"""Local LLM server lifecycle management.

The file keeps the historical ``vllm_server`` module path for compatibility,
but the implementation is provider-aware and defaults to MLX on Apple Silicon.
Supported backends:

- ``mlx``: launches ``python -m mlx_lm.server`` and talks OpenAI-compatible HTTP
- ``ollama``: uses Ollama's native API plus OpenAI-compatible endpoints
- ``vllm``: launches the legacy vLLM OpenAI server
- ``openai``: assumes an already-running OpenAI-compatible endpoint
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any

import openai

from src.config import ModelConfig

logger = logging.getLogger(__name__)


class ServerStatus(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"


class LLMServer:
    """Manages a local model server behind an OpenAI-compatible client."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        log_dir: str | Path = "./logs",
        health_poll_interval: float = 2.0,
        health_timeout: float = 300.0,
        shutdown_grace_seconds: float = 15.0,
    ) -> None:
        self.config = config
        self.log_dir = Path(log_dir)
        self.health_poll_interval = health_poll_interval
        self.health_timeout = health_timeout
        self.shutdown_grace_seconds = shutdown_grace_seconds

        self._process: subprocess.Popen[bytes] | None = None
        self._status: ServerStatus = ServerStatus.STOPPED
        self._log_file_path: Path | None = None
        self._client: openai.AsyncOpenAI | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def status(self) -> ServerStatus:
        return self._status

    @property
    def provider(self) -> str:
        return self.config.normalized_provider

    @property
    def base_url(self) -> str:
        return f"http://{self.config.resolved_host}:{self.config.resolved_port}/v1"

    @property
    def api_url(self) -> str:
        return f"http://{self.config.resolved_host}:{self.config.resolved_port}/api"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Ensure the configured local LLM server is running and ready."""
        if self._status in (ServerStatus.STARTING, ServerStatus.READY):
            logger.warning("LLM server is already %s", self._status.value)
            return

        self._status = ServerStatus.STARTING
        logger.info(
            "Starting LLM server: provider=%s model=%s host=%s port=%d",
            self.provider,
            self.config.name,
            self.config.resolved_host,
            self.config.resolved_port,
        )

        if self.provider == "ollama":
            await self._ensure_ollama_running()
            await self._ensure_model_available()
            await self._warmup_ollama_model()
        else:
            await self._ensure_openai_server_running()

        self._client = openai.AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.config.resolved_api_key,
        )
        await self._wait_for_openai_ready(warmup=self.provider in {"mlx", "vllm"})

        self._status = ServerStatus.READY
        logger.info(
            "LLM server is ready: provider=%s model=%s at %s",
            self.provider,
            self.config.name,
            self.base_url,
        )

    async def stop(self) -> None:
        """Stop the managed server or unload the configured model."""
        if self._status == ServerStatus.STOPPED:
            return

        self._status = ServerStatus.STOPPING

        if self.provider == "ollama":
            await self._unload_ollama_model()

        if self._process is not None:
            logger.info("Stopping managed %s server (pid=%d)", self.provider, self._process.pid)
            self._process.terminate()
            try:
                await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(None, self._process.wait),
                    timeout=self.shutdown_grace_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "%s server did not exit within %.0fs; killing",
                    self.provider,
                    self.shutdown_grace_seconds,
                )
                self._process.kill()
                await asyncio.get_running_loop().run_in_executor(None, self._process.wait)
            self._process = None

        self._client = None
        self._status = ServerStatus.STOPPED
        logger.info("LLM server stopped")

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> LLMServer:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: str | list[str] | None = None,
        top_p: float = 1.0,
        think: bool = False,
        **kwargs: Any,
    ) -> openai.types.chat.ChatCompletion:
        """Send a chat-completion request to the configured local server."""
        self._ensure_ready()

        if self.provider == "ollama" and not think:
            result = await self._native_chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                think=False,
            )
            return _wrap_native_response(result)

        assert self._client is not None
        response = await self._client.chat.completions.create(
            model=self.config.name,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            top_p=top_p,
            **kwargs,
        )
        if self.provider == "ollama":
            _fix_thinking_response(response)
        return response

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        think: bool = False,
    ) -> str:
        """Convenience: send a chat request and return the text content."""
        resp = await self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            think=think,
        )
        return resp.choices[0].message.content or ""

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: str | list[str] | None = None,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> openai.types.Completion:
        """Send a raw completion request to the configured server."""
        self._ensure_ready()
        assert self._client is not None
        return await self._client.completions.create(
            model=self.config.name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            top_p=top_p,
            **kwargs,
        )

    async def _native_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        think: bool = False,
    ) -> dict[str, Any]:
        """Call the native Ollama /api/chat endpoint (supports think param)."""
        import aiohttp

        payload: dict[str, Any] = {
            "model": self.config.name,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p,
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_url}/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Ollama /api/chat failed ({resp.status}): {text}")
                return await resp.json()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_openai_server_running(self) -> None:
        if await self._probe_openai_ready():
            return

        if not self.config.auto_start:
            raise RuntimeError(
                f"{self.provider} server is not reachable at {self.base_url} and auto_start is disabled."
            )

        cmd = self._build_server_command()
        if not cmd:
            raise RuntimeError(
                f"No auto-start command configured for provider={self.provider}."
            )

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file_path = self.log_dir / f"{self.provider}_{int(time.time())}.log"
        log_fh = open(self._log_file_path, "wb")  # noqa: SIM115
        env = os.environ.copy()

        logger.info("Launching %s server: %s", self.provider, " ".join(cmd))
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        self._process = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            env=env,
        )

    def _build_server_command(self) -> list[str]:
        if self.config.server_command:
            return shlex.split(self.config.server_command)

        if self.provider == "mlx":
            return [
                sys.executable,
                "-m",
                "mlx_lm.server",
                "--model",
                self.config.name,
                "--host",
                self.config.resolved_host,
                "--port",
                str(self.config.resolved_port),
            ]

        if self.provider == "vllm":
            cmd = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                self.config.name,
                "--port",
                str(self.config.resolved_port),
                "--host",
                self.config.resolved_host,
                "--gpu-memory-utilization",
                str(self.config.vllm_gpu_memory_utilization),
            ]
            if self.config.quantization:
                cmd += ["--quantization", self.config.quantization]
            return cmd

        return []

    async def _wait_for_openai_ready(self, *, warmup: bool) -> None:
        deadline = time.monotonic() + self.health_timeout
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"{self.provider} server exited with code {self._process.returncode}. "
                    f"See logs at {self._log_file_path}"
                )

            if await self._probe_openai_ready():
                if warmup:
                    await self._warmup_openai_model()
                return

            await asyncio.sleep(self.health_poll_interval)

        raise TimeoutError(
            f"{self.provider} server did not become ready within {self.health_timeout}s. "
            f"See logs at {self._log_file_path or 'stdout/stderr'}"
        )

    async def _probe_openai_ready(self) -> bool:
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        return True
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.config.name,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 1,
                        "temperature": 0.0,
                    },
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def _warmup_openai_model(self) -> None:
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.config.name,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 1,
                        "temperature": 0.0,
                    },
                    timeout=aiohttp.ClientTimeout(total=self.health_timeout),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(
                            f"Warmup failed for provider={self.provider} (status={resp.status}): {text}"
                        )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"{self.provider} warmup timed out after {self.health_timeout}s"
            ) from exc

    async def _ensure_ollama_running(self) -> None:
        """Check that the Ollama service is reachable, start it if not."""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{self.config.ollama_host}:{self.config.ollama_port}/",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        logger.info("Ollama service is running")
                        return
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass

        # Try to start Ollama service
        logger.info("Ollama not reachable, attempting to start ...")
        if sys.platform == "win32":
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        # Wait for it to come up
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://{self.config.ollama_host}:{self.config.ollama_port}/",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as resp:
                        if resp.status == 200:
                            logger.info("Ollama service started")
                            return
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(1.0)

        raise RuntimeError("Failed to start Ollama service within 30s")

    async def _ensure_model_available(self) -> None:
        """Check if the model is pulled, pull it if not."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            # Check local models
            async with session.get(
                f"{self.api_url}/tags",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    model_names = [m.get("name", "") for m in data.get("models", [])]
                    # Check if our model is in the list (handle tag variations)
                    model_base = self.config.name.split(":")[0]
                    for name in model_names:
                        if model_base in name:
                            logger.info("Model %s is already available locally", self.config.name)
                            return

        logger.info("Model %s not found locally, pulling ...", self.config.name)
        proc = await asyncio.create_subprocess_exec(
            "ollama", "pull", self.config.name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to pull model {self.config.name}: {stdout.decode()}"
            )
        logger.info("Model %s pulled successfully", self.config.name)

    async def _warmup_ollama_model(self) -> None:
        """Load the model into GPU memory with a dummy request."""
        import aiohttp

        logger.info("Warming up model %s (loading into GPU) ...", self.config.name)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/generate",
                    json={
                        "model": self.config.name,
                        "prompt": "Hello",
                        "stream": False,
                        "options": {"num_predict": 1},
                    },
                    timeout=aiohttp.ClientTimeout(total=self.health_timeout),
                ) as resp:
                    if resp.status == 200:
                        logger.info("Model loaded into GPU memory")
                    else:
                        text = await resp.text()
                        raise RuntimeError(f"Warmup failed (status={resp.status}): {text}")
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Model warmup timed out after {self.health_timeout}s"
            )

    async def _unload_ollama_model(self) -> None:
        import aiohttp

        logger.info("Unloading model %s from Ollama ...", self.config.name)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/generate",
                    json={"model": self.config.name, "keep_alive": 0},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        logger.warning("Ollama model unload returned status %d", resp.status)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to unload Ollama model: %s", exc)

    def _ensure_ready(self) -> None:
        if self._status != ServerStatus.READY:
            raise RuntimeError(
                f"LLM server is not ready (status={self._status.value}). "
                "Call start() first or use the async context manager."
            )


# Backward-compatible alias
VLLMServer = LLMServer


# ---------------------------------------------------------------------------
# Thinking-model response helpers
# ---------------------------------------------------------------------------


def _fix_thinking_response(response: openai.types.chat.ChatCompletion) -> None:
    """Fix responses from thinking models (Qwen 3.5) where content is empty.

    When Qwen 3.5 uses thinking mode via the OpenAI-compatible API, the actual
    response text goes into the ``reasoning`` field and ``content`` is empty.
    This function moves reasoning into content when content is empty.
    """
    for choice in response.choices:
        msg = choice.message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning", None) or ""

        if not content.strip() and reasoning.strip():
            # Extract the actual answer from reasoning (after thinking)
            # Qwen often wraps thinking in <think> tags
            import re
            # Try to get text after </think> tag
            match = re.split(r"</think>", reasoning, maxsplit=1)
            if len(match) > 1 and match[1].strip():
                msg.content = match[1].strip()
            else:
                # No think tags — use the full reasoning as content
                msg.content = reasoning.strip()


def _wrap_native_response(data: dict) -> openai.types.chat.ChatCompletion:
    """Wrap a native Ollama /api/chat response as an OpenAI ChatCompletion."""
    msg = data.get("message", {})
    content = msg.get("content", "")
    thinking = msg.get("thinking", "")

    # Build a minimal ChatCompletion-compatible object
    return openai.types.chat.ChatCompletion(
        id="ollama-native",
        choices=[
            openai.types.chat.chat_completion.Choice(
                index=0,
                finish_reason="stop",
                message=openai.types.chat.ChatCompletionMessage(
                    role="assistant",
                    content=content or thinking or "",
                ),
            )
        ],
        created=int(time.time()),
        model=data.get("model", ""),
        object="chat.completion",
        usage=openai.types.CompletionUsage(
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        ),
    )
