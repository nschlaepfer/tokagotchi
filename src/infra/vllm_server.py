"""LLM server lifecycle management via Ollama with OpenAI-compatible client.

Ollama runs natively on Windows, serves GGUF-quantized models, and exposes
an OpenAI-compatible API at localhost:11434/v1. This replaces vLLM for our
setup since vLLM doesn't support Windows natively.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import time
from enum import Enum
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
    """Manages Ollama model serving and exposes an async OpenAI-compatible client.

    Ollama runs as a background service on Windows. This class ensures the
    model is loaded and provides chat_completion / generate methods that
    the rest of the codebase uses.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        health_poll_interval: float = 2.0,
        health_timeout: float = 300.0,
    ) -> None:
        self.config = config
        self.health_poll_interval = health_poll_interval
        self.health_timeout = health_timeout

        self._status: ServerStatus = ServerStatus.STOPPED
        self._client: openai.AsyncOpenAI | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def status(self) -> ServerStatus:
        return self._status

    @property
    def base_url(self) -> str:
        return f"http://{self.config.ollama_host}:{self.config.ollama_port}/v1"

    @property
    def api_url(self) -> str:
        return f"http://{self.config.ollama_host}:{self.config.ollama_port}/api"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Ensure Ollama is running and the model is loaded."""
        if self._status in (ServerStatus.STARTING, ServerStatus.READY):
            logger.warning("LLM server is already %s", self._status.value)
            return

        self._status = ServerStatus.STARTING
        logger.info("Starting LLM server: model=%s", self.config.name)

        # 1. Check if Ollama service is reachable
        await self._ensure_ollama_running()

        # 2. Ensure model is pulled
        await self._ensure_model_available()

        # 3. Warm up the model by loading it into GPU memory
        await self._warmup_model()

        # 4. Create OpenAI-compatible client
        self._client = openai.AsyncOpenAI(
            base_url=self.base_url,
            api_key="ollama",  # Ollama ignores this but openai lib requires it
        )

        self._status = ServerStatus.READY
        logger.info("LLM server is ready: model=%s at %s", self.config.name, self.base_url)

    async def stop(self) -> None:
        """Unload the model from GPU memory."""
        if self._status == ServerStatus.STOPPED:
            return

        self._status = ServerStatus.STOPPING
        logger.info("Unloading model %s from Ollama ...", self.config.name)

        import aiohttp

        # Retry unload with single session — Ollama sometimes needs multiple nudges
        async with aiohttp.ClientSession() as session:
            for attempt in range(3):
                try:
                    async with session.post(
                        f"{self.api_url}/generate",
                        json={"model": self.config.name, "keep_alive": 0},
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        if resp.status == 200:
                            logger.info("Model unload requested (attempt %d)", attempt + 1)
                        else:
                            logger.warning("Model unload returned status %d", resp.status)
                except Exception as e:
                    logger.warning("Failed to unload model (attempt %d): %s", attempt + 1, e)

                # Wait for VRAM to free, then verify
                await asyncio.sleep(3)
                free_mb = await _query_gpu_free_mb()
                if free_mb and free_mb > 25000:
                    logger.info("GPU memory freed: %.0f MiB available", free_mb)
                    break
                logger.info("GPU still held: %.0f MiB free, retrying...", free_mb or 0)

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
        think: bool = True,
        **kwargs: Any,
    ) -> openai.types.chat.ChatCompletion:
        """Send a chat-completion request to the local Ollama server.

        Parameters
        ----------
        think:
            If True (default), enables Qwen 3.5's thinking mode. The model
            puts reasoning in the ``thinking`` field and the action in
            ``content``. This is REQUIRED for qwen3.5-abliterated which
            cannot produce content without thinking first.
        """
        self._ensure_ready()

        # Always use native Ollama API (supports think parameter)
        result = await self._native_chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            think=think,
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
        # Fix thinking model responses: extract reasoning if content is empty
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
        """Send a raw completion request to the local Ollama server."""
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

    async def _warmup_model(self) -> None:
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


async def _query_gpu_free_mb() -> float | None:
    """Quick GPU free memory check via nvidia-smi."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        return float(stdout.decode().strip().split("\n")[0])
    except Exception:
        return None
