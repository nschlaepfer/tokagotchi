"""vLLM server lifecycle management and OpenAI-compatible inference client."""

from __future__ import annotations

import asyncio
import logging
import signal
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any, Sequence

import openai

from src.config import ModelConfig

logger = logging.getLogger(__name__)


class ServerStatus(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"


class VLLMServer:
    """Manages a vLLM subprocess and exposes an async OpenAI-compatible client."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        log_dir: str | Path = "./logs",
        health_poll_interval: float = 2.0,
        health_timeout: float = 300.0,
        shutdown_grace_seconds: float = 15.0,
        extra_vllm_args: Sequence[str] = (),
    ) -> None:
        self.config = config
        self.log_dir = Path(log_dir)
        self.health_poll_interval = health_poll_interval
        self.health_timeout = health_timeout
        self.shutdown_grace_seconds = shutdown_grace_seconds
        self.extra_vllm_args = list(extra_vllm_args)

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
    def base_url(self) -> str:
        return f"http://{self.config.vllm_host}:{self.config.vllm_port}/v1"

    @property
    def health_url(self) -> str:
        return f"http://{self.config.vllm_host}:{self.config.vllm_port}/health"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch vLLM as a subprocess and wait until it is healthy."""
        if self._status in (ServerStatus.STARTING, ServerStatus.READY):
            logger.warning("vLLM server is already %s", self._status.value)
            return

        self._status = ServerStatus.STARTING
        logger.info(
            "Starting vLLM server: model=%s quant=%s port=%d gpu_mem=%.2f",
            self.config.name,
            self.config.quantization,
            self.config.vllm_port,
            self.config.vllm_gpu_memory_utilization,
        )

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file_path = self.log_dir / f"vllm_{int(time.time())}.log"

        cmd = self._build_command()
        logger.info("vLLM command: %s", " ".join(cmd))

        log_fh = open(self._log_file_path, "wb")  # noqa: SIM115
        self._process = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            # On Windows SIGTERM is not available; we rely on terminate().
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP
                if sys.platform == "win32"
                else 0
            ),
        )

        try:
            await self._wait_for_healthy()
        except Exception:
            logger.error("vLLM failed to become healthy; shutting down process")
            await self.stop()
            raise

        self._client = openai.AsyncOpenAI(
            base_url=self.base_url,
            api_key="unused",  # vLLM does not require a key
        )
        self._status = ServerStatus.READY
        logger.info("vLLM server is ready at %s", self.base_url)

    async def stop(self) -> None:
        """Gracefully stop the vLLM process (SIGTERM then SIGKILL)."""
        if self._process is None or self._status == ServerStatus.STOPPED:
            self._status = ServerStatus.STOPPED
            return

        self._status = ServerStatus.STOPPING
        logger.info("Stopping vLLM server (pid=%d) ...", self._process.pid)

        # Attempt graceful termination.
        self._process.terminate()
        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, self._process.wait),
                timeout=self.shutdown_grace_seconds,
            )
            logger.info("vLLM process terminated gracefully")
        except asyncio.TimeoutError:
            logger.warning(
                "vLLM did not exit within %.0fs; sending SIGKILL",
                self.shutdown_grace_seconds,
            )
            self._process.kill()
            await asyncio.get_event_loop().run_in_executor(None, self._process.wait)
            logger.info("vLLM process killed")

        self._process = None
        self._client = None
        self._status = ServerStatus.STOPPED
        logger.info("vLLM server stopped")

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> VLLMServer:
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
        **kwargs: Any,
    ) -> openai.types.chat.ChatCompletion:
        """Send a chat-completion request to the local vLLM server."""
        self._ensure_ready()
        assert self._client is not None
        return await self._client.chat.completions.create(
            model=self.config.name,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            top_p=top_p,
            **kwargs,
        )

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
        """Send a raw (non-chat) completion request to the local vLLM server."""
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_command(self) -> list[str]:
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.name,
            "--port", str(self.config.vllm_port),
            "--host", self.config.vllm_host,
            "--gpu-memory-utilization", str(self.config.vllm_gpu_memory_utilization),
        ]
        if self.config.quantization:
            cmd += ["--quantization", self.config.quantization]
        cmd.extend(self.extra_vllm_args)
        return cmd

    async def _wait_for_healthy(self) -> None:
        """Poll the /health endpoint until vLLM reports ready."""
        import aiohttp

        deadline = time.monotonic() + self.health_timeout
        logger.info("Waiting up to %.0fs for vLLM health check ...", self.health_timeout)

        while time.monotonic() < deadline:
            # If the subprocess exited, abort early.
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"vLLM process exited with code {self._process.returncode} "
                    f"before becoming healthy. See logs at {self._log_file_path}"
                )

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.health_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            return
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass

            await asyncio.sleep(self.health_poll_interval)

        raise TimeoutError(
            f"vLLM server did not become healthy within {self.health_timeout}s. "
            f"See logs at {self._log_file_path}"
        )

    def _ensure_ready(self) -> None:
        if self._status != ServerStatus.READY:
            raise RuntimeError(
                f"vLLM server is not ready (status={self._status.value}). "
                "Call start() first or use the async context manager."
            )
