"""VRAM phase scheduler for single-GPU (RTX 5090 32 GB) serving/training transitions."""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.infra.vllm_server import VLLMServer

logger = logging.getLogger(__name__)


class Phase(str, Enum):
    SERVING = "serving"
    TRAINING = "training"
    TRANSITIONING = "transitioning"


class VRAMScheduler:
    """Coordinates exclusive VRAM access between vLLM serving and training.

    Only one phase is active at a time.  An async lock prevents concurrent
    transitions.  The scheduler does *not* own the ``VLLMServer`` — it just
    orchestrates its start/stop around training needs.
    """

    def __init__(
        self,
        server: VLLMServer,
        *,
        vram_free_wait: float = 5.0,
        vram_free_timeout: float = 60.0,
        vram_free_target_mb: int = 30_000,
    ) -> None:
        self._server = server
        self._phase: Phase = Phase.TRAINING  # nothing running initially
        self._lock = asyncio.Lock()
        self._vram_free_wait = vram_free_wait
        self._vram_free_timeout = vram_free_timeout
        self._vram_free_target_mb = vram_free_target_mb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_phase(self) -> str:
        """Return the current phase as a plain string."""
        return self._phase.value

    async def enter_serving_phase(self, *, max_retries: int = 3) -> None:
        """Start the vLLM server and switch to serving phase."""
        async with self._lock:
            if self._phase == Phase.SERVING:
                logger.info("Already in serving phase — no-op")
                return

            self._phase = Phase.TRANSITIONING
            logger.info("[%s] Transitioning -> SERVING", _ts())

            for attempt in range(1, max_retries + 1):
                try:
                    await self._server.start()
                    break
                except Exception:
                    logger.exception(
                        "Server start failed (attempt %d/%d)", attempt, max_retries,
                    )
                    if attempt == max_retries:
                        self._phase = Phase.TRAINING  # stay in safe state
                        raise
                    await asyncio.sleep(10.0)

            self._phase = Phase.SERVING
            logger.info("[%s] Phase is now SERVING", _ts())

    async def enter_training_phase(self) -> None:
        """Stop the vLLM server, wait for VRAM to free, switch to training."""
        async with self._lock:
            if self._phase == Phase.TRAINING:
                logger.info("Already in training phase — no-op")
                return

            self._phase = Phase.TRANSITIONING
            logger.info("[%s] Transitioning -> TRAINING", _ts())

            await self._server.stop()
            await self._wait_for_vram_free()

            self._phase = Phase.TRAINING
            logger.info("[%s] Phase is now TRAINING", _ts())

    # ------------------------------------------------------------------
    # VRAM introspection
    # ------------------------------------------------------------------

    @staticmethod
    async def query_gpu_free_memory_mb() -> float | None:
        """Return free GPU memory in MiB by calling ``nvidia-smi``.

        Returns ``None`` if nvidia-smi is unavailable or parsing fails.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            # Take the first GPU line.
            value = stdout.decode().strip().splitlines()[0]
            return float(value)
        except Exception as exc:  # noqa: BLE001
            logger.debug("nvidia-smi query failed: %s", exc)
            return None

    async def _wait_for_vram_free(self) -> None:
        """Poll nvidia-smi until enough VRAM is free or timeout is reached."""
        deadline = time.monotonic() + self._vram_free_timeout
        target = self._vram_free_target_mb

        while time.monotonic() < deadline:
            free = await self.query_gpu_free_memory_mb()
            if free is None:
                # nvidia-smi not available; assume memory is freed after
                # the process has exited.
                logger.info(
                    "nvidia-smi unavailable — assuming VRAM freed after server stop"
                )
                return

            logger.info(
                "GPU free VRAM: %.0f MiB (target >= %d MiB)", free, target
            )
            if free >= target:
                return

            await asyncio.sleep(self._vram_free_wait)

        logger.warning(
            "VRAM did not reach target %d MiB within %.0fs — proceeding anyway",
            target,
            self._vram_free_timeout,
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _ts() -> str:
    """Return a compact timestamp for log messages."""
    return time.strftime("%H:%M:%S")
