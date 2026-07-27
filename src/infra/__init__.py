"""Infrastructure modules for serving, scheduling, and evaluation.

Exports are resolved lazily so lightweight helpers such as ``ollama_utils`` do
not require optional OpenAI/vLLM dependencies at import time.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "EvalHarness",
    "VLLMServer",
    "VRAMScheduler",
]


def __getattr__(name: str) -> Any:
    if name == "EvalHarness":
        from src.infra.eval_harness import EvalHarness

        return EvalHarness
    if name == "VLLMServer":
        from src.infra.vllm_server import VLLMServer

        return VLLMServer
    if name == "VRAMScheduler":
        from src.infra.vram_scheduler import VRAMScheduler

        return VRAMScheduler
    raise AttributeError(name)
