"""Weights & Biases integration for tokagotchi training tracking.

Provides a singleton tracker that logs:
- Genome evaluations (success rate, tool accuracy, steps, quality)
- Mutation events (type, parent scores, child scores)
- SDPO contrastive pairs generated
- SFT training metrics (loss, learning rate, steps)
- Overall pipeline status (budget, buffer size, frontier size)
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_run = None  # wandb.Run singleton
_initialized = False


def init(
    project: str = "tokagotchi",
    name: str | None = None,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> bool:
    """Initialize a wandb run. Returns True if successful."""
    global _run, _initialized

    if _initialized:
        return _run is not None

    try:
        import wandb

        _run = wandb.init(
            project=project,
            name=name,
            config=config or {},
            tags=tags or ["tokagotchi"],
            resume="allow",
            reinit=True,
        )
        _initialized = True
        logger.info("wandb initialized: %s/%s", project, _run.name)
        return True
    except Exception as exc:
        logger.warning("wandb init failed (continuing without tracking): %s", exc)
        _initialized = True  # Don't retry
        return False


def log_genome_eval(
    genome_id: str,
    generation: int,
    success_rate: float,
    avg_steps: float,
    tool_accuracy: float,
    code_quality: float,
    mutation_type: str = "",
    step: int | None = None,
) -> None:
    """Log a genome evaluation result."""
    if _run is None:
        return

    data = {
        "eval/success_rate": success_rate,
        "eval/avg_steps": avg_steps,
        "eval/tool_accuracy": tool_accuracy,
        "eval/code_quality": code_quality,
        "eval/generation": generation,
    }
    if mutation_type:
        data["eval/mutation_type"] = mutation_type

    try:
        _run.log(data, step=step)
    except Exception:
        logger.debug("wandb log failed", exc_info=True)


def log_iteration(
    iteration: int,
    generation: int,
    frontier_size: int,
    mutations_succeeded: int,
    mutations_proposed: int,
    best_success: float,
    best_tool_acc: float = 0.0,
    duration_seconds: float = 0.0,
) -> None:
    """Log a GEPA iteration summary."""
    if _run is None:
        return

    try:
        _run.log({
            "gepa/iteration": iteration,
            "gepa/generation": generation,
            "gepa/frontier_size": frontier_size,
            "gepa/mutations_succeeded": mutations_succeeded,
            "gepa/mutations_proposed": mutations_proposed,
            "gepa/best_success_rate": best_success,
            "gepa/best_tool_accuracy": best_tool_acc,
            "gepa/iteration_duration_s": duration_seconds,
        })
    except Exception:
        logger.debug("wandb log failed", exc_info=True)


def log_mutation(
    genome_id: str,
    parent_id: str,
    mutation_type: str,
    parent_success: float = 0.0,
    child_success: float | None = None,
    divergence: float | None = None,
) -> None:
    """Log a mutation event."""
    if _run is None:
        return

    data: dict[str, Any] = {
        "mutation/type": mutation_type,
        "mutation/parent_success": parent_success,
    }
    if child_success is not None:
        data["mutation/child_success"] = child_success
        data["mutation/improvement"] = child_success - parent_success

    try:
        _run.log(data)
    except Exception:
        logger.debug("wandb log failed", exc_info=True)


def log_sdpo(
    trajectory_id: str,
    num_pairs: int,
    num_steps_checked: int,
    avg_divergence: float = 0.0,
    escalated_to_opus: bool = False,
) -> None:
    """Log an SDPO re-evaluation event."""
    if _run is None:
        return

    try:
        _run.log({
            "sdpo/contrastive_pairs": num_pairs,
            "sdpo/steps_checked": num_steps_checked,
            "sdpo/avg_divergence": avg_divergence,
            "sdpo/escalated_to_opus": int(escalated_to_opus),
        })
    except Exception:
        logger.debug("wandb log failed", exc_info=True)


def log_buffer_status(
    size: int,
    is_ready: bool,
    sdpo_examples: int = 0,
    opus_examples: int = 0,
) -> None:
    """Log pending buffer status."""
    if _run is None:
        return

    try:
        _run.log({
            "buffer/size": size,
            "buffer/is_ready": int(is_ready),
            "buffer/sdpo_examples": sdpo_examples,
            "buffer/opus_examples": opus_examples,
        })
    except Exception:
        logger.debug("wandb log failed", exc_info=True)


def log_training(
    loss: float,
    learning_rate: float,
    epoch: int,
    step: int,
    num_examples: int = 0,
) -> None:
    """Log SFT/RL training metrics."""
    if _run is None:
        return

    try:
        _run.log({
            "train/loss": loss,
            "train/learning_rate": learning_rate,
            "train/epoch": epoch,
            "train/step": step,
            "train/num_examples": num_examples,
        })
    except Exception:
        logger.debug("wandb log failed", exc_info=True)


def log_budget(
    hourly_usd: float,
    daily_usd: float,
    opus_calls: int = 0,
) -> None:
    """Log API budget usage."""
    if _run is None:
        return

    try:
        _run.log({
            "budget/hourly_usd": hourly_usd,
            "budget/daily_usd": daily_usd,
            "budget/opus_calls": opus_calls,
        })
    except Exception:
        logger.debug("wandb log failed", exc_info=True)


def log_pipeline_status(
    uptime_seconds: float,
    loop1_status: str,
    loop2_status: str,
    loop3_status: str,
) -> None:
    """Log overall pipeline status."""
    if _run is None:
        return

    try:
        _run.log({
            "pipeline/uptime_hours": uptime_seconds / 3600,
        })
    except Exception:
        logger.debug("wandb log failed", exc_info=True)


def finish() -> None:
    """Finalize the wandb run."""
    global _run, _initialized
    if _run is not None:
        try:
            _run.finish()
        except Exception:
            pass
    _run = None
    _initialized = False
