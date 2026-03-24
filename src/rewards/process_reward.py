"""Opus-based per-step process reward (expensive, sampled).

Sends the full trajectory to Opus for structured per-step evaluation across
four quality dimensions.  This module is intentionally expensive and should
only be called on a sampled subset of trajectories.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

from src.models import TaskSpec, Trajectory
from src.orchestrator.opus_client import OpusClient

logger = logging.getLogger(__name__)

# The four evaluation dimensions and their descriptions (used in the prompt)
DIMENSIONS = {
    "correctness": "Was this the right action given current knowledge and task state?",
    "efficiency": "Did this step make meaningful progress toward the goal?",
    "reasoning_quality": "Was the thinking/reasoning sound and well-structured?",
    "information_utilization": "Did it make good use of information available from prior steps?",
}

_PROCESS_REWARD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "step_ratings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_idx": {"type": "integer"},
                    "correctness": {"type": "number"},
                    "efficiency": {"type": "number"},
                    "reasoning_quality": {"type": "number"},
                    "info_utilization": {"type": "number"},
                    "feedback": {"type": "string"},
                },
                "required": [
                    "step_idx", "correctness", "efficiency",
                    "reasoning_quality", "info_utilization",
                ],
            },
        },
        "critical_decision_point": {"type": "integer"},
        "critical_decision_rationale": {"type": "string"},
        "overall_process_score": {"type": "number"},
    },
    "required": ["step_ratings", "critical_decision_point", "overall_process_score"],
}


async def compute_process_reward(
    trajectory: Trajectory,
    task_spec: TaskSpec,
    opus_client: OpusClient,
) -> list[dict[str, Any]]:
    """Compute per-step process rewards via Opus evaluation.

    Sends the full trajectory and task specification to Opus with a structured
    prompt.  Opus rates each step on four dimensions (0.0 -- 1.0):

    * **correctness** -- was this the right action given current knowledge?
    * **efficiency** -- did this step make meaningful progress?
    * **reasoning_quality** -- was the thinking sound?
    * **information_utilization** -- did it use available info well?

    Opus also identifies the *critical decision point* -- the step where
    success or failure was most strongly determined.

    Parameters
    ----------
    trajectory:
        The completed agent trajectory.
    task_spec:
        The task specification for context.
    opus_client:
        An :class:`OpusClient` for making the evaluation call.

    Returns
    -------
    list[dict]
        A list of per-step rating dicts.  Each dict has keys ``step_idx``,
        ``correctness``, ``efficiency``, ``reasoning_quality``,
        ``info_utilization``, and optionally ``feedback``.  An extra entry
        with key ``__summary__`` is appended containing the aggregate score
        and critical decision point.
    """
    prompt = _build_evaluation_prompt(trajectory, task_spec)

    response = await opus_client.query(
        prompt,
        json_schema=_PROCESS_REWARD_SCHEMA,
        max_budget_usd=0.20,
        max_turns=1,
    )

    if response.is_error:
        logger.error(
            "Process reward evaluation failed for trajectory %s: %s",
            trajectory.trajectory_id, response.error_message,
        )
        return []

    data = response.raw_json
    step_ratings: list[dict[str, Any]] = data.get("step_ratings", [])
    critical_point: int | None = data.get("critical_decision_point")
    critical_rationale: str = data.get("critical_decision_rationale", "")
    overall_score: float = data.get("overall_process_score", 0.0)

    # Clamp all dimension scores to [0, 1]
    for rating in step_ratings:
        for dim in ("correctness", "efficiency", "reasoning_quality", "info_utilization"):
            val = rating.get(dim, 0.0)
            rating[dim] = max(0.0, min(1.0, float(val)))

    # Append summary entry
    step_ratings.append({
        "__summary__": True,
        "overall_process_score": overall_score,
        "critical_decision_point": critical_point,
        "critical_decision_rationale": critical_rationale,
        "avg_correctness": _avg_dim(step_ratings, "correctness"),
        "avg_efficiency": _avg_dim(step_ratings, "efficiency"),
        "avg_reasoning_quality": _avg_dim(step_ratings, "reasoning_quality"),
        "avg_info_utilization": _avg_dim(step_ratings, "info_utilization"),
    })

    logger.info(
        "Process reward for %s: overall=%.3f, critical_step=%s, %d steps rated",
        trajectory.trajectory_id, overall_score, critical_point,
        len(step_ratings) - 1,  # exclude summary
    )

    return step_ratings


def average_process_score(step_ratings: list[dict[str, Any]]) -> float:
    """Collapse per-step ratings into a single scalar process reward.

    Uses the ``overall_process_score`` from the summary entry if available,
    otherwise averages the per-step dimension means.

    Parameters
    ----------
    step_ratings:
        The list returned by :func:`compute_process_reward`.

    Returns
    -------
    float
        Scalar process reward in [0.0, 1.0].
    """
    # Check for summary entry first
    for entry in step_ratings:
        if entry.get("__summary__"):
            score = entry.get("overall_process_score", 0.0)
            return max(0.0, min(1.0, float(score)))

    # Fallback: average all four dimensions across steps
    if not step_ratings:
        return 0.0

    dims = ("correctness", "efficiency", "reasoning_quality", "info_utilization")
    total = 0.0
    count = 0
    for rating in step_ratings:
        if rating.get("__summary__"):
            continue
        step_avg = sum(rating.get(d, 0.0) for d in dims) / len(dims)
        total += step_avg
        count += 1

    return round(total / max(count, 1), 4)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_evaluation_prompt(trajectory: Trajectory, task_spec: TaskSpec) -> str:
    """Construct the structured prompt for Opus process evaluation."""
    dim_descriptions = "\n".join(
        f"  - **{name}**: {desc}" for name, desc in DIMENSIONS.items()
    )

    trajectory_json = json.dumps(asdict(trajectory), default=str, indent=2)
    task_json = json.dumps(asdict(task_spec), default=str, indent=2)

    return (
        "You are an expert evaluator assessing the quality of an AI agent's "
        "problem-solving process. Rate EACH step of the trajectory on four "
        "dimensions, each scored from 0.0 (terrible) to 1.0 (excellent):\n\n"
        f"{dim_descriptions}\n\n"
        "Additionally, identify the **critical decision point** -- the single "
        "step where the agent's success or failure was most strongly determined. "
        "Explain why this step was pivotal.\n\n"
        "Finally, provide an **overall_process_score** (0.0-1.0) reflecting the "
        "quality of the agent's entire problem-solving process.\n\n"
        f"## Task Specification\n```json\n{task_json}\n```\n\n"
        f"## Agent Trajectory\n```json\n{trajectory_json}\n```\n\n"
        "Return your evaluation as structured JSON."
    )


def _avg_dim(
    ratings: list[dict[str, Any]],
    dimension: str,
) -> float:
    """Average a dimension score across non-summary step ratings."""
    values = [
        r[dimension]
        for r in ratings
        if not r.get("__summary__") and dimension in r
    ]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)
