"""Composite reward module -- combines outcome, process, and efficiency signals.

Provides the :class:`CompositeReward` class that orchestrates the three reward
components and produces a single :class:`RewardResult`.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from src.config import RewardWeights
from src.models import RewardResult, TaskSpec, Trajectory
from src.orchestrator.opus_client import OpusClient
from src.rewards.efficiency_penalty import compute_efficiency_penalty
from src.rewards.outcome_reward import compute_outcome_reward
from src.rewards.process_reward import (
    average_process_score,
    compute_process_reward,
)

logger = logging.getLogger(__name__)

# Default process reward sampling rate if not provided
DEFAULT_SAMPLE_RATE = 0.20


class CompositeReward:
    """Combines outcome, process, and efficiency reward signals.

    The composite reward is computed as::

        composite = w_outcome * outcome + w_process * process - w_efficiency * penalty

    where the weights are drawn from a :class:`RewardWeights` configuration.

    Parameters
    ----------
    weights:
        Reward component weights.  Defaults to ``RewardWeights()`` which gives
        outcome=0.6, process=0.3, efficiency=0.1.
    sample_rate:
        Probability of invoking the expensive Opus-based process reward on any
        given trajectory.  Defaults to 0.20 (20%).
    """

    def __init__(
        self,
        weights: RewardWeights | None = None,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.weights = weights or RewardWeights()
        self.sample_rate = sample_rate

    async def compute(
        self,
        trajectory: Trajectory,
        task_spec: TaskSpec,
        container_id: str,
        docker_manager,
        opus_client: OpusClient,
        use_process_reward: bool = False,
    ) -> RewardResult:
        """Compute the composite reward for a trajectory.

        Parameters
        ----------
        trajectory:
            The completed agent trajectory.
        task_spec:
            The task specification used to evaluate the outcome.
        container_id:
            Docker container ID where the task ran.
        docker_manager:
            A :class:`DockerManager` instance for in-container verification.
        opus_client:
            An :class:`OpusClient` for the (optional) process reward call.
        use_process_reward:
            If ``True``, the Opus-based process reward is computed regardless
            of the sampling rate.  If ``False``, the process reward is skipped
            and its contribution is zero.

        Returns
        -------
        RewardResult
            A result object containing all individual components, the composite
            score, per-step rewards, and diagnostic details.
        """
        # 1. Outcome reward (always computed)
        outcome = await compute_outcome_reward(
            trajectory, task_spec, container_id, docker_manager,
        )

        # 2. Efficiency penalty (always computed, cheap)
        efficiency_pen = compute_efficiency_penalty(trajectory)

        # 3. Process reward (expensive, optionally computed)
        process_score = 0.0
        per_step_ratings: list[dict[str, Any]] = []

        if use_process_reward:
            per_step_ratings = await compute_process_reward(
                trajectory, task_spec, opus_client,
            )
            process_score = average_process_score(per_step_ratings)

        # 4. Composite score
        composite = (
            self.weights.outcome * outcome
            + self.weights.process * process_score
            - self.weights.efficiency * efficiency_pen
        )
        # Clamp to [0, 1]
        composite = max(0.0, min(1.0, composite))

        # Build per-step reward list from process ratings
        per_step_rewards = _extract_per_step_rewards(per_step_ratings, len(trajectory.steps))

        # Assemble details dict
        details: dict[str, Any] = {
            "weights": {
                "outcome": self.weights.outcome,
                "process": self.weights.process,
                "efficiency": self.weights.efficiency,
            },
            "process_reward_used": use_process_reward,
            "per_step_ratings": per_step_ratings,
        }

        logger.info(
            "Composite reward for %s: %.3f (outcome=%.3f, process=%.3f, penalty=%.3f)",
            trajectory.trajectory_id, composite, outcome, process_score, efficiency_pen,
        )

        return RewardResult(
            outcome_reward=round(outcome, 4),
            process_reward=round(process_score, 4),
            efficiency_penalty=round(efficiency_pen, 4),
            composite=round(composite, 4),
            per_step_rewards=per_step_rewards,
            details=details,
        )

    def should_use_process_reward(self) -> bool:
        """Randomly decide whether to invoke the process reward.

        Returns ``True`` with probability equal to ``self.sample_rate``.
        This allows amortizing the cost of Opus calls across many trajectories.

        Returns
        -------
        bool
            Whether to compute the process reward for the current trajectory.
        """
        return random.random() < self.sample_rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_per_step_rewards(
    per_step_ratings: list[dict[str, Any]],
    num_steps: int,
) -> list[float]:
    """Convert per-step ratings into a simple list of floats.

    Each step's reward is the average of its four dimension scores.
    Steps without ratings get 0.0.
    """
    rewards = [0.0] * num_steps

    dims = ("correctness", "efficiency", "reasoning_quality", "info_utilization")
    for rating in per_step_ratings:
        if rating.get("__summary__"):
            continue
        idx = rating.get("step_idx")
        if idx is not None and 0 <= idx < num_steps:
            step_avg = sum(rating.get(d, 0.0) for d in dims) / len(dims)
            rewards[idx] = round(step_avg, 4)

    return rewards
