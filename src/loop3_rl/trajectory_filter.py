"""RAGEN-style trajectory filtering to prevent echo traps and degenerate rollouts.

Before feeding trajectories into RL training, this module removes degenerate
examples that would poison the gradient signal: echo traps (same action
repeated endlessly), format-collapsed outputs, excessively low-reward
episodes, and trajectories lacking action diversity.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from src.config import Loop3Config
from src.models import ActionType, StepRecord, TaskSpec, Trajectory, RewardResult

logger = logging.getLogger(__name__)


class TrajectoryFilter:
    """Filters degenerate trajectories from RL training batches.

    Parameters
    ----------
    config:
        Loop 3 configuration providing thresholds for echo detection,
        minimum reward, etc.
    """

    def __init__(self, config: Loop3Config) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def filter_batch(
        self,
        trajectories: list[Trajectory],
        rewards: list[float],
    ) -> tuple[list[Trajectory], list[float]]:
        """Remove degenerate trajectories from a training batch.

        Parameters
        ----------
        trajectories:
            Candidate trajectories.
        rewards:
            Per-trajectory scalar rewards (same length as *trajectories*).

        Returns
        -------
        tuple[list[Trajectory], list[float]]
            Filtered trajectories and their corresponding rewards.
        """
        before_count = len(trajectories)
        kept_trajs: list[Trajectory] = []
        kept_rewards: list[float] = []

        for traj, reward in zip(trajectories, rewards):
            if self.is_echo_trap(traj):
                logger.debug(
                    "Filtered echo-trap trajectory %s", traj.trajectory_id
                )
                continue

            if self.is_degenerate(traj, reward):
                logger.debug(
                    "Filtered degenerate trajectory %s (reward=%.4f)",
                    traj.trajectory_id,
                    reward,
                )
                continue

            if self.is_format_collapsed(traj):
                logger.debug(
                    "Filtered format-collapsed trajectory %s",
                    traj.trajectory_id,
                )
                continue

            if not self.has_diversity(traj):
                logger.debug(
                    "Filtered low-diversity trajectory %s",
                    traj.trajectory_id,
                )
                continue

            kept_trajs.append(traj)
            kept_rewards.append(reward)

        after_count = len(kept_trajs)
        stats = self.filter_stats(before_count, after_count)
        logger.info(
            "Trajectory filter: kept %d/%d (%.1f%% filtered)",
            after_count,
            before_count,
            stats["filtered_pct"],
        )

        return kept_trajs, kept_rewards

    # ------------------------------------------------------------------
    # Detection criteria
    # ------------------------------------------------------------------

    def is_echo_trap(self, trajectory: Trajectory) -> bool:
        """Detect echo traps: same action repeated consecutively.

        An echo trap occurs when the same (action_type, action_content)
        pair appears ``>= echo_trap_threshold`` times in a row.

        Parameters
        ----------
        trajectory:
            The trajectory to check.

        Returns
        -------
        bool
            ``True`` if an echo trap is detected.
        """
        threshold = self.config.echo_trap_threshold
        if len(trajectory.steps) < threshold:
            return False

        streak = 1
        for i in range(1, len(trajectory.steps)):
            prev = trajectory.steps[i - 1]
            curr = trajectory.steps[i]
            if (
                curr.action_type == prev.action_type
                and curr.action_content.strip() == prev.action_content.strip()
            ):
                streak += 1
                if streak >= threshold:
                    return True
            else:
                streak = 1

        return False

    def is_degenerate(
        self,
        trajectory: Trajectory,
        reward: float | None = None,
    ) -> bool:
        """Check whether a trajectory is degenerate based on reward.

        Parameters
        ----------
        trajectory:
            The trajectory to check.
        reward:
            External reward value.  If ``None``, falls back to
            ``trajectory.total_reward``.

        Returns
        -------
        bool
            ``True`` if the reward is below ``config.min_trajectory_reward``.
        """
        r = reward if reward is not None else trajectory.total_reward
        return r < self.config.min_trajectory_reward

    def is_format_collapsed(self, trajectory: Trajectory) -> bool:
        """Detect format collapse: all assistant outputs follow an identical template.

        If every step's ``action_content`` is identical (modulo leading/
        trailing whitespace) and there are at least 3 steps, the trajectory
        is considered format-collapsed.

        Additionally checks for near-identical outputs by normalising
        whitespace and comparing.

        Parameters
        ----------
        trajectory:
            The trajectory to check.

        Returns
        -------
        bool
            ``True`` if format collapse is detected.
        """
        steps = trajectory.steps
        if len(steps) < 3:
            return False

        # Check for exact duplicates
        contents = [s.action_content.strip() for s in steps]
        unique_contents = set(contents)
        if len(unique_contents) == 1:
            return True

        # Check for near-identical outputs (normalise whitespace)
        _ws = re.compile(r"\s+")
        normalised = {_ws.sub(" ", c) for c in contents}
        if len(normalised) == 1:
            return True

        # Check for repeated exact phrases across all outputs
        # If > 80% of outputs match a single template, flag it
        content_counts = Counter(contents)
        most_common_count = content_counts.most_common(1)[0][1]
        if most_common_count / len(contents) > 0.8:
            return True

        return False

    def has_diversity(self, trajectory: Trajectory) -> bool:
        """Check whether the trajectory uses at least 2 different action types.

        Parameters
        ----------
        trajectory:
            The trajectory to check.

        Returns
        -------
        bool
            ``True`` if at least 2 distinct action types are used.
        """
        if len(trajectory.steps) <= 1:
            # Single-step trajectories get a pass on diversity
            return True

        action_types_used = {s.action_type for s in trajectory.steps}
        return len(action_types_used) >= 2

    # ------------------------------------------------------------------
    # Scoring and stats
    # ------------------------------------------------------------------

    @staticmethod
    def compute_diversity_score(trajectory: Trajectory) -> float:
        """Compute a diversity score for a trajectory.

        Higher scores indicate more diverse action usage.  The score is
        based on the normalised entropy of the action-type distribution.

        Parameters
        ----------
        trajectory:
            The trajectory to score.

        Returns
        -------
        float
            Diversity score in ``[0.0, 1.0]``.  1.0 = maximum diversity.
        """
        import math

        if not trajectory.steps:
            return 0.0

        counts = Counter(s.action_type for s in trajectory.steps)
        total = len(trajectory.steps)
        n_types = len(counts)

        if n_types <= 1:
            return 0.0

        # Shannon entropy
        entropy = -sum(
            (c / total) * math.log2(c / total) for c in counts.values()
        )
        # Normalise by max possible entropy
        max_entropy = math.log2(n_types)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    @staticmethod
    def filter_stats(before_count: int, after_count: int) -> dict[str, Any]:
        """Produce logging-friendly statistics about filtering.

        Parameters
        ----------
        before_count:
            Number of trajectories before filtering.
        after_count:
            Number of trajectories after filtering.

        Returns
        -------
        dict
            Dictionary with ``before``, ``after``, ``removed``, and
            ``filtered_pct`` keys.
        """
        removed = before_count - after_count
        pct = (removed / before_count * 100) if before_count > 0 else 0.0
        return {
            "before": before_count,
            "after": after_count,
            "removed": removed,
            "filtered_pct": round(pct, 2),
        }
