"""Capability boundary detection via binary-search probing.

Determines the difficulty threshold at which the agent's success rate
drops below 50% for each capability dimension, giving a precise map
of the agent's frontier.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.config import ArenaConfig, MasterConfig, load_config
from src.models import TaskSpec, TaskType
from src.orchestrator.opus_client import OpusClient

logger = logging.getLogger(__name__)

# Capability dimensions to probe
DIMENSIONS = [
    "file_manipulation",
    "code_debugging",
    "api_usage",
    "error_recovery",
    "multi_step_planning",
]

# Map dimensions to TaskType for task generation
_DIMENSION_TASK_TYPE: dict[str, TaskType] = {
    "file_manipulation": TaskType.CODE_DEBUGGING,
    "code_debugging": TaskType.CODE_DEBUGGING,
    "api_usage": TaskType.API_ORCHESTRATION,
    "error_recovery": TaskType.CODE_DEBUGGING,
    "multi_step_planning": TaskType.OPEN_ENDED,
}

# Binary search parameters
DEFAULT_TOLERANCE = 0.05
DEFAULT_MAX_ITERATIONS = 8
DEFAULT_TRIALS_PER_PROBE = 4
DEFAULT_CACHE_TTL_SECONDS = 3600  # 1 hour


@dataclass
class ProbeResult:
    """Result of probing a single dimension."""

    dimension: str
    frontier_difficulty: float
    confidence: float
    trials_run: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class _CacheEntry:
    """Cached probe result with expiry."""

    result: float
    timestamp: float
    ttl: float

    @property
    def expired(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl


class FrontierProber:
    """Detects the agent's capability boundary via binary search.

    For each dimension, performs a binary search over difficulty levels
    to find the threshold where the agent's success rate drops below
    50%.

    Parameters
    ----------
    opus_client:
        OpusClient for generating probe tasks on the fly.
    cache_ttl:
        Time-to-live in seconds for cached probe results.
    tolerance:
        Binary search terminates when the search interval is smaller
        than this value.
    max_iterations:
        Maximum binary search iterations per dimension.
    trials_per_probe:
        Number of episodes to run at each difficulty level to estimate
        success rate.
    """

    def __init__(
        self,
        opus_client: OpusClient,
        *,
        cache_ttl: float = DEFAULT_CACHE_TTL_SECONDS,
        tolerance: float = DEFAULT_TOLERANCE,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        trials_per_probe: int = DEFAULT_TRIALS_PER_PROBE,
    ) -> None:
        self._opus = opus_client
        self._cache_ttl = cache_ttl
        self._tolerance = tolerance
        self._max_iterations = max_iterations
        self._trials_per_probe = trials_per_probe
        self._cache: dict[str, _CacheEntry] = {}

    async def probe_dimension(
        self,
        dimension: str,
        vllm_server: Any,
        arena_manager: Any,
    ) -> float:
        """Binary search for the difficulty where success drops below 50%.

        Parameters
        ----------
        dimension:
            Capability dimension to probe (e.g. ``"code_debugging"``).
        vllm_server:
            The VLLMServer instance for running agent inference.
        arena_manager:
            Object that can run episodes. Must support an
            ``async run_episode(task: TaskSpec, vllm_server) -> bool``
            method returning success/failure.

        Returns
        -------
        float
            The estimated difficulty frontier (0.0-1.0) for this
            dimension.
        """
        # Check cache
        cached = self._cache.get(dimension)
        if cached is not None and not cached.expired:
            logger.debug(
                "Using cached frontier for %s: %.3f", dimension, cached.result
            )
            return cached.result

        logger.info("Probing frontier for dimension: %s", dimension)

        low = 0.0
        high = 1.0
        task_type = _DIMENSION_TASK_TYPE.get(dimension, TaskType.CODE_DEBUGGING)

        for iteration in range(self._max_iterations):
            mid = (low + high) / 2.0
            logger.debug(
                "  %s iteration %d: probing difficulty=%.3f [%.3f, %.3f]",
                dimension, iteration, mid, low, high,
            )

            # Generate a probe task at this difficulty
            tasks = await self._generate_probe_tasks(
                dimension, task_type, mid, self._trials_per_probe
            )

            if not tasks:
                logger.warning("Failed to generate probe tasks; using midpoint")
                break

            # Run trials
            successes = 0
            for task in tasks:
                try:
                    success = await arena_manager.run_episode(task, vllm_server)
                    if success:
                        successes += 1
                except Exception:
                    logger.warning(
                        "Probe trial failed for %s at difficulty %.2f",
                        dimension, mid,
                        exc_info=True,
                    )

            success_rate = successes / len(tasks)
            logger.debug(
                "  %s at difficulty %.3f: %d/%d success (%.1f%%)",
                dimension, mid, successes, len(tasks), success_rate * 100,
            )

            # Binary search logic
            if success_rate >= 0.5:
                low = mid  # Agent can handle this; try harder
            else:
                high = mid  # Too hard; try easier

            if (high - low) < self._tolerance:
                break

        frontier = (low + high) / 2.0
        logger.info("Frontier for %s: %.3f", dimension, frontier)

        # Cache the result
        self._cache[dimension] = _CacheEntry(
            result=frontier,
            timestamp=time.time(),
            ttl=self._cache_ttl,
        )

        return frontier

    async def probe_all_dimensions(
        self,
        vllm_server: Any,
        arena_manager: Any,
    ) -> dict[str, float]:
        """Probe the capability frontier across all dimensions.

        Runs probes sequentially to avoid overloading the inference
        server and arena.

        Parameters
        ----------
        vllm_server:
            The VLLMServer instance.
        arena_manager:
            The arena manager for running episodes.

        Returns
        -------
        dict[str, float]
            Mapping of dimension name to frontier difficulty level.
        """
        profile: dict[str, float] = {}

        for dimension in DIMENSIONS:
            try:
                frontier = await self.probe_dimension(
                    dimension, vllm_server, arena_manager
                )
                profile[dimension] = frontier
            except Exception:
                logger.error(
                    "Failed to probe dimension %s", dimension, exc_info=True
                )
                profile[dimension] = 0.5  # Default midpoint on failure

        logger.info("Full capability profile: %s", profile)
        return profile

    def invalidate_cache(self, dimension: str | None = None) -> None:
        """Clear cached probe results.

        Parameters
        ----------
        dimension:
            If provided, only clear this dimension. Otherwise clear all.
        """
        if dimension is not None:
            self._cache.pop(dimension, None)
        else:
            self._cache.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _generate_probe_tasks(
        self,
        dimension: str,
        task_type: TaskType,
        target_difficulty: float,
        count: int,
    ) -> list[TaskSpec]:
        """Generate probe tasks at a specific difficulty level."""
        schema = {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "initial_files": {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                            },
                            "test_commands": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "expected_output": {"type": ["string", "null"]},
                        },
                        "required": ["description"],
                    },
                },
            },
            "required": ["tasks"],
        }

        prompt = (
            f"Generate exactly {count} evaluation tasks for an AI coding agent.\n\n"
            f"**Dimension being tested:** {dimension}\n"
            f"**Target difficulty:** {target_difficulty:.2f} (0.0=trivial, 1.0=expert)\n"
            f"**Task type:** {task_type.value}\n\n"
            f"Each task should specifically test '{dimension}' ability at "
            f"difficulty level {target_difficulty:.2f}. Include initial_files "
            f"and test_commands for each task.\n"
        )

        resp = await self._opus.query(
            prompt,
            json_schema=schema,
            max_budget_usd=0.30,
        )

        if resp.is_error:
            logger.error("Probe task generation failed: %s", resp.error_message)
            return []

        tasks_data = resp.raw_json.get("tasks", [])
        specs: list[TaskSpec] = []
        for t in tasks_data:
            spec = TaskSpec(
                task_type=task_type,
                description=t.get("description", ""),
                initial_files=t.get("initial_files", {}),
                test_commands=t.get("test_commands", []),
                expected_output=t.get("expected_output"),
                difficulty=target_difficulty,
                metadata={
                    "probe_dimension": dimension,
                    "probe_target_difficulty": target_difficulty,
                },
            )
            specs.append(spec)

        return specs
