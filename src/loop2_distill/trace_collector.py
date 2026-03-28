"""Collects Qwen rollouts and identifies failure points in trajectories.

Runs the agent through tasks concurrently, records full trajectories,
and applies heuristics to locate and classify failure modes.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Any

from src.models import (
    ActionType,
    PromptGenome,
    StepRecord,
    TaskSpec,
    Trajectory,
    TraceAnalysis,
)
from src.arena.game import AgentArenaGame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Failure classification labels
# ---------------------------------------------------------------------------

FAILURE_WRONG_TOOL = "wrong_tool"
FAILURE_REASONING_ERROR = "reasoning_error"
FAILURE_HALLUCINATION = "hallucination"
FAILURE_LOOP = "loop"
FAILURE_TIMEOUT = "timeout"
FAILURE_INCORRECT_OUTPUT = "incorrect_output"

# Thresholds for heuristic detection
_LOOP_REPEAT_THRESHOLD = 3
_REASONING_CONTRADICTION_KEYWORDS = [
    "but earlier", "contradicts", "wait, actually", "no,", "I was wrong",
]


class TraceCollector:
    """Collects Qwen rollouts and identifies where trajectories fail.

    Parameters
    ----------
    concurrency:
        Maximum number of rollouts to execute in parallel.
    timeout_seconds:
        Per-episode wall-clock timeout before marking as timed out.
    """

    def __init__(
        self,
        concurrency: int = 4,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.concurrency = concurrency
        self.timeout_seconds = timeout_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def collect_rollouts(
        self,
        tasks: list[TaskSpec],
        n_per_task: int,
        vllm_server: Any,
        arena_manager: Any,
        genome: PromptGenome,
    ) -> list[Trajectory]:
        """Run Qwen through each task ``n_per_task`` times and collect trajectories.

        Parameters
        ----------
        tasks:
            List of task specifications to evaluate.
        n_per_task:
            Number of rollouts per task.
        vllm_server:
            A running VLLMServer instance for Qwen inference.
        arena_manager:
            DockerManager instance for container management.
        genome:
            The current PromptGenome to use as the system prompt.

        Returns
        -------
        list[Trajectory]
            All collected trajectories across tasks and rollouts.
        """
        # Build the full list of (task, rollout_index) pairs
        work_items: list[tuple[TaskSpec, int]] = []
        for task in tasks:
            for i in range(n_per_task):
                work_items.append((task, i))

        logger.info(
            "Collecting %d rollouts across %d tasks (concurrency=%d)",
            len(work_items),
            len(tasks),
            self.concurrency,
        )

        semaphore = asyncio.Semaphore(self.concurrency)
        trajectories: list[Trajectory] = []

        async def _run_one(task: TaskSpec, rollout_idx: int) -> Trajectory | None:
            async with semaphore:
                return await self._execute_rollout(
                    task, vllm_server, arena_manager, genome,
                )

        coros = [_run_one(task, idx) for task, idx in work_items]
        results = await asyncio.gather(*coros, return_exceptions=True)

        for result in results:
            if isinstance(result, Trajectory):
                trajectories.append(result)
            elif isinstance(result, Exception):
                logger.error("Rollout failed with exception: %s", result)

        logger.info(
            "Collected %d trajectories (%d failed)",
            len(trajectories),
            len(work_items) - len(trajectories),
        )
        return trajectories

    @staticmethod
    def identify_failure_points(trajectory: Trajectory) -> list[int]:
        """Identify step indices where the trajectory likely went wrong.

        Applies multiple heuristics and returns a sorted, deduplicated
        list of step indices.

        Heuristics
        ----------
        1. Step before first stderr error.
        2. Step where reasoning contradicts prior evidence.
        3. Step where the agent loops (repeats the same action).
        4. Last step before timeout if the episode timed out.
        """
        failure_indices: set[int] = set()
        steps = trajectory.steps

        if not steps:
            return []

        # 1. Step before first error in stderr
        for step in steps:
            stderr = step.metadata.get("stderr", "")
            if stderr and step.reward < 0:
                idx = max(0, step.step_idx - 1)
                failure_indices.add(idx)
                break

        # 2. Step where reasoning contradicts evidence
        for step in steps:
            reasoning_lower = step.reasoning.lower()
            for keyword in _REASONING_CONTRADICTION_KEYWORDS:
                if keyword in reasoning_lower:
                    failure_indices.add(step.step_idx)
                    break

        # 3. Step where the agent loops / repeats the same action
        if len(steps) >= _LOOP_REPEAT_THRESHOLD:
            for i in range(len(steps) - _LOOP_REPEAT_THRESHOLD + 1):
                window = steps[i : i + _LOOP_REPEAT_THRESHOLD]
                actions = [
                    (s.action_type, s.action_content) for s in window
                ]
                if len(set(actions)) == 1:
                    # The first repeated step is the failure point
                    failure_indices.add(window[0].step_idx)

        # 4. Last step before timeout
        timed_out = (
            not trajectory.success
            and trajectory.steps
            and trajectory.steps[-1].metadata.get("reason") == "max_tool_calls_exceeded"
        )
        if timed_out and len(steps) >= 2:
            failure_indices.add(steps[-2].step_idx)

        return sorted(failure_indices)

    @staticmethod
    def classify_failure(trajectory: Trajectory) -> str:
        """Classify the dominant failure mode of a trajectory.

        Returns one of: wrong_tool, reasoning_error, hallucination,
        loop, timeout, incorrect_output.
        """
        steps = trajectory.steps

        if not steps:
            return FAILURE_INCORRECT_OUTPUT

        # Check for timeout
        last = steps[-1]
        if last.metadata.get("reason") == "max_tool_calls_exceeded":
            return FAILURE_TIMEOUT

        # Check for loops
        if len(steps) >= _LOOP_REPEAT_THRESHOLD:
            for i in range(len(steps) - _LOOP_REPEAT_THRESHOLD + 1):
                window = steps[i : i + _LOOP_REPEAT_THRESHOLD]
                actions = [
                    (s.action_type, s.action_content) for s in window
                ]
                if len(set(actions)) == 1:
                    return FAILURE_LOOP

        # Check for hallucination: agent references things not in observations
        for step in steps:
            reasoning_lower = step.reasoning.lower()
            if any(
                phrase in reasoning_lower
                for phrase in [
                    "i can see that", "the output shows", "as shown above",
                ]
            ):
                # If the preceding observation is empty, it is a hallucination
                if not step.observation.strip():
                    return FAILURE_HALLUCINATION

        # Check for wrong tool usage
        for step in steps:
            stderr = step.metadata.get("stderr", "")
            if "command not found" in stderr or "unknown action" in stderr:
                return FAILURE_WRONG_TOOL

        # Check for reasoning contradictions
        for step in steps:
            reasoning_lower = step.reasoning.lower()
            for keyword in _REASONING_CONTRADICTION_KEYWORDS:
                if keyword in reasoning_lower:
                    return FAILURE_REASONING_ERROR

        # Default: the output was simply wrong
        return FAILURE_INCORRECT_OUTPUT

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _execute_rollout(
        self,
        task: TaskSpec,
        vllm_server: Any,
        arena_manager: Any,
        genome: PromptGenome,
    ) -> Trajectory:
        """Execute a single rollout of Qwen on a task.

        Uses AgentArenaGame for the episode and drives Qwen via the
        vLLM chat completion API.
        """
        game = AgentArenaGame(arena_mgr=arena_manager)

        async with game:
            initial_obs = await game.reset(task)

            # Build the conversation history
            messages: list[dict[str, str]] = [
                {"role": "system", "content": genome.to_system_message()},
                {"role": "user", "content": initial_obs},
            ]

            timed_out = False
            try:
                timed_out = await asyncio.wait_for(
                    self._run_episode_loop(game, messages, vllm_server),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                timed_out = True
                logger.warning(
                    "Rollout timed out for task %s after %.0fs",
                    task.task_id,
                    self.timeout_seconds,
                )

            trajectory = game.get_trajectory()
            trajectory.model_id = vllm_server.config.name
            trajectory.prompt_genome_id = genome.genome_id

            return trajectory

    async def _run_episode_loop(
        self,
        game: AgentArenaGame,
        messages: list[dict[str, str]],
        vllm_server: Any,
    ) -> bool:
        """Drive the agent loop until done or tool call limit.

        Returns True if the episode ended normally, False otherwise.
        """
        max_iterations = 50  # Safety cap

        for _ in range(max_iterations):
            # Get Qwen's next action via vLLM
            completion = await vllm_server.chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )

            assistant_msg = completion.choices[0].message.content or ""
            if not assistant_msg.strip():
                logger.debug(
                    "Empty LLM response (raw choice: %s)",
                    repr(completion.choices[0])[:200],
                )
            messages.append({"role": "assistant", "content": assistant_msg})

            # Execute the action in the arena
            step_result = await game.step(assistant_msg)

            if step_result.observation:
                messages.append(
                    {"role": "user", "content": step_result.observation}
                )

            if step_result.done:
                return True

        return False
