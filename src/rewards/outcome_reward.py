"""Automated, verifiable outcome reward -- no Opus calls required.

Computes a scalar reward in [0.0, 1.0] by running task-specific verification
logic inside the Docker container or by comparing submitted answers.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from src.models import ActionType, TaskSpec, Trajectory

logger = logging.getLogger(__name__)


async def compute_outcome_reward(
    trajectory: Trajectory,
    task_spec: TaskSpec,
    container_id: str,
    docker_manager,
) -> float:
    """Compute an automated outcome reward for a completed trajectory.

    The reward strategy depends on the task type:

    * **code_debugging** -- run ``task_spec.test_commands`` in the container and
      return the fraction of commands that exit 0.
    * **info_gathering** -- compare the agent's submitted answer to
      ``task_spec.expected_output`` using exact or fuzzy matching.
    * **api_orchestration** -- check whether the submitted answer matches the
      expected output.
    * **open_ended** -- measure a speedup ratio by running a benchmark command
      before/after the agent's changes.

    Parameters
    ----------
    trajectory:
        The agent's completed trajectory (used to extract the submitted answer).
    task_spec:
        The task specification with test commands and expected outputs.
    container_id:
        ID of the Docker container where the task was executed.
    docker_manager:
        A :class:`DockerManager` instance for running commands in the container.

    Returns
    -------
    float
        Reward in [0.0, 1.0] with support for partial credit.
    """
    task_type = task_spec.task_type.value

    if task_type == "code_debugging":
        return await _reward_code_debugging(task_spec, container_id, docker_manager)
    elif task_type == "info_gathering":
        return _reward_info_gathering(trajectory, task_spec)
    elif task_type == "api_orchestration":
        return _reward_api_orchestration(trajectory, task_spec)
    elif task_type == "open_ended_optimization":
        return await _reward_open_ended(task_spec, container_id, docker_manager)
    else:
        logger.warning("Unknown task type %r; returning 0.0", task_type)
        return 0.0


# ---------------------------------------------------------------------------
# Task-type specific strategies
# ---------------------------------------------------------------------------


async def _reward_code_debugging(
    task_spec: TaskSpec,
    container_id: str,
    docker_manager,
) -> float:
    """Run test commands and return fraction passing (exit code 0)."""
    if not task_spec.test_commands:
        logger.warning("code_debugging task has no test_commands; returning 0.0")
        return 0.0

    passed = 0
    total = len(task_spec.test_commands)

    for cmd in task_spec.test_commands:
        try:
            stdout, stderr, exit_code = await docker_manager.async_exec_in_container(
                container_id, cmd, timeout=30,
            )
            if exit_code == 0:
                passed += 1
            else:
                logger.debug("Test command failed (exit %d): %s", exit_code, cmd)
        except TimeoutError:
            logger.warning("Test command timed out: %s", cmd)
        except Exception:
            logger.exception("Error running test command: %s", cmd)

    return passed / total


async def _reward_open_ended(
    task_spec: TaskSpec,
    container_id: str,
    docker_manager,
) -> float:
    """Measure speedup ratio for open-ended optimization tasks.

    Expects ``task_spec.metadata`` to contain:
    * ``benchmark_command`` -- the command to time.
    * ``baseline_seconds`` -- the pre-optimization baseline (float).

    If the baseline is not provided, we run the benchmark command once first
    to establish it.  The reward is ``min(1.0, baseline / post)``.
    """
    benchmark_cmd = task_spec.metadata.get("benchmark_command")
    if not benchmark_cmd:
        logger.warning("open_ended task missing benchmark_command; returning 0.0")
        return 0.0

    baseline = task_spec.metadata.get("baseline_seconds")

    if baseline is None:
        baseline = await _time_command(benchmark_cmd, container_id, docker_manager)
        if baseline is None:
            return 0.0

    post = await _time_command(benchmark_cmd, container_id, docker_manager)
    if post is None or post <= 0:
        return 0.0

    speedup = float(baseline) / post
    return min(1.0, speedup)


async def _time_command(
    command: str,
    container_id: str,
    docker_manager,
) -> float | None:
    """Run a command and return its wall-clock duration in seconds."""
    timed_cmd = f"bash -c 'start=$(date +%s%N); {command}; end=$(date +%s%N); echo \"__ELAPSED_NS__=$(($end - $start))\"'"
    try:
        stdout, _, exit_code = await docker_manager.async_exec_in_container(
            container_id, timed_cmd, timeout=60,
        )
        match = re.search(r"__ELAPSED_NS__=(\d+)", stdout)
        if match:
            return int(match.group(1)) / 1_000_000_000
    except Exception:
        logger.exception("Failed to time command: %s", command)
    return None


# ---------------------------------------------------------------------------
# Answer-comparison strategies
# ---------------------------------------------------------------------------


def _extract_submitted_answer(trajectory: Trajectory) -> str | None:
    """Extract the answer from the last submit action in the trajectory."""
    for step in reversed(trajectory.steps):
        if step.action_type == ActionType.SUBMIT:
            return step.action_content.strip()
    return None


def _reward_info_gathering(trajectory: Trajectory, task_spec: TaskSpec) -> float:
    """Compare submitted answer to expected output (exact or fuzzy)."""
    submitted = _extract_submitted_answer(trajectory)
    expected = task_spec.expected_output

    if submitted is None:
        logger.debug("No submit action found in trajectory")
        return 0.0
    if expected is None:
        logger.warning("info_gathering task has no expected_output; returning 0.0")
        return 0.0

    # Exact match (case-insensitive, whitespace-normalized)
    norm_sub = _normalize(submitted)
    norm_exp = _normalize(expected)

    if norm_sub == norm_exp:
        return 1.0

    # Fuzzy match -- partial credit based on sequence similarity
    ratio = SequenceMatcher(None, norm_sub, norm_exp).ratio()
    return round(ratio, 4)


def _reward_api_orchestration(trajectory: Trajectory, task_spec: TaskSpec) -> float:
    """Check if the submitted answer matches expected output for API tasks."""
    submitted = _extract_submitted_answer(trajectory)
    expected = task_spec.expected_output

    if submitted is None:
        return 0.0
    if expected is None:
        logger.warning("api_orchestration task has no expected_output; returning 0.0")
        return 0.0

    norm_sub = _normalize(submitted)
    norm_exp = _normalize(expected)

    if norm_sub == norm_exp:
        return 1.0

    # Partial credit: check if expected is contained within the submission
    if norm_exp in norm_sub:
        return 0.8

    ratio = SequenceMatcher(None, norm_sub, norm_exp).ratio()
    return round(ratio, 4)


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip."""
    return " ".join(text.lower().split())
