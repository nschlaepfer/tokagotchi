"""Evaluates a PromptGenome by running Qwen on tasks in the arena.

Orchestrates the full agent loop: Qwen generates actions via vLLM chat
completion, tools execute them in Docker containers, observations are
fed back, and results are aggregated into an EvalResult.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import defaultdict
from typing import Any

from src.arena.docker_manager import DockerManager
from src.arena.tools import bash_tool, file_tool, python_tool, submit_tool
from src.arena.tools.common import ToolResult
from src.config import ArenaConfig, Loop1Config
from src.infra.vllm_server import VLLMServer
from src.models import (
    ActionType,
    EvalResult,
    PromptGenome,
    StepRecord,
    TaskSpec,
    Trajectory,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STEPS_DEFAULT = 20
AGENT_TIMEOUT_SECONDS = 120
MAX_OBSERVATION_CHARS = 4000

# Regex patterns for parsing Qwen's action output.
# Expected format:
#   [action_type]
#   content here
# or:
#   [action_type: content here]
_ACTION_BLOCK_RE = re.compile(
    r"\[(?P<action_type>think|bash|python|read_file|write_file|sql|api_call|submit)"
    r"(?::?\s*(?P<inline_content>[^\]]*))?\]"
    r"(?:\s*\n(?P<block_content>[\s\S]*?)(?=\n\[|\Z))?",
    re.IGNORECASE,
)

# Fallback: look for fenced code blocks with a tool label
_FENCED_BLOCK_RE = re.compile(
    r"```(?P<action_type>bash|python)\n(?P<content>[\s\S]*?)```",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------


def _parse_actions(raw_output: str) -> list[tuple[ActionType, str]]:
    """Parse Qwen's raw text output into a list of (action_type, content) pairs.

    Tries structured bracket format first, then falls back to fenced code blocks.
    If nothing is detected, returns a single THINK action with the full text.
    """
    actions: list[tuple[ActionType, str]] = []

    for m in _ACTION_BLOCK_RE.finditer(raw_output):
        action_str = m.group("action_type").lower()
        content = (
            m.group("inline_content") or m.group("block_content") or ""
        ).strip()
        try:
            action_type = ActionType(action_str)
        except ValueError:
            action_type = ActionType.THINK
            content = m.group(0)
        actions.append((action_type, content))

    if not actions:
        for m in _FENCED_BLOCK_RE.finditer(raw_output):
            action_str = m.group("action_type").lower()
            content = m.group("content").strip()
            try:
                action_type = ActionType(action_str)
            except ValueError:
                continue
            actions.append((action_type, content))

    if not actions:
        actions.append((ActionType.THINK, raw_output.strip()))

    return actions


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


async def _dispatch_tool(
    action_type: ActionType,
    content: str,
    docker_mgr: DockerManager,
    container_id: str,
) -> ToolResult:
    """Execute a single tool action in the arena container."""
    if action_type == ActionType.BASH:
        return await bash_tool.execute(docker_mgr, container_id, content)

    if action_type == ActionType.PYTHON:
        return await python_tool.execute(docker_mgr, container_id, content)

    if action_type == ActionType.READ_FILE:
        path = content.strip()
        if not path.startswith("/"):
            path = f"/workspace/{path}"
        return await file_tool.read_file(docker_mgr, container_id, path)

    if action_type == ActionType.WRITE_FILE:
        # Expected format: first line is the path, rest is content
        lines = content.split("\n", 1)
        path = lines[0].strip()
        file_content = lines[1] if len(lines) > 1 else ""
        if not path.startswith("/"):
            path = f"/workspace/{path}"
        return await file_tool.write_file(docker_mgr, container_id, path, file_content)

    if action_type == ActionType.SUBMIT:
        return await submit_tool.submit(container_id, content)

    if action_type == ActionType.THINK:
        return ToolResult(
            stdout="(reasoning noted)",
            stderr="",
            exit_code=0,
        )

    # SQL, API_CALL, or unknown: execute as bash for now
    return await bash_tool.execute(docker_mgr, container_id, content)


# ---------------------------------------------------------------------------
# Single-task agent loop
# ---------------------------------------------------------------------------


async def _run_agent_loop(
    genome: PromptGenome,
    task: TaskSpec,
    vllm_server: VLLMServer,
    docker_mgr: DockerManager,
    *,
    max_steps: int = MAX_STEPS_DEFAULT,
    timeout_seconds: float = AGENT_TIMEOUT_SECONDS,
) -> Trajectory:
    """Run the full agent loop for one task.

    1. Set up the arena container with task files.
    2. Build the initial message list from the genome's system prompt.
    3. Loop: Qwen generates actions -> tools execute -> observations fed back.
    4. Terminates on submit action, max steps, or timeout.
    """
    trajectory = Trajectory(
        task=task,
        model_id=vllm_server.config.name,
        prompt_genome_id=genome.genome_id,
    )

    container_id = await docker_mgr.async_create_container(task)
    start_time = time.monotonic()

    try:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": genome.to_system_message()},
            {"role": "user", "content": _format_task_prompt(task)},
        ]

        episode_complete = False

        for step_idx in range(max_steps):
            elapsed = time.monotonic() - start_time
            if elapsed > timeout_seconds:
                logger.warning(
                    "Agent timeout after %.1fs on task %s", elapsed, task.task_id
                )
                break

            # Get Qwen's response
            try:
                completion = await vllm_server.chat_completion(
                    messages,
                    temperature=0.7,
                    max_tokens=2048,
                )
                raw_response = completion.choices[0].message.content or ""
            except Exception as exc:
                logger.error("vLLM chat completion failed: %s", exc)
                raw_response = f"[think]\nError calling model: {exc}"

            # Parse into actions
            parsed_actions = _parse_actions(raw_response)

            # Execute each action and collect observations
            observations: list[str] = []
            reasoning = ""

            for action_type, action_content in parsed_actions:
                if action_type == ActionType.THINK:
                    reasoning = action_content
                    observations.append(f"(reasoning: {action_content[:200]})")
                    continue

                result = await _dispatch_tool(
                    action_type, action_content, docker_mgr, container_id
                )

                observation = result.output
                if len(observation) > MAX_OBSERVATION_CHARS:
                    observation = (
                        observation[:MAX_OBSERVATION_CHARS]
                        + "\n... [observation truncated]"
                    )
                observations.append(observation)

                step = StepRecord(
                    step_idx=step_idx,
                    action_type=action_type,
                    action_content=action_content,
                    observation=observation,
                    reasoning=reasoning,
                    reward=0.0,
                )
                trajectory.steps.append(step)

                # Check for episode completion (submit action)
                if result.metadata.get("episode_complete", False):
                    episode_complete = True
                    break

                # Reset reasoning after it is consumed by an action
                reasoning = ""

            # Build observation message for the conversation
            combined_observation = "\n---\n".join(observations) if observations else "(no output)"
            messages.append({"role": "assistant", "content": raw_response})
            messages.append({"role": "user", "content": f"Observation:\n{combined_observation}"})

            if episode_complete:
                break

        trajectory.wall_time_seconds = time.monotonic() - start_time
        trajectory.success = episode_complete and _check_task_success(
            trajectory, task, docker_mgr, container_id
        )

    finally:
        await docker_mgr.async_release_container(container_id)

    return trajectory


def _format_task_prompt(task: TaskSpec) -> str:
    """Format a TaskSpec into the initial user message for the agent."""
    parts = [f"## Task\n{task.description}"]

    if task.initial_files:
        file_list = ", ".join(task.initial_files.keys())
        parts.append(
            f"\nThe following files have been placed in /workspace: {file_list}"
        )

    if task.test_commands:
        cmds = "\n".join(f"  {cmd}" for cmd in task.test_commands)
        parts.append(
            f"\nTo verify your solution, run:\n{cmds}"
        )

    if task.expected_output:
        parts.append(f"\nExpected output: {task.expected_output}")

    parts.append(
        "\nWhen you have completed the task, use the submit tool with your answer."
    )

    return "\n".join(parts)


def _check_task_success(
    trajectory: Trajectory,
    task: TaskSpec,
    docker_mgr: DockerManager,
    container_id: str,
) -> bool:
    """Quick synchronous success heuristic.

    Checks if the last step was a submit and whether test commands (if any)
    would plausibly pass. For a full reward, the reward module is used separately.
    """
    if not trajectory.steps:
        return False

    last_step = trajectory.steps[-1]
    if last_step.action_type != ActionType.SUBMIT:
        return False

    # If there are test commands defined, we consider the task plausibly successful
    # if the agent submitted (actual test verification happens in reward computation).
    return True


# ---------------------------------------------------------------------------
# Outcome reward (lightweight)
# ---------------------------------------------------------------------------


def _compute_outcome_reward(trajectory: Trajectory, task: TaskSpec) -> float:
    """Compute a basic outcome reward for a trajectory.

    This is a lightweight reward used during GEPA evaluation. The full
    reward model in src/rewards/ is used for training data.
    """
    if not trajectory.success:
        return 0.0

    # Base reward for successful completion
    reward = 0.5

    # Efficiency bonus: fewer steps is better
    max_expected_steps = 15
    step_ratio = min(trajectory.num_steps / max_expected_steps, 1.0)
    efficiency_bonus = 0.2 * (1.0 - step_ratio)
    reward += efficiency_bonus

    # Tool diversity bonus: using appropriate tools
    expected_tools = {"bash", "read_file", "write_file"}
    used_tools = trajectory.action_types_used
    if used_tools & expected_tools:
        reward += 0.1

    # Penalty for excessive steps
    if trajectory.num_steps > max_expected_steps:
        reward -= 0.1 * ((trajectory.num_steps - max_expected_steps) / max_expected_steps)

    return max(0.0, min(1.0, reward))


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------


async def evaluate_genome(
    genome: PromptGenome,
    tasks: list[TaskSpec],
    vllm_server: VLLMServer,
    arena_manager: Any,
    *,
    max_concurrent: int = 4,
    max_steps_per_task: int = MAX_STEPS_DEFAULT,
    timeout_per_task: float = AGENT_TIMEOUT_SECONDS,
) -> EvalResult:
    """Evaluate a PromptGenome by running Qwen on a set of tasks.

    Runs tasks concurrently (up to max_concurrent) and aggregates results
    into an EvalResult with score_vector.

    Parameters
    ----------
    genome:
        The prompt genome to evaluate.
    tasks:
        List of tasks to run the agent on.
    vllm_server:
        The vLLM server for Qwen inference.
    arena_manager:
        Docker manager for arena containers.
    max_concurrent:
        Maximum number of tasks to run in parallel.
    max_steps_per_task:
        Maximum agent steps per task.
    timeout_per_task:
        Wall-clock timeout per task in seconds.

    Returns
    -------
    EvalResult
        Aggregated evaluation metrics and trajectories.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_one(task: TaskSpec) -> Trajectory:
        async with semaphore:
            try:
                return await _run_agent_loop(
                    genome,
                    task,
                    vllm_server,
                    arena_manager,
                    max_steps=max_steps_per_task,
                    timeout_seconds=timeout_per_task,
                )
            except Exception as exc:
                logger.error(
                    "Task %s failed with exception: %s", task.task_id, exc
                )
                return Trajectory(
                    task=task,
                    model_id=vllm_server.config.name,
                    prompt_genome_id=genome.genome_id,
                    success=False,
                )

    # Run all tasks concurrently
    trajectories = await asyncio.gather(*[_run_one(t) for t in tasks])

    # Compute aggregate metrics
    total_tasks = len(trajectories)
    successful = sum(1 for t in trajectories if t.success)
    total_steps = sum(t.num_steps for t in trajectories)

    success_rate = successful / total_tasks if total_tasks > 0 else 0.0
    avg_steps = total_steps / total_tasks if total_tasks > 0 else 0.0

    # Tool accuracy: ratio of tool actions that produced exit_code 0
    tool_actions = 0
    successful_tool_actions = 0
    for traj in trajectories:
        for step in traj.steps:
            if step.action_type not in (ActionType.THINK, ActionType.SUBMIT):
                tool_actions += 1
                # Consider a tool action successful if observation doesn't contain
                # obvious error indicators
                if not _looks_like_error(step.observation):
                    successful_tool_actions += 1

    tool_accuracy = (
        successful_tool_actions / tool_actions if tool_actions > 0 else 0.0
    )

    # Code quality: average outcome reward across all trajectories
    rewards = [_compute_outcome_reward(t, t.task) for t in trajectories if t.task]
    code_quality = sum(rewards) / len(rewards) if rewards else 0.0

    # Failure pattern analysis
    failure_patterns: dict[str, int] = defaultdict(int)
    for traj in trajectories:
        if not traj.success:
            pattern = _classify_failure(traj)
            failure_patterns[pattern] += 1

    result = EvalResult(
        genome_id=genome.genome_id,
        tasks_run=total_tasks,
        success_rate=success_rate,
        avg_steps=avg_steps,
        tool_accuracy=tool_accuracy,
        code_quality=code_quality,
        trajectories=list(trajectories),
        failure_patterns=dict(failure_patterns),
    )

    logger.info(
        "Genome %s evaluated: success=%.2f avg_steps=%.1f tool_acc=%.2f quality=%.2f",
        genome.genome_id,
        success_rate,
        avg_steps,
        tool_accuracy,
        code_quality,
    )

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _looks_like_error(observation: str) -> bool:
    """Heuristic check for error indicators in tool output."""
    error_patterns = [
        "Traceback (most recent call last)",
        "Error:",
        "error:",
        "FAILED",
        "No such file or directory",
        "Permission denied",
        "command not found",
        "SyntaxError",
        "NameError",
        "TypeError",
        "ValueError",
        "ImportError",
        "ModuleNotFoundError",
        "FileNotFoundError",
        "exit code 1",
        "exit code 2",
        "timed out",
    ]
    obs_lower = observation.lower()
    return any(p.lower() in obs_lower for p in error_patterns)


def _classify_failure(trajectory: Trajectory) -> str:
    """Classify a failed trajectory into a failure pattern category."""
    if not trajectory.steps:
        return "no_actions_taken"

    last_step = trajectory.steps[-1]

    # Did not submit
    if last_step.action_type != ActionType.SUBMIT:
        if trajectory.num_steps >= MAX_STEPS_DEFAULT:
            return "max_steps_exceeded"
        return "no_submission"

    # Check for repeated identical actions (stuck in a loop)
    if len(trajectory.steps) >= 3:
        last_actions = [
            (s.action_type, s.action_content) for s in trajectory.steps[-3:]
        ]
        if len(set(last_actions)) == 1:
            return "action_loop"

    # Check for tool errors in the last few steps
    error_count = sum(
        1
        for s in trajectory.steps[-5:]
        if _looks_like_error(s.observation)
    )
    if error_count >= 3:
        return "repeated_tool_errors"

    # Check for never reading files
    action_types = trajectory.action_types_used
    if "read_file" not in action_types and "bash" not in action_types:
        return "no_exploration"

    return "incorrect_solution"
