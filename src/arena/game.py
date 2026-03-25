"""Main game environment for agent arena episodes.

Provides the AgentArenaGame class that manages a single agent episode:
container lifecycle, action parsing, tool dispatch, trajectory tracking,
and step/reward accounting.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

from src.arena.tools import bash_tool, python_tool, file_tool, submit_tool
from src.arena.tools.common import ToolResult
from src.arena.tools.sql_tool import execute as sql_execute
from src.arena.tools.api_tool import execute as api_execute
from src.models import ActionType, StepRecord, TaskSpec, Trajectory

logger = logging.getLogger(__name__)

# Type alias — both DockerManager and SubprocessManager implement the same
# async_create_container / async_exec_in_container / async_destroy_container API.
# We use Any here to avoid circular imports; runtime duck-typing is fine.
ArenaManagerLike = Any

# Regex to parse "[action_type]: content" or "action_type: content"
_ACTION_PATTERN = re.compile(
    r"^\[?(?P<action_type>[a-z_]+)\]?\s*:\s*(?P<content>.*)",
    re.DOTALL | re.IGNORECASE,
)

DEFAULT_MAX_TOOL_CALLS = 20


@dataclass
class StepResult:
    """Result of a single game step."""

    observation: str
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class AgentArenaGame:
    """Game environment for a single agent episode.

    Manages the full lifecycle: container/sandbox creation, action dispatch,
    trajectory recording, and cleanup.

    Parameters
    ----------
    arena_mgr:
        Any arena manager (DockerManager or SubprocessManager) that
        implements async_create_container, async_exec_in_container,
        and async_destroy_container.
    max_tool_calls:
        Maximum number of tool-executing steps before the episode is
        forcibly terminated.
    """

    def __init__(
        self,
        arena_mgr: ArenaManagerLike,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    ) -> None:
        self._docker_mgr = arena_mgr  # kept as _docker_mgr for minimal internal churn
        self._max_tool_calls = max_tool_calls

        self._container_id: str | None = None
        self._task: TaskSpec | None = None
        self._steps: list[StepRecord] = []
        self._step_count: int = 0
        self._tool_call_count: int = 0
        self._done: bool = False
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> AgentArenaGame:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    async def reset(self, task_spec: TaskSpec) -> str:
        """Start a new episode.

        Spins up a Docker container, seeds it with the task's initial
        files, and returns the initial observation (task description
        plus available tools).

        Parameters
        ----------
        task_spec:
            The task specification for this episode.

        Returns
        -------
        str
            The initial observation presented to the agent.
        """
        # Clean up any prior episode
        if self._container_id is not None:
            await self.close()

        self._task = task_spec
        self._steps = []
        self._step_count = 0
        self._tool_call_count = 0
        self._done = False
        self._start_time = time.monotonic()

        # Create and seed container
        self._container_id = await self._docker_mgr.async_create_container(task_spec)
        logger.info(
            "Episode started: task=%s container=%s",
            task_spec.task_id,
            self._container_id[:12],
        )

        # Build initial observation
        available_tools = ", ".join(at.value for at in ActionType)
        initial_obs = (
            f"## Task\n{task_spec.description}\n\n"
            f"## Available Tools\n{available_tools}\n\n"
            f"## Action Format\n"
            f"Use `[action_type]: content` to invoke a tool.\n"
            f"Use `[think]: your reasoning` to reason without executing.\n"
            f"Use `[submit]: your answer` when done.\n\n"
            f"## Constraints\n"
            f"- Maximum tool calls: {self._max_tool_calls}\n"
            f"- All files are in /workspace\n"
        )
        return initial_obs

    async def step(self, action: str) -> StepResult:
        """Execute one agent action and return the result.

        Parses the action string into (action_type, content), dispatches
        to the appropriate tool, and records the step in the trajectory.

        Parameters
        ----------
        action:
            Raw action string from the agent, e.g. ``[bash]: ls -la``.

        Returns
        -------
        StepResult
            Contains the observation, reward, done flag, and info dict.
        """
        if self._done:
            return StepResult(
                observation="Episode is already complete.",
                reward=0.0,
                done=True,
                info={"error": "episode_already_done"},
            )

        if self._container_id is None:
            return StepResult(
                observation="No active container. Call reset() first.",
                reward=0.0,
                done=True,
                info={"error": "no_container"},
            )

        # Parse action
        action_type, content = self._parse_action(action)

        # Record step
        self._step_count += 1

        # Handle think action (no execution)
        if action_type == ActionType.THINK:
            record = StepRecord(
                step_idx=self._step_count,
                action_type=action_type,
                action_content=content,
                observation="",
                reasoning=content,
            )
            self._steps.append(record)
            return StepResult(
                observation="",
                reward=0.0,
                done=False,
                info={"action_type": action_type.value},
            )

        # Check tool call limit
        self._tool_call_count += 1
        if self._tool_call_count > self._max_tool_calls:
            self._done = True
            record = StepRecord(
                step_idx=self._step_count,
                action_type=action_type,
                action_content=content,
                observation="Tool call limit reached. Episode terminated.",
                reward=-0.5,
            )
            self._steps.append(record)
            return StepResult(
                observation="Tool call limit reached. Episode terminated.",
                reward=-0.5,
                done=True,
                info={"reason": "max_tool_calls_exceeded"},
            )

        # Dispatch to tool
        tool_result = await self._dispatch(action_type, content)

        # Check for episode completion (submit)
        done = False
        reward = 0.0
        info: dict[str, Any] = {"action_type": action_type.value}

        if tool_result.metadata.get("episode_complete"):
            done = True
            info["submitted_answer"] = tool_result.stdout

        if not tool_result.success:
            reward = -0.1  # Small penalty for errors
            info["tool_error"] = True

        self._done = done

        record = StepRecord(
            step_idx=self._step_count,
            action_type=action_type,
            action_content=content,
            observation=tool_result.output,
            reward=reward,
            metadata=tool_result.metadata,
        )
        self._steps.append(record)

        return StepResult(
            observation=tool_result.output,
            reward=reward,
            done=done,
            info=info,
        )

    def get_trajectory(self) -> Trajectory:
        """Return the full trajectory so far.

        Returns
        -------
        Trajectory
            Contains the task spec, all recorded steps, and timing info.
        """
        wall_time = time.monotonic() - self._start_time if self._start_time else 0.0
        total_reward = sum(s.reward for s in self._steps)
        return Trajectory(
            task=self._task,
            steps=list(self._steps),
            success=self._done and any(
                s.metadata.get("episode_complete") for s in self._steps
            ),
            total_reward=total_reward,
            wall_time_seconds=wall_time,
        )

    async def close(self) -> None:
        """Destroy the container and clean up resources."""
        if self._container_id is not None:
            try:
                await self._docker_mgr.async_destroy_container(self._container_id)
            except Exception:
                logger.warning(
                    "Failed to destroy container %s",
                    self._container_id[:12] if self._container_id else "?",
                    exc_info=True,
                )
            self._container_id = None
        self._done = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_action(self, action: str) -> tuple[ActionType, str]:
        """Parse a raw action string into (ActionType, content).

        Supports formats:
        - ``[action_type]: content``
        - ``action_type: content``

        Falls back to THINK if parsing fails.
        """
        action = action.strip()
        match = _ACTION_PATTERN.match(action)
        if match is None:
            logger.warning("Could not parse action, treating as think: %s", action[:80])
            return ActionType.THINK, action

        raw_type = match.group("action_type").lower()
        content = match.group("content").strip()

        try:
            action_type = ActionType(raw_type)
        except ValueError:
            logger.warning("Unknown action type '%s', treating as think", raw_type)
            return ActionType.THINK, content

        return action_type, content

    async def _dispatch(self, action_type: ActionType, content: str) -> ToolResult:
        """Route an action to its corresponding tool handler."""
        assert self._container_id is not None

        try:
            if action_type == ActionType.BASH:
                return await bash_tool.execute(
                    self._docker_mgr, self._container_id, content
                )

            elif action_type == ActionType.PYTHON:
                return await python_tool.execute(
                    self._docker_mgr, self._container_id, content
                )

            elif action_type == ActionType.READ_FILE:
                return await file_tool.read_file(
                    self._docker_mgr, self._container_id, content.strip()
                )

            elif action_type == ActionType.WRITE_FILE:
                # Content format: first line is path, rest is file content
                lines = content.split("\n", 1)
                path = lines[0].strip()
                file_content = lines[1] if len(lines) > 1 else ""
                return await file_tool.write_file(
                    self._docker_mgr, self._container_id, path, file_content
                )

            elif action_type == ActionType.SQL:
                return await sql_execute(
                    self._docker_mgr, self._container_id, content.strip()
                )

            elif action_type == ActionType.API_CALL:
                # Content format: endpoint [params]
                parts = content.strip().split(None, 1)
                endpoint = parts[0] if parts else ""
                params = parts[1] if len(parts) > 1 else ""
                return await api_execute(
                    self._docker_mgr, self._container_id, endpoint, params
                )

            elif action_type == ActionType.SUBMIT:
                return await submit_tool.submit(self._container_id, content)

            else:
                return ToolResult(
                    stdout="",
                    stderr=f"Unknown action type: {action_type.value}",
                    exit_code=1,
                )

        except TimeoutError as exc:
            return ToolResult(
                stdout="",
                stderr=f"Tool execution timed out: {exc}",
                exit_code=124,
            )
        except Exception as exc:
            logger.error("Tool dispatch error: %s", exc, exc_info=True)
            return ToolResult(
                stdout="",
                stderr=f"Internal tool error: {exc}",
                exit_code=1,
            )
