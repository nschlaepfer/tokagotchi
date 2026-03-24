"""Implements the 80/20 mentored vs teacher-annotated split for on-policy distillation.

In mentored mode, Qwen attempts tasks with Opus providing hints when stuck.
In teacher-demo mode, Opus solves tasks outright as a demonstration.
Both modes generate SFT training examples.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
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
from src.orchestrator.opus_client import OpusClient
from src.config import Loop2Config

logger = logging.getLogger(__name__)

# How many steps without progress before Opus intervenes
_STUCK_THRESHOLD = 3

# Maximum iterations per mentored episode
_MAX_MENTORED_STEPS = 50


class MentorSession:
    """Manages mentored and teacher-demonstrated training example generation.

    Implements the 80/20 split: most tasks are attempted by Qwen with
    Opus nudging when stuck, while a minority are solved entirely by
    Opus as full demonstrations.

    Parameters
    ----------
    config:
        Loop2 configuration with teacher_ratio and mentor_ratio.
    """

    def __init__(self, config: Loop2Config) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Mentored session
    # ------------------------------------------------------------------

    async def run_mentored_session(
        self,
        task: TaskSpec,
        vllm_server: Any,
        arena_manager: Any,
        opus_client: OpusClient,
        genome: PromptGenome,
    ) -> tuple[Trajectory, list[dict[str, Any]]]:
        """Run a mentored session: Qwen attempts the task with Opus hints when stuck.

        When Qwen makes no progress for ``_STUCK_THRESHOLD`` steps or
        repeats the same action, Opus provides a nudge (not the full
        answer). Training examples are generated from the hint points
        showing the correct next action.

        Parameters
        ----------
        task:
            The task for Qwen to attempt.
        vllm_server:
            A running VLLMServer for Qwen inference.
        arena_manager:
            DockerManager for container management.
        opus_client:
            OpusClient for generating hints.
        genome:
            The PromptGenome providing the system prompt.

        Returns
        -------
        tuple[Trajectory, list[dict]]
            The full trajectory and any training examples generated
            from hint intervention points.
        """
        from src.arena.game import AgentArenaGame

        game = AgentArenaGame(docker_mgr=arena_manager)
        training_examples: list[dict[str, Any]] = []

        async with game:
            initial_obs = await game.reset(task)

            messages: list[dict[str, str]] = [
                {"role": "system", "content": genome.to_system_message()},
                {"role": "user", "content": initial_obs},
            ]

            # Track recent actions for stuck detection
            recent_actions: list[str] = []
            recent_observations: list[str] = []
            hints_given = 0

            for iteration in range(_MAX_MENTORED_STEPS):
                # Check if Qwen is stuck
                is_stuck = self._detect_stuck(recent_actions, recent_observations)

                if is_stuck and hints_given < 5:
                    # Generate a hint from Opus
                    hint = await self._generate_hint(
                        task, messages, opus_client,
                    )
                    hints_given += 1

                    logger.info(
                        "Mentor hint #%d for task %s: %s",
                        hints_given,
                        task.task_id,
                        hint[:100],
                    )

                    # Inject the hint as a user message (system nudge)
                    hint_message = (
                        f"[Mentor hint]: {hint}\n"
                        "Continue working on the task."
                    )
                    messages.append({"role": "user", "content": hint_message})

                    # Generate a training example at this hint point
                    example = self._create_hint_example(task, messages, hint)
                    if example:
                        training_examples.append(example)

                    # Reset stuck tracking
                    recent_actions.clear()
                    recent_observations.clear()

                # Get Qwen's next action
                completion = await vllm_server.chat_completion(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2048,
                )

                assistant_msg = completion.choices[0].message.content or ""
                messages.append({"role": "assistant", "content": assistant_msg})
                recent_actions.append(assistant_msg)

                # Execute in the arena
                step_result = await game.step(assistant_msg)
                recent_observations.append(step_result.observation)

                if step_result.observation:
                    messages.append(
                        {"role": "user", "content": step_result.observation}
                    )

                if step_result.done:
                    break

            trajectory = game.get_trajectory()
            trajectory.model_id = vllm_server.config.name
            trajectory.prompt_genome_id = genome.genome_id

        logger.info(
            "Mentored session for task %s: %d steps, %d hints, success=%s",
            task.task_id,
            trajectory.num_steps,
            hints_given,
            trajectory.success,
        )

        return trajectory, training_examples

    # ------------------------------------------------------------------
    # Teacher demonstration
    # ------------------------------------------------------------------

    async def run_teacher_demo(
        self,
        task: TaskSpec,
        opus_client: OpusClient,
    ) -> list[dict[str, Any]]:
        """Have Opus solve a task completely, producing a full demonstration.

        Opus generates a step-by-step solution trajectory which is then
        converted into a chat-format training example.

        Parameters
        ----------
        task:
            The task for Opus to solve.
        opus_client:
            OpusClient for the teacher demonstration.

        Returns
        -------
        list[dict]
            Training examples from the teacher demonstration (typically one).
        """
        schema = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action_type": {"type": "string"},
                            "action_content": {"type": "string"},
                            "reasoning": {"type": "string"},
                            "expected_observation": {"type": "string"},
                        },
                        "required": ["action_type", "action_content", "reasoning"],
                    },
                },
                "final_answer": {"type": "string"},
            },
            "required": ["steps", "final_answer"],
        }

        available_tools = ", ".join(at.value for at in ActionType)
        prompt = (
            "You are an expert coding agent demonstrating the ideal approach "
            "to solving a task. Produce a complete, step-by-step solution.\n\n"
            "For each step, specify:\n"
            "- action_type: one of " + available_tools + "\n"
            "- action_content: the exact command or content\n"
            "- reasoning: why this step is necessary\n"
            "- expected_observation: what you expect to see\n\n"
            "End with a submit action containing the final answer.\n\n"
            f"## Task\n{task.description}\n\n"
            f"## Initial Files\n```json\n{json.dumps(task.initial_files)}\n```\n\n"
            f"## Test Commands\n{json.dumps(task.test_commands)}"
        )

        resp = await opus_client.query(
            prompt,
            json_schema=schema,
            max_budget_usd=0.50,
        )

        if resp.is_error:
            logger.error(
                "Teacher demo failed for task %s: %s",
                task.task_id,
                resp.error_message,
            )
            return []

        data = resp.raw_json
        demo_steps = data.get("steps", [])
        final_answer = data.get("final_answer", "")

        if not demo_steps:
            logger.warning("Teacher demo returned no steps for task %s", task.task_id)
            return []

        # Convert to a training example
        example = self._demo_to_training_example(task, demo_steps, final_answer)
        if example:
            logger.info(
                "Teacher demo for task %s: %d steps",
                task.task_id,
                len(demo_steps),
            )
            return [example]

        return []

    # ------------------------------------------------------------------
    # Batch generation with 80/20 split
    # ------------------------------------------------------------------

    async def generate_batch(
        self,
        tasks: list[TaskSpec],
        vllm_server: Any,
        arena_manager: Any,
        opus_client: OpusClient,
        genome: PromptGenome,
    ) -> list[dict[str, Any]]:
        """Generate training examples using the 80/20 mentored/teacher split.

        Tasks are randomly partitioned: ``mentor_ratio`` fraction go to
        mentored sessions, the rest to teacher demonstrations.

        Parameters
        ----------
        tasks:
            Tasks to generate examples from.
        vllm_server:
            VLLMServer for Qwen inference (mentored sessions).
        arena_manager:
            DockerManager for container management.
        opus_client:
            OpusClient for hints and demonstrations.
        genome:
            PromptGenome for Qwen's system prompt.

        Returns
        -------
        list[dict]
            All generated training examples from both modes.
        """
        # Shuffle and split
        shuffled = list(tasks)
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * self._config.mentor_ratio)
        mentored_tasks = shuffled[:split_idx]
        teacher_tasks = shuffled[split_idx:]

        logger.info(
            "Batch generation: %d mentored, %d teacher demo (total %d tasks)",
            len(mentored_tasks),
            len(teacher_tasks),
            len(tasks),
        )

        all_examples: list[dict[str, Any]] = []

        # Run mentored sessions concurrently
        mentored_coros = [
            self.run_mentored_session(
                task, vllm_server, arena_manager, opus_client, genome,
            )
            for task in mentored_tasks
        ]
        mentored_results = await asyncio.gather(
            *mentored_coros, return_exceptions=True,
        )

        for result in mentored_results:
            if isinstance(result, tuple):
                _trajectory, examples = result
                all_examples.extend(examples)
            elif isinstance(result, Exception):
                logger.error("Mentored session failed: %s", result)

        # Run teacher demos concurrently
        teacher_coros = [
            self.run_teacher_demo(task, opus_client) for task in teacher_tasks
        ]
        teacher_results = await asyncio.gather(
            *teacher_coros, return_exceptions=True,
        )

        for result in teacher_results:
            if isinstance(result, list):
                all_examples.extend(result)
            elif isinstance(result, Exception):
                logger.error("Teacher demo failed: %s", result)

        logger.info(
            "Batch generation complete: %d total examples "
            "(%d mentored tasks, %d teacher tasks)",
            len(all_examples),
            len(mentored_tasks),
            len(teacher_tasks),
        )

        return all_examples

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_stuck(
        recent_actions: list[str],
        recent_observations: list[str],
    ) -> bool:
        """Detect whether Qwen is stuck based on recent history.

        Stuck conditions:
        1. No progress for ``_STUCK_THRESHOLD`` steps (same observations).
        2. Repeated identical actions.
        """
        if len(recent_actions) < _STUCK_THRESHOLD:
            return False

        window = recent_actions[-_STUCK_THRESHOLD:]

        # Check for repeated actions
        if len(set(window)) == 1:
            return True

        # Check for no progress: observations are all the same or empty
        obs_window = recent_observations[-_STUCK_THRESHOLD:]
        if len(obs_window) >= _STUCK_THRESHOLD:
            non_empty = [o for o in obs_window if o.strip()]
            if not non_empty:
                return True
            if len(set(non_empty)) == 1:
                return True

        return False

    @staticmethod
    async def _generate_hint(
        task: TaskSpec,
        conversation: list[dict[str, str]],
        opus_client: OpusClient,
    ) -> str:
        """Ask Opus for a hint (nudge, not a full answer).

        The hint guides Qwen toward the right approach without giving
        away the solution.
        """
        # Summarise recent conversation to keep the prompt manageable
        recent_msgs = conversation[-10:]  # Last 10 messages
        conversation_summary = json.dumps(recent_msgs, default=str)

        prompt = (
            "An AI coding agent is stuck on a task. Provide a SHORT hint "
            "(1-2 sentences) to nudge it in the right direction. Do NOT "
            "give the full answer or exact commands. Just point toward "
            "what to look at or try differently.\n\n"
            f"## Task\n{task.description}\n\n"
            f"## Recent Conversation\n```json\n{conversation_summary}\n```\n\n"
            "Hint:"
        )

        resp = await opus_client.query(
            prompt,
            max_budget_usd=0.10,
            max_turns=1,
        )

        if resp.is_error:
            return "Consider re-reading the error message and trying a different approach."

        return resp.text.strip()

    @staticmethod
    def _create_hint_example(
        task: TaskSpec,
        conversation: list[dict[str, str]],
        hint: str,
    ) -> dict[str, Any] | None:
        """Create a training example from a hint intervention point.

        The example shows the conversation up to the stuck point,
        with the hint incorporated as context, and the expected
        next action as the assistant response.
        """
        if len(conversation) < 3:
            return None

        # Build a training example:
        # - System message
        # - Conversation history up to the hint
        # - The hint-informed context as the final user message
        messages: list[dict[str, str]] = []

        for msg in conversation:
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

        return {
            "messages": messages,
            "metadata": {
                "task_id": task.task_id,
                "source": "mentor_hint",
                "hint": hint,
            },
        }

    @staticmethod
    def _demo_to_training_example(
        task: TaskSpec,
        demo_steps: list[dict[str, Any]],
        final_answer: str,
    ) -> dict[str, Any] | None:
        """Convert a teacher demonstration into a chat-format training example.

        Builds an alternating user/assistant conversation from the
        demonstration steps.
        """
        messages: list[dict[str, str]] = []

        # System message
        messages.append({
            "role": "system",
            "content": (
                "You are a skilled coding agent. Solve the task step by step "
                "using the available tools. Think carefully before acting."
            ),
        })

        # Initial task observation
        available_tools = ", ".join(at.value for at in ActionType)
        initial_obs = (
            f"## Task\n{task.description}\n\n"
            f"## Available Tools\n{available_tools}\n\n"
            f"Use [action_type]: content to invoke a tool.\n"
            f"Use [submit]: your answer when done."
        )
        messages.append({"role": "user", "content": initial_obs})

        # Interleave assistant actions and expected observations
        for step in demo_steps:
            action_type = step.get("action_type", "think")
            action_content = step.get("action_content", "")
            reasoning = step.get("reasoning", "")
            expected_obs = step.get("expected_observation", "")

            # Assistant action
            action_str = f"[{action_type}]: {action_content}"
            if reasoning:
                action_str = f"[think]: {reasoning}\n{action_str}"
            messages.append({"role": "assistant", "content": action_str})

            # Environment observation (if not the final step)
            if expected_obs and action_type != "submit":
                messages.append({"role": "user", "content": expected_obs})

        # Final submit if not already present
        if demo_steps and demo_steps[-1].get("action_type") != "submit":
            messages.append({
                "role": "assistant",
                "content": f"[submit]: {final_answer}",
            })

        return {
            "messages": messages,
            "metadata": {
                "task_id": task.task_id,
                "source": "teacher_demo",
                "num_steps": len(demo_steps),
            },
        }
