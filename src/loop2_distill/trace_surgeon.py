"""Opus-powered trace surgery on failed Qwen trajectories.

Sends full trajectories to Opus for diagnosis, receives corrected steps
from the failure point forward, and converts them into chat-format
training examples for SFT.
"""

from __future__ import annotations

import asyncio
import json
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
from src.orchestrator.opus_client import OpusClient
from src.config import Loop2Config

logger = logging.getLogger(__name__)

# Default rate-limit parameters for batch processing
_DEFAULT_MAX_CONCURRENT = 3
_DEFAULT_DELAY_SECONDS = 1.0


class TraceSurgeon:
    """Performs Opus-guided trace surgery on failed trajectories.

    Sends trajectories to Opus for diagnosis, extracts corrected steps,
    and converts them into SFT-ready training examples.

    Parameters
    ----------
    max_concurrent:
        Maximum number of concurrent Opus surgery calls.
    inter_call_delay:
        Minimum delay in seconds between Opus calls for rate limiting.
    """

    def __init__(
        self,
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
        inter_call_delay: float = _DEFAULT_DELAY_SECONDS,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.inter_call_delay = inter_call_delay
        self._semaphore: asyncio.Semaphore | None = None

    # ------------------------------------------------------------------
    # Core surgery
    # ------------------------------------------------------------------

    async def perform_surgery(
        self,
        trajectory: Trajectory,
        task_spec: TaskSpec,
        opus_client: OpusClient,
    ) -> TraceAnalysis:
        """Send a trajectory to Opus for diagnosis and correction.

        Opus identifies the exact failure step, diagnoses the root cause,
        and produces corrected steps from that point forward.

        Parameters
        ----------
        trajectory:
            The failed Qwen trajectory to analyse.
        task_spec:
            The original task specification for context.
        opus_client:
            An initialised OpusClient for making Opus queries.

        Returns
        -------
        TraceAnalysis
            Contains failure_step, diagnosis, and corrected_steps.
        """
        schema = {
            "type": "object",
            "properties": {
                "trajectory_id": {"type": "string"},
                "failure_step": {"type": ["integer", "null"]},
                "diagnosis": {"type": "string"},
                "corrected_steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_idx": {"type": "integer"},
                            "action_type": {"type": "string"},
                            "action_content": {"type": "string"},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["step_idx", "action_type", "action_content"],
                    },
                },
                "overall_assessment": {"type": "string"},
            },
            "required": [
                "trajectory_id",
                "failure_step",
                "diagnosis",
                "corrected_steps",
                "overall_assessment",
            ],
        }

        prompt = (
            "You are performing trace surgery on a failed agent trajectory. "
            "Carefully examine every step, identify the EXACT step where the "
            "agent first went wrong, diagnose the root cause, and provide "
            "corrected steps from that point forward.\n\n"
            "The corrected steps should demonstrate the ideal approach: correct "
            "tool usage, sound reasoning, and progress toward the goal.\n\n"
            f"## Task\n{task_spec.description}\n\n"
            f"## Trajectory\n```json\n"
            f"{json.dumps(asdict(trajectory), default=str)}\n```\n\n"
            "Return the analysis as structured JSON."
        )

        resp = await opus_client.query(
            prompt,
            json_schema=schema,
            max_budget_usd=0.50,
        )

        if resp.is_error:
            logger.error(
                "Trace surgery failed for trajectory %s: %s",
                trajectory.trajectory_id,
                resp.error_message,
            )
            return TraceAnalysis(trajectory_id=trajectory.trajectory_id)

        data = resp.raw_json
        return TraceAnalysis(
            trajectory_id=data.get("trajectory_id", trajectory.trajectory_id),
            failure_step=data.get("failure_step"),
            diagnosis=data.get("diagnosis", ""),
            corrected_steps=data.get("corrected_steps", []),
            overall_assessment=data.get("overall_assessment", ""),
        )

    # ------------------------------------------------------------------
    # Training example generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_training_example(
        trajectory: Trajectory,
        analysis: TraceAnalysis,
    ) -> dict[str, Any]:
        """Convert a trajectory and its surgery analysis into a chat-format training example.

        The example preserves all correct steps up to the failure point,
        then splices in Opus's corrected steps from the failure onward.

        Format::

            {
                "messages": [
                    {"role": "system", "content": "..."},
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."},
                    ...
                ],
                "metadata": {
                    "trajectory_id": "...",
                    "failure_step": N,
                    "diagnosis": "...",
                }
            }

        The assistant messages use the corrected trace from the failure
        point onward, teaching Qwen the correct behaviour.
        """
        messages: list[dict[str, str]] = []

        # System message from the trajectory's genome (if available)
        system_content = (
            "You are a skilled coding agent. Solve the task step by step "
            "using the available tools. Think carefully before acting."
        )
        messages.append({"role": "system", "content": system_content})

        # Initial task description as user message
        if trajectory.task:
            available_tools = ", ".join(at.value for at in ActionType)
            initial_obs = (
                f"## Task\n{trajectory.task.description}\n\n"
                f"## Available Tools\n{available_tools}\n\n"
                f"Use [action_type]: content to invoke a tool.\n"
                f"Use [submit]: your answer when done."
            )
            messages.append({"role": "user", "content": initial_obs})

        failure_step = analysis.failure_step
        if failure_step is None:
            failure_step = 0

        # Add original correct steps up to the failure point
        for step in trajectory.steps:
            if step.step_idx >= failure_step:
                break

            # Assistant action
            action_str = f"[{step.action_type.value}]: {step.action_content}"
            if step.reasoning:
                action_str = f"[think]: {step.reasoning}\n{action_str}"
            messages.append({"role": "assistant", "content": action_str})

            # Environment observation
            if step.observation:
                messages.append({"role": "user", "content": step.observation})

        # Splice in corrected steps from Opus
        for corrected in analysis.corrected_steps:
            action_type = corrected.get("action_type", "think")
            action_content = corrected.get("action_content", "")
            reasoning = corrected.get("reasoning", "")

            action_str = f"[{action_type}]: {action_content}"
            if reasoning:
                action_str = f"[think]: {reasoning}\n{action_str}"
            messages.append({"role": "assistant", "content": action_str})

            # For non-terminal corrected steps, add a placeholder observation
            # The actual observations aren't available for corrected steps
            if action_type != "submit":
                messages.append(
                    {"role": "user", "content": "(observation from environment)"}
                )

        return {
            "messages": messages,
            "metadata": {
                "trajectory_id": trajectory.trajectory_id,
                "task_id": trajectory.task.task_id if trajectory.task else "",
                "failure_step": analysis.failure_step,
                "diagnosis": analysis.diagnosis,
                "num_corrected_steps": len(analysis.corrected_steps),
            },
        }

    # ------------------------------------------------------------------
    # Batch processing with rate limiting
    # ------------------------------------------------------------------

    async def batch_surgery(
        self,
        trajectories: list[Trajectory],
        task_specs: dict[str, TaskSpec],
        opus_client: OpusClient,
    ) -> list[tuple[Trajectory, TraceAnalysis]]:
        """Perform trace surgery on multiple trajectories with rate limiting.

        Parameters
        ----------
        trajectories:
            Failed trajectories to analyse.
        task_specs:
            Mapping of task_id to TaskSpec for context lookup.
        opus_client:
            An initialised OpusClient.

        Returns
        -------
        list[tuple[Trajectory, TraceAnalysis]]
            Pairs of (original trajectory, surgery analysis).
        """
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        results: list[tuple[Trajectory, TraceAnalysis]] = []

        logger.info(
            "Starting batch surgery on %d trajectories (max_concurrent=%d)",
            len(trajectories),
            self.max_concurrent,
        )

        async def _process_one(
            traj: Trajectory,
        ) -> tuple[Trajectory, TraceAnalysis]:
            assert self._semaphore is not None
            async with self._semaphore:
                task_id = traj.task.task_id if traj.task else ""
                task_spec = task_specs.get(
                    task_id,
                    TaskSpec(description="(unknown task)"),
                )
                analysis = await self.perform_surgery(traj, task_spec, opus_client)
                # Rate-limit delay
                await asyncio.sleep(self.inter_call_delay)
                return traj, analysis

        coros = [_process_one(traj) for traj in trajectories]
        gathered = await asyncio.gather(*coros, return_exceptions=True)

        for item in gathered:
            if isinstance(item, tuple):
                results.append(item)
            elif isinstance(item, Exception):
                logger.error("Surgery failed for a trajectory: %s", item)

        logger.info(
            "Batch surgery complete: %d/%d succeeded",
            len(results),
            len(trajectories),
        )
        return results

    async def batch_generate_examples(
        self,
        surgery_results: list[tuple[Trajectory, TraceAnalysis]],
    ) -> list[dict[str, Any]]:
        """Generate training examples from a batch of surgery results.

        Parameters
        ----------
        surgery_results:
            Pairs of (trajectory, analysis) from :meth:`batch_surgery`.

        Returns
        -------
        list[dict]
            Chat-format training examples ready for SFT.
        """
        examples: list[dict[str, Any]] = []

        for trajectory, analysis in surgery_results:
            if analysis.failure_step is None and not analysis.corrected_steps:
                logger.debug(
                    "Skipping trajectory %s: no failure identified",
                    trajectory.trajectory_id,
                )
                continue

            example = self.generate_training_example(trajectory, analysis)
            if len(example["messages"]) > 2:  # More than just system + task
                examples.append(example)

        logger.info("Generated %d training examples from surgery results", len(examples))
        return examples
