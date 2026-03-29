"""Logprob-free SDPO: Self-Distillation via Behavioral Divergence.

After a failed agent episode, replays the trajectory through the model
with error feedback injected.  Compares original vs feedback-conditioned
actions to produce contrastive training pairs.  No logprobs needed —
divergence is measured at the action level.

Reference: "Reinforcement Learning via Self-Distillation" (SDPO, 2026)
Adapted for Ollama (no logprob support) by using behavioral divergence.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.loop2_distill.trace_collector import (
    TraceCollector,
    FAILURE_WRONG_TOOL,
    FAILURE_REASONING_ERROR,
    FAILURE_LOOP,
    FAILURE_TIMEOUT,
    FAILURE_INCORRECT_OUTPUT,
)
from src.models import ActionType, PromptGenome, StepRecord, Trajectory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(
    r"\[?(?P<action_type>[a-z_]+)\]?\s*:\s*(?P<content>.*)",
    re.DOTALL | re.IGNORECASE,
)
_BRACKET_RE = re.compile(
    r"\[(?P<action_type>[a-z_]+)\s+(?P<content>[^\]]+)\]",
    re.DOTALL | re.IGNORECASE,
)
_THINK_STRIP = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_ORPHAN = re.compile(r"^</think>\s*", re.DOTALL)


@dataclass
class ContrastivePair:
    """A single step where the model changed its action after seeing feedback."""

    step_idx: int
    context_messages: list[dict[str, str]]  # conversation up to this step (clean)
    negative: str  # original (failed) action text
    positive: str  # re-evaluated (feedback-conditioned) action text
    weight: float  # divergence score 0.0–1.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class SDPOReevaluator:
    """Generates contrastive training pairs from failed trajectories.

    For each failed episode, replays the trajectory step-by-step with
    error feedback injected.  Steps where the model changes its action
    become contrastive training pairs (train toward the re-evaluated
    action, away from the original).

    Parameters
    ----------
    divergence_threshold:
        Minimum divergence score (0–1) for a step to become a training pair.
    max_reeval_steps:
        Maximum number of steps to re-evaluate per trajectory (saves compute).
    """

    def __init__(
        self,
        divergence_threshold: float = 0.3,
        max_reeval_steps: int = 8,
    ) -> None:
        self.divergence_threshold = divergence_threshold
        self.max_reeval_steps = max_reeval_steps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def reevaluate(
        self,
        trajectory: Trajectory,
        vllm_server: Any,
        genome: PromptGenome,
    ) -> list[ContrastivePair]:
        """Re-evaluate a failed trajectory with error feedback.

        Returns contrastive pairs for steps where the model changes
        its action after seeing what went wrong.
        """
        if trajectory.success:
            return []

        error_feedback = self._build_error_feedback(trajectory)
        system_msg = genome.to_system_message()
        task_desc = trajectory.task.description if trajectory.task else ""

        # Identify which steps to re-evaluate (near failure point)
        steps_to_check = self._select_steps(trajectory)
        pairs: list[ContrastivePair] = []

        for step in steps_to_check:
            # Skip think-only steps
            if step.action_type == ActionType.THINK:
                continue

            # Build clean context up to this step
            context = self._build_context(
                system_msg, task_desc, trajectory, step.step_idx
            )

            # Build feedback-injected context
            feedback_context = list(context) + [
                {
                    "role": "user",
                    "content": (
                        f"IMPORTANT: Your previous attempt at this task FAILED.\n"
                        f"Error analysis: {error_feedback}\n\n"
                        f"Given this information, what is the best action to take "
                        f"at this point? Re-evaluate carefully and respond with "
                        f"a single [action_type]: content."
                    ),
                }
            ]

            # Get re-evaluated action from model
            try:
                completion = await vllm_server.chat_completion(
                    messages=feedback_context,
                    temperature=0.1,
                    max_tokens=1024,
                )
                reeval_response = completion.choices[0].message.content or ""
            except Exception as exc:
                logger.debug("Re-evaluation call failed: %s", exc)
                continue

            # Parse re-evaluated action
            reeval_type, reeval_content = self._parse_action(reeval_response)

            # Compute divergence
            div = self._compute_divergence(
                step.action_type.value,
                step.action_content,
                reeval_type,
                reeval_content,
            )

            if div >= self.divergence_threshold:
                pairs.append(
                    ContrastivePair(
                        step_idx=step.step_idx,
                        context_messages=context,
                        negative=f"[{step.action_type.value}]: {step.action_content}",
                        positive=reeval_response,
                        weight=div,
                        metadata={
                            "trajectory_id": trajectory.trajectory_id,
                            "task_id": trajectory.task.task_id if trajectory.task else "",
                            "original_action": step.action_type.value,
                            "reeval_action": reeval_type,
                            "divergence": div,
                        },
                    )
                )

        if pairs:
            logger.info(
                "SDPO: %d contrastive pairs from %d re-evaluated steps "
                "(trajectory %s)",
                len(pairs),
                len(steps_to_check),
                trajectory.trajectory_id[:12],
            )

        return pairs

    def generate_training_examples(
        self,
        pairs: list[ContrastivePair],
        trajectory: Trajectory,
    ) -> list[dict[str, Any]]:
        """Convert contrastive pairs into chat-format training examples.

        Each example trains the model to produce the re-evaluated (positive)
        action given the clean context (without error feedback injection).
        """
        examples: list[dict[str, Any]] = []

        for pair in pairs:
            # Training example: clean context → positive action
            messages = list(pair.context_messages) + [
                {"role": "assistant", "content": pair.positive}
            ]

            task_type = "unknown"
            if trajectory.task:
                task_type = trajectory.task.task_type.value

            example = {
                "example": {"messages": messages},
                "metadata": {
                    "task_type": task_type,
                    "failure_mode": "sdpo_divergent",
                    "source": "sdpo",
                    "sdpo_weight": pair.weight,
                    "step_idx": pair.step_idx,
                    "trajectory_id": pair.metadata.get("trajectory_id", ""),
                    "difficulty": trajectory.task.difficulty if trajectory.task else 0.5,
                },
            }
            examples.append(example)

        return examples

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_steps(self, trajectory: Trajectory) -> list[StepRecord]:
        """Select which steps to re-evaluate, focusing near the failure point."""
        failure_points = TraceCollector.identify_failure_points(trajectory)

        if failure_points:
            # Re-evaluate steps around the first failure point
            center = failure_points[0]
            start = max(0, center - 2)
            end = min(len(trajectory.steps), center + 3)
            candidates = trajectory.steps[start:end]
        else:
            # No clear failure point — check the last N steps
            candidates = trajectory.steps[-self.max_reeval_steps :]

        return candidates[: self.max_reeval_steps]

    def _build_context(
        self,
        system_msg: str,
        task_desc: str,
        trajectory: Trajectory,
        up_to_step: int,
    ) -> list[dict[str, str]]:
        """Build the conversation context up to (but not including) a step."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": task_desc},
        ]

        for step in trajectory.steps[:up_to_step]:
            # Assistant turn: the action
            action_str = f"[{step.action_type.value}]: {step.action_content}"
            if step.reasoning:
                action_str = f"[think]: {step.reasoning}\n{action_str}"
            messages.append({"role": "assistant", "content": action_str})

            # User turn: the observation
            if step.observation:
                messages.append({"role": "user", "content": step.observation})

        return messages

    def _build_error_feedback(self, trajectory: Trajectory) -> str:
        """Extract a concise error summary from the failed trajectory."""
        parts: list[str] = []

        # Task description
        if trajectory.task:
            parts.append(f"Task: {trajectory.task.description[:200]}")

        # Last observation (often contains the error)
        if trajectory.steps:
            last_step = trajectory.steps[-1]
            if last_step.observation:
                parts.append(f"Last observation: {last_step.observation[:300]}")

            # Look for stderr in metadata
            stderr = last_step.metadata.get("stderr", "")
            if stderr:
                parts.append(f"Error output: {stderr[:300]}")

        # Failure classification
        failure_type = TraceCollector.classify_failure(trajectory)
        if failure_type:
            parts.append(f"Failure type: {failure_type}")

        # Outcome
        parts.append(f"Result: Task was NOT completed successfully.")

        return "\n".join(parts)

    @staticmethod
    def _parse_action(response: str) -> tuple[str, str]:
        """Parse an action from a model response. Returns (type, content)."""
        cleaned = _THINK_STRIP.sub("", response)
        cleaned = _THINK_ORPHAN.sub("", cleaned).strip()
        if not cleaned:
            cleaned = response.strip()

        # Try each line
        for line in cleaned.split("\n"):
            line = line.strip()
            if not line:
                continue
            m = _ACTION_RE.match(line)
            if m is None:
                m = _BRACKET_RE.match(line)
            if m:
                return m.group("action_type").lower(), m.group("content").strip()

        return "think", cleaned

    @staticmethod
    def _compute_divergence(
        orig_type: str,
        orig_content: str,
        reeval_type: str,
        reeval_content: str,
    ) -> float:
        """Compute behavioral divergence between original and re-evaluated actions.

        Returns 0.0 (identical) to 1.0 (completely different).
        """
        if orig_type != reeval_type:
            return 1.0

        # Same action type — compare content
        similarity = difflib.SequenceMatcher(
            None, orig_content[:500], reeval_content[:500]
        ).ratio()

        return 0.5 + 0.5 * (1.0 - similarity)
