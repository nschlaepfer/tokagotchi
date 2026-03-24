"""Opus-powered task generation.

Uses the OpusClient to generate new evaluation tasks, adversarial
trap tasks targeting known weaknesses, and difficulty gradients
for systematic capability probing.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict
from typing import Any

from src.config import OpusConfig
from src.models import TaskSpec, TaskType
from src.orchestrator.opus_client import OpusClient

logger = logging.getLogger(__name__)

# Map human-friendly dimension names to TaskType values
_DIMENSION_TO_TYPE: dict[str, str] = {
    "file_manipulation": "code_debugging",
    "code_debugging": "code_debugging",
    "api_usage": "api_orchestration",
    "error_recovery": "code_debugging",
    "multi_step_planning": "open_ended_optimization",
    "info_gathering": "info_gathering",
}


class TaskGenerator:
    """Generates evaluation tasks via Opus.

    Parameters
    ----------
    opus_client:
        An OpusClient instance for calling Opus.
    """

    def __init__(self, opus_client: OpusClient) -> None:
        self._opus = opus_client

    async def generate_tasks(
        self,
        n: int,
        capability_profile: dict[str, Any],
        focus_type: str | None = None,
    ) -> list[TaskSpec]:
        """Generate N new tasks guided by the current capability profile.

        Parameters
        ----------
        n:
            Number of tasks to generate.
        capability_profile:
            Dict summarizing success rates by type, by difficulty, and
            overall. Used to identify areas needing improvement.
        focus_type:
            Optional TaskType value to focus generation on. If None,
            Opus decides the mix based on the profile.

        Returns
        -------
        list[TaskSpec]
            Generated task specifications.
        """
        # Identify weak areas from the profile
        weak_areas = self._identify_weak_areas(capability_profile)

        schema = {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "task_type": {"type": "string"},
                            "initial_files": {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                            },
                            "test_commands": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "expected_output": {"type": ["string", "null"]},
                            "difficulty": {"type": "number"},
                        },
                        "required": ["description", "task_type", "difficulty"],
                    },
                },
            },
            "required": ["tasks"],
        }

        focus_instruction = ""
        if focus_type:
            focus_instruction = f"\nFocus all tasks on type: {focus_type}\n"

        prompt = (
            f"Generate exactly {n} new evaluation tasks for an AI coding agent. "
            f"Each task should be self-contained with all necessary files.\n\n"
            f"## Current Capability Profile\n"
            f"```json\n{json.dumps(capability_profile, default=str, indent=2)}\n```\n\n"
            f"## Areas Needing Improvement\n"
            f"{json.dumps(weak_areas, default=str, indent=2)}\n\n"
            f"{focus_instruction}"
            f"## Task Types Available\n"
            f"- info_gathering: finding information from files/APIs\n"
            f"- code_debugging: fixing bugs in provided code\n"
            f"- api_orchestration: using APIs to accomplish goals\n"
            f"- open_ended_optimization: multi-step planning and execution\n\n"
            f"## Requirements\n"
            f"- Each task must have a clear success criterion\n"
            f"- Include initial_files with actual file contents\n"
            f"- Include test_commands that verify the solution\n"
            f"- Difficulty should be 0.0-1.0 (target the 0.3-0.7 range for most tasks)\n"
            f"- Tasks should be solvable in under 20 tool calls\n"
        )

        # Scale budget with number of tasks
        budget = min(0.30 + 0.10 * n, 2.0)

        resp = await self._opus.query(
            prompt,
            json_schema=schema,
            max_budget_usd=budget,
        )

        if resp.is_error:
            logger.error("Task generation failed: %s", resp.error_message)
            return []

        tasks_data = resp.raw_json.get("tasks", [])
        return [self._parse_task(t) for t in tasks_data]

    async def generate_trap_task(self, weakness: str) -> TaskSpec:
        """Generate an adversarial task that exploits a known weakness.

        Parameters
        ----------
        weakness:
            Description of the weakness to target, e.g.
            "agent fails to handle missing files gracefully".

        Returns
        -------
        TaskSpec
            A task specifically designed to test the weakness.
        """
        schema = {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "task_type": {"type": "string"},
                "initial_files": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "test_commands": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "expected_output": {"type": ["string", "null"]},
                "difficulty": {"type": "number"},
                "trap_explanation": {"type": "string"},
            },
            "required": ["description", "task_type", "difficulty", "trap_explanation"],
        }

        prompt = (
            f"Generate an adversarial 'trap' task for an AI coding agent that "
            f"specifically tests this known weakness:\n\n"
            f"**Weakness:** {weakness}\n\n"
            f"The task should look straightforward but contain a subtle element "
            f"that will trigger the weakness. Include:\n"
            f"- A clear task description that doesn't reveal the trap\n"
            f"- Initial files that set up the trap scenario\n"
            f"- Test commands that verify correct handling\n"
            f"- An explanation of why this tests the weakness\n"
        )

        resp = await self._opus.query(
            prompt,
            json_schema=schema,
            max_budget_usd=0.30,
        )

        if resp.is_error:
            logger.error("Trap task generation failed: %s", resp.error_message)
            return TaskSpec(description=f"(trap generation failed for: {weakness})")

        data = resp.raw_json
        spec = self._parse_task(data)
        spec.metadata["trap"] = True
        spec.metadata["trap_explanation"] = data.get("trap_explanation", "")
        spec.metadata["target_weakness"] = weakness
        return spec

    async def generate_difficulty_gradient(
        self,
        task_type: str,
        levels: int = 5,
    ) -> list[TaskSpec]:
        """Generate a gradient of tasks from easy to hard for a task type.

        Parameters
        ----------
        task_type:
            The TaskType value to generate tasks for.
        levels:
            Number of difficulty levels to produce.

        Returns
        -------
        list[TaskSpec]
            Tasks ordered from easiest to hardest.
        """
        schema = {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "task_type": {"type": "string"},
                            "initial_files": {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                            },
                            "test_commands": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "expected_output": {"type": ["string", "null"]},
                            "difficulty": {"type": "number"},
                            "difficulty_rationale": {"type": "string"},
                        },
                        "required": ["description", "difficulty"],
                    },
                },
            },
            "required": ["tasks"],
        }

        # Build difficulty targets: evenly spaced from 0.1 to 0.9
        targets = [round(0.1 + (0.8 * i / max(levels - 1, 1)), 2) for i in range(levels)]

        prompt = (
            f"Generate exactly {levels} tasks of type '{task_type}' forming a "
            f"difficulty gradient from easy to hard.\n\n"
            f"Target difficulty levels: {targets}\n\n"
            f"Each task should be self-contained. Make the difficulty progression "
            f"clear and meaningful:\n"
            f"- Easy tasks: single-step, obvious solution\n"
            f"- Medium tasks: 3-5 steps, some reasoning needed\n"
            f"- Hard tasks: 8+ steps, subtle issues, multiple files\n\n"
            f"Include initial_files, test_commands, and expected_output for each.\n"
        )

        budget = min(0.30 + 0.08 * levels, 1.50)

        resp = await self._opus.query(
            prompt,
            json_schema=schema,
            max_budget_usd=budget,
        )

        if resp.is_error:
            logger.error("Difficulty gradient generation failed: %s", resp.error_message)
            return []

        tasks_data = resp.raw_json.get("tasks", [])
        specs = [self._parse_task(t) for t in tasks_data]

        # Sort by difficulty to ensure proper ordering
        specs.sort(key=lambda s: s.difficulty)

        # Tag with gradient metadata
        for i, spec in enumerate(specs):
            spec.metadata["gradient_level"] = i
            spec.metadata["gradient_total"] = len(specs)

        return specs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _identify_weak_areas(profile: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract areas below 50% success rate from the profile."""
        weak: list[dict[str, Any]] = []

        by_type = profile.get("by_type", {})
        for task_type, rate in by_type.items():
            if rate < 0.5:
                weak.append({
                    "dimension": "task_type",
                    "value": task_type,
                    "success_rate": rate,
                    "severity": "critical" if rate < 0.2 else "moderate",
                })

        by_diff = profile.get("by_difficulty", {})
        for bucket, rate in by_diff.items():
            if rate < 0.5:
                weak.append({
                    "dimension": "difficulty",
                    "value": bucket,
                    "success_rate": rate,
                    "severity": "critical" if rate < 0.2 else "moderate",
                })

        return weak

    @staticmethod
    def _parse_task(data: dict[str, Any]) -> TaskSpec:
        """Parse a task dict from Opus output into a TaskSpec."""
        raw_type = data.get("task_type", "code_debugging")
        try:
            task_type = TaskType(raw_type)
        except ValueError:
            # Try mapping from dimension name
            mapped = _DIMENSION_TO_TYPE.get(raw_type, "code_debugging")
            task_type = TaskType(mapped)

        return TaskSpec(
            task_id=uuid.uuid4().hex[:12],
            task_type=task_type,
            description=data.get("description", ""),
            initial_files=data.get("initial_files", {}),
            test_commands=data.get("test_commands", []),
            expected_output=data.get("expected_output"),
            difficulty=data.get("difficulty", 0.5),
            metadata=data.get("metadata", {}),
        )
