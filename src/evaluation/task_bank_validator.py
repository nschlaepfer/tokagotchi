"""Task-bank validation before tasks are used for optimization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.evaluation.task_judge import TaskJudge, TaskJudgeResult
from src.models import ActionType, StepRecord, TaskSpec, TaskType, Trajectory


@dataclass
class TaskValidationResult:
    """Validation result for one task-bank entry."""

    task_id: str
    valid: bool
    issues: list[str] = field(default_factory=list)
    starter_result: dict[str, Any] | None = None
    reference_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskBankValidator:
    """Validate task specs structurally and, when possible, executably."""

    def __init__(self, *, judge: TaskJudge | None = None) -> None:
        self.judge = judge or TaskJudge()

    def validate_static(self, task: TaskSpec) -> TaskValidationResult:
        issues: list[str] = []
        if not task.description.strip():
            issues.append("missing_description")
        if not 0.0 <= float(task.difficulty) <= 1.0:
            issues.append("difficulty_out_of_range")

        has_executable_oracle = bool(task.test_commands)
        has_answer_oracle = task.expected_output is not None

        if task.task_type == TaskType.CODE_DEBUGGING and not has_executable_oracle:
            issues.append("code_task_missing_test_commands")
        if task.task_type in (TaskType.INFO_GATHERING, TaskType.API_ORCHESTRATION):
            if not has_executable_oracle and not has_answer_oracle:
                issues.append("answer_task_missing_oracle")
        if task.task_type == TaskType.OPEN_ENDED:
            if "benchmark_command" not in task.metadata:
                issues.append("open_ended_missing_benchmark_command")
            if "baseline_seconds" not in task.metadata:
                issues.append("open_ended_missing_baseline_seconds")
            if not has_executable_oracle and "test_command" not in task.metadata:
                issues.append("open_ended_missing_test_command")

        if has_executable_oracle and "reference_files" not in task.metadata:
            issues.append("executable_task_missing_reference_files")

        return TaskValidationResult(
            task_id=task.task_id,
            valid=not issues,
            issues=issues,
        )

    async def validate_executable(
        self,
        task: TaskSpec,
        arena_manager: Any,
    ) -> TaskValidationResult:
        """Validate starter failure and reference pass for executable tasks."""

        result = self.validate_static(task)
        if not task.test_commands:
            return result

        starter = await self._judge_task_state(task, arena_manager)
        result.starter_result = starter.to_dict()
        if starter.oracle_passed:
            result.issues.append("starter_already_passes")

        reference_files = task.metadata.get("reference_files")
        if not isinstance(reference_files, dict) or not reference_files:
            result.issues.append("missing_executable_reference_files")
        else:
            reference = await self._judge_task_state(
                task,
                arena_manager,
                patch_files={str(k): str(v) for k, v in reference_files.items()},
            )
            result.reference_result = reference.to_dict()
            if not reference.success:
                result.issues.append("reference_does_not_pass")

        result.issues = sorted(set(result.issues))
        result.valid = not result.issues
        return result

    async def _judge_task_state(
        self,
        task: TaskSpec,
        arena_manager: Any,
        *,
        patch_files: dict[str, str] | None = None,
    ) -> TaskJudgeResult:
        container_id = await arena_manager.async_create_container(task)
        try:
            if patch_files:
                await arena_manager.async_copy_files_to_container(container_id, patch_files)
            trajectory = Trajectory(
                task=task,
                steps=[
                    StepRecord(
                        step_idx=0,
                        action_type=ActionType.SUBMIT,
                        action_content="validator submitted for oracle check",
                        observation="",
                    )
                ],
            )
            return await self.judge.judge(
                trajectory,
                task,
                arena_manager=arena_manager,
                container_id=container_id,
            )
        finally:
            await arena_manager.async_destroy_container(container_id)


def load_task_bank(path: str | Path) -> list[TaskSpec]:
    """Load a JSON task bank into TaskSpec objects."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        rows = data.get("tasks", [])
    else:
        rows = data
    tasks: list[TaskSpec] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        task_type = row.get("task_type", TaskType.CODE_DEBUGGING.value)
        tasks.append(
            TaskSpec(
                task_id=row.get("task_id", ""),
                task_type=TaskType(task_type),
                description=row.get("description", ""),
                initial_files=row.get("initial_files", {}) or {},
                test_commands=row.get("test_commands", []) or [],
                expected_output=row.get("expected_output"),
                difficulty=row.get("difficulty", 0.5),
                metadata=row.get("metadata", {}) or {},
            )
        )
    return tasks
