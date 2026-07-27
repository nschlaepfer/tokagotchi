"""Canonical oracle-backed task judge.

This is the single success authority for arena tasks. A submit action ends an
episode; it does not by itself prove task success.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any

from src.models import ActionType, TaskSpec, Trajectory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TestResult:
    """Result for one executable oracle command."""

    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    timed_out: bool = False

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@dataclass(frozen=True)
class TaskJudgeResult:
    """Canonical task judgment consumed by all loops."""

    task_id: str
    task_type: str
    submitted: bool
    submitted_answer: str = ""
    oracle_passed: bool = False
    success: bool = False
    partial_score: float = 0.0
    test_results: tuple[TestResult, ...] = ()
    safety_violations: tuple[str, ...] = ()
    failure_reason: str = ""
    reward_components: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskJudge:
    """Evaluate a completed trajectory against a task oracle."""

    def __init__(self, *, command_timeout_seconds: int = 60) -> None:
        self.command_timeout_seconds = command_timeout_seconds

    async def judge(
        self,
        trajectory: Trajectory,
        task_spec: TaskSpec | None = None,
        *,
        arena_manager: Any | None = None,
        container_id: str | None = None,
    ) -> TaskJudgeResult:
        """Return the canonical judgment for one trajectory.

        ``arena_manager`` and ``container_id`` are required for executable
        oracles such as code-debugging tests. They must refer to the same live
        arena state modified by the agent.
        """

        task = task_spec or trajectory.task
        if task is None:
            return TaskJudgeResult(
                task_id="",
                task_type="unknown",
                submitted=False,
                failure_reason="missing_task",
                safety_violations=("missing_task",),
            )

        submitted, submitted_answer = _extract_submission(trajectory)
        safety_violations = _collect_safety_violations(trajectory)

        task_type = task.task_type.value
        if task.test_commands and task_type in {
            "code_debugging",
            "info_gathering",
            "api_orchestration",
        }:
            result = await self._judge_executable_task(
                task,
                submitted=submitted,
                submitted_answer=submitted_answer,
                safety_violations=safety_violations,
                arena_manager=arena_manager,
                container_id=container_id,
            )
        elif task_type == "code_debugging":
            result = _result(
                task,
                submitted,
                submitted_answer,
                safety_violations,
                failure_reason="missing_test_commands",
            )
        elif task_type in {"info_gathering", "api_orchestration"}:
            result = self._judge_answer_task(
                task,
                submitted=submitted,
                submitted_answer=submitted_answer,
                safety_violations=safety_violations,
            )
        elif task_type == "open_ended_optimization":
            result = await self._judge_open_ended_task(
                task,
                submitted=submitted,
                submitted_answer=submitted_answer,
                safety_violations=safety_violations,
                arena_manager=arena_manager,
                container_id=container_id,
            )
        else:
            result = TaskJudgeResult(
                task_id=task.task_id,
                task_type=task_type,
                submitted=submitted,
                submitted_answer=submitted_answer,
                safety_violations=tuple(safety_violations),
                failure_reason="unknown_task_type",
            )

        trajectory.success = result.success
        trajectory.total_reward = result.partial_score
        trajectory.evaluation = result.to_dict()
        return result

    async def _judge_executable_task(
        self,
        task: TaskSpec,
        *,
        submitted: bool,
        submitted_answer: str,
        safety_violations: list[str],
        arena_manager: Any | None,
        container_id: str | None,
    ) -> TaskJudgeResult:
        if not task.test_commands:
            return _result(
                task,
                submitted,
                submitted_answer,
                safety_violations,
                failure_reason="missing_test_commands",
            )
        if arena_manager is None or not container_id:
            return _result(
                task,
                submitted,
                submitted_answer,
                safety_violations + ["missing_live_arena_for_tests"],
                failure_reason="missing_live_arena_for_tests",
            )

        test_results = [
            await self._run_oracle_command(arena_manager, container_id, command)
            for command in task.test_commands
        ]

        passed = sum(1 for item in test_results if item.passed)
        total = len(test_results)
        oracle_passed = total > 0 and passed == total
        partial = passed / total if total else 0.0
        failure = "" if oracle_passed else "test_commands_failed"
        return _result(
            task,
            submitted,
            submitted_answer,
            safety_violations,
            oracle_passed=oracle_passed,
            partial_score=partial,
            test_results=tuple(test_results),
            failure_reason=failure,
            reward_components={"oracle": partial},
        )

    def _judge_answer_task(
        self,
        task: TaskSpec,
        *,
        submitted: bool,
        submitted_answer: str,
        safety_violations: list[str],
    ) -> TaskJudgeResult:
        if task.expected_output is None:
            return _result(
                task,
                submitted,
                submitted_answer,
                safety_violations,
                failure_reason="missing_expected_output",
            )
        expected = _normalize(task.expected_output)
        actual = _normalize(submitted_answer)
        exact = bool(actual) and actual == expected
        score = 1.0 if exact else round(SequenceMatcher(None, actual, expected).ratio(), 4)
        return _result(
            task,
            submitted,
            submitted_answer,
            safety_violations,
            oracle_passed=exact,
            partial_score=score if submitted else 0.0,
            failure_reason="" if exact else "answer_mismatch",
            reward_components={"answer_match": score if submitted else 0.0},
            details={"expected": task.expected_output},
        )

    async def _judge_open_ended_task(
        self,
        task: TaskSpec,
        *,
        submitted: bool,
        submitted_answer: str,
        safety_violations: list[str],
        arena_manager: Any | None,
        container_id: str | None,
    ) -> TaskJudgeResult:
        benchmark_cmd = task.metadata.get("benchmark_command")
        baseline_seconds = task.metadata.get("baseline_seconds")
        if not benchmark_cmd or baseline_seconds is None:
            return _result(
                task,
                submitted,
                submitted_answer,
                safety_violations,
                failure_reason="missing_benchmark_or_baseline",
            )
        try:
            baseline = float(baseline_seconds)
        except (TypeError, ValueError):
            return _result(
                task,
                submitted,
                submitted_answer,
                safety_violations,
                failure_reason="invalid_baseline_seconds",
            )
        if baseline <= 0:
            return _result(
                task,
                submitted,
                submitted_answer,
                safety_violations,
                failure_reason="invalid_baseline_seconds",
            )
        if arena_manager is None or not container_id:
            return _result(
                task,
                submitted,
                submitted_answer,
                safety_violations + ["missing_live_arena_for_benchmark"],
                failure_reason="missing_live_arena_for_benchmark",
            )

        test_cmd = task.metadata.get("test_command")
        correctness_commands = list(task.test_commands or ([test_cmd] if test_cmd else []))
        if not correctness_commands:
            return _result(
                task,
                submitted,
                submitted_answer,
                safety_violations,
                oracle_passed=False,
                partial_score=0.0,
                failure_reason="open_ended_requires_test_command",
            )

        correctness_results = [
            await self._run_oracle_command(arena_manager, container_id, command)
            for command in correctness_commands
        ]
        correctness_passed = sum(1 for item in correctness_results if item.passed)
        correctness_total = len(correctness_results)
        correctness_score = correctness_passed / correctness_total if correctness_total else 0.0
        correctness_ok = correctness_total > 0 and correctness_passed == correctness_total

        benchmark_result = await self._run_oracle_command(
            arena_manager,
            container_id,
            str(benchmark_cmd),
        )
        benchmark_seconds = benchmark_result.duration_seconds
        benchmark_exited = benchmark_result.passed
        benchmark_ok = benchmark_exited and benchmark_seconds <= baseline
        if benchmark_exited and benchmark_seconds > 0:
            benchmark_score = min(1.0, baseline / benchmark_seconds)
        elif benchmark_exited:
            benchmark_score = 1.0
        else:
            benchmark_score = 0.0

        if not correctness_ok:
            failure = "test_commands_failed"
        elif not benchmark_exited:
            failure = "benchmark_command_failed"
        elif not benchmark_ok:
            failure = "benchmark_slower_than_baseline"
        else:
            failure = ""

        oracle_passed = correctness_ok and benchmark_ok
        speed_ratio = baseline / benchmark_seconds if benchmark_seconds > 0 else None
        partial_score = round(correctness_score * benchmark_score, 4)
        return _result(
            task,
            submitted,
            submitted_answer,
            safety_violations,
            oracle_passed=oracle_passed,
            partial_score=partial_score,
            test_results=tuple(correctness_results + [benchmark_result]),
            failure_reason=failure,
            reward_components={
                "correctness": correctness_score,
                "benchmark": round(benchmark_score, 4),
                "oracle": partial_score,
            },
            details={
                "baseline_seconds": baseline,
                "benchmark_seconds": benchmark_seconds,
                "benchmark_passed": benchmark_ok,
                "speed_ratio": round(speed_ratio, 4) if speed_ratio is not None else None,
            },
        )

    async def _run_oracle_command(
        self,
        arena_manager: Any,
        container_id: str,
        command: str,
    ) -> TestResult:
        started = time.monotonic()
        try:
            stdout, stderr, exit_code = await arena_manager.async_exec_in_container(
                container_id,
                command,
                timeout=self.command_timeout_seconds,
            )
            return TestResult(
                command=command,
                exit_code=exit_code,
                stdout=stdout[-4000:],
                stderr=stderr[-4000:],
                duration_seconds=time.monotonic() - started,
            )
        except TimeoutError as exc:
            return TestResult(
                command=command,
                exit_code=124,
                stderr=str(exc),
                duration_seconds=time.monotonic() - started,
                timed_out=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Task judge failed to run oracle: %s", command)
            return TestResult(
                command=command,
                exit_code=1,
                stderr=str(exc),
                duration_seconds=time.monotonic() - started,
            )


def _result(
    task: TaskSpec,
    submitted: bool,
    submitted_answer: str,
    safety_violations: list[str],
    *,
    oracle_passed: bool = False,
    partial_score: float = 0.0,
    test_results: tuple[TestResult, ...] = (),
    failure_reason: str = "",
    reward_components: dict[str, float] | None = None,
    details: dict[str, Any] | None = None,
) -> TaskJudgeResult:
    success = submitted and oracle_passed and not safety_violations
    if not submitted and not failure_reason:
        failure_reason = "not_submitted"
    elif safety_violations and not failure_reason:
        failure_reason = "safety_violation"

    return TaskJudgeResult(
        task_id=task.task_id,
        task_type=task.task_type.value,
        submitted=submitted,
        submitted_answer=submitted_answer,
        oracle_passed=oracle_passed,
        success=success,
        partial_score=partial_score if submitted else 0.0,
        test_results=test_results,
        safety_violations=tuple(safety_violations),
        failure_reason=failure_reason,
        reward_components=reward_components or {},
        details=details or {},
    )


def _extract_submission(trajectory: Trajectory) -> tuple[bool, str]:
    for step in reversed(trajectory.steps):
        if step.action_type == ActionType.SUBMIT:
            return True, step.action_content.strip()
    return False, ""


def _collect_safety_violations(trajectory: Trajectory) -> list[str]:
    violations: list[str] = []
    for step in trajectory.steps:
        violation = step.metadata.get("safety_violation")
        if violation:
            violations.append(str(violation))
    return violations


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
