"""Hard gates for unsafe autonomous self-improvement actions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import MasterConfig


class SafetyGateError(RuntimeError):
    """Raised when a protected autonomous operation is not allowed."""


def require_autonomous_sft_enabled(config: MasterConfig) -> None:
    _require_operation(config, "autonomous_sft", config.safety.enable_autonomous_sft)


def require_autonomous_rl_enabled(config: MasterConfig) -> None:
    _require_operation(config, "autonomous_rl", config.safety.enable_autonomous_rl)


def require_checkpoint_promotion_enabled(config: MasterConfig) -> None:
    _require_operation(
        config,
        "checkpoint_promotion",
        config.safety.enable_checkpoint_promotion,
    )


def _require_operation(config: MasterConfig, operation: str, enabled: bool) -> None:
    if not enabled:
        raise SafetyGateError(
            f"{operation} is disabled by safety config. "
            "Enable it only after canonical TaskJudge and benchmark validation evidence pass."
        )

    evidence = load_gate_evidence(config.safety.gate_evidence_path)
    issues = validate_gate_evidence(evidence)
    if issues:
        raise SafetyGateError(
            f"{operation} requires complete truth-gate evidence at "
            f"{config.safety.gate_evidence_path}: {', '.join(issues)}"
        )


def load_gate_evidence(path: str | Path) -> dict[str, Any]:
    evidence_path = Path(path)
    if not evidence_path.exists():
        return {}
    try:
        data = json.loads(evidence_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def validate_gate_evidence(evidence: dict[str, Any]) -> list[str]:
    """Return fail-closed issue codes for autonomous learning evidence."""

    issues: list[str] = []
    if not evidence:
        return ["missing_gate_evidence"]
    if evidence.get("schema_version") != 1:
        issues.append("schema_version_must_be_1")
    if evidence.get("truth_grounding_passed") is not True:
        issues.append("truth_grounding_not_passed")
    if evidence.get("human_reviewed") is not True:
        issues.append("human_review_required")
    if not evidence.get("git_commit"):
        issues.append("missing_git_commit")
    if evidence.get("git_dirty") is not False:
        issues.append("git_tree_must_be_clean")
    if evidence.get("task_judge_canonical") is not True:
        issues.append("task_judge_not_canonical")

    _validate_arena_evidence(evidence.get("arena"), issues)
    _validate_task_bank_evidence(evidence.get("task_bank"), issues)
    _validate_test_evidence(evidence.get("tests"), issues)
    _validate_reproducible_commands(evidence.get("reproducible_commands"), issues)
    return issues


def _validate_arena_evidence(arena: Any, issues: list[str]) -> None:
    if not isinstance(arena, dict):
        issues.append("missing_arena_evidence")
        return
    if arena.get("backend") != "docker":
        issues.append("docker_arena_validation_required")
    if arena.get("network") != "none":
        issues.append("arena_network_must_be_none")
    if arena.get("fail_closed_checked") is not True:
        issues.append("arena_fail_closed_not_checked")
    if arena.get("unsafe_host_execution") is not False:
        issues.append("unsafe_host_execution_not_allowed_for_gate")


def _validate_task_bank_evidence(task_bank: Any, issues: list[str]) -> None:
    if not isinstance(task_bank, dict):
        issues.append("missing_task_bank_evidence")
        return

    tasks = _as_int(task_bank.get("tasks"))
    executable_tasks = _as_int(task_bank.get("executable_tasks"))
    starters_failed = _as_int(task_bank.get("starters_failed"))
    references_passed = _as_int(task_bank.get("references_passed"))

    if task_bank.get("static_valid") is not True:
        issues.append("task_bank_static_not_valid")
    if task_bank.get("executable_valid") is not True:
        issues.append("task_bank_executable_not_valid")
    if task_bank.get("invalid_task_ids") not in ([], None):
        issues.append("task_bank_has_invalid_tasks")
    if tasks is None or tasks <= 0:
        issues.append("task_bank_tasks_missing")
    if executable_tasks is None or executable_tasks <= 0:
        issues.append("task_bank_executable_tasks_missing")
    if starters_failed is None:
        issues.append("starter_failure_count_missing")
    elif executable_tasks is not None and starters_failed != executable_tasks:
        issues.append("starter_failure_proof_incomplete")
    if references_passed is None:
        issues.append("reference_pass_count_missing")
    elif executable_tasks is not None and references_passed != executable_tasks:
        issues.append("reference_pass_proof_incomplete")
    if (
        tasks is not None
        and executable_tasks is not None
        and executable_tasks > tasks
    ):
        issues.append("task_bank_executable_count_invalid")

    benchmark_tasks = _as_int(task_bank.get("benchmark_tasks"))
    benchmarks_passed = _as_int(task_bank.get("benchmarks_passed"))
    if benchmark_tasks is None:
        issues.append("benchmark_task_count_missing")
    elif benchmarks_passed is None:
        issues.append("benchmark_pass_count_missing")
    elif benchmarks_passed != benchmark_tasks:
        issues.append("benchmark_proof_incomplete")


def _validate_test_evidence(tests: Any, issues: list[str]) -> None:
    if not isinstance(tests, dict):
        issues.append("missing_test_evidence")
        return
    if tests.get("failures") != 0:
        issues.append("test_failures_present")
    passed = _as_int(tests.get("passed"))
    total = _as_int(tests.get("total"))
    skipped = _as_int(tests.get("skipped")) or 0
    failures = _as_int(tests.get("failures")) or 0
    if passed is None or passed <= 0:
        issues.append("test_pass_count_missing")
    if total is None or total <= 0:
        issues.append("test_total_missing")
    if total is not None and passed is not None and passed + skipped + failures != total:
        issues.append("test_counts_do_not_sum")


def _validate_reproducible_commands(commands: Any, issues: list[str]) -> None:
    if not isinstance(commands, list) or not commands:
        issues.append("missing_reproducible_commands")
        return
    if len(commands) < 4:
        issues.append("reproducible_command_coverage_incomplete")

    command_texts: list[str] = []
    for item in commands:
        if not isinstance(item, dict):
            issues.append("invalid_reproducible_command_entry")
            continue
        command = item.get("command")
        if not isinstance(command, str) or not command.strip():
            issues.append("invalid_reproducible_command_entry")
        else:
            command_texts.append(command)
        if item.get("exit_code") != 0:
            issues.append("reproducible_command_failed")

    joined = "\n".join(command_texts)
    required_markers = {
        "compileall": "missing_compile_proof_command",
        "git diff --check": "missing_diff_check_command",
        "scripts/test_all_loops.py": "missing_integration_test_command",
        "validate_task_bank.py": "missing_task_bank_validation_command",
    }
    for marker, issue in required_markers.items():
        if marker not in joined:
            issues.append(issue)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None
