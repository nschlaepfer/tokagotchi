#!/usr/bin/env python3
"""Validate a tokagotchi task bank before optimization/training uses it."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.arena.docker_manager import create_arena_manager  # noqa: E402
from src.evaluation.task_judge import TaskJudge  # noqa: E402
from src.evaluation.task_bank_validator import TaskBankValidator, load_task_bank  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate task-bank oracle integrity.")
    parser.add_argument("task_bank", type=Path, help="Path to a JSON task bank.")
    parser.add_argument("--static-only", action="store_true", help="Skip executable starter/reference checks.")
    parser.add_argument(
        "--command-timeout-seconds",
        type=int,
        default=10,
        help="Per-oracle command timeout for executable validation.",
    )
    parser.add_argument(
        "--unsafe-host-code-execution",
        action="store_true",
        help="Allow explicit host subprocess execution for local validation only.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a compact proof summary instead of full per-task details.",
    )
    parser.add_argument("--json-out", type=Path, help="Optional path to write validation JSON.")
    return parser.parse_args()


async def main_async() -> int:
    args = parse_args()
    tasks = load_task_bank(args.task_bank)
    validator = TaskBankValidator(
        judge=TaskJudge(command_timeout_seconds=args.command_timeout_seconds)
    )
    arena_manager = None
    if not args.static_only:
        arena_manager = create_arena_manager(
            use_docker=False if args.unsafe_host_code_execution else None,
            allow_unsafe_host_execution=args.unsafe_host_code_execution,
        )

    results = []
    for task in tasks:
        if args.static_only:
            result = validator.validate_static(task)
        else:
            result = await validator.validate_executable(task, arena_manager)
        results.append(result.to_dict())

    report = {
        "task_bank": str(args.task_bank),
        "tasks": len(results),
        "valid": all(row["valid"] for row in results),
        "results": results,
    }

    print_payload = _summarize(report) if args.summary else report
    text = json.dumps(print_payload, indent=2)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["valid"] else 1


def _summarize(report: dict) -> dict:
    results = report["results"]
    executable = [
        row for row in results
        if row.get("starter_result") is not None or row.get("reference_result") is not None
    ]
    starters_failed = sum(
        1 for row in executable
        if (row.get("starter_result") or {}).get("oracle_passed") is False
    )
    references_passed = sum(
        1 for row in executable
        if (row.get("reference_result") or {}).get("success") is True
    )
    benchmark = [
        row for row in executable
        if (
            (row.get("reference_result") or {}).get("task_type")
            == "open_ended_optimization"
        )
    ]
    benchmarks_passed = sum(
        1 for row in benchmark
        if (
            (row.get("reference_result") or {})
            .get("details", {})
            .get("benchmark_passed")
            is True
        )
    )
    invalid = [row["task_id"] for row in results if not row["valid"]]
    return {
        "task_bank": report["task_bank"],
        "tasks": report["tasks"],
        "valid": report["valid"],
        "executable_tasks": len(executable),
        "starters_failed": starters_failed,
        "references_passed": references_passed,
        "benchmark_tasks": len(benchmark),
        "benchmarks_passed": benchmarks_passed,
        "invalid_task_ids": invalid,
    }


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
