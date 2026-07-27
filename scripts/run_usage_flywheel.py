#!/usr/bin/env python3
"""Run one real-use tokagotchi product flywheel task."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.usage_flywheel import UsageFlywheel, UsageTraceStore, trace_trainability  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record one local user task, optionally try the local Qwen student, "
            "optionally boost with Codex, and save it for explicit user review."
        )
    )
    parser.add_argument("task", nargs="*", help="User task text. Use --task-file for longer tasks.")
    parser.add_argument("--task-file", type=Path, help="Read the user task from a file.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config", help="Config directory.")
    parser.add_argument("--trace-dir", type=Path, help="Override usage trace directory.")
    parser.add_argument("--pending-jsonl", type=Path, help="Override pending training buffer path.")
    parser.add_argument("--dry-run", action="store_true", help="Write a trace shape without calling Ollama or Codex.")
    parser.add_argument("--skip-student", action="store_true", help="Do not call the local Ollama student.")
    parser.add_argument(
        "--codex-boost",
        choices=["never", "on-failure", "always"],
        help="Override Codex boost policy for this task.",
    )
    parser.add_argument("--write", action="store_true", help="Allow Codex workspace-write sandbox for this task.")
    parser.add_argument(
        "--no-pending",
        action="store_true",
        help="Deprecated no-op. Usage traces are never appended before review.",
    )
    return parser.parse_args()


def read_task(args: argparse.Namespace) -> str:
    if args.task_file:
        return args.task_file.read_text(encoding="utf-8").strip()
    task = " ".join(args.task).strip()
    if not task:
        raise SystemExit("Provide a task argument or --task-file.")
    return task


async def main_async() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.trace_dir:
        cfg.usage_flywheel.trace_dir = str(args.trace_dir)
    if args.pending_jsonl:
        cfg.usage_flywheel.pending_jsonl = str(args.pending_jsonl)

    store = UsageTraceStore(cfg.usage_flywheel.trace_dir)
    flywheel = UsageFlywheel(cfg, repo_root=PROJECT_ROOT, trace_store=store)
    result = await flywheel.run_task(
        read_task(args),
        dry_run=args.dry_run,
        skip_student=args.skip_student,
        codex_boost=args.codex_boost,
        write=args.write,
        append_pending=False,
    )

    trace = result.trace
    trainability = trace_trainability(trace)
    print(f"trace_id: {trace.trace_id}")
    print(f"status: {trace.status}")
    print(f"student_status: {trace.student_status}")
    print(f"codex_status: {trace.codex_status}")
    print(f"boost_used: {trace.boost_used}")
    print(f"trainable: {trainability.trainable}")
    print(f"trainability_reason: {trainability.reason}")
    print(f"trace_path: {result.trace_path}")
    print(
        "review_next: "
        f"python scripts/review_usage_trace.py accept {trace.trace_id} --rating 5 --promote"
    )
    return 0 if trace.status not in {"codex_error"} else 1


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
