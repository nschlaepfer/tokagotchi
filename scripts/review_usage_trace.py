#!/usr/bin/env python3
"""Review and promote local product-use traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.usage_flywheel import (  # noqa: E402
    UsageTraceStore,
    apply_trace_feedback,
    promote_trace_to_pending,
    trace_trainability,
)
from src.usage_flywheel.redaction import redact_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review tokagotchi usage traces.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config")
    parser.add_argument("--trace-dir", type=Path, help="Override usage trace directory.")
    parser.add_argument("--pending-jsonl", type=Path, help="Override pending buffer path.")
    sub = parser.add_subparsers(dest="command", required=True)

    list_p = sub.add_parser("list", help="List recent traces with trainability reasons.")
    list_p.add_argument("-n", type=int, default=10)

    show_p = sub.add_parser("show", help="Show one trace summary and trainability reason.")
    show_p.add_argument("trace_id")
    show_p.add_argument("--full", action="store_true", help="Print the full trace JSON.")

    accept_p = sub.add_parser("accept", help="Accept a trace for possible training.")
    accept_p.add_argument("trace_id")
    _feedback_args(accept_p, allow_answer=True)
    accept_p.add_argument("--promote", action="store_true", help="Append to pending buffer if trainable.")

    reject_p = sub.add_parser("reject", help="Reject a trace from training.")
    reject_p.add_argument("trace_id")
    _feedback_args(reject_p)

    edit_p = sub.add_parser("edit", help="Edit the selected training answer.")
    edit_p.add_argument("trace_id")
    _feedback_args(edit_p, allow_answer=True, require_answer=True)

    private_p = sub.add_parser("private", help="Mark a trace private and no-training.")
    private_p.add_argument("trace_id")
    _feedback_args(private_p)

    sensitive_p = sub.add_parser("sensitive", help="Mark a trace sensitive and no-training.")
    sensitive_p.add_argument("trace_id")
    _feedback_args(sensitive_p)

    rate_p = sub.add_parser("rate", help="Rate trace usefulness from 1 to 5.")
    rate_p.add_argument("trace_id")
    rate_p.add_argument("rating", type=int)
    rate_p.add_argument("--note", default="")

    train_p = sub.add_parser("trainability", help="Show whether a trace is trainable.")
    train_p.add_argument("trace_id")

    promote_p = sub.add_parser("promote", help="Promote an accepted trace to pending training data.")
    promote_p.add_argument("trace_id")

    return parser.parse_args()


def _feedback_args(
    parser: argparse.ArgumentParser,
    *,
    allow_answer: bool = False,
    require_answer: bool = False,
) -> None:
    parser.add_argument("--rating", type=int, help="Usefulness rating from 1 to 5.")
    parser.add_argument("--note", default="", help="User review note.")
    if allow_answer:
        group = parser.add_mutually_exclusive_group(required=require_answer)
        group.add_argument("--answer", help="Edited selected answer.")
        group.add_argument("--answer-file", type=Path, help="Read edited selected answer from file.")


def main() -> None:
    raise SystemExit(main_sync())


def main_sync() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    trace_dir = args.trace_dir or Path(cfg.usage_flywheel.trace_dir)
    pending_jsonl = args.pending_jsonl or Path(cfg.usage_flywheel.pending_jsonl)
    store = UsageTraceStore(trace_dir)

    if args.command == "list":
        for row in store.latest(args.n):
            trace = store.load(row["trace_id"])
            trainability = trace_trainability(trace)
            print(
                f"{trace.trace_id} status={trace.status} review={trace.review_status} "
                f"trainable={trainability.trainable} reason={trainability.reason} "
                f"rating={trace.usefulness_rating} private={trace.is_private} "
                f"sensitive={trace.is_sensitive} task={trace.user_task[:80]!r}"
            )
        return 0

    if args.command == "show":
        trace = store.load(args.trace_id)
        trainability = trace_trainability(trace)
        payload = trace.to_dict() if args.full else trace.summary()
        payload["trainability"] = trainability.to_dict()
        print(json.dumps(payload, indent=2))
        return 0 if trainability.trainable else 2

    if args.command == "trainability":
        trace = store.load(args.trace_id)
        trainability = trace_trainability(trace)
        print(json.dumps(trainability.to_dict(), indent=2))
        return 0 if trainability.trainable else 2

    if args.command == "promote":
        result = promote_trace_to_pending(store, args.trace_id, pending_jsonl)
        _print_promotion(result)
        return 0 if result.promoted else 2

    trace = store.load(args.trace_id)
    if args.command == "accept":
        apply_trace_feedback(
            trace,
            decision="accepted",
            rating=args.rating,
            note=args.note,
            selected_output=_read_answer(args),
        )
        store.save(trace)
        if args.promote:
            result = promote_trace_to_pending(store, trace.trace_id, pending_jsonl)
            _print_promotion(result)
            return 0 if result.promoted else 2
    elif args.command == "reject":
        apply_trace_feedback(trace, decision="rejected", rating=args.rating, note=args.note)
        store.save(trace)
    elif args.command == "edit":
        apply_trace_feedback(
            trace,
            rating=args.rating,
            note=args.note,
            selected_output=_read_answer(args),
        )
        store.save(trace)
    elif args.command == "private":
        apply_trace_feedback(trace, rating=args.rating, note=args.note, mark_private=True)
        store.save(trace)
    elif args.command == "sensitive":
        apply_trace_feedback(trace, rating=args.rating, note=args.note, mark_sensitive=True)
        store.save(trace)
    elif args.command == "rate":
        apply_trace_feedback(trace, rating=args.rating, note=args.note)
        store.save(trace)
    else:
        raise SystemExit(f"Unsupported command: {args.command}")

    trainability = trace_trainability(store.load(trace.trace_id))
    print(f"trace_id: {trace.trace_id}")
    print(f"status: {store.load(trace.trace_id).status}")
    print(f"trainable: {trainability.trainable}")
    print(f"reason: {trainability.reason}")
    if trainability.detail:
        print(f"detail: {trainability.detail}")
    return 0


def _read_answer(args: argparse.Namespace) -> str | None:
    answer = getattr(args, "answer", None)
    answer_file = getattr(args, "answer_file", None)
    if answer is None and answer_file is None:
        return None
    text = answer_file.read_text(encoding="utf-8") if answer_file else answer
    redacted, _ = redact_text(text or "")
    return redacted


def _print_promotion(result) -> None:
    print(f"trace_id: {result.trace.trace_id}")
    print(f"promoted: {result.promoted}")
    print(f"trainable: {result.trainability.trainable}")
    print(f"reason: {result.trainability.reason}")
    if result.pending_example_path:
        print(f"pending_example_path: {result.pending_example_path}")
    if result.trainability.detail:
        print(f"detail: {result.trainability.detail}")


if __name__ == "__main__":
    main()
