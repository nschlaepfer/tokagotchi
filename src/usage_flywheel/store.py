"""Filesystem store for local product-use traces."""

from __future__ import annotations

import json
import re
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

from src.usage_flywheel.models import UsageEvent, UsageTrace


class UsageTraceStore:
    """Append-friendly local trace store.

    Full traces are written to ``traces/<trace_id>.json``. A compact append-only
    index is written to ``usage_index.jsonl`` so future optimizers can scan
    recent usage without loading every full trace.
    """

    def __init__(self, base_dir: str | Path = "data/usage_traces") -> None:
        self.base_dir = Path(base_dir)
        self.trace_dir = self.base_dir / "traces"
        self.index_path = self.base_dir / "usage_index.jsonl"
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def save(self, trace: UsageTrace) -> Path:
        """Persist a full trace and append its latest summary to the index."""

        path = self.path_for(trace.trace_id)
        path.write_text(json.dumps(trace.to_dict(), indent=2, default=_json_default), encoding="utf-8")
        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace.summary(), default=_json_default) + "\n")
        return path

    def load(self, trace_id: str) -> UsageTrace:
        """Load one full trace by id."""

        data = json.loads(self.path_for(trace_id).read_text(encoding="utf-8"))
        events = [
            UsageEvent(
                event_type=e.get("event_type", "unknown"),
                content=e.get("content", ""),
                metadata=e.get("metadata", {}),
                timestamp=e.get("timestamp", 0.0),
            )
            for e in data.pop("events", [])
        ]
        trace = UsageTrace(**data)
        trace.events = events
        return trace

    def latest(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the latest unique compact trace summaries."""

        if not self.index_path.exists():
            return []
        lines = [line for line in self.index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for line in reversed(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            trace_id = str(row.get("trace_id", ""))
            if not trace_id or trace_id in seen:
                continue
            seen.add(trace_id)
            rows.append(row)
            if len(rows) >= n:
                break
        rows.reverse()
        return rows

    def path_for(self, trace_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", trace_id):
            raise ValueError(f"Invalid trace id: {trace_id!r}")
        return self.trace_dir / f"{trace_id}.json"


def append_pending_example(
    path: str | Path,
    *,
    example: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    """Append one chat-format example using PendingBuffer's JSONL shape."""

    pending_path = Path(path)
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"example": example, "metadata": metadata}
    with open(pending_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")
    return pending_path


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return value.__dict__
    return str(value)
