"""Shared data structures for arena tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Standard result returned by every arena tool."""

    stdout: str
    stderr: str
    exit_code: int
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    @property
    def output(self) -> str:
        """Combined human-readable output."""
        parts: list[str] = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"[stderr] {self.stderr}")
        return "\n".join(parts) if parts else "(no output)"
