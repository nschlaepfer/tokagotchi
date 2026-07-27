"""Small local redaction helpers for persisted usage traces."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "private_key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            flags=re.DOTALL,
        ),
    ),
    (
        "secret_assignment",
        re.compile(
            r"(?i)\b(password|passwd|secret|api[_-]?key|token)\s*=\s*([^\s\"']{8,}|[\"'][^\"']{8,}[\"'])"
        ),
    ),
]


@dataclass
class RedactionReport:
    """Counts of redactions applied to one text payload."""

    replacements: dict[str, int] = field(default_factory=dict)
    truncated: bool = False
    original_chars: int = 0
    stored_chars: int = 0

    @property
    def total_replacements(self) -> int:
        return sum(self.replacements.values())

    def as_dict(self) -> dict[str, object]:
        return {
            "replacements": dict(self.replacements),
            "total_replacements": self.total_replacements,
            "truncated": self.truncated,
            "original_chars": self.original_chars,
            "stored_chars": self.stored_chars,
        }


def redact_text(text: str, *, max_chars: int = 20000) -> tuple[str, RedactionReport]:
    """Redact likely credentials and bound persisted text size.

    This is intentionally conservative and local. It is not a full DLP system;
    it only prevents common high-risk tokens from being written into training
    traces by default.
    """

    report = RedactionReport(original_chars=len(text))
    redacted = text

    for name, pattern in _SECRET_PATTERNS:
        redacted, count = pattern.subn(f"[REDACTED:{name}]", redacted)
        if count:
            report.replacements[name] = count

    if len(redacted) > max_chars:
        redacted = redacted[:max_chars] + "\n[TRUNCATED]"
        report.truncated = True

    report.stored_chars = len(redacted)
    return redacted, report
