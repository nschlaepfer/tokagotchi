"""Dataclasses for product-use traces."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


def _now() -> float:
    return time.time()


@dataclass
class UsageEvent:
    """One event inside a product-use trace."""

    event_type: str
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=_now)


@dataclass
class UsageTrace:
    """A complete local trace for one user task."""

    user_task: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    status: str = "created"
    repo_root: str = ""
    git_branch: str = ""
    git_commit: str = ""
    git_dirty: bool = False
    privacy_mode: str = "local-only"
    training_consent: str = "local_training_only"
    review_status: str = "unreviewed"
    feedback_note: str = ""
    usefulness_rating: int | None = None
    is_sensitive: bool = False
    is_private: bool = False
    redaction_report: dict[str, Any] = field(default_factory=dict)
    student_model: str = ""
    teacher_provider: str = "codex"
    teacher_model: str = "gpt-5.6-sol"
    student_output: str = ""
    student_status: str = "not_run"
    codex_output: str = ""
    codex_status: str = "not_run"
    boost_used: bool = False
    failure_mode: str = ""
    selected_output: str = ""
    score: float | None = None
    evaluator_feedback: str = ""
    pending_example_path: str = ""
    events: list[UsageEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_event(self, event_type: str, content: str = "", **metadata: Any) -> None:
        self.events.append(UsageEvent(event_type=event_type, content=content, metadata=metadata))
        self.updated_at = _now()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def selected_answer(self) -> str:
        """Return the current candidate answer for training, if any."""

        if self.selected_output.strip():
            return self.selected_output
        if self.boost_used and self.codex_output.strip():
            return self.codex_output
        return self.student_output

    def summary(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "repo_root": self.repo_root,
            "git_branch": self.git_branch,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "privacy_mode": self.privacy_mode,
            "review_status": self.review_status,
            "usefulness_rating": self.usefulness_rating,
            "is_sensitive": self.is_sensitive,
            "is_private": self.is_private,
            "student_model": self.student_model,
            "student_status": self.student_status,
            "teacher_provider": self.teacher_provider,
            "teacher_model": self.teacher_model,
            "codex_status": self.codex_status,
            "boost_used": self.boost_used,
            "failure_mode": self.failure_mode,
            "score": self.score,
            "pending_example_path": self.pending_example_path,
            "trainability": self.metadata.get("trainability", {}),
            "user_task_preview": self.user_task[:240],
        }
