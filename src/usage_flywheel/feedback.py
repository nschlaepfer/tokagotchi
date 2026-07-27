"""User feedback and trace promotion controls for product-use traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.usage_flywheel.models import UsageTrace
from src.usage_flywheel.redaction import redact_text
from src.usage_flywheel.store import UsageTraceStore, append_pending_example


ACCEPTED = "accepted"
REJECTED = "rejected"
UNREVIEWED = "unreviewed"
MIN_USEFULNESS_RATING = 3


@dataclass(frozen=True)
class Trainability:
    """Explains whether a usage trace may enter the pending training buffer."""

    trainable: bool
    reason: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionResult:
    """Result from attempting to promote one trace into pending training data."""

    promoted: bool
    trainability: Trainability
    trace: UsageTrace
    pending_example_path: Path | None = None


def apply_trace_feedback(
    trace: UsageTrace,
    *,
    decision: str | None = None,
    rating: int | None = None,
    note: str = "",
    selected_output: str | None = None,
    mark_sensitive: bool | None = None,
    mark_private: bool | None = None,
    redact: bool = True,
) -> UsageTrace:
    """Apply explicit user feedback to a trace in-place and return it."""

    if rating is not None and not 1 <= int(rating) <= 5:
        raise ValueError("rating must be an integer from 1 to 5")

    if selected_output is not None:
        output = selected_output
        if redact:
            output, report = redact_text(output)
            trace.redaction_report.setdefault("feedback_redactions", []).append(
                report.as_dict()
            )
        trace.selected_output = output
        trace.add_event("trace_edited", "Selected output edited by user.")

    if rating is not None:
        trace.usefulness_rating = int(rating)
        trace.add_event("trace_rated", str(int(rating)), rating=int(rating))

    if note:
        trace.feedback_note = note
        trace.add_event("feedback_note", note)

    if mark_sensitive is not None:
        trace.is_sensitive = bool(mark_sensitive)
        trace.add_event(
            "sensitivity_changed",
            "sensitive" if trace.is_sensitive else "not_sensitive",
        )
        if trace.is_sensitive:
            trace.training_consent = "no_training"

    if mark_private is not None:
        trace.is_private = bool(mark_private)
        trace.add_event(
            "privacy_changed",
            "private" if trace.is_private else "not_private",
        )
        if trace.is_private:
            trace.privacy_mode = "private"
            trace.training_consent = "no_training"

    if decision is not None:
        normalized = decision.lower().strip()
        if normalized in {"accept", ACCEPTED}:
            trace.review_status = ACCEPTED
            trace.add_event("trace_accepted", note)
        elif normalized in {"reject", REJECTED}:
            trace.review_status = REJECTED
            trace.add_event("trace_rejected", note)
        elif normalized in {"unreviewed", "reset"}:
            trace.review_status = UNREVIEWED
            trace.add_event("trace_unreviewed", note)
        else:
            raise ValueError("decision must be accepted, rejected, or unreviewed")

    _refresh_trainability(trace)
    trace.status = _reviewed_status(trace)
    return trace


def trace_trainability(trace: UsageTrace) -> Trainability:
    """Return the current trainability decision and reason for one trace."""

    if trace.pending_example_path:
        return Trainability(False, "already_promoted", "Trace is already in the pending buffer.")
    if trace.review_status != ACCEPTED:
        if trace.review_status == REJECTED:
            return Trainability(False, "rejected", "User rejected this trace.")
        return Trainability(False, "not_accepted", "User has not accepted this trace.")
    if trace.is_private:
        return Trainability(False, "marked_private", "Private traces are never trainable.")
    if trace.is_sensitive:
        return Trainability(False, "marked_sensitive", "Sensitive traces are never trainable.")
    if trace.training_consent == "no_training":
        return Trainability(False, "no_training_consent", "Trace is marked no-training.")
    if trace.usefulness_rating is not None and trace.usefulness_rating < MIN_USEFULNESS_RATING:
        return Trainability(
            False,
            "usefulness_rating_too_low",
            f"Rating {trace.usefulness_rating}/5 is below {MIN_USEFULNESS_RATING}/5.",
        )
    if not trace.selected_answer().strip():
        return Trainability(False, "missing_selected_output", "No selected answer is available.")
    return Trainability(True, "accepted", "Accepted, non-private trace with usable output.")


def promote_trace_to_pending(
    store: UsageTraceStore,
    trace_id: str,
    pending_jsonl: str | Path,
) -> PromotionResult:
    """Append an accepted trace to the pending SFT buffer, or explain refusal."""

    trace = store.load(trace_id)
    trainability = _refresh_trainability(trace)
    if not trainability.trainable:
        trace.add_event(
            "promotion_blocked",
            trainability.reason,
            detail=trainability.detail,
        )
        store.save(trace)
        return PromotionResult(False, trainability, trace)

    from src.usage_flywheel.flywheel import build_training_example

    example, metadata = build_training_example(trace, trace.selected_answer())
    metadata.update(
        {
            "review_status": trace.review_status,
            "usefulness_rating": trace.usefulness_rating,
            "is_sensitive": trace.is_sensitive,
            "is_private": trace.is_private,
            "trainability_reason": trainability.reason,
        }
    )
    pending_path = append_pending_example(
        pending_jsonl,
        example=example,
        metadata=metadata,
    )
    trace.pending_example_path = str(pending_path)
    trace.status = "promoted"
    trace.add_event("pending_example", str(pending_path), **metadata)
    _refresh_trainability(trace)
    store.save(trace)
    return PromotionResult(True, trainability, trace, pending_path)


def _refresh_trainability(trace: UsageTrace) -> Trainability:
    trainability = trace_trainability(trace)
    trace.metadata["trainability"] = trainability.to_dict()
    return trainability


def _reviewed_status(trace: UsageTrace) -> str:
    if trace.pending_example_path:
        return "promoted"
    if trace.is_private:
        return "private"
    if trace.is_sensitive:
        return "sensitive"
    if trace.review_status == ACCEPTED:
        return "accepted"
    if trace.review_status == REJECTED:
        return "rejected"
    if trace.selected_answer().strip():
        return "needs_review"
    return trace.status or "recorded"
