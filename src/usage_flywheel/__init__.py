"""Product-use data flywheel for tokagotchi.

The usage flywheel records local user tasks, captures local student attempts,
optionally boosts failures through Codex, and promotes only explicitly accepted
non-private traces into pending SFT examples.
"""

from src.usage_flywheel.feedback import (
    PromotionResult,
    Trainability,
    apply_trace_feedback,
    promote_trace_to_pending,
    trace_trainability,
)
from src.usage_flywheel.flywheel import FlywheelResult, UsageFlywheel
from src.usage_flywheel.models import UsageEvent, UsageTrace
from src.usage_flywheel.store import UsageTraceStore

__all__ = [
    "FlywheelResult",
    "PromotionResult",
    "Trainability",
    "UsageEvent",
    "UsageFlywheel",
    "UsageTrace",
    "UsageTraceStore",
    "apply_trace_feedback",
    "promote_trace_to_pending",
    "trace_trainability",
]
