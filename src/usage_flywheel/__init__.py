"""Product-use data flywheel for tokagotchi.

The usage flywheel records local user tasks, captures local student attempts,
optionally boosts failures through Codex, and turns accepted boosted completions
into pending SFT examples.
"""

from src.usage_flywheel.flywheel import FlywheelResult, UsageFlywheel
from src.usage_flywheel.models import UsageEvent, UsageTrace
from src.usage_flywheel.store import UsageTraceStore

__all__ = [
    "FlywheelResult",
    "UsageEvent",
    "UsageFlywheel",
    "UsageTrace",
    "UsageTraceStore",
]
