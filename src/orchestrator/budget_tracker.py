"""Budget tracker with circuit breakers for Opus API spend.

Tracks cumulative spend per hour, per day, and per loop iteration.
Thread-safe and persists state to disk as JSON.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BudgetExhaustedError(Exception):
    """Raised when a budget circuit breaker trips."""

    def __init__(self, limit_type: str, limit_usd: float, current_usd: float) -> None:
        self.limit_type = limit_type
        self.limit_usd = limit_usd
        self.current_usd = current_usd
        super().__init__(
            f"Budget exhausted: {limit_type} limit ${limit_usd:.2f} "
            f"reached (current: ${current_usd:.2f})"
        )


@dataclass
class SpendRecord:
    """A single API call cost record."""

    timestamp: float
    cost_usd: float
    loop_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class BudgetTracker:
    """Thread-safe budget tracker with hourly/daily circuit breakers.

    Parameters
    ----------
    hourly_limit_usd:
        Maximum allowed spend in any rolling 1-hour window.
    daily_limit_usd:
        Maximum allowed spend in any rolling 24-hour window.
    persist_path:
        File path to save/load budget state. ``None`` disables persistence.
    """

    def __init__(
        self,
        hourly_limit_usd: float = 10.0,
        daily_limit_usd: float = 50.0,
        persist_path: str | Path | None = None,
    ) -> None:
        self.hourly_limit_usd = hourly_limit_usd
        self.daily_limit_usd = daily_limit_usd
        self.persist_path = Path(persist_path) if persist_path else None
        self._records: list[SpendRecord] = []
        self._lock = threading.Lock()

        if self.persist_path and self.persist_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_spend(self, amount_usd: float, loop_id: str = "") -> bool:
        """Check whether spending *amount_usd* would stay within limits.

        Returns ``True`` if the spend is allowed, ``False`` otherwise.
        Does **not** record the spend.
        """
        with self._lock:
            now = time.time()
            hourly = self._sum_window(now, 3600) + amount_usd
            daily = self._sum_window(now, 86400) + amount_usd
            return hourly <= self.hourly_limit_usd and daily <= self.daily_limit_usd

    def record_spend(
        self,
        amount_usd: float,
        loop_id: str,
        metadata: dict[str, Any] | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record an API call's cost and enforce circuit breakers.

        Raises
        ------
        BudgetExhaustedError
            If the new spend pushes the hourly or daily total over the limit.
        """
        with self._lock:
            now = time.time()

            # Check hourly limit
            hourly_total = self._sum_window(now, 3600) + amount_usd
            if hourly_total > self.hourly_limit_usd:
                raise BudgetExhaustedError("hourly", self.hourly_limit_usd, hourly_total)

            # Check daily limit
            daily_total = self._sum_window(now, 86400) + amount_usd
            if daily_total > self.daily_limit_usd:
                raise BudgetExhaustedError("daily", self.daily_limit_usd, daily_total)

            record = SpendRecord(
                timestamp=now,
                cost_usd=amount_usd,
                loop_id=loop_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                metadata=metadata or {},
            )
            self._records.append(record)
            logger.debug(
                "Recorded spend $%.4f for loop=%s (hourly=$%.2f, daily=$%.2f)",
                amount_usd,
                loop_id,
                hourly_total,
                daily_total,
            )
            self._save()

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of spend broken down by loop, hour, and day.

        Returns a dict with keys:
        - ``total_usd``: lifetime total
        - ``hourly_usd``: spend in the last rolling hour
        - ``daily_usd``: spend in the last rolling 24 hours
        - ``by_loop``: ``{loop_id: total_usd}``
        - ``by_hour``: ``{iso_hour_str: total_usd}`` for hours with spend
        - ``by_day``: ``{iso_date_str: total_usd}`` for days with spend
        - ``num_calls``: total number of recorded API calls
        - ``total_prompt_tokens``: sum of prompt tokens
        - ``total_completion_tokens``: sum of completion tokens
        """
        with self._lock:
            now = time.time()
            by_loop: dict[str, float] = defaultdict(float)
            by_hour: dict[str, float] = defaultdict(float)
            by_day: dict[str, float] = defaultdict(float)
            total_prompt = 0
            total_completion = 0

            for r in self._records:
                by_loop[r.loop_id] += r.cost_usd
                dt = datetime.fromtimestamp(r.timestamp, tz=timezone.utc)
                by_hour[dt.strftime("%Y-%m-%dT%H")] += r.cost_usd
                by_day[dt.strftime("%Y-%m-%d")] += r.cost_usd
                total_prompt += r.prompt_tokens
                total_completion += r.completion_tokens

            return {
                "total_usd": sum(r.cost_usd for r in self._records),
                "hourly_usd": self._sum_window(now, 3600),
                "daily_usd": self._sum_window(now, 86400),
                "by_loop": dict(by_loop),
                "by_hour": dict(by_hour),
                "by_day": dict(by_day),
                "num_calls": len(self._records),
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Persist current state to disk (called under lock)."""
        if not self.persist_path:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "hourly_limit_usd": self.hourly_limit_usd,
                "daily_limit_usd": self.daily_limit_usd,
                "records": [asdict(r) for r in self._records],
            }
            tmp = self.persist_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self.persist_path)
        except OSError:
            logger.exception("Failed to save budget state to %s", self.persist_path)

    def _load(self) -> None:
        """Load state from disk (called in __init__, no lock needed)."""
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            payload = json.loads(self.persist_path.read_text())
            for raw in payload.get("records", []):
                self._records.append(SpendRecord(
                    timestamp=raw["timestamp"],
                    cost_usd=raw["cost_usd"],
                    loop_id=raw["loop_id"],
                    prompt_tokens=raw.get("prompt_tokens", 0),
                    completion_tokens=raw.get("completion_tokens", 0),
                    metadata=raw.get("metadata", {}),
                ))
            logger.info(
                "Loaded %d spend records from %s", len(self._records), self.persist_path
            )
        except (OSError, json.JSONDecodeError, KeyError):
            logger.exception("Failed to load budget state from %s", self.persist_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sum_window(self, now: float, window_seconds: float) -> float:
        """Sum cost_usd for records within *window_seconds* of *now*.

        Must be called under ``self._lock``.
        """
        cutoff = now - window_seconds
        return sum(r.cost_usd for r in self._records if r.timestamp >= cutoff)
