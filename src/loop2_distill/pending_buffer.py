"""Accumulates corrected traces for SFT in a persistent buffer.

Tracks training examples with metadata, enforces diversity constraints,
and persists to disk via JSONL for crash recovery.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from src.models import (
    Trajectory,
    TraceAnalysis,
    TaskSpec,
    StepRecord,
    ActionType,
    PromptGenome,
)
from src.config import Loop2Config

logger = logging.getLogger(__name__)

_DEFAULT_PERSIST_PATH = Path("data/pending.jsonl")


class PendingBuffer:
    """Thread-safe buffer that accumulates corrected traces for SFT training.

    Tracks metadata distributions (task types, failure modes, difficulty)
    and enforces diversity constraints before declaring the buffer ready
    for a training run.

    Parameters
    ----------
    config:
        Loop2 configuration with min_buffer_size, max_buffer_size, and
        diversity thresholds.
    persist_path:
        Path to the JSONL file for persistence. Defaults to
        ``data/pending.jsonl``.
    """

    def __init__(
        self,
        config: Loop2Config,
        persist_path: str | Path = _DEFAULT_PERSIST_PATH,
    ) -> None:
        self._config = config
        self._persist_path = Path(persist_path)

        self._examples: list[dict[str, Any]] = []
        self._metadata: list[dict[str, Any]] = []
        self._lock = threading.Lock()

        # Distribution counters
        self._task_type_counts: Counter[str] = Counter()
        self._failure_mode_counts: Counter[str] = Counter()
        self._difficulty_counts: Counter[str] = Counter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, example: dict[str, Any], metadata: dict[str, Any]) -> None:
        """Add a training example to the buffer.

        Parameters
        ----------
        example:
            A chat-format training example with ``messages`` key.
        metadata:
            Metadata dict with keys like ``task_type``, ``failure_mode``,
            ``difficulty``, ``trajectory_id``, etc.
        """
        with self._lock:
            self._examples.append(example)
            self._metadata.append(metadata)

            # Update distribution counters
            task_type = metadata.get("task_type", "unknown")
            failure_mode = metadata.get("failure_mode", "unknown")
            difficulty = metadata.get("difficulty", "medium")

            # Bucket difficulty into low/medium/high
            if isinstance(difficulty, (int, float)):
                if difficulty < 0.33:
                    difficulty = "low"
                elif difficulty < 0.66:
                    difficulty = "medium"
                else:
                    difficulty = "high"

            self._task_type_counts[task_type] += 1
            self._failure_mode_counts[failure_mode] += 1
            self._difficulty_counts[str(difficulty)] += 1

            logger.debug(
                "Buffer: added example (size=%d, task_type=%s, failure_mode=%s)",
                len(self._examples),
                task_type,
                failure_mode,
            )

    def size(self) -> int:
        """Return the current number of examples in the buffer."""
        with self._lock:
            return len(self._examples)

    def sample(self, n: int) -> list[dict[str, Any]]:
        """Return a random sample of n examples with metadata, without modifying the buffer."""
        import random
        with self._lock:
            if not self._examples:
                return []
            indices = random.sample(range(len(self._examples)), min(n, len(self._examples)))
            return [
                {"example": self._examples[i], "metadata": self._metadata[i], "index": i}
                for i in indices
            ]

    def is_ready(self) -> bool:
        """Check whether the buffer is ready for a training run.

        The buffer is ready when it meets both size and diversity thresholds.

        Returns
        -------
        bool
            True if size >= min_buffer_size AND diversity checks pass.
        """
        with self._lock:
            if len(self._examples) < self._config.min_buffer_size:
                return False

            diverse, _ = self._diversity_check_unlocked()
            return diverse

    def diversity_check(self) -> tuple[bool, dict[str, Any]]:
        """Check if the buffer has sufficient diversity.

        Returns
        -------
        tuple[bool, dict]
            (passes, details) where details contains the distribution
            information and which checks passed/failed.
        """
        with self._lock:
            return self._diversity_check_unlocked()

    def get_training_batch(self) -> list[dict[str, Any]]:
        """Return all examples and clear the buffer.

        This drains the buffer entirely, returning all accumulated
        examples for training. The buffer is reset afterward.

        Returns
        -------
        list[dict]
            All training examples that were in the buffer.
        """
        with self._lock:
            batch = list(self._examples)
            self._examples.clear()
            self._metadata.clear()
            self._task_type_counts.clear()
            self._failure_mode_counts.clear()
            self._difficulty_counts.clear()

            logger.info("Drained buffer: %d examples for training", len(batch))
            return batch

    def save(self) -> None:
        """Persist the buffer to disk as JSONL.

        Each line is a JSON object containing both the training example
        and its metadata.
        """
        with self._lock:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self._persist_path, "w", encoding="utf-8") as f:
                for example, meta in zip(self._examples, self._metadata):
                    record = {"example": example, "metadata": meta}
                    f.write(json.dumps(record, default=str) + "\n")

            logger.info(
                "Saved %d examples to %s",
                len(self._examples),
                self._persist_path,
            )

    def load(self) -> None:
        """Load the buffer from disk, replacing current contents.

        Reads from the JSONL file at ``persist_path`` and rebuilds
        all internal state including distribution counters.
        """
        if not self._persist_path.exists():
            logger.info("No pending buffer file found at %s", self._persist_path)
            return

        with self._lock:
            self._examples.clear()
            self._metadata.clear()
            self._task_type_counts.clear()
            self._failure_mode_counts.clear()
            self._difficulty_counts.clear()

        # Read outside lock, then add via public API
        records: list[dict[str, Any]] = []
        with open(self._persist_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed line %d in %s",
                        line_num,
                        self._persist_path,
                    )

        for record in records:
            self.add(
                example=record.get("example", {}),
                metadata=record.get("metadata", {}),
            )

        logger.info(
            "Loaded %d examples from %s",
            self.size(),
            self._persist_path,
        )

    def get_stats(self) -> dict[str, Any]:
        """Return buffer statistics including size and distributions.

        Returns
        -------
        dict
            Statistics dictionary with size, distributions, and readiness.
        """
        with self._lock:
            diverse, diversity_details = self._diversity_check_unlocked()
            return {
                "size": len(self._examples),
                "min_threshold": self._config.min_buffer_size,
                "max_threshold": self._config.max_buffer_size,
                "is_ready": (
                    len(self._examples) >= self._config.min_buffer_size
                    and diverse
                ),
                "task_type_distribution": dict(self._task_type_counts),
                "failure_mode_distribution": dict(self._failure_mode_counts),
                "difficulty_distribution": dict(self._difficulty_counts),
                "diversity": diversity_details,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _diversity_check_unlocked(self) -> tuple[bool, dict[str, Any]]:
        """Check diversity without acquiring the lock (caller must hold it).

        Returns
        -------
        tuple[bool, dict]
            (passes, details) with specifics on each constraint.
        """
        num_task_types = len(self._task_type_counts)
        num_failure_modes = len(self._failure_mode_counts)

        min_types = self._config.diversity_min_task_types
        min_failures = self._config.diversity_min_failure_modes

        task_type_ok = num_task_types >= min_types
        failure_mode_ok = num_failure_modes >= min_failures

        details = {
            "task_types": {
                "count": num_task_types,
                "required": min_types,
                "pass": task_type_ok,
                "values": dict(self._task_type_counts),
            },
            "failure_modes": {
                "count": num_failure_modes,
                "required": min_failures,
                "pass": failure_mode_ok,
                "values": dict(self._failure_mode_counts),
            },
        }

        return (task_type_ok and failure_mode_ok), details
