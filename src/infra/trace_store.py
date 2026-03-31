"""Meta-Harness style filesystem-based trace store.

Logs every pipeline action as structured JSON/JSONL so that both
Codex (GPT-5.4) and Opus (Claude) can read historical traces for
causal reasoning over prior failures.

Reference: Meta-Harness (arxiv 2603.28052) — "the main advantage is
search with selective access to prior diagnostic experience."
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TraceStore:
    """Filesystem-based structured trace logger.

    Directory layout::

        data/traces/
        ├── loop1/gen_NNN/       mutations.jsonl, evals.jsonl, frontier.json
        ├── loop2/cycle_NNN/     buffer_snapshot.json, training_log.jsonl, ...
        ├── curriculum/          codex_gen_NNN.json, profiles/
        └── evals/               eval_YYYYMMDD_HHMMSS.json
    """

    def __init__(self, base_dir: str | Path = "data/traces") -> None:
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self._gen_counter = self._detect_counter("loop1")
        self._cycle_counter = self._detect_counter("loop2")
        self._curric_counter = self._detect_counter("curriculum")

    # ------------------------------------------------------------------
    # Loop 1: GEPA mutation traces
    # ------------------------------------------------------------------

    def log_mutation(
        self,
        gen: int,
        genome_id: str,
        original: dict[str, Any],
        mutated: dict[str, Any],
        scores: dict[str, Any],
        codex_review: dict[str, Any] | None = None,
    ) -> None:
        """Log a single mutation proposal + optional Codex review."""
        gen_dir = self.base / "loop1" / f"gen_{gen:04d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.time(),
            "genome_id": genome_id,
            "mutation_type": mutated.get("mutation_type", "unknown"),
            "original_prompt_len": len(str(original.get("system_prompt", ""))),
            "mutated_prompt_len": len(str(mutated.get("system_prompt", ""))),
            "scores": scores,
            "codex_review": codex_review,
        }
        self._append_jsonl(gen_dir / "mutations.jsonl", record)

    def log_eval(
        self,
        gen: int,
        genome_id: str,
        scores: dict[str, Any],
        failure_patterns: dict[str, int] | None = None,
    ) -> None:
        """Log a genome evaluation result."""
        gen_dir = self.base / "loop1" / f"gen_{gen:04d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.time(),
            "genome_id": genome_id,
            "scores": scores,
            "failure_patterns": failure_patterns or {},
        }
        self._append_jsonl(gen_dir / "evals.jsonl", record)

    def log_frontier(self, gen: int, frontier_summary: dict[str, Any]) -> None:
        """Snapshot the Pareto frontier at this generation."""
        gen_dir = self.base / "loop1" / f"gen_{gen:04d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(gen_dir / "frontier.json", {
            "ts": time.time(), "gen": gen, **frontier_summary,
        })

    # ------------------------------------------------------------------
    # Loop 2: Distillation traces
    # ------------------------------------------------------------------

    def log_training_cycle(
        self,
        cycle_id: int | str,
        buffer_stats: dict[str, Any],
        codex_review: dict[str, Any] | None = None,
        loss_curve: list[float] | None = None,
        export_result: dict[str, Any] | None = None,
    ) -> None:
        """Log a complete SFT training cycle."""
        cycle_dir = self.base / "loop2" / f"cycle_{cycle_id}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(cycle_dir / "buffer_snapshot.json", {
            "ts": time.time(), **buffer_stats,
        })
        if codex_review:
            self._write_json(cycle_dir / "codex_review.json", codex_review)
        if loss_curve:
            self._write_json(cycle_dir / "training_log.json", {
                "ts": time.time(), "loss_curve": loss_curve,
            })
        if export_result:
            self._write_json(cycle_dir / "export_log.json", export_result)

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def log_curriculum(
        self,
        source: str,
        tasks: list[dict[str, Any]],
        capability_profile: dict[str, Any],
    ) -> None:
        """Log a curriculum generation event."""
        curric_dir = self.base / "curriculum"
        curric_dir.mkdir(parents=True, exist_ok=True)
        self._curric_counter += 1
        self._write_json(
            curric_dir / f"{source}_gen_{self._curric_counter:04d}.json",
            {"ts": time.time(), "source": source, "n_tasks": len(tasks),
             "tasks": tasks, "capability_profile": capability_profile},
        )
        # Also snapshot the profile
        profiles_dir = curric_dir / "profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(
            profiles_dir / f"profile_{time.strftime('%Y%m%d_%H%M%S')}.json",
            {"ts": time.time(), **capability_profile},
        )

    # ------------------------------------------------------------------
    # Periodic evaluation
    # ------------------------------------------------------------------

    def log_periodic_eval(
        self,
        result: dict[str, Any],
        comparison: dict[str, Any] | None = None,
    ) -> None:
        """Log a periodic benchmark evaluation."""
        eval_dir = self.base / "evals"
        eval_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(
            eval_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json",
            {"ts": time.time(), "result": result, "comparison": comparison},
        )

    # ------------------------------------------------------------------
    # Retrieval (for Codex/Opus context injection)
    # ------------------------------------------------------------------

    def get_recent_traces(self, category: str, n: int = 5) -> list[dict[str, Any]]:
        """Read the last n trace entries from a category.

        category: "loop1", "loop2", "curriculum", "evals"
        """
        cat_dir = self.base / category
        if not cat_dir.exists():
            return []

        # Find most recent JSONL/JSON files
        files = sorted(cat_dir.rglob("*.json*"), key=lambda p: p.stat().st_mtime, reverse=True)
        results: list[dict[str, Any]] = []
        for f in files[:n * 2]:  # Read extra in case some are empty
            try:
                if f.suffix == ".jsonl":
                    lines = f.read_text(encoding="utf-8").strip().split("\n")
                    for line in lines[-n:]:
                        if line.strip():
                            results.append(json.loads(line))
                else:
                    results.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                continue
            if len(results) >= n:
                break
        return results[:n]

    def get_failure_summary(self, last_n_gens: int = 3) -> str:
        """Aggregate failure patterns from recent Loop 1 eval traces.

        Returns a compressed summary string suitable for injection
        into Codex/Opus prompts.
        """
        loop1_dir = self.base / "loop1"
        if not loop1_dir.exists():
            return "No Loop 1 traces available."

        gen_dirs = sorted(loop1_dir.iterdir(), reverse=True)[:last_n_gens]
        all_patterns: dict[str, int] = {}
        all_scores: list[dict[str, Any]] = []

        for gd in gen_dirs:
            evals_file = gd / "evals.jsonl"
            if evals_file.exists():
                for line in evals_file.read_text(encoding="utf-8").strip().split("\n"):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        for pat, count in record.get("failure_patterns", {}).items():
                            all_patterns[pat] = all_patterns.get(pat, 0) + count
                        all_scores.append(record.get("scores", {}))
                    except Exception:
                        continue

        if not all_patterns:
            return "No failure patterns recorded in recent generations."

        # Build compressed summary
        sorted_pats = sorted(all_patterns.items(), key=lambda x: x[1], reverse=True)
        avg_success = 0.0
        if all_scores:
            successes = [s.get("success_rate", 0) for s in all_scores if "success_rate" in s]
            avg_success = sum(successes) / len(successes) if successes else 0

        lines = [
            f"Failure summary (last {last_n_gens} generations, avg success={avg_success:.1%}):",
        ]
        for pat, count in sorted_pats[:8]:
            lines.append(f"  - {pat}: {count} occurrences")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def _detect_counter(self, category: str) -> int:
        cat_dir = self.base / category
        if not cat_dir.exists():
            return 0
        dirs = [d for d in cat_dir.iterdir() if d.is_dir()]
        if not dirs:
            return 0
        # Extract numbers from dir names like gen_0015 or cycle_003
        nums = []
        for d in dirs:
            parts = d.name.split("_")
            for p in parts:
                try:
                    nums.append(int(p))
                except ValueError:
                    continue
        return max(nums) if nums else 0
