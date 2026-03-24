"""Multi-objective Pareto frontier tracker for prompt genome evolution.

Maintains a set of non-dominated genomes across multiple objectives
and provides tournament selection for parent selection.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from src.models import PromptGenome
from src.loop1_gepa.prompt_genome import dict_to_genome, genome_to_dict

logger = logging.getLogger(__name__)

# Objectives where higher is better.
_MAXIMIZE_OBJECTIVES = frozenset({"success_rate", "tool_accuracy", "code_quality"})

# Objectives where lower is better.
_MINIMIZE_OBJECTIVES = frozenset({"avg_steps"})


class ParetoTracker:
    """Tracks a Pareto frontier of PromptGenomes across multiple objectives.

    Handles mixed objectives: some are maximized (success_rate, tool_accuracy,
    code_quality) and some are minimized (avg_steps). Internally, minimization
    objectives are negated so that all comparisons use "higher is better".

    Parameters
    ----------
    maximize_objectives:
        Set of objective names where higher values are better.
    minimize_objectives:
        Set of objective names where lower values are better.
    """

    def __init__(
        self,
        maximize_objectives: frozenset[str] | None = None,
        minimize_objectives: frozenset[str] | None = None,
    ) -> None:
        self._maximize = maximize_objectives or _MAXIMIZE_OBJECTIVES
        self._minimize = minimize_objectives or _MINIMIZE_OBJECTIVES
        self._all_objectives = self._maximize | self._minimize

        # Frontier storage: genome_id -> (genome, normalized_scores)
        self._frontier: dict[str, tuple[PromptGenome, dict[str, float]]] = {}

        # History of all genomes ever added (for analysis)
        self._history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize_scores(self, scores: dict[str, float]) -> dict[str, float]:
        """Normalize scores so that higher is always better.

        Minimization objectives are negated.
        """
        normalized: dict[str, float] = {}
        for obj in self._all_objectives:
            val = scores.get(obj, 0.0)
            if obj in self._minimize:
                normalized[obj] = -val
            else:
                normalized[obj] = val
        return normalized

    # ------------------------------------------------------------------
    # Dominance
    # ------------------------------------------------------------------

    @staticmethod
    def dominates(scores_a: dict[str, float], scores_b: dict[str, float]) -> bool:
        """Return True if scores_a dominates scores_b.

        A dominates B iff A is >= B on all objectives and strictly > on at least one.
        Both dicts must already be in normalized form (higher = better).
        """
        all_keys = set(scores_a) | set(scores_b)
        if not all_keys:
            return False

        at_least_as_good = True
        strictly_better = False

        for k in all_keys:
            a_val = scores_a.get(k, 0.0)
            b_val = scores_b.get(k, 0.0)
            if a_val < b_val:
                at_least_as_good = False
                break
            if a_val > b_val:
                strictly_better = True

        return at_least_as_good and strictly_better

    # ------------------------------------------------------------------
    # Frontier management
    # ------------------------------------------------------------------

    def add(self, genome: PromptGenome, scores: dict[str, float]) -> bool:
        """Add a genome to the tracker and update the Pareto frontier.

        Parameters
        ----------
        genome:
            The genome to consider.
        scores:
            Raw objective scores (before normalization).

        Returns
        -------
        bool
            True if the genome is on the Pareto frontier after insertion.
        """
        normalized = self._normalize_scores(scores)

        # Record history
        self._history.append({
            "genome_id": genome.genome_id,
            "generation": genome.generation,
            "raw_scores": dict(scores),
            "normalized_scores": dict(normalized),
        })

        # Check if this genome is dominated by any existing frontier member
        for fid, (_, f_scores) in list(self._frontier.items()):
            if self.dominates(f_scores, normalized):
                logger.debug(
                    "Genome %s dominated by frontier member %s",
                    genome.genome_id,
                    fid,
                )
                return False

        # This genome is not dominated; remove any frontier members it dominates
        dominated_ids = []
        for fid, (_, f_scores) in self._frontier.items():
            if self.dominates(normalized, f_scores):
                dominated_ids.append(fid)

        for fid in dominated_ids:
            logger.debug(
                "Genome %s dominates frontier member %s; removing",
                genome.genome_id,
                fid,
            )
            del self._frontier[fid]

        # Add to frontier
        genome.scores = dict(scores)
        self._frontier[genome.genome_id] = (genome, normalized)

        logger.info(
            "Genome %s added to Pareto frontier (size=%d). Removed %d dominated.",
            genome.genome_id,
            len(self._frontier),
            len(dominated_ids),
        )
        return True

    def get_frontier(self) -> list[PromptGenome]:
        """Return all non-dominated genomes on the current frontier."""
        return [genome for genome, _ in self._frontier.values()]

    @property
    def frontier_size(self) -> int:
        """Number of genomes on the current frontier."""
        return len(self._frontier)

    # ------------------------------------------------------------------
    # Parent selection
    # ------------------------------------------------------------------

    def select_parents(self, n: int, tournament_size: int = 3) -> list[PromptGenome]:
        """Select n parent genomes from the frontier via tournament selection.

        For each parent slot, ``tournament_size`` random frontier members
        are sampled, and the one with the best crowding distance (spread
        across objectives) is selected. This promotes diversity.

        Parameters
        ----------
        n:
            Number of parents to select.
        tournament_size:
            Number of candidates per tournament round.

        Returns
        -------
        list[PromptGenome]
            Selected parent genomes. May contain duplicates if frontier
            is smaller than n.
        """
        frontier_list = list(self._frontier.values())
        if not frontier_list:
            return []

        if len(frontier_list) <= n:
            return [g for g, _ in frontier_list]

        # Compute crowding distances for diversity pressure
        crowding = self._compute_crowding_distances(frontier_list)

        selected: list[PromptGenome] = []
        for _ in range(n):
            candidates = random.sample(
                range(len(frontier_list)),
                min(tournament_size, len(frontier_list)),
            )
            # Pick the candidate with the highest crowding distance
            best_idx = max(candidates, key=lambda i: crowding[i])
            selected.append(frontier_list[best_idx][0])

        return selected

    def _compute_crowding_distances(
        self,
        members: list[tuple[PromptGenome, dict[str, float]]],
    ) -> list[float]:
        """Compute NSGA-II style crowding distances for frontier members."""
        n = len(members)
        if n <= 2:
            return [float("inf")] * n

        distances = [0.0] * n
        objectives = list(self._all_objectives)

        for obj in objectives:
            # Sort by this objective
            indices = list(range(n))
            indices.sort(key=lambda i: members[i][1].get(obj, 0.0))

            # Boundary points get infinite distance
            distances[indices[0]] = float("inf")
            distances[indices[-1]] = float("inf")

            obj_min = members[indices[0]][1].get(obj, 0.0)
            obj_max = members[indices[-1]][1].get(obj, 0.0)
            obj_range = obj_max - obj_min

            if obj_range == 0:
                continue

            for i in range(1, n - 1):
                prev_val = members[indices[i - 1]][1].get(obj, 0.0)
                next_val = members[indices[i + 1]][1].get(obj, 0.0)
                distances[indices[i]] += (next_val - prev_val) / obj_range

        return distances

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the frontier and history to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": 1,
            "frontier": [
                {
                    "genome": genome_to_dict(genome),
                    "normalized_scores": normalized,
                }
                for genome, normalized in self._frontier.values()
            ],
            "history": self._history[-1000:],  # Keep last 1000 entries
            "maximize_objectives": sorted(self._maximize),
            "minimize_objectives": sorted(self._minimize),
        }

        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
        logger.info("Saved Pareto frontier (%d members) to %s", self.frontier_size, path)

    def load(self, path: str | Path) -> None:
        """Load frontier and history from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        self._frontier.clear()
        for entry in data.get("frontier", []):
            genome = dict_to_genome(entry["genome"])
            normalized = entry["normalized_scores"]
            self._frontier[genome.genome_id] = (genome, normalized)

        self._history = data.get("history", [])

        logger.info(
            "Loaded Pareto frontier (%d members) from %s",
            self.frontier_size,
            path,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a summary of the current frontier state."""
        if not self._frontier:
            return {
                "frontier_size": 0,
                "objective_ranges": {},
                "history_length": len(self._history),
            }

        objective_ranges: dict[str, dict[str, float]] = {}
        for obj in self._all_objectives:
            raw_values = [
                g.scores.get(obj, 0.0)
                for g, _ in self._frontier.values()
            ]
            objective_ranges[obj] = {
                "min": min(raw_values),
                "max": max(raw_values),
                "mean": sum(raw_values) / len(raw_values),
            }

        return {
            "frontier_size": self.frontier_size,
            "objective_ranges": objective_ranges,
            "history_length": len(self._history),
            "generations": sorted(
                set(g.generation for g, _ in self._frontier.values())
            ),
        }
