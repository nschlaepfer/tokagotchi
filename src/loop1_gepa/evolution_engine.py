"""Main GEPA evolutionary loop for prompt genome optimization.

Orchestrates the iterative cycle of parent selection, Opus-driven mutation,
genome evaluation, and Pareto frontier updates. Supports concurrent
evaluations and budget-aware rate limiting.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.arena.docker_manager import DockerManager
from src.config import Loop1Config, MasterConfig, load_config
from src.infra.vllm_server import VLLMServer
from src.models import EvalResult, PromptGenome, TaskSpec
from src.orchestrator.opus_client import OpusClient

from src.loop1_gepa.evaluator import evaluate_genome
from src.loop1_gepa.mutation_operators import MutationOperator, propose_mutation
from src.loop1_gepa.pareto_tracker import ParetoTracker
from src.loop1_gepa.prompt_genome import (
    create_seed_genome,
    crossover,
    load_population,
    save_population,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment history record
# ---------------------------------------------------------------------------


@dataclass
class IterationRecord:
    """Record of a single GEPA iteration for logging and analysis."""

    iteration: int
    generation: int
    timestamp: float = field(default_factory=time.time)
    parents_selected: int = 0
    mutations_proposed: int = 0
    mutations_succeeded: int = 0
    crossovers_performed: int = 0
    genomes_evaluated: int = 0
    frontier_additions: int = 0
    frontier_size: int = 0
    best_scores: dict[str, float] = field(default_factory=dict)
    mutation_types: list[str] = field(default_factory=list)
    eval_duration_seconds: float = 0.0
    opus_cost_usd: float = 0.0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class GEPAEngine:
    """The main GEPA evolutionary engine.

    Runs an iterative loop:
    1. Select parents from the Pareto frontier.
    2. Propose mutations via Opus (with failure trace analysis).
    3. Optionally perform crossover between frontier members.
    4. Evaluate mutated/crossed genomes against the task set.
    5. Update the Pareto frontier.
    6. Log results and persist state.

    Parameters
    ----------
    config:
        Master configuration (uses config.loop1 for GEPA-specific settings).
    opus_client:
        Client for Opus-driven mutation proposals.
    vllm_server:
        vLLM server for Qwen inference during evaluation.
    arena_manager:
        Docker manager for arena containers.
    tasks:
        Evaluation task set. If None, must be provided when calling run().
    data_dir:
        Directory for persisting population, frontier, and history.
    """

    def __init__(
        self,
        config: MasterConfig,
        opus_client: OpusClient,
        vllm_server: VLLMServer,
        arena_manager: Any,
        tasks: list[TaskSpec] | None = None,
        data_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.loop1_config: Loop1Config = config.loop1
        self.opus_client = opus_client
        self.vllm_server = vllm_server
        self.arena_manager = arena_manager
        self.tasks = tasks or []

        self.data_dir = Path(data_dir or config.data_dir) / "loop1_gepa"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.pareto_tracker = ParetoTracker()
        self.population: list[PromptGenome] = []
        self.generation: int = 0
        self.history: list[IterationRecord] = []

        # Rate limiting: track Opus calls per hour
        self._opus_call_timestamps: list[float] = []

        # Paths
        self._population_path = self.data_dir / "population.json"
        self._frontier_path = self.data_dir / "frontier.json"
        self._history_path = self.data_dir / "history.json"
        self._mutation_log_path = self.data_dir / "mutation_log.jsonl"

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize the population, either from disk or by seeding."""
        # Try loading existing state
        if self._population_path.exists():
            logger.info("Loading existing population from %s", self._population_path)
            self.population = load_population(self._population_path)
            if self.population:
                self.generation = max(g.generation for g in self.population)
        if self._frontier_path.exists():
            logger.info("Loading existing frontier from %s", self._frontier_path)
            self.pareto_tracker.load(self._frontier_path)

        # Seed population if empty
        if not self.population:
            logger.info(
                "Seeding initial population of %d genomes",
                self.loop1_config.population_size,
            )
            self.population = [
                create_seed_genome()
                for _ in range(self.loop1_config.population_size)
            ]

            # Evaluate seed population
            if self.tasks:
                await self._evaluate_and_update(self.population)

        logger.info(
            "GEPA initialized: population=%d, frontier=%d, generation=%d",
            len(self.population),
            self.pareto_tracker.frontier_size,
            self.generation,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, n_iterations: int, tasks: list[TaskSpec] | None = None) -> None:
        """Run the GEPA evolutionary loop for n_iterations.

        Parameters
        ----------
        n_iterations:
            Number of evolution iterations to run.
        tasks:
            Optional override for the evaluation task set.
        """
        if tasks is not None:
            self.tasks = tasks

        if not self.tasks:
            raise ValueError("No evaluation tasks provided. Supply tasks to run().")

        # Initialize if not already done
        if not self.population:
            await self.initialize()

        logger.info("Starting GEPA evolution for %d iterations", n_iterations)

        for iteration in range(n_iterations):
            iter_start = time.monotonic()

            record = IterationRecord(
                iteration=iteration,
                generation=self.generation,
            )

            try:
                # --- Step 1: Select parents from Pareto frontier ---
                n_parents = max(
                    2,
                    self.loop1_config.population_size // 4,
                )
                parents = self._select_parents(n_parents)
                record.parents_selected = len(parents)

                if not parents:
                    # No frontier yet; use random population members
                    parents = random.sample(
                        self.population,
                        min(n_parents, len(self.population)),
                    )

                # --- Step 2: Propose mutations via Opus ---
                offspring = await self._propose_mutations(parents, record)

                # --- Step 3: Crossover (optional) ---
                crossover_offspring = await self._perform_crossovers(record)
                offspring.extend(crossover_offspring)

                if not offspring:
                    logger.warning(
                        "Iteration %d: no offspring produced, skipping evaluation",
                        iteration,
                    )
                    continue

                # --- Step 4: Evaluate offspring ---
                eval_start = time.monotonic()
                await self._evaluate_and_update(offspring)
                record.eval_duration_seconds = time.monotonic() - eval_start
                record.genomes_evaluated = len(offspring)

                # --- Step 5: Update population ---
                self.generation += 1
                record.generation = self.generation
                self._update_population(offspring)

                # --- Step 6: Record results ---
                record.frontier_size = self.pareto_tracker.frontier_size
                record.best_scores = self._get_best_scores()
                record.timestamp = time.time()

                self.history.append(record)

                # Persist state
                self._save_state()

                elapsed = time.monotonic() - iter_start
                logger.info(
                    "Iteration %d/%d complete (%.1fs): "
                    "gen=%d frontier=%d mutations=%d/%d best_success=%.2f",
                    iteration + 1,
                    n_iterations,
                    elapsed,
                    self.generation,
                    record.frontier_size,
                    record.mutations_succeeded,
                    record.mutations_proposed,
                    record.best_scores.get("success_rate", 0.0),
                )

            except Exception:
                logger.exception("Error in GEPA iteration %d", iteration)
                # Continue with the next iteration
                continue

        logger.info(
            "GEPA evolution complete: %d iterations, final frontier size=%d",
            n_iterations,
            self.pareto_tracker.frontier_size,
        )

    # ------------------------------------------------------------------
    # Parent selection
    # ------------------------------------------------------------------

    def _select_parents(self, n: int) -> list[PromptGenome]:
        """Select parent genomes from the Pareto frontier."""
        frontier_parents = self.pareto_tracker.select_parents(n)
        if frontier_parents:
            return frontier_parents

        # Fallback: select from elite population members by score
        scored = [g for g in self.population if g.scores]
        if not scored:
            return []

        scored.sort(
            key=lambda g: g.scores.get("success_rate", 0.0),
            reverse=True,
        )
        return scored[: min(n, len(scored))]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    async def _propose_mutations(
        self,
        parents: list[PromptGenome],
        record: IterationRecord,
    ) -> list[PromptGenome]:
        """Propose mutations for each parent genome via Opus.

        Uses concurrent calls with rate limiting.
        """
        offspring: list[PromptGenome] = []
        record.mutations_proposed = len(parents)

        # Rate limit: ensure we don't exceed experiments_per_hour
        await self._rate_limit()

        # Run mutation proposals concurrently with a semaphore
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent Opus calls

        async def _mutate_one(parent: PromptGenome) -> tuple[MutationOperator, PromptGenome] | None:
            async with semaphore:
                await self._rate_limit()

                # Get the parent's eval result if available, or create a minimal one
                eval_result = self._get_latest_eval(parent)
                if eval_result is None:
                    return None

                try:
                    mutation_type, mutated = await propose_mutation(
                        self.opus_client,
                        parent,
                        eval_result,
                    )
                    self._opus_call_timestamps.append(time.monotonic())
                    return mutation_type, mutated
                except Exception as exc:
                    logger.error(
                        "Mutation proposal failed for %s: %s",
                        parent.genome_id,
                        exc,
                    )
                    return None

        results = await asyncio.gather(
            *[_mutate_one(p) for p in parents],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                logger.error("Mutation task raised: %s", result)
                continue
            if result is None:
                continue
            mutation_type, mutated = result
            offspring.append(mutated)
            record.mutation_types.append(mutation_type.value)
            record.mutations_succeeded += 1

            # Append to persistent mutation log (JSONL)
            self._log_mutation(mutated, mutation_type)

        return offspring

    def _log_mutation(
        self, genome: PromptGenome, mutation_type: MutationOperator
    ) -> None:
        """Append a mutation event to the JSONL mutation log."""
        import json as _json

        entry = {
            "timestamp": genome.created_at or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "genome_id": genome.genome_id,
            "parent_ids": genome.parent_ids,
            "generation": genome.generation,
            "mutation_type": mutation_type.value,
            "diagnosis": genome.mutation_diagnosis,
            "rationale": genome.mutation_rationale,
            "parent_scores": {},
        }

        # Include parent's scores if available
        if genome.parent_ids:
            for p in self.population:
                if p.genome_id == genome.parent_ids[0]:
                    entry["parent_scores"] = p.scores
                    break

        try:
            with open(self._mutation_log_path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(entry) + "\n")
        except Exception:
            logger.debug("Failed to write mutation log entry", exc_info=True)

    def _get_latest_eval(self, genome: PromptGenome) -> EvalResult | None:
        """Retrieve or construct an EvalResult for a genome.

        Looks through history to find the most recent evaluation, or constructs
        a minimal EvalResult from the genome's scores if available.
        """
        if genome.scores:
            return EvalResult(
                genome_id=genome.genome_id,
                tasks_run=self.loop1_config.eval_tasks_per_genome,
                success_rate=genome.scores.get("success_rate", 0.0),
                avg_steps=genome.scores.get("avg_steps", 0.0),
                tool_accuracy=genome.scores.get("tool_accuracy", 0.0),
                code_quality=genome.scores.get("code_quality", 0.0),
                failure_patterns={"unknown": 1},
            )
        return EvalResult(
            genome_id=genome.genome_id,
            tasks_run=0,
            failure_patterns={"not_yet_evaluated": 1},
        )

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    async def _perform_crossovers(
        self,
        record: IterationRecord,
    ) -> list[PromptGenome]:
        """Perform crossover between frontier members based on crossover_rate."""
        offspring: list[PromptGenome] = []

        if random.random() > self.loop1_config.crossover_rate:
            return offspring

        frontier = self.pareto_tracker.get_frontier()
        if len(frontier) < 2:
            return offspring

        # Pick two random frontier members for crossover
        n_crossovers = max(1, int(len(frontier) * self.loop1_config.crossover_rate))
        for _ in range(n_crossovers):
            parent1, parent2 = random.sample(frontier, 2)
            child = crossover(parent1, parent2)
            offspring.append(child)
            record.crossovers_performed += 1

        logger.info("Produced %d crossover offspring", len(offspring))
        return offspring

    # ------------------------------------------------------------------
    # Evaluation and frontier update
    # ------------------------------------------------------------------

    async def _evaluate_and_update(self, genomes: list[PromptGenome]) -> None:
        """Evaluate genomes and update the Pareto frontier."""
        # Select a task subset for evaluation
        eval_tasks = self._select_eval_tasks()

        for genome in genomes:
            try:
                eval_result = await evaluate_genome(
                    genome,
                    eval_tasks,
                    self.vllm_server,
                    self.arena_manager,
                    max_concurrent=4,
                )

                # Update genome scores
                genome.scores = eval_result.score_vector

                # Update Pareto frontier
                self.pareto_tracker.add(genome, eval_result.score_vector)

            except Exception as exc:
                logger.error(
                    "Evaluation failed for genome %s: %s",
                    genome.genome_id,
                    exc,
                )

    def _select_eval_tasks(self) -> list[TaskSpec]:
        """Select a subset of tasks for evaluation."""
        n = self.loop1_config.eval_tasks_per_genome
        if len(self.tasks) <= n:
            return list(self.tasks)
        return random.sample(self.tasks, n)

    # ------------------------------------------------------------------
    # Population management
    # ------------------------------------------------------------------

    def _update_population(self, offspring: list[PromptGenome]) -> None:
        """Update the population with new offspring.

        Uses elitism: the top elite_size members are always retained.
        Remaining slots are filled by offspring and then randomly from
        the existing population.
        """
        max_size = self.loop1_config.population_size
        elite_size = self.loop1_config.elite_size

        # Sort current population by success_rate
        scored_pop = sorted(
            self.population,
            key=lambda g: g.scores.get("success_rate", 0.0),
            reverse=True,
        )

        # Keep elites
        elites = scored_pop[:elite_size]
        elite_ids = {g.genome_id for g in elites}

        # Build new population: elites + offspring + remaining
        new_population = list(elites)

        for g in offspring:
            if g.genome_id not in elite_ids and len(new_population) < max_size:
                new_population.append(g)

        # Fill remaining slots from existing population
        remaining = [g for g in scored_pop if g.genome_id not in elite_ids]
        random.shuffle(remaining)
        for g in remaining:
            if len(new_population) >= max_size:
                break
            if g.genome_id not in {ng.genome_id for ng in new_population}:
                new_population.append(g)

        self.population = new_population[:max_size]

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    async def _rate_limit(self) -> None:
        """Enforce Opus call rate limits (experiments_per_hour).

        Sleeps if the current rate would exceed the configured limit.
        """
        now = time.monotonic()
        one_hour_ago = now - 3600

        # Prune old timestamps
        self._opus_call_timestamps = [
            t for t in self._opus_call_timestamps if t > one_hour_ago
        ]

        max_per_hour = self.loop1_config.experiments_per_hour
        if len(self._opus_call_timestamps) >= max_per_hour:
            oldest = self._opus_call_timestamps[0]
            wait_time = 3600 - (now - oldest) + 1.0
            if wait_time > 0:
                logger.info(
                    "Rate limit reached (%d/%d per hour). Waiting %.1fs",
                    len(self._opus_call_timestamps),
                    max_per_hour,
                    wait_time,
                )
                await asyncio.sleep(wait_time)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Persist population, frontier, and history to disk."""
        try:
            save_population(self.population, self._population_path)
            self.pareto_tracker.save(self._frontier_path)
            self._save_history()
        except Exception:
            logger.exception("Failed to save GEPA state")

    def _save_history(self) -> None:
        """Save iteration history to JSON."""
        import json
        from dataclasses import asdict

        payload = {
            "version": 1,
            "generation": self.generation,
            "records": [asdict(r) for r in self.history[-500:]],
        }
        tmp = self._history_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._history_path)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _get_best_scores(self) -> dict[str, float]:
        """Get the best score for each objective across the frontier."""
        frontier = self.pareto_tracker.get_frontier()
        if not frontier:
            return {}

        best: dict[str, float] = {}
        maximize = {"success_rate", "tool_accuracy", "code_quality"}
        minimize = {"avg_steps"}

        for obj in maximize | minimize:
            values = [g.scores.get(obj, 0.0) for g in frontier if obj in g.scores]
            if values:
                if obj in maximize:
                    best[obj] = max(values)
                else:
                    best[obj] = min(values)

        return best

    def get_status(self) -> dict[str, Any]:
        """Return a status summary of the engine."""
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "frontier_size": self.pareto_tracker.frontier_size,
            "total_iterations": len(self.history),
            "best_scores": self._get_best_scores(),
            "pareto_summary": self.pareto_tracker.summary(),
            "tasks_available": len(self.tasks),
        }
