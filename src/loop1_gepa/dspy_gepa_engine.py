"""DSPy GEPA engine — wraps real DSPy GEPA optimizer in the GEPAEngine interface.

Drop-in replacement for the custom GEPAEngine. Uses DSPy's built-in GEPA
optimizer with trace-aware reflection, proper Pareto selection, and the
ability to use different LLMs for the student (Qwen) and reflector (Opus).

Usage in master_loop.py::

    if cfg.loop1.use_dspy_gepa:
        from src.loop1_gepa.dspy_gepa_engine import DspyGEPAEngine
        engine = DspyGEPAEngine(config=cfg, ...)
    else:
        engine = GEPAEngine(config=cfg, ...)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import dspy

from src.config import MasterConfig
from src.loop1_gepa.dspy_metric import arena_metric, set_arena_manager, task_to_example
from src.loop1_gepa.dspy_program import AgentProgram
from src.loop1_gepa.dspy_tools import build_dspy_tools
from src.models import TaskSpec

logger = logging.getLogger(__name__)


class DspyGEPAEngine:
    """Wraps DSPy's GEPA optimizer in the GEPAEngine interface.

    Same constructor signature and public API as ``GEPAEngine`` so
    ``master_loop.py`` can use either engine transparently.

    Parameters
    ----------
    config : MasterConfig
        Full merged config.
    opus_client : Any
        OpusClient instance (used for budget tracking, not direct calls).
    vllm_server : Any
        LLMServer instance (for model name and endpoint info).
    arena_manager : Any
        DockerManager or SubprocessManager for sandbox execution.
    tasks : list[TaskSpec]
        Seed tasks for optimization.
    data_dir : Path or str
        Directory for saving state and logs.
    """

    def __init__(
        self,
        config: MasterConfig,
        opus_client: Any,
        vllm_server: Any,
        arena_manager: Any,
        tasks: list[TaskSpec],
        data_dir: Path | str,
    ) -> None:
        self.config = config
        self.opus_client = opus_client
        self.vllm_server = vllm_server
        self.arena_manager = arena_manager
        self.tasks = tasks
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # DSPy state
        self._program: AgentProgram | None = None
        self._student_lm: dspy.LM | None = None
        self._reflection_lm: dspy.LM | None = None
        self._trainset: list[dspy.Example] = []

        # Tracking
        self._generation: int = 0
        self._total_iterations: int = 0
        self._best_score: float = 0.0
        self._last_run_stats: dict[str, Any] = {}
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API (matches GEPAEngine)
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Set up DSPy LMs, build the program, and load any saved state."""
        self._start_time = time.time()

        # Configure student LM against the local serving backend.
        model_name = self.config.model.name
        provider = self.config.model.normalized_provider
        if provider == "ollama":
            self._student_lm = dspy.LM(
                model=f"ollama_chat/{model_name}",
                api_base=f"http://{self.config.model.ollama_host}:{self.config.model.ollama_port}",
                api_key="",
                temperature=0.7,
                max_tokens=2048,
            )
        else:
            self._student_lm = dspy.LM(
                model=f"openai/{model_name}",
                api_base=self.vllm_server.base_url,
                api_key=self.config.model.resolved_api_key,
                model_type="chat",
                temperature=0.7,
                max_tokens=2048,
            )

        # Configure reflection LM (Claude Opus via CLI)
        # Uses the Claude CLI's built-in auth — no ANTHROPIC_API_KEY needed.
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            # Direct API access (faster, preferred)
            self._reflection_lm = dspy.LM(
                model=f"anthropic/{self.config.opus.model}",
                api_key=api_key,
                temperature=1.0,
                max_tokens=4096,
            )
            logger.info("Reflection LM: direct Anthropic API")
        else:
            # Fall back to Claude CLI wrapper (uses CLI auth)
            from src.loop1_gepa.claude_cli_lm import ClaudeCliLM
            self._reflection_lm = ClaudeCliLM(model=self.config.opus.model)
            logger.info("Reflection LM: Claude CLI (no API key, using CLI auth)")

        # Set student as default
        dspy.configure(lm=self._student_lm)

        # Set arena manager for metric function
        set_arena_manager(self.arena_manager)

        # Build trainset from TaskSpec list
        self._trainset = []
        for task in self.tasks:
            example = task_to_example(task)
            # Stash TaskSpec in a private attribute for the metric
            example._task_spec = task
            self._trainset.append(example)

        logger.info(
            "DSPy trainset: %d examples from %d tasks",
            len(self._trainset), len(self.tasks),
        )

        # Build the program
        tools = build_dspy_tools(self.arena_manager)
        self._program = AgentProgram(tools=tools, max_iters=15)

        # Load saved state if available
        state_path = self.data_dir / "dspy_optimized.json"
        if state_path.exists():
            try:
                self._program.load(str(state_path))
                logger.info("Loaded saved DSPy program from %s", state_path)
            except Exception as e:
                logger.warning("Failed to load saved state: %s", e)

        logger.info(
            "DspyGEPAEngine initialized: provider=%s student=%s, reflector=%s, tasks=%d",
            provider,
            model_name,
            self.config.opus.model,
            len(self._trainset),
        )

    async def run(self, n_iterations: int = 10, tasks: list[TaskSpec] | None = None) -> None:
        """Run DSPy GEPA optimization.

        Parameters
        ----------
        n_iterations : int
            Controls optimization budget via ``max_metric_calls``.
        tasks : list[TaskSpec], optional
            Override tasks for this run.
        """
        if self._program is None:
            await self.initialize()

        if tasks:
            self._trainset = []
            for task in tasks:
                example = task_to_example(task)
                example._task_spec = task
                self._trainset.append(example)

        # Configure budget
        max_calls = getattr(
            self.config.loop1, "dspy_max_metric_calls",
            n_iterations * 10,
        )
        num_threads = getattr(
            self.config.loop1, "dspy_num_threads", 4
        )

        logger.info(
            "Starting DSPy GEPA optimization: max_calls=%d, threads=%d, trainset=%d",
            max_calls, num_threads, len(self._trainset),
        )

        # Create the optimizer
        optimizer = dspy.GEPA(
            metric=arena_metric,
            max_metric_calls=max_calls,
            reflection_lm=self._reflection_lm,
            candidate_selection_strategy="pareto",
            num_threads=num_threads,
            track_stats=True,
            log_dir=str(self.data_dir / "dspy_logs"),
        )

        # Run GEPA in a thread (it's synchronous internally)
        try:
            optimized = await asyncio.to_thread(
                optimizer.compile,
                student=self._program,
                trainset=self._trainset,
            )
        except Exception as e:
            logger.error("DSPy GEPA optimization failed: %s", e, exc_info=True)
            self._last_run_stats = {"status": "error", "error": str(e)}
            return

        # Update state
        self._program = optimized
        self._generation += 1
        self._total_iterations += n_iterations

        # Extract stats
        self._last_run_stats = self._extract_stats(optimizer)

        # Save optimized program
        try:
            save_path = self.data_dir / "dspy_optimized.json"
            optimized.save(str(save_path))
            logger.info("Saved optimized program to %s", save_path)
        except Exception as e:
            logger.warning("Failed to save optimized program: %s", e)

        # Save run summary
        self._save_run_summary()

        logger.info(
            "DSPy GEPA run complete: generation=%d, best_score=%.4f",
            self._generation,
            self._best_score,
        )

    def get_status(self) -> dict[str, Any]:
        """Return a snapshot of the engine status."""
        return {
            "engine": "dspy_gepa",
            "generation": self._generation,
            "total_iterations": self._total_iterations,
            "best_score": self._best_score,
            "trainset_size": len(self._trainset),
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            "last_run": self._last_run_stats,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_stats(self, optimizer: dspy.GEPA) -> dict[str, Any]:
        """Extract statistics from a completed GEPA optimizer run."""
        stats: dict[str, Any] = {"status": "completed"}

        try:
            # DSPy GEPA exposes stats via track_stats
            if hasattr(optimizer, "best_score"):
                self._best_score = max(self._best_score, optimizer.best_score)
                stats["best_score"] = optimizer.best_score

            if hasattr(optimizer, "trial_logs"):
                stats["num_trials"] = len(optimizer.trial_logs)
                scores = [t.get("score", 0) for t in optimizer.trial_logs if isinstance(t, dict)]
                if scores:
                    stats["score_mean"] = sum(scores) / len(scores)
                    stats["score_max"] = max(scores)
                    stats["score_min"] = min(scores)
        except Exception as e:
            logger.debug("Failed to extract optimizer stats: %s", e)

        return stats

    def _save_run_summary(self) -> None:
        """Save a summary of the run to disk."""
        summary = {
            "generation": self._generation,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "best_score": self._best_score,
            "stats": self._last_run_stats,
            "config": {
                "student_model": self.config.model.name,
                "reflection_model": self.config.opus.model,
                "trainset_size": len(self._trainset),
            },
        }

        summary_path = self.data_dir / "dspy_run_history.jsonl"
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, default=str) + "\n")
