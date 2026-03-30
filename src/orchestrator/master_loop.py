"""Top-level orchestrator that schedules all three self-improvement loops.

MasterLoop runs continuously, coordinating:
- Loop 1 (GEPA): prompt evolution (continuous)
- Loop 2 (distillation): SFT when the pending buffer is ready
- Loop 3 (RL): overnight Tree-GRPO + DAPO training
- Periodic evaluation against held-out benchmarks
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.arena.docker_manager import create_arena_manager
from src.config import MasterConfig, load_config
from src.curriculum.sec_engine import SECEngine
from src.infra.eval_harness import EvalHarness
from src.infra.vllm_server import VLLMServer
from src.infra.vram_scheduler import VRAMScheduler
from src.loop1_gepa import GEPAEngine, create_seed_genome, load_population
from src.loop2_distill import (
    MentorSession,
    PendingBuffer,
    SFTLauncher,
    TraceCollector,
    TraceSurgeon,
)
from src.loop2_distill.sdpo_reevaluator import SDPOReevaluator
from src.loop3_rl import RLRunner
from src.models import EvalResult, PromptGenome, TaskSpec
from src.orchestrator.budget_tracker import BudgetTracker
from src.orchestrator.experiment_git import ExperimentGit
from src.orchestrator.opus_client import OpusClient
from src.rewards import CompositeReward
from src.infra import wandb_tracker

logger = logging.getLogger(__name__)


class MasterLoop:
    """Orchestrates all three self-improvement loops.

    Parameters
    ----------
    config:
        Fully merged MasterConfig.
    """

    def __init__(self, config: MasterConfig) -> None:
        self.config = config
        self._shutdown_event = asyncio.Event()
        self._tasks: list[asyncio.Task[Any]] = []

        # Status tracking
        self._loop_status: dict[str, str] = {
            "loop1_gepa": "idle",
            "loop2_distill": "idle",
            "loop3_rl": "idle",
            "eval": "idle",
        }
        self._metrics: dict[str, Any] = {}
        self._start_time: float = 0.0

        # Subsystems (initialized in _init_subsystems)
        self.budget_tracker: BudgetTracker | None = None
        self.opus_client: OpusClient | None = None
        self.vllm_server: VLLMServer | None = None
        self.vram_scheduler: VRAMScheduler | None = None
        self.arena_manager: Any | None = None  # DockerManager or SubprocessManager
        self.curriculum: SECEngine | None = None
        self.composite_reward: CompositeReward | None = None
        self.experiment_git: ExperimentGit | None = None
        self.eval_harness: EvalHarness | None = None

        # Loop-specific engines
        self._gepa_engine: GEPAEngine | None = None
        self._trace_collector: TraceCollector | None = None
        self._trace_surgeon: TraceSurgeon | None = None
        self._sdpo_reevaluator: SDPOReevaluator | None = None
        self._pending_buffer: PendingBuffer | None = None
        self._sft_launcher: SFTLauncher | None = None
        self._rl_runner: RLRunner | None = None

        # Current best genome
        self._best_genome: PromptGenome | None = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main async entry point. Runs until SIGINT/SIGTERM or fatal error."""
        self._start_time = time.time()
        logger.info("MasterLoop starting up")

        # Install signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._signal_handler)
            except NotImplementedError:
                # Windows does not support add_signal_handler; fall back to
                # signal.signal which works cross-platform for these signals.
                signal.signal(sig, lambda s, f: self._signal_handler())

        try:
            await self._init_subsystems()

            # Launch concurrent loops
            self._tasks = [
                asyncio.create_task(self._run_loop1(), name="loop1_gepa"),
                asyncio.create_task(self._run_loop2(), name="loop2_distill"),
                asyncio.create_task(self._run_loop3(), name="loop3_rl"),
                asyncio.create_task(self._run_eval(), name="eval_periodic"),
                asyncio.create_task(self._run_status_reporter(), name="status"),
            ]

            # Wait for shutdown signal or any task to raise
            done, _ = await asyncio.wait(
                self._tasks,
                return_when=asyncio.FIRST_EXCEPTION,
            )

            # If a task failed, log it
            for task in done:
                if task.exception() and not self._shutdown_event.is_set():
                    logger.error(
                        "Task %s failed: %s",
                        task.get_name(),
                        task.exception(),
                    )

        except asyncio.CancelledError:
            logger.info("MasterLoop cancelled")
        finally:
            await self._shutdown()

    # ------------------------------------------------------------------
    # Subsystem initialisation
    # ------------------------------------------------------------------

    async def _init_subsystems(self) -> None:
        """Create and wire up all subsystem instances."""
        cfg = self.config
        data_dir = Path(cfg.data_dir)

        # Budget tracker
        self.budget_tracker = BudgetTracker(
            hourly_limit_usd=cfg.opus.hourly_budget_usd,
            daily_limit_usd=cfg.opus.daily_budget_usd,
            persist_path=data_dir / "budget_state.json",
        )

        # Opus client
        self.opus_client = OpusClient(
            config=cfg.opus,
            budget_tracker=self.budget_tracker,
        )

        # vLLM server
        self.vllm_server = VLLMServer(cfg.model)

        # VRAM scheduler
        self.vram_scheduler = VRAMScheduler(self.vllm_server)

        # Arena (auto-detects Docker, falls back to subprocess)
        self.arena_manager = create_arena_manager()
        logger.info("Arena backend: %s", type(self.arena_manager).__name__)

        # Curriculum
        self.curriculum = SECEngine(task_bank_path=data_dir / "task_bank.json")

        # Rewards
        self.composite_reward = CompositeReward(weights=cfg.reward_weights)

        # Git tracking
        self.experiment_git = ExperimentGit(repo_path=".")
        await self.experiment_git.init_repo()

        # Eval harness
        self.eval_harness = EvalHarness()

        # Loop 1 — GEPA
        seed_tasks = self.eval_harness.load_benchmark_tasks(
            str(data_dir / "curriculum" / "seed_tasks.json")
        )
        if getattr(cfg.loop1, "use_dspy_gepa", False):
            from src.loop1_gepa.dspy_gepa_engine import DspyGEPAEngine
            self._gepa_engine = DspyGEPAEngine(
                config=cfg,
                opus_client=self.opus_client,
                vllm_server=self.vllm_server,
                arena_manager=self.arena_manager,
                tasks=seed_tasks,
                data_dir=data_dir / "loop1_gepa",
            )
            logger.info("Loop 1 engine: DspyGEPAEngine (real DSPy GEPA)")
        else:
            self._gepa_engine = GEPAEngine(
                config=cfg,
                opus_client=self.opus_client,
                vllm_server=self.vllm_server,
                arena_manager=self.arena_manager,
                tasks=seed_tasks,
                data_dir=data_dir,
            )
            logger.info("Loop 1 engine: GEPAEngine (custom)")

        # Loop 2 — Distillation
        self._trace_collector = TraceCollector()
        self._trace_surgeon = TraceSurgeon()
        self._sdpo_reevaluator = SDPOReevaluator()
        self._pending_buffer = PendingBuffer(
            config=cfg.loop2,
            persist_path=data_dir / "pending.jsonl",
        )
        self._pending_buffer.load()  # Restore examples from previous runs
        self._sft_launcher = SFTLauncher(
            output_dir=data_dir / "checkpoints",
        )

        # Loop 3 — RL
        self._rl_runner = RLRunner(
            config=cfg.loop3,
            output_dir=data_dir / "checkpoints" / "loop3",
        )

        # Load or create best genome
        pop_path = data_dir / "loop1_gepa" / "population.json"
        if pop_path.exists():
            population = load_population(pop_path)
            if population:
                self._best_genome = max(
                    population,
                    key=lambda g: g.scores.get("success_rate", 0.0),
                )
        if self._best_genome is None:
            self._best_genome = create_seed_genome()

        # Enter serving phase so Loop 1 can run
        await self.vram_scheduler.enter_serving_phase()

        logger.info("All subsystems initialized")

        # Initialize wandb tracking
        wandb_tracker.init(
            project="tokagotchi",
            name=f"run-{time.strftime('%Y%m%d-%H%M')}",
            config={
                "model": self.config.model.name,
                "gpu": "RTX 5090 32GB",
                "loop1_enabled": True,
                "loop2_sdpo": True,
                "loop3_overnight": True,
            },
            tags=["tokagotchi", self.config.model.name.split("/")[-1]],
        )

    # ------------------------------------------------------------------
    # Loop 1: GEPA (continuous prompt evolution)
    # ------------------------------------------------------------------

    async def _run_loop1(self) -> None:
        """Run Loop 1 (GEPA) continuously in the background."""
        assert self._gepa_engine is not None
        self._loop_status["loop1_gepa"] = "running"
        logger.info("Loop 1 (GEPA) started")

        try:
            await self._gepa_engine.initialize()

            while not self._shutdown_event.is_set():
                try:
                    # Run a batch of iterations
                    await self._gepa_engine.run(n_iterations=10)

                    # Update best genome
                    status = self._gepa_engine.get_status()
                    self._metrics["loop1"] = status

                    # Yield control briefly
                    await asyncio.sleep(1.0)

                except Exception:
                    logger.exception("Loop 1 iteration error")
                    await asyncio.sleep(30.0)

        except asyncio.CancelledError:
            pass
        finally:
            self._loop_status["loop1_gepa"] = "stopped"
            logger.info("Loop 1 (GEPA) stopped")

    # ------------------------------------------------------------------
    # Loop 2: Distillation (buffer-triggered SFT)
    # ------------------------------------------------------------------

    async def _run_loop2(self) -> None:
        """Monitor the pending buffer and trigger SFT when ready."""
        assert self._pending_buffer is not None
        assert self._trace_collector is not None
        assert self._sft_launcher is not None
        assert self.vram_scheduler is not None
        assert self.vllm_server is not None

        self._loop_status["loop2_distill"] = "monitoring"
        logger.info("Loop 2 (distillation) started — monitoring buffer")

        try:
            while not self._shutdown_event.is_set():
                try:
                    # Check if buffer is ready for training
                    if self._pending_buffer.is_ready():
                        logger.info("Loop 2: buffer ready — triggering SFT")
                        self._loop_status["loop2_distill"] = "training"

                        # Drain training examples from the buffer
                        training_data = self._pending_buffer.get_training_batch()

                        # Transition to training phase (stop vLLM)
                        await self.vram_scheduler.enter_training_phase()

                        try:
                            # Use HF model path for training (not Ollama tag)
                            hf_model = str(Path(self.config.model.hf_model_path).resolve())
                            if not Path(hf_model).exists():
                                hf_model = str(Path(".") / self.config.model.hf_model_path)
                            checkpoint_path = await self._sft_launcher.launch_training(
                                training_data=training_data,
                                config=self.config.loop2,
                                base_model_path=hf_model,
                            )
                            logger.info("Loop 2: SFT complete — checkpoint at %s", checkpoint_path)

                            # Export to Ollama (merge LoRA → GGUF → ollama create)
                            try:
                                ollama_name = await self._sft_launcher.export_to_ollama(
                                    base_model_path=hf_model,
                                    adapter_path=checkpoint_path,
                                    ollama_model_name="tokagotchi:latest",
                                    quantization="q4_k_m",
                                )
                                logger.info("Loop 2: Exported to Ollama as %s", ollama_name)
                            except Exception:
                                logger.exception("Loop 2: Ollama export failed (adapter still saved)")

                            # Commit results via git
                            if self.experiment_git:
                                branch = await self.experiment_git.create_experiment_branch(
                                    "loop2", "sft_training",
                                )
                                await self.experiment_git.commit_results(
                                    [checkpoint_path],
                                    f"Loop 2 SFT: {len(training_data)} examples",
                                )
                        finally:
                            # Free any leftover PyTorch GPU memory before Ollama restarts
                            self._free_training_vram()
                            # Return to serving phase
                            await self.vram_scheduler.enter_serving_phase()

                        self._loop_status["loop2_distill"] = "monitoring"
                    else:
                        # Collect more traces if Loop 1 is running
                        await self._collect_traces_for_buffer()

                        # Persist buffer to disk after each collection round
                        if self._pending_buffer.size() > 0:
                            self._pending_buffer.save()
                            logger.debug(
                                "Buffer saved: %d examples", self._pending_buffer.size()
                            )

                except Exception:
                    logger.exception("Loop 2 error")
                    self._loop_status["loop2_distill"] = "error"
                    await asyncio.sleep(60.0)
                    self._loop_status["loop2_distill"] = "monitoring"

                await asyncio.sleep(30.0)

        except asyncio.CancelledError:
            pass
        finally:
            self._loop_status["loop2_distill"] = "stopped"
            logger.info("Loop 2 (distillation) stopped")

    @staticmethod
    def _free_training_vram() -> None:
        """Release all PyTorch GPU memory so Ollama can reclaim VRAM."""
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                free_mb = torch.cuda.mem_get_info()[0] / 1024 / 1024
                logger.info("GPU memory freed before serving: %.0f MiB available", free_mb)
        except Exception as e:
            logger.warning("Failed to free GPU memory: %s", e)

    async def _collect_traces_for_buffer(self) -> None:
        """Collect rollout traces and feed corrected versions into the buffer."""
        assert self._trace_collector is not None
        assert self._trace_surgeon is not None
        assert self._pending_buffer is not None
        assert self.vllm_server is not None
        assert self.arena_manager is not None
        assert self.opus_client is not None

        if self._best_genome is None:
            return

        # Sample tasks from curriculum
        if self.curriculum is None:
            return
        tasks = self.curriculum.sample_tasks(batch_size=5)
        if not tasks:
            return

        # Collect rollouts
        trajectories = await self._trace_collector.collect_rollouts(
            tasks=tasks,
            n_per_task=1,
            vllm_server=self.vllm_server,
            arena_manager=self.arena_manager,
            genome=self._best_genome,
        )

        # Single pass: count and collect failed trajectories
        failed = [t for t in trajectories if not t.success]
        logger.info(
            "Trace collection: %d trajectories (%d success, %d failed)",
            len(trajectories), len(trajectories) - len(failed), len(failed),
        )

        # Process failed trajectories: SDPO first (free), Opus fallback (costly)
        for traj in failed:
                # SDPO: self-distillation using Qwen's own re-evaluation
                pairs: list = []
                if self._sdpo_reevaluator is not None and self._best_genome is not None:
                    try:
                        pairs = await self._sdpo_reevaluator.reevaluate(
                            traj, self.vllm_server, self._best_genome,
                        )
                        if pairs:
                            examples = self._sdpo_reevaluator.generate_training_examples(
                                pairs, traj,
                            )
                            for ex in examples:
                                self._pending_buffer.add(
                                    example=ex["example"],
                                    metadata=ex["metadata"],
                                )
                            logger.info(
                                "SDPO: %d contrastive pairs from trajectory %s (buffer=%d)",
                                len(pairs),
                                traj.trajectory_id[:12],
                                self._pending_buffer.size(),
                            )
                            # Track in wandb
                            avg_div = sum(p.weight for p in pairs) / len(pairs) if pairs else 0
                            wandb_tracker.log_sdpo(
                                trajectory_id=traj.trajectory_id,
                                num_pairs=len(pairs),
                                num_steps_checked=len(pairs),
                                avg_divergence=avg_div,
                                escalated_to_opus=False,
                            )
                    except Exception:
                        logger.exception("SDPO re-evaluation failed")

                # Opus fallback: only if SDPO produced nothing
                if not pairs:
                    try:
                        analysis = await self.opus_client.correct_trace(traj)
                        if analysis and analysis.corrected_steps:
                            ex = self._trace_surgeon.generate_training_example(
                                traj, analysis,
                            )
                            if ex:
                                task_type = (
                                    traj.task.task_type.value
                                    if traj.task
                                    else "unknown"
                                )
                                self._pending_buffer.add(
                                    example=ex,
                                    metadata={
                                        "task_type": task_type,
                                        "failure_mode": "opus_corrected",
                                        "source": "opus",
                                        "difficulty": traj.task.difficulty if traj.task else 0.5,
                                    },
                                )
                                logger.info(
                                    "Opus: corrected trajectory %s (%d steps)",
                                    traj.trajectory_id[:12],
                                    len(analysis.corrected_steps),
                                )
                    except Exception:
                        logger.exception("Opus trace surgery failed")

    # ------------------------------------------------------------------
    # Loop 3: Overnight RL
    # ------------------------------------------------------------------

    async def _run_loop3(self) -> None:
        """Start RL training during the overnight window."""
        assert self._rl_runner is not None
        assert self.vram_scheduler is not None
        self._loop_status["loop3_rl"] = "waiting"
        logger.info("Loop 3 (RL) started — waiting for overnight window")

        # Wait for Loop 1 to finish initial population + at least one GEPA cycle
        # before stealing the GPU for RL training
        logger.info("Loop 3: waiting 5 minutes for Loop 1/2 to collect initial data...")
        await asyncio.sleep(300)

        try:
            while not self._shutdown_event.is_set():
                try:
                    if self._is_overnight_window():
                        logger.info("Loop 3: overnight window — starting RL")
                        self._loop_status["loop3_rl"] = "training"

                        result = await self._rl_runner.run_overnight(
                            config=self.config.loop3,
                            vram_scheduler=self.vram_scheduler,
                            vllm_server=self.vllm_server,
                            arena_manager=self.arena_manager,
                            opus_client=self.opus_client,
                            curriculum_engine=self.curriculum,
                        )

                        self._metrics["loop3"] = result
                        logger.info("Loop 3: RL training complete: %s", result)

                        # Tag the best checkpoint
                        if self.experiment_git and result.get("improved"):
                            await self.experiment_git.tag_best(
                                f"best-rl-{time.strftime('%Y%m%d')}"
                            )

                        self._loop_status["loop3_rl"] = "waiting"

                        # Sleep until the next overnight window
                        await asyncio.sleep(3600)
                    else:
                        await asyncio.sleep(300)

                except Exception:
                    logger.exception("Loop 3 error")
                    self._loop_status["loop3_rl"] = "error"
                    await asyncio.sleep(600)
                    self._loop_status["loop3_rl"] = "waiting"

        except asyncio.CancelledError:
            pass
        finally:
            self._loop_status["loop3_rl"] = "stopped"
            logger.info("Loop 3 (RL) stopped")

    def _is_overnight_window(self) -> bool:
        """Check whether the current time falls in the overnight training window."""
        hour = datetime.now().hour
        start = self.config.schedule.loop3_start_hour
        end = self.config.schedule.loop3_end_hour

        if start > end:
            # Window wraps midnight (e.g. 22:00 - 06:00)
            return hour >= start or hour < end
        return start <= hour < end

    # ------------------------------------------------------------------
    # Periodic evaluation
    # ------------------------------------------------------------------

    async def _run_eval(self) -> None:
        """Run periodic evaluation against held-out benchmarks."""
        assert self.eval_harness is not None
        self._loop_status["eval"] = "idle"
        logger.info("Periodic evaluation started")

        frequency_seconds = self.config.schedule.eval_frequency_minutes * 60
        data_dir = Path(self.config.data_dir)
        benchmark_path = str(data_dir / "curriculum" / "seed_tasks.json")
        results_dir = data_dir / "eval_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        previous_result: EvalResult | None = None

        try:
            while not self._shutdown_event.is_set():
                await asyncio.sleep(frequency_seconds)

                if self._shutdown_event.is_set():
                    break

                try:
                    self._loop_status["eval"] = "running"
                    tasks = self.eval_harness.load_benchmark_tasks(benchmark_path)
                    if not tasks or self._best_genome is None:
                        continue

                    result = await self.eval_harness.run_evaluation(
                        tasks=tasks,
                        vllm_server=self.vllm_server,
                        arena_manager=self.arena_manager,
                        genome=self._best_genome,
                    )

                    # Save results
                    result_path = str(
                        results_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
                    )
                    self.eval_harness.save_results(result, result_path)
                    self._metrics["latest_eval"] = result.score_vector

                    # Regression check
                    if previous_result is not None:
                        passed, details = self.eval_harness.run_regression_check(
                            before=previous_result,
                            after=result,
                        )
                        if not passed:
                            logger.warning(
                                "Regression detected: %s", details
                            )
                            self._metrics["regression"] = details
                        else:
                            logger.info("Eval passed regression check")

                    previous_result = result
                    self._loop_status["eval"] = "idle"

                except Exception:
                    logger.exception("Periodic evaluation error")
                    self._loop_status["eval"] = "error"

        except asyncio.CancelledError:
            pass
        finally:
            self._loop_status["eval"] = "stopped"
            logger.info("Periodic evaluation stopped")

    # ------------------------------------------------------------------
    # Status reporting
    # ------------------------------------------------------------------

    async def _run_status_reporter(self) -> None:
        """Periodically log system status."""
        try:
            while not self._shutdown_event.is_set():
                await asyncio.sleep(120)
                self._report_status()
        except asyncio.CancelledError:
            pass

    def _report_status(self) -> None:
        """Log the current status of all loops."""
        uptime = time.time() - self._start_time if self._start_time else 0
        budget_summary = (
            self.budget_tracker.get_summary() if self.budget_tracker else {}
        )

        hourly_usd = budget_summary.get("hourly_usd", 0)
        daily_usd = budget_summary.get("daily_usd", 0)

        logger.info(
            "=== MasterLoop Status (uptime %.0fs) ===\n"
            "  Loops: %s\n"
            "  Budget: $%.2f hourly / $%.2f daily\n"
            "  Metrics: %s",
            uptime,
            self._loop_status,
            hourly_usd,
            daily_usd,
            {k: v for k, v in self._metrics.items() if k != "loop1"},
        )

        # Track in wandb
        wandb_tracker.log_budget(
            hourly_usd=hourly_usd,
            daily_usd=daily_usd,
            opus_calls=budget_summary.get("num_calls", 0),
        )
        wandb_tracker.log_pipeline_status(
            uptime_seconds=uptime,
            loop1_status=self._loop_status.get("loop1_gepa", "unknown"),
            loop2_status=self._loop_status.get("loop2_distill", "unknown"),
            loop3_status=self._loop_status.get("loop3_rl", "unknown"),
        )

    def get_status(self) -> dict[str, Any]:
        """Return a snapshot of the current system status."""
        return {
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            "loops": dict(self._loop_status),
            "metrics": dict(self._metrics),
            "budget": self.budget_tracker.get_summary() if self.budget_tracker else {},
        }

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _signal_handler(self) -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.info("Shutdown signal received")
        self._shutdown_event.set()
        for task in self._tasks:
            task.cancel()

    async def _shutdown(self) -> None:
        """Clean up all subsystems."""
        logger.info("MasterLoop shutting down...")

        # Cancel any remaining tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to finish
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Stop vLLM server
        if self.vllm_server:
            await self.vllm_server.stop()

        logger.info("MasterLoop shutdown complete")
