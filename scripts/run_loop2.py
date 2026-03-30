#!/usr/bin/env python3
"""CLI entry point for Loop 2 (on-policy distillation).

Collects Qwen rollouts, runs Opus trace surgery on failures,
accumulates corrected traces in the pending buffer, and triggers
QLoRA SFT when the buffer meets readiness criteria.

Usage::

    python scripts/run_loop2.py --config config/ --collect-rounds 20
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import MasterConfig, load_config
from src.arena.docker_manager import create_arena_manager
from src.curriculum.sec_engine import SECEngine
from src.infra.eval_harness import EvalHarness
from src.infra.vllm_server import VLLMServer
from src.infra.vram_scheduler import VRAMScheduler
from src.loop1_gepa import create_seed_genome, load_population
from src.loop2_distill import PendingBuffer, SFTLauncher, TraceCollector, TraceSurgeon
from src.models import PromptGenome
from src.orchestrator.budget_tracker import BudgetTracker
from src.orchestrator.opus_client import OpusClient

logger = logging.getLogger("run_loop2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Loop 2 (on-policy distillation).",
    )
    parser.add_argument(
        "--config", type=str, default="config",
        help="Path to the configuration directory",
    )
    parser.add_argument(
        "--collect-rounds", type=int, default=20,
        help="Number of trace collection rounds before checking buffer readiness",
    )
    parser.add_argument(
        "--tasks-per-round", type=int, default=5,
        help="Number of tasks to sample per collection round",
    )
    parser.add_argument(
        "--force-train", action="store_true",
        help="Force SFT training even if buffer is below threshold",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override data directory from config",
    )
    parser.add_argument(
        "--sandbox", type=str, default="auto",
        choices=["docker", "subprocess", "auto"],
        help="Arena sandbox backend: docker, subprocess, or auto-detect",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    cfg: MasterConfig = load_config(args.config)
    data_dir = Path(args.data_dir or cfg.data_dir)

    # Initialise subsystems
    budget_tracker = BudgetTracker(
        hourly_limit_usd=cfg.opus.hourly_budget_usd,
        daily_limit_usd=cfg.opus.daily_budget_usd,
        persist_path=data_dir / "budget_state.json",
    )
    opus_client = OpusClient(config=cfg.opus, budget_tracker=budget_tracker)
    vllm_server = VLLMServer(cfg.model)
    vram_scheduler = VRAMScheduler(vllm_server)

    # Arena sandbox: auto-detect Docker, fall back to subprocess
    use_docker = None  # auto-detect
    if args.sandbox == "docker":
        use_docker = True
    elif args.sandbox == "subprocess":
        use_docker = False
    arena_manager = create_arena_manager(use_docker=use_docker)
    logger.info("Arena backend: %s", type(arena_manager).__name__)

    trace_collector = TraceCollector()
    trace_surgeon = TraceSurgeon()
    pending_buffer = PendingBuffer(
        config=cfg.loop2,
        persist_path=data_dir / "pending.jsonl",
    )
    sft_launcher = SFTLauncher(output_dir=data_dir / "checkpoints")

    # Curriculum for task sampling
    curriculum = SECEngine(task_bank_path=data_dir / "task_bank.json")

    # Load best genome
    pop_path = data_dir / "loop1_gepa" / "population.json"
    genome: PromptGenome
    if pop_path.exists():
        population = load_population(pop_path)
        genome = max(population, key=lambda g: g.scores.get("success_rate", 0.0)) if population else create_seed_genome()
    else:
        genome = create_seed_genome()

    # Enter serving phase for trace collection
    logger.info("Starting vLLM server for trace collection...")
    await vram_scheduler.enter_serving_phase()

    try:
        # --- Phase 1: Collect traces ---
        for round_idx in range(args.collect_rounds):
            logger.info("Collection round %d/%d", round_idx + 1, args.collect_rounds)

            tasks = curriculum.sample(n=args.tasks_per_round)
            if not tasks:
                harness = EvalHarness()
                tasks = harness.load_benchmark_tasks(
                    str(data_dir / "curriculum" / "seed_tasks.json")
                )
                if not tasks:
                    logger.error("No tasks available for trace collection")
                    break
                tasks = tasks[: args.tasks_per_round]

            # Collect rollouts
            trajectories = await trace_collector.collect_rollouts(
                tasks=tasks,
                n_per_task=1,
                vllm_server=vllm_server,
                arena_manager=arena_manager,
                genome=genome,
            )

            # Run trace surgery on failures
            n_corrected = 0
            for traj in trajectories:
                if not traj.success:
                    analysis = await opus_client.correct_trace(traj)
                    if analysis.corrected_steps:
                        pending_buffer.add(trajectory=traj, analysis=analysis)
                        n_corrected += 1

            logger.info(
                "Round %d: %d trajectories, %d failures corrected, buffer=%d",
                round_idx + 1,
                len(trajectories),
                n_corrected,
                pending_buffer.size,
            )

            # Check if buffer is ready
            if pending_buffer.is_ready():
                logger.info("Buffer is ready for training (%d examples)", pending_buffer.size)
                break

        # --- Phase 2: SFT Training ---
        should_train = pending_buffer.is_ready() or args.force_train
        if should_train and pending_buffer.size > 0:
            logger.info("Starting SFT training with %d examples", pending_buffer.size)
            training_data = pending_buffer.drain()

            # Switch to training phase (stop vLLM, free VRAM)
            await vram_scheduler.enter_training_phase()

            try:
                # Use HF model path for training (not the Ollama tag)
                hf_model = str(Path(cfg.model.hf_model_path).resolve())
                if not Path(hf_model).exists():
                    # Fall back: try relative to project root
                    hf_model = str(_project_root / cfg.model.hf_model_path)
                logger.info("SFT base model: %s", hf_model)
                checkpoint = await sft_launcher.launch_training(
                    training_data=training_data,
                    config=cfg.loop2,
                    base_model_path=hf_model,
                )
                deployment_ts = time.strftime("%Y%m%d_%H%M%S")
                merged_path = str(data_dir / "checkpoints" / f"merged_{deployment_ts}")
                deployed_tag = await sft_launcher.deploy_adapter(
                    base_model_path=hf_model,
                    adapter_path=checkpoint,
                    tag=f"tokagotchi-loop2:{deployment_ts}",
                    merged_output_path=merged_path,
                )
                cfg.model.name = deployed_tag
                vllm_server.config.name = deployed_tag
                print(
                    f"\nSFT training complete. Checkpoint: {checkpoint}\n"
                    f"Deployed merged model to Ollama tag: {deployed_tag}"
                )
            finally:
                # Return to serving phase
                await vram_scheduler.enter_serving_phase()
        else:
            print(f"\nBuffer not ready for training (size={pending_buffer.size})")

        # Print summary
        budget = budget_tracker.get_summary()
        print("\n" + "=" * 60)
        print("Loop 2 (Distillation) — Summary")
        print("=" * 60)
        print(f"  Collection rounds:  {args.collect_rounds}")
        print(f"  Buffer size:        {pending_buffer.size}")
        print(f"  Training triggered: {should_train}")
        print(f"  Opus spend:         ${budget['total_usd']:.2f}")
        print(f"  API calls:          {budget['num_calls']}")
        print("=" * 60)

    finally:
        await vllm_server.stop()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    asyncio.run(main(args))
