#!/usr/bin/env python3
"""CLI entry point for Loop 1 (GEPA prompt evolution).

Usage::

    python scripts/run_loop1.py --config config/ --iterations 100
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.arena.docker_manager import create_arena_manager
from src.config import MasterConfig, load_config
from src.infra.eval_harness import EvalHarness
from src.infra.vllm_server import VLLMServer
from src.infra.vram_scheduler import VRAMScheduler
from src.loop1_gepa import GEPAEngine
from src.orchestrator.budget_tracker import BudgetTracker
from src.orchestrator.opus_client import OpusClient

logger = logging.getLogger("run_loop1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Loop 1 (GEPA) prompt evolution.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config",
        help="Path to the configuration directory (default: config/)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of GEPA iterations to run (default: 100)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory from config",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Path to task JSON file (overrides seed_tasks.json)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    cfg: MasterConfig = load_config(args.config)
    data_dir = Path(args.data_dir or cfg.data_dir)

    # Load evaluation tasks
    harness = EvalHarness()
    task_path = args.tasks or str(data_dir / "curriculum" / "seed_tasks.json")
    tasks = harness.load_benchmark_tasks(task_path)
    if not tasks:
        logger.error("No tasks loaded from %s — aborting", task_path)
        sys.exit(1)
    logger.info("Loaded %d evaluation tasks", len(tasks))

    # Initialize subsystems
    budget_tracker = BudgetTracker(
        hourly_limit_usd=cfg.opus.hourly_budget_usd,
        daily_limit_usd=cfg.opus.daily_budget_usd,
        persist_path=data_dir / "budget_state.json",
    )
    opus_client = OpusClient(config=cfg.opus, budget_tracker=budget_tracker)
    vllm_server = VLLMServer(cfg.model, log_dir=data_dir / "logs")
    arena_manager = create_arena_manager()

    # Start vLLM
    logger.info("Starting local LLM server (%s)...", cfg.model.normalized_provider)
    await vllm_server.start()

    try:
        # Create and run GEPA engine
        engine = GEPAEngine(
            config=cfg,
            opus_client=opus_client,
            vllm_server=vllm_server,
            arena_manager=arena_manager,
            tasks=tasks,
            data_dir=data_dir,
        )

        await engine.initialize()
        await engine.run(n_iterations=args.iterations)

        # Print results summary
        status = engine.get_status()
        print("\n" + "=" * 60)
        print("GEPA Loop 1 — Results Summary")
        print("=" * 60)
        print(f"  Generations completed:  {status['generation']}")
        print(f"  Population size:        {status['population_size']}")
        print(f"  Pareto frontier size:   {status['frontier_size']}")
        print(f"  Total iterations:       {status['total_iterations']}")
        print(f"  Tasks available:        {status['tasks_available']}")
        if status["best_scores"]:
            print("  Best scores:")
            for metric, value in status["best_scores"].items():
                print(f"    {metric}: {value:.4f}")
        print("=" * 60)

        # Budget summary
        budget = budget_tracker.get_summary()
        print(f"\n  Opus spend: ${budget['total_usd']:.2f} total")
        print(f"    Hourly:  ${budget['hourly_usd']:.2f}")
        print(f"    Daily:   ${budget['daily_usd']:.2f}")
        print(f"    Calls:   {budget['num_calls']}")

    finally:
        await vllm_server.stop()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    asyncio.run(main(args))
