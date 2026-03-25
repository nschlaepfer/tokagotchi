#!/usr/bin/env python3
"""CLI entry point for Loop 3 (overnight RL training).

Runs Tree-GRPO with DAPO clipping for reinforcement learning.
Handles VRAM phase transitions to free GPU memory for training.

Usage::

    python scripts/run_loop3.py --config config/
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.arena.docker_manager import create_arena_manager
from src.config import MasterConfig, load_config
from src.curriculum.sec_engine import SECEngine
from src.infra.vllm_server import VLLMServer
from src.infra.vram_scheduler import VRAMScheduler
from src.loop3_rl import RLRunner
from src.orchestrator.budget_tracker import BudgetTracker
from src.orchestrator.opus_client import OpusClient

logger = logging.getLogger("run_loop3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Loop 3 (overnight RL training with Tree-GRPO + DAPO).",
    )
    parser.add_argument(
        "--config", type=str, default="config",
        help="Path to the configuration directory",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs (default: from config)",
    )
    parser.add_argument(
        "--skip-overnight-check", action="store_true",
        help="Run immediately without waiting for the overnight window",
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

    if args.epochs is not None:
        cfg.loop3.total_epochs = args.epochs

    # Initialise subsystems
    budget_tracker = BudgetTracker(
        hourly_limit_usd=cfg.opus.hourly_budget_usd,
        daily_limit_usd=cfg.opus.daily_budget_usd,
        persist_path=data_dir / "budget_state.json",
    )
    opus_client = OpusClient(config=cfg.opus, budget_tracker=budget_tracker)
    vllm_server = VLLMServer(cfg.model, log_dir=data_dir / "logs")
    vram_scheduler = VRAMScheduler(vllm_server)

    # Arena sandbox: auto-detect Docker, fall back to subprocess
    use_docker = None
    if args.sandbox == "docker":
        use_docker = True
    elif args.sandbox == "subprocess":
        use_docker = False
    arena_manager = create_arena_manager(use_docker=use_docker)
    logger.info("Arena backend: %s", type(arena_manager).__name__)

    curriculum = SECEngine(task_bank_path=data_dir / "task_bank.json")

    rl_runner = RLRunner(
        config=cfg.loop3,
        output_dir=data_dir / "checkpoints" / "loop3",
    )

    # Wait for overnight window unless skipped
    if not args.skip_overnight_check:
        from datetime import datetime
        hour = datetime.now().hour
        start = cfg.schedule.loop3_start_hour
        end = cfg.schedule.loop3_end_hour

        in_window = (hour >= start or hour < end) if start > end else (start <= hour < end)
        if not in_window:
            logger.warning(
                "Not in overnight window (%02d:00-%02d:00, current=%02d:00). "
                "Use --skip-overnight-check to run anyway.",
                start, end, hour,
            )
            sys.exit(1)

    # Transition to training phase (stop vLLM, free VRAM)
    logger.info("Entering training phase...")
    await vram_scheduler.enter_training_phase()

    try:
        logger.info("Starting RL training (epochs=%d)", cfg.loop3.total_epochs)
        result = await rl_runner.run_overnight(
            config=cfg.loop3,
            vram_scheduler=vram_scheduler,
            vllm_server=vllm_server,
            arena_manager=arena_manager,
            opus_client=opus_client,
            curriculum_engine=curriculum,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("Loop 3 (RL Training) — Summary")
        print("=" * 60)
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  Result: {result}")

        budget = budget_tracker.get_summary()
        print(f"\n  Opus spend: ${budget['total_usd']:.2f}")
        print(f"  API calls:  {budget['num_calls']}")
        print("=" * 60)

    finally:
        # Return to serving phase
        await vram_scheduler.enter_serving_phase()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    asyncio.run(main(args))
