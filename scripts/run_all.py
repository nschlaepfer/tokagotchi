#!/usr/bin/env python3
"""Full pipeline entry point — runs MasterLoop with all three loops.

Initializes the complete self-improvement pipeline, installs signal
handlers for graceful shutdown, and runs indefinitely.

Usage::

    python scripts/run_all.py --config config/
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import load_config
from src.orchestrator.master_loop import MasterLoop

logger = logging.getLogger("run_all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Qwen self-improvement pipeline (all loops).",
    )
    parser.add_argument(
        "--config", type=str, default="config",
        help="Path to the configuration directory (default: config/)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override data directory from config",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Write logs to this file in addition to stdout",
    )
    return parser.parse_args()


def setup_logging(level: str, log_file: str | None = None) -> None:
    """Configure logging with optional file output."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=handlers,
    )


async def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.data_dir:
        cfg.data_dir = args.data_dir

    logger.info("Starting Qwen self-improvement pipeline")
    logger.info(
        "Config: model=%s, budget=$%.0f/day, loop3=%02d:00-%02d:00",
        cfg.model.name,
        cfg.opus.daily_budget_usd,
        cfg.schedule.loop3_start_hour,
        cfg.schedule.loop3_end_hour,
    )

    master = MasterLoop(config=cfg)
    await master.run()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level, args.log_file)

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.exception("Fatal error in run_all")
        sys.exit(1)
