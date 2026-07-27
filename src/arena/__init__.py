"""Agent Arena: sandboxed execution environments for RL episodes.

Provides two backends:
  - DockerManager:      production isolation via Docker containers
  - SubprocessManager:  unsafe host subprocess backend for explicit local tests

Use ``create_arena_manager()`` for fail-closed Docker execution. Host
subprocess execution requires an explicit unsafe opt-in.
"""

from src.arena.docker_manager import (
    ArenaUnavailableError,
    DockerManager,
    UnsafeArenaBackendError,
    create_arena_manager,
)
from src.arena.subprocess_manager import SubprocessManager
from src.arena.game import AgentArenaGame

# Type alias for anything that game.py / trace_collector / rl_runner accept.
# Both DockerManager and SubprocessManager implement this interface.
ArenaManager = DockerManager | SubprocessManager

__all__ = [
    "AgentArenaGame",
    "ArenaManager",
    "ArenaUnavailableError",
    "DockerManager",
    "SubprocessManager",
    "UnsafeArenaBackendError",
    "create_arena_manager",
]
