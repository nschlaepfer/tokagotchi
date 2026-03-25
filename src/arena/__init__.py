"""Agent Arena: sandboxed execution environments for RL episodes.

Provides two backends:
  - DockerManager:      production isolation via Docker containers
  - SubprocessManager:  lightweight isolation via temp-dir + subprocess (dev/CI)

Use ``create_arena_manager()`` to auto-detect which is available,
or pass ``--sandbox docker|subprocess`` in CLI scripts.
"""

from src.arena.docker_manager import DockerManager, create_arena_manager
from src.arena.subprocess_manager import SubprocessManager
from src.arena.game import AgentArenaGame

# Type alias for anything that game.py / trace_collector / rl_runner accept.
# Both DockerManager and SubprocessManager implement this interface.
ArenaManager = DockerManager | SubprocessManager

__all__ = [
    "AgentArenaGame",
    "ArenaManager",
    "DockerManager",
    "SubprocessManager",
    "create_arena_manager",
]
