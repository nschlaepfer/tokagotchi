"""Bash command execution tool for the agent arena."""

from __future__ import annotations

import re

from src.arena.docker_manager import DockerManager
from src.arena.tools.common import ToolResult

# Commands that are never allowed, even inside longer pipelines.
_BLOCKED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+-[^\s]*r[^\s]*f[^\s]*\s+/\s*$"),  # rm -rf /
    re.compile(r"\brm\s+-[^\s]*f[^\s]*r[^\s]*\s+/\s*$"),  # rm -fr /
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+.*of=/dev/"),
    re.compile(r":(){ :\|:& };:"),  # fork bomb
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\binit\s+0\b"),
]

DEFAULT_TIMEOUT = 30
MAX_OUTPUT_CHARS = 10_000


def _is_blocked(command: str) -> str | None:
    """Return a reason string if the command is blocked, else None."""
    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(command):
            return f"Blocked dangerous command matching pattern: {pattern.pattern}"
    return None


async def execute(
    docker_mgr: DockerManager,
    container_id: str,
    command: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> ToolResult:
    """Execute a bash command inside the container.

    Args:
        docker_mgr: The Docker manager instance.
        container_id: Target container.
        command: Shell command to run.
        timeout: Max seconds to wait.

    Returns:
        ToolResult with stdout, stderr, exit_code, and truncation flag.
    """
    blocked_reason = _is_blocked(command)
    if blocked_reason is not None:
        return ToolResult(
            stdout="",
            stderr=blocked_reason,
            exit_code=1,
            truncated=False,
        )

    try:
        stdout, stderr, exit_code = await docker_mgr.async_exec_in_container(
            container_id, command, timeout=timeout
        )
    except TimeoutError:
        return ToolResult(
            stdout="",
            stderr=f"Command timed out after {timeout}s",
            exit_code=124,
            truncated=False,
        )

    truncated = False
    if len(stdout) > MAX_OUTPUT_CHARS:
        stdout = stdout[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"
        truncated = True
    if len(stderr) > MAX_OUTPUT_CHARS:
        stderr = stderr[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"
        truncated = True

    return ToolResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        truncated=truncated,
    )
