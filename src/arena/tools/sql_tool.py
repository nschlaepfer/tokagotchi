"""SQLite query tool for the agent arena.

Executes SQL queries against the /workspace/data.db database inside the
container.  By default, dangerous write operations (DROP, DELETE, ALTER,
TRUNCATE) are rejected unless explicitly allowed.
"""

from __future__ import annotations

import re

from src.arena.docker_manager import DockerManager
from src.arena.tools.common import ToolResult

DEFAULT_TIMEOUT = 15
MAX_OUTPUT_CHARS = 10_000
DB_PATH = "/workspace/data.db"

# Patterns for dangerous write operations
_DANGEROUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bDROP\b", re.IGNORECASE),
    re.compile(r"\bDELETE\b", re.IGNORECASE),
    re.compile(r"\bALTER\b", re.IGNORECASE),
    re.compile(r"\bTRUNCATE\b", re.IGNORECASE),
]


def _is_dangerous(query: str) -> str | None:
    """Return a reason string if the query contains a dangerous operation."""
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(query):
            return (
                f"Blocked: query contains disallowed operation "
                f"matching '{pattern.pattern}'. "
                f"Only SELECT / INSERT / UPDATE / CREATE are permitted."
            )
    return None


async def execute(
    docker_mgr: DockerManager,
    container_id: str,
    query: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    allow_dangerous: bool = False,
) -> ToolResult:
    """Execute a SQLite query inside the container.

    Runs ``sqlite3 /workspace/data.db "<query>"`` and returns the
    formatted results.

    Parameters
    ----------
    docker_mgr:
        The Docker manager instance.
    container_id:
        Target container.
    query:
        SQL query string.
    timeout:
        Max seconds to wait for the query to finish.
    allow_dangerous:
        If True, skip the safety check for DROP/DELETE/ALTER/TRUNCATE.

    Returns
    -------
    ToolResult
        Query results as formatted text, or an error message.
    """
    query = query.strip().rstrip(";")
    if not query:
        return ToolResult(
            stdout="",
            stderr="Empty query.",
            exit_code=1,
        )

    # Read-only safety check
    if not allow_dangerous:
        blocked = _is_dangerous(query)
        if blocked is not None:
            return ToolResult(
                stdout="",
                stderr=blocked,
                exit_code=1,
            )

    # Escape single quotes in the query for the shell command
    escaped_query = query.replace("'", "'\\''")
    command = (
        f"sqlite3 -header -column {DB_PATH} '{escaped_query};'"
    )

    try:
        stdout, stderr, exit_code = await docker_mgr.async_exec_in_container(
            container_id, command, timeout=timeout
        )
    except TimeoutError:
        return ToolResult(
            stdout="",
            stderr=f"SQL query timed out after {timeout}s",
            exit_code=124,
        )

    truncated = False
    if len(stdout) > MAX_OUTPUT_CHARS:
        stdout = stdout[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"
        truncated = True

    return ToolResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        truncated=truncated,
    )
