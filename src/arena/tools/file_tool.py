"""File read/write tool for the agent arena.

All paths are restricted to /workspace inside the container.
"""

from __future__ import annotations

import posixpath

from src.arena.docker_manager import DockerManager
from src.arena.tools.common import ToolResult

WORKSPACE_ROOT = "/workspace"


def _validate_path(path: str) -> str | None:
    """Return an error message if the path is not under /workspace, else None."""
    normalized = posixpath.normpath(path)
    if not normalized.startswith(WORKSPACE_ROOT):
        return f"Path must be under {WORKSPACE_ROOT}, got: {path}"
    return None


async def read_file(
    docker_mgr: DockerManager,
    container_id: str,
    path: str,
) -> ToolResult:
    """Read a file from the container's /workspace.

    Args:
        docker_mgr: The Docker manager instance.
        container_id: Target container.
        path: Absolute path inside the container (must be under /workspace).

    Returns:
        ToolResult with file contents in stdout.
    """
    error = _validate_path(path)
    if error is not None:
        return ToolResult(stdout="", stderr=error, exit_code=1, truncated=False)

    try:
        stdout, stderr, exit_code = await docker_mgr.async_exec_in_container(
            container_id, f"cat {path}", timeout=10
        )
    except TimeoutError:
        return ToolResult(
            stdout="", stderr="Read timed out", exit_code=124, truncated=False
        )

    return ToolResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        truncated=False,
    )


async def write_file(
    docker_mgr: DockerManager,
    container_id: str,
    path: str,
    content: str,
) -> ToolResult:
    """Write content to a file in the container's /workspace.

    Args:
        docker_mgr: The Docker manager instance.
        container_id: Target container.
        path: Absolute path inside the container (must be under /workspace).
        content: File content to write.

    Returns:
        ToolResult confirming the write.
    """
    error = _validate_path(path)
    if error is not None:
        return ToolResult(stdout="", stderr=error, exit_code=1, truncated=False)

    # Ensure parent directory exists, then write via heredoc
    dir_path = posixpath.dirname(path)
    # Use copy_files_to_container for reliable writes
    relative = posixpath.relpath(path, WORKSPACE_ROOT)
    try:
        await docker_mgr.async_copy_files_to_container(
            container_id, {relative: content}
        )
    except Exception as exc:
        return ToolResult(
            stdout="",
            stderr=f"Write failed: {exc}",
            exit_code=1,
            truncated=False,
        )

    return ToolResult(
        stdout=f"Wrote {len(content)} bytes to {path}",
        stderr="",
        exit_code=0,
        truncated=False,
    )
