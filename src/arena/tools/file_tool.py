"""File read/write tool for the agent arena.

All paths are restricted to /workspace inside the container.
"""

from __future__ import annotations

import shlex

from typing import Any

from src.arena.path_safety import WORKSPACE_ROOT, workspace_relative_path
from src.arena.tools.common import ToolResult


def _validate_and_relativize(path: str) -> tuple[str | None, str]:
    """Validate path is under /workspace and return (error, relative_path).

    Accepts:
    - /workspace/foo.py → foo.py
    - foo.py → foo.py  (already relative)
    - src/foo.py → src/foo.py  (already relative)

    Returns (error_msg, relative_path). error_msg is None on success.
    """
    try:
        return None, workspace_relative_path(path)
    except ValueError as exc:
        return str(exc), ""


async def read_file(
    docker_mgr: Any,
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
    error, rel_path = _validate_and_relativize(path)
    if error is not None:
        return ToolResult(stdout="", stderr=error, exit_code=1, truncated=False)

    try:
        # Use relative path — works in both Docker (/workspace is cwd) and
        # SubprocessManager (temp dir workspace is cwd)
        stdout, stderr, exit_code = await docker_mgr.async_exec_in_container(
            container_id, f"cat -- {shlex.quote(rel_path)}", timeout=10
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
    docker_mgr: Any,
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
    error, relative = _validate_and_relativize(path)
    if error is not None:
        return ToolResult(stdout="", stderr=error, exit_code=1, truncated=False)

    # Use copy_files_to_container for reliable writes
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
