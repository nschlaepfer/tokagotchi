"""Python code execution tool for the agent arena."""

from __future__ import annotations

import uuid

from src.arena.docker_manager import DockerManager
from src.arena.tools.common import ToolResult

DEFAULT_TIMEOUT = 30
MAX_OUTPUT_CHARS = 10_000


async def execute(
    docker_mgr: DockerManager,
    container_id: str,
    code: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> ToolResult:
    """Write Python code to a temp file and execute it inside the container.

    Args:
        docker_mgr: The Docker manager instance.
        container_id: Target container.
        code: Python source code to run.
        timeout: Max seconds to wait.

    Returns:
        ToolResult with captured stdout, stderr, and exit code.
    """
    script_name = f"/tmp/_agent_script_{uuid.uuid4().hex[:8]}.py"

    # Write the code into the container
    await docker_mgr.async_copy_files_to_container(
        container_id,
        {script_name.lstrip("/"): code},
    )

    # Execute with python3, capturing both stdout and stderr
    command = f"python3 {script_name} 2>&1; echo '::EXIT_CODE::'$?"
    try:
        raw_stdout, _, _ = await docker_mgr.async_exec_in_container(
            container_id, command, timeout=timeout
        )
    except TimeoutError:
        return ToolResult(
            stdout="",
            stderr=f"Python execution timed out after {timeout}s",
            exit_code=124,
            truncated=False,
        )

    # Parse exit code from the sentinel line
    stdout = raw_stdout
    stderr = ""
    exit_code = 0

    sentinel = "::EXIT_CODE::"
    if sentinel in raw_stdout:
        parts = raw_stdout.rsplit(sentinel, 1)
        stdout = parts[0]
        try:
            exit_code = int(parts[1].strip())
        except (ValueError, IndexError):
            exit_code = 1

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
