"""DSPy ReAct tool adapters wrapping the arena's async tools.

DSPy ReAct expects synchronous Python callables with type hints and
docstrings.  These adapters bridge to the arena's async sandbox tools
by running a per-thread event loop.

Usage::

    from src.loop1_gepa.dspy_tools import build_dspy_tools
    tools = build_dspy_tools(arena_manager)
    react = dspy.ReAct("task -> solution", tools=tools, max_iters=20)
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from src.models import TaskSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-thread context for sandbox state
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """Holds the arena manager and active sandbox ID for the current thread."""
    arena_manager: Any = None
    container_id: str = ""


_tool_ctx: contextvars.ContextVar[ToolContext] = contextvars.ContextVar(
    "dspy_tool_ctx", default=ToolContext()
)

# Per-thread event loops for running async arena calls
_thread_loops: dict[int, asyncio.AbstractEventLoop] = {}
_thread_loops_lock = threading.Lock()


def _get_thread_loop() -> asyncio.AbstractEventLoop:
    """Get or create an event loop for the current thread."""
    tid = threading.get_ident()
    with _thread_loops_lock:
        if tid not in _thread_loops:
            loop = asyncio.new_event_loop()
            _thread_loops[tid] = loop
        return _thread_loops[tid]


def _run_async(coro: Any) -> Any:
    """Run an async coroutine synchronously in the current thread's loop."""
    loop = _get_thread_loop()
    return loop.run_until_complete(coro)


def set_tool_context(arena_manager: Any, container_id: str) -> None:
    """Set the arena context for the current thread/task."""
    _tool_ctx.set(ToolContext(arena_manager=arena_manager, container_id=container_id))


def get_tool_context() -> ToolContext:
    """Get the current thread's tool context."""
    return _tool_ctx.get()


# ---------------------------------------------------------------------------
# Tool adapters (sync wrappers around async arena tools)
# ---------------------------------------------------------------------------

def bash(command: str) -> str:
    """Execute a bash command in the sandbox workspace.

    Use this to run shell commands like ls, cat, grep, find, etc.
    The working directory is /workspace where task files are located.

    Args:
        command: The bash command to execute.

    Returns:
        The command output (stdout + stderr combined).
    """
    ctx = get_tool_context()
    if not ctx.arena_manager or not ctx.container_id:
        return "Error: No active sandbox. Tool context not set."
    try:
        stdout, stderr, exit_code = _run_async(
            ctx.arena_manager.async_exec_in_container(
                ctx.container_id, command, timeout=30
            )
        )
        output = stdout
        if stderr:
            output += f"\n[stderr]: {stderr}"
        if exit_code != 0:
            output += f"\n[exit_code]: {exit_code}"
        return output[:4000]  # Truncate long output
    except TimeoutError:
        return "Error: Command timed out after 30s."
    except Exception as e:
        return f"Error executing bash: {e}"


def python_exec(code: str) -> str:
    """Execute Python code in the sandbox.

    Use this to run Python scripts for data processing, computation,
    or testing. The code runs in the sandbox's Python interpreter.

    Args:
        code: The Python code to execute.

    Returns:
        The script output (stdout + stderr).
    """
    ctx = get_tool_context()
    if not ctx.arena_manager or not ctx.container_id:
        return "Error: No active sandbox."
    try:
        # Write code to a temp file and execute it
        escaped_code = code.replace("'", "'\\''")
        command = f"python3 -c '{escaped_code}'"
        stdout, stderr, exit_code = _run_async(
            ctx.arena_manager.async_exec_in_container(
                ctx.container_id, command, timeout=30
            )
        )
        output = stdout
        if stderr:
            output += f"\n[stderr]: {stderr}"
        if exit_code != 0:
            output += f"\n[exit_code]: {exit_code}"
        return output[:4000]
    except TimeoutError:
        return "Error: Python execution timed out after 30s."
    except Exception as e:
        return f"Error executing Python: {e}"


def read_file(path: str) -> str:
    """Read the contents of a file in the sandbox workspace.

    Args:
        path: Relative path from /workspace (e.g., 'data.csv', 'src/main.py').

    Returns:
        The file contents, or an error message.
    """
    ctx = get_tool_context()
    if not ctx.arena_manager or not ctx.container_id:
        return "Error: No active sandbox."
    try:
        stdout, stderr, exit_code = _run_async(
            ctx.arena_manager.async_exec_in_container(
                ctx.container_id, f"cat '{path}'", timeout=10
            )
        )
        if exit_code != 0:
            return f"Error reading file: {stderr}"
        return stdout[:8000]
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a file in the sandbox workspace.

    Args:
        path: Relative path from /workspace.
        content: The content to write.

    Returns:
        Confirmation message or error.
    """
    ctx = get_tool_context()
    if not ctx.arena_manager or not ctx.container_id:
        return "Error: No active sandbox."
    try:
        # Use heredoc to handle special characters
        escaped = content.replace("\\", "\\\\").replace("'", "'\\''")
        command = f"cat > '{path}' << 'DSPY_EOF'\n{content}\nDSPY_EOF"
        stdout, stderr, exit_code = _run_async(
            ctx.arena_manager.async_exec_in_container(
                ctx.container_id, command, timeout=10
            )
        )
        if exit_code != 0:
            return f"Error writing file: {stderr}"
        return f"Successfully wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def submit_answer(answer: str) -> str:
    """Submit your final answer for the task.

    Call this when you have completed the task and are ready to submit
    your solution. Write any output files first, then submit.

    Args:
        answer: Your final answer or summary of what you did.

    Returns:
        Confirmation that the answer was submitted.
    """
    # Submit is a marker — the answer is recorded by the metric function
    return f"SUBMITTED: {answer}"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_dspy_tools(arena_manager: Any = None) -> list:
    """Build the list of DSPy-compatible tool functions.

    Returns a list of callables suitable for ``dspy.ReAct(tools=...)``.
    """
    return [bash, python_exec, read_file, write_file, submit_answer]
