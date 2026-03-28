"""Manages local subprocess sandboxes for agent arena episodes.

Drop-in replacement for DockerManager that uses temporary directories
and asyncio subprocesses instead of Docker containers.  Useful for
development, CI, and environments where Docker is unavailable.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models import TaskSpec

logger = logging.getLogger(__name__)

DEFAULT_EXEC_TIMEOUT = 30
WORKSPACE_SUBDIR = "workspace"

# ---------------------------------------------------------------------------
# Python shim: ensure 'python' command is available even if only python3 exists
# ---------------------------------------------------------------------------

_PYTHON_SHIM_DIR: str | None = None
_shim_checked = False


def _ensure_python_shim() -> None:
    """Create a 'python' shim so that 'python' maps to 'python3' in bash.

    On many systems (especially Windows with Git Bash / MSYS2), 'python3'
    is on PATH but 'python' is not.  We create a tiny shell script that
    delegates to python3.
    """
    global _PYTHON_SHIM_DIR, _shim_checked
    if _shim_checked:
        return
    _shim_checked = True

    shim_dir = tempfile.mkdtemp(prefix="arena-pyshim-")
    shim_path = os.path.join(shim_dir, "python")
    # Use 'exec python3' — bash will resolve python3 from its own PATH
    with open(shim_path, "w", newline="\n") as f:
        f.write('#!/bin/bash\nexec python3 "$@"\n')
    os.chmod(shim_path, 0o755)

    _PYTHON_SHIM_DIR = shim_dir
    logger.info("Created python->python3 shim at %s", shim_dir)

# ---------------------------------------------------------------------------
# Safety: blocked command patterns (mirrors bash_tool.py)
# ---------------------------------------------------------------------------

_BLOCKED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+-[^\s]*r[^\s]*f[^\s]*\s+/\s*$"),   # rm -rf /
    re.compile(r"\brm\s+-[^\s]*f[^\s]*r[^\s]*\s+/\s*$"),   # rm -fr /
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+.*of=/dev/"),
    re.compile(r":(){ :\|:& };:"),                           # fork bomb
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\binit\s+0\b"),
]


def _is_blocked(command: str) -> str | None:
    """Return a reason string if the command is blocked, else None."""
    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(command):
            return f"Blocked dangerous command matching pattern: {pattern.pattern}"
    return None


# ---------------------------------------------------------------------------
# Sandbox bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class _SandboxInfo:
    """Tracks a single subprocess sandbox."""

    sandbox_id: str
    root_dir: str  # top-level temp directory
    workspace: str  # root_dir / workspace


class SubprocessManager:
    """Manages local subprocess sandboxes for the agent arena.

    Provides the same async interface as ``DockerManager`` so that callers
    (e.g. ``AgentArenaGame``) can use either backend transparently.
    """

    def __init__(self, default_timeout: int = DEFAULT_EXEC_TIMEOUT) -> None:
        self.default_timeout = default_timeout
        self._sandboxes: dict[str, _SandboxInfo] = {}

    # ------------------------------------------------------------------
    # Container lifecycle  (container == temp-dir sandbox here)
    # ------------------------------------------------------------------

    def create_container(self, task_spec: TaskSpec) -> str:
        """Create a temp-dir sandbox and seed it with the task's files.

        Returns a sandbox ID (used in place of a Docker container ID).
        """
        sandbox_id = uuid.uuid4().hex
        root_dir = tempfile.mkdtemp(prefix=f"arena-{sandbox_id[:8]}-")
        workspace = os.path.join(root_dir, WORKSPACE_SUBDIR)
        os.makedirs(workspace, exist_ok=True)

        info = _SandboxInfo(
            sandbox_id=sandbox_id,
            root_dir=root_dir,
            workspace=workspace,
        )
        self._sandboxes[sandbox_id] = info
        logger.info("Created sandbox %s at %s", sandbox_id[:12], root_dir)

        # Seed initial files
        if task_spec.initial_files:
            self.copy_files_to_container(sandbox_id, task_spec.initial_files)

        return sandbox_id

    async def async_create_container(self, task_spec: TaskSpec) -> str:
        return await asyncio.to_thread(self.create_container, task_spec)

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def exec_in_container(
        self,
        container_id: str,
        command: str,
        timeout: int | None = None,
    ) -> tuple[str, str, int]:
        """Execute *command* synchronously (blocking) in the sandbox.

        Returns ``(stdout, stderr, exit_code)``.
        """
        return asyncio.get_event_loop().run_until_complete(
            self.async_exec_in_container(container_id, command, timeout)
        )

    async def async_exec_in_container(
        self,
        container_id: str,
        command: str,
        timeout: int | None = None,
    ) -> tuple[str, str, int]:
        """Execute *command* asynchronously in the sandbox.

        Returns ``(stdout, stderr, exit_code)``.
        Raises ``TimeoutError`` if the command exceeds the timeout.
        """
        timeout = timeout or self.default_timeout

        # Safety gate
        blocked = _is_blocked(command)
        if blocked is not None:
            return ("", blocked, 1)

        info = self._sandboxes.get(container_id)
        if info is None:
            return ("", f"Unknown sandbox: {container_id}", 1)

        try:
            env = os.environ.copy()
            # Ensure 'python' resolves: add a shim dir to PATH if needed
            _ensure_python_shim()
            if _PYTHON_SHIM_DIR:
                env["PATH"] = _PYTHON_SHIM_DIR + os.pathsep + env.get("PATH", "")
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c", command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=info.workspace,
                env=env,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
            stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
            exit_code = proc.returncode if proc.returncode is not None else -1
            return (stdout, stderr, exit_code)
        except asyncio.TimeoutError:
            # Kill the runaway process
            try:
                proc.kill()  # type: ignore[possibly-undefined]
            except Exception:
                pass
            raise TimeoutError(f"Command exceeded {timeout}s timeout") from None

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def copy_files_to_container(
        self,
        container_id: str,
        files: dict[str, str],
    ) -> None:
        """Write *files* into the sandbox workspace directory.

        Args:
            container_id: Target sandbox ID.
            files: Mapping of relative path -> file content.
        """
        info = self._sandboxes.get(container_id)
        if info is None:
            raise ValueError(f"Unknown sandbox: {container_id}")

        for rel_path, content in files.items():
            dest = os.path.join(info.workspace, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "w", encoding="utf-8") as fh:
                fh.write(content)

    async def async_copy_files_to_container(
        self,
        container_id: str,
        files: dict[str, str],
    ) -> None:
        await asyncio.to_thread(self.copy_files_to_container, container_id, files)

    # ------------------------------------------------------------------
    # Cleanup / destroy
    # ------------------------------------------------------------------

    def destroy_container(self, container_id: str) -> None:
        """Remove the sandbox temp directory and stop tracking it."""
        info = self._sandboxes.pop(container_id, None)
        if info is None:
            logger.debug("Sandbox %s already removed", container_id[:12])
            return
        try:
            shutil.rmtree(info.root_dir, ignore_errors=True)
            logger.info("Destroyed sandbox %s", container_id[:12])
        except Exception as exc:
            logger.error("Failed to destroy sandbox %s: %s", container_id[:12], exc)

    async def async_destroy_container(self, container_id: str) -> None:
        await asyncio.to_thread(self.destroy_container, container_id)

    def release_container(self, container_id: str) -> None:
        """For SubprocessManager, release == destroy (no pooling)."""
        self.destroy_container(container_id)

    async def async_release_container(self, container_id: str) -> None:
        await asyncio.to_thread(self.release_container, container_id)

    def destroy_all(self) -> None:
        """Destroy every tracked sandbox."""
        for cid in list(self._sandboxes):
            self.destroy_container(cid)
        logger.info("All sandboxes destroyed")

    async def async_destroy_all(self) -> None:
        await asyncio.to_thread(self.destroy_all)
