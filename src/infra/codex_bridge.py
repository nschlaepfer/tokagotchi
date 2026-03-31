"""Async bridge to the Codex CLI companion runtime.

Provides fire-and-forget background task delegation and polling for
results, enabling the pipeline to offload diagnosis, code review,
and investigation tasks to GPT-5.4 via the Codex plugin.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Locate the codex-companion script from the installed plugin
_COMPANION_SCRIPT: str | None = None
for candidate in [
    Path.home() / ".claude" / "plugins" / "cache" / "openai-codex" / "codex" / "1.0.0" / "scripts" / "codex-companion.mjs",
    Path.home() / ".claude" / "plugins" / "marketplaces" / "openai-codex" / "plugins" / "codex" / "scripts" / "codex-companion.mjs",
]:
    if candidate.exists():
        _COMPANION_SCRIPT = str(candidate)
        break


def _get_codex_env() -> dict[str, str]:
    """Build an environment dict with npm global bin dir in PATH."""
    env = {**os.environ}
    npm_bin = Path.home() / "AppData" / "Roaming" / "npm"
    if npm_bin.exists():
        env["PATH"] = f"{npm_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


class CodexBridge:
    """Async interface to the Codex companion task runtime.

    Usage::

        bridge = CodexBridge(cwd="/path/to/repo")
        job_id = await bridge.spawn_task("Investigate why training failed...")
        # ... later ...
        result = await bridge.poll_result(job_id, timeout=300)
    """

    def __init__(
        self,
        cwd: str | Path = ".",
        *,
        model: str | None = None,
        effort: str = "high",
        write: bool = False,
    ) -> None:
        self.cwd = str(Path(cwd).resolve())
        self.model = model
        self.effort = effort
        self.write = write
        self._available: bool | None = None

    async def is_available(self) -> bool:
        """Check if Codex CLI is installed."""
        if self._available is not None:
            return self._available

        try:
            proc = await asyncio.create_subprocess_shell(
                "codex --version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_get_codex_env(),
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            self._available = proc.returncode == 0
            if self._available:
                logger.info("Codex CLI available: %s", stdout.decode().strip())
        except Exception:
            self._available = False

        if not self._available:
            logger.info("Codex CLI not available — Codex integration disabled")
        return self._available

    async def spawn_task(
        self,
        prompt: str,
        *,
        background: bool = True,
        model: str | None = None,
        effort: str | None = None,
        write: bool | None = None,
        fresh: bool = True,
    ) -> str | None:
        """Spawn a Codex task and return the job ID (background) or result (foreground).

        Returns None if Codex is not available.
        """
        if not await self.is_available():
            return None

        # Use codex CLI directly (shell=True for Windows .cmd resolution)
        _write = write if write is not None else self.write
        _model = model or self.model
        _effort = effort or self.effort

        # Escape the prompt for shell safety
        safe_prompt = prompt.replace('"', '\\"')
        parts = ["codex"]
        if _write:
            parts.append("--full-auto")
        if _model:
            parts.extend(["--model", _model])
        parts.append(f'"{safe_prompt}"')
        shell_cmd = " ".join(parts)

        logger.info("Spawning Codex task (background=%s): %.120s...", background, prompt)

        try:
            env = _get_codex_env()

            if background:
                # Run detached so it doesn't block the pipeline
                proc = await asyncio.create_subprocess_shell(
                    shell_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=self.cwd,
                    env=env,
                )
                # Don't wait for completion — just log that it started
                logger.info("Codex task spawned (pid=%s)", proc.pid)
                # Give it a moment to start, then return
                await asyncio.sleep(2)
                return f"codex-pid-{proc.pid}"
            else:
                proc = await asyncio.create_subprocess_shell(
                    shell_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=self.cwd,
                    env=env,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
                output = stdout.decode(errors="replace").strip()
                logger.info("Codex task completed (exit %s)", proc.returncode)
                return output

        except asyncio.TimeoutError:
            logger.warning("Codex task timed out")
            return None
        except Exception:
            logger.exception("Failed to spawn Codex task")
            return None

    async def get_status(self, job_id: str | None = None) -> dict[str, Any]:
        """Get status of a Codex job or all jobs."""
        if not await self.is_available():
            return {"error": "codex unavailable"}

        cmd = ["node", _COMPANION_SCRIPT, "status", "--cwd", self.cwd, "--json"]
        if job_id:
            cmd.extend(["--job-id", job_id])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            return json.loads(stdout.decode(errors="replace"))
        except Exception:
            logger.debug("Failed to get Codex status", exc_info=True)
            return {"error": "status check failed"}

    async def get_result(self, job_id: str) -> str | None:
        """Get the rendered result of a completed Codex job."""
        if not await self.is_available():
            return None

        cmd = [
            "node", _COMPANION_SCRIPT, "result",
            "--cwd", self.cwd,
            "--job-id", job_id,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            return stdout.decode(errors="replace").strip()
        except Exception:
            logger.debug("Failed to get Codex result", exc_info=True)
            return None

    async def poll_result(
        self,
        job_id: str,
        timeout: float = 600,
        poll_interval: float = 15,
    ) -> str | None:
        """Poll for a background Codex job to complete, then return the result."""
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            status = await self.get_status(job_id)
            state = status.get("status", "unknown")

            if state in ("completed", "failed", "cancelled"):
                if state == "completed":
                    return await self.get_result(job_id)
                logger.warning("Codex job %s ended with status: %s", job_id, state)
                return None

            await asyncio.sleep(poll_interval)

        logger.warning("Codex job %s timed out after %.0fs", job_id, timeout)
        return None

    async def diagnose(
        self,
        context: str,
        *,
        log_path: str | None = None,
        background: bool = True,
    ) -> str | None:
        """Convenience: spawn a diagnosis task with optional log file context."""
        prompt_parts = [context]
        if log_path and Path(log_path).exists():
            prompt_parts.append(f"\n\nRelevant log file: {log_path}")
        return await self.spawn_task("\n".join(prompt_parts), background=background)

    async def review_code(
        self,
        description: str,
        *,
        background: bool = True,
    ) -> str | None:
        """Run a code review via Codex."""
        if not await self.is_available():
            return None

        cmd = [
            "node", _COMPANION_SCRIPT, "review",
            "--cwd", self.cwd,
        ]
        if background:
            cmd.append("--background")
        cmd.append(description)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.cwd,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
            return stdout.decode(errors="replace").strip()
        except Exception:
            logger.exception("Codex review failed")
            return None
