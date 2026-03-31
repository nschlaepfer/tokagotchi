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
        """Check if Codex CLI is installed and the companion script exists."""
        if self._available is not None:
            return self._available

        if _COMPANION_SCRIPT is None:
            logger.info("Codex companion script not found — Codex integration disabled")
            self._available = False
            return False

        try:
            proc = await asyncio.create_subprocess_exec(
                "node", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            self._available = proc.returncode == 0
        except FileNotFoundError:
            self._available = False

        if not self._available:
            logger.info("Node.js not available — Codex integration disabled")
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

        cmd = [
            "node", _COMPANION_SCRIPT, "task",
            "--cwd", self.cwd,
        ]

        if background:
            cmd.append("--background")

        _model = model or self.model
        if _model:
            cmd.extend(["--model", _model])

        _effort = effort or self.effort
        if _effort:
            cmd.extend(["--effort", _effort])

        if write if write is not None else self.write:
            cmd.append("--write")

        if fresh:
            cmd.append("--fresh")

        cmd.append(prompt)

        logger.info("Spawning Codex task (background=%s): %.120s...", background, prompt)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.cwd,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
            output = stdout.decode(errors="replace").strip()

            if background:
                # Parse job ID from output
                for line in output.splitlines():
                    line = line.strip()
                    if line.startswith("job:") or line.startswith("Job "):
                        # Extract job ID
                        parts = line.split()
                        for part in parts:
                            if len(part) > 8 and part.replace("-", "").isalnum():
                                logger.info("Codex background task spawned: %s", part)
                                return part
                # Fallback: return raw output
                logger.info("Codex task output: %s", output[:200])
                return output
            else:
                return output

        except asyncio.TimeoutError:
            logger.warning("Codex task spawn timed out")
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
