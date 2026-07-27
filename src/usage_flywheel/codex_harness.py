"""Codex CLI harness wrapper used by the product-use flywheel."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


_STDIN_PROMPT_THRESHOLD = 4000


@dataclass
class CodexHarnessResult:
    """Result from one non-interactive Codex harness run."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    last_message: str
    duration_seconds: float
    output_path: str = ""
    events: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and bool(self.last_message.strip() or self.stdout.strip())


class CodexHarness:
    """Run ``codex exec`` with clean final-message capture.

    This intentionally wraps the installed Codex CLI instead of vendoring the
    entire Codex repo. The CLI is the stable harness boundary; full vendoring can
    still be added later if tokagotchi needs to patch Codex internals.
    """

    def __init__(
        self,
        cwd: str | Path = ".",
        *,
        model: str = "gpt-5.6-sol",
        effort: str = "medium",
        sandbox: str = "read-only",
        timeout_seconds: float = 300.0,
    ) -> None:
        self.cwd = Path(cwd).resolve()
        self.model = model
        self.effort = effort
        self.sandbox = sandbox
        self.timeout_seconds = timeout_seconds

    def build_command(self, prompt: str, *, output_path: str | Path) -> list[str]:
        """Build the Codex exec command for one task."""

        prompt_arg = "-" if self._use_stdin(prompt) else prompt
        return [
            "codex",
            "exec",
            "--model",
            self.model,
            "-c",
            f'model_reasoning_effort="{self.effort}"',
            "--sandbox",
            self.sandbox,
            "--json",
            "--output-last-message",
            str(output_path),
            prompt_arg,
        ]

    async def run(self, prompt: str) -> CodexHarnessResult:
        """Run one Codex task and capture stdout plus final assistant text."""

        fd, output_name = tempfile.mkstemp(prefix="tokagotchi-codex-", suffix=".txt")
        os.close(fd)
        output_path = Path(output_name)
        cmd = self.build_command(prompt, output_path=output_path)
        stdin_payload = prompt.encode("utf-8") if self._use_stdin(prompt) else None
        start = time.monotonic()

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if stdin_payload is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.cwd),
        )

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(input=stdin_payload),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            stdout_b = b""
            stderr_b = f"Codex timed out after {self.timeout_seconds:.0f}s".encode()

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        try:
            last_message = output_path.read_text(encoding="utf-8").strip()
        except OSError:
            last_message = ""

        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass

        return CodexHarnessResult(
            command=cmd,
            returncode=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout.strip(),
            stderr=stderr.strip(),
            last_message=last_message,
            duration_seconds=time.monotonic() - start,
            output_path=str(output_path),
            events=[line for line in stdout.splitlines() if line.strip()],
        )

    @staticmethod
    def _use_stdin(prompt: str) -> bool:
        return len(prompt) > _STDIN_PROMPT_THRESHOLD
