"""Product-use flywheel orchestration."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import MasterConfig, load_config
from src.infra.ollama_utils import ollama_api_urls
from src.usage_flywheel.codex_harness import CodexHarness, CodexHarnessResult
from src.usage_flywheel.models import UsageTrace
from src.usage_flywheel.redaction import redact_text
from src.usage_flywheel.store import UsageTraceStore, append_pending_example


@dataclass
class StudentAttempt:
    """Local student attempt result."""

    status: str
    output: str = ""
    error: str = ""
    duration_seconds: float = 0.0
    raw: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok" and bool(self.output.strip())


@dataclass
class FlywheelResult:
    """Summary returned after one usage flywheel run."""

    trace: UsageTrace
    trace_path: Path
    pending_example_path: Path | None = None
    student_attempt: StudentAttempt | None = None
    codex_result: CodexHarnessResult | None = None


class UsageFlywheel:
    """Run the local usage trace -> Codex boost -> pending-example loop."""

    def __init__(
        self,
        config: MasterConfig | None = None,
        *,
        repo_root: str | Path = ".",
        trace_store: UsageTraceStore | None = None,
    ) -> None:
        self.config = config or load_config("config")
        self.repo_root = Path(repo_root).resolve()
        self.trace_store = trace_store or UsageTraceStore(self.config.usage_flywheel.trace_dir)

    async def run_task(
        self,
        user_task: str,
        *,
        dry_run: bool = False,
        skip_student: bool = False,
        codex_boost: str | None = None,
        write: bool = False,
        append_pending: bool = True,
    ) -> FlywheelResult:
        """Run one user task through the product-use flywheel.

        ``dry_run`` records the task and builds the same trace shape without
        external model calls. It is used by tests and setup checks.
        """

        redacted_task, redaction_report = redact_text(user_task)
        git_meta = _git_metadata(self.repo_root)
        trace = UsageTrace(
            user_task=redacted_task,
            repo_root=str(self.repo_root),
            git_branch=git_meta.get("branch", ""),
            git_commit=git_meta.get("commit", ""),
            git_dirty=git_meta.get("dirty", False),
            privacy_mode=self.config.usage_flywheel.privacy_mode,
            redaction_report=redaction_report.as_dict(),
            student_model=self.config.model.name,
            teacher_provider=self.config.opus.provider,
            teacher_model=self.config.opus.model,
            metadata={"dry_run": dry_run},
        )
        trace.add_event("user_task", redacted_task)

        student_attempt: StudentAttempt | None = None
        codex_result: CodexHarnessResult | None = None

        if dry_run:
            trace.status = "dry_run"
            trace.student_status = "skipped"
            trace.codex_status = "skipped"
            trace.add_event("dry_run", "No local model or Codex call was made.")
            trace_path = self.trace_store.save(trace)
            return FlywheelResult(trace=trace, trace_path=trace_path)

        if skip_student:
            trace.student_status = "skipped"
            trace.add_event("student_skipped", "Local student attempt was skipped by CLI option.")
        else:
            student_attempt = await self._run_student(user_task)
            trace.student_status = student_attempt.status
            trace.student_output = _redact_if_enabled(
                student_attempt.output or student_attempt.error,
                enabled=self.config.usage_flywheel.redact_secrets,
            )
            trace.add_event(
                "student_attempt",
                trace.student_output,
                status=student_attempt.status,
                duration_seconds=student_attempt.duration_seconds,
            )
            if not student_attempt.ok:
                trace.failure_mode = student_attempt.status

        boost_policy = (codex_boost or self.config.usage_flywheel.codex_boost).replace("-", "_")
        should_boost = boost_policy == "always" or (
            boost_policy == "on_failure" and not (student_attempt and student_attempt.ok)
        )

        if should_boost and self.config.opus.provider.lower() == "codex":
            codex_prompt = _build_codex_boost_prompt(
                user_task=user_task,
                student_output=student_attempt.output if student_attempt else "",
                student_status=trace.student_status,
            )
            harness = CodexHarness(
                cwd=self.repo_root,
                model=self.config.opus.model,
                effort=self.config.opus.model_reasoning_effort,
                sandbox="workspace-write" if write else self.config.usage_flywheel.codex_sandbox,
                timeout_seconds=self.config.usage_flywheel.codex_timeout_seconds,
            )
            codex_result = await harness.run(codex_prompt)
            trace.codex_status = "ok" if codex_result.ok else "error"
            trace.codex_output = _redact_if_enabled(
                codex_result.last_message or codex_result.stdout or codex_result.stderr,
                enabled=self.config.usage_flywheel.redact_secrets,
            )
            trace.boost_used = codex_result.ok
            trace.add_event(
                "codex_boost",
                trace.codex_output,
                status=trace.codex_status,
                duration_seconds=codex_result.duration_seconds,
                returncode=codex_result.returncode,
            )
        elif should_boost:
            trace.codex_status = "provider_not_codex"
            trace.add_event("codex_boost_skipped", "Configured teacher provider is not Codex.")
        else:
            trace.codex_status = "skipped"
            trace.add_event("codex_boost_skipped", f"Boost policy is {boost_policy}.")

        pending_path: Path | None = None
        selected_answer = trace.codex_output if trace.boost_used else trace.student_output
        if append_pending and selected_answer.strip():
            example, metadata = build_training_example(trace, selected_answer)
            pending_path = append_pending_example(
                self.config.usage_flywheel.pending_jsonl,
                example=example,
                metadata=metadata,
            )
            trace.pending_example_path = str(pending_path)
            trace.add_event("pending_example", str(pending_path), **metadata)

        trace.status = _final_status(trace)
        trace_path = self.trace_store.save(trace)
        return FlywheelResult(
            trace=trace,
            trace_path=trace_path,
            pending_example_path=pending_path,
            student_attempt=student_attempt,
            codex_result=codex_result,
        )

    async def _run_student(self, user_task: str) -> StudentAttempt:
        """Ask the local Ollama-served student to attempt the task once."""

        import aiohttp

        started = time.monotonic()
        payload = {
            "model": self.config.model.name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are tokagotchi, a local coding assistant. "
                        "Answer with the concrete next action or solution. "
                        "If you cannot complete the task, state the blocker."
                    ),
                },
                {"role": "user", "content": user_task},
            ],
            "stream": False,
            "think": True,
            "options": {
                "temperature": 0.2,
                "num_predict": self.config.usage_flywheel.max_student_tokens,
                "num_ctx": self.config.usage_flywheel.student_num_ctx,
            },
        }
        errors: list[str] = []
        try:
            timeout = aiohttp.ClientTimeout(total=self.config.usage_flywheel.student_timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for api_base in ollama_api_urls(self.config.model.ollama_host, self.config.model.ollama_port):
                    try:
                        async with session.post(f"{api_base}/chat", json=payload) as resp:
                            text = await resp.text()
                            if resp.status != 200:
                                return StudentAttempt(
                                    status=f"ollama_http_{resp.status}",
                                    error=text[:2000],
                                    duration_seconds=time.monotonic() - started,
                                )
                            data = json.loads(text)
                            break
                    except aiohttp.ClientConnectionError as exc:
                        errors.append(f"{api_base}: {exc}")
                else:
                    return StudentAttempt(
                        status="unavailable",
                        error="Could not reach Ollama. Tried: " + "; ".join(errors),
                        duration_seconds=time.monotonic() - started,
                    )
        except asyncio.TimeoutError:
            return StudentAttempt(status="timeout", error="Ollama student timed out.", duration_seconds=time.monotonic() - started)
        except Exception as exc:  # noqa: BLE001
            return StudentAttempt(status="unavailable", error=str(exc), duration_seconds=time.monotonic() - started)

        message = data.get("message", {})
        content = message.get("content", "") or message.get("thinking", "")
        return StudentAttempt(
            status="ok" if content.strip() else "empty",
            output=content,
            duration_seconds=time.monotonic() - started,
            raw=data,
        )


def build_training_example(trace: UsageTrace, selected_answer: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one chat-format training example from a completed usage trace."""

    system = (
        "You are tokagotchi, a local coding agent. Complete the user's task with "
        "clear, correct, minimal steps. Use local tools when available, preserve "
        "user data, and surface blockers explicitly."
    )
    example = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": trace.user_task},
            {"role": "assistant", "content": selected_answer},
        ]
    }
    metadata = {
        "source": "usage_flywheel",
        "trace_id": trace.trace_id,
        "task_type": "real_user_task",
        "failure_mode": trace.failure_mode or "none",
        "difficulty": 0.5,
        "student_model": trace.student_model,
        "student_status": trace.student_status,
        "teacher_provider": trace.teacher_provider,
        "teacher_model": trace.teacher_model,
        "boosted_by_codex": trace.boost_used,
        "privacy_mode": trace.privacy_mode,
        "created_at": trace.created_at,
    }
    return example, metadata


def _build_codex_boost_prompt(*, user_task: str, student_output: str, student_status: str) -> str:
    return (
        "You are the Codex teacher harness for tokagotchi.\n\n"
        "Goal: complete or repair the user's real local task so the resulting answer "
        "can become high-quality local supervision for a smaller student model.\n\n"
        "Requirements:\n"
        "- Be concrete and correct.\n"
        "- If the task needs file changes, inspect before editing and keep changes minimal.\n"
        "- If the task cannot be completed safely, state the blocker and the exact next check.\n"
        "- Do not include private credentials, tokens, or unrelated file content in the final answer.\n\n"
        f"Student status: {student_status}\n\n"
        f"User task:\n{user_task}\n\n"
        f"Student attempt:\n{student_output or '(no usable student attempt)'}\n"
    )


def _redact_if_enabled(text: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    redacted, _ = redact_text(text)
    return redacted


def _final_status(trace: UsageTrace) -> str:
    if trace.boost_used:
        return "boosted"
    if trace.student_status == "ok":
        return "student_ok"
    if trace.codex_status == "error":
        return "codex_error"
    return "recorded"


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def run_git(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except Exception:
            return ""
        return result.stdout.strip() if result.returncode == 0 else ""

    status = run_git("status", "--porcelain")
    return {
        "branch": run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": bool(status),
    }
