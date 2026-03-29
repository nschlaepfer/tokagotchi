"""Async wrapper around Claude Code headless mode for programmatic Opus 4.6 calls.

Uses ``claude -p`` CLI subprocess with structured JSON output to drive
trace analysis, prompt mutation, task generation, and trajectory rating.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.config import OpusConfig
from src.models import (
    PromptGenome,
    TaskSpec,
    TaskType,
    TraceAnalysis,
    Trajectory,
)
from src.orchestrator.budget_tracker import BudgetTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Response container
# ---------------------------------------------------------------------------


@dataclass
class OpusResponse:
    """Parsed response from a single Opus CLI call."""

    raw_json: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    session_id: str = ""
    duration_seconds: float = 0.0
    is_error: bool = False
    error_message: str = ""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OpusClient:
    """Async wrapper around the ``claude`` CLI for structured Opus queries.

    Parameters
    ----------
    config:
        Opus-specific configuration (budget limits, model, session dir, etc.).
    budget_tracker:
        Shared budget tracker instance for cost accounting.
    max_retries:
        Number of retries on transient failures (timeout, non-zero exit).
    base_backoff:
        Initial backoff in seconds for exponential retry delay.
    """

    def __init__(
        self,
        config: OpusConfig | None = None,
        budget_tracker: BudgetTracker | None = None,
        max_retries: int = 3,
        base_backoff: float = 2.0,
    ) -> None:
        self.config = config or OpusConfig()
        self.budget = budget_tracker or BudgetTracker(
            hourly_limit_usd=self.config.hourly_budget_usd,
            daily_limit_usd=self.config.daily_budget_usd,
            persist_path=Path(self.config.session_dir) / "budget_state.json",
        )
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self._call_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core CLI invocation
    # ------------------------------------------------------------------

    async def query(
        self,
        prompt: str,
        *,
        tools: list[str] | None = None,
        json_schema: dict[str, Any] | None = None,
        max_budget_usd: float | None = None,
        session_id: str | None = None,
        max_turns: int | None = None,
    ) -> OpusResponse:
        """Send a prompt to Opus via ``claude -p`` and return a parsed response.

        Parameters
        ----------
        prompt:
            The user-turn text to send.
        tools:
            Optional list of allowed tool names (maps to ``--allowedTools``).
        json_schema:
            If provided, passed via ``--json-schema`` to constrain output.
        max_budget_usd:
            Per-call budget cap (``--max-budget-usd``). Falls back to config default.
        session_id:
            Session ID for ``--resume`` to continue a conversation.
        max_turns:
            Maximum agentic turns (``--max-turns``).
        """
        budget = max_budget_usd or self.config.default_max_budget_per_call_usd
        turns = max_turns or self.config.default_max_turns

        # Pre-flight budget check
        if not self.budget.can_spend(budget, loop_id="query"):
            return OpusResponse(
                is_error=True,
                error_message="Budget pre-check failed: would exceed limits",
            )

        cmd, stdin_text = self._build_command(
            prompt=prompt,
            tools=tools,
            json_schema=json_schema,
            max_budget_usd=budget,
            session_id=session_id,
            max_turns=turns,
        )

        response = await self._run_with_retries(cmd, stdin_text=stdin_text, budget_usd=budget, loop_id="query")
        return response

    # ------------------------------------------------------------------
    # High-level methods
    # ------------------------------------------------------------------

    async def analyze_trace(self, trace: Trajectory) -> TraceAnalysis:
        """Send a full trajectory to Opus for diagnosis and analysis.

        Returns a :class:`TraceAnalysis` populated from Opus's structured output.
        """
        schema = {
            "type": "object",
            "properties": {
                "trajectory_id": {"type": "string"},
                "failure_step": {"type": ["integer", "null"]},
                "diagnosis": {"type": "string"},
                "corrected_steps": {
                    "type": "array",
                    "items": {"type": "object"},
                },
                "step_ratings": {
                    "type": "array",
                    "items": {"type": "object"},
                },
                "overall_assessment": {"type": "string"},
                "suggested_mutation": {"type": "string"},
            },
            "required": ["trajectory_id", "diagnosis", "overall_assessment"],
        }

        prompt = (
            "Analyze the following agent trajectory. Identify the first failure step "
            "(if any), diagnose root causes, rate each step, and suggest how to improve "
            "the prompt or strategy.\n\n"
            f"```json\n{json.dumps(asdict(trace), default=str)}\n```"
        )

        resp = await self.query(prompt, json_schema=schema, max_budget_usd=0.50)
        if resp.is_error:
            logger.error("analyze_trace failed: %s", resp.error_message)
            return TraceAnalysis(trajectory_id=trace.trajectory_id)

        data = resp.raw_json
        return TraceAnalysis(
            trajectory_id=data.get("trajectory_id", trace.trajectory_id),
            failure_step=data.get("failure_step"),
            diagnosis=data.get("diagnosis", ""),
            corrected_steps=data.get("corrected_steps", []),
            step_ratings=data.get("step_ratings", []),
            overall_assessment=data.get("overall_assessment", ""),
            suggested_mutation=data.get("suggested_mutation", ""),
        )

    async def mutate_prompt(
        self,
        genome: PromptGenome,
        traces: list[Trajectory],
    ) -> PromptGenome:
        """Ask Opus to propose a mutation to a prompt genome based on traces.

        Returns a new :class:`PromptGenome` with the proposed changes applied.
        """
        schema = {
            "type": "object",
            "properties": {
                "system_prompt": {"type": "string"},
                "cot_scaffold": {"type": "string"},
                "tool_instructions": {"type": "string"},
                "error_recovery_hints": {"type": "string"},
                "few_shot_examples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "user": {"type": "string"},
                            "assistant": {"type": "string"},
                        },
                    },
                },
                "mutation_rationale": {"type": "string"},
            },
            "required": ["system_prompt", "mutation_rationale"],
        }

        trace_summaries = []
        for t in traces:
            trace_summaries.append({
                "id": t.trajectory_id,
                "success": t.success,
                "num_steps": t.num_steps,
                "total_reward": t.total_reward,
                "action_types": list(t.action_types_used),
            })

        prompt = (
            "You are an expert prompt engineer. Given the current prompt genome and "
            "recent trajectory results, propose a targeted mutation to improve agent "
            "performance. Preserve what works; change what fails.\n\n"
            f"## Current Genome\n```json\n{json.dumps(asdict(genome), default=str)}\n```\n\n"
            f"## Recent Traces\n```json\n{json.dumps(trace_summaries, default=str)}\n```\n\n"
            "Return the full mutated fields plus a rationale."
        )

        resp = await self.query(prompt, json_schema=schema, max_budget_usd=0.50)
        if resp.is_error:
            logger.error("mutate_prompt failed: %s", resp.error_message)
            return genome

        data = resp.raw_json
        mutated = PromptGenome(
            genome_id=uuid.uuid4().hex[:8],
            system_prompt=data.get("system_prompt", genome.system_prompt),
            few_shot_examples=data.get("few_shot_examples", genome.few_shot_examples),
            cot_scaffold=data.get("cot_scaffold", genome.cot_scaffold),
            tool_instructions=data.get("tool_instructions", genome.tool_instructions),
            error_recovery_hints=data.get("error_recovery_hints", genome.error_recovery_hints),
            generation=genome.generation + 1,
            parent_ids=[genome.genome_id],
        )
        return mutated

    async def generate_task(
        self,
        capability_profile: dict[str, Any],
        task_type: str = "code_debugging",
    ) -> TaskSpec:
        """Ask Opus to create a new evaluation task targeting agent weaknesses.

        Parameters
        ----------
        capability_profile:
            Dict summarizing current agent strengths/weaknesses (e.g. scores by type).
        task_type:
            One of the :class:`TaskType` values as a string.
        """
        schema = {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "initial_files": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "test_commands": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "expected_output": {"type": ["string", "null"]},
                "difficulty": {"type": "number"},
            },
            "required": ["description", "difficulty"],
        }

        prompt = (
            "Generate a new evaluation task for an AI coding agent. The task should "
            "target the agent's weak areas based on the capability profile below.\n\n"
            f"Task type: {task_type}\n\n"
            f"## Capability Profile\n```json\n{json.dumps(capability_profile, default=str)}\n```\n\n"
            "Return a complete task specification."
        )

        resp = await self.query(prompt, json_schema=schema, max_budget_usd=0.30)
        if resp.is_error:
            logger.error("generate_task failed: %s", resp.error_message)
            return TaskSpec(task_type=TaskType(task_type), description="(generation failed)")

        data = resp.raw_json
        return TaskSpec(
            task_type=TaskType(task_type),
            description=data.get("description", ""),
            initial_files=data.get("initial_files", {}),
            test_commands=data.get("test_commands", []),
            expected_output=data.get("expected_output"),
            difficulty=data.get("difficulty", 0.5),
        )

    async def rate_trajectory(self, trajectory: Trajectory) -> list[dict[str, Any]]:
        """Ask Opus to assign a quality rating to each step in a trajectory.

        Returns a list of dicts, one per step, with at least ``step_idx`` and
        ``rating`` (float 0-1) keys.
        """
        schema = {
            "type": "object",
            "properties": {
                "ratings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_idx": {"type": "integer"},
                            "rating": {"type": "number"},
                            "comment": {"type": "string"},
                        },
                        "required": ["step_idx", "rating"],
                    },
                },
            },
            "required": ["ratings"],
        }

        prompt = (
            "Rate each step of the following agent trajectory on a 0.0-1.0 scale. "
            "Consider correctness, efficiency, and whether the step moved toward "
            "the goal.\n\n"
            f"```json\n{json.dumps(asdict(trajectory), default=str)}\n```"
        )

        resp = await self.query(prompt, json_schema=schema, max_budget_usd=0.30)
        if resp.is_error:
            logger.error("rate_trajectory failed: %s", resp.error_message)
            return []

        return resp.raw_json.get("ratings", [])

    async def correct_trace(self, trajectory: Trajectory) -> TraceAnalysis:
        """Perform trace surgery: identify and correct faulty steps.

        Similar to :meth:`analyze_trace` but focuses on producing concrete
        corrected step content that can be used as training signal.
        """
        schema = {
            "type": "object",
            "properties": {
                "trajectory_id": {"type": "string"},
                "failure_step": {"type": ["integer", "null"]},
                "diagnosis": {"type": "string"},
                "corrected_steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_idx": {"type": "integer"},
                            "action_type": {"type": "string"},
                            "action_content": {"type": "string"},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["step_idx", "action_type", "action_content"],
                    },
                },
                "overall_assessment": {"type": "string"},
            },
            "required": ["trajectory_id", "corrected_steps", "overall_assessment"],
        }

        prompt = (
            "You are performing trace surgery on an agent trajectory. Identify every "
            "step that was incorrect or sub-optimal and provide a corrected version. "
            "Focus on producing actionable corrections, not just commentary.\n\n"
            f"```json\n{json.dumps(asdict(trajectory), default=str)}\n```"
        )

        resp = await self.query(prompt, json_schema=schema, max_budget_usd=0.50)
        if resp.is_error:
            logger.error("correct_trace failed: %s", resp.error_message)
            return TraceAnalysis(trajectory_id=trajectory.trajectory_id)

        data = resp.raw_json
        return TraceAnalysis(
            trajectory_id=data.get("trajectory_id", trajectory.trajectory_id),
            failure_step=data.get("failure_step"),
            diagnosis=data.get("diagnosis", ""),
            corrected_steps=data.get("corrected_steps", []),
            overall_assessment=data.get("overall_assessment", ""),
        )

    # ------------------------------------------------------------------
    # CLI command builder
    # ------------------------------------------------------------------

    def _build_command(
        self,
        prompt: str,
        *,
        tools: list[str] | None = None,
        json_schema: dict[str, Any] | None = None,
        max_budget_usd: float = 0.50,
        session_id: str | None = None,
        max_turns: int = 10,
    ) -> tuple[list[str], str | None]:
        """Build the ``claude`` CLI argument list.

        Returns (cmd, stdin_text) — if the prompt is long (>4000 chars),
        it's piped via stdin to avoid Windows command-line length limits.
        """
        stdin_text: str | None = None

        cli_cmd = _resolve_claude_cli()  # returns a list like ["node", "cli.js"]

        # Windows has ~8191 char command-line limit; use stdin for safety at 4000
        if len(prompt) > 4000:
            cmd = [
                *cli_cmd,
                "-p", "-",  # read from stdin
                "--output-format", "json",
                "--max-turns", str(max_turns),
                "--max-budget-usd", f"{max_budget_usd:.2f}",
            ]
            stdin_text = prompt
        else:
            cmd = [
                *cli_cmd,
                "-p", prompt,
                "--output-format", "json",
                "--max-turns", str(max_turns),
                "--max-budget-usd", f"{max_budget_usd:.2f}",
            ]

        # Restrict tool usage: by default no tools (pure text generation)
        # to prevent the model from trying to write files or use agents
        if tools:
            cmd.extend(["--allowedTools", json.dumps(tools)])
        else:
            cmd.extend(["--allowedTools", ""])

        if json_schema:
            cmd.extend(["--json-schema", json.dumps(json_schema)])

        if session_id:
            cmd.extend(["--resume", session_id])

        return cmd, stdin_text

    # ------------------------------------------------------------------
    # Subprocess runner with retries
    # ------------------------------------------------------------------

    async def _run_with_retries(
        self,
        cmd: list[str],
        *,
        stdin_text: str | None = None,
        budget_usd: float,
        loop_id: str,
    ) -> OpusResponse:
        """Execute *cmd* with exponential-backoff retries on failure."""
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            start = time.monotonic()
            try:
                response = await self._execute(cmd, stdin_text=stdin_text)
                response.duration_seconds = time.monotonic() - start

                # Record cost
                if not response.is_error and response.cost_usd > 0:
                    self.budget.record_spend(
                        amount_usd=response.cost_usd,
                        loop_id=loop_id,
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        metadata={"cmd_preview": " ".join(cmd[:6])},
                    )

                self._log_call(cmd, response, attempt)
                return response

            except asyncio.TimeoutError:
                last_error = f"Timeout on attempt {attempt}"
                logger.warning(last_error)
            except Exception as exc:  # noqa: BLE001
                last_error = f"Attempt {attempt} failed: {exc}"
                logger.warning(last_error)

            if attempt < self.max_retries:
                backoff = self.base_backoff * (2 ** (attempt - 1))
                logger.info("Retrying in %.1fs...", backoff)
                await asyncio.sleep(backoff)

        return OpusResponse(is_error=True, error_message=f"All retries exhausted. Last: {last_error}")

    async def _execute(self, cmd: list[str], *, stdin_text: str | None = None) -> OpusResponse:
        """Run a single ``claude`` CLI invocation and parse its output."""
        timeout = 300  # 5 minutes

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if stdin_text else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdin_bytes = stdin_text.encode("utf-8") if stdin_text else None
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=stdin_bytes), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise

        stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            return OpusResponse(
                is_error=True,
                error_message=f"CLI exited with code {proc.returncode}: {stderr or stdout}",
            )

        # Parse the JSON output from claude --output-format json
        return self._parse_cli_output(stdout)

    def _parse_cli_output(self, stdout: str) -> OpusResponse:
        """Parse the JSON blob emitted by ``claude -p --output-format json``."""
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            return OpusResponse(
                is_error=True,
                error_message=f"Failed to parse CLI JSON output: {stdout[:500]}",
            )

        # The CLI JSON envelope typically has 'result', 'cost_usd', 'session_id', etc.
        # Extract the structured result (may itself be JSON if --json-schema was used).
        result = data.get("result", data)
        text = ""
        raw_json: dict[str, Any] = {}

        if isinstance(result, str):
            text = result
            # Attempt to parse if it looks like JSON
            try:
                raw_json = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                raw_json = {}
        elif isinstance(result, dict):
            raw_json = result
            text = json.dumps(result)
        else:
            text = str(result)

        # Cost: CLI uses "total_cost_usd" (or fallback "cost_usd")
        cost = data.get("total_cost_usd", data.get("cost_usd", 0.0))

        # Usage: CLI uses "input_tokens" / "output_tokens" in usage dict
        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        completion_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))

        # Check for CLI-level error
        is_error = data.get("is_error", False)
        error_message = ""
        if is_error:
            error_message = data.get("error", data.get("result", "Unknown CLI error"))
            if isinstance(error_message, dict):
                error_message = json.dumps(error_message)

        return OpusResponse(
            raw_json=raw_json,
            text=text,
            cost_usd=float(cost) if cost else 0.0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            session_id=data.get("session_id", ""),
            is_error=is_error,
            error_message=str(error_message) if is_error else "",
        )

    # ------------------------------------------------------------------
    # Debug logging
    # ------------------------------------------------------------------

    def _log_call(self, cmd: list[str], response: OpusResponse, attempt: int) -> None:
        """Append a record to the in-memory call log for debugging."""
        entry = {
            "timestamp": time.time(),
            "cmd_preview": " ".join(cmd[:6]),
            "attempt": attempt,
            "cost_usd": response.cost_usd,
            "duration_s": response.duration_seconds,
            "is_error": response.is_error,
            "error_message": response.error_message,
            "session_id": response.session_id,
        }
        self._call_log.append(entry)
        logger.debug("Opus call: %s", json.dumps(entry))

    @property
    def call_log(self) -> list[dict[str, Any]]:
        """Return a copy of the call log for inspection."""
        return list(self._call_log)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_CLAUDE_CLI_CMD: list[str] | None = None


def _resolve_claude_cli() -> list[str]:
    """Find the ``claude`` CLI and return the command list to invoke it.

    On Windows, ``.cmd`` shims hang when invoked via ``create_subprocess_exec``,
    so we bypass the wrapper and call ``node cli.js`` directly.

    Returns a list like ``["node", "/path/to/cli.js"]`` or ``["claude"]``.
    """
    global _CLAUDE_CLI_CMD
    if _CLAUDE_CLI_CMD is not None:
        return _CLAUDE_CLI_CMD

    import os
    import shutil
    import sys

    if sys.platform == "win32":
        # On Windows, find the actual Node.js entry point to avoid .CMD wrapper issues
        npm_dir = os.path.expandvars(r"%APPDATA%\npm")
        cli_js = os.path.join(npm_dir, "node_modules", "@anthropic-ai", "claude-code", "cli.js")

        if os.path.isfile(cli_js):
            node = shutil.which("node") or "node"
            _CLAUDE_CLI_CMD = [node, cli_js]
            return _CLAUDE_CLI_CMD

        # Fallback: try the .cmd via shell
        cmd_path = os.path.join(npm_dir, "claude.cmd")
        if os.path.isfile(cmd_path):
            _CLAUDE_CLI_CMD = [cmd_path]
            return _CLAUDE_CLI_CMD

    # Non-Windows or fallback
    found = shutil.which("claude")
    _CLAUDE_CLI_CMD = [found] if found else ["claude"]
    return _CLAUDE_CLI_CMD
