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

    # ------------------------------------------------------------------
    # Domain-specific gates (Meta-Harness trace-aware)
    # ------------------------------------------------------------------

    async def review_training_data(
        self,
        examples_sample: list[dict[str, Any]],
        buffer_stats: dict[str, Any],
        trace_context: str = "",
    ) -> str | None:
        """Review training data quality before SFT. Non-blocking."""
        prompt = (
            "You are reviewing training data for a self-improving coding agent (Qwen 3.5 9B).\n\n"
            f"## Buffer Stats\n{json.dumps(buffer_stats, indent=2, default=str)}\n\n"
            f"## Sample Examples ({len(examples_sample)} of buffer)\n"
        )
        for i, ex in enumerate(examples_sample[:10]):
            meta = ex.get("metadata", {})
            msgs = ex.get("example", {}).get("messages", [])
            prompt += f"\n### Example {i+1} (type={meta.get('task_type','?')}, source={meta.get('source','?')})\n"
            for m in msgs[-3:]:  # Last 3 messages only
                role = m.get("role", "?")
                content = str(m.get("content", ""))[:200]
                prompt += f"  [{role}]: {content}...\n"
        if trace_context:
            prompt += f"\n## Recent Training History\n{trace_context}\n"
        prompt += (
            "\n## Task\n"
            "Rate each example 1-5 (5=high quality). Flag any with:\n"
            "- Obviously wrong corrections\n"
            "- Truncated or incomplete traces\n"
            "- Degenerate patterns (repeated actions, empty observations)\n"
            "- Task-specific memorization (won't generalize)\n"
            "Return JSON: {flagged_indices: int[], quality_summary: str, avg_score: float}\n"
        )
        result = await self.spawn_task(prompt, background=True)
        self._log_review("training_data", {"stats": buffer_stats, "n_samples": len(examples_sample)})
        return result

    async def generate_curriculum(
        self,
        capability_profile: dict[str, Any],
        failure_patterns: dict[str, int],
        trace_context: str = "",
        n_tasks: int = 5,
    ) -> str | None:
        """Generate new curriculum tasks targeting weak areas. Non-blocking."""
        prompt = (
            "You are a curriculum designer for a self-improving coding agent.\n\n"
            f"## Current Capability Profile\n{json.dumps(capability_profile, indent=2, default=str)}\n\n"
            f"## Recent Failure Patterns\n{json.dumps(failure_patterns, indent=2, default=str)}\n\n"
        )
        if trace_context:
            prompt += f"## Historical Trace Context\n{trace_context}\n\n"
        prompt += (
            f"## Task\n"
            f"Generate {n_tasks} new coding tasks that:\n"
            "1. Target weak areas (success rate < 50%)\n"
            "2. Stay in the 0.3-0.7 difficulty band (learnable, not impossible)\n"
            "3. Bridge from strong areas to weak ones gradually\n"
            "4. Include diverse task types\n\n"
            "Return JSON array of tasks, each with:\n"
            '{"task_id": "gen_xxx", "task_type": "code_debugging|api_orchestration|info_gathering|open_ended_optimization",\n'
            ' "description": "...", "initial_files": {"filename": "content"}, "test_commands": ["..."],\n'
            ' "expected_output": "..." or null, "difficulty": 0.3-0.7}\n'
        )
        result = await self.spawn_task(prompt, background=True)
        self._log_review("curriculum_gen", {"profile": capability_profile, "n_requested": n_tasks})
        return result

    async def diagnose_failures(
        self,
        failure_trajectories: list[dict[str, Any]],
        failure_patterns: dict[str, int],
        genome_summary: dict[str, Any],
        trace_context: str = "",
    ) -> str | None:
        """Analyze failure patterns to produce compressed diagnostics for Opus. Non-blocking."""
        prompt = (
            "You are diagnosing failures in a coding agent's evaluation runs.\n\n"
            f"## Failure Pattern Distribution\n{json.dumps(failure_patterns, indent=2, default=str)}\n\n"
            f"## Current Genome Summary\n{json.dumps(genome_summary, indent=2, default=str)}\n\n"
            f"## Failed Trajectory Summaries ({len(failure_trajectories)} trajectories)\n"
        )
        for i, traj in enumerate(failure_trajectories[:10]):
            prompt += f"\n### Trajectory {i+1}: {traj.get('task_desc', '?')[:150]}\n"
            prompt += f"  Steps: {traj.get('num_steps', '?')}, Actions: {traj.get('actions_used', [])}\n"
            for step in traj.get("last_steps", [])[-3:]:
                prompt += f"  [{step.get('action','')}] {str(step.get('content',''))[:80]} -> {str(step.get('obs',''))[:80]}\n"
        if trace_context:
            prompt += f"\n## Historical Context\n{trace_context}\n"
        prompt += (
            "\n## Task\n"
            "Produce a structured diagnosis:\n"
            "1. Top 3 root causes (with evidence from trajectories)\n"
            "2. For each: recommended mutation type (add_example, modify_tool_instructions, strengthen_instruction, add_cot_step, add_error_recovery)\n"
            "3. One-paragraph genome weakness summary\n"
            "Return JSON: {root_causes: [{cause, evidence, mutation_type, priority}], genome_weakness: str}\n"
        )
        result = await self.spawn_task(prompt, background=True)
        self._log_review("failure_diagnosis", {"n_trajectories": len(failure_trajectories), "patterns": failure_patterns})
        return result

    async def review_mutation(
        self,
        original_genome: dict[str, Any],
        mutated_genome: dict[str, Any],
        mutation_metadata: dict[str, Any],
        trace_context: str = "",
    ) -> str | None:
        """Review a proposed prompt mutation for obvious regressions. Non-blocking."""
        # Compute diffs for the prompt
        diffs = {}
        for key in ["system_prompt", "few_shot_examples", "tool_instructions", "error_recovery_hints"]:
            orig = str(original_genome.get(key, ""))[:500]
            mut = str(mutated_genome.get(key, ""))[:500]
            if orig != mut:
                diffs[key] = {"before": orig, "after": mut}

        if not diffs:
            return None  # No changes to review

        prompt = (
            "You are reviewing a prompt mutation for a coding agent.\n\n"
            f"## Mutation Type: {mutation_metadata.get('mutation_type', '?')}\n"
            f"## Diagnosis: {str(mutation_metadata.get('diagnosis', ''))[:300]}\n"
            f"## Rationale: {str(mutation_metadata.get('rationale', ''))[:300]}\n\n"
            f"## Changes\n{json.dumps(diffs, indent=2, default=str)}\n\n"
        )
        if trace_context:
            prompt += f"## Historical Context\n{trace_context}\n\n"
        prompt += (
            "## Task\n"
            "Assess this mutation:\n"
            "1. Does it address the diagnosed failure?\n"
            "2. Are there obvious regressions (removing critical instructions, contradictions)?\n"
            "3. Is the system prompt coherent after the change?\n"
            "Return JSON: {verdict: 'approve'|'flag'|'reject', issues: str[], confidence: float}\n"
        )
        result = await self.spawn_task(prompt, background=True)
        self._log_review("mutation_review", {"type": mutation_metadata.get("mutation_type"), "fields_changed": list(diffs.keys())})
        return result

    # ------------------------------------------------------------------
    # Review logging
    # ------------------------------------------------------------------

    def _log_review(self, category: str, data: dict[str, Any]) -> None:
        """Persist a Codex review record to disk for auditability."""
        try:
            review_dir = Path(self.cwd) / "data" / "codex_reviews"
            review_dir.mkdir(parents=True, exist_ok=True)
            path = review_dir / f"{category}_{time.strftime('%Y%m%d_%H%M%S')}.json"
            path.write_text(json.dumps(
                {"category": category, "timestamp": time.time(), **data},
                indent=2, default=str,
            ))
        except Exception:
            logger.debug("Failed to log Codex review", exc_info=True)
