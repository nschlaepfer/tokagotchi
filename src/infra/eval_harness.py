"""Structured evaluation harness for benchmarking prompt genomes.

Runs tasks in the agent arena, collects trajectories, computes rewards,
and provides regression testing between evaluation snapshots.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.models import EvalResult, PromptGenome, TaskSpec, TaskType, Trajectory

logger = logging.getLogger(__name__)

# Regression threshold: fail if any metric drops more than 5%
REGRESSION_THRESHOLD = 0.05


class EvalHarness:
    """Runs structured evaluations and regression checks.

    Provides methods to benchmark a prompt genome against a set of tasks,
    compare before/after results, and persist evaluation artefacts.
    """

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    async def run_evaluation(
        self,
        tasks: list[TaskSpec],
        vllm_server: Any,
        arena_manager: Any,
        genome: PromptGenome,
    ) -> EvalResult:
        """Evaluate a genome against a set of tasks.

        Parameters
        ----------
        tasks:
            Task specifications to run.
        vllm_server:
            Running VLLMServer instance for Qwen inference.
        arena_manager:
            DockerManager for arena containers.
        genome:
            The prompt genome to evaluate.

        Returns
        -------
        EvalResult
            Structured results with full trajectory traces.
        """
        from src.arena.game import AgentArenaGame

        logger.info(
            "Starting evaluation: %d tasks, genome=%s",
            len(tasks),
            genome.genome_id,
        )
        start = time.monotonic()

        trajectories: list[Trajectory] = []
        successes = 0
        total_steps = 0
        correct_tool_uses = 0
        total_tool_uses = 0
        failure_patterns: dict[str, int] = {}

        for task in tasks:
            try:
                game = AgentArenaGame(
                    arena_mgr=arena_manager,
                )
                async with game:
                    await game.reset(task)
                    system_message = genome.to_system_message()

                    # Run the agent through the task
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": task.description},
                    ]

                    trajectory = Trajectory(
                        task=task,
                        model_id="qwen",
                        prompt_genome_id=genome.genome_id,
                    )

                    step_start = time.monotonic()
                    done = False
                    step_idx = 0

                    while not done and step_idx < 20:
                        # Get model response
                        response = await vllm_server.chat_completion(
                            messages=messages,
                            temperature=0.2,
                            max_tokens=2048,
                        )
                        action_text = response.choices[0].message.content or ""

                        # Execute in arena
                        step_result = await game.step(action_text)

                        from src.models import ActionType, StepRecord

                        record = StepRecord(
                            step_idx=step_idx,
                            action_type=ActionType.BASH,
                            action_content=action_text,
                            observation=step_result.observation,
                            reward=step_result.reward,
                        )
                        trajectory.steps.append(record)
                        total_tool_uses += 1

                        messages.append({"role": "assistant", "content": action_text})
                        messages.append({"role": "user", "content": step_result.observation})

                        done = step_result.done
                        step_idx += 1

                    trajectory.wall_time_seconds = time.monotonic() - step_start
                    trajectory.success = done and step_result.reward > 0
                    trajectory.total_reward = sum(s.reward for s in trajectory.steps)

                    if trajectory.success:
                        successes += 1
                        correct_tool_uses += step_idx
                    else:
                        pattern = _classify_failure(trajectory)
                        failure_patterns[pattern] = failure_patterns.get(pattern, 0) + 1

                    total_steps += trajectory.num_steps
                    trajectories.append(trajectory)

            except Exception as exc:
                logger.error("Evaluation failed for task %s: %s", task.task_id, exc)
                failure_patterns["eval_error"] = failure_patterns.get("eval_error", 0) + 1

        n_tasks = len(tasks)
        elapsed = time.monotonic() - start

        result = EvalResult(
            genome_id=genome.genome_id,
            tasks_run=n_tasks,
            success_rate=successes / n_tasks if n_tasks > 0 else 0.0,
            avg_steps=total_steps / n_tasks if n_tasks > 0 else 0.0,
            tool_accuracy=correct_tool_uses / total_tool_uses if total_tool_uses > 0 else 0.0,
            code_quality=_estimate_code_quality(trajectories),
            trajectories=trajectories,
            failure_patterns=failure_patterns,
        )

        logger.info(
            "Evaluation complete (%.1fs): success=%.2f avg_steps=%.1f tasks=%d",
            elapsed,
            result.success_rate,
            result.avg_steps,
            n_tasks,
        )

        return result

    # ------------------------------------------------------------------
    # Regression check
    # ------------------------------------------------------------------

    def run_regression_check(
        self,
        before: EvalResult,
        after: EvalResult,
    ) -> tuple[bool, dict[str, Any]]:
        """Compare before/after evaluation results for regression.

        Parameters
        ----------
        before:
            Baseline evaluation result.
        after:
            New evaluation result to check against the baseline.

        Returns
        -------
        tuple[bool, dict]
            ``(passed, details)`` where ``passed`` is False if any metric
            regressed more than 5%.
        """
        details: dict[str, Any] = {}
        passed = True

        before_vec = before.score_vector
        after_vec = after.score_vector

        for metric in ("success_rate", "tool_accuracy", "code_quality"):
            bval = before_vec.get(metric, 0.0)
            aval = after_vec.get(metric, 0.0)

            if bval > 0:
                change = (aval - bval) / bval
            else:
                change = 0.0

            details[metric] = {
                "before": round(bval, 4),
                "after": round(aval, 4),
                "change_pct": round(change * 100, 2),
            }

            if change < -REGRESSION_THRESHOLD:
                passed = False
                details[metric]["regressed"] = True

        # avg_steps: lower is better, so regression means increase
        b_steps = before_vec.get("avg_steps", 0.0)
        a_steps = after_vec.get("avg_steps", 0.0)
        if b_steps > 0:
            step_change = (a_steps - b_steps) / b_steps
        else:
            step_change = 0.0

        details["avg_steps"] = {
            "before": round(b_steps, 2),
            "after": round(a_steps, 2),
            "change_pct": round(step_change * 100, 2),
        }
        if step_change > REGRESSION_THRESHOLD:
            passed = False
            details["avg_steps"]["regressed"] = True

        details["passed"] = passed
        return passed, details

    # ------------------------------------------------------------------
    # Benchmark loading
    # ------------------------------------------------------------------

    def load_benchmark_tasks(self, path: str) -> list[TaskSpec]:
        """Load held-out evaluation tasks from a JSON file.

        Parameters
        ----------
        path:
            Path to a JSON file containing a list of task specifications.

        Returns
        -------
        list[TaskSpec]
            Parsed task specifications.
        """
        task_path = Path(path)
        if not task_path.exists():
            logger.warning("Benchmark tasks file not found: %s", path)
            return []

        try:
            with open(task_path) as f:
                raw_tasks = json.load(f)

            tasks: list[TaskSpec] = []
            for raw in raw_tasks:
                task_type_str = raw.get("task_type", "code_debugging")
                try:
                    task_type = TaskType(task_type_str)
                except ValueError:
                    task_type = TaskType.CODE_DEBUGGING

                tasks.append(TaskSpec(
                    task_id=raw.get("task_id", ""),
                    task_type=task_type,
                    description=raw.get("description", ""),
                    initial_files=raw.get("initial_files", {}),
                    test_commands=raw.get("test_commands", []),
                    expected_output=raw.get("expected_output"),
                    difficulty=raw.get("difficulty", 0.5),
                    metadata=raw.get("metadata", {}),
                ))

            logger.info("Loaded %d benchmark tasks from %s", len(tasks), path)
            return tasks

        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load benchmark tasks from %s: %s", path, exc)
            return []

    # ------------------------------------------------------------------
    # Results persistence
    # ------------------------------------------------------------------

    def save_results(self, result: EvalResult, path: str) -> None:
        """Save evaluation results to a JSON file.

        Parameters
        ----------
        result:
            The evaluation result to save.
        path:
            File path for the output JSON.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "genome_id": result.genome_id,
            "tasks_run": result.tasks_run,
            "score_vector": result.score_vector,
            "failure_patterns": result.failure_patterns,
            "trajectories": [
                {
                    "trajectory_id": t.trajectory_id,
                    "task_id": t.task.task_id if t.task else None,
                    "task_description": t.task.description if t.task else None,
                    "success": t.success,
                    "num_steps": t.num_steps,
                    "total_reward": t.total_reward,
                    "wall_time_seconds": t.wall_time_seconds,
                    "action_types_used": list(t.action_types_used),
                    "steps": [
                        {
                            "step_idx": s.step_idx,
                            "action_type": s.action_type.value,
                            "action_content": s.action_content[:500],
                            "observation": s.observation[:500],
                            "reasoning": s.reasoning[:300],
                            "reward": s.reward,
                        }
                        for s in t.steps
                    ],
                }
                for t in result.trajectories
            ],
        }

        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(out_path)
        logger.info("Saved eval results to %s", path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify_failure(trajectory: Trajectory) -> str:
    """Heuristically classify the failure mode of a trajectory."""
    if not trajectory.steps:
        return "no_steps"

    actions = [s.action_content for s in trajectory.steps]

    # Detect loops: same action repeated 3+ times
    if len(actions) >= 3:
        for i in range(len(actions) - 2):
            if actions[i] == actions[i + 1] == actions[i + 2]:
                return "action_loop"

    # Detect timeout (hit max steps without submission)
    last_step = trajectory.steps[-1]
    if "submit" not in last_step.action_content.lower() and trajectory.num_steps >= 18:
        return "timeout"

    # Detect wrong output
    if trajectory.steps and trajectory.total_reward <= 0:
        return "incorrect_output"

    return "unknown"


def _estimate_code_quality(trajectories: list[Trajectory]) -> float:
    """Estimate code quality from trajectory patterns.

    Uses heuristics: shorter successful trajectories with fewer retries
    indicate better code quality.
    """
    if not trajectories:
        return 0.0

    scores: list[float] = []
    for traj in trajectories:
        if not traj.steps:
            scores.append(0.0)
            continue

        # Base score from success
        score = 0.6 if traj.success else 0.1

        # Bonus for efficiency (fewer steps)
        if traj.success and traj.num_steps > 0:
            efficiency_bonus = max(0.0, 0.4 * (1.0 - traj.num_steps / 20.0))
            score += efficiency_bonus

        scores.append(min(score, 1.0))

    return sum(scores) / len(scores)
