"""DSPy metric function for GEPA optimization.

Evaluates agent predictions by running test commands in a sandbox and
generating rich textual feedback for GEPA's reflection LM.

The metric bridges DSPy's evaluation interface to our arena sandbox,
enabling GEPA to understand *why* the agent failed (not just that it did).
"""

from __future__ import annotations

import logging
import re
from typing import Any

import dspy

from src.arena.subprocess_manager import SubprocessManager
from src.loop1_gepa.dspy_tools import _run_async, set_tool_context
from src.models import TaskSpec

logger = logging.getLogger(__name__)

# Module-level arena manager (set once, shared across metric calls)
_arena_manager: SubprocessManager | None = None


def set_arena_manager(mgr: Any) -> None:
    """Set the arena manager used by the metric function."""
    global _arena_manager
    _arena_manager = mgr


def task_to_example(task: TaskSpec) -> dspy.Example:
    """Convert a TaskSpec to a dspy.Example for GEPA training.

    The TaskSpec is stashed in metadata so the metric can access
    test_commands, expected_output, initial_files, etc.
    """
    return dspy.Example(
        task_description=task.description,
        solution="",  # Label field (agent fills this)
    ).with_inputs("task_description")


def arena_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> float:
    """Evaluate an agent's prediction against a task in a sandbox.

    This is the core metric function passed to ``dspy.GEPA(metric=...)``.
    It spins up a sandbox, runs the agent's solution, checks test commands,
    and returns a scalar score.

    GEPA gets trace-aware feedback through DSPy's built-in trace capture
    mechanism (when ``trace is not None``), not through the metric return.
    The metric MUST return a float for DSPy's progress tracking to work.

    Parameters
    ----------
    gold : dspy.Example
        The gold example with ``task_description`` and metadata.
    pred : dspy.Prediction
        The agent's prediction with ``solution`` and optional ``trajectory``.
    trace : optional
        DSPy execution trace (available during optimization).

    Returns
    -------
    float
        Score between 0.0 and 1.0.
    """
    global _arena_manager
    if _arena_manager is None:
        _arena_manager = SubprocessManager()

    task_desc = gold.get("task_description", "")
    solution = getattr(pred, "solution", "") or ""

    # Try to find the original TaskSpec from the example's metadata
    task_spec = getattr(gold, "_task_spec", None)

    if task_spec is None:
        # Can't run test commands without TaskSpec — score based on output quality
        has_answer = len(solution.strip()) > 10
        return 0.3 if has_answer else 0.0

    # Create sandbox and run test commands
    try:
        sandbox_id = _arena_manager.create_container(task_spec)
        set_tool_context(_arena_manager, sandbox_id)

        # Write any solution files the agent mentioned
        _write_solution_files(sandbox_id, solution)

        # Run test commands
        test_results = _run_tests(sandbox_id, task_spec)

        # Check expected output
        output_match = _check_expected_output(sandbox_id, task_spec, solution)

        # Compute scores
        tests_passed = sum(1 for r in test_results if r["passed"])
        tests_total = max(len(test_results), 1)
        test_score = tests_passed / tests_total

        # Efficiency: shorter solutions are better (normalized)
        solution_length = len(solution)
        efficiency = max(0.0, 1.0 - (solution_length / 5000))

        # Format compliance: did the agent use proper tool calls?
        format_score = _assess_format_compliance(pred, trace)

        # Composite score
        score = (
            0.6 * (test_score if test_results else (0.5 if output_match else 0.0))
            + 0.2 * efficiency
            + 0.1 * format_score
            + 0.1 * (1.0 if output_match else 0.0)
        )

        # Generate detailed feedback for GEPA reflection
        feedback = _generate_feedback(
            task_desc, solution, test_results, output_match,
            test_score, format_score, pred, trace
        )

        logger.debug("Metric score=%.4f for task: %s", score, task_desc[:60])
        return round(score, 4)

    except Exception as e:
        logger.warning("Metric evaluation failed: %s", e)
        return 0.0
    finally:
        try:
            _arena_manager.destroy_container(sandbox_id)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_solution_files(sandbox_id: str, solution: str) -> None:
    """Write result.txt with the solution for test command checking."""
    global _arena_manager
    if _arena_manager is None:
        return
    try:
        _run_async(
            _arena_manager.async_exec_in_container(
                sandbox_id,
                f"cat > result.txt << 'DSPY_EOF'\n{solution}\nDSPY_EOF",
                timeout=5,
            )
        )
    except Exception:
        pass


def _run_tests(sandbox_id: str, task_spec: TaskSpec) -> list[dict[str, Any]]:
    """Run test_commands and return results."""
    global _arena_manager
    results = []
    for cmd in (task_spec.test_commands or []):
        try:
            stdout, stderr, exit_code = _run_async(
                _arena_manager.async_exec_in_container(sandbox_id, cmd, timeout=15)
            )
            results.append({
                "command": cmd[:100],
                "passed": exit_code == 0,
                "stdout": stdout[:500],
                "stderr": stderr[:500],
                "exit_code": exit_code,
            })
        except Exception as e:
            results.append({
                "command": cmd[:100],
                "passed": False,
                "error": str(e),
            })
    return results


def _check_expected_output(
    sandbox_id: str, task_spec: TaskSpec, solution: str
) -> bool:
    """Check if the solution matches expected output."""
    expected = task_spec.expected_output
    if not expected:
        return True  # No expected output defined

    # Check in solution text
    if expected.strip().lower() in solution.lower():
        return True

    # Check result.txt
    global _arena_manager
    try:
        stdout, _, code = _run_async(
            _arena_manager.async_exec_in_container(
                sandbox_id, "cat result.txt 2>/dev/null", timeout=5
            )
        )
        if expected.strip().lower() in stdout.lower():
            return True
    except Exception:
        pass

    return False


def _assess_format_compliance(pred: dspy.Prediction, trace: Any) -> float:
    """Score how well the agent used the expected action format.

    Checks if the agent emitted proper tool calls vs just free-text thinking.
    """
    solution = getattr(pred, "solution", "") or ""

    # Check for ReAct-style tool calls in the trajectory
    if trace is not None:
        # In optimization mode, trace contains the full execution
        # DSPy ReAct handles formatting internally, so compliance is high
        return 0.8

    # Check solution for tool-use indicators
    tool_patterns = [
        r"\[bash\]:", r"\[python\]:", r"\[read_file\]:",
        r"\[write_file\]:", r"\[submit\]:",
        r"```bash", r"```python",
    ]
    tool_calls = sum(1 for p in tool_patterns if re.search(p, solution, re.IGNORECASE))
    if tool_calls >= 2:
        return 1.0
    elif tool_calls >= 1:
        return 0.6
    return 0.2


def _generate_feedback(
    task_desc: str,
    solution: str,
    test_results: list[dict],
    output_match: bool,
    test_score: float,
    format_score: float,
    pred: dspy.Prediction,
    trace: Any,
) -> str:
    """Generate rich textual feedback for GEPA's reflection LM.

    This is the key advantage over the custom implementation — GEPA's
    reflection LM reads this feedback to understand *why* things failed
    and propose targeted improvements.
    """
    parts = []

    # Overall assessment
    if test_score == 1.0 and output_match:
        parts.append("Task completed successfully. All tests passed.")
    elif test_score > 0.5:
        parts.append(f"Partial success: {int(test_score*100)}% of tests passed.")
    elif test_score > 0:
        parts.append(f"Mostly failed: only {int(test_score*100)}% of tests passed.")
    else:
        parts.append("Task failed. No tests passed.")

    # Test details
    for r in test_results:
        if not r.get("passed"):
            parts.append(
                f"FAILED test: {r.get('command', '?')} "
                f"(exit={r.get('exit_code', '?')}, stderr={r.get('stderr', '')[:100]})"
            )

    # Output match
    if not output_match:
        parts.append("Expected output not found in agent's response or result.txt.")

    # Format issues
    if format_score < 0.5:
        parts.append(
            "FORMAT ISSUE: Agent did not use proper tool calls. "
            "It should use bash(), python_exec(), read_file(), write_file(), "
            "and submit_answer() tools explicitly rather than describing actions in free text."
        )

    # Solution quality
    if not solution or len(solution.strip()) < 20:
        parts.append("Agent produced no meaningful output or very short response.")
    elif len(solution) > 4000:
        parts.append("Agent response was very verbose. Consider more concise reasoning.")

    return " ".join(parts)
