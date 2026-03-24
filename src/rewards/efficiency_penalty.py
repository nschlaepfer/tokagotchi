"""Local efficiency penalty -- no Opus calls required.

Detects wasteful patterns in an agent trajectory and returns a penalty
value in [0.0, 0.3].  Higher values indicate more waste.
"""

from __future__ import annotations

import logging
from collections import Counter

from src.models import ActionType, Trajectory

logger = logging.getLogger(__name__)

# Penalty constants (per occurrence)
REPEATED_ACTION_PENALTY = 0.05
REDUNDANT_READ_PENALTY = 0.03
UNNECESSARY_THINK_PENALTY = 0.02
SUBOPTIMAL_TOOL_PENALTY = 0.02
MAX_PENALTY = 0.3


def compute_efficiency_penalty(trajectory: Trajectory) -> float:
    """Compute an efficiency penalty for a trajectory.

    Detects and penalizes the following wasteful patterns:

    * **Repeated identical actions** -- the same ``(action_type, content)`` pair
      appearing two or more times.  Penalty: 0.05 per repeat (after the first).
    * **Redundant file reads** -- reading the same file path multiple times.
      Penalty: 0.03 per redundant read (after the first).
    * **Unnecessary think actions** -- ``think`` steps that are not followed by
      a meaningful action (another ``think`` immediately after, or the
      trajectory ends).  Penalty: 0.02 per occurrence.
    * **Suboptimal tool choice** -- using ``bash`` for operations that have a
      dedicated tool (e.g. ``cat`` / ``head`` instead of ``read_file``).
      Penalty: 0.02 per occurrence.

    The total penalty is capped at 0.3.

    Parameters
    ----------
    trajectory:
        The completed agent trajectory to evaluate.

    Returns
    -------
    float
        Penalty in [0.0, 0.3].  Positive values indicate inefficiency.
    """
    total = 0.0
    details: dict[str, int] = {}

    repeated = _count_repeated_actions(trajectory)
    if repeated > 0:
        details["repeated_actions"] = repeated
        total += repeated * REPEATED_ACTION_PENALTY

    redundant = _count_redundant_reads(trajectory)
    if redundant > 0:
        details["redundant_reads"] = redundant
        total += redundant * REDUNDANT_READ_PENALTY

    unnecessary = _count_unnecessary_thinks(trajectory)
    if unnecessary > 0:
        details["unnecessary_thinks"] = unnecessary
        total += unnecessary * UNNECESSARY_THINK_PENALTY

    suboptimal = _count_suboptimal_tool_uses(trajectory)
    if suboptimal > 0:
        details["suboptimal_tool_uses"] = suboptimal
        total += suboptimal * SUBOPTIMAL_TOOL_PENALTY

    capped = min(total, MAX_PENALTY)

    if capped > 0:
        logger.debug(
            "Efficiency penalty %.3f (uncapped %.3f) for trajectory %s: %s",
            capped, total, trajectory.trajectory_id, details,
        )

    return round(capped, 4)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _count_repeated_actions(trajectory: Trajectory) -> int:
    """Count how many times (action_type, content) pairs repeat beyond the first."""
    action_counts: Counter[tuple[str, str]] = Counter()
    for step in trajectory.steps:
        key = (step.action_type.value, step.action_content.strip())
        action_counts[key] += 1

    # Each occurrence beyond the first is a repeat
    return sum(max(0, count - 1) for count in action_counts.values())


def _count_redundant_reads(trajectory: Trajectory) -> int:
    """Count redundant file reads (same file read more than once).

    Detects both ``read_file`` actions and ``bash`` actions that look like
    ``cat <file>`` or ``head <file>``.
    """
    files_read: Counter[str] = Counter()

    for step in trajectory.steps:
        if step.action_type == ActionType.READ_FILE:
            path = step.action_content.strip()
            files_read[path] += 1
        elif step.action_type == ActionType.BASH:
            path = _extract_read_path_from_bash(step.action_content)
            if path:
                files_read[path] += 1

    return sum(max(0, count - 1) for count in files_read.values())


def _extract_read_path_from_bash(content: str) -> str | None:
    """Try to extract a file path from a bash read-like command."""
    content = content.strip()
    for prefix in ("cat ", "head ", "tail ", "less ", "more "):
        if content.startswith(prefix):
            # Grab the first argument after the command
            parts = content[len(prefix):].strip().split()
            if parts:
                # Skip flags (start with -)
                for part in parts:
                    if not part.startswith("-"):
                        return part
    return None


def _count_unnecessary_thinks(trajectory: Trajectory) -> int:
    """Count think actions that don't lead to progress.

    A think step is considered unnecessary if:
    - It is immediately followed by another think step (thinking without acting).
    - It is the last step (thinking but never acting on it).
    """
    count = 0
    steps = trajectory.steps

    for i, step in enumerate(steps):
        if step.action_type != ActionType.THINK:
            continue

        is_last = i == len(steps) - 1
        next_is_think = (
            not is_last and steps[i + 1].action_type == ActionType.THINK
        )

        if is_last or next_is_think:
            count += 1

    return count


def _count_suboptimal_tool_uses(trajectory: Trajectory) -> int:
    """Count bash commands that should have used a dedicated tool.

    Patterns detected:
    - ``cat/head/tail <file>`` should use ``read_file``
    - ``echo "..." > <file>`` or ``printf ... > <file>`` should use ``write_file``
    - ``curl`` for API calls should use ``api_call``
    """
    count = 0

    for step in trajectory.steps:
        if step.action_type != ActionType.BASH:
            continue

        content = step.action_content.strip()

        # File reading via bash
        if _is_bash_file_read(content):
            count += 1
        # File writing via bash
        elif _is_bash_file_write(content):
            count += 1
        # API calls via bash curl/wget
        elif _is_bash_api_call(content):
            count += 1

    return count


def _is_bash_file_read(content: str) -> bool:
    """Check if a bash command is essentially a file read."""
    for cmd in ("cat ", "head ", "tail "):
        if content.startswith(cmd):
            # Exclude piped commands -- those may be legitimate
            if "|" not in content and ">" not in content:
                return True
    return False


def _is_bash_file_write(content: str) -> bool:
    """Check if a bash command is essentially a file write."""
    for pattern in ("echo ", "printf "):
        if content.startswith(pattern) and ">" in content:
            return True
    return False


def _is_bash_api_call(content: str) -> bool:
    """Check if a bash command is essentially an API call."""
    return content.startswith(("curl ", "wget ")) and "|" not in content
