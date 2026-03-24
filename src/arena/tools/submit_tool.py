"""Submit tool for the agent arena.

Marks the episode as complete and returns the agent's final answer
for reward computation.
"""

from __future__ import annotations

from src.arena.tools.common import ToolResult


async def submit(
    container_id: str,
    answer: str,
) -> ToolResult:
    """Submit a final answer for the current episode.

    Args:
        container_id: The container this episode is running in.
        answer: The agent's final answer / solution.

    Returns:
        ToolResult with the submitted answer and a completion flag in metadata.
    """
    if not answer.strip():
        return ToolResult(
            stdout="",
            stderr="Cannot submit an empty answer.",
            exit_code=1,
            truncated=False,
        )

    return ToolResult(
        stdout=answer,
        stderr="",
        exit_code=0,
        truncated=False,
        metadata={
            "episode_complete": True,
            "container_id": container_id,
        },
    )
