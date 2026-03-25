"""DSPy Module wrapping the agent as a ReAct program.

This is the "student" program that GEPA optimizes. It uses DSPy's
built-in ReAct module with our arena tools, so GEPA can evolve the
instructions, prompts, and few-shot examples that control the agent.
"""

from __future__ import annotations

import dspy

from src.loop1_gepa.dspy_tools import build_dspy_tools


class AgentProgram(dspy.Module):
    """A coding agent wrapped as a DSPy Module for GEPA optimization.

    Uses ``dspy.ReAct`` with arena tools (bash, python, read_file,
    write_file, submit_answer). GEPA evolves the instruction prefix
    and few-shot demos to improve task completion.

    Parameters
    ----------
    tools : list, optional
        List of tool callables. If None, builds the default arena tools.
    max_iters : int
        Maximum ReAct reasoning steps per task.
    """

    def __init__(
        self,
        tools: list | None = None,
        max_iters: int = 20,
    ) -> None:
        super().__init__()
        self.tools = tools or build_dspy_tools()
        self.react = dspy.ReAct(
            "task_description -> solution",
            tools=self.tools,
            max_iters=max_iters,
        )

    def forward(self, task_description: str) -> dspy.Prediction:
        """Run the agent on a task and return the solution.

        Parameters
        ----------
        task_description : str
            Natural language description of the coding task.

        Returns
        -------
        dspy.Prediction
            Contains ``solution`` (the agent's final answer) and
            any intermediate ReAct trajectory state.
        """
        return self.react(task_description=task_description)
