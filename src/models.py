"""Shared data models used across all modules."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    INFO_GATHERING = "info_gathering"
    CODE_DEBUGGING = "code_debugging"
    API_ORCHESTRATION = "api_orchestration"
    OPEN_ENDED = "open_ended_optimization"


class ActionType(str, Enum):
    THINK = "think"
    BASH = "bash"
    PYTHON = "python"
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    SQL = "sql"
    API_CALL = "api_call"
    SUBMIT = "submit"


@dataclass
class TaskSpec:
    """A task for the agent arena."""
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_type: TaskType = TaskType.CODE_DEBUGGING
    description: str = ""
    initial_files: dict[str, str] = field(default_factory=dict)
    test_commands: list[str] = field(default_factory=list)
    expected_output: str | None = None
    difficulty: float = 0.5  # 0.0 = trivial, 1.0 = very hard
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepRecord:
    """A single step in an agent trajectory."""
    step_idx: int
    action_type: ActionType
    action_content: str
    observation: str
    reasoning: str = ""
    reward: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """A full agent trajectory (episode)."""
    trajectory_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task: TaskSpec | None = None
    steps: list[StepRecord] = field(default_factory=list)
    success: bool = False
    total_reward: float = 0.0
    wall_time_seconds: float = 0.0
    model_id: str = ""
    prompt_genome_id: str = ""

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def action_types_used(self) -> set[str]:
        return {s.action_type.value for s in self.steps}


@dataclass
class TraceAnalysis:
    """Opus's analysis of a trajectory."""
    trajectory_id: str = ""
    failure_step: int | None = None
    diagnosis: str = ""
    corrected_steps: list[dict[str, str]] = field(default_factory=list)
    step_ratings: list[dict[str, float]] = field(default_factory=list)
    overall_assessment: str = ""
    suggested_mutation: str = ""


@dataclass
class PromptGenome:
    """A prompt configuration as an evolvable genome."""
    genome_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    system_prompt: str = ""
    few_shot_examples: list[dict[str, str]] = field(default_factory=list)
    cot_scaffold: str = ""
    tool_instructions: str = ""
    error_recovery_hints: str = ""
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    # Mutation lineage — filled when genome is created via mutation
    mutation_type: str = ""          # e.g. "add_example", "modify_tool_instructions"
    mutation_diagnosis: str = ""     # Opus's analysis of why mutation was needed
    mutation_rationale: str = ""     # Opus's explanation of the change
    created_at: str = ""             # ISO timestamp of genome creation

    def to_system_message(self) -> str:
        parts = [self.system_prompt]
        if self.tool_instructions:
            parts.append(f"\n## Tool Usage\n{self.tool_instructions}")
        if self.cot_scaffold:
            parts.append(f"\n## Reasoning Approach\n{self.cot_scaffold}")
        if self.error_recovery_hints:
            parts.append(f"\n## Error Recovery\n{self.error_recovery_hints}")
        if self.few_shot_examples:
            parts.append("\n## Examples")
            for ex in self.few_shot_examples:
                parts.append(f"\nUser: {ex.get('user', '')}\nAssistant: {ex.get('assistant', '')}")
        return "\n".join(parts)


@dataclass
class EvalResult:
    """Structured evaluation result with full traces."""
    genome_id: str = ""
    tasks_run: int = 0
    success_rate: float = 0.0
    avg_steps: float = 0.0
    tool_accuracy: float = 0.0
    code_quality: float = 0.0
    trajectories: list[Trajectory] = field(default_factory=list)
    failure_patterns: dict[str, int] = field(default_factory=dict)

    @property
    def score_vector(self) -> dict[str, float]:
        return {
            "success_rate": self.success_rate,
            "avg_steps": self.avg_steps,
            "tool_accuracy": self.tool_accuracy,
            "code_quality": self.code_quality,
        }


@dataclass
class RewardResult:
    """Composite reward for a trajectory."""
    outcome_reward: float = 0.0
    process_reward: float = 0.0
    efficiency_penalty: float = 0.0
    composite: float = 0.0
    per_step_rewards: list[float] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
