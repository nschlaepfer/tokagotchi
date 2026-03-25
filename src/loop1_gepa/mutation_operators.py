"""Opus-driven mutation operators for prompt genome evolution.

Each mutation is proposed by Claude Opus based on failure analysis of
evaluation results, then applied to produce a new PromptGenome.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict
from enum import Enum
from typing import Any

from src.models import EvalResult, PromptGenome
from src.orchestrator.opus_client import OpusClient

logger = logging.getLogger(__name__)


class MutationOperator(str, Enum):
    """The set of possible mutation operations on a PromptGenome."""

    REPHRASE_SECTION = "rephrase_section"
    ADD_EXAMPLE = "add_example"
    REMOVE_EXAMPLE = "remove_example"
    SWAP_EXAMPLE_ORDER = "swap_example_order"
    STRENGTHEN_INSTRUCTION = "strengthen_instruction"
    ADD_COT_STEP = "add_cot_step"
    REMOVE_COT_STEP = "remove_cot_step"
    MODIFY_TOOL_INSTRUCTIONS = "modify_tool_instructions"
    ADD_ERROR_RECOVERY = "add_error_recovery"


# ---------------------------------------------------------------------------
# JSON schema for Opus structured output
# ---------------------------------------------------------------------------

_MUTATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mutation_type": {
            "type": "string",
            "enum": [m.value for m in MutationOperator],
        },
        "diagnosis": {
            "type": "string",
            "description": "Root cause analysis of the failure patterns.",
        },
        "rationale": {
            "type": "string",
            "description": "Why this mutation type was chosen and what it should fix.",
        },
        "system_prompt": {"type": "string"},
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
        "cot_scaffold": {"type": "string"},
        "tool_instructions": {"type": "string"},
        "error_recovery_hints": {"type": "string"},
    },
    "required": ["mutation_type", "diagnosis", "rationale", "system_prompt"],
}


# ---------------------------------------------------------------------------
# Core mutation function
# ---------------------------------------------------------------------------


def _build_mutation_prompt(
    genome: PromptGenome,
    eval_result: EvalResult,
) -> str:
    """Construct the prompt sent to Opus for mutation proposal."""

    # Compute action format diagnostics across ALL trajectories
    total_steps = 0
    think_fallback_steps = 0
    successful_tool_calls = 0
    total_tool_calls = 0
    for traj in eval_result.trajectories:
        for s in traj.steps:
            total_steps += 1
            if s.action_type.value == "think":
                think_fallback_steps += 1
            else:
                total_tool_calls += 1
                if s.observation and "error" not in s.observation.lower()[:50]:
                    successful_tool_calls += 1

    format_diagnostics = {
        "total_steps": total_steps,
        "think_fallback_steps": think_fallback_steps,
        "think_fallback_pct": round(100 * think_fallback_steps / max(total_steps, 1), 1),
        "successful_tool_calls": successful_tool_calls,
        "total_tool_calls": total_tool_calls,
        "diagnosis": (
            "CRITICAL: Agent is not formatting tool calls correctly. "
            f"{think_fallback_steps}/{total_steps} steps ({round(100 * think_fallback_steps / max(total_steps, 1))}%) "
            "were unparseable and fell back to 'think'. The agent needs explicit "
            "instructions on the exact [action_type]: content format."
            if think_fallback_steps > total_steps * 0.3
            else "Action format is acceptable."
        ),
    }

    # Summarize failure trajectories for context
    failure_summaries: list[dict[str, Any]] = []
    for traj in eval_result.trajectories:
        if not traj.success:
            steps_summary = []
            for s in traj.steps[-5:]:  # Last 5 steps of failures
                steps_summary.append({
                    "step": s.step_idx,
                    "action": s.action_type.value,
                    "content_preview": s.action_content[:200],
                    "observation_preview": s.observation[:200],
                })
            failure_summaries.append({
                "trajectory_id": traj.trajectory_id,
                "task_description": traj.task.description if traj.task else "",
                "num_steps": traj.num_steps,
                "total_reward": traj.total_reward,
                "actions_used": list(traj.action_types_used),
                "final_steps": steps_summary,
            })

    # Cap failure summaries to avoid token overflow
    if len(failure_summaries) > 5:
        failure_summaries = failure_summaries[:5]

    genome_fields = {
        "system_prompt": genome.system_prompt,
        "few_shot_examples": genome.few_shot_examples,
        "cot_scaffold": genome.cot_scaffold,
        "tool_instructions": genome.tool_instructions,
        "error_recovery_hints": genome.error_recovery_hints,
    }

    scores = {
        "success_rate": eval_result.success_rate,
        "avg_steps": eval_result.avg_steps,
        "tool_accuracy": eval_result.tool_accuracy,
        "code_quality": eval_result.code_quality,
        "tasks_run": eval_result.tasks_run,
    }

    mutation_types_desc = "\n".join(
        f"- {m.value}: {_MUTATION_DESCRIPTIONS[m]}" for m in MutationOperator
    )

    return f"""\
You are an expert prompt engineer performing targeted mutation on an AI agent's \
prompt genome. Your goal is to diagnose why the agent is failing and propose a \
single, focused mutation to improve performance.

## Current Genome
```json
{json.dumps(genome_fields, indent=2)}
```

## Evaluation Scores
```json
{json.dumps(scores, indent=2)}
```

## Action Format Diagnostics
```json
{json.dumps(format_diagnostics, indent=2)}
```

## Failure Patterns
{json.dumps(eval_result.failure_patterns, indent=2)}

## Failed Trajectory Samples
```json
{json.dumps(failure_summaries, indent=2, default=str)}
```

## Available Mutation Types
{mutation_types_desc}

## CRITICAL INSTRUCTIONS
1. Read the Action Format Diagnostics FIRST. If think_fallback_pct > 30%, the \
agent is failing to format tool calls. Use modify_tool_instructions or \
add_example to fix this — do NOT use rephrase_section for format issues.
2. Diagnose the root cause from trajectories and failure patterns.
3. Select the MOST IMPACTFUL mutation type. DO NOT default to rephrase_section \
unless rephrasing is genuinely the best fix. Consider:
   - add_example: if the agent lacks concrete examples of correct behavior
   - modify_tool_instructions: if tool calls are malformed or missing
   - strengthen_instruction: if the agent ignores important constraints
   - add_error_recovery: if the agent fails to recover from errors
   - add_cot_step: if the agent skips important reasoning steps
4. Apply the mutation to ALL genome fields. Return the complete genome.
5. If success_rate < 0.3, focus on the single biggest failure mode, not general improvements.
6. The agent uses [bash]: command, [python]: code, [read_file]: path, \
[write_file]: path\\ncontent, [submit]: answer format. Make sure tool_instructions \
specify this EXACTLY.

Return a JSON object with: mutation_type, diagnosis, rationale, system_prompt, \
few_shot_examples, cot_scaffold, tool_instructions, error_recovery_hints."""


_MUTATION_DESCRIPTIONS: dict[MutationOperator, str] = {
    MutationOperator.REPHRASE_SECTION: (
        "Rephrase a section of the system prompt for clarity or emphasis."
    ),
    MutationOperator.ADD_EXAMPLE: (
        "Add a new few-shot example demonstrating correct behavior for a failure pattern."
    ),
    MutationOperator.REMOVE_EXAMPLE: (
        "Remove a few-shot example that may be misleading or irrelevant."
    ),
    MutationOperator.SWAP_EXAMPLE_ORDER: (
        "Reorder few-shot examples to put the most relevant ones first."
    ),
    MutationOperator.STRENGTHEN_INSTRUCTION: (
        "Make an instruction more explicit or emphatic to prevent a recurring mistake."
    ),
    MutationOperator.ADD_COT_STEP: (
        "Add a chain-of-thought reasoning step to the scaffold."
    ),
    MutationOperator.REMOVE_COT_STEP: (
        "Remove an unnecessary chain-of-thought step to reduce verbosity."
    ),
    MutationOperator.MODIFY_TOOL_INSTRUCTIONS: (
        "Refine tool usage instructions to correct misuse patterns."
    ),
    MutationOperator.ADD_ERROR_RECOVERY: (
        "Add or improve error recovery guidance for a specific failure mode."
    ),
}


async def propose_mutation(
    opus_client: OpusClient,
    genome: PromptGenome,
    eval_result: EvalResult,
) -> tuple[MutationOperator, PromptGenome]:
    """Use Opus to diagnose failures and propose a targeted genome mutation.

    Sends the current genome and failure traces to Opus, which picks the
    best mutation type and produces the mutated genome.

    Parameters
    ----------
    opus_client:
        The Opus CLI client for making queries.
    genome:
        The current prompt genome to mutate.
    eval_result:
        Evaluation results containing trajectories and failure patterns.

    Returns
    -------
    tuple[MutationOperator, PromptGenome]
        The chosen mutation type and the new mutated genome.
        On failure, returns (REPHRASE_SECTION, original_genome).
    """
    prompt = _build_mutation_prompt(genome, eval_result)

    response = await opus_client.query(
        prompt,
        json_schema=_MUTATION_RESPONSE_SCHEMA,
        max_budget_usd=0.50,
        max_turns=1,
    )

    if response.is_error:
        logger.error(
            "Mutation proposal failed for genome %s: %s",
            genome.genome_id,
            response.error_message,
        )
        return MutationOperator.REPHRASE_SECTION, genome

    data = response.raw_json

    # Parse mutation type
    mutation_type_str = data.get("mutation_type", MutationOperator.REPHRASE_SECTION.value)
    try:
        mutation_type = MutationOperator(mutation_type_str)
    except ValueError:
        logger.warning(
            "Unknown mutation type '%s', falling back to REPHRASE_SECTION",
            mutation_type_str,
        )
        mutation_type = MutationOperator.REPHRASE_SECTION

    logger.info(
        "Opus proposed mutation %s for genome %s: %s",
        mutation_type.value,
        genome.genome_id,
        data.get("rationale", "")[:120],
    )

    # Build the mutated genome
    mutated = PromptGenome(
        genome_id=uuid.uuid4().hex[:8],
        system_prompt=data.get("system_prompt", genome.system_prompt),
        few_shot_examples=data.get("few_shot_examples", genome.few_shot_examples),
        cot_scaffold=data.get("cot_scaffold", genome.cot_scaffold),
        tool_instructions=data.get("tool_instructions", genome.tool_instructions),
        error_recovery_hints=data.get("error_recovery_hints", genome.error_recovery_hints),
        generation=genome.generation + 1,
        parent_ids=[genome.genome_id],
        scores={},
    )

    return mutation_type, mutated
