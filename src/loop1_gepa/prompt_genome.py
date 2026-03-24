"""Utility functions for PromptGenome management.

Provides creation, serialization, deserialization, population I/O,
and crossover operations for prompt genomes.
"""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any

from src.models import PromptGenome


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def genome_to_dict(g: PromptGenome) -> dict[str, Any]:
    """Convert a PromptGenome to a JSON-serializable dict."""
    return {
        "genome_id": g.genome_id,
        "system_prompt": g.system_prompt,
        "few_shot_examples": g.few_shot_examples,
        "cot_scaffold": g.cot_scaffold,
        "tool_instructions": g.tool_instructions,
        "error_recovery_hints": g.error_recovery_hints,
        "generation": g.generation,
        "parent_ids": g.parent_ids,
        "scores": g.scores,
    }


def dict_to_genome(d: dict[str, Any]) -> PromptGenome:
    """Reconstruct a PromptGenome from a dict (inverse of genome_to_dict)."""
    return PromptGenome(
        genome_id=d.get("genome_id", uuid.uuid4().hex[:8]),
        system_prompt=d.get("system_prompt", ""),
        few_shot_examples=d.get("few_shot_examples", []),
        cot_scaffold=d.get("cot_scaffold", ""),
        tool_instructions=d.get("tool_instructions", ""),
        error_recovery_hints=d.get("error_recovery_hints", ""),
        generation=d.get("generation", 0),
        parent_ids=d.get("parent_ids", []),
        scores=d.get("scores", {}),
    )


# ---------------------------------------------------------------------------
# Population I/O
# ---------------------------------------------------------------------------


def save_population(population: list[PromptGenome], path: str | Path) -> None:
    """Persist a population of genomes to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": 1,
        "count": len(population),
        "genomes": [genome_to_dict(g) for g in population],
    }

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_population(path: str | Path) -> list[PromptGenome]:
    """Load a population of genomes from a JSON file."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return [dict_to_genome(d) for d in data.get("genomes", [])]


# ---------------------------------------------------------------------------
# Seed genome
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = """\
You are an expert coding agent. You solve tasks by reading files, writing code, \
running commands, and reasoning step by step. Always verify your work before \
submitting.

When given a task:
1. Understand the requirements fully before acting.
2. Explore the existing codebase and relevant files.
3. Plan your approach, then implement changes incrementally.
4. Test after every significant change.
5. If something fails, diagnose the root cause before retrying.
6. Submit your answer only when you are confident it is correct."""

_DEFAULT_COT_SCAFFOLD = """\
Before each action, think through:
- What information do I still need?
- What is the most direct path to the goal?
- What could go wrong with this action?
- How will I verify the result?"""

_DEFAULT_TOOL_INSTRUCTIONS = """\
Available tools:
- bash: Run shell commands. Use for exploring files, installing packages, running tests.
- python: Execute Python code. Use for data processing, complex logic, scripting.
- read_file: Read a file from the workspace. Prefer this for targeted reads.
- write_file: Write content to a file in the workspace.
- submit: Submit your final answer when the task is complete.

Guidelines:
- Prefer small, focused commands over long pipelines.
- Always check exit codes and output for errors.
- Use read_file before modifying a file to understand its contents.
- Run tests after making changes to verify correctness."""

_DEFAULT_ERROR_RECOVERY = """\
When you encounter an error:
- Read the full error message carefully, including stack traces.
- Identify whether it is a syntax error, runtime error, or logic error.
- Check recent changes that may have introduced the issue.
- If stuck after 2 failed attempts at the same approach, try an alternative strategy.
- Never repeat the exact same failing command without changes."""

_DEFAULT_FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "user": "Fix the failing test in tests/test_utils.py. The test_parse_config "
                "test expects a dict but gets None.",
        "assistant": "Let me start by reading the test file to understand the expected "
                     "behavior, then look at the implementation.\n\n"
                     "[read_file tests/test_utils.py]\n"
                     "I see the test expects parse_config('valid.yaml') to return a dict. "
                     "Let me check the implementation.\n\n"
                     "[read_file src/utils.py]\n"
                     "The function returns None when the file exists but is empty. "
                     "I need to return an empty dict instead.\n\n"
                     "[write_file src/utils.py with the fix]\n"
                     "[bash: python -m pytest tests/test_utils.py::test_parse_config -v]\n"
                     "Test passes. Submitting.\n\n"
                     "[submit: Fixed parse_config to return {} instead of None for empty files]",
    },
]


def create_seed_genome() -> PromptGenome:
    """Create a reasonable default coding agent genome."""
    return PromptGenome(
        genome_id=uuid.uuid4().hex[:8],
        system_prompt=_DEFAULT_SYSTEM_PROMPT,
        few_shot_examples=list(_DEFAULT_FEW_SHOT_EXAMPLES),
        cot_scaffold=_DEFAULT_COT_SCAFFOLD,
        tool_instructions=_DEFAULT_TOOL_INSTRUCTIONS,
        error_recovery_hints=_DEFAULT_ERROR_RECOVERY,
        generation=0,
        parent_ids=[],
        scores={},
    )


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------


def crossover(parent1: PromptGenome, parent2: PromptGenome) -> PromptGenome:
    """Produce a child genome by combining fields from two parents.

    For string fields, one parent's version is chosen at random per field.
    For few-shot examples, a random subset from both parents is merged
    (deduplicated by user prompt text, capped at the larger parent's count).
    """
    child_id = uuid.uuid4().hex[:8]
    new_generation = max(parent1.generation, parent2.generation) + 1

    # Per-field random selection
    system_prompt = random.choice([parent1.system_prompt, parent2.system_prompt])
    cot_scaffold = random.choice([parent1.cot_scaffold, parent2.cot_scaffold])
    tool_instructions = random.choice(
        [parent1.tool_instructions, parent2.tool_instructions]
    )
    error_recovery_hints = random.choice(
        [parent1.error_recovery_hints, parent2.error_recovery_hints]
    )

    # Merge few-shot examples: combine, deduplicate, sample
    seen_prompts: set[str] = set()
    merged_examples: list[dict[str, str]] = []
    all_examples = parent1.few_shot_examples + parent2.few_shot_examples
    random.shuffle(all_examples)

    max_examples = max(
        len(parent1.few_shot_examples),
        len(parent2.few_shot_examples),
        1,
    )

    for ex in all_examples:
        key = ex.get("user", "")
        if key not in seen_prompts:
            seen_prompts.add(key)
            merged_examples.append(ex)
        if len(merged_examples) >= max_examples:
            break

    # Merge scores by averaging shared objectives
    merged_scores: dict[str, float] = {}
    all_keys = set(parent1.scores) | set(parent2.scores)
    for k in all_keys:
        vals = []
        if k in parent1.scores:
            vals.append(parent1.scores[k])
        if k in parent2.scores:
            vals.append(parent2.scores[k])
        merged_scores[k] = sum(vals) / len(vals)

    return PromptGenome(
        genome_id=child_id,
        system_prompt=system_prompt,
        few_shot_examples=merged_examples,
        cot_scaffold=cot_scaffold,
        tool_instructions=tool_instructions,
        error_recovery_hints=error_recovery_hints,
        generation=new_generation,
        parent_ids=[parent1.genome_id, parent2.genome_id],
        scores=merged_scores,
    )
