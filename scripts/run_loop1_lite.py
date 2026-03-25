"""Loop 1 Lite: GEPA prompt evolution without Docker.

Runs the core Opus-analyzes-Qwen-and-mutates-prompts loop using a
local OpenAI-compatible model server plus Claude CLI for Opus mutations.
No Docker arena needed — tasks are evaluated purely through generated
action plans (not executed).

Usage:
    python scripts/run_loop1_lite.py --iterations 5
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import openai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("loop1_lite")

# ---------------------------------------------------------------------------
# Claude CLI helper
# ---------------------------------------------------------------------------

def find_claude_cli() -> str:
    found = shutil.which("claude")
    if found:
        return found
    npm_path = os.path.expandvars(r"%APPDATA%\npm\claude.cmd")
    if os.path.exists(npm_path):
        return npm_path
    return "claude"

CLAUDE_BIN = find_claude_cli()
MODEL_NAME = "mlx-community/Qwen3-14B-4bit"
LOCAL_LLM_BASE = "http://127.0.0.1:8080/v1"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PromptGenome:
    genome_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    system_prompt: str = ""
    tool_instructions: str = ""
    cot_scaffold: str = ""
    generation: int = 0
    scores: dict = field(default_factory=dict)


@dataclass
class TaskResult:
    task_description: str = ""
    qwen_response: str = ""
    score: float = 0.0
    action_count: int = 0
    has_tool_calls: bool = False
    has_reasoning: bool = False
    has_submission: bool = False


# ---------------------------------------------------------------------------
# Evaluation: run Qwen on tasks and grade the response
# ---------------------------------------------------------------------------

EVAL_TASKS = [
    "There are 3 CSV files in /data/: sales_q1.csv, sales_q2.csv, sales_q3.csv. Each has columns: customer_id, amount, date. Find the customer with the highest total spend across all files.",
    "The Python file /repo/calculator.py has a bug: the divide function doesn't handle division by zero. The test file /repo/test_calculator.py has 4 tests, and test_divide_by_zero is failing. Fix the bug.",
    "Using the REST API at localhost:5000, GET /orders returns paginated results (100 per page). Find all orders from March 2024 that were shipped late (shipped_date > expected_date). Summarize patterns.",
    "The function in /code/sort.py uses bubble sort. Optimize it to run at least 3x faster on the test input (10000 random integers). The benchmark is in /code/bench.py.",
    "Read the JSON config at /config/settings.json. It has nested keys. Find all keys whose values are null or empty strings, and report them with their full path (e.g., 'database.host').",
]


async def evaluate_genome(genome: PromptGenome, client: openai.AsyncOpenAI) -> list[TaskResult]:
    """Run Qwen on each task with the genome's system prompt, grade responses."""
    results = []

    system_msg = genome.system_prompt
    if genome.tool_instructions:
        system_msg += f"\n\n## Tools\n{genome.tool_instructions}"
    if genome.cot_scaffold:
        system_msg += f"\n\n## Approach\n{genome.cot_scaffold}"

    for task in EVAL_TASKS:
        try:
            t0 = time.time()
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": task},
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            elapsed = time.time() - t0
            content = response.choices[0].message.content or ""

            # Score the response based on quality signals
            score = 0.0
            action_count = 0
            has_tool_calls = False
            has_reasoning = False
            has_submission = False

            # Check for tool usage patterns
            for tool in ["[bash]", "[python]", "[read_file]", "[write_file]", "[sql]", "[api_call]", "[submit]",
                         "bash:", "python:", "read_file:", "write_file:"]:
                if tool.lower() in content.lower():
                    has_tool_calls = True
                    action_count += content.lower().count(tool.lower())

            # Check for reasoning
            reasoning_signals = ["think", "step", "first", "then", "because", "therefore", "let me", "approach"]
            for signal in reasoning_signals:
                if signal in content.lower():
                    has_reasoning = True
                    break

            # Check for submission/answer
            if "[submit]" in content.lower() or "submit:" in content.lower() or "answer:" in content.lower() or "result:" in content.lower():
                has_submission = True

            # Composite score
            if has_tool_calls:
                score += 0.3
            if has_reasoning:
                score += 0.3
            if has_submission:
                score += 0.2
            if action_count >= 2:
                score += 0.1  # Multi-step plan
            if len(content) > 100:
                score += 0.1  # Substantive response

            results.append(TaskResult(
                task_description=task[:80],
                qwen_response=content[:500],
                score=score,
                action_count=action_count,
                has_tool_calls=has_tool_calls,
                has_reasoning=has_reasoning,
                has_submission=has_submission,
            ))
            logger.info("  Task %d: score=%.2f tools=%s reasoning=%s submit=%s (%.1fs)",
                        len(results), score, has_tool_calls, has_reasoning, has_submission, elapsed)

        except Exception as e:
            logger.error("  Task %d failed: %s", len(results) + 1, e)
            results.append(TaskResult(task_description=task[:80], score=0.0))

    return results


# ---------------------------------------------------------------------------
# Mutation: Opus analyzes results and proposes improvements
# ---------------------------------------------------------------------------

async def propose_mutation(genome: PromptGenome, results: list[TaskResult]) -> PromptGenome:
    """Send genome + eval results to Opus, get a mutated genome back."""

    # Build analysis context
    results_summary = []
    for r in results:
        results_summary.append({
            "task": r.task_description,
            "score": r.score,
            "has_tools": r.has_tool_calls,
            "has_reasoning": r.has_reasoning,
            "has_submission": r.has_submission,
            "response_preview": r.qwen_response[:200],
        })

    avg_score = sum(r.score for r in results) / len(results) if results else 0

    prompt = f"""You are optimizing a system prompt for a coding agent.
The agent must use tools like [bash], [python], [read_file], [write_file], [sql], [api_call], [submit] to solve coding tasks.

CURRENT SYSTEM PROMPT:
{genome.system_prompt}

CURRENT TOOL INSTRUCTIONS:
{genome.tool_instructions}

CURRENT COT SCAFFOLD:
{genome.cot_scaffold}

EVALUATION RESULTS (avg score: {avg_score:.2f}/1.0):
{json.dumps(results_summary, indent=2)}

Scoring criteria: tool_usage (0.3), reasoning (0.3), submission (0.2), multi_step (0.1), substantive (0.1)

TASK: Analyze the failures and propose an IMPROVED version of the system prompt. Focus on:
1. Getting the agent to actually use the tool format [tool_name]: content
2. Getting it to reason step-by-step before acting
3. Getting it to submit a final answer

Return ONLY a JSON object with these exact fields:
{{
  "diagnosis": "what's wrong with the current prompt",
  "system_prompt": "the full improved system prompt",
  "tool_instructions": "improved tool usage instructions",
  "cot_scaffold": "improved reasoning scaffold"
}}"""

    try:
        proc = await asyncio.create_subprocess_exec(
            CLAUDE_BIN, "-p", prompt,
            "--output-format", "text",
            "--max-turns", "1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        output = stdout.decode().strip()

        # Try to parse JSON from the response
        # Handle cases where Opus wraps JSON in markdown
        json_str = output
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()

        data = json.loads(json_str)

        new_genome = PromptGenome(
            system_prompt=data.get("system_prompt", genome.system_prompt),
            tool_instructions=data.get("tool_instructions", genome.tool_instructions),
            cot_scaffold=data.get("cot_scaffold", genome.cot_scaffold),
            generation=genome.generation + 1,
        )

        logger.info("  Opus diagnosis: %s", data.get("diagnosis", "N/A")[:200])
        return new_genome

    except json.JSONDecodeError as e:
        logger.warning("  Opus returned non-JSON, using raw output as system prompt")
        return PromptGenome(
            system_prompt=output[:3000],
            tool_instructions=genome.tool_instructions,
            cot_scaffold=genome.cot_scaffold,
            generation=genome.generation + 1,
        )
    except Exception as e:
        logger.error("  Mutation failed: %s", e)
        return genome  # Return unchanged


# ---------------------------------------------------------------------------
# Main GEPA loop
# ---------------------------------------------------------------------------

async def run_gepa(iterations: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    client = openai.AsyncOpenAI(base_url=LOCAL_LLM_BASE, api_key="local")

    # Seed genome
    genome = PromptGenome(
        system_prompt=(
            "You are a skilled coding agent. You solve tasks by using tools to interact with files, "
            "run code, query databases, and call APIs. Always think step-by-step before acting."
        ),
        tool_instructions=(
            "Available tools (format your actions exactly like this):\n"
            "[bash]: <shell command>\n"
            "[python]: <python code>\n"
            "[read_file]: <file path>\n"
            "[write_file]: <file path>\n<content>\n"
            "[sql]: <SQL query>\n"
            "[api_call]: <endpoint> <params>\n"
            "[submit]: <your final answer>\n\n"
            "Always end with [submit] when you have your answer."
        ),
        cot_scaffold=(
            "For each task:\n"
            "1. Understand what is being asked\n"
            "2. Plan your approach (which tools to use and in what order)\n"
            "3. Execute step by step, checking results at each stage\n"
            "4. Submit your final answer with [submit]"
        ),
    )

    best_genome = genome
    best_score = 0.0
    history = []

    for i in range(iterations):
        logger.info("")
        logger.info("=" * 60)
        logger.info("ITERATION %d/%d (generation %d)", i + 1, iterations, genome.generation)
        logger.info("=" * 60)

        # Evaluate current genome
        logger.info("Evaluating genome %s ...", genome.genome_id)
        results = await evaluate_genome(genome, client)
        avg_score = sum(r.score for r in results) / len(results) if results else 0
        genome.scores = {
            "avg_score": avg_score,
            "tool_usage": sum(1 for r in results if r.has_tool_calls) / len(results),
            "reasoning": sum(1 for r in results if r.has_reasoning) / len(results),
            "submission": sum(1 for r in results if r.has_submission) / len(results),
        }

        logger.info("Scores: avg=%.3f tools=%.0f%% reasoning=%.0f%% submit=%.0f%%",
                     avg_score,
                     genome.scores["tool_usage"] * 100,
                     genome.scores["reasoning"] * 100,
                     genome.scores["submission"] * 100)

        # Track best
        if avg_score > best_score:
            best_score = avg_score
            best_genome = genome
            logger.info("New best! Score: %.3f", best_score)

        # Save iteration results
        iteration_data = {
            "iteration": i + 1,
            "genome_id": genome.genome_id,
            "generation": genome.generation,
            "scores": genome.scores,
            "system_prompt": genome.system_prompt[:500],
        }
        history.append(iteration_data)

        # Propose mutation via Opus
        if i < iterations - 1:  # Don't mutate on last iteration
            logger.info("Proposing mutation via Opus ...")
            genome = await propose_mutation(genome, results)
            logger.info("New genome: %s (gen %d)", genome.genome_id, genome.generation)

    # Save final results
    final = {
        "best_score": best_score,
        "best_genome": {
            "genome_id": best_genome.genome_id,
            "system_prompt": best_genome.system_prompt,
            "tool_instructions": best_genome.tool_instructions,
            "cot_scaffold": best_genome.cot_scaffold,
            "generation": best_genome.generation,
            "scores": best_genome.scores,
        },
        "history": history,
    }

    results_path = output_dir / "loop1_results.json"
    with open(results_path, "w") as f:
        json.dump(final, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("LOOP 1 COMPLETE")
    logger.info("=" * 60)
    logger.info("Iterations: %d", iterations)
    logger.info("Best score: %.3f (genome %s, gen %d)", best_score, best_genome.genome_id, best_genome.generation)
    logger.info("Results saved to: %s", results_path)
    logger.info("")
    logger.info("Best system prompt:")
    logger.info(best_genome.system_prompt[:300])


def main():
    parser = argparse.ArgumentParser(description="Loop 1 Lite: GEPA prompt evolution")
    parser.add_argument("--iterations", type=int, default=3, help="Number of evolution iterations")
    parser.add_argument("--output-dir", type=str, default="./data/loop1_results", help="Output directory")
    args = parser.parse_args()

    asyncio.run(run_gepa(args.iterations, Path(args.output_dir)))


if __name__ == "__main__":
    main()
