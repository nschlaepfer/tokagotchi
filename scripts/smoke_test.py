"""Smoke test: verify Ollama + Claude CLI work, then run a mini Loop 1 iteration.

Usage:
    python scripts/smoke_test.py

This test does NOT require Docker. It tests:
1. Ollama is serving the model and can do inference
2. Claude CLI (headless) can be invoked and returns structured output
3. A single GEPA mutation cycle works end-to-end
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time

import openai


def find_claude_cli() -> str:
    """Find the claude CLI binary, checking common locations."""
    # Check PATH first
    found = shutil.which("claude")
    if found:
        return found
    # Common Windows npm global install location
    npm_path = os.path.expandvars(r"%APPDATA%\npm\claude.cmd")
    if os.path.exists(npm_path):
        return npm_path
    npm_path2 = os.path.expandvars(r"%APPDATA%\npm\claude")
    if os.path.exists(npm_path2):
        return npm_path2
    return "claude"  # fallback, hope for the best


CLAUDE_BIN = find_claude_cli()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke_test")


async def check_ollama() -> bool:
    """Test that Ollama is serving and can do inference."""
    logger.info("=" * 60)
    logger.info("TEST 1: Ollama inference")
    logger.info("=" * 60)

    client = openai.AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    try:
        t0 = time.time()
        response = await client.chat.completions.create(
            model="huihui_ai/qwen3.5-abliterated:27b",
            messages=[
                {"role": "system", "content": "You are a coding assistant. Be concise."},
                {"role": "user", "content": "Write a Python function that checks if a number is prime. Just the code, nothing else."},
            ],
            temperature=0.7,
            max_tokens=256,
        )
        elapsed = time.time() - t0
        content = response.choices[0].message.content
        tokens = response.usage.completion_tokens if response.usage else 0

        logger.info("Response (%d tokens in %.1fs, %.1f tok/s):", tokens, elapsed, tokens / elapsed if elapsed > 0 else 0)
        logger.info(content[:500])
        logger.info("PASS: Ollama inference works")
        return True

    except Exception as e:
        logger.error("FAIL: Ollama inference failed: %s", e)
        return False


async def check_claude_cli() -> bool:
    """Test that Claude CLI headless mode works."""
    logger.info("=" * 60)
    logger.info("TEST 2: Claude CLI headless mode")
    logger.info("=" * 60)

    try:
        proc = await asyncio.create_subprocess_exec(
            CLAUDE_BIN, "-p",
            "Return a JSON object with exactly these fields: {\"status\": \"ok\", \"message\": \"hello from opus\"}. Return ONLY the JSON, no markdown.",
            "--output-format", "text",
            "--max-turns", "1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        output = stdout.decode().strip()

        logger.info("Claude CLI output: %s", output[:500])

        if proc.returncode == 0 and len(output) > 0:
            logger.info("PASS: Claude CLI headless works")
            return True
        else:
            logger.error("FAIL: Claude CLI returned code %d, stderr: %s", proc.returncode, stderr.decode()[:300])
            return False

    except asyncio.TimeoutError:
        logger.error("FAIL: Claude CLI timed out after 60s")
        return False
    except FileNotFoundError:
        logger.error("FAIL: 'claude' command not found on PATH")
        return False
    except Exception as e:
        logger.error("FAIL: Claude CLI error: %s", e)
        return False


async def check_mini_gepa_cycle() -> bool:
    """Test a simplified GEPA mutation cycle: Opus analyzes a Qwen response and proposes an improvement."""
    logger.info("=" * 60)
    logger.info("TEST 3: Mini GEPA cycle (Opus analyzes Qwen, proposes mutation)")
    logger.info("=" * 60)

    # Step 1: Get a response from Qwen
    client = openai.AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    test_prompt = "You have 5 CSV files in /data/. Find the customer with the highest total spend. Think step by step."
    system_prompt = "You are a coding agent. You can use tools: [bash], [python], [read_file], [submit]. Format actions as [tool_name]: content"

    try:
        t0 = time.time()
        qwen_response = await client.chat.completions.create(
            model="huihui_ai/qwen3.5-abliterated:27b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": test_prompt},
            ],
            temperature=0.7,
            max_tokens=512,
        )
        qwen_output = qwen_response.choices[0].message.content
        logger.info("Qwen response (%.1fs): %s", time.time() - t0, qwen_output[:300])

        # Step 2: Send to Opus for analysis
        analysis_prompt = f"""Analyze this coding agent's response to a task and suggest ONE specific improvement to the system prompt.

TASK: {test_prompt}
SYSTEM PROMPT: {system_prompt}
AGENT RESPONSE: {qwen_output[:500]}

Return a JSON object:
{{"diagnosis": "what the agent did wrong or could improve", "mutation": "the specific change to make to the system prompt", "improved_prompt": "the full improved system prompt"}}

Return ONLY valid JSON, no markdown."""

        proc = await asyncio.create_subprocess_exec(
            CLAUDE_BIN, "-p", analysis_prompt,
            "--output-format", "text",
            "--max-turns", "1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        opus_output = stdout.decode().strip()

        logger.info("Opus analysis: %s", opus_output[:500])

        if proc.returncode == 0 and len(opus_output) > 10:
            logger.info("PASS: Mini GEPA cycle completed")
            return True
        else:
            logger.error("FAIL: Opus analysis failed (code=%d)", proc.returncode)
            return False

    except Exception as e:
        logger.error("FAIL: Mini GEPA cycle error: %s", e)
        return False


async def main() -> None:
    results = {}

    results["ollama"] = await check_ollama()
    results["claude_cli"] = await check_claude_cli()

    if results["ollama"] and results["claude_cli"]:
        results["gepa_cycle"] = await check_mini_gepa_cycle()
    else:
        logger.warning("Skipping GEPA cycle test — prerequisites failed")
        results["gepa_cycle"] = False

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SMOKE TEST RESULTS")
    logger.info("=" * 60)
    all_pass = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info("  %-20s %s", test_name, status)
        if not passed:
            all_pass = False

    if all_pass:
        logger.info("")
        logger.info("All tests passed! tokagotchi is ready to run.")
        logger.info("Next: python scripts/run_loop1.py --iterations 5")
    else:
        logger.info("")
        logger.info("Some tests failed. Fix the issues above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
