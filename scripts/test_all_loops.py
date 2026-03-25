#!/usr/bin/env python3
"""Comprehensive integration test for all tokagotchi system components.

Runs end-to-end tests WITHOUT Docker, using subprocess sandboxing instead.
Each test prints PASS / FAIL / SKIP with timing information.

Usage:
    python scripts/test_all_loops.py
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup -- ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Test runner infrastructure
# ---------------------------------------------------------------------------

_results: list[dict[str, Any]] = []


def _run_test(name: str, func):
    """Run a single test function, catching all exceptions."""
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    status = "FAIL"
    detail = ""
    try:
        result = func()
        # Handle coroutines from async test functions
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
        status = "PASS"
        detail = str(result) if result else ""
    except _SkipTest as exc:
        status = "SKIP"
        detail = str(exc)
    except Exception as exc:
        status = "FAIL"
        detail = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    elapsed = time.perf_counter() - t0
    _results.append({"name": name, "status": status, "elapsed": elapsed, "detail": detail})

    tag = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[status]
    print(f"\n  [{tag}] {name}  ({elapsed:.3f}s)")
    if detail:
        # Truncate long details for display
        short = detail if len(detail) < 300 else detail[:300] + "..."
        print(f"         {short}")


class _SkipTest(Exception):
    """Raised to mark a test as skipped."""


def skip(reason: str):
    raise _SkipTest(reason)


# ===================================================================
# Test 0: Imports
# ===================================================================

def test_00_imports():
    """Verify all major modules import without errors."""
    import src.config
    import src.models
    import src.rewards.efficiency_penalty
    import src.rewards.outcome_reward
    import src.loop1_gepa.prompt_genome
    import src.loop1_gepa.pareto_tracker
    import src.loop1_gepa.evaluator
    import src.loop1_gepa.mutation_operators
    import src.loop1_gepa.evolution_engine
    import src.loop2_distill.pending_buffer
    import src.loop3_rl.trajectory_filter
    import src.loop3_rl.dapo_clipping
    import src.loop3_rl.tree_grpo
    import src.curriculum.sec_engine
    import src.orchestrator.budget_tracker
    import src.orchestrator.opus_client
    return "All modules imported successfully"


# ===================================================================
# Test 1: Config loading
# ===================================================================

def test_01_config_loading():
    """Load config from config/ directory and verify MasterConfig structure."""
    from src.config import load_config, MasterConfig

    cfg = load_config(PROJECT_ROOT / "config")
    assert isinstance(cfg, MasterConfig), f"Expected MasterConfig, got {type(cfg)}"
    assert cfg.model.resolved_port > 0, f"Expected a positive serving port, got {cfg.model.resolved_port}"
    assert cfg.opus.daily_budget_usd > 0, "Daily budget should be positive"
    assert cfg.loop1.population_size > 0, "Population size should be positive"
    assert cfg.loop3.algorithm == "grpo", f"Expected grpo, got {cfg.loop3.algorithm}"
    assert len(cfg.loop1.pareto_objectives) >= 3, "Should have at least 3 pareto objectives"
    return f"MasterConfig loaded: model={cfg.model.name}, budget=${cfg.opus.daily_budget_usd}/day"


# ===================================================================
# Test 2: Models -- TaskSpec, Trajectory, PromptGenome serialization
# ===================================================================

def test_02_models():
    """Create TaskSpec, Trajectory, PromptGenome and verify serialization."""
    from src.models import (
        TaskSpec, TaskType, Trajectory, StepRecord, ActionType, PromptGenome,
    )

    # TaskSpec
    task = TaskSpec(
        task_type=TaskType.CODE_DEBUGGING,
        description="Fix the broken test",
        initial_files={"main.py": "print('hello')"},
        test_commands=["python -m pytest test_main.py"],
        expected_output="All tests pass",
        difficulty=0.5,
    )
    task_dict = asdict(task)
    assert "task_id" in task_dict
    assert task_dict["task_type"] == "code_debugging"

    # StepRecord + Trajectory
    step = StepRecord(
        step_idx=0,
        action_type=ActionType.BASH,
        action_content="ls -la",
        observation="total 4\n-rw-r--r-- 1 user user 15 main.py",
    )
    traj = Trajectory(
        task=task,
        steps=[step],
        success=True,
        total_reward=0.8,
        wall_time_seconds=3.2,
        model_id="qwen-test",
        prompt_genome_id="abc123",
    )
    assert traj.num_steps == 1
    assert "bash" in traj.action_types_used

    traj_dict = asdict(traj)
    assert isinstance(traj_dict, dict)
    json_str = json.dumps(traj_dict, default=str)
    assert len(json_str) > 50

    # PromptGenome
    genome = PromptGenome(
        system_prompt="You are a coding agent.",
        cot_scaffold="Think step by step.",
        tool_instructions="Use bash for commands.",
        generation=0,
    )
    msg = genome.to_system_message()
    assert "coding agent" in msg
    assert "Think step by step" in msg

    genome_dict = asdict(genome)
    json_genome = json.dumps(genome_dict, default=str)
    assert len(json_genome) > 20

    return "TaskSpec, Trajectory, PromptGenome all serialize correctly"


# ===================================================================
# Test 3: Local LLM inference
# ===================================================================

def test_03_local_llm_inference():
    """Make a chat completion call to the configured local model server."""
    try:
        from openai import OpenAI
    except ImportError:
        skip("openai package not installed")

    from src.config import load_config
    cfg = load_config(PROJECT_ROOT / "config")

    import requests

    api_url = f"http://{cfg.model.resolved_host}:{cfg.model.resolved_port}"

    try:
        resp = requests.post(
            f"{api_url}/v1/chat/completions",
            json={
                "model": cfg.model.name,
                "messages": [
                    {"role": "system", "content": "Reply in one short sentence."},
                    {"role": "user", "content": "What is 2+2?"},
                ],
                "max_tokens": 64,
                "temperature": 0.0,
            },
            timeout=120,
        )
        resp.raise_for_status()
    except Exception as exc:
        skip(f"Local LLM server not reachable at {api_url}: {exc}")

    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content", "")
    tokens = (data.get("usage") or {}).get("completion_tokens", 0)
    assert len(content) > 0, f"Empty response from local LLM server: {data}"
    return f"Local LLM responded ({tokens} tokens): {content[:80]}"


# ===================================================================
# Test 4: Claude CLI
# ===================================================================

def test_04_claude_cli():
    """Invoke claude -p with a simple prompt."""
    try:
        result = subprocess.run(
            ["claude", "-p", "Reply with exactly: PING_OK", "--output-format", "json", "--max-turns", "1"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        skip("claude CLI not found on PATH")
    except subprocess.TimeoutExpired:
        skip("claude CLI timed out after 60s")

    if result.returncode != 0:
        skip(f"claude CLI exited with code {result.returncode}: {result.stderr[:200]}")

    stdout = result.stdout.strip()
    assert len(stdout) > 0, "Empty stdout from claude CLI"

    # Try to parse as JSON
    try:
        data = json.loads(stdout)
        text = data.get("result", stdout)
    except json.JSONDecodeError:
        text = stdout

    return f"Claude CLI responded: {str(text)[:120]}"


# ===================================================================
# Test 5: Budget Tracker
# ===================================================================

def test_05_budget_tracker():
    """Create a BudgetTracker, record spend, check limits."""
    from src.orchestrator.budget_tracker import BudgetTracker, BudgetExhaustedError

    tracker = BudgetTracker(
        hourly_limit_usd=1.00,
        daily_limit_usd=5.00,
        persist_path=None,  # in-memory only
    )

    # Should be able to spend within limits
    assert tracker.can_spend(0.50, loop_id="test"), "Should allow $0.50"
    tracker.record_spend(0.50, loop_id="loop1", prompt_tokens=100, completion_tokens=50)

    assert tracker.can_spend(0.40, loop_id="test"), "Should allow another $0.40"
    tracker.record_spend(0.40, loop_id="loop2", prompt_tokens=80, completion_tokens=40)

    # Should NOT allow exceeding hourly limit
    assert not tracker.can_spend(0.20, loop_id="test"), "Should deny $0.20 (would exceed hourly $1.00)"

    # Check summary
    summary = tracker.get_summary()
    assert summary["num_calls"] == 2
    assert abs(summary["total_usd"] - 0.90) < 0.001
    assert summary["total_prompt_tokens"] == 180
    assert summary["total_completion_tokens"] == 90

    # Verify circuit breaker raises
    raised = False
    try:
        tracker.record_spend(0.20, loop_id="over_limit")
    except BudgetExhaustedError as exc:
        raised = True
        assert "hourly" in exc.limit_type
    assert raised, "Should have raised BudgetExhaustedError"

    return f"Budget tracker OK: {summary['num_calls']} calls, ${summary['total_usd']:.2f} total"


# ===================================================================
# Test 6: Opus Client (real call, small/cheap)
# ===================================================================

def test_06_opus_client():
    """Make a real (small, cheap) call via OpusClient to analyze a mock trace."""

    async def _run():
        from src.orchestrator.opus_client import OpusClient
        from src.orchestrator.budget_tracker import BudgetTracker
        from src.config import OpusConfig
        from src.models import Trajectory, TaskSpec, StepRecord, ActionType, TaskType

        # Check that claude CLI is available
        try:
            probe = subprocess.run(
                ["claude", "--version"],
                capture_output=True, text=True, timeout=10,
            )
            if probe.returncode != 0:
                skip("claude CLI not working")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            skip("claude CLI not available")

        cfg = OpusConfig(
            daily_budget_usd=5.0,
            hourly_budget_usd=2.0,
            default_max_budget_per_call_usd=0.10,
            default_max_turns=1,
        )
        budget = BudgetTracker(hourly_limit_usd=2.0, daily_limit_usd=5.0)
        client = OpusClient(config=cfg, budget_tracker=budget)

        # Build a tiny mock trajectory
        task = TaskSpec(task_type=TaskType.CODE_DEBUGGING, description="Fix a bug")
        step = StepRecord(
            step_idx=0,
            action_type=ActionType.BASH,
            action_content="cat main.py",
            observation="def add(a, b): return a - b  # BUG",
        )
        traj = Trajectory(task=task, steps=[step], success=False, total_reward=0.0)

        resp = await client.query(
            "In one sentence, what is wrong with this code? def add(a, b): return a - b",
            max_budget_usd=0.05,
            max_turns=1,
        )

        if resp.is_error:
            skip(f"OpusClient query failed: {resp.error_message}")

        assert len(resp.text) > 0, "Empty response text"
        return f"OpusClient OK (cost=${resp.cost_usd:.4f}): {resp.text[:100]}"

    return asyncio.run(_run())


# ===================================================================
# Test 7: Loop 1 GEPA lite
# ===================================================================

def test_07_gepa_lite():
    """Run 1 mini GEPA iteration: create seed genome, evaluate via the local LLM, mutate via Opus."""

    async def _run():
        from src.loop1_gepa.prompt_genome import create_seed_genome
        from src.models import PromptGenome

        # Step 1: Create seed genome
        genome = create_seed_genome()
        assert genome.system_prompt, "Seed genome should have a system prompt"
        assert genome.generation == 0

        # Step 2: Evaluate via the local model server (lightweight -- just test it responds)
        try:
            from openai import OpenAI
        except ImportError:
            skip("openai package not installed")

        from src.config import load_config
        cfg = load_config(PROJECT_ROOT / "config")

        try:
            import requests as _req
            api_url = f"http://{cfg.model.resolved_host}:{cfg.model.resolved_port}"
            resp = _req.post(
                f"{api_url}/v1/chat/completions",
                json={
                    "model": cfg.model.name,
                    "messages": [
                        {"role": "system", "content": genome.to_system_message()[:500]},
                        {"role": "user", "content": "What is the first step to fix a failing Python test?"},
                    ],
                    "max_tokens": 100,
                    "temperature": 0.7,
                },
                timeout=120,
            )
            resp.raise_for_status()
            local_llm_answer = ((resp.json().get("choices") or [{}])[0].get("message") or {}).get("content", "")
            assert len(local_llm_answer) > 0, "Empty local LLM response"
            local_llm_ok = True
        except Exception:
            local_llm_ok = False

        # Step 3: Mutate via Opus (if available)
        opus_ok = False
        try:
            probe = subprocess.run(
                ["claude", "--version"], capture_output=True, text=True, timeout=10,
            )
            if probe.returncode == 0:
                from src.orchestrator.opus_client import OpusClient
                from src.orchestrator.budget_tracker import BudgetTracker
                from src.config import OpusConfig

                opus_cfg = OpusConfig(
                    default_max_budget_per_call_usd=0.05,
                    default_max_turns=1,
                )
                opus_budget = BudgetTracker(hourly_limit_usd=2.0, daily_limit_usd=5.0)
                opus_client = OpusClient(config=opus_cfg, budget_tracker=opus_budget)

                mutation_resp = await opus_client.query(
                    f"Given this system prompt for a coding agent, suggest one small improvement "
                    f"in 1-2 sentences:\n\n{genome.system_prompt[:300]}",
                    max_budget_usd=0.05,
                    max_turns=1,
                )
                opus_ok = not mutation_resp.is_error and len(mutation_resp.text) > 0
        except Exception:
            pass

        if not local_llm_ok and not opus_ok:
            skip("Neither the local LLM server nor Claude CLI is available for the GEPA lite test")

        parts = []
        if local_llm_ok:
            parts.append("Local LLM eval OK")
        else:
            parts.append("Local LLM eval SKIP")
        if opus_ok:
            parts.append("Opus mutation OK")
        else:
            parts.append("Opus mutation SKIP")

        return f"GEPA lite: seed genome created, {', '.join(parts)}"

    return asyncio.run(_run())


# ===================================================================
# Test 8: Efficiency penalty
# ===================================================================

def test_08_efficiency_penalty():
    """Compute efficiency penalty on a mock trajectory."""
    from src.rewards.efficiency_penalty import compute_efficiency_penalty
    from src.models import Trajectory, StepRecord, ActionType, TaskSpec, TaskType

    task = TaskSpec(task_type=TaskType.CODE_DEBUGGING, description="test")

    # Build a trajectory with known wasteful patterns
    steps = [
        # Repeated action (same bash command twice)
        StepRecord(step_idx=0, action_type=ActionType.BASH,
                   action_content="cat main.py", observation="code here"),
        StepRecord(step_idx=1, action_type=ActionType.BASH,
                   action_content="cat main.py", observation="code here"),
        # Unnecessary think (think followed by think)
        StepRecord(step_idx=2, action_type=ActionType.THINK,
                   action_content="hmm let me think", observation=""),
        StepRecord(step_idx=3, action_type=ActionType.THINK,
                   action_content="still thinking", observation=""),
        # Suboptimal tool use (cat via bash instead of read_file)
        StepRecord(step_idx=4, action_type=ActionType.BASH,
                   action_content="cat test.py", observation="test code"),
        # Normal action
        StepRecord(step_idx=5, action_type=ActionType.WRITE_FILE,
                   action_content="main.py\nfixed code", observation="ok"),
        StepRecord(step_idx=6, action_type=ActionType.SUBMIT,
                   action_content="done", observation=""),
    ]
    traj = Trajectory(task=task, steps=steps, success=True)

    penalty = compute_efficiency_penalty(traj)
    assert 0.0 < penalty <= 0.3, f"Expected penalty in (0, 0.3], got {penalty}"

    # A clean trajectory should have zero penalty
    clean_steps = [
        StepRecord(step_idx=0, action_type=ActionType.READ_FILE,
                   action_content="main.py", observation="code"),
        StepRecord(step_idx=1, action_type=ActionType.WRITE_FILE,
                   action_content="main.py\nfixed", observation="ok"),
        StepRecord(step_idx=2, action_type=ActionType.SUBMIT,
                   action_content="done", observation=""),
    ]
    clean_traj = Trajectory(task=task, steps=clean_steps, success=True)
    clean_penalty = compute_efficiency_penalty(clean_traj)
    assert clean_penalty == 0.0, f"Expected 0 penalty for clean traj, got {clean_penalty}"

    return f"Efficiency penalty: wasteful={penalty:.4f}, clean={clean_penalty:.4f}"


# ===================================================================
# Test 9: Outcome reward
# ===================================================================

def test_09_outcome_reward():
    """Compute outcome reward on mock trajectories (info_gathering type)."""
    from src.rewards.outcome_reward import _reward_info_gathering
    from src.models import Trajectory, StepRecord, ActionType, TaskSpec, TaskType

    # Exact match
    task = TaskSpec(
        task_type=TaskType.INFO_GATHERING,
        description="Find the capital of France",
        expected_output="Paris",
    )
    steps = [
        StepRecord(step_idx=0, action_type=ActionType.SUBMIT,
                   action_content="Paris", observation=""),
    ]
    traj = Trajectory(task=task, steps=steps, success=True)
    reward = _reward_info_gathering(traj, task)
    assert reward == 1.0, f"Expected 1.0 for exact match, got {reward}"

    # Partial match
    steps2 = [
        StepRecord(step_idx=0, action_type=ActionType.SUBMIT,
                   action_content="The capital is Paris, France", observation=""),
    ]
    traj2 = Trajectory(task=task, steps=steps2, success=True)
    reward2 = _reward_info_gathering(traj2, task)
    assert 0.0 < reward2 < 1.0, f"Expected partial credit, got {reward2}"

    # No submission
    steps3 = [
        StepRecord(step_idx=0, action_type=ActionType.THINK,
                   action_content="I should look this up", observation=""),
    ]
    traj3 = Trajectory(task=task, steps=steps3, success=False)
    reward3 = _reward_info_gathering(traj3, task)
    assert reward3 == 0.0, f"Expected 0.0 for no submission, got {reward3}"

    return f"Outcome reward: exact={reward}, partial={reward2:.4f}, none={reward3}"


# ===================================================================
# Test 10: Trajectory filter
# ===================================================================

def test_10_trajectory_filter():
    """Filter mock trajectories using TrajectoryFilter."""
    from src.loop3_rl.trajectory_filter import TrajectoryFilter
    from src.config import Loop3Config
    from src.models import Trajectory, StepRecord, ActionType, TaskSpec, TaskType

    config = Loop3Config(echo_trap_threshold=3, min_trajectory_reward=0.1)
    filt = TrajectoryFilter(config)

    task = TaskSpec(task_type=TaskType.CODE_DEBUGGING, description="test")

    # Good trajectory
    good_steps = [
        StepRecord(step_idx=0, action_type=ActionType.READ_FILE,
                   action_content="main.py", observation="code"),
        StepRecord(step_idx=1, action_type=ActionType.BASH,
                   action_content="python main.py", observation="output"),
        StepRecord(step_idx=2, action_type=ActionType.WRITE_FILE,
                   action_content="main.py\nfixed", observation="ok"),
        StepRecord(step_idx=3, action_type=ActionType.SUBMIT,
                   action_content="done", observation=""),
    ]
    good_traj = Trajectory(task=task, steps=good_steps, success=True)

    # Echo trap trajectory (same action 3+ times)
    echo_steps = [
        StepRecord(step_idx=i, action_type=ActionType.BASH,
                   action_content="cat main.py", observation="code")
        for i in range(5)
    ]
    echo_traj = Trajectory(task=task, steps=echo_steps, success=False)

    # Low reward trajectory
    low_steps = [
        StepRecord(step_idx=0, action_type=ActionType.THINK,
                   action_content="hmm", observation=""),
        StepRecord(step_idx=1, action_type=ActionType.BASH,
                   action_content="ls", observation="files"),
    ]
    low_traj = Trajectory(task=task, steps=low_steps, success=False)

    trajs = [good_traj, echo_traj, low_traj]
    rewards = [0.8, 0.05, 0.02]

    kept_trajs, kept_rewards = filt.filter_batch(trajs, rewards)
    assert len(kept_trajs) == 1, f"Expected 1 kept trajectory, got {len(kept_trajs)}"
    assert kept_rewards[0] == 0.8

    # Verify detection helpers
    assert filt.is_echo_trap(echo_traj), "Should detect echo trap"
    assert not filt.is_echo_trap(good_traj), "Good traj should not be echo trap"
    assert filt.is_degenerate(low_traj, 0.02), "Low reward should be degenerate"

    return f"Trajectory filter: {len(trajs)} in, {len(kept_trajs)} kept"


# ===================================================================
# Test 11: DAPO clipper
# ===================================================================

def test_11_dapo_clipper():
    """Compute DAPO-clipped policy loss on mock tensors."""
    try:
        import torch
    except ImportError:
        skip("PyTorch not installed")

    from src.loop3_rl.dapo_clipping import DAPOClipper
    from src.config import DAPOConfig

    config = DAPOConfig(epsilon_low=0.1, epsilon_high=0.28)
    clipper = DAPOClipper(config)

    # Scalar clip tests
    # Positive advantage: upper bound is 1 + 0.28 = 1.28
    clipped = clipper.clip_ratio(1.5, advantage=1.0)
    assert abs(clipped - 1.28) < 1e-6, f"Expected 1.28, got {clipped}"

    # Negative advantage: symmetric clip, upper = 1 + 0.1 = 1.1
    clipped_neg = clipper.clip_ratio(1.5, advantage=-1.0)
    assert abs(clipped_neg - 1.1) < 1e-6, f"Expected 1.1, got {clipped_neg}"

    # Lower bound: 1 - 0.1 = 0.9
    clipped_low = clipper.clip_ratio(0.5, advantage=1.0)
    assert abs(clipped_low - 0.9) < 1e-6, f"Expected 0.9, got {clipped_low}"

    # Tensor-level policy loss
    batch_size = 8
    log_probs_new = torch.randn(batch_size)
    log_probs_old = torch.randn(batch_size)
    advantages = torch.randn(batch_size)

    loss = clipper.compute_policy_loss(log_probs_new, log_probs_old, advantages)
    assert loss.shape == (), f"Expected scalar loss, got shape {loss.shape}"
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"

    # Entropy bonus
    logits = torch.randn(batch_size, 100)
    bonus = DAPOClipper.entropy_bonus(logits, config)
    assert bonus.shape == (), f"Expected scalar bonus, got shape {bonus.shape}"
    assert bonus >= 0, f"Entropy bonus should be non-negative, got {bonus}"

    return f"DAPO clipper: loss={loss.item():.4f}, entropy_bonus={bonus.item():.6f}"


# ===================================================================
# Test 12: GRPO advantages
# ===================================================================

def test_12_grpo_advantages():
    """Compute GRPO advantages on mock rewards."""
    from src.loop3_rl.tree_grpo import TreeGRPO
    from src.models import Trajectory, TaskSpec, TaskType

    task = TaskSpec(task_type=TaskType.CODE_DEBUGGING, description="test")
    trajs = [Trajectory(task=task) for _ in range(4)]
    rewards = [0.2, 0.5, 0.8, 0.3]

    advantages = TreeGRPO.compute_grpo_advantages(trajs, rewards)
    assert len(advantages) == 4, f"Expected 4 advantages, got {len(advantages)}"

    # Advantages should be mean-centered (sum close to 0)
    adv_sum = sum(advantages)
    assert abs(adv_sum) < 1e-6, f"Expected sum ~0, got {adv_sum}"

    # Highest reward should get highest advantage
    max_adv_idx = advantages.index(max(advantages))
    max_reward_idx = rewards.index(max(rewards))
    assert max_adv_idx == max_reward_idx, "Max advantage should match max reward"

    # Empty case
    empty_adv = TreeGRPO.compute_grpo_advantages([], [])
    assert empty_adv == [], f"Expected empty list, got {empty_adv}"

    return f"GRPO advantages: {[f'{a:.4f}' for a in advantages]}"


# ===================================================================
# Test 13: Curriculum SEC engine
# ===================================================================

def test_13_sec_engine():
    """Register tasks, sample, and update stats in the SEC engine."""
    import tempfile
    from src.curriculum.sec_engine import SECEngine
    from src.models import TaskSpec, TaskType, Trajectory, StepRecord, ActionType

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        bank_path = f.name

    try:
        engine = SECEngine(task_bank_path=bank_path)

        # Register tasks
        tasks = [
            TaskSpec(task_id="t1", task_type=TaskType.CODE_DEBUGGING,
                     description="Fix bug 1", difficulty=0.3),
            TaskSpec(task_id="t2", task_type=TaskType.INFO_GATHERING,
                     description="Find info 1", difficulty=0.6),
            TaskSpec(task_id="t3", task_type=TaskType.API_ORCHESTRATION,
                     description="Call API 1", difficulty=0.9),
        ]
        engine.add_tasks(tasks)
        assert engine.task_count == 3, f"Expected 3 tasks, got {engine.task_count}"
        assert engine.active_task_count == 3

        # Sample tasks
        sampled = engine.sample_tasks(batch_size=2)
        assert len(sampled) == 2, f"Expected 2 sampled, got {len(sampled)}"
        assert all(isinstance(s, TaskSpec) for s in sampled)

        # Update stats
        traj = Trajectory(
            steps=[
                StepRecord(step_idx=0, action_type=ActionType.BASH,
                           action_content="ls", observation="output"),
                StepRecord(step_idx=1, action_type=ActionType.SUBMIT,
                           action_content="done", observation=""),
            ],
            success=True,
        )
        engine.update_stats("t1", success=True, trajectory=traj)
        engine.update_stats("t1", success=True, trajectory=traj)

        # Get capability profile
        profile = engine.get_capability_profile()
        assert profile["total_tasks"] == 3
        assert profile["overall"] > 0

        # Verify retrieval
        retrieved = engine.get_task("t2")
        assert retrieved is not None
        assert retrieved.task_id == "t2"

        # Save and verify
        engine.save()
        assert Path(bank_path).exists()

    finally:
        Path(bank_path).unlink(missing_ok=True)

    return f"SEC engine: {engine.task_count} tasks, profile={profile['overall']:.2f} overall"


# ===================================================================
# Test 14: Pending buffer
# ===================================================================

def test_14_pending_buffer():
    """Add examples to PendingBuffer, check readiness, drain."""
    import tempfile
    from src.loop2_distill.pending_buffer import PendingBuffer
    from src.config import Loop2Config

    config = Loop2Config(
        min_buffer_size=5,
        max_buffer_size=100,
        diversity_min_task_types=2,
        diversity_min_failure_modes=2,
    )

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        persist_path = f.name

    try:
        buf = PendingBuffer(config=config, persist_path=persist_path)

        # Initially empty and not ready
        assert buf.size() == 0
        assert not buf.is_ready()

        # Add examples with diversity
        task_types = ["code_debugging", "info_gathering", "api_orchestration"]
        failure_modes = ["wrong_tool", "timeout", "syntax_error"]

        for i in range(6):
            buf.add(
                example={"messages": [{"role": "user", "content": f"task {i}"}]},
                metadata={
                    "task_type": task_types[i % len(task_types)],
                    "failure_mode": failure_modes[i % len(failure_modes)],
                    "difficulty": 0.3 + (i * 0.1),
                },
            )

        assert buf.size() == 6
        assert buf.is_ready(), "Buffer should be ready (size >= 5, diversity met)"

        # Check diversity
        diverse, details = buf.diversity_check()
        assert diverse, f"Diversity check failed: {details}"
        assert details["task_types"]["count"] >= 2

        # Get stats
        stats = buf.get_stats()
        assert stats["size"] == 6
        assert stats["is_ready"]

        # Drain
        batch = buf.get_training_batch()
        assert len(batch) == 6
        assert buf.size() == 0
        assert not buf.is_ready()

        # Verify persistence
        buf.add(
            example={"messages": [{"role": "user", "content": "persist test"}]},
            metadata={"task_type": "code_debugging", "failure_mode": "test"},
        )
        buf.save()
        assert Path(persist_path).stat().st_size > 0

    finally:
        Path(persist_path).unlink(missing_ok=True)

    return f"Pending buffer: added 6, drained {len(batch)}, persistence OK"


# ===================================================================
# Test 15: Pareto tracker
# ===================================================================

def test_15_pareto_tracker():
    """Add genomes to ParetoTracker, check frontier."""
    from src.loop1_gepa.pareto_tracker import ParetoTracker
    from src.models import PromptGenome

    tracker = ParetoTracker()

    # Genome A: great success rate, bad step count
    genome_a = PromptGenome(genome_id="aaa", system_prompt="A", generation=0)
    on_frontier_a = tracker.add(genome_a, {
        "success_rate": 0.9, "avg_steps": 15.0,
        "tool_accuracy": 0.7, "code_quality": 0.6,
    })
    assert on_frontier_a, "First genome should be on frontier"
    assert tracker.frontier_size == 1

    # Genome B: moderate everything -- not dominated by A
    genome_b = PromptGenome(genome_id="bbb", system_prompt="B", generation=1)
    on_frontier_b = tracker.add(genome_b, {
        "success_rate": 0.7, "avg_steps": 5.0,
        "tool_accuracy": 0.8, "code_quality": 0.7,
    })
    assert on_frontier_b, "B should be on frontier (better avg_steps)"
    assert tracker.frontier_size == 2

    # Genome C: dominated by B on all axes
    genome_c = PromptGenome(genome_id="ccc", system_prompt="C", generation=1)
    on_frontier_c = tracker.add(genome_c, {
        "success_rate": 0.5, "avg_steps": 20.0,
        "tool_accuracy": 0.4, "code_quality": 0.3,
    })
    assert not on_frontier_c, "C should be dominated"
    assert tracker.frontier_size == 2

    # Genome D: dominates A on all axes
    genome_d = PromptGenome(genome_id="ddd", system_prompt="D", generation=2)
    on_frontier_d = tracker.add(genome_d, {
        "success_rate": 0.95, "avg_steps": 10.0,
        "tool_accuracy": 0.75, "code_quality": 0.65,
    })
    assert on_frontier_d, "D should be on frontier"
    # D dominates A, so A should be removed
    frontier_ids = {g.genome_id for g in tracker.get_frontier()}
    assert "aaa" not in frontier_ids, "A should have been removed (dominated by D)"
    assert "bbb" in frontier_ids
    assert "ddd" in frontier_ids

    # Select parents
    parents = tracker.select_parents(2)
    assert len(parents) == 2
    assert all(isinstance(p, PromptGenome) for p in parents)

    # Summary
    summary = tracker.summary()
    assert summary["frontier_size"] == tracker.frontier_size
    assert "objective_ranges" in summary

    return (
        f"Pareto tracker: frontier_size={tracker.frontier_size}, "
        f"members={frontier_ids}, history={summary['history_length']}"
    )


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 60)
    print("  TOKAGOTCHI INTEGRATION TEST SUITE")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    tests = [
        ("Test 00: Imports", test_00_imports),
        ("Test 01: Config loading", test_01_config_loading),
        ("Test 02: Models serialization", test_02_models),
        ("Test 03: Local LLM inference", test_03_local_llm_inference),
        ("Test 04: Claude CLI", test_04_claude_cli),
        ("Test 05: Budget Tracker", test_05_budget_tracker),
        ("Test 06: Opus Client (real call)", test_06_opus_client),
        ("Test 07: Loop 1 GEPA lite", test_07_gepa_lite),
        ("Test 08: Efficiency penalty", test_08_efficiency_penalty),
        ("Test 09: Outcome reward", test_09_outcome_reward),
        ("Test 10: Trajectory filter", test_10_trajectory_filter),
        ("Test 11: DAPO clipper", test_11_dapo_clipper),
        ("Test 12: GRPO advantages", test_12_grpo_advantages),
        ("Test 13: Curriculum SEC engine", test_13_sec_engine),
        ("Test 14: Pending buffer", test_14_pending_buffer),
        ("Test 15: Pareto tracker", test_15_pareto_tracker),
    ]

    for name, func in tests:
        _run_test(name, func)

    # Summary
    print("\n")
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    total_time = sum(r["elapsed"] for r in _results)
    pass_count = sum(1 for r in _results if r["status"] == "PASS")
    fail_count = sum(1 for r in _results if r["status"] == "FAIL")
    skip_count = sum(1 for r in _results if r["status"] == "SKIP")

    for r in _results:
        tag = r["status"]
        pad = " " * (4 - len(tag))
        print(f"  [{tag}]{pad} {r['name']:45s}  {r['elapsed']:.3f}s")

    print(f"\n  Total: {len(_results)} tests | "
          f"PASS: {pass_count} | FAIL: {fail_count} | SKIP: {skip_count} | "
          f"Time: {total_time:.2f}s")

    if fail_count > 0:
        print("\n  ** FAILURES DETECTED **")
        for r in _results:
            if r["status"] == "FAIL":
                print(f"     - {r['name']}: {r['detail'][:200]}")
        sys.exit(1)
    else:
        print("\n  All non-skipped tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
