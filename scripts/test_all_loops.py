#!/usr/bin/env python3
"""Comprehensive integration test for all tokagotchi system components.

Runs end-to-end tests WITHOUT Docker, using subprocess sandboxing instead.
Each test prints PASS / FAIL / SKIP with timing information.

Usage:
    python scripts/test_all_loops.py
"""

from __future__ import annotations

import asyncio
import argparse
import json
import os
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


def _external_probes_disabled() -> bool:
    return os.environ.get("TOKAGOTCHI_SKIP_EXTERNAL_PROBES") == "1"


def _external_probe_timeout_seconds(default: float = 120.0) -> float:
    raw = os.environ.get("TOKAGOTCHI_EXTERNAL_PROBE_TIMEOUT_SECONDS")
    if raw is None:
        return default
    try:
        return max(0.1, float(raw))
    except ValueError:
        return default


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
    import src.evaluation.task_bank_validator
    import src.evaluation.task_judge
    import src.infra.ollama_utils
    import src.orchestrator.budget_tracker
    import src.orchestrator.opus_client
    import src.orchestrator.safety_gates
    import src.usage_flywheel
    import src.usage_flywheel.codex_harness
    import src.usage_flywheel.feedback
    import src.usage_flywheel.flywheel
    import src.usage_flywheel.redaction
    import src.usage_flywheel.store
    return "All modules imported successfully"


# ===================================================================
# Test 1: Config loading
# ===================================================================

def test_01_config_loading():
    """Load config from config/ directory and verify MasterConfig structure."""
    from src.config import load_config, MasterConfig

    cfg = load_config(PROJECT_ROOT / "config")
    assert isinstance(cfg, MasterConfig), f"Expected MasterConfig, got {type(cfg)}"
    assert cfg.model.ollama_port == 11434, f"Expected port 11434, got {cfg.model.ollama_port}"
    assert cfg.opus.daily_budget_usd > 0, "Daily budget should be positive"
    assert cfg.opus.provider == "codex", f"Expected default provider codex, got {cfg.opus.provider}"
    assert cfg.opus.model == "gpt-5.6-sol", f"Expected GPT-5.6 Sol, got {cfg.opus.model}"
    assert cfg.usage_flywheel.enabled is True, "Usage flywheel should be enabled"
    assert cfg.usage_flywheel.codex_boost == "on_failure", cfg.usage_flywheel.codex_boost
    assert cfg.loop1.population_size > 0, "Population size should be positive"
    assert cfg.loop3.algorithm == "grpo", f"Expected grpo, got {cfg.loop3.algorithm}"
    assert len(cfg.loop1.pareto_objectives) >= 3, "Should have at least 3 pareto objectives"
    return f"MasterConfig loaded: student={cfg.model.name}, teacher={cfg.opus.provider}/{cfg.opus.model}"


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
# Test 3: Ollama inference
# ===================================================================

def test_03_ollama_inference():
    """Make a chat completion call to Ollama via the openai client."""
    if _external_probes_disabled():
        skip("external Ollama probe disabled by TOKAGOTCHI_SKIP_EXTERNAL_PROBES")

    try:
        from openai import OpenAI
    except ImportError:
        skip("openai package not installed")

    from src.config import load_config
    from src.infra.ollama_utils import ollama_base_urls
    cfg = load_config(PROJECT_ROOT / "config")

    # Use native Ollama API with think=false to get direct content
    # (Qwen thinking models may put content in the reasoning/thinking field)
    import requests

    errors = []
    timeout = _external_probe_timeout_seconds()
    for api_url in ollama_base_urls(cfg.model.ollama_host, cfg.model.ollama_port):
        try:
            resp = requests.post(
                f"{api_url}/api/chat",
                json={
                    "model": cfg.model.name,
                    "messages": [
                        {"role": "system", "content": "Reply in one short sentence."},
                        {"role": "user", "content": "What is 2+2?"},
                    ],
                    "stream": False,
                    "think": False,
                    "options": {
                        "num_predict": 64,
                        "temperature": 0.0,
                        "num_ctx": cfg.model.ollama_num_ctx,
                    },
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            break
        except Exception as exc:
            errors.append(f"{api_url}: {exc}")
    else:
        skip("Ollama not reachable at configured endpoints: " + "; ".join(errors))

    data = resp.json()
    content = data.get("message", {}).get("content", "")
    tokens = data.get("eval_count", 0)
    assert len(content) > 0, f"Empty response from Ollama: {data}"
    return f"Ollama responded via {api_url} ({tokens} tokens): {content[:80]}"


# ===================================================================
# Test 4: Codex CLI
# ===================================================================

def test_04_codex_cli():
    """Verify the Codex CLI is installed."""
    try:
        result = subprocess.run(
            ["codex", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        skip("codex CLI not found on PATH")
    except subprocess.TimeoutExpired:
        skip("codex CLI timed out after 10s")

    if result.returncode != 0:
        skip(f"codex CLI exited with code {result.returncode}: {result.stderr[:200]}")

    stdout = result.stdout.strip()
    assert len(stdout) > 0, "Empty stdout from codex CLI"

    return f"Codex CLI available: {stdout[:120]}"


# ===================================================================
# Test 5: Budget Tracker
# ===================================================================

def test_05_budget_tracker():
    """Create a BudgetTracker and verify subscription-mode spend tracking."""
    from src.orchestrator.budget_tracker import BudgetTracker

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

    # Subscription mode: limits are observability guardrails, not hard blockers.
    assert tracker.can_spend(0.20, loop_id="test"), "Subscription mode should not hard-block spend"

    # Check summary
    summary = tracker.get_summary()
    assert summary["num_calls"] == 2
    assert abs(summary["total_usd"] - 0.90) < 0.001
    assert summary["total_prompt_tokens"] == 180
    assert summary["total_completion_tokens"] == 90

    # Verify over-limit records are still tracked rather than rejected.
    tracker.record_spend(0.20, loop_id="over_limit")
    summary = tracker.get_summary()
    assert summary["num_calls"] == 3
    assert abs(summary["total_usd"] - 1.10) < 0.001

    return f"Budget tracker OK: {summary['num_calls']} calls, ${summary['total_usd']:.2f} total"


# ===================================================================
# Test 6: Teacher Client command builder
# ===================================================================

def test_06_teacher_client():
    """Verify OpusClient builds the default Codex command without calling a model."""

    from src.orchestrator.opus_client import OpusClient
    from src.orchestrator.budget_tracker import BudgetTracker
    from src.config import OpusConfig

    cfg = OpusConfig(
        provider="codex",
        model="gpt-5.6-sol",
        model_reasoning_effort="medium",
        daily_budget_usd=5.0,
        hourly_budget_usd=2.0,
        default_max_budget_per_call_usd=0.10,
        default_max_turns=1,
    )
    budget = BudgetTracker(hourly_limit_usd=2.0, daily_limit_usd=5.0)
    client = OpusClient(config=cfg, budget_tracker=budget)

    cmd, stdin_text = client._build_command(
        "Return {\"status\":\"ok\"}",
        json_schema={"type": "object", "required": ["status"]},
        max_turns=1,
    )
    assert cmd[:4] == ["codex", "exec", "--model", "gpt-5.6-sol"]
    assert "--sandbox" in cmd and "read-only" in cmd
    assert any("model_reasoning_effort" in arg for arg in cmd)
    assert stdin_text is None

    return f"Teacher client default OK: {' '.join(cmd[:8])}"


# ===================================================================
# Test 7: Loop 1 GEPA lite
# ===================================================================

def test_07_gepa_lite():
    """Run a mini GEPA wiring check: seed genome, optional Ollama eval, Codex mutation wiring."""

    async def _run():
        from src.loop1_gepa.prompt_genome import create_seed_genome
        from src.models import PromptGenome

        # Step 1: Create seed genome
        genome = create_seed_genome()
        assert genome.system_prompt, "Seed genome should have a system prompt"
        assert genome.generation == 0

        # Step 2: Evaluate via Ollama (lightweight -- just test the model responds)
        try:
            from openai import OpenAI
        except ImportError:
            skip("openai package not installed")

        from src.config import load_config
        from src.infra.ollama_utils import ollama_base_urls
        cfg = load_config(PROJECT_ROOT / "config")

        try:
            if _external_probes_disabled():
                ollama_ok = False
            else:
                import requests as _req
                timeout = _external_probe_timeout_seconds()
                for api_url in ollama_base_urls(cfg.model.ollama_host, cfg.model.ollama_port):
                    try:
                        resp = _req.post(
                            f"{api_url}/api/chat",
                            json={
                                "model": cfg.model.name,
                                "messages": [
                                    {"role": "system", "content": genome.to_system_message()[:500]},
                                    {"role": "user", "content": "What is the first step to fix a failing Python test?"},
                                ],
                                "stream": False,
                                "think": False,
                                "options": {
                                    "num_predict": 100,
                                    "temperature": 0.7,
                                    "num_ctx": cfg.model.ollama_num_ctx,
                                },
                            },
                            timeout=timeout,
                        )
                        resp.raise_for_status()
                        break
                    except Exception:
                        continue
                else:
                    raise RuntimeError("Ollama not reachable at configured endpoints")
                ollama_answer = resp.json().get("message", {}).get("content", "")
                assert len(ollama_answer) > 0, "Empty Ollama response"
                ollama_ok = True
        except Exception:
            ollama_ok = False

        # Step 3: Verify Codex teacher mutation command wiring without a remote call.
        from src.orchestrator.opus_client import OpusClient
        from src.orchestrator.budget_tracker import BudgetTracker

        teacher_budget = BudgetTracker(hourly_limit_usd=2.0, daily_limit_usd=5.0)
        teacher_client = OpusClient(config=cfg.opus, budget_tracker=teacher_budget)
        teacher_cmd, _ = teacher_client._build_command(
            f"Suggest one small improvement:\n\n{genome.system_prompt[:300]}",
            max_turns=1,
        )
        teacher_ok = teacher_cmd[:2] == ["codex", "exec"] if cfg.opus.provider == "codex" else len(teacher_cmd) > 0

        if not ollama_ok and not teacher_ok:
            skip("Neither Ollama nor teacher command wiring available for GEPA lite test")

        parts = []
        if ollama_ok:
            parts.append("Ollama eval OK")
        else:
            parts.append("Ollama eval SKIP")
        if teacher_ok:
            parts.append("teacher mutation wiring OK")
        else:
            parts.append("teacher mutation wiring SKIP")

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

    with tempfile.TemporaryDirectory() as tmp:
        bank_path = str(Path(tmp) / "task_bank.json")
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

        # Transactional peek, then clear after a confirmed handoff
        batch = buf.peek_training_batch()
        assert len(batch) == 6
        assert buf.size() == 6
        buf.clear()
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

    return f"Pending buffer: added 6, peeked+cleared {len(batch)}, persistence OK"


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
# Test 16: Usage flywheel trace store + redaction
# ===================================================================

def test_16_usage_flywheel_store():
    """Persist a local product-use trace and verify secret redaction."""
    import tempfile
    from src.usage_flywheel.models import UsageTrace
    from src.usage_flywheel.redaction import redact_text
    from src.usage_flywheel.store import UsageTraceStore, append_pending_example

    with tempfile.TemporaryDirectory() as tmp:
        store = UsageTraceStore(Path(tmp) / "usage")
        redacted, report = redact_text("debug this with token sk-proj-abcdefghijklmnopqrstuvwxyz123456")
        assert "[REDACTED:openai_key]" in redacted
        assert report.total_replacements == 1

        trace = UsageTrace(user_task=redacted, redaction_report=report.as_dict())
        trace.student_model = "qwen3.6:27b"
        trace.teacher_model = "gpt-5.6-sol"
        trace.student_status = "empty"
        trace.codex_status = "ok"
        trace.boost_used = True
        trace.add_event("user_task", redacted)
        trace_path = store.save(trace)

        loaded = store.load(trace.trace_id)
        assert loaded.trace_id == trace.trace_id
        assert loaded.events[0].event_type == "user_task"
        assert store.latest(1)[0]["trace_id"] == trace.trace_id
        trace.status = "updated"
        store.save(trace)
        latest = store.latest(10)
        assert len([row for row in latest if row["trace_id"] == trace.trace_id]) == 1
        assert latest[-1]["status"] == "updated"
        try:
            store.load("../escape")
            raise AssertionError("Expected unsafe trace id rejection")
        except ValueError:
            pass

        pending_path = append_pending_example(
            Path(tmp) / "pending.jsonl",
            example={"messages": [{"role": "user", "content": "x"}]},
            metadata={"source": "usage_flywheel", "trace_id": trace.trace_id},
        )
        assert pending_path.read_text(encoding="utf-8").count("\n") == 1

    return f"Usage trace persisted: {trace_path.name}, redactions={report.total_replacements}"


# ===================================================================
# Test 17: Usage flywheel dry run
# ===================================================================

def test_17_usage_flywheel_dry_run():
    """Run the product-use flywheel without external model calls."""
    import tempfile
    from src.config import load_config
    from src.usage_flywheel import UsageFlywheel, UsageTraceStore

    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(PROJECT_ROOT / "config")
            cfg.usage_flywheel.trace_dir = str(Path(tmp) / "usage")
            cfg.usage_flywheel.pending_jsonl = str(Path(tmp) / "pending.jsonl")
            store = UsageTraceStore(cfg.usage_flywheel.trace_dir)
            flywheel = UsageFlywheel(cfg, repo_root=PROJECT_ROOT, trace_store=store)
            result = await flywheel.run_task(
                "Explain how to run tokagotchi in Codex mode.",
                dry_run=True,
                append_pending=True,
            )
            assert result.trace.status == "dry_run"
            assert result.trace.student_status == "skipped"
            assert result.trace.codex_status == "skipped"
            assert result.pending_example_path is None
            assert result.trace_path.exists()
            assert not Path(cfg.usage_flywheel.pending_jsonl).exists()
            assert store.load(result.trace.trace_id).status == "dry_run"
            return result.trace.trace_id

    trace_id = asyncio.run(_run())
    return f"Usage flywheel dry-run trace OK: {trace_id}"


# ===================================================================
# Test 18: Usage flywheel feedback controls
# ===================================================================

def test_18_usage_flywheel_feedback_controls():
    """Accepted non-private traces promote; rejected/private/low-rated traces do not."""
    import tempfile

    from src.usage_flywheel.feedback import (
        apply_trace_feedback,
        promote_trace_to_pending,
        trace_trainability,
    )
    from src.usage_flywheel.models import UsageTrace
    from src.usage_flywheel.store import UsageTraceStore

    with tempfile.TemporaryDirectory() as tmp:
        usage_dir = Path(tmp) / "usage"
        pending_path = Path(tmp) / "pending.jsonl"
        store = UsageTraceStore(usage_dir)

        trace = UsageTrace(
            user_task="Explain the failing test.",
            student_output="Initial weaker answer",
            selected_output="Initial weaker answer",
            student_status="ok",
        )
        store.save(trace)
        first = trace_trainability(trace)
        assert not first.trainable
        assert first.reason == "not_accepted"

        apply_trace_feedback(
            trace,
            decision="accepted",
            rating=5,
            selected_output="Edited high-quality answer",
            note="Useful after edit.",
        )
        store.save(trace)
        accepted = trace_trainability(store.load(trace.trace_id))
        assert accepted.trainable, accepted

        result = promote_trace_to_pending(store, trace.trace_id, pending_path)
        assert result.promoted
        assert pending_path.read_text(encoding="utf-8").count("\n") == 1
        record = json.loads(pending_path.read_text(encoding="utf-8").strip())
        assert record["example"]["messages"][-1]["content"] == "Edited high-quality answer"
        assert record["metadata"]["review_status"] == "accepted"
        assert store.load(trace.trace_id).status == "promoted"
        duplicate = promote_trace_to_pending(store, trace.trace_id, pending_path)
        assert not duplicate.promoted
        assert duplicate.trainability.reason == "already_promoted"
        assert pending_path.read_text(encoding="utf-8").count("\n") == 1

        rejected = UsageTrace(user_task="Bad trace", selected_output="bad")
        apply_trace_feedback(rejected, decision="rejected", rating=5)
        store.save(rejected)
        rejected_result = promote_trace_to_pending(store, rejected.trace_id, pending_path)
        assert not rejected_result.promoted
        assert rejected_result.trainability.reason == "rejected"

        private = UsageTrace(user_task="Private trace", selected_output="private")
        apply_trace_feedback(private, decision="accepted", rating=5, mark_private=True)
        store.save(private)
        private_result = promote_trace_to_pending(store, private.trace_id, pending_path)
        assert not private_result.promoted
        assert private_result.trainability.reason == "marked_private"

        sensitive = UsageTrace(user_task="Sensitive trace", selected_output="secret")
        apply_trace_feedback(sensitive, decision="accepted", rating=5, mark_sensitive=True)
        store.save(sensitive)
        sensitive_result = promote_trace_to_pending(store, sensitive.trace_id, pending_path)
        assert not sensitive_result.promoted
        assert sensitive_result.trainability.reason == "marked_sensitive"

        low = UsageTrace(user_task="Low usefulness", selected_output="meh")
        apply_trace_feedback(low, decision="accepted", rating=2)
        store.save(low)
        low_result = promote_trace_to_pending(store, low.trace_id, pending_path)
        assert not low_result.promoted
        assert low_result.trainability.reason == "usefulness_rating_too_low"
        assert pending_path.read_text(encoding="utf-8").count("\n") == 1

    return "Usage feedback gates promote only accepted, useful, non-private traces"


# ===================================================================
# Test 19: Codex harness command construction
# ===================================================================

def test_19_codex_harness_command():
    """Verify Codex harness command flags without calling Codex."""
    from src.usage_flywheel.codex_harness import CodexHarness

    harness = CodexHarness(
        cwd=PROJECT_ROOT,
        model="gpt-5.6-sol",
        effort="medium",
        sandbox="read-only",
    )
    cmd = harness.build_command(
        "Reply OK",
        output_path=PROJECT_ROOT / "data" / "usage_traces" / "last.txt",
    )
    assert cmd[:4] == ["codex", "exec", "--model", "gpt-5.6-sol"]
    assert "--json" in cmd
    assert "--output-last-message" in cmd
    assert "--sandbox" in cmd and "read-only" in cmd
    assert any("model_reasoning_effort" in arg for arg in cmd)
    assert harness.build_command("x" * 5000, output_path="/tmp/out.txt")[-1] == "-"
    return f"Codex harness command OK: {' '.join(cmd[:8])}"


# ===================================================================
# Test 19: Arena fail-closed behavior
# ===================================================================

def test_19_arena_fail_closed():
    """Docker auto-detect must not silently fall back to host subprocess."""
    import src.arena.docker_manager as dm
    from src.arena.subprocess_manager import SubprocessManager

    command = dm._make_workspace_write_command({"nested/ok.txt": "hello from task"})
    assert "/workspace" in command
    assert "nested/ok.txt" in command
    assert "hello from task" not in command, "File contents should be base64 encoded"
    assert "mode=1777" in dm.WORKSPACE_TMPFS_OPTIONS, (
        "Docker tmpfs workspace must be writable by the non-root agent user "
        "when the container rootfs is read-only"
    )
    arena_dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile.arena").read_text()
    assert "pytest" in arena_dockerfile, "Arena image must include pytest for seed-task oracles"
    try:
        dm._make_workspace_write_command({"../escape.txt": "bad"})
        raise AssertionError("Expected traversal rejection in workspace writer")
    except ValueError:
        pass

    original_docker = dm.docker
    dm.docker = None
    try:
        try:
            dm.create_arena_manager(use_docker=None)
            raise AssertionError("Expected fail-closed Docker auto-detect")
        except dm.ArenaUnavailableError:
            pass

        try:
            dm.create_arena_manager(use_docker=False)
            raise AssertionError("Expected unsafe subprocess opt-in failure")
        except dm.UnsafeArenaBackendError:
            pass

        mgr = dm.create_arena_manager(
            use_docker=False,
            allow_unsafe_host_execution=True,
        )
        assert isinstance(mgr, SubprocessManager)
        assert mgr.inherit_environment is False
    finally:
        dm.docker = original_docker

    return "Arena factory fails closed and unsafe host backend requires explicit opt-in"


# ===================================================================
# Test 20: Host subprocess containment
# ===================================================================

def test_20_subprocess_containment():
    """Unsafe host backend must still reject path traversal and secret env inheritance."""
    import asyncio
    import os
    import time
    from src.arena.subprocess_manager import SubprocessManager
    from src.arena.tools.file_tool import read_file
    from src.models import TaskSpec

    async def _run():
        os.environ["TOKAGOTCHI_TEST_SECRET_SHOULD_NOT_LEAK"] = "leak-me"
        mgr = SubprocessManager(inherit_environment=False)
        cid = await mgr.async_create_container(TaskSpec(initial_files={"ok.txt": "OK"}))
        try:
            stdout, _, code = await mgr.async_exec_in_container(
                cid,
                "python3 - <<'PY'\nimport os\nprint(os.getenv('TOKAGOTCHI_TEST_SECRET_SHOULD_NOT_LEAK', ''))\nPY",
                timeout=5,
            )
            assert code == 0
            assert "leak-me" not in stdout

            try:
                await mgr.async_copy_files_to_container(cid, {"../escape.txt": "bad"})
                raise AssertionError("Expected copy path traversal rejection")
            except ValueError:
                pass

            result = await read_file(mgr, cid, "../../etc/passwd")
            assert result.exit_code == 1
            assert "escapes" in result.stderr

            started = time.monotonic()
            try:
                await mgr.async_exec_in_container(cid, "sleep 5", timeout=1)
                raise AssertionError("Expected timeout")
            except TimeoutError:
                elapsed = time.monotonic() - started
                assert elapsed < 3.0, f"Timeout did not terminate promptly: {elapsed:.2f}s"
        finally:
            await mgr.async_destroy_container(cid)
            os.environ.pop("TOKAGOTCHI_TEST_SECRET_SHOULD_NOT_LEAK", None)

    asyncio.run(_run())
    return "Subprocess backend strips secret env, rejects traversal, and kills timeouts"


# ===================================================================
# Test 21: Canonical TaskJudge proof
# ===================================================================

def test_21_task_judge_oracle():
    """Submit-only code task fails; reference patch passes through the same judge."""
    import asyncio
    from src.arena.subprocess_manager import SubprocessManager
    from src.evaluation.task_judge import TaskJudge
    from src.models import ActionType, StepRecord, TaskSpec, TaskType, Trajectory

    async def _run():
        task = TaskSpec(
            task_id="judge-proof",
            task_type=TaskType.CODE_DEBUGGING,
            description="Write OK to answer.txt",
            initial_files={"answer.txt": "NO\n"},
            test_commands=["grep -qx OK answer.txt"],
        )
        judge = TaskJudge(command_timeout_seconds=5)
        mgr = SubprocessManager(inherit_environment=False)

        cid = await mgr.async_create_container(task)
        try:
            submit_only = Trajectory(
                task=task,
                steps=[
                    StepRecord(
                        step_idx=0,
                        action_type=ActionType.SUBMIT,
                        action_content="done",
                        observation="",
                    )
                ],
            )
            failed = await judge.judge(submit_only, task, arena_manager=mgr, container_id=cid)
            assert failed.submitted is True
            assert failed.oracle_passed is False
            assert failed.success is False
        finally:
            await mgr.async_destroy_container(cid)

        cid = await mgr.async_create_container(task)
        try:
            await mgr.async_copy_files_to_container(cid, {"answer.txt": "OK\n"})
            reference = Trajectory(
                task=task,
                steps=[
                    StepRecord(
                        step_idx=0,
                        action_type=ActionType.SUBMIT,
                        action_content="done",
                        observation="",
                    )
                ],
            )
            passed = await judge.judge(reference, task, arena_manager=mgr, container_id=cid)
            assert passed.oracle_passed is True
            assert passed.success is True
            assert reference.success is True
        finally:
            await mgr.async_destroy_container(cid)

        opt_task = TaskSpec(
            task_id="benchmark-proof",
            task_type=TaskType.OPEN_ENDED,
            description="Return OK quickly enough",
            initial_files={
                "solver.py": "import time\n\ndef work():\n    time.sleep(0.6)\n    return 'OK'\n",
                "test_solver.py": "from solver import work\n\ndef test_work():\n    assert work() == 'OK'\n",
            },
            test_commands=["python -m pytest test_solver.py -q"],
            metadata={
                "benchmark_command": "python - <<'PY'\nfrom solver import work\nwork()\nPY",
                "baseline_seconds": 0.4,
            },
        )
        cid = await mgr.async_create_container(opt_task)
        try:
            slow = Trajectory(
                task=opt_task,
                steps=[
                    StepRecord(
                        step_idx=0,
                        action_type=ActionType.SUBMIT,
                        action_content="done",
                        observation="",
                    )
                ],
            )
            slow_result = await judge.judge(slow, opt_task, arena_manager=mgr, container_id=cid)
            assert slow_result.oracle_passed is False
            assert slow_result.success is False
            assert slow_result.failure_reason == "benchmark_slower_than_baseline"
            assert slow_result.details["benchmark_seconds"] > slow_result.details["baseline_seconds"]
        finally:
            await mgr.async_destroy_container(cid)

        cid = await mgr.async_create_container(opt_task)
        try:
            await mgr.async_copy_files_to_container(
                cid,
                {"solver.py": "def work():\n    return 'OK'\n"},
            )
            fast = Trajectory(
                task=opt_task,
                steps=[
                    StepRecord(
                        step_idx=0,
                        action_type=ActionType.SUBMIT,
                        action_content="done",
                        observation="",
                    )
                ],
            )
            fast_result = await judge.judge(fast, opt_task, arena_manager=mgr, container_id=cid)
            assert fast_result.oracle_passed is True
            assert fast_result.success is True
            assert fast_result.details["benchmark_passed"] is True
            assert fast_result.reward_components["benchmark"] > 0
        finally:
            await mgr.async_destroy_container(cid)

    asyncio.run(_run())
    return "TaskJudge: submit-only fails, reference patch passes, open-ended benchmark gate enforced"


# ===================================================================
# Test 22: Task-bank validator proof
# ===================================================================

def test_22_task_bank_validator():
    """Validator detects missing references and proves starter/reference lifecycle."""
    import asyncio
    from src.arena.subprocess_manager import SubprocessManager
    from scripts.validate_task_bank import _summarize
    from src.evaluation.task_bank_validator import TaskBankValidator
    from src.models import TaskSpec, TaskType

    async def _run():
        validator = TaskBankValidator()
        missing_ref = TaskSpec(
            task_id="missing-ref",
            task_type=TaskType.CODE_DEBUGGING,
            description="Fix answer",
            initial_files={"answer.txt": "NO\n"},
            test_commands=["grep -qx OK answer.txt"],
        )
        static = validator.validate_static(missing_ref)
        assert not static.valid
        assert "executable_task_missing_reference_files" in static.issues

        valid_task = TaskSpec(
            task_id="valid-ref",
            task_type=TaskType.CODE_DEBUGGING,
            description="Fix answer",
            initial_files={"answer.txt": "NO\n"},
            test_commands=["grep -qx OK answer.txt"],
            metadata={"reference_files": {"answer.txt": "OK\n"}},
        )
        mgr = SubprocessManager(inherit_environment=False)
        executable = await validator.validate_executable(valid_task, mgr)
        assert executable.valid, executable.issues
        assert executable.starter_result["oracle_passed"] is False
        assert executable.reference_result["success"] is True

        summary = _summarize(
            {
                "task_bank": "inline",
                "tasks": 1,
                "valid": True,
                "results": [executable.to_dict()],
            }
        )
        assert summary["executable_tasks"] == 1
        assert summary["starters_failed"] == 1
        assert summary["references_passed"] == 1
        assert summary["benchmarks_passed"] == 0

    asyncio.run(_run())
    return "Task-bank validator detects missing reference and proves starter/reference lifecycle"


# ===================================================================
# Test 23: Autonomous safety gates
# ===================================================================

def test_23_autonomous_safety_gates():
    """Autonomous SFT/RL/promotion require flags plus complete evidence."""
    import tempfile

    from src.config import load_config
    from src.orchestrator.safety_gates import (
        SafetyGateError,
        load_gate_evidence,
        require_autonomous_rl_enabled,
        require_autonomous_sft_enabled,
        require_checkpoint_promotion_enabled,
        validate_gate_evidence,
    )

    cfg = load_config(PROJECT_ROOT / "config")
    for gate in (
        require_autonomous_sft_enabled,
        require_autonomous_rl_enabled,
        require_checkpoint_promotion_enabled,
    ):
        try:
            gate(cfg)
            raise AssertionError(f"Expected {gate.__name__} to block")
        except SafetyGateError:
            pass

    with tempfile.TemporaryDirectory(prefix="tokagotchi-gate-") as tmp:
        evidence_path = Path(tmp) / "truth_gate.json"
        cfg.safety.gate_evidence_path = str(evidence_path)
        cfg.safety.enable_autonomous_sft = True
        cfg.safety.enable_autonomous_rl = True
        cfg.safety.enable_checkpoint_promotion = True

        evidence_path.write_text(
            json.dumps({"truth_grounding_passed": True}) + "\n",
            encoding="utf-8",
        )
        weak_issues = validate_gate_evidence(load_gate_evidence(evidence_path))
        assert "schema_version_must_be_1" in weak_issues
        assert "git_tree_must_be_clean" in weak_issues
        assert "missing_arena_evidence" in weak_issues
        assert "missing_reproducible_commands" in weak_issues
        for gate in (
            require_autonomous_sft_enabled,
            require_autonomous_rl_enabled,
            require_checkpoint_promotion_enabled,
        ):
            try:
                gate(cfg)
                raise AssertionError(f"Expected weak evidence to block {gate.__name__}")
            except SafetyGateError:
                pass

        host_evidence = _truth_gate_evidence(arena_backend="subprocess")
        evidence_path.write_text(json.dumps(host_evidence) + "\n", encoding="utf-8")
        host_issues = validate_gate_evidence(load_gate_evidence(evidence_path))
        assert "docker_arena_validation_required" in host_issues
        assert "unsafe_host_execution_not_allowed_for_gate" in host_issues
        try:
            require_autonomous_sft_enabled(cfg)
            raise AssertionError("Expected unsafe-host evidence to block SFT")
        except SafetyGateError:
            pass

        missing_counts = _truth_gate_evidence(arena_backend="docker")
        del missing_counts["task_bank"]["starters_failed"]
        evidence_path.write_text(json.dumps(missing_counts) + "\n", encoding="utf-8")
        count_issues = validate_gate_evidence(load_gate_evidence(evidence_path))
        assert "starter_failure_count_missing" in count_issues

        docker_evidence = _truth_gate_evidence(arena_backend="docker")
        evidence_path.write_text(json.dumps(docker_evidence) + "\n", encoding="utf-8")
        assert validate_gate_evidence(load_gate_evidence(evidence_path)) == []
        for gate in (
            require_autonomous_sft_enabled,
            require_autonomous_rl_enabled,
            require_checkpoint_promotion_enabled,
        ):
            gate(cfg)

    return (
        "Autonomous SFT, RL, and checkpoint promotion require flags plus "
        "complete Docker-backed truth-gate evidence"
    )


def _truth_gate_evidence(*, arena_backend: str) -> dict[str, Any]:
    """Build complete test evidence for safety-gate regression tests."""

    return {
        "schema_version": 1,
        "truth_grounding_passed": True,
        "human_reviewed": True,
        "git_commit": "test-commit",
        "git_dirty": False,
        "task_judge_canonical": True,
        "arena": {
            "backend": arena_backend,
            "network": "none",
            "fail_closed_checked": True,
            "unsafe_host_execution": arena_backend != "docker",
        },
        "task_bank": {
            "path": "data/curriculum/seed_tasks.json",
            "static_valid": True,
            "executable_valid": True,
            "tasks": 20,
            "executable_tasks": 20,
            "starters_failed": 20,
            "references_passed": 20,
            "benchmark_tasks": 5,
            "benchmarks_passed": 5,
            "invalid_task_ids": [],
        },
        "tests": {
            "suite": "scripts/test_all_loops.py",
            "total": 29,
            "passed": 28,
            "failures": 0,
            "skipped": 1,
        },
        "reproducible_commands": [
            {
                "command": "python3 -m compileall -q src scripts",
                "exit_code": 0,
            },
            {
                "command": "git diff --check",
                "exit_code": 0,
            },
            {
                "command": (
                    "python3 scripts/validate_task_bank.py "
                    "data/curriculum/seed_tasks.json --static-only --summary"
                ),
                "exit_code": 0,
            },
            {
                "command": (
                    "python3 scripts/validate_task_bank.py "
                    "data/curriculum/seed_tasks.json --summary"
                ),
                "exit_code": 0,
            },
            {
                "command": "python scripts/test_all_loops.py --json-out /proof/integration_tests.json",
                "exit_code": 0,
            },
        ],
    }


# ===================================================================
# Test 24: Teacher-generated oracle gate
# ===================================================================

def test_24_teacher_generated_oracle_gate():
    """Teacher-generated task oracles must not enter the active bank by default."""
    import tempfile

    from src.curriculum.sec_engine import SECEngine

    with tempfile.TemporaryDirectory(prefix="tokagotchi-sec-") as tmp:
        engine = SECEngine(Path(tmp) / "task_bank.json")
        generated = {
            "task_id": "teacher-oracle",
            "task_type": "info_gathering",
            "description": "Answer the local file question.",
            "initial_files": {"answer.txt": "42\n"},
            "expected_output": "42",
            "difficulty": 0.4,
        }

        blocked = engine.add_external_tasks(
            [generated],
            source="codex",
            allow_teacher_generated_tests=False,
        )
        assert blocked == 0
        assert engine.task_count == 0

        allowed = engine.add_external_tasks(
            [generated],
            source="codex",
            allow_teacher_generated_tests=True,
        )
        assert allowed == 1
        task = engine.get_task("teacher-oracle")
        assert task is not None
        assert task.metadata["teacher_generated"] is True
        assert task.metadata["oracle_trusted"] is True
        assert task.metadata["static_validation"]["valid"] is True

    return "Teacher-generated oracles are blocked unless explicitly trusted"


# ===================================================================
# Test 25: Strict benchmark loading
# ===================================================================

def test_25_strict_benchmark_loading():
    """EvalHarness can filter invalid benchmark tasks before optimization."""
    import tempfile

    from src.infra.eval_harness import EvalHarness

    with tempfile.TemporaryDirectory(prefix="tokagotchi-bench-") as tmp:
        path = Path(tmp) / "tasks.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "task_id": "invalid-answer",
                        "task_type": "info_gathering",
                        "description": "Missing expected output.",
                        "difficulty": 0.5,
                    },
                    {
                        "task_id": "valid-answer",
                        "task_type": "info_gathering",
                        "description": "Answer exactly 42.",
                        "expected_output": "42",
                        "difficulty": 0.5,
                    },
                ]
            ),
            encoding="utf-8",
        )

        harness = EvalHarness()
        loose = harness.load_benchmark_tasks(str(path), require_valid=False)
        strict = harness.load_benchmark_tasks(str(path), require_valid=True)

        assert len(loose) == 2
        assert loose[0].metadata["static_validation"]["valid"] is False
        assert len(strict) == 1
        assert strict[0].task_id == "valid-answer"

    return "Strict benchmark loading filters invalid task specs"


# ===================================================================
# Test 26: Docker proof runner dry-run artifact
# ===================================================================

def test_26_docker_proof_runner_dry_run():
    """Proof runner should emit a reproducible command plan without Docker."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="tokagotchi-proof-") as tmp:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/prove_docker_gate.py",
                "--dry-run",
                "--output-root",
                tmp,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        proof_files = list(Path(tmp).glob("*/proof.json"))
        assert len(proof_files) == 1, proof_files
        proof = json.loads(proof_files[0].read_text(encoding="utf-8"))
        assert proof["status"] == "not_run"
        command_plan = "\n".join(proof["command_plan"])
        assert "build -t qwen-arena:latest" in command_plan
        assert "docker/Dockerfile.proof" in command_plan
        assert "scripts/prove_docker_gate.py --inside-container" in command_plan

    return "Docker proof runner emits a dry-run command plan"


# ===================================================================
# Test 27: Tokagotchi doctor
# ===================================================================

def test_27_tokagotchi_doctor():
    """Doctor command should report dogfood readiness and locked autonomy."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="tokagotchi-doctor-") as tmp:
        json_out = Path(tmp) / "doctor.json"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/tokagotchi_doctor.py",
                "--skip-codex",
                "--skip-docker",
                "--json-out",
                str(json_out),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        report = json.loads(json_out.read_text(encoding="utf-8"))
        assert report["schema_version"] == 1
        assert report["overall"] in {"ok", "warn"}
        checks = {row["name"]: row for row in report["checks"]}
        assert checks["safety_config"]["status"] == "ok"
        assert checks["gate_evidence"]["status"] == "locked"
        assert report["autonomy_locked"] is True

    return "Doctor reports local readiness and keeps autonomy locked"


# ===================================================================
# Main
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tokagotchi integration tests.")
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to write a structured integration-test report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 60)
    print("  TOKAGOTCHI INTEGRATION TEST SUITE")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Time: {started_at}")
    print("=" * 60)

    tests = [
        ("Test 00: Imports", test_00_imports),
        ("Test 01: Config loading", test_01_config_loading),
        ("Test 02: Models serialization", test_02_models),
        ("Test 03: Ollama inference", test_03_ollama_inference),
        ("Test 04: Codex CLI", test_04_codex_cli),
        ("Test 05: Budget Tracker", test_05_budget_tracker),
        ("Test 06: Teacher Client command builder", test_06_teacher_client),
        ("Test 07: Loop 1 GEPA lite", test_07_gepa_lite),
        ("Test 08: Efficiency penalty", test_08_efficiency_penalty),
        ("Test 09: Outcome reward", test_09_outcome_reward),
        ("Test 10: Trajectory filter", test_10_trajectory_filter),
        ("Test 11: DAPO clipper", test_11_dapo_clipper),
        ("Test 12: GRPO advantages", test_12_grpo_advantages),
        ("Test 13: Curriculum SEC engine", test_13_sec_engine),
        ("Test 14: Pending buffer", test_14_pending_buffer),
        ("Test 15: Pareto tracker", test_15_pareto_tracker),
        ("Test 16: Usage flywheel store", test_16_usage_flywheel_store),
        ("Test 17: Usage flywheel dry run", test_17_usage_flywheel_dry_run),
        ("Test 18: Usage flywheel feedback controls", test_18_usage_flywheel_feedback_controls),
        ("Test 19: Codex harness command", test_19_codex_harness_command),
        ("Test 20: Arena fail-closed", test_19_arena_fail_closed),
        ("Test 21: Subprocess containment", test_20_subprocess_containment),
        ("Test 22: TaskJudge oracle", test_21_task_judge_oracle),
        ("Test 23: Task-bank validator", test_22_task_bank_validator),
        ("Test 24: Autonomous safety gates", test_23_autonomous_safety_gates),
        ("Test 25: Teacher-generated oracle gate", test_24_teacher_generated_oracle_gate),
        ("Test 26: Strict benchmark loading", test_25_strict_benchmark_loading),
        ("Test 27: Docker proof runner dry run", test_26_docker_proof_runner_dry_run),
        ("Test 28: Tokagotchi doctor", test_27_tokagotchi_doctor),
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

    report = {
        "suite": "scripts/test_all_loops.py",
        "project_root": str(PROJECT_ROOT),
        "python": sys.version.split()[0],
        "started_at": started_at,
        "total": len(_results),
        "passed": pass_count,
        "failures": fail_count,
        "skipped": skip_count,
        "elapsed_seconds": total_time,
        "success": fail_count == 0,
        "results": _results,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\n  JSON report: {args.json_out}")

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
