# Truth and Safety Gates

Updated: 2026-07-27

This document defines the current proof boundary for tokagotchi self-improvement.

The short version: tokagotchi may collect traces and run judged evaluations, but autonomous SFT, RL, and checkpoint promotion are disabled by default until the task bank and evaluation evidence are trustworthy.

## Current safety posture

### 1. Arena execution is fail-closed

`create_arena_manager()` now requires Docker by default. If Docker is unavailable, tokagotchi raises an arena error instead of silently falling back to host subprocess execution.

Host subprocess execution still exists for local development, but it requires an explicit unsafe opt-in:

```bash
python scripts/run_loop1.py --sandbox subprocess --unsafe-host-code-execution
python scripts/run_loop2.py --sandbox subprocess --unsafe-host-code-execution
python scripts/run_loop3.py --sandbox subprocess --unsafe-host-code-execution
```

Even in unsafe host mode, the subprocess backend strips inherited environment secrets, rejects workspace path traversal, and kills timed-out process groups.

### 2. `TaskJudge` is the canonical success authority

A `submit` action only means the agent ended the episode. It is not success.

The canonical rule is:

```text
success = submitted AND oracle_passed AND no_safety_violations
```

Any task with `test_commands` uses those commands as the live executable oracle. Answer-only Info/API tasks require `expected_output`. Open-ended tasks require correctness commands plus `benchmark_command` and `baseline_seconds`; success requires both correctness and measured benchmark runtime at or below the declared baseline.

The judge writes structured evaluation evidence onto each trajectory:

- `submitted`
- `oracle_passed`
- `success`
- `partial_score`
- per-command test results
- safety violations
- failure reason

### 3. Task banks must validate before optimization

Use:

```bash
python scripts/validate_task_bank.py data/curriculum/seed_tasks.json --static-only
python scripts/validate_task_bank.py data/curriculum/seed_tasks.json --unsafe-host-code-execution --command-timeout-seconds 5 --summary
```

or, after installing the package:

```bash
tokagotchi-validate-tasks data/curriculum/seed_tasks.json --static-only
```

For default executable proof, run without `--static-only` with Docker available. On a local development machine without Docker, `--unsafe-host-code-execution` forces the explicit host-subprocess backend.

Strict benchmark loading can filter invalid tasks before GEPA/evaluation uses them. The master loop and Loop 1 default to validated tasks.

### 4. Model-generated task oracles are untrusted by default

Teacher/Codex-generated tasks are tagged:

```json
{
  "teacher_generated": true,
  "oracle_trusted": false
}
```

Tasks with model-generated `test_commands` or `expected_output` are blocked from entering the active curriculum unless the caller explicitly sets `allow_teacher_generated_tests=true` and the task passes static validation.

### 5. Autonomous training and promotion are disabled by default

These config flags default to `false`:

```yaml
safety:
  allow_unsafe_host_execution: false
  enable_autonomous_sft: false
  enable_autonomous_rl: false
  enable_checkpoint_promotion: false
  allow_teacher_generated_tests: false
  gate_evidence_path: "./data/safety/truth_gate.json"
```

Enabling SFT, RL, or checkpoint promotion also requires complete gate evidence at `gate_evidence_path`. A bare `truth_grounding_passed: true` file is rejected.

Loop 2 reads the pending buffer transactionally for training. It only clears examples after serving-model promotion succeeds; if training, export, or a safety gate fails, the pending examples stay available for review/retry.

The evidence must also identify a clean git tree. A dirty working tree cannot satisfy the autonomous-learning gate because the proof would not identify exactly what code was validated.

### 6. Docker proof runner writes reviewable evidence

Use:

```bash
python3 scripts/prove_docker_gate.py
```

This builds the arena image, builds a clean proof image, runs validation inside that proof image, and writes artifacts under `data/proofs/docker_gate/<timestamp>/`.

The generated `truth_gate_candidate.json` keeps `human_reviewed: false` by design. It is a candidate for human review, not an automatic unlock file.

The proof requires a mountable Unix Docker socket for nested arena validation. See `docs/DOCKER_PROOF.md`.

## Gate criteria before enabling autonomous learning

Do not enable autonomous SFT/RL/promotion until all of these are true:

1. The task bank passes static validation.
2. Executable tasks pass validation: starter state fails, reference state passes.
3. Evaluation uses `TaskJudge` output, not heuristic `submit` checks.
4. Full integration tests pass.
5. Docker arena runtime is available and hardened.
6. A held-out benchmark report exists and is reproducible from committed code.
7. A human has reviewed the evidence file before toggling safety flags.

Example evidence shape:

```json
{
  "schema_version": 1,
  "truth_grounding_passed": true,
  "human_reviewed": true,
  "git_commit": "COMMITTED_SHA",
  "git_dirty": false,
  "task_judge_canonical": true,
  "arena": {
    "backend": "docker",
    "network": "none",
    "fail_closed_checked": true,
    "unsafe_host_execution": false
  },
  "task_bank": {
    "path": "data/curriculum/seed_tasks.json",
    "static_valid": true,
    "executable_valid": true,
    "tasks": 20,
    "executable_tasks": 20,
    "starters_failed": 20,
    "references_passed": 20,
    "benchmark_tasks": 5,
    "benchmarks_passed": 5,
    "invalid_task_ids": []
  },
  "tests": {
    "suite": "scripts/test_all_loops.py",
    "total": 29,
    "passed": 28,
    "failures": 0,
    "skipped": 1
  },
  "reproducible_commands": [
    {
      "command": "python3 -m compileall -q src scripts",
      "exit_code": 0
    },
    {
      "command": "git diff --check",
      "exit_code": 0
    },
    {
      "command": "python3 scripts/validate_task_bank.py data/curriculum/seed_tasks.json --static-only --summary",
      "exit_code": 0
    },
    {
      "command": "python3 scripts/validate_task_bank.py data/curriculum/seed_tasks.json --summary",
      "exit_code": 0
    },
    {
      "command": "python scripts/test_all_loops.py --json-out /proof/integration_tests.json",
      "exit_code": 0
    }
  ],
  "proof_artifacts": {
    "proof_dir": "data/proofs/docker_gate/TIMESTAMP",
    "inside_results": "data/proofs/docker_gate/TIMESTAMP/inside-results.json",
    "integration_tests": "data/proofs/docker_gate/TIMESTAMP/integration_tests.json",
    "task_bank_static": "data/proofs/docker_gate/TIMESTAMP/task_bank_static.json",
    "task_bank_docker": "data/proofs/docker_gate/TIMESTAMP/task_bank_docker.json"
  }
}
```

## Verification proof from 2026-07-27

Commands run:

```bash
python3 -m compileall -q src scripts
git diff --check
/tmp/tokagotchi-proof-venv/bin/python scripts/test_all_loops.py --json-out /tmp/tokagotchi-local-tests.json
python3 scripts/validate_task_bank.py data/curriculum/seed_tasks.json --static-only
python3 scripts/validate_task_bank.py data/curriculum/seed_tasks.json --unsafe-host-code-execution --command-timeout-seconds 5 --summary
python3 scripts/tokagotchi_doctor.py --json-out /tmp/tokagotchi-doctor.json
python3 scripts/prove_docker_gate.py --dry-run
```

Results:

- Compile: passed.
- Whitespace diff check: passed.
- Integration suite in a dependency-complete venv: 29 total, 28 pass, 1 skip, 0 fail.
- Skip reason: PyTorch was not installed for the DAPO clipper test in the lightweight validation environment.
- Current seed task bank before repair: failed validation, as expected.
  - 4/20 tasks were statically valid.
  - Invalid tasks were missing expected outputs, reference files, or open-ended benchmark metadata.
- Current repaired seed task bank: passed static validation.
- Current repaired seed task bank: passed executable starter/reference validation with 20/20 executable tasks, 20/20 starters failing, 20/20 references passing, and 5/5 optimization benchmark proofs passing for reference artifacts.
- Docker proof runner dry-run: passed and emitted the exact command plan.
- Docker proof runner real run in current WSL: blocked cleanly because Docker daemon integration is unavailable; it wrote a blocked proof artifact under `data/proofs/docker_gate/`.
- GitHub Actions Docker proof passed for commit `d2a17ce396ae4e00ecc1497d51133dbd3227e029` in run `30316051838` on 2026-07-28 UTC:
  - `truth_grounding_passed: true`
  - git tree clean
  - task bank valid: 20/20 tasks, 20/20 starters failed, 20/20 references passed, 5/5 benchmarks passed
  - integration suite: 29 total, 26 passed, 0 failed, 3 skipped

That failed seed-bank validation is intentional proof that the validator catches weak benchmark entries. The repaired seed bank must pass both static validation and executable starter/reference validation before it can become trusted benchmark evidence.

## Remaining risks

- The generated `truth_gate_candidate.json` still needs human review before it can become local gate evidence.
- Loop 3 RL math/checkpoint quality is not scientifically revalidated; it is gated off.
- Product-use trace feedback controls are currently CLI-based; a richer UI can come later, but pending-buffer promotion is now gated by explicit acceptance.
