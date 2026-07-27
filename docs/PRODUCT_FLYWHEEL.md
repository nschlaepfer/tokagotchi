# Tokagotchi Product Flywheel

Updated: 2026-07-27

## Goal

Tokagotchi should learn from real local use, not only from synthetic benchmark loops.

The product loop is:

1. A user gives tokagotchi a real local task.
2. The local student model tries the task first.
3. Tokagotchi records a local trace: task, repo metadata, student output, status, and later feedback.
4. If the student fails, stalls, or is unavailable, Codex GPT-5.6 Sol can rescue the task through `codex exec`.
5. The rescued result becomes a high-quality local supervision example.
6. GEPA/OmniGEPA-style optimizers use these traces to improve prompts, harness settings, curriculum, and eventually LoRA/RL training data.

The point is not to collect random chat logs. The point is to collect structured examples where a weaker local model attempted real work and a stronger harness produced a better outcome.

## Current implementation

The first product-use slice lives in `src/usage_flywheel/`:

- `models.py` defines `UsageTrace` and `UsageEvent`.
- `redaction.py` redacts common secret/token patterns before persistence.
- `store.py` writes full traces under `data/usage_traces/traces/` and an append-only `usage_index.jsonl`.
- `codex_harness.py` wraps the installed open-source Codex CLI harness through `codex exec`.
- `flywheel.py` runs the local student attempt, optional Codex boost, and pending-buffer export.
- `src/infra/ollama_utils.py` resolves Ollama endpoints and falls back from WSL `localhost` to the Windows host gateway when needed.
- `scripts/run_usage_flywheel.py` is the CLI entry point.

Generated traces are ignored by Git by default because they may contain user work, repo context, or private task details.

## Why use the Codex CLI harness this way

Codex CLI is the right boundary to start with because it already supplies the agent loop: local workspace access, sandbox modes, model selection, JSON event output, and final-message capture. The current wrapper uses:

```bash
codex exec \
  --model gpt-5.6-sol \
  -c 'model_reasoning_effort="medium"' \
  --sandbox read-only \
  --json \
  --output-last-message /tmp/result.txt \
  "task..."
```

Do not vendor the entire Codex repository yet. That adds maintenance burden before tokagotchi needs to patch Codex internals. The stable first step is a harness contract around the CLI. Vendoring becomes worth it only if tokagotchi needs deeper control over tool traces, session persistence, custom policies, or local OSS-provider orchestration that the CLI cannot expose.

## Privacy boundary

Default mode is local trace storage:

- traces stay under `data/usage_traces/`;
- common API keys, GitHub tokens, Slack tokens, AWS access keys, private-key blocks, and obvious secret assignments are redacted before persistence;
- traces are not committed because `data/usage_traces/` is in `.gitignore`;
- Codex boosting still sends task context to Codex/OpenAI through the user-authenticated Codex CLI.

If a task is too sensitive to send to Codex, run:

```bash
python scripts/run_usage_flywheel.py "your task" --codex-boost never
```

## Usage

Dry-run the trace path without external model calls:

```bash
python scripts/run_usage_flywheel.py "Explain how to run tokagotchi in Codex mode." --dry-run
```

Run the local student first, then let Codex rescue only if the student fails or is unavailable:

```bash
python scripts/run_usage_flywheel.py "Fix the failing unit test and explain the change."
```

The default local student request uses `num_ctx=2048`. This is intentional for 32GB VRAM stability with `qwen3.6:27b`; raise it only after a fresh smoke test passes.

Skip the local student and use Codex as the boosted teacher for a trace:

```bash
python scripts/run_usage_flywheel.py "Create a minimal repro for the parser bug." --skip-student --codex-boost always
```

Allow Codex to write in the workspace for that task:

```bash
python scripts/run_usage_flywheel.py "Update the docs for the flywheel." --skip-student --codex-boost always --write
```

## How this feeds training

When a usable student or Codex answer exists, the flywheel appends one chat-format example to the pending SFT buffer:

```json
{
  "example": {
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  },
  "metadata": {
    "source": "usage_flywheel",
    "trace_id": "...",
    "task_type": "real_user_task",
    "boosted_by_codex": true
  }
}
```

That keeps the existing Loop 2 pending-buffer shape. The next hardening step is adding explicit user accept/reject feedback before promotion to training.

## OmniGEPA research notes

GEPA's core idea is trace-aware reflective optimization: capture execution traces, convert them into actionable side information, mutate candidate artifacts, and keep candidates that improve a score.

The recent OmniGEPA work generalizes that idea. Its `optimize_anything` interface treats prompts, code, harnesses, configurations, and other artifacts as candidates with the same candidate/score/feedback loop. The system compares multiple optimizer families:

- GEPA: a reflective LLM mutates a Pareto parent.
- AutoResearch: a long-horizon coding-agent session owns more of the optimization loop.
- Meta-Harness: an agent proposes candidate mutations while the framework owns evaluation and selection.

The key result for tokagotchi is not the exact leaderboard number. The useful design lesson is that no single optimizer dominated every problem. OmniGEPA improved results by running or sequencing different optimizers and continuing from the best candidate.

For tokagotchi, that means the optimizer should not only mutate prompts. It should eventually mutate:

- system prompts and tool instructions;
- Codex boost policy (`never`, `on_failure`, `always`);
- sandbox mode for a task class;
- student decoding parameters;
- curriculum sampling;
- trace-to-training-example filters;
- redaction and context-selection policy;
- evaluator prompts and acceptance thresholds.

## Success criteria

This goal is not complete when the code merely runs once. Success means:

1. A user can run a real task through the flywheel from the CLI.
2. The local student can attempt the task through Ollama when available.
3. Codex GPT-5.6 Sol is the default boosted teacher path.
4. Claude Code / Opus remains available only as an optional provider.
5. A trace is persisted locally with repo metadata, statuses, events, redaction info, and selected output.
6. A usable answer can become a pending SFT example in the existing Loop 2 format.
7. Generated traces are not committed.
8. Tests cover trace persistence, redaction, dry-run behavior, and Codex harness command construction.
9. The default Qwen model is actually pulled and smoke-tested locally, or the exact blocker is recorded.
10. The docs state what is local, what is sent to Codex, and what remains future work.

## Validation status on 2026-07-27

- `qwen3.6:27b` was pulled through Ollama and listed locally as a 17GB model.
- A first unbounded smoke test failed with CUDA out-of-memory during model startup.
- A bounded smoke test with `num_ctx=2048`, `num_predict=16`, and `temperature=0` returned `TOKA_QWEN_OK`.
- In WSL, `localhost:11434` was not reachable, but the Windows host gateway `172.24.224.1:11434` was reachable. The code now tries that gateway automatically after the configured localhost endpoint.
- The integration suite passed with 18 passes, 1 skip, and 0 failures. The only skip was the DAPO clipper because PyTorch was not installed in the lightweight validation venv.
- Direct flywheel smokes passed for both the local student path (`TOKA_FLYWHEEL_OK`) and the Codex boost path (`TOKA_CODEX_FLYWHEEL_OK`).
- The Ollama-backed compatibility server passed a start/chat/stop smoke through the WSL gateway (`TOKA_SERVER_OK`).

## Future work

- Add explicit user feedback: accept, reject, edit, rate.
- Add per-task context manifests so file content is captured intentionally, not accidentally.
- Store Codex JSONL event streams in a compact normalized form.
- Add an OmniGEPA adapter that optimizes flywheel configuration and training-example filters.
- Add a promotion gate so only accepted traces move from `pending` to training.
- Evaluate post-training deltas on held-out real-use traces, not only synthetic tasks.

## References

- Codex CLI repository: https://github.com/openai/codex
- Codex CLI docs: https://learn.chatgpt.com/docs/codex/cli
- GEPA docs: https://gepa-ai.github.io/gepa/
- OmniGEPA announcement: https://gepa-ai.github.io/gepa/blog/2026/07/22/optimize-anything-omni/
- Qwen3.6 27B on Ollama: https://ollama.com/library/qwen3.6:27b
- Qwen3.6 27B on Hugging Face: https://huggingface.co/Qwen/Qwen3.6-27B
