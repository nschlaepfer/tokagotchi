# tokagotchi

**Raise your own AI on a single GPU.**

tokagotchi is a self-improving AI system that runs on a single RTX 5090 (32GB). It uses Codex GPT-5.6 Sol as the default teacher/judge/reviewer and a local Qwen3.6 27B model as the student. Claude Opus 5 is still supported as an optional provider if you have Claude Code access.

Think of it as a Tamagotchi for LLMs: you feed it tasks, it learns from its mistakes, and it grows stronger overnight.

## How It Works

Three training loops run at different timescales, each building on the last:

### Loop 1 — Prompt Evolution (minutes)
GEPA-style evolutionary optimization of prompts and context. No weight updates — just finding the best way to talk to your model. The configured teacher model defaults to Codex GPT-5.6 Sol, which analyzes execution traces, diagnoses failures, and proposes targeted mutations. **Forced mutation diversity** cycles through 5 high-impact types (add_example, modify_tool_instructions, strengthen_instruction, add_error_recovery, add_cot_step) to prevent defaulting to shallow rephrasing. Based on [Training-Free GRPO](https://arxiv.org/abs/2503.04644) and [GEPA](https://arxiv.org/abs/2502.02968).

### Loop 2 — On-Policy Distillation + SDPO (hours)
**Two-tier training signal generation from failed trajectories:**

1. **SDPO (free)**: Self-Distillation via Behavioral Divergence. After a failed episode, replays the trajectory through Qwen with error feedback injected. Steps where the model changes its action become contrastive training pairs — at zero API cost. See our paper on [Divergence-Gated Hierarchical Distillation](#research).

2. **Teacher trace surgery (fallback)**: When SDPO produces zero contrastive pairs (the model is "confidently wrong" even after seeing the error), the configured teacher performs targeted correction. The default teacher is Codex GPT-5.6 Sol; Claude Opus 5 remains optional.

Corrections accumulate into a diversity-aware training buffer, then **LoRA fine-tuning via Unsloth** bakes the lessons into weights. Based on [SCoRe](https://arxiv.org/abs/2504.01408), [SDPO](https://arxiv.org/abs/2601.20802), and [OPSD](https://arxiv.org/abs/2601.18734).

### Loop 3 — Reinforcement Learning (overnight)
Tree-GRPO with shared prefix rollouts for 4x efficiency. DAPO's asymmetric clipping prevents entropy collapse. RAGEN's trajectory filtering catches echo traps. Quantization noise from fitting on 32GB actually helps exploration. Based on [Tree-GRPO](https://arxiv.org/abs/2504.07641), [QeRL](https://arxiv.org/abs/2502.15405), [DAPO](https://arxiv.org/abs/2503.14476), and [RAGEN](https://arxiv.org/abs/2504.11723).

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Teacher Provider                                   │
│  Default: Codex GPT-5.6 Sol via codex exec          │
│  Optional: Claude Opus 5 via claude -p              │
└──────────┬────────────────┬───────────────┬─────────┘
           │                │               │
    ┌──────▼──────┐  ┌──────▼──────┐  ┌────▼────────┐
    │  Loop 1     │  │  Loop 2     │  │  Loop 3     │
    │  Prompt     │  │  SDPO +     │  │  Tree-GRPO  │
    │  Evolution  │  │  Unsloth    │  │  RL         │
    └──────┬──────┘  └──────┬──────┘  └────┬────────┘
           │                │               │
    ┌──────▼────────────────▼───────────────▼─────────┐
    │  Qwen3.6 27B (think=true) — RTX 5090 32GB        │
    │  Ollama serving / Unsloth LoRA training          │
    └──────────────────────┬──────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────┐
    │  Agent Arena (subprocess sandboxes)              │
    │  bash, python, files, SQL, APIs                  │
    │  Self-Evolving Curriculum + 3-tier rewards       │
    └─────────────────────────────────────────────────┘
```

## Key Technical Details

- **Training via Unsloth**: Qwen3.6 uses the newer Qwen3.5-family architecture. The supported Windows path is [Unsloth](https://unsloth.ai) rather than raw bitsandbytes 4-bit loading. See `docs/KNOWN_ISSUES.md` for details.
- **Thinking mode**: Qwen thinking models should run with `think=true` in this harness. The system handles reasoning/thinking fields natively.
- **Action parser**: Robust multi-format parser handles Qwen's output patterns including `<think>` blocks, orphaned `</think>` tags, bracket-style `[action content]`, and reasoning text before actions.
- **Subscription auth**: Default teacher calls route through `codex exec`, so normal Codex subscription login is used instead of project-level API keys. If you opt into Claude, calls route through `claude -p`.
- **Sandbox backend**: Auto-detects Docker; falls back to subprocess sandboxes with a `python->python3` shim for Windows/Git Bash compatibility.
- **Mutation lineage**: Every genome stores its mutation type, teacher diagnosis, rationale, and creation timestamp. Full mutation history logs to `mutation_log.jsonl`.
- **Trajectory persistence**: Eval results save full step-by-step data (actions, observations, reasoning, rewards) for replay and analysis.
- **Weights & Biases**: Real-time tracking of genome evals, SDPO pairs, training loss, budget, and pipeline status at [wandb.ai](https://wandb.ai).

## Model Defaults

Verified on 2026-07-27:

| Role | Default | Why |
|------|---------|-----|
| Student / local serving | `qwen3.6:27b` | Current open Qwen3.6 dense 27B model with Ollama support; fits a 32GB RTX 5090 in Q4_K_M. See [Qwen3.6-27B](https://qwen.ai/blog?id=qwen3.6-27b) and [Ollama qwen3.6](https://ollama.com/library/qwen3.6). |
| Teacher / judge / reviewer | `gpt-5.6-sol` with `medium` effort | OpenAI's GPT-5.6 flagship model for complex coding/reasoning. OpenAI lists Sol, Terra, and Luna; this repo defaults to Sol because quality matters more than throughput for mutations, trace surgery, and reviews. See [OpenAI model guidance](https://developers.openai.com/api/docs/guides/latest-model) and [Codex models](https://developers.openai.com/codex/models). |
| Optional Claude provider | `claude-opus-5` with `high` effort | Kept for users with Claude Code access. To use it, set `opus.provider: "claude"`, `opus.model: "claude-opus-5"`, and `opus.model_reasoning_effort: "high"` in `config/master.yaml`. See [Claude models](https://platform.claude.com/docs/en/about-claude/models/overview) and [Claude Code model settings](https://code.claude.com/docs/en/settings). |

## Requirements

- **GPU**: NVIDIA RTX 5090 (32GB) or similar
- **OS**: Windows 11 (Git Bash / MSYS2) — Docker optional
- **CLI**: Codex CLI logged into your ChatGPT/Codex subscription. Claude Code CLI is optional and only needed if you switch `opus.provider` to `claude`.
- **Python**: 3.11+
- **Model**: Pulled via Ollama (`ollama pull qwen3.6:27b`)
- **Training**: `pip install unsloth triton-windows`; for LoRA training, download `Qwen/Qwen3.6-27B` to `models/Qwen3.6-27B`
- **Storage**: ~25GB for Ollama-only use; ~80GB+ if keeping Hugging Face weights and checkpoints for training

## Quick Start

```bash
# Clone and setup
git clone https://github.com/nschlaepfer/tokagotchi.git
cd tokagotchi
pip install -e .
pip install unsloth triton-windows

# Login/check CLIs
codex login
codex --version

# Pull the local student model
ollama pull qwen3.6:27b

# Download HF weights for Loop 2 / Loop 3 training
huggingface-cli download Qwen/Qwen3.6-27B --local-dir models/Qwen3.6-27B

# Run the full self-improving pipeline
python scripts/run_all.py --config config/ --log-file data/logs/run.log --log-level INFO

# Or run just Loop 1 (prompt evolution, cheapest)
python scripts/run_loop1.py --iterations 10
```

## Project Structure

```
tokagotchi/
├── config/              # YAML configs for all loops, arena, rewards
├── src/
│   ├── orchestrator/    # Teacher client, budget tracker, master loop, git experiments
│   ├── loop1_gepa/      # Prompt evolution: genome, mutations, Pareto frontier
│   ├── loop2_distill/   # SDPO + Unsloth SFT: trace surgery, training, mentor sessions
│   ├── loop3_rl/        # RL: Tree-GRPO, DAPO clipping, trajectory filtering
│   ├── arena/           # Subprocess sandboxes + tools (bash, python, SQL, APIs)
│   ├── curriculum/      # Self-Evolving Curriculum, task generation, frontier probing
│   ├── rewards/         # Outcome, process (teacher-judged), efficiency, composite
│   └── infra/           # Ollama server, VRAM scheduler, eval harness, wandb tracker
├── paper/               # DGHD paper (LaTeX)
├── docs/                # Reference papers (PDFs) + KNOWN_ISSUES.md
├── data/                # Seed prompts, seed tasks, generated data, checkpoints
├── scripts/             # CLI entry points + setup
└── eval/                # Benchmarks + regression suite
```

## Research

This project introduces **Divergence-Gated Hierarchical Distillation (DGHD)** — a novel training framework that uses behavioral divergence as a gating mechanism to route failed trajectories between free self-distillation and costly expert supervision. The core insight: when a model doesn't change its behavior after seeing error feedback, it's in a blind spot that only an external teacher can fix. See `paper/main.tex` for the full writeup.

### Key Papers

| Paper | What we use |
|-------|------------|
| [SDPO](https://arxiv.org/abs/2601.20802) (2026) | Self-distillation from feedback; 6x faster than GRPO |
| [OPSD](https://arxiv.org/abs/2601.18734) (2026) | On-policy self-distillation; 10-100x cheaper than RL |
| [QeRL](https://arxiv.org/abs/2502.15405) (ICLR 2026) | Quantized RL on single GPU; noise helps exploration |
| [RAGEN](https://arxiv.org/abs/2504.11723) | StarPO framework; echo trap prevention |
| [Tree-GRPO](https://arxiv.org/abs/2504.07641) (ICLR 2026) | 4x rollout efficiency via shared prefixes |
| [DAPO](https://arxiv.org/abs/2503.14476) | Clip-Higher fixes for GRPO entropy collapse |
| [GEPA](https://arxiv.org/abs/2502.02968) (ICLR 2026) | Evolutionary prompt optimization |
| [SCoRe](https://arxiv.org/abs/2504.01408) | Student-explores, teacher-corrects distillation |
| [Training-Free GRPO](https://arxiv.org/abs/2503.04644) | Context-space optimization beats weight updates |
| [WEBRL](https://arxiv.org/abs/2411.02337) (ICLR 2025) | Self-evolving curriculum; 9x improvement |
| [HCAPO](https://arxiv.org/abs/2603.08754) (2026) | Hindsight credit assignment for long-horizon agents |

## Cost Estimates

| Component | Rate | Daily Estimate |
|-----------|------|---------------|
| Loop 1 mutations | Codex subscription / provider-limited | tracked for observability |
| Loop 2 SDPO (local) | $0 | $0 |
| Loop 2 teacher fallback | Codex subscription / provider-limited | tracked for observability |
| Optional Claude mode | Claude Code subscription / provider-limited | only used if `opus.provider: "claude"` |
| Loop 2 training (local, Unsloth) | $0 | $0 |
| Loop 3 RL (local) | $0 (overnight) | $0 |
| **Local compute total** | | **$0 marginal API spend** |

SDPO reduces teacher usage by handling most failures locally before escalating to teacher trace surgery.

## VRAM Management

The single GPU serves double duty:
- **Serving phase** (~8-20GB): Ollama runs Qwen for inference during Loops 1-2
- **Training phase** (~8GB via Unsloth 4-bit): Ollama stops, Unsloth loads model for LoRA training

Phase transitions are automatic — the VRAM scheduler stops Ollama (with retry + nvidia-smi verification), runs training, then restarts serving.

## Scaling Beyond 27B

The default is now the 27B dense model. To experiment with the larger sparse coding model:

1. Download: `huggingface-cli download Qwen/Qwen3.6-35B-A3B --local-dir models/Qwen3.6-35B-A3B`
2. Pull: `ollama pull qwen3.6:35b-a3b`
3. Change `model.name`, `model.base_ollama_model`, and `model.hf_model_path` in `config/master.yaml`
4. Re-run the smoke tests before starting a long training loop

Optional: Build [Madreag's TurboQuant fork](https://github.com/Madreag/turbo3-cuda) for 4.6x KV cache compression on the RTX 5090 — enables 262K+ context for the 27B model.

## License

MIT
