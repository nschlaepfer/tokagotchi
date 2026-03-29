# tokagotchi

**Raise your own AI on a single GPU.**

tokagotchi is a self-improving AI system that runs on a single RTX 5090 (32GB). It uses Claude Opus 4.6 as a teacher/judge to continuously evolve and train a local Qwen 3.5 9B model — making it increasingly capable as a coding agent over time. The 27B model is supported for later scaling.

Think of it as a Tamagotchi for LLMs: you feed it tasks, it learns from its mistakes, and it grows stronger overnight.

## How It Works

Three training loops run at different timescales, each building on the last:

### Loop 1 — Prompt Evolution (minutes)
GEPA-style evolutionary optimization of prompts and context. No weight updates — just finding the best way to talk to your model. Opus analyzes execution traces, diagnoses failures, and proposes targeted mutations. **Forced mutation diversity** cycles through 5 high-impact types (add_example, modify_tool_instructions, strengthen_instruction, add_error_recovery, add_cot_step) to prevent defaulting to shallow rephrasing. Based on [Training-Free GRPO](https://arxiv.org/abs/2503.04644) and [GEPA](https://arxiv.org/abs/2502.02968).

### Loop 2 — On-Policy Distillation + SDPO (hours)
**Two-tier training signal generation from failed trajectories:**

1. **SDPO (free)**: Self-Distillation via Behavioral Divergence. After a failed episode, replays the trajectory through Qwen with error feedback injected. Steps where the model changes its action become contrastive training pairs — at zero API cost. See our paper on [Divergence-Gated Hierarchical Distillation](#research).

2. **Opus trace surgery (fallback)**: When SDPO produces zero contrastive pairs (the model is "confidently wrong" even after seeing the error), Opus performs targeted correction. This concentrates expensive expert supervision on the model's blind spots.

Corrections accumulate into a diversity-aware training buffer, then QLoRA fine-tuning bakes the lessons into weights. Based on [SCoRe](https://arxiv.org/abs/2504.01408), [SDPO](https://arxiv.org/abs/2601.20802), and [OPSD](https://arxiv.org/abs/2601.18734).

### Loop 3 — Reinforcement Learning (overnight)
Tree-GRPO with shared prefix rollouts for 4x efficiency. DAPO's asymmetric clipping prevents entropy collapse. RAGEN's trajectory filtering catches echo traps. Quantization noise from fitting on 32GB actually helps exploration. Based on [Tree-GRPO](https://arxiv.org/abs/2504.07641), [QeRL](https://arxiv.org/abs/2502.15405), [DAPO](https://arxiv.org/abs/2503.14476), and [RAGEN](https://arxiv.org/abs/2504.11723).

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Opus 4.6 (Claude Code headless)                    │
│  Teacher / Judge / Curriculum Designer               │
└──────────┬────────────────┬───────────────┬─────────┘
           │                │               │
    ┌──────▼──────┐  ┌──────▼──────┐  ┌────▼────────┐
    │  Loop 1     │  │  Loop 2     │  │  Loop 3     │
    │  Prompt     │  │  SDPO +     │  │  Tree-GRPO  │
    │  Evolution  │  │  Distill    │  │  RL         │
    └──────┬──────┘  └──────┬──────┘  └────┬────────┘
           │                │               │
    ┌──────▼────────────────▼───────────────▼─────────┐
    │  Qwen 3.5 9B (think=true) — RTX 5090 32GB      │
    │  Ollama serving / QLoRA training                 │
    └──────────────────────┬──────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────┐
    │  Agent Arena (subprocess sandboxes)              │
    │  bash, python, files, SQL, APIs                  │
    │  Self-Evolving Curriculum + 3-tier rewards       │
    └─────────────────────────────────────────────────┘
```

## Key Technical Details

- **Thinking mode**: Qwen 3.5 abliterated requires `think=true` — the model cannot produce content without reasoning first. The system handles this natively.
- **Action parser**: Robust multi-format parser handles Qwen's output patterns including `<think>` blocks, orphaned `</think>` tags, bracket-style `[action content]`, and reasoning text before actions.
- **Sandbox backend**: Auto-detects Docker; falls back to subprocess sandboxes with a `python→python3` shim for Windows/Git Bash compatibility.
- **Mutation lineage**: Every genome stores its mutation type, Opus's diagnosis, rationale, and creation timestamp. Full mutation history logged to `mutation_log.jsonl`.
- **Trajectory persistence**: Eval results save full step-by-step data (actions, observations, reasoning, rewards) for replay and analysis.

## Requirements

- **GPU**: NVIDIA RTX 5090 (32GB) or similar
- **OS**: Windows 11 (Git Bash / MSYS2) — Docker optional
- **CLI**: Claude Code CLI (`npm install -g @anthropic-ai/claude-code`)
- **Python**: 3.11+
- **Model**: Pulled via Ollama (`ollama pull huihui_ai/qwen3.5-abliterated:9b`)
- **Storage**: ~20GB for model weights + ~5GB for data/checkpoints

## Quick Start

```bash
# Clone and setup
git clone https://github.com/nschlaepfer/tokagotchi.git
cd tokagotchi
pip install -e .

# Pull the model
ollama pull huihui_ai/qwen3.5-abliterated:9b

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
│   ├── orchestrator/    # Opus client, budget tracker, master loop, git experiments
│   ├── loop1_gepa/      # Prompt evolution: genome, mutations, Pareto frontier
│   ├── loop2_distill/   # SDPO + distillation: trace surgery, SFT, mentor sessions
│   ├── loop3_rl/        # RL: Tree-GRPO, DAPO clipping, trajectory filtering
│   ├── arena/           # Subprocess sandboxes + tools (bash, python, SQL, APIs)
│   ├── curriculum/      # Self-Evolving Curriculum, task generation, frontier probing
│   ├── rewards/         # Outcome, process (Opus-judged), efficiency, composite
│   └── infra/           # Ollama server wrapper, VRAM scheduler, eval harness
├── paper/               # DGHD paper (LaTeX)
├── docs/                # Reference papers (PDFs)
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
| Loop 1 mutations (Opus) | ~$0.03/call, 50/hr | ~$3-5 |
| Loop 2 SDPO (local) | $0 | $0 |
| Loop 2 Opus fallback | ~$0.03-0.05/call, as needed | ~$2-5 |
| Loop 3 RL (local) | $0 (overnight) | $0 |
| **Total** | | **~$5-10/day** |

SDPO reduces Loop 2 costs by an estimated 70-80% by handling most failures locally.

## VRAM Management

The single GPU serves double duty:
- **Serving phase** (~8GB for 9B, ~17GB for 27B): Ollama runs Qwen for inference during Loops 1-2
- **Training phase** (32GB): Ollama stops, full GPU for QLoRA/GRPO

Phase transitions are automatic — the VRAM scheduler handles the lifecycle.

## Scaling to 27B

The 9B model is the proof of concept. To scale up:
1. Download: `huggingface-cli download huihui-ai/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated`
2. Pull: `ollama pull huihui_ai/qwen3.5-abliterated:27b`
3. Change `model.name` in `config/master.yaml`
4. Everything else (GEPA, SDPO, QLoRA, RL) just works on the bigger model

Optional: Build [Madreag's TurboQuant fork](https://github.com/Madreag/turbo3-cuda) for 4.6x KV cache compression on the RTX 5090 — enables 262K+ context for the 27B model.

## License

MIT
