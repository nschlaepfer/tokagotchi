# tokagotchi

**Raise your own AI on a single GPU.**

tokagotchi is a self-improving AI system that runs on a single RTX 5090 (32GB). It uses Claude Opus 4.6 as a teacher/judge to continuously evolve and train a local Qwen 3.5 27B model — making it increasingly capable as a coding agent over time.

Think of it as a Tamagotchi for LLMs: you feed it tasks, it learns from its mistakes, and it grows stronger overnight.

## How It Works

Three training loops run at different timescales, each building on the last:

### Loop 1 — Prompt Evolution (minutes)
GEPA-style evolutionary optimization of prompts and context. No weight updates — just finding the best way to talk to your model. Opus analyzes execution traces, diagnoses failures, and proposes targeted mutations. Based on [Training-Free GRPO](https://arxiv.org/abs/2503.04644) and [GEPA](https://arxiv.org/abs/2502.02968).

### Loop 2 — On-Policy Distillation (hours)
Opus watches Qwen attempt tasks, identifies exactly where reasoning breaks down, and performs "trace surgery" — correcting the trajectory from the failure point forward. Corrections accumulate into a training buffer, then QLoRA fine-tuning bakes the lessons into weights. Based on [SCoRe](https://arxiv.org/abs/2504.01408) and on-policy distillation research.

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
    │  Prompt     │  │  Distill    │  │  Tree-GRPO  │
    │  Evolution  │  │  (QLoRA)    │  │  RL         │
    └──────┬──────┘  └──────┬──────┘  └────┬────────┘
           │                │               │
    ┌──────▼────────────────▼───────────────▼─────────┐
    │  Qwen 3.5 27B (Q4) — RTX 5090 32GB             │
    │  vLLM serving / QLoRA training                   │
    └──────────────────────┬──────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────┐
    │  Agent Arena (Docker sandboxes)                  │
    │  bash, python, files, SQL, APIs                  │
    │  Self-Evolving Curriculum + 3-tier rewards       │
    └─────────────────────────────────────────────────┘
```

## Requirements

- **GPU**: NVIDIA RTX 5090 (32GB) or similar
- **OS**: Windows 11 with WSL2 + Docker Desktop
- **API**: Claude API key (Opus 4.6) — budget ~$30-50/day
- **Python**: 3.11+
- **Storage**: ~50GB for model weights + data

## Quick Start

```bash
# Clone and setup
git clone https://github.com/nschlaepfer/tokagotchi.git
cd tokagotchi

# Install everything (deps, model, Docker image)
bash scripts/setup.sh

# Test prompt evolution (Loop 1 only, cheapest to run)
python scripts/run_loop1.py --iterations 10

# Run the full self-improving pipeline
python scripts/run_all.py
```

## Project Structure

```
tokagotchi/
├── config/              # YAML configs for all loops, arena, rewards
├── src/
│   ├── orchestrator/    # Opus client, budget tracker, master loop, git experiments
│   ├── loop1_gepa/      # Prompt evolution: genome, mutations, Pareto frontier
│   ├── loop2_distill/   # Distillation: trace surgery, SFT, mentor sessions
│   ├── loop3_rl/        # RL: Tree-GRPO, DAPO clipping, trajectory filtering
│   ├── arena/           # Docker sandboxes + tools (bash, python, SQL, APIs)
│   ├── curriculum/      # Self-Evolving Curriculum, task generation, frontier probing
│   ├── rewards/         # Outcome, process (Opus-judged), efficiency, composite
│   └── infra/           # vLLM server, VRAM scheduler, eval harness
├── docker/              # Arena sandbox Dockerfile + mock API server
├── data/                # Seed prompts, seed tasks, generated data
├── scripts/             # CLI entry points + setup
└── eval/                # Benchmarks + regression suite
```

## Key Papers

This project implements ideas from:

| Paper | What we use |
|-------|------------|
| [QeRL](https://arxiv.org/abs/2502.15405) (ICLR 2026) | Quantized RL on single GPU; noise helps exploration |
| [RAGEN](https://arxiv.org/abs/2504.11723) | StarPO framework; echo trap prevention |
| [Tree-GRPO](https://arxiv.org/abs/2504.07641) (ICLR 2026) | 4x rollout efficiency via shared prefixes |
| [DAPO](https://arxiv.org/abs/2503.14476) | Clip-Higher fixes for GRPO entropy collapse |
| [GEPA](https://arxiv.org/abs/2502.02968) (ICLR 2026) | Evolutionary prompt optimization |
| [SCoRe](https://arxiv.org/abs/2504.01408) | Student-explores, teacher-corrects distillation |
| [Training-Free GRPO](https://arxiv.org/abs/2503.04644) | Context-space optimization beats weight updates |
| [WEBRL](https://arxiv.org/abs/2411.02337) (ICLR 2025) | Self-evolving curriculum; 9x improvement |
| [AgentPRM](https://arxiv.org/abs/2502.03492) | Process rewards for agents |
| [iStar](https://arxiv.org/abs/2502.12459) (ICLR 2026) | Implicit step rewards for agentic RL |

## Cost Estimates

| Component | Rate | Daily Estimate |
|-----------|------|---------------|
| Loop 1 mutations | $0.05-0.10/call, ~50/hr | ~$8-15 |
| Loop 2 trace surgery | $0.20-0.50/call, ~20/hr | ~$10-20 |
| Process rewards | $0.15-0.30/call, ~10/hr | ~$5-10 |
| Task generation | $0.30-0.50/call, ~5/hr | ~$3-5 |
| **Total** | | **~$30-50/day** |

Budget circuit breakers automatically pause loops when limits are exceeded.

## VRAM Management

The single GPU serves double duty:
- **Serving phase** (~16GB): vLLM runs Qwen Q4 for inference during Loops 1-2
- **Training phase** (32GB): vLLM stops, full GPU for QLoRA/GRPO during overnight RL

Phase transitions are automatic — the VRAM scheduler handles the lifecycle.

## License

MIT
