# tokagotchi

**Raise your own coding agent on Apple Silicon.**

This branch tracks `main`'s newer orchestration and GEPA work, but pivots the local inference path to **MLX** instead of Ollama/vLLM. The default setup now assumes:

- Apple Silicon
- `mlx-lm` serving a local model over an OpenAI-compatible endpoint
- Claude Opus as the teacher/judge for prompt evolution and trace analysis

It also vendors [Flash-MoE](./vendor/flash-moe) and the accompanying macOS/iOS app work under `vendor/flash-moe/FlashMoE-iOS`.

## Current Branch Focus

### Loop 1
Fully aligned with the MLX path. `scripts/run_loop1.py`, the merged GEPA engines, and the smoke/integration scripts now target a local MLX-backed OpenAI-compatible server by default.

### Loop 2 / Loop 3
The latest `main` training code is present, but the training stack is still the existing Hugging Face / PyTorch path. It has not been ported to MLX training in this branch.

## Default Runtime

The default model config in `config/master.yaml` points to:

```yaml
model:
  provider: "mlx"
  name: "mlx-community/Qwen3-14B-4bit"
  server_host: "127.0.0.1"
  server_port: 8080
```

By default the local server manager will auto-start:

```bash
python -m mlx_lm.server --model mlx-community/Qwen3-14B-4bit --host 127.0.0.1 --port 8080
```

You can still switch `model.provider` to `ollama`, `vllm`, or another OpenAI-compatible endpoint if you want to override the MLX default.

## Quick Start

```bash
git clone https://github.com/nschlaepfer/tokagotchi.git
cd tokagotchi

bash scripts/setup.sh
python scripts/smoke_test.py
python scripts/run_loop1.py --iterations 10
```

## Project Structure

```text
tokagotchi/
├── config/
├── data/
├── docker/
├── scripts/
├── src/
│   ├── arena/
│   ├── curriculum/
│   ├── infra/            # Local server manager, scheduler, eval harness
│   ├── loop1_gepa/
│   ├── loop2_distill/
│   ├── loop3_rl/
│   └── orchestrator/
└── vendor/
    └── flash-moe/
```

## Notes

- `scripts/setup.sh` installs the `mlx` extra on macOS and the `training` extra elsewhere.
- The DSPy GEPA path merged from `main` is preserved. The default config still leaves `loop1.use_dspy_gepa: false`, but MLX-backed OpenAI-compatible endpoints are now wired in if you enable it.
- The historical module name `src/infra/vllm_server.py` remains for compatibility, even though it now manages MLX/Ollama/vLLM backends.

## License

MIT
