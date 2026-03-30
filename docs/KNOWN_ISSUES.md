# Known Issues & Workarounds

## bitsandbytes 4-bit Segfault on Qwen 3.5

**Status**: Active workaround in place
**Affected**: QLoRA training (Loop 2 SFT, Loop 3 RL)
**Date**: March 29, 2026

### Problem

Loading Qwen 3.5 (9B or 27B) with `BitsAndBytesConfig(load_in_4bit=True)` causes a segmentation fault during model weight loading. The crash happens deep in the CUDA quantization kernels.

### Root Cause

Qwen 3.5 uses **Gated Delta Networks** (linear attention layers) which are not fully supported by bitsandbytes 0.49.x 4-bit quantization. The combination of:
- bitsandbytes 0.49.2
- transformers 5.3.0
- PyTorch 2.11.0+cu128
- NVIDIA driver 595.79 (RTX 5090, SM120/Blackwell)

triggers a segfault in `bitsandbytes.backends.cuda.ops` during the NF4 quantization of the Gated Delta Network weight matrices.

### Workaround

Use BF16 loading instead of 4-bit quantization:

```python
# CRASHES:
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb_config)

# WORKS:
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto")
```

### VRAM Impact

| Config | 9B Model | 27B Model | Training Headroom (32GB) |
|--------|----------|-----------|--------------------------|
| 4-bit | ~6 GB | ~16 GB | ~26 GB / ~16 GB |
| BF16 | ~13 GB | ~54 GB (won't fit) | ~19 GB / N/A |

For the 9B model, BF16 uses ~13GB leaving ~19GB for LoRA + optimizer — sufficient for training on RTX 5090 32GB.

For the 27B model, BF16 won't fit. Options when scaling up:
1. Wait for bitsandbytes fix for Gated Delta Networks
2. Use 8-bit quantization (`load_in_8bit=True`) — untested
3. Use GPTQ/AWQ pre-quantized weights from HuggingFace
4. Use Unsloth which has its own quantization path

### Reproduction

```bash
cd /e/Documents/qwen-self-improve
python -c "
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    'models/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated',
    quantization_config=bnb, device_map='auto', trust_remote_code=True
)
"
# Expected: Segmentation fault (exit code 139)
```

### Files Affected

- `src/loop2_distill/sft_launcher.py` — Changed from 4-bit to BF16 loading
- `src/infra/vram_scheduler.py` — VRAM target may need adjustment for BF16

---

## Ollama Model Unload Timing

**Status**: Mitigated with retry loop
**Affected**: VRAM phase transitions (serving → training)

### Problem

`POST /api/generate {"keep_alive": 0}` requests Ollama to unload the model, but VRAM isn't freed immediately. The original code checked VRAM once after the request and proceeded, finding only 9.3GB free (model still in GPU memory).

### Workaround

`VLLMServer.stop()` now retries up to 3 times with 3-second waits, checking nvidia-smi after each attempt. Proceeds when free VRAM exceeds 30,000 MiB.

### Files Affected

- `src/infra/vllm_server.py` — Retry loop with GPU memory verification

---

## Qwen 3.5 Abliterated Requires think=true

**Status**: Fixed in codebase
**Affected**: All inference calls

### Problem

The `huihui_ai/qwen3.5-abliterated` model cannot produce content with `think: false`. With thinking disabled, the model generates `<think>` as its first token, which Ollama suppresses, resulting in `content: ""` (empty string) for every response.

### Fix

`VLLMServer.chat_completion()` defaults to `think=True`. The model puts reasoning in the `thinking` field and the actual action in `content`.

### Files Affected

- `src/infra/vllm_server.py` — Default `think=True`
- `src/arena/game.py` — Action parser strips `<think>` blocks and orphaned `</think>` tags
