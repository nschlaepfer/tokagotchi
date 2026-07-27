# Known Issues & Workarounds

## bitsandbytes 4-bit Segfault on Qwen 3.5 / Qwen3.6-family training

**Status**: Mitigated via Unsloth; raw bitsandbytes is disabled by default
**Affected**: QLoRA training (Loop 2 SFT, Loop 3 RL)
**Date**: March 29, 2026
**Resolution Date**: March 29, 2026

### Problem

Loading Qwen 3.5 (9B or 27B) with `BitsAndBytesConfig(load_in_4bit=True)` caused a segmentation fault during model weight loading. The crash happened deep in the CUDA quantization kernels.

The current default is Qwen3.6 27B. Keep the Unsloth path for Qwen3.6 unless raw bitsandbytes 4-bit loading has been explicitly revalidated on the target driver/CUDA stack. Do not "fix" this by switching training back to raw `BitsAndBytesConfig(load_in_4bit=True)`.

### Root Cause

Qwen 3.5 uses **Gated Delta Networks** (linear attention layers) which were not fully supported by bitsandbytes 0.49.x 4-bit quantization. The failing combination was:
- bitsandbytes 0.49.2
- transformers 5.3.0
- PyTorch 2.11.0+cu128
- NVIDIA driver 595.79 (RTX 5090, SM120/Blackwell)

triggers a segfault in `bitsandbytes.backends.cuda.ops` during the NF4 quantization of the Gated Delta Network weight matrices.

### Supported Path: Unsloth

**Unsloth's FastModel handles the Qwen 3.5/3.6 architecture natively on Windows** — use it instead of raw bitsandbytes loading. Install `pip install unsloth triton-windows`.

```python
# CRASHES (raw transformers + bitsandbytes):
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb_config)

# CRASHES (BF16 transformers — triton/fla dependency):
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto")

# WORKS (Unsloth):
from unsloth import FastModel
model, processor = FastModel.from_pretrained(path, max_seq_length=2048, load_in_4bit=True)
model = FastModel.get_peft_model(model, r=16, lora_alpha=16, target_modules=[...])
```

### VRAM Impact

| Config | 9B Model | 27B Model | Training Headroom (32GB) |
|--------|----------|-----------|--------------------------|
| Unsloth 4-bit | **~8 GB** | ~22 GB | **~24 GB** / ~10 GB |
| BF16 (crashes) | ~13 GB | ~54 GB (won't fit) | ~19 GB / N/A |
| bnb 4-bit (crashes) | segfault | segfault | N/A |

For the 27B model, BF16 won't fit on a 32GB card. Options when scaling up:
1. Wait for bitsandbytes fix for Gated Delta Networks
2. Use 8-bit quantization (`load_in_8bit=True`) — untested
3. Use GPTQ/AWQ pre-quantized weights from HuggingFace
4. Use Unsloth which has its own quantization path

### Reproduction

```bash
cd /e/Documents/toka
python -c "
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    'models/Qwen3.6-27B',
    quantization_config=bnb, device_map='auto', trust_remote_code=True
)
"
# Expected: Segmentation fault (exit code 139)
```

### Files Affected

- `src/loop2_distill/sft_launcher.py` — Uses Unsloth FastModel instead of raw bitsandbytes loading
- `scripts/smoke_test_training.py` — Uses Unsloth FastModel for the supported smoke path
- `pyproject.toml` — Keeps `bitsandbytes` out of the default training extra until revalidated

---

## Qwen3.6 27B Ollama startup OOM with large context

**Status**: Mitigated by bounded default context
**Affected**: Local student inference through Ollama
**Date**: July 27, 2026

### Problem

`qwen3.6:27b` can fail during Ollama startup with a CUDA out-of-memory error when the serving context/compute allocation is too large for the current GPU state. On the tested RTX 5090 32GB setup, the unbounded smoke test failed while allocating compute buffers.

### Supported Path

Use a bounded context for day-one local student calls:

```json
{
  "options": {
    "num_ctx": 2048,
    "num_predict": 16,
    "temperature": 0
  }
}
```

With `num_ctx=2048`, the local smoke test returned the expected `TOKA_QWEN_OK` response.

### Files Affected

- `config/master.yaml` — Sets `model.ollama_num_ctx` and `usage_flywheel.student_num_ctx` to 2048
- `src/usage_flywheel/flywheel.py` — Sends bounded local student requests
- `src/infra/vllm_server.py` — Uses bounded warmup requests

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

## Qwen thinking models require think=true for this harness

**Status**: Fixed in codebase
**Affected**: All inference calls

### Problem

Some Qwen thinking-first variants cannot produce reliable `content` with `think: false`. With thinking disabled, the model may generate a thinking token first that the serving layer suppresses, resulting in `content: ""` (empty string).

### Fix

`VLLMServer.chat_completion()` defaults to `think=True`. The model puts reasoning in the `thinking` field and the actual action in `content`.

### Files Affected

- `src/infra/vllm_server.py` — Default `think=True`
- `src/arena/game.py` — Action parser strips `<think>` blocks and orphaned `</think>` tags
