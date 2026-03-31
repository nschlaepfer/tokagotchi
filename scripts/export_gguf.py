#!/usr/bin/env python3
"""Standalone LoRA → merged safetensors → Ollama export script.

Runs in a SEPARATE subprocess from the training process so it gets
a clean VRAM state. The parent process should free all PyTorch memory
before spawning this script.

Strategy: merge LoRA into base model via Unsloth's save_pretrained_merged
(writes 16-bit safetensors to output_dir), then hand off to
``ollama create FROM <dir>`` which handles GGUF conversion natively.
This avoids needing llama.cpp's converter or 18GB BF16 intermediates.

Usage::

    python scripts/export_gguf.py \
        --base-model models/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated \
        --adapter data/checkpoints/adapter_27ex \
        --output-dir data/checkpoints/merged_export \
        --ollama-tag tokagotchi:latest
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Suppress tqdm progress bars (crashes on headless Windows)
os.environ["TQDM_DISABLE"] = "1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export LoRA adapter to Ollama")
    p.add_argument("--base-model", required=True, help="Path to base HF model")
    p.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    p.add_argument("--output-dir", default="C:/temp/tokagotchi_export",
                   help="Directory for merged safetensors output (use C: for space)")
    p.add_argument("--ollama-tag", default="tokagotchi:latest",
                   help="Ollama model tag to create")
    # Keep for compat but not used directly — Ollama picks quantization
    p.add_argument("--quantization", default="q4_k_m", help="(ignored — Ollama handles quant)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_model = str(Path(args.base_model).resolve())
    adapter = str(Path(args.adapter).resolve())
    output_dir = str(Path(args.output_dir).resolve())

    if not Path(base_model).exists():
        print(f"ERROR: Base model not found: {base_model}", file=sys.stderr)
        sys.exit(1)
    if not Path(adapter).exists():
        print(f"ERROR: Adapter not found: {adapter}", file=sys.stderr)
        sys.exit(1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Base model: {base_model}")
    print(f"Adapter:    {adapter}")
    print(f"Output:     {output_dir}")
    print(f"Tag:        {args.ollama_tag}")
    sys.stdout.flush()

    # ---- Load model + adapter ----
    import torch
    from unsloth import FastModel

    print("Loading base model (4-bit)...")
    sys.stdout.flush()
    model, processor = FastModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    print("Base model loaded")

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter)
    print("LoRA adapter applied")
    sys.stdout.flush()

    # ---- Save merged model as 16-bit safetensors ----
    # This dequantizes 4-bit → 16-bit and merges LoRA deltas in one pass.
    # Output is ~18GB of safetensors that Ollama can natively convert.
    print("Saving merged model (16-bit safetensors)...")
    sys.stdout.flush()
    model.save_pretrained_merged(
        output_dir,
        tokenizer,
        save_method="merged_16bit",
    )
    print("Merged model saved")

    # ---- Free GPU ----
    del model, tokenizer, processor
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("GPU memory freed")

    # ---- Write Modelfile for Ollama ----
    modelfile_path = Path(output_dir) / "Modelfile"
    abs_output = str(Path(output_dir).resolve())
    modelfile_path.write_text(f'FROM "{abs_output}"\n', encoding="utf-8")
    print(f"Modelfile written: {modelfile_path}")

    # ---- Import into Ollama ----
    print(f"Running: ollama create {args.ollama_tag} ...")
    sys.stdout.flush()
    result = subprocess.run(
        ["ollama", "create", args.ollama_tag, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
        timeout=900,  # Ollama conversion can take 10+ min
        cwd=output_dir,
    )
    if result.returncode == 0:
        print(f"Ollama model created: {args.ollama_tag}")
    else:
        print(f"WARNING: ollama create failed (exit {result.returncode})", file=sys.stderr)
        print(f"stdout: {result.stdout[:500]}", file=sys.stderr)
        print(f"stderr: {result.stderr[:500]}", file=sys.stderr)
        # Still print EXPORT_OK if merged safetensors are saved
        # The user can manually run ollama create later

    print("EXPORT_OK")


if __name__ == "__main__":
    main()
