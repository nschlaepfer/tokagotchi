#!/usr/bin/env python3
"""Standalone LoRA adapter -> merged GGUF -> Ollama export. Zero-GPU."""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

os.environ["TQDM_DISABLE"] = "1"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--ollama-base", default="huihui_ai/qwen3.5-abliterated:9b")
    p.add_argument("--ollama-tag", default="tokagotchi:latest")
    p.add_argument("--output-dir", default="C:/temp/tokagotchi_gguf")
    p.add_argument("--quantization", default="q4_k_m")
    return p.parse_args()

def find_tool(name):
    for c in [Path.home()/".unsloth"/"llama.cpp"/name,
              Path.home()/".unsloth"/"llama.cpp"/"build"/"bin"/"Release"/f"{name}.exe"]:
        if c.exists(): return c
    return None

def get_base_gguf(tag):
    r = subprocess.run(["ollama","show",tag,"--modelfile"], capture_output=True, text=True, timeout=10)
    for line in r.stdout.splitlines():
        l = line.strip()
        if l.startswith("FROM ") and ("sha256" in l or l.endswith(".gguf")):
            return l[5:].strip()
    return None

def main():
    args = parse_args()
    base_model = str(Path(args.base_model).resolve())
    adapter = str(Path(args.adapter).resolve())
    output_dir = str(Path(args.output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not Path(adapter).exists():
        print(f"ERROR: Adapter not found: {adapter}"); sys.exit(1)
    if not Path(base_model).exists():
        print(f"ERROR: Base model not found: {base_model}"); sys.exit(1)

    base_gguf = get_base_gguf(args.ollama_base)
    if not base_gguf:
        print(f"ERROR: No GGUF for {args.ollama_base}"); sys.exit(1)
    print(f"Base GGUF: {base_gguf}")

    lora_gguf = str(Path(output_dir) / "adapter.gguf")
    merged_gguf = str(Path(output_dir) / "tokagotchi-merged.gguf")

    # Step 1: Convert LoRA safetensors -> GGUF
    converter = find_tool("convert_lora_to_gguf.py")
    if not converter:
        print("ERROR: convert_lora_to_gguf.py not found"); sys.exit(1)
    print("Step 1: LoRA safetensors -> GGUF adapter...")
    r = subprocess.run([sys.executable, str(converter), "--base", base_model,
        "--outfile", lora_gguf, "--outtype", "f16", adapter],
        capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f"FAILED: {r.stderr[:500]}"); sys.exit(1)
    print(f"   Done: {Path(lora_gguf).stat().st_size/1024/1024:.0f} MB")

    # Step 2: Merge base GGUF + LoRA GGUF
    export_lora = find_tool("llama-export-lora")
    if not export_lora:
        print("ERROR: llama-export-lora not found"); sys.exit(1)
    print("Step 2: Merging base + LoRA...")
    r = subprocess.run([str(export_lora), "--model", base_gguf, "--lora", lora_gguf,
        "--output", merged_gguf], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"FAILED: {r.stderr[:500]}"); sys.exit(1)
    print(f"   Done: {Path(merged_gguf).stat().st_size/1024/1024/1024:.1f} GB")

    # Cleanup adapter GGUF
    try: Path(lora_gguf).unlink()
    except: pass

    # Step 3: Import to Ollama
    mf = Path(output_dir) / "Modelfile"
    mf.write_text(f'FROM "{Path(merged_gguf).resolve()}"\n', encoding="utf-8")
    print(f"Step 3: ollama create {args.ollama_tag}...")
    r = subprocess.run(["ollama","create",args.ollama_tag,"-f",str(mf)],
        capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f"FAILED: {r.stderr[:300]}"); sys.exit(1)
    print(f"   Ollama model: {args.ollama_tag}")

    # Cleanup merged GGUF (now in Ollama's blob store)
    try: Path(merged_gguf).unlink()
    except: pass

    print("EXPORT_OK")

if __name__ == "__main__":
    main()
