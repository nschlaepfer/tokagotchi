#!/usr/bin/env python3
"""Convert a merged HF checkpoint to GGUF and import into Ollama.

Pipeline:
  1. Merge LoRA adapter into base HF model (if adapter_path given)
  2. Convert merged HF model to GGUF via llama.cpp
  3. Create Ollama Modelfile and import as a new model tag

Usage:
  python scripts/convert_to_ollama.py \
    --base-model models/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated \
    --adapter-path data/checkpoints/adapter_500ex \
    --output-tag tokagotchi-v1 \
    --quantization q4_k_m
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def merge_adapter(base_model: str, adapter_path: str, output_dir: str) -> str:
    """Merge a LoRA adapter into the base model and save."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Merging adapter %s into %s", adapter_path, base_model)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # merge on CPU to save VRAM
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    merged = model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Merged model saved to %s", output_dir)
    return output_dir


def convert_to_gguf(hf_model_dir: str, output_path: str, quantization: str = "q4_k_m") -> str:
    """Convert HF model to GGUF format using llama.cpp's convert script.

    Tries multiple approaches:
    1. llama-quantize (if llama.cpp is installed)
    2. Python convert_hf_to_gguf.py from llama.cpp repo
    3. huggingface-cli export (if available)
    """
    output_path = str(Path(output_path).with_suffix(".gguf"))

    # Approach 1: Try llama-quantize / llama-cpp-python
    try:
        # First convert to f16 GGUF
        f16_path = output_path.replace(".gguf", "-f16.gguf")

        # Try using the Python converter from llama.cpp
        convert_script = shutil.which("convert_hf_to_gguf.py")
        if convert_script is None:
            # Check common locations
            for candidate in [
                "llama.cpp/convert_hf_to_gguf.py",
                "../llama.cpp/convert_hf_to_gguf.py",
                os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
            ]:
                if os.path.exists(candidate):
                    convert_script = candidate
                    break

        if convert_script:
            logger.info("Converting HF -> GGUF F16 via %s", convert_script)
            subprocess.run(
                [sys.executable, convert_script, hf_model_dir, "--outfile", f16_path, "--outtype", "f16"],
                check=True,
            )

            # Then quantize
            quantize_bin = shutil.which("llama-quantize")
            if quantize_bin:
                logger.info("Quantizing F16 -> %s via llama-quantize", quantization)
                subprocess.run(
                    [quantize_bin, f16_path, output_path, quantization.upper()],
                    check=True,
                )
                os.remove(f16_path)
                logger.info("GGUF saved to %s", output_path)
                return output_path

        logger.warning("llama.cpp tools not found, trying alternative...")
    except Exception as e:
        logger.warning("llama.cpp conversion failed: %s", e)

    # Approach 2: Use the `llama-cpp-python` package if available
    try:
        from llama_cpp import llama_model_quantize

        logger.info("Using llama-cpp-python for quantization")
        # This would need the f16 GGUF first...
        raise NotImplementedError("llama-cpp-python quantize not yet wired up")
    except (ImportError, NotImplementedError):
        pass

    # Approach 3: Export guidance for manual conversion
    logger.error(
        "Could not auto-convert to GGUF. To convert manually:\n"
        "  1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp\n"
        "  2. pip install -r llama.cpp/requirements.txt\n"
        "  3. python llama.cpp/convert_hf_to_gguf.py %s --outfile model-f16.gguf --outtype f16\n"
        "  4. ./llama.cpp/build/bin/llama-quantize model-f16.gguf %s %s\n"
        "\nOr use Ollama's built-in import (see import_to_ollama below).",
        hf_model_dir,
        output_path,
        quantization.upper(),
    )
    return ""


def import_to_ollama(model_path: str, tag: str, base_ollama_model: str = "") -> bool:
    """Import a model into Ollama.

    Strategy 1: If we have a GGUF file, create a Modelfile and import.
    Strategy 2: If we have HF safetensors, Ollama can import directly (v0.5+).
    """
    model_path_p = Path(model_path)

    # Strategy 1: GGUF file
    if model_path_p.suffix == ".gguf" and model_path_p.exists():
        modelfile_content = f'FROM "{model_path}"\n'
        modelfile_path = model_path_p.parent / "Modelfile"
        modelfile_path.write_text(modelfile_content)

        logger.info("Importing GGUF into Ollama as %s", tag)
        result = subprocess.run(
            ["ollama", "create", tag, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("Successfully imported as ollama model: %s", tag)
            return True
        else:
            logger.error("Ollama import failed: %s", result.stderr)
            return False

    # Strategy 2: HF safetensors directory (Ollama v0.5+ supports this)
    if model_path_p.is_dir() and any(model_path_p.glob("*.safetensors")):
        modelfile_content = f'FROM "{model_path}"\n'
        modelfile_path = model_path_p / "Modelfile"
        modelfile_path.write_text(modelfile_content)

        logger.info("Importing HF safetensors into Ollama as %s", tag)
        result = subprocess.run(
            ["ollama", "create", tag, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("Successfully imported as ollama model: %s", tag)
            return True
        else:
            logger.error("Ollama HF import failed: %s\nTrying GGUF conversion...", result.stderr)

    # Strategy 3: Use base Ollama model + LoRA (if adapter is separate)
    if base_ollama_model:
        logger.info(
            "Note: For adapter-only updates, consider using:\n"
            "  ollama create %s --from %s --adapter %s",
            tag, base_ollama_model, model_path,
        )

    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HF model to Ollama")
    parser.add_argument("--base-model", required=True, help="Path to base HF model")
    parser.add_argument("--adapter-path", default="", help="Path to LoRA adapter (optional)")
    parser.add_argument("--output-tag", default="tokagotchi-v1", help="Ollama model tag")
    parser.add_argument("--quantization", default="q4_k_m", help="GGUF quantization type")
    parser.add_argument("--work-dir", default="data/conversion", help="Working directory")
    parser.add_argument("--skip-gguf", action="store_true", help="Skip GGUF, import HF directly")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Merge adapter if provided
    if args.adapter_path:
        merged_dir = str(work_dir / "merged")
        merge_adapter(args.base_model, args.adapter_path, merged_dir)
        hf_model = merged_dir
    else:
        hf_model = args.base_model

    # Step 2: Convert to GGUF (or skip)
    if args.skip_gguf:
        gguf_path = ""
    else:
        gguf_path = convert_to_gguf(
            hf_model,
            str(work_dir / f"{args.output_tag}.gguf"),
            args.quantization,
        )

    # Step 3: Import to Ollama
    import_path = gguf_path if gguf_path else hf_model
    success = import_to_ollama(import_path, args.output_tag)

    if success:
        logger.info("Done! Run: ollama run %s", args.output_tag)
    else:
        logger.warning(
            "Auto-import failed. You can manually import:\n"
            "  ollama create %s -f Modelfile",
            args.output_tag,
        )


if __name__ == "__main__":
    main()
