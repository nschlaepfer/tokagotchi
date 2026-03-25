#!/usr/bin/env python3
"""Smoke test: verify model loads in 4-bit + LoRA trains on RTX 5090.

This is the critical path test for Loops 2 and 3. If this passes,
the full SFT and RL pipelines will work with the real model.

Usage:
  python scripts/smoke_test_training.py [--model-path PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_4bit_load_and_train(model_path: str) -> None:
    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print()

    # ---- Step 1: Load in 4-bit ----
    print("=" * 50)
    print("STEP 1: Loading model in 4-bit NF4")
    print("=" * 50)
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    load_time = time.time() - t0
    vram_model = torch.cuda.memory_allocated() / 1e9
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  VRAM (model): {vram_model:.2f} GB")
    print(f"  Model type: {type(model).__name__}")
    print()

    # ---- Step 2: Add LoRA ----
    print("=" * 50)
    print("STEP 2: Applying LoRA (r=64, all projection layers)")
    print("=" * 50)

    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    vram_lora = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM (model + LoRA): {vram_lora:.2f} GB")
    print()

    # ---- Step 3: Forward + backward pass ----
    print("=" * 50)
    print("STEP 3: Forward + backward pass")
    print("=" * 50)

    # Simulate a training example
    text = (
        "<|system|>\nYou are a skilled coding agent. Use tools to solve problems.\n"
        "<|user|>\nFind all Python files in the current directory.\n"
        "<|assistant|>\nI'll use bash to find Python files.\n"
        "```bash\nfind . -name '*.py' -type f\n```\n<|end|>"
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    t0 = time.time()
    outputs = peft_model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    fwd_time = time.time() - t0
    print(f"  Forward pass: {fwd_time:.2f}s, loss={loss.item():.4f}")

    t0 = time.time()
    loss.backward()
    bwd_time = time.time() - t0
    print(f"  Backward pass: {bwd_time:.2f}s")

    grad_params = sum(1 for p in peft_model.parameters() if p.grad is not None)
    total_params = sum(1 for p in peft_model.parameters() if p.requires_grad)
    print(f"  Gradients: {grad_params}/{total_params} trainable params have grads")

    vram_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Peak VRAM: {vram_peak:.2f} GB / {torch.cuda.get_device_properties(0).total_mem / 1e9:.0f} GB")
    print()

    # ---- Step 4: Optimizer step ----
    print("=" * 50)
    print("STEP 4: Optimizer step (AdamW)")
    print("=" * 50)

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=2e-5)
    torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)

    t0 = time.time()
    optimizer.step()
    opt_time = time.time() - t0
    optimizer.zero_grad()
    print(f"  Optimizer step: {opt_time:.2f}s")

    vram_after_opt = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Peak VRAM after optimizer: {vram_after_opt:.2f} GB")
    print()

    # ---- Step 5: Generation test (verify model still works) ----
    print("=" * 50)
    print("STEP 5: Generation test (verify model still works)")
    print("=" * 50)

    peft_model.eval()
    gen_input = tokenizer("What is 2+2?", return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen_output = peft_model.generate(
            **gen_input,
            max_new_tokens=50,
            do_sample=False,
        )
    generated = tokenizer.decode(gen_output[0], skip_special_tokens=True)
    print(f"  Generated: {generated[:150]}")
    print()

    # ---- Step 6: Save LoRA adapter ----
    print("=" * 50)
    print("STEP 6: Save LoRA adapter")
    print("=" * 50)

    adapter_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "checkpoints", "smoke_test_adapter",
    )
    peft_model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    adapter_size = sum(
        os.path.getsize(os.path.join(adapter_dir, f))
        for f in os.listdir(adapter_dir)
        if f.endswith((".safetensors", ".bin"))
    ) / 1e6
    print(f"  Adapter saved to: {adapter_dir}")
    print(f"  Adapter size: {adapter_size:.1f} MB")
    print()

    # ---- Cleanup ----
    del peft_model, model, optimizer
    torch.cuda.empty_cache()

    # ---- Summary ----
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Model: {model_path}")
    print(f"  4-bit load: {load_time:.1f}s")
    print(f"  VRAM (model): {vram_model:.1f} GB")
    print(f"  VRAM (peak): {vram_after_opt:.1f} GB")
    print(f"  Forward: {fwd_time:.2f}s | Backward: {bwd_time:.2f}s | Optimizer: {opt_time:.2f}s")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Adapter size: {adapter_size:.1f} MB")
    print()
    print("  ✓ ALL STEPS PASSED — Training pipeline is functional!")
    print()
    headroom = torch.cuda.get_device_properties(0).total_mem / 1e9 - vram_after_opt
    print(f"  VRAM headroom: {headroom:.1f} GB (available for larger batches)")
    if headroom > 5:
        print("  → Plenty of room. Can increase batch_size or LoRA rank.")
    elif headroom > 2:
        print("  → Comfortable. Current settings are good.")
    else:
        print("  → Tight. Consider reducing batch_size or LoRA rank.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated",
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at {args.model_path}")
        print("Download it first: huggingface-cli download huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated")
        sys.exit(1)

    test_4bit_load_and_train(args.model_path)


if __name__ == "__main__":
    main()
