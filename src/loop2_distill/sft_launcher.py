"""Triggers QLoRA fine-tuning on corrected traces.

Manages the full SFT lifecycle: data preparation, PEFT LoRA configuration,
training via TRL/transformers, adapter merging, and checkpoint validation.
Designed for a single 32GB VRAM budget during training phases.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from src.models import (
    Trajectory,
    TraceAnalysis,
    TaskSpec,
    StepRecord,
    ActionType,
    PromptGenome,
)
from src.orchestrator.opus_client import OpusClient
from src.config import Loop2Config

logger = logging.getLogger(__name__)


class SFTLauncher:
    """Launches QLoRA fine-tuning runs on corrected trace data.

    Handles data serialisation, PEFT/LoRA configuration, SFT training,
    adapter merging, and quick checkpoint validation.

    Parameters
    ----------
    output_dir:
        Base directory for saving adapters and merged checkpoints.
    """

    def __init__(self, output_dir: str | Path = "data/checkpoints") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    async def launch_training(
        self,
        training_data: list[dict[str, Any]],
        config: Loop2Config,
        base_model_path: str,
    ) -> str:
        """Run LoRA SFT on the provided training examples via Unsloth.

        Uses Unsloth's FastModel which handles Qwen 3.5's Gated Delta
        Networks natively on Windows without requiring triton/causal-conv1d.
        Loads in 4-bit (~8GB VRAM) leaving room for training.

        Parameters
        ----------
        training_data:
            List of chat-format training examples (each with ``messages``).
        config:
            Loop2 configuration with LoRA and training hyperparameters.
        base_model_path:
            Path or HuggingFace model ID for the base model.

        Returns
        -------
        str
            Path to the saved LoRA adapter directory.
        """
        # Lazy imports — fix tqdm stderr crash on Windows background processes
        import io, os, sys
        if not hasattr(sys.stderr, 'fileno') or not sys.stderr.isatty():
            os.environ["TQDM_DISABLE"] = "1"
        import torch
        from datasets import Dataset
        from unsloth import FastModel
        from trl import SFTTrainer, SFTConfig

        logger.info(
            "Launching SFT (Unsloth): %d examples, base_model=%s, lora_rank=%d",
            len(training_data),
            base_model_path,
            config.lora.rank,
        )

        # 1. Save training data
        data_path = self._save_training_data(training_data)
        logger.info("Training data saved to %s", data_path)

        # 2. Load dataset
        dataset = self._load_dataset(data_path)

        # 3. Load model via Unsloth FastModel (handles Qwen 3.5 natively)
        logger.info("Loading model via Unsloth FastModel...")
        model, processor = FastModel.from_pretrained(
            model_name=base_model_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )

        # Extract text tokenizer from processor (Qwen 3.5 is multimodal)
        tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 4. Apply LoRA via Unsloth (uses its own optimized PEFT)
        logger.info("Applying LoRA adapters...")
        # Unsloth recommends dropout=0 for fast patching
        model = FastModel.get_peft_model(
            model,
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=0.0,
        )

        # 5. Configure training
        adapter_output = str(
            self.output_dir / f"adapter_{len(training_data)}ex"
        )

        training_args = SFTConfig(
            output_dir=adapter_output,
            num_train_epochs=1,
            max_steps=min(config.max_steps, len(training_data) * 3),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            bf16=config.bf16,
            fp16=not config.bf16,
            gradient_checkpointing=config.gradient_checkpointing,
            logging_steps=1,
            save_strategy="no",
            remove_unused_columns=False,
            report_to="none",
            max_seq_length=2048,
            gradient_checkpointing_kwargs={"use_reentrant": False}
            if config.gradient_checkpointing
            else None,
        )

        # 6. Formatting function for chat data
        #    Unsloth's SFTTrainer expects a list of strings (batched mode)
        def _formatting_func(examples: dict[str, Any]) -> list[str]:
            """Format chat conversations into strings for SFT."""
            texts = []
            for messages in examples["messages"]:
                parts = []
                for msg in messages:
                    if isinstance(msg, str):
                        parts.append(msg)
                        continue
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    parts.append(f"<|{role}|>\n{content}")
                parts.append("<|end|>")
                texts.append("\n".join(parts))
            return texts

        # 7. Create trainer and run
        try:
            logger.info("Creating SFTTrainer...")
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=dataset,
                formatting_func=_formatting_func,
            )
            logger.info("SFTTrainer created, starting training...")
        except Exception:
            logger.exception("SFTTrainer creation failed")
            raise

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, trainer.train)
            logger.info("Training complete")
        except Exception:
            logger.exception("Training failed")
            raise

        # Save the adapter
        await loop.run_in_executor(None, trainer.save_model, adapter_output)
        logger.info("LoRA adapter saved to %s", adapter_output)

        # Free training VRAM (model, optimizer, etc.)
        del trainer, model, tokenizer, processor, dataset
        _free_gpu_memory()

        return adapter_output

    async def train_and_export(
        self,
        training_data: list[dict[str, Any]],
        config: Loop2Config,
        base_model_path: str,
        ollama_model_name: str = "tokagotchi:latest",
        quantization: str = "q4_k_m",
    ) -> tuple[str, str | None]:
        """Train LoRA, then export to GGUF+Ollama using the already-loaded model.

        This avoids reloading the model for export (saves ~18GB RAM).

        Returns (adapter_path, ollama_model_name or None on export failure).
        """
        adapter_path, model, tokenizer = await self.launch_training(
            training_data, config, base_model_path,
        )

        # Export GGUF while model+LoRA are still in memory
        gguf_dir = str(self.output_dir / "gguf_export")
        Path(gguf_dir).mkdir(parents=True, exist_ok=True)
        ollama_name: str | None = None

        loop = asyncio.get_event_loop()
        try:
            logger.info(
                "Exporting GGUF (in-memory): quant=%s, dest=%s",
                quantization, gguf_dir,
            )

            def _save_gguf() -> None:
                model.save_pretrained_gguf(
                    gguf_dir,
                    tokenizer,
                    quantization_method=quantization,
                )

            await loop.run_in_executor(None, _save_gguf)
            logger.info("GGUF export complete: %s", gguf_dir)

            # Import into Ollama
            modelfile_path = Path(gguf_dir) / "Modelfile"
            if modelfile_path.exists():
                proc = await asyncio.create_subprocess_exec(
                    "ollama", "create", ollama_model_name,
                    "-f", str(modelfile_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=gguf_dir,
                )
                out, _ = await proc.communicate()
                if proc.returncode == 0:
                    logger.info("Ollama model created: %s", ollama_model_name)
                    ollama_name = ollama_model_name
                else:
                    logger.error(
                        "ollama create failed: %s",
                        out.decode(errors="replace"),
                    )
            else:
                logger.warning("No Modelfile at %s", modelfile_path)
        except Exception:
            logger.exception("GGUF export failed (adapter still saved at %s)", adapter_path)
        finally:
            del model, tokenizer
            _free_gpu_memory()

        return adapter_path, ollama_name

    async def export_to_ollama(
        self,
        base_model_path: str,
        adapter_path: str,
        ollama_model_name: str = "tokagotchi:latest",
        quantization: str = "q4_k_m",
    ) -> str:
        """Merge LoRA adapter, convert to GGUF, and import into Ollama.

        Runs the export in a **separate Python subprocess** via
        ``scripts/export_gguf.py`` so the parent process's VRAM is
        fully free — no double-load OOM.

        Parameters
        ----------
        base_model_path:
            Path to the base HF model.
        adapter_path:
            Path to the saved LoRA adapter directory.
        ollama_model_name:
            Name for the Ollama model (e.g. "tokagotchi:latest").
        quantization:
            GGUF quantization method (q4_k_m, q8_0, f16).

        Returns
        -------
        str
            The Ollama model name that was created.
        """
        import sys

        # Locate the export script relative to the project root
        project_root = Path(__file__).resolve().parent.parent.parent
        export_script = project_root / "scripts" / "export_gguf.py"
        if not export_script.exists():
            raise FileNotFoundError(f"Export script not found: {export_script}")

        # Use C: drive for the merged safetensors — E: is too full with pagefile
        output_dir = "C:/temp/tokagotchi_export"

        logger.info(
            "Exporting to Ollama (subprocess): adapter=%s, quant=%s, tag=%s",
            adapter_path, quantization, ollama_model_name,
        )

        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(export_script),
            "--base-model", str(Path(base_model_path).resolve()),
            "--adapter", str(Path(adapter_path).resolve()),
            "--ollama-base", "huihui_ai/qwen3.5-abliterated:9b",
            "--ollama-tag", ollama_model_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(project_root),
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode(errors="replace")

        if "EXPORT_OK" in output:
            logger.info("GGUF export + Ollama import complete: %s", ollama_model_name)
        else:
            logger.error("Export subprocess failed (exit %d):\n%s", proc.returncode, output[-2000:])
            raise RuntimeError("GGUF export subprocess failed")

        return ollama_model_name

    # ------------------------------------------------------------------
    # GPU memory management
    # ------------------------------------------------------------------

    async def launch_training_and_export(
        self,
        training_data: list[dict[str, Any]],
        config: Loop2Config,
        base_model_path: str,
        ollama_model_name: str = "tokagotchi:latest",
        quantization: str = "q4_k_m",
    ) -> tuple[str, str]:
        """Train LoRA then export to Ollama in one pass (saves a model reload).

        Returns (adapter_path, ollama_model_name).
        """
        adapter_path = await self.launch_training(
            training_data, config, base_model_path,
        )
        # Free training VRAM before export reload
        _free_gpu_memory()

        ollama_name = await self.export_to_ollama(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            ollama_model_name=ollama_model_name,
            quantization=quantization,
        )
        return adapter_path, ollama_name

    # ------------------------------------------------------------------
    # Adapter merging (legacy — use export_to_ollama instead)
    # ------------------------------------------------------------------

    async def merge_adapter(
        self,
        base_model_path: str,
        adapter_path: str,
        output_path: str,
    ) -> None:
        """Merge a LoRA adapter into the base model and save.

        Parameters
        ----------
        base_model_path:
            Path or model ID for the base model.
        adapter_path:
            Path to the LoRA adapter directory.
        output_path:
            Path to save the merged model.
        """
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(
            "Merging adapter %s into %s -> %s",
            adapter_path,
            base_model_path,
            output_path,
        )

        loop = asyncio.get_event_loop()

        def _merge() -> None:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_path, trust_remote_code=True
            )
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                elif tokenizer.unk_token is not None:
                    tokenizer.pad_token = tokenizer.unk_token
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

            model = PeftModel.from_pretrained(base_model, adapter_path)
            merged = model.merge_and_unload()
            generation_config = getattr(merged, "generation_config", None)
            if generation_config is not None:
                pad_token_id = tokenizer.pad_token_id
                eos_token_id = tokenizer.eos_token_id
                if pad_token_id is None or pad_token_id < 0:
                    pad_token_id = eos_token_id
                if pad_token_id is not None and pad_token_id >= 0:
                    generation_config.pad_token_id = pad_token_id
                if eos_token_id is not None and eos_token_id >= 0:
                    generation_config.eos_token_id = eos_token_id

            output = Path(output_path)
            output.mkdir(parents=True, exist_ok=True)
            merged.save_pretrained(str(output))
            tokenizer.save_pretrained(str(output))

        await loop.run_in_executor(None, _merge)
        logger.info("Merged model saved to %s", output_path)

    async def import_to_ollama(
        self,
        model_path: str,
        tag: str,
    ) -> bool:
        """Import a merged HF checkpoint into Ollama as a local model tag.

        Parameters
        ----------
        model_path:
            Directory containing the merged HuggingFace checkpoint.
        tag:
            Ollama tag to create.
        """
        model_dir = Path(model_path).resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Merged model directory not found: {model_dir}")

        if not any(model_dir.glob("*.safetensors")):
            raise FileNotFoundError(
                f"No safetensors files found in merged model directory: {model_dir}"
            )

        modelfile_path = model_dir / "Modelfile"
        modelfile_path.write_text(f'FROM "{model_dir}"\n', encoding="utf-8")

        logger.info("Importing merged checkpoint into Ollama as %s", tag)
        proc = await asyncio.create_subprocess_exec(
            "ollama",
            "create",
            tag,
            "-f",
            str(modelfile_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            logger.error(
                "Ollama import failed for %s: %s",
                tag,
                stdout.decode(errors="replace"),
            )
            return False

        logger.info("Imported Ollama model tag %s from %s", tag, model_dir)
        return True

    async def deploy_adapter(
        self,
        base_model_path: str,
        adapter_path: str,
        tag: str,
        merged_output_path: str,
    ) -> str:
        """Merge a LoRA adapter and import the merged checkpoint into Ollama.

        Returns the Ollama tag to serve on success.
        """
        await self.merge_adapter(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            output_path=merged_output_path,
        )
        ok = await self.import_to_ollama(merged_output_path, tag)
        if not ok:
            raise RuntimeError(f"Failed to import merged checkpoint into Ollama: {tag}")
        return tag

    # ------------------------------------------------------------------
    # Checkpoint validation
    # ------------------------------------------------------------------

    async def validate_checkpoint(
        self,
        checkpoint_path: str,
        eval_tasks: list[TaskSpec],
        vllm_server: Any | None = None,
        arena_manager: Any | None = None,
        genome: PromptGenome | None = None,
    ) -> dict[str, Any]:
        """Quick validation of a new checkpoint against evaluation tasks.

        Runs a small number of tasks and returns success rate comparison
        metrics. If vllm_server and arena_manager are not provided,
        performs a dry-run validation (tokenizer load and generation test).

        Parameters
        ----------
        checkpoint_path:
            Path to the model checkpoint to validate.
        eval_tasks:
            Tasks to evaluate on.
        vllm_server:
            Optional VLLMServer for full evaluation.
        arena_manager:
            Optional DockerManager for full evaluation.
        genome:
            Optional PromptGenome for full evaluation.

        Returns
        -------
        dict
            Validation results including success_rate, num_tasks,
            and per-task outcomes.
        """
        logger.info(
            "Validating checkpoint %s on %d tasks",
            checkpoint_path,
            len(eval_tasks),
        )

        # If full evaluation infrastructure is available, use it
        if vllm_server is not None and arena_manager is not None:
            return await self._full_validation(
                checkpoint_path, eval_tasks, vllm_server, arena_manager, genome,
            )

        # Otherwise, do a minimal dry-run validation
        return await self._dry_run_validation(checkpoint_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_training_data(self, training_data: list[dict[str, Any]]) -> Path:
        """Save training data to a temporary JSONL file.

        Returns the path to the file.
        """
        data_dir = self.output_dir / "training_data"
        data_dir.mkdir(parents=True, exist_ok=True)

        path = data_dir / "sft_data.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for example in training_data:
                f.write(json.dumps(example, default=str) + "\n")

        return path

    @staticmethod
    def _load_dataset(data_path: Path) -> Any:
        """Load a JSONL file as a HuggingFace Dataset."""
        from datasets import Dataset

        records: list[dict[str, Any]] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    records.append(record)

        # Extract just the messages for the dataset
        messages_list = [r.get("messages", []) for r in records]
        return Dataset.from_dict({"messages": messages_list})

    async def _full_validation(
        self,
        checkpoint_path: str,
        eval_tasks: list[TaskSpec],
        vllm_server: Any,
        arena_manager: Any,
        genome: PromptGenome | None,
    ) -> dict[str, Any]:
        """Run full agent evaluation with the checkpoint."""
        from src.loop2_distill.trace_collector import TraceCollector

        collector = TraceCollector(concurrency=2, timeout_seconds=60.0)
        default_genome = genome or PromptGenome(
            system_prompt="You are a skilled coding agent."
        )

        trajectories = await collector.collect_rollouts(
            tasks=eval_tasks,
            n_per_task=1,
            vllm_server=vllm_server,
            arena_manager=arena_manager,
            genome=default_genome,
        )

        successes = sum(1 for t in trajectories if t.success)
        total = len(trajectories)

        results = {
            "checkpoint": checkpoint_path,
            "num_tasks": len(eval_tasks),
            "num_trajectories": total,
            "successes": successes,
            "success_rate": successes / total if total > 0 else 0.0,
            "avg_steps": (
                sum(t.num_steps for t in trajectories) / total
                if total > 0
                else 0.0
            ),
            "per_task": [
                {
                    "task_id": t.task.task_id if t.task else "",
                    "success": t.success,
                    "num_steps": t.num_steps,
                    "total_reward": t.total_reward,
                }
                for t in trajectories
            ],
        }

        logger.info(
            "Validation: %d/%d success (%.1f%%)",
            successes,
            total,
            results["success_rate"] * 100,
        )
        return results

    async def _dry_run_validation(
        self, checkpoint_path: str,
    ) -> dict[str, Any]:
        """Minimal validation: load the model and run a test generation."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        loop = asyncio.get_event_loop()

        def _validate() -> dict[str, Any]:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    checkpoint_path, trust_remote_code=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )

                # Quick generation test
                inputs = tokenizer(
                    "Hello, I am a coding agent.",
                    return_tensors="pt",
                ).to(model.device)
                outputs = model.generate(
                    **inputs, max_new_tokens=32, do_sample=False
                )
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

                return {
                    "checkpoint": checkpoint_path,
                    "dry_run": True,
                    "model_loads": True,
                    "generates_text": len(generated) > 0,
                    "sample_output": generated[:200],
                }
            except Exception as e:
                logger.error("Dry-run validation failed: %s", e)
                return {
                    "checkpoint": checkpoint_path,
                    "dry_run": True,
                    "model_loads": False,
                    "error": str(e),
                }

        return await loop.run_in_executor(None, _validate)


def _free_gpu_memory() -> None:
    """Release all PyTorch GPU memory so Ollama can reclaim VRAM."""
    try:
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info(
                "GPU memory freed: %.0f MiB available",
                torch.cuda.mem_get_info()[0] / 1024 / 1024,
            )
    except Exception as e:
        logger.warning("Failed to free GPU memory: %s", e)
