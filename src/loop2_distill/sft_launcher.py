"""Triggers QLoRA fine-tuning on corrected traces.

Manages the full SFT lifecycle: data preparation, PEFT LoRA configuration,
training via TRL/transformers, adapter merging, and checkpoint validation.
Designed for a single 32GB VRAM budget during training phases.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
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
        """Run QLoRA SFT on the provided training examples.

        Saves training data to a temporary JSONL file, configures PEFT
        LoRA, and runs supervised fine-tuning. The full 32GB VRAM budget
        is assumed available during training.

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
        # Lazy imports to avoid loading torch at module level
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType as PeftTaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer, SFTConfig

        logger.info(
            "Launching SFT: %d examples, base_model=%s, lora_rank=%d",
            len(training_data),
            base_model_path,
            config.lora.rank,
        )

        # 1. Save training data to a temporary JSONL file
        data_path = self._save_training_data(training_data)
        logger.info("Training data saved to %s", data_path)

        # 2. Load dataset
        dataset = self._load_dataset(data_path)

        # 3. Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # BF16 loading — bitsandbytes 4-bit segfaults on Qwen 3.5's
        # Gated Delta Network layers with current driver/library combo.
        # BF16 9B model uses ~13GB, leaves ~19GB for LoRA + optimizer.
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # 4. Configure PEFT LoRA
        lora_config = LoraConfig(
            task_type=PeftTaskType.CAUSAL_LM,
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.dropout,
            bias="none",
        )

        # 5. Configure training
        adapter_output = str(
            self.output_dir / f"adapter_{len(training_data)}ex"
        )

        training_args = SFTConfig(
            output_dir=adapter_output,
            num_train_epochs=1,
            max_steps=config.max_steps,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            bf16=config.bf16,
            fp16=not config.bf16,
            gradient_checkpointing=config.gradient_checkpointing,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            remove_unused_columns=False,
            report_to="none",
            gradient_checkpointing_kwargs={"use_reentrant": False}
            if config.gradient_checkpointing
            else None,
        )

        # 6. Create trainer and run
        def _formatting_func(example: dict[str, Any]) -> str:
            """Format a single chat conversation into a string for SFT.

            TRL 0.29+ calls this with batched=False, so example is a
            single row: {"messages": [{"role": ..., "content": ...}, ...]}.
            """
            messages = example["messages"]
            parts = []
            for msg in messages:
                if isinstance(msg, str):
                    parts.append(msg)
                    continue
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    parts.append(f"<|system|>\n{content}")
                elif role == "user":
                    parts.append(f"<|user|>\n{content}")
                elif role == "assistant":
                    parts.append(f"<|assistant|>\n{content}")
            parts.append("<|end|>")
            return "\n".join(parts)

        try:
            logger.info("Creating SFTTrainer...")
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                peft_config=lora_config,
                formatting_func=_formatting_func,
            )
            logger.info("SFTTrainer created, starting training...")
        except Exception:
            logger.exception("SFTTrainer creation failed")
            raise

        # Run training (blocking, on the event loop executor)
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

        return adapter_output

    # ------------------------------------------------------------------
    # Adapter merging
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
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

            model = PeftModel.from_pretrained(base_model, adapter_path)
            merged = model.merge_and_unload()

            output = Path(output_path)
            output.mkdir(parents=True, exist_ok=True)
            merged.save_pretrained(str(output))
            tokenizer.save_pretrained(str(output))

        await loop.run_in_executor(None, _merge)
        logger.info("Merged model saved to %s", output_path)

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
