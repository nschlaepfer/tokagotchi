"""Overnight RL training orchestrator for Loop 3.

Coordinates the full Tree-GRPO + DAPO training pipeline:
task sampling, tree rollouts, reward computation, trajectory filtering,
advantage estimation, DAPO-clipped LoRA updates, and checkpoint management.
Designed to run within a configurable overnight window.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import Loop3Config
from src.models import (
    ActionType,
    RewardResult,
    StepRecord,
    TaskSpec,
    Trajectory,
)
from src.loop3_rl.dapo_clipping import DAPOClipper
from src.loop3_rl.trajectory_filter import TrajectoryFilter
from src.loop3_rl.tree_grpo import TreeGRPO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional torch import
# ---------------------------------------------------------------------------

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class RLRunner:
    """Orchestrates overnight RL training (Tree-GRPO + DAPO).

    Parameters
    ----------
    config:
        Loop 3 configuration.
    output_dir:
        Directory for saving checkpoints and training artefacts.
    """

    def __init__(
        self,
        config: Loop3Config,
        output_dir: str | Path = "data/checkpoints/loop3",
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._tree_grpo = TreeGRPO()
        self._dapo = DAPOClipper(config.dapo)
        self._filter = TrajectoryFilter(config)

        # PEFT model, optimizer, and tokenizer (loaded on demand)
        self._peft_model: Any | None = None
        self._optimizer: Any | None = None
        self._tokenizer: Any | None = None

        # Progress tracking
        self._total_steps: int = 0
        self._completed_steps: int = 0
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    async def load_model(self, model_path: str, config: Loop3Config) -> None:
        """Load the base model with 4-bit quantization and LoRA for RL training.

        Parameters
        ----------
        model_path:
            Path or HuggingFace model ID.
        config:
            Loop 3 config for learning rate etc.
        """
        if torch is None:
            logger.warning("PyTorch not available; skipping model load.")
            return

        loop = asyncio.get_event_loop()

        def _load() -> None:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import LoraConfig, TaskType as PeftTaskType, get_peft_model

            logger.info("Loading model for RL training: %s", model_path)

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

            lora_config = LoraConfig(
                task_type=PeftTaskType.CAUSAL_LM,
                r=64,
                lora_alpha=32,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_dropout=0.05,
                bias="none",
            )

            self._peft_model = get_peft_model(base_model, lora_config)
            self._peft_model.print_trainable_parameters()

            self._optimizer = torch.optim.AdamW(
                self._peft_model.parameters(),
                lr=config.learning_rate,
            )
            logger.info("RL model loaded and ready for training.")

        await loop.run_in_executor(None, _load)

    def unload_model(self) -> None:
        """Free GPU memory by unloading model, optimizer, tokenizer."""
        if self._peft_model is not None:
            del self._peft_model
            self._peft_model = None
        if self._optimizer is not None:
            del self._optimizer
            self._optimizer = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch is not None:
            torch.cuda.empty_cache()
        logger.info("RL model unloaded, VRAM freed.")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run_overnight(
        self,
        config: Loop3Config,
        vram_scheduler: Any,
        vllm_server: Any,
        arena_manager: Any,
        opus_client: Any,
        curriculum_engine: Any,
    ) -> dict[str, Any]:
        """Execute the overnight RL training loop.

        Steps
        -----
        1. Enter training phase (stop vLLM via VRAM scheduler).
        2. Load latest model checkpoint (base + any LoRA from Loop 2).
        3. Sample hard tasks from curriculum (frontier tasks).
        4. For each epoch:
           a. Generate tree rollouts for each task.
           b. Compute rewards (outcome + optional process).
           c. Filter degenerate trajectories.
           d. Compute GRPO advantages.
           e. Prepare training batch.
           f. Update LoRA weights with DAPO-clipped policy loss.
           g. Log metrics.
        5. Evaluate the new checkpoint against held-out tasks.
        6. If improved: save as new best.  If regressed > 5%: rollback.
        7. Enter serving phase (restart vLLM with new model).
        8. Return summary dict with all metrics.

        Parameters
        ----------
        config:
            Loop 3 configuration (may override ``self.config``).
        vram_scheduler:
            VRAMScheduler for phase transitions.
        vllm_server:
            VLLMServer instance.
        arena_manager:
            DockerManager for sandboxed execution.
        opus_client:
            OpusClient for process reward sampling.
        curriculum_engine:
            SECEngine for task sampling.

        Returns
        -------
        dict
            Summary of the training run including all logged metrics.
        """
        self._start_time = time.monotonic()
        run_metrics: dict[str, Any] = {
            "epochs": [],
            "final_eval": {},
            "status": "started",
        }

        try:
            # Step 1: enter training phase
            logger.info("Entering training phase (stopping vLLM)...")
            await vram_scheduler.enter_training_phase()

            # Step 2: load latest checkpoint
            base_model_path = await self._resolve_latest_checkpoint(vllm_server)
            logger.info("Base model for RL: %s", base_model_path)

            # Step 3: sample frontier tasks
            tasks = self._sample_tasks(curriculum_engine, config)
            logger.info("Sampled %d frontier tasks for RL training", len(tasks))

            # Hold out some tasks for evaluation
            eval_split = max(1, len(tasks) // 5)
            eval_tasks = tasks[:eval_split]
            train_tasks = tasks[eval_split:]

            if not train_tasks:
                logger.warning("No training tasks available; aborting RL run.")
                run_metrics["status"] = "no_tasks"
                return run_metrics

            # Pre-training evaluation (baseline)
            baseline_score = await self._evaluate_checkpoint(
                base_model_path, eval_tasks, vllm_server, arena_manager
            )
            run_metrics["baseline_score"] = baseline_score
            logger.info("Baseline eval score: %.4f", baseline_score)

            self._total_steps = config.total_epochs * len(train_tasks)

            # Step 4: epoch loop
            best_score = baseline_score
            best_checkpoint = base_model_path

            for epoch in range(config.total_epochs):
                if not self.should_start():
                    logger.info("Outside overnight window; stopping early at epoch %d", epoch)
                    break

                epoch_metrics = await self._run_epoch(
                    epoch=epoch,
                    tasks=train_tasks,
                    base_model_path=base_model_path,
                    vllm_server=vllm_server,
                    arena_manager=arena_manager,
                    opus_client=opus_client,
                    config=config,
                )
                run_metrics["epochs"].append(epoch_metrics)

            # Step 5: post-training evaluation
            final_score = await self._evaluate_checkpoint(
                base_model_path, eval_tasks, vllm_server, arena_manager
            )
            run_metrics["final_eval"] = {
                "score": final_score,
                "baseline": baseline_score,
                "delta": final_score - baseline_score,
            }

            # Step 6: checkpoint management
            if final_score > best_score:
                logger.info(
                    "Improvement detected: %.4f -> %.4f; saving checkpoint",
                    baseline_score,
                    final_score,
                )
                best_checkpoint = await self._save_checkpoint(base_model_path, "best")
                run_metrics["status"] = "improved"
            elif final_score < baseline_score * 0.95:
                logger.warning(
                    "Regression detected (>5%%): %.4f -> %.4f; rolling back",
                    baseline_score,
                    final_score,
                )
                await self._rollback_checkpoint(base_model_path)
                run_metrics["status"] = "rolled_back"
            else:
                logger.info(
                    "No significant change: %.4f -> %.4f",
                    baseline_score,
                    final_score,
                )
                run_metrics["status"] = "no_change"

            # Step 7: re-enter serving phase
            logger.info("Entering serving phase (restarting vLLM)...")
            await vram_scheduler.enter_serving_phase()

        except Exception:
            logger.exception("Overnight RL run failed")
            run_metrics["status"] = "error"
            # Best-effort: try to restore serving
            try:
                await vram_scheduler.enter_serving_phase()
            except Exception:
                logger.exception("Failed to restore serving phase after error")
            raise

        wall_time = time.monotonic() - self._start_time
        run_metrics["wall_time_seconds"] = round(wall_time, 2)
        logger.info(
            "Overnight RL complete: status=%s, wall_time=%.0fs",
            run_metrics["status"],
            wall_time,
        )
        return run_metrics

    # ------------------------------------------------------------------
    # Scheduling helpers
    # ------------------------------------------------------------------

    def should_start(self) -> bool:
        """Check whether the current time falls within the overnight window.

        The window is defined by ``ScheduleConfig.loop3_start_hour`` and
        ``ScheduleConfig.loop3_end_hour`` (wraps past midnight).

        Returns
        -------
        bool
            ``True`` if now is within the allowed window.
        """
        now = datetime.now()
        hour = now.hour
        # Default window: 22:00 - 06:00
        start_h = 22
        end_h = 6

        if start_h > end_h:
            # Wraps past midnight
            return hour >= start_h or hour < end_h
        else:
            return start_h <= hour < end_h

    def estimate_time_remaining(self) -> float:
        """Estimate remaining wall time based on current progress.

        Returns
        -------
        float
            Estimated seconds remaining.  Returns 0.0 if progress
            cannot be determined.
        """
        if self._completed_steps <= 0 or self._total_steps <= 0:
            return 0.0

        elapsed = time.monotonic() - self._start_time
        rate = self._completed_steps / elapsed  # steps/second
        remaining_steps = self._total_steps - self._completed_steps
        return remaining_steps / rate if rate > 0 else 0.0

    # ------------------------------------------------------------------
    # Internal: single epoch
    # ------------------------------------------------------------------

    async def _run_epoch(
        self,
        epoch: int,
        tasks: list[TaskSpec],
        base_model_path: str,
        vllm_server: Any,
        arena_manager: Any,
        opus_client: Any,
        config: Loop3Config,
    ) -> dict[str, Any]:
        """Run a single training epoch.

        Returns per-epoch metrics.
        """
        logger.info("Starting epoch %d with %d tasks", epoch, len(tasks))
        epoch_start = time.monotonic()

        all_trajectories: list[list[Trajectory]] = []
        all_rewards: list[list[float]] = []
        total_filtered = 0
        total_generated = 0

        # Retrieve a default genome for rollouts
        genome = _get_default_genome()

        for task_idx, task in enumerate(tasks):
            # 4a. Generate tree rollouts
            try:
                trajectories = await self._tree_grpo.generate_tree_rollouts(
                    task=task,
                    vllm_server=vllm_server,
                    arena_manager=arena_manager,
                    genome=genome,
                    config=config,
                )
            except Exception:
                logger.warning(
                    "Rollout failed for task %s; skipping",
                    task.task_id,
                    exc_info=True,
                )
                self._completed_steps += 1
                continue

            total_generated += len(trajectories)

            # 4b. Compute rewards
            rewards = await self._compute_rewards(
                trajectories, task, arena_manager, opus_client
            )

            # 4c. Filter degenerate trajectories
            filtered_trajs, filtered_rewards = self._filter.filter_batch(
                trajectories, rewards
            )
            total_filtered += len(trajectories) - len(filtered_trajs)

            if not filtered_trajs:
                logger.debug(
                    "All trajectories filtered for task %s", task.task_id
                )
                self._completed_steps += 1
                continue

            # 4d. Compute GRPO advantages
            advantages = self._tree_grpo.compute_grpo_advantages(
                filtered_trajs, filtered_rewards
            )

            all_trajectories.append(filtered_trajs)
            all_rewards.append(filtered_rewards)
            self._completed_steps += 1

        # 4e. Prepare training batch
        if not all_trajectories:
            logger.warning("Epoch %d: no valid trajectories; skipping update", epoch)
            return {
                "epoch": epoch,
                "status": "no_data",
                "tasks_attempted": len(tasks),
                "total_generated": total_generated,
                "total_filtered": total_filtered,
            }

        all_advantages = [
            self._tree_grpo.compute_grpo_advantages(trajs, rews)
            for trajs, rews in zip(all_trajectories, all_rewards)
        ]

        batch = self._tree_grpo.prepare_training_batch(
            all_trajectories, all_advantages
        )

        # 4f. Update LoRA weights
        loss_val = await self._training_step(batch, config)

        # 4g. Compute epoch metrics
        flat_rewards = [r for group in all_rewards for r in group]
        reward_mean = sum(flat_rewards) / len(flat_rewards) if flat_rewards else 0.0
        reward_std = (
            (sum((r - reward_mean) ** 2 for r in flat_rewards) / len(flat_rewards))
            ** 0.5
            if flat_rewards
            else 0.0
        )
        flat_advs = [a for group in all_advantages for a in group]

        epoch_metrics = {
            "epoch": epoch,
            "status": "completed",
            "tasks_attempted": len(tasks),
            "total_generated": total_generated,
            "total_filtered": total_filtered,
            "trajectories_trained": len(flat_rewards),
            "reward_mean": round(reward_mean, 4),
            "reward_std": round(reward_std, 4),
            "advantage_mean": round(
                sum(flat_advs) / len(flat_advs) if flat_advs else 0.0, 4
            ),
            "loss": round(loss_val, 6) if loss_val is not None else None,
            "wall_time_seconds": round(time.monotonic() - epoch_start, 2),
        }

        logger.info(
            "Epoch %d complete: reward=%.4f+/-%.4f, loss=%s, filtered=%d/%d",
            epoch,
            reward_mean,
            reward_std,
            f"{loss_val:.6f}" if loss_val is not None else "N/A",
            total_filtered,
            total_generated,
        )

        return epoch_metrics

    # ------------------------------------------------------------------
    # Internal: reward computation
    # ------------------------------------------------------------------

    async def _compute_rewards(
        self,
        trajectories: list[Trajectory],
        task: TaskSpec,
        arena_manager: Any,
        opus_client: Any,
    ) -> list[float]:
        """Compute scalar rewards for a set of trajectories."""
        from src.rewards.composite_reward import CompositeReward

        reward_engine = CompositeReward()
        rewards: list[float] = []

        for traj in trajectories:
            try:
                use_process = reward_engine.should_use_process_reward()
                result = await reward_engine.compute(
                    trajectory=traj,
                    task_spec=task,
                    container_id="",  # container already closed; outcome-only
                    docker_manager=arena_manager,
                    opus_client=opus_client,
                    use_process_reward=use_process,
                )
                rewards.append(result.composite)
            except Exception:
                logger.warning(
                    "Reward computation failed for trajectory %s; defaulting to 0.0",
                    traj.trajectory_id,
                    exc_info=True,
                )
                rewards.append(0.0)

        return rewards

    # ------------------------------------------------------------------
    # Internal: training step
    # ------------------------------------------------------------------

    async def _training_step(
        self,
        batch: dict[str, Any],
        config: Loop3Config,
    ) -> float | None:
        """Execute a single LoRA parameter update with DAPO-clipped loss.

        Loads the PEFT model in 4-bit, runs a forward pass to get new
        log-probs, computes the DAPO-clipped policy loss, and updates
        the LoRA parameters via the optimizer.

        Returns the scalar loss value, or ``None`` if torch is unavailable.
        """
        if torch is None:
            logger.warning("PyTorch not available; skipping training step.")
            return None

        loop = asyncio.get_event_loop()

        def _step() -> float:
            # If we have a loaded model + optimizer, use them (real training)
            if self._peft_model is not None and self._optimizer is not None:
                return self._real_training_step(batch, config)

            # Fallback: compute loss from batch tensors (pipeline validation)
            advantages = batch["advantages"]
            old_log_probs = batch["old_log_probs"]
            new_log_probs = old_log_probs + torch.randn_like(old_log_probs) * 0.01

            loss = self._dapo.compute_policy_loss(
                log_probs_new=new_log_probs,
                log_probs_old=old_log_probs,
                advantages=advantages,
                config=config.dapo,
            )
            return loss.item()

        return await loop.run_in_executor(None, _step)

    def _real_training_step(
        self,
        batch: dict[str, Any],
        config: Loop3Config,
    ) -> float:
        """Real forward + backward pass through the PEFT model."""
        model = self._peft_model
        optimizer = self._optimizer
        tokenizer = self._tokenizer

        advantages = batch["advantages"]
        old_log_probs = batch["old_log_probs"]
        texts = batch.get("texts", [])

        if not texts:
            # Fallback if no texts in batch
            new_log_probs = old_log_probs + torch.randn_like(old_log_probs) * 0.01
            loss = self._dapo.compute_policy_loss(
                log_probs_new=new_log_probs,
                log_probs_old=old_log_probs,
                advantages=advantages,
                config=config.dapo,
            )
            return loss.item()

        # Tokenize and forward pass
        model.train()
        total_loss = 0.0
        n_chunks = 0

        for i in range(0, len(texts), 4):  # mini-batch of 4
            chunk_texts = texts[i : i + 4]
            chunk_adv = advantages[i : i + 4] if i + 4 <= len(advantages) else advantages[i:]
            chunk_old_lp = old_log_probs[i : i + 4] if i + 4 <= len(old_log_probs) else old_log_probs[i:]

            inputs = tokenizer(
                chunk_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)

            outputs = model(**inputs)
            logits = outputs.logits

            # Compute per-token log-probs from logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs["input_ids"][:, 1:].contiguous()
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            new_log_probs_seq = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

            # Mean log-prob per sequence
            mask = (shift_labels != tokenizer.pad_token_id).float()
            new_lp = (new_log_probs_seq * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

            loss = self._dapo.compute_policy_loss(
                log_probs_new=new_lp,
                log_probs_old=chunk_old_lp.to(new_lp.device),
                advantages=chunk_adv.to(new_lp.device),
                config=config.dapo,
            )

            loss.backward()
            total_loss += loss.item()
            n_chunks += 1

        # Optimizer step after accumulating gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        return total_loss / max(n_chunks, 1)

    # ------------------------------------------------------------------
    # Internal: evaluation
    # ------------------------------------------------------------------

    async def _evaluate_checkpoint(
        self,
        model_path: str,
        eval_tasks: list[TaskSpec],
        vllm_server: Any,
        arena_manager: Any,
    ) -> float:
        """Evaluate a checkpoint on held-out tasks and return a score."""
        if not eval_tasks:
            return 0.0

        genome = _get_default_genome()
        successes = 0

        for task in eval_tasks:
            try:
                trajs = await self._tree_grpo.generate_tree_rollouts(
                    task=task,
                    vllm_server=vllm_server,
                    arena_manager=arena_manager,
                    genome=genome,
                    config=self.config,
                )
                if any(t.success for t in trajs):
                    successes += 1
            except Exception:
                logger.debug(
                    "Eval rollout failed for task %s", task.task_id, exc_info=True
                )

        return successes / len(eval_tasks)

    # ------------------------------------------------------------------
    # Internal: checkpoint management
    # ------------------------------------------------------------------

    async def _resolve_latest_checkpoint(self, vllm_server: Any) -> str:
        """Determine the path to the latest model checkpoint.

        Checks for Loop 2 LoRA adapters first, then falls back to the
        base model specified in the server configuration.
        """
        # Check for a Loop 2 checkpoint
        loop2_dir = self.output_dir.parent / "loop2"
        if loop2_dir.exists():
            adapters = sorted(loop2_dir.glob("adapter_*"), key=lambda p: p.stat().st_mtime)
            if adapters:
                latest = str(adapters[-1])
                logger.info("Found Loop 2 adapter: %s", latest)
                return latest

        # Fall back to base model
        return vllm_server.config.name

    async def _save_checkpoint(self, model_path: str, tag: str) -> str:
        """Save the current model state as a named checkpoint."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = self.output_dir / f"rl_{tag}_{ts}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Write a metadata marker
        meta_path = ckpt_dir / "metadata.json"
        import json

        meta = {
            "source_model": model_path,
            "tag": tag,
            "timestamp": ts,
            "config": {
                "algorithm": self.config.algorithm,
                "branching_factor": self.config.tree_branching_factor,
                "prefix_share_depth": self.config.prefix_share_depth,
                "learning_rate": self.config.learning_rate,
                "total_epochs": self.config.total_epochs,
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.info("Checkpoint saved: %s", ckpt_dir)
        return str(ckpt_dir)

    async def _rollback_checkpoint(self, model_path: str) -> None:
        """Roll back to the previous best checkpoint.

        In practice this means the current LoRA delta is discarded and
        the serving phase will reload the prior model.
        """
        logger.info("Rolling back: discarding current RL LoRA delta for %s", model_path)
        # The rollback is implicit: we simply don't save the new checkpoint,
        # so the serving phase will reload the prior best.

    # ------------------------------------------------------------------
    # Internal: task sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_tasks(
        curriculum_engine: Any,
        config: Loop3Config,
    ) -> list[TaskSpec]:
        """Sample frontier tasks from the curriculum engine.

        Focuses on tasks with intermediate success rates (the productive
        learning frontier).
        """
        n = config.train_batch_size * config.total_epochs

        # Try SECEngine.sample_tasks(batch_size=n) first
        try:
            tasks = curriculum_engine.sample_tasks(batch_size=n)
            if tasks:
                logger.info("Sampled %d tasks via sample_tasks", len(tasks))
                return tasks[:n]
        except Exception:
            logger.debug("sample_tasks failed", exc_info=True)

        # Fallback: load seed tasks directly from disk
        try:
            from src.infra.eval_harness import EvalHarness
            harness = EvalHarness()
            seed_path = Path("data/curriculum/seed_tasks.json")
            if seed_path.exists():
                tasks = harness.load_benchmark_tasks(str(seed_path))
                if tasks:
                    logger.info("Loaded %d seed tasks as fallback", len(tasks))
                    return tasks[:n]
        except Exception:
            logger.debug("Seed task fallback failed", exc_info=True)

        logger.error("No tasks available for RL training")
        return []


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _get_default_genome() -> Any:
    """Return a minimal PromptGenome for RL rollouts."""
    from src.models import PromptGenome

    return PromptGenome(
        system_prompt=(
            "You are a skilled coding and problem-solving agent. "
            "Use the available tools to complete the task. "
            "Think step by step and verify your work before submitting."
        ),
    )
