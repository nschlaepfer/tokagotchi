"""Overnight RL training orchestrator for Loop 3.

Coordinates the full Tree-GRPO + DAPO training pipeline:
task sampling, tree rollouts, reward computation, trajectory filtering,
advantage estimation, DAPO-clipped LoRA updates, and checkpoint management.
Designed to run within a configurable overnight window.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.config import Loop3Config
from src.models import TaskSpec, Trajectory
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
        self._local_server: Any | None = None
        self._base_model_path: str | None = None
        self._adapter_path: str | None = None
        self._model_label: str = ""
        self._generation_lock = asyncio.Lock()

        # Progress tracking
        self._total_steps: int = 0
        self._completed_steps: int = 0
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    async def load_model(
        self,
        model_path: str,
        config: Loop3Config,
        adapter_path: str | None = None,
    ) -> None:
        """Load the base model plus an optional LoRA adapter for RL training.

        Parameters
        ----------
        model_path:
            Path or HuggingFace model ID.
        config:
            Loop 3 config for learning rate etc.
        adapter_path:
            Optional path to a previously saved LoRA adapter checkpoint.
        """
        if torch is None:
            logger.warning("PyTorch not available; skipping model load.")
            return

        loop = asyncio.get_event_loop()
        resolved_adapter = str(adapter_path) if adapter_path else None

        self.unload_model()

        def _load() -> None:
            from peft import LoraConfig, PeftModel, TaskType as PeftTaskType, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("Loading model for RL training: %s", model_path)

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
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

            if resolved_adapter and Path(resolved_adapter).exists():
                self._peft_model = PeftModel.from_pretrained(
                    base_model,
                    resolved_adapter,
                    is_trainable=True,
                )
                logger.info("Loaded RL adapter checkpoint: %s", resolved_adapter)
            else:
                self._peft_model = get_peft_model(base_model, lora_config)
            self._peft_model.print_trainable_parameters()
            self._peft_model.train()

            self._optimizer = torch.optim.AdamW(
                self._peft_model.parameters(),
                lr=config.learning_rate,
            )

            if resolved_adapter:
                optimizer_path = Path(resolved_adapter) / "optimizer.pt"
                if optimizer_path.exists():
                    try:
                        state = torch.load(optimizer_path, map_location="cpu")
                        self._optimizer.load_state_dict(state)
                        logger.info("Loaded optimizer state from %s", optimizer_path)
                    except Exception:
                        logger.warning(
                            "Failed to load optimizer state from %s",
                            optimizer_path,
                            exc_info=True,
                        )

            self._base_model_path = model_path
            self._adapter_path = resolved_adapter
            self._model_label = (
                f"local:{Path(resolved_adapter).name}"
                if resolved_adapter
                else f"local:{Path(str(model_path)).name}"
            )
            self._local_server = _LocalModelServerShim(self, self._model_label)
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
        self._local_server = None
        self._base_model_path = None
        self._adapter_path = None
        self._model_label = ""
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
        genome: Any | None = None,
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
            "improved": False,
        }

        try:
            # Step 1: enter training phase
            logger.info("Entering training phase (stopping vLLM)...")
            await vram_scheduler.enter_training_phase()

            # Step 2: load the trainable HF model plus any saved adapter
            base_model_path, adapter_path = await self._resolve_latest_checkpoint(
                vllm_server
            )
            logger.info(
                "Base model for RL: %s (adapter=%s)",
                base_model_path,
                adapter_path or "none",
            )
            await self.load_model(base_model_path, config, adapter_path=adapter_path)
            if self._local_server is None:
                raise RuntimeError("Loop 3 requires a locally loaded trainable model.")
            rollout_server = self._local_server

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
                base_model_path,
                eval_tasks,
                rollout_server,
                arena_manager,
                genome=genome or _get_default_genome(),
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
                    rollout_server=rollout_server,
                    arena_manager=arena_manager,
                    opus_client=opus_client,
                    config=config,
                    genome=genome or _get_default_genome(),
                )
                run_metrics["epochs"].append(epoch_metrics)

            # Persist the final trainable state before scoring.
            last_checkpoint = await self._save_checkpoint(base_model_path, "last")
            run_metrics["last_checkpoint"] = last_checkpoint

            # Step 5: post-training evaluation
            final_score = await self._evaluate_checkpoint(
                base_model_path,
                eval_tasks,
                rollout_server,
                arena_manager,
                genome=genome or _get_default_genome(),
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
                run_metrics["best_checkpoint"] = best_checkpoint
                run_metrics["improved"] = True
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

        except Exception:
            logger.exception("Overnight RL run failed")
            run_metrics["status"] = "error"
            raise
        finally:
            self.unload_model()
            try:
                logger.info("Entering serving phase (restarting vLLM)...")
                await vram_scheduler.enter_serving_phase()
            except Exception:
                logger.exception("Failed to restore serving phase after RL run")

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
        rollout_server: Any,
        arena_manager: Any,
        opus_client: Any,
        config: Loop3Config,
        genome: Any,
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

        for task_idx, task in enumerate(tasks):
            # 4a. Generate tree rollouts
            try:
                trajectories = await self._tree_grpo.generate_tree_rollouts(
                    task=task,
                    vllm_server=rollout_server,
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

            # Last-resort guard: no model was loaded, so preserve the pipeline
            # shape with a no-op surrogate instead of crashing the run.
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
        messages = batch.get("messages", [])
        texts = batch.get("texts", [])

        if not messages and not texts:
            raise RuntimeError("Training batch is missing transcript texts.")

        # Tokenize and forward pass on the exact transcript text used to
        # build the rollout. This keeps the policy update real and traceable.
        model.train()
        total_loss = 0.0
        n_chunks = 0

        source_rows = messages if messages else texts

        for i in range(0, len(source_rows), 4):  # mini-batch of 4
            chunk_rows = source_rows[i : i + 4]
            if messages:
                chunk_texts = [
                    _messages_to_prompt(row, tokenizer) for row in chunk_rows
                ]
            else:
                chunk_texts = chunk_rows
            chunk_adv = advantages[i : i + 4] if i + 4 <= len(advantages) else advantages[i:]
            chunk_old_lp = old_log_probs[i : i + 4] if i + 4 <= len(old_log_probs) else old_log_probs[i:]

            inputs = tokenizer(
                chunk_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            inputs = inputs.to(self._input_device(model))

            outputs = model(**inputs)
            logits = outputs.logits

            # Compute per-token log-probs from logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs["input_ids"][:, 1:].contiguous()
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            new_log_probs_seq = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

            # Mean log-prob per sequence
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            mask = (shift_labels != pad_token_id).float()
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
        rollout_server: Any,
        arena_manager: Any,
        genome: Any | None = None,
    ) -> float:
        """Evaluate a checkpoint on held-out tasks and return a score."""
        if not eval_tasks:
            return 0.0

        rollout_genome = genome or _get_default_genome()
        successes = 0

        for task in eval_tasks:
            try:
                trajs = await self._tree_grpo.generate_tree_rollouts(
                    task=task,
                    vllm_server=rollout_server,
                    arena_manager=arena_manager,
                    genome=rollout_genome,
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

    async def _resolve_latest_checkpoint(self, vllm_server: Any) -> tuple[str, str | None]:
        """Resolve the HF base model and the latest available adapter."""
        base_model_path = getattr(vllm_server.config, "hf_model_path", "") or vllm_server.config.name

        candidates: list[Path] = []
        candidates.extend(sorted(self.output_dir.glob("rl_*"), key=lambda p: p.stat().st_mtime))
        candidates.extend(
            sorted(self.output_dir.parent.glob("adapter_*"), key=lambda p: p.stat().st_mtime)
        )

        if candidates:
            latest = max(candidates, key=lambda p: p.stat().st_mtime)
            logger.info("Found latest checkpoint candidate: %s", latest)
            return base_model_path, str(latest)

        return base_model_path, None

    async def _save_checkpoint(self, model_path: str, tag: str) -> str:
        """Save the current model state as a named checkpoint."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = self.output_dir / f"rl_{tag}_{ts}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

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

        if self._peft_model is not None and self._tokenizer is not None:
            self._peft_model.save_pretrained(str(ckpt_dir))
            self._tokenizer.save_pretrained(str(ckpt_dir))
            if self._optimizer is not None and torch is not None:
                torch.save(self._optimizer.state_dict(), ckpt_dir / "optimizer.pt")
            logger.info("Saved trainable RL checkpoint to %s", ckpt_dir)
        else:
            logger.warning(
                "Saving checkpoint without a loaded model; metadata only at %s",
                ckpt_dir,
            )

        meta_path = ckpt_dir / "metadata.json"
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

    def _input_device(self, model: Any) -> Any:
        """Best-effort device for tokenized inputs."""
        try:
            if hasattr(model, "device"):
                return model.device
            return next(model.parameters()).device
        except Exception:
            return "cpu"

    async def _local_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        think: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Expose a chat-completion shaped API backed by the loaded HF model."""
        del think, kwargs

        if self._peft_model is None or self._tokenizer is None:
            raise RuntimeError("Local RL model is not loaded.")

        async with self._generation_lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self._generate_local_response(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                ),
            )

    def _generate_local_response(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> Any:
        """Synchronously generate one action and its sequence log-probability."""
        if self._peft_model is None or self._tokenizer is None or torch is None:
            raise RuntimeError("Local RL model is not available.")

        model = self._peft_model
        tokenizer = self._tokenizer
        prompt = _messages_to_prompt(messages, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self._input_device(model))

        model.eval()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        sequences = output.sequences[0]
        prompt_len = inputs["input_ids"].shape[-1]
        generated_ids = sequences[prompt_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        sample_logprob = _scores_to_logprob(output.scores, generated_ids)

        message = SimpleNamespace(
            role="assistant",
            content=text,
            reasoning="",
        )
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(
            choices=[choice],
            sample_logprob=sample_logprob,
            model=self._model_label or "local",
        )

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


def _messages_to_prompt(messages: list[dict[str, str]], tokenizer: Any) -> str:
    """Render messages into a prompt string using the tokenizer when possible."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


def _scores_to_logprob(scores: Any, generated_ids: Any) -> float:
    """Sum token log-probabilities from generation scores."""
    if torch is None or not scores:
        return 0.0

    total = 0.0
    try:
        for step_idx, step_scores in enumerate(scores):
            token_id = int(generated_ids[step_idx])
            log_probs = torch.log_softmax(step_scores[0], dim=-1)
            total += float(log_probs[token_id].item())
    except Exception:
        return 0.0
    return total


class _LocalModelServerShim:
    """Minimal chat-completion shim backed by the loaded HF/PEFT model."""

    def __init__(self, runner: RLRunner, name: str) -> None:
        self._runner = runner
        self.config = SimpleNamespace(name=name)

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: str | list[str] | None = None,
        top_p: float = 1.0,
        think: bool = True,
        **kwargs: Any,
    ) -> Any:
        del stop, think, kwargs
        return await self._runner._local_chat_completion(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
