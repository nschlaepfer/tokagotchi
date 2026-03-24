"""Tree-GRPO: Group Relative Policy Optimisation with shared-prefix rollouts.

Instead of K fully independent rollouts per task, Tree-GRPO generates a
single trajectory up to a configurable branch point and then forks into K
independent completions.  This shares the (expensive) prefix computation
while still producing the diversity required for GRPO advantage estimation.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from typing import Any

from src.config import Loop3Config
from src.models import (
    ActionType,
    RewardResult,
    StepRecord,
    TaskSpec,
    Trajectory,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional torch import
# ---------------------------------------------------------------------------

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class TreeGRPO:
    """Tree-structured Group Relative Policy Optimisation.

    Generates shared-prefix rollouts for a task, computes GRPO advantages
    across the group, and prepares RL training batches.
    """

    # ------------------------------------------------------------------
    # Tree rollout generation
    # ------------------------------------------------------------------

    @staticmethod
    async def generate_tree_rollouts(
        task: TaskSpec,
        vllm_server: Any,
        arena_manager: Any,
        genome: Any,
        config: Loop3Config,
    ) -> list[Trajectory]:
        """Generate K rollouts that share a common prefix.

        Parameters
        ----------
        task:
            The task to solve.
        vllm_server:
            Running VLLMServer instance used for model inference.
        arena_manager:
            DockerManager for sandboxed execution.
        genome:
            The current PromptGenome driving the system prompt.
        config:
            Loop 3 configuration (branching factor, prefix depth, etc.).

        Returns
        -------
        list[Trajectory]
            ``config.tree_branching_factor`` trajectories, each containing the
            shared prefix followed by an independently sampled suffix.
        """
        from src.arena.game import AgentArenaGame

        branching_factor: int = config.tree_branching_factor
        prefix_depth: int = config.prefix_share_depth

        # --- Phase 1: shared prefix -----------------------------------------
        prefix_game = AgentArenaGame(arena_manager)
        initial_obs = await prefix_game.reset(task)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": genome.to_system_message()},
            {"role": "user", "content": initial_obs},
        ]

        prefix_steps: list[StepRecord] = []
        done = False

        for step_idx in range(prefix_depth):
            if done:
                break

            response = await _sample_action(
                vllm_server, messages, config, temperature=config.rollout_temperature
            )
            step_result = await prefix_game.step(response)

            record = StepRecord(
                step_idx=step_idx,
                action_type=_parse_action_type(response),
                action_content=response,
                observation=step_result.observation,
                reasoning="",
                reward=step_result.reward,
                metadata=step_result.info,
            )
            prefix_steps.append(record)

            messages.append({"role": "assistant", "content": response})
            if step_result.observation:
                messages.append({"role": "user", "content": step_result.observation})

            done = step_result.done

        # Snapshot the prefix state
        prefix_messages = list(messages)
        await prefix_game.close()

        # If the prefix already completed the task, duplicate it K times
        if done:
            return _replicate_trajectory(task, prefix_steps, branching_factor, genome)

        # --- Phase 2: fork into K independent continuations ------------------
        branch_coros = [
            _run_branch(
                branch_idx=i,
                task=task,
                prefix_steps=prefix_steps,
                prefix_messages=prefix_messages,
                vllm_server=vllm_server,
                arena_manager=arena_manager,
                genome=genome,
                config=config,
            )
            for i in range(branching_factor)
        ]

        trajectories = await asyncio.gather(*branch_coros, return_exceptions=True)

        results: list[Trajectory] = []
        for traj in trajectories:
            if isinstance(traj, BaseException):
                logger.warning("Branch rollout failed: %s", traj)
                continue
            results.append(traj)

        # Guarantee at least one trajectory
        if not results:
            results.append(
                Trajectory(
                    task=task,
                    steps=list(prefix_steps),
                    success=False,
                    total_reward=sum(s.reward for s in prefix_steps),
                    model_id=getattr(vllm_server, "config", None)
                    and vllm_server.config.name
                    or "",
                    prompt_genome_id=getattr(genome, "genome_id", ""),
                )
            )

        return results

    # ------------------------------------------------------------------
    # GRPO advantage computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_grpo_advantages(
        trajectories: list[Trajectory],
        rewards: list[float],
    ) -> list[float]:
        """Compute GRPO advantages within a group of rollouts.

        For each trajectory *i* in the group:

            advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)

        Parameters
        ----------
        trajectories:
            K rollouts for the same task (used only for length validation).
        rewards:
            Per-trajectory scalar rewards, same length as *trajectories*.

        Returns
        -------
        list[float]
            Per-trajectory advantage values.
        """
        if len(rewards) == 0:
            return []

        n = len(rewards)
        mean_r = sum(rewards) / n
        var_r = sum((r - mean_r) ** 2 for r in rewards) / max(n, 1)
        std_r = var_r ** 0.5

        eps = 1e-8
        return [(r - mean_r) / (std_r + eps) for r in rewards]

    # ------------------------------------------------------------------
    # Training batch preparation
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_training_batch(
        all_trajectories: list[list[Trajectory]],
        all_advantages: list[list[float]],
    ) -> dict[str, Any]:
        """Convert grouped trajectories + advantages into a training batch.

        Each training example is built from the full message sequence of
        the trajectory and annotated with its advantage and an estimated
        old log-probability placeholder (to be filled by a reference
        forward pass before the actual gradient step).

        Parameters
        ----------
        all_trajectories:
            Outer list = per-task groups; inner list = K trajectories.
        all_advantages:
            Matching structure with per-trajectory advantage values.

        Returns
        -------
        dict
            Batch dictionary with keys:
            ``input_ids``, ``attention_mask``, ``advantages``,
            ``old_log_probs``, ``metadata``.
        """
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for prepare_training_batch but is not installed."
            )

        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_advantages: list[float] = []
        batch_old_log_probs: list[float] = []
        batch_metadata: list[dict[str, Any]] = []

        for task_trajs, task_advs in zip(all_trajectories, all_advantages):
            for traj, adv in zip(task_trajs, task_advs):
                tokens = _trajectory_to_token_ids(traj)
                batch_input_ids.append(tokens)
                batch_attention_mask.append([1] * len(tokens))
                batch_advantages.append(adv)
                # Placeholder — the caller must run a reference forward pass
                # to fill in actual old log-probs before the gradient step.
                batch_old_log_probs.append(0.0)
                batch_metadata.append(
                    {
                        "trajectory_id": traj.trajectory_id,
                        "task_id": traj.task.task_id if traj.task else "",
                        "num_steps": traj.num_steps,
                        "advantage": adv,
                    }
                )

        # Pad to uniform length
        max_len = max((len(ids) for ids in batch_input_ids), default=0)
        for i in range(len(batch_input_ids)):
            pad_len = max_len - len(batch_input_ids[i])
            batch_input_ids[i].extend([0] * pad_len)
            batch_attention_mask[i].extend([0] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "advantages": torch.tensor(batch_advantages, dtype=torch.float32),
            "old_log_probs": torch.tensor(batch_old_log_probs, dtype=torch.float32),
            "metadata": batch_metadata,
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _sample_action(
    vllm_server: Any,
    messages: list[dict[str, str]],
    config: Loop3Config,
    temperature: float | None = None,
) -> str:
    """Request a single action from the model via the vLLM server."""
    client = vllm_server._client  # AsyncOpenAI client
    temp = temperature if temperature is not None else config.rollout_temperature

    response = await client.chat.completions.create(
        model=vllm_server.config.name,
        messages=messages,  # type: ignore[arg-type]
        temperature=temp,
        top_p=config.rollout_top_p,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def _parse_action_type(raw: str) -> ActionType:
    """Best-effort parse of the action type from a raw action string."""
    import re

    pattern = re.compile(r"^\[?([a-z_]+)\]?\s*:", re.IGNORECASE)
    m = pattern.match(raw.strip())
    if m:
        try:
            return ActionType(m.group(1).lower())
        except ValueError:
            pass
    return ActionType.THINK


async def _run_branch(
    branch_idx: int,
    task: TaskSpec,
    prefix_steps: list[StepRecord],
    prefix_messages: list[dict[str, str]],
    vllm_server: Any,
    arena_manager: Any,
    genome: Any,
    config: Loop3Config,
) -> Trajectory:
    """Run a single continuation branch from the shared prefix."""
    from src.arena.game import AgentArenaGame

    game = AgentArenaGame(arena_manager)
    initial_obs = await game.reset(task)

    # Replay prefix steps inside this fresh container
    messages = list(prefix_messages)
    replayed_steps: list[StepRecord] = []

    for step in prefix_steps:
        result = await game.step(step.action_content)
        replayed_steps.append(
            StepRecord(
                step_idx=step.step_idx,
                action_type=step.action_type,
                action_content=step.action_content,
                observation=result.observation,
                reasoning=step.reasoning,
                reward=result.reward,
                metadata=result.info,
            )
        )

    # Continue from the branch point
    all_steps = list(replayed_steps)
    done = False
    max_suffix_steps = config.arena.max_tool_calls if hasattr(config, "arena") else 20
    step_idx = len(prefix_steps)
    start_time = time.monotonic()

    while not done and step_idx < max_suffix_steps:
        response = await _sample_action(
            vllm_server, messages, config, temperature=config.rollout_temperature
        )
        result = await game.step(response)

        record = StepRecord(
            step_idx=step_idx,
            action_type=_parse_action_type(response),
            action_content=response,
            observation=result.observation,
            reasoning="",
            reward=result.reward,
            metadata=result.info,
        )
        all_steps.append(record)

        messages.append({"role": "assistant", "content": response})
        if result.observation:
            messages.append({"role": "user", "content": result.observation})

        done = result.done
        step_idx += 1

    await game.close()
    wall_time = time.monotonic() - start_time

    return Trajectory(
        task=task,
        steps=all_steps,
        success=done
        and any(s.metadata.get("episode_complete") for s in all_steps),
        total_reward=sum(s.reward for s in all_steps),
        wall_time_seconds=wall_time,
        model_id=getattr(vllm_server, "config", None)
        and vllm_server.config.name
        or "",
        prompt_genome_id=getattr(genome, "genome_id", ""),
    )


def _replicate_trajectory(
    task: TaskSpec,
    prefix_steps: list[StepRecord],
    k: int,
    genome: Any,
) -> list[Trajectory]:
    """Create K identical trajectory copies when the prefix already solved the task."""
    trajs: list[Trajectory] = []
    for _ in range(k):
        trajs.append(
            Trajectory(
                task=task,
                steps=copy.deepcopy(prefix_steps),
                success=True,
                total_reward=sum(s.reward for s in prefix_steps),
                prompt_genome_id=getattr(genome, "genome_id", ""),
            )
        )
    return trajs


def _trajectory_to_token_ids(trajectory: Trajectory) -> list[int]:
    """Convert a trajectory into a pseudo-token-id sequence.

    In production, this would use the model tokenizer.  Here we encode
    each step's action content as UTF-8 byte values so that the batch
    structure is correct.  The actual tokenisation is deferred to the
    training loop which has access to the tokenizer.
    """
    token_ids: list[int] = []
    for step in trajectory.steps:
        encoded = step.action_content.encode("utf-8", errors="replace")
        token_ids.extend(list(encoded))
    return token_ids if token_ids else [0]
