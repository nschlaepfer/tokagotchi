"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    # Local inference backend for Loop 1 / arena episodes.
    provider: str = "mlx"
    name: str = "mlx-community/Qwen3-14B-4bit"
    server_host: str = "127.0.0.1"
    server_port: int = 8080
    api_key: str = "local"
    auto_start: bool = True
    server_command: str = ""
    quantization: str = ""
    # HuggingFace model for training (Loop 2 SFT, Loop 3 RL). These loops
    # still use the existing PyTorch pipeline and are not MLX-native yet.
    hf_model_path: str = "models/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated"
    hf_model_path_27b: str = "huihui-ai/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated"
    # Legacy backend fields kept for backward compatibility / provider-specific
    # overrides when switching away from the MLX default.
    ollama_port: int = 11434
    ollama_host: str = "localhost"
    vllm_gpu_memory_utilization: float = 0.50
    vllm_port: int = 11434
    vllm_host: str = "localhost"

    @property
    def normalized_provider(self) -> str:
        return (self.provider or "mlx").strip().lower()

    @property
    def resolved_host(self) -> str:
        if self.server_host:
            return self.server_host
        if self.normalized_provider == "ollama":
            return self.ollama_host
        if self.normalized_provider == "vllm":
            return self.vllm_host
        return "127.0.0.1"

    @property
    def resolved_port(self) -> int:
        if self.server_port:
            return self.server_port
        if self.normalized_provider == "ollama":
            return self.ollama_port
        if self.normalized_provider == "vllm":
            return self.vllm_port
        return 8080

    @property
    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        if self.normalized_provider == "ollama":
            return "ollama"
        return "local"


@dataclass
class OpusConfig:
    daily_budget_usd: float = 50.0
    hourly_budget_usd: float = 10.0
    session_dir: str = "./data/opus_sessions"
    model: str = "claude-opus-4-6"
    default_max_turns: int = 10
    default_max_budget_per_call_usd: float = 0.50


@dataclass
class VRAMConfig:
    total_gb: int = 32
    serving_allocation_gb: int = 16
    training_allocation_gb: int = 32


@dataclass
class ScheduleConfig:
    loop1_continuous: bool = True
    loop2_trigger_threshold: int = 500
    loop2_diversity_min_types: int = 3
    loop3_start_hour: int = 22
    loop3_end_hour: int = 6
    eval_frequency_minutes: int = 60


@dataclass
class LoRAConfig:
    rank: int = 64
    alpha: int = 32
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    dropout: float = 0.05


@dataclass
class Loop1Config:
    population_size: int = 20
    experiments_per_hour: int = 50
    eval_tasks_per_genome: int = 10
    pareto_objectives: list[str] = field(default_factory=lambda: [
        "success_rate", "avg_steps", "tool_accuracy", "code_quality",
    ])
    mutation_temperature: float = 0.7
    crossover_rate: float = 0.2
    elite_size: int = 5
    max_prompt_length: int = 4000
    # DSPy GEPA integration
    use_dspy_gepa: bool = True
    dspy_num_threads: int = 4
    dspy_max_metric_calls: int = 150


@dataclass
class Loop2Config:
    teacher_ratio: float = 0.20
    mentor_ratio: float = 0.80
    min_buffer_size: int = 500
    max_buffer_size: int = 5000
    diversity_min_task_types: int = 3
    diversity_min_failure_modes: int = 3
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_steps: int = 500
    gradient_checkpointing: bool = True
    bf16: bool = True


@dataclass
class DAPOConfig:
    epsilon_low: float = 0.1
    epsilon_high: float = 0.28


@dataclass
class Loop3Config:
    algorithm: str = "grpo"
    tree_branching_factor: int = 4
    prefix_share_depth: int = 3
    rollout_temperature: float = 0.8
    rollout_top_p: float = 0.95
    dapo: DAPOConfig = field(default_factory=DAPOConfig)
    min_entropy: float = 0.01
    entropy_coeff: float = 0.01
    echo_trap_threshold: int = 3
    min_trajectory_reward: float = 0.1
    train_batch_size: int = 16
    rollout_n: int = 4
    learning_rate: float = 5e-7
    total_epochs: int = 3
    kl_loss_coef: float = 0.001
    gradient_checkpointing: bool = True
    bf16: bool = True


@dataclass
class ArenaConfig:
    docker_image: str = "qwen-arena:latest"
    memory_limit: str = "2g"
    cpu_limit: int = 2
    timeout_seconds: int = 60
    network: str = "none"
    max_tool_calls: int = 20
    warmup_containers: int = 4


@dataclass
class RewardWeights:
    outcome: float = 0.6
    process: float = 0.3
    efficiency: float = 0.1


@dataclass
class MasterConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    opus: OpusConfig = field(default_factory=OpusConfig)
    vram: VRAMConfig = field(default_factory=VRAMConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    loop1: Loop1Config = field(default_factory=Loop1Config)
    loop2: Loop2Config = field(default_factory=Loop2Config)
    loop3: Loop3Config = field(default_factory=Loop3Config)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    data_dir: str = "./data"


def _merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def _apply_dict(obj: Any, d: dict) -> None:
    """Recursively apply dict values to a dataclass."""
    for k, v in d.items():
        if hasattr(obj, k):
            attr = getattr(obj, k)
            if hasattr(attr, "__dataclass_fields__") and isinstance(v, dict):
                _apply_dict(attr, v)
            else:
                setattr(obj, k, v)


def load_config(config_dir: str | Path = "config") -> MasterConfig:
    """Load and merge all YAML configs into a MasterConfig."""
    config_dir = Path(config_dir)
    merged: dict = {}

    yaml_files = [
        "master.yaml", "loop1_gepa.yaml", "loop2_distill.yaml",
        "loop3_rl.yaml", "arena.yaml", "curriculum.yaml", "rewards.yaml",
    ]

    for fname in yaml_files:
        fpath = config_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                data = yaml.safe_load(f) or {}
            merged = _merge_dicts(merged, data)

    cfg = MasterConfig()
    _apply_dict(cfg, merged)
    return cfg
