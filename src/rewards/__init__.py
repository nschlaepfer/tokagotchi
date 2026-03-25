"""Reward modules for trajectory evaluation."""


def __getattr__(name: str):
    if name == "CompositeReward":
        from src.rewards.composite_reward import CompositeReward
        return CompositeReward
    if name == "compute_efficiency_penalty":
        from src.rewards.efficiency_penalty import compute_efficiency_penalty
        return compute_efficiency_penalty
    if name == "compute_outcome_reward":
        from src.rewards.outcome_reward import compute_outcome_reward
        return compute_outcome_reward
    if name == "compute_process_reward":
        from src.rewards.process_reward import compute_process_reward
        return compute_process_reward
    if name == "average_process_score":
        from src.rewards.process_reward import average_process_score
        return average_process_score
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CompositeReward",
    "compute_efficiency_penalty",
    "compute_outcome_reward",
    "compute_process_reward",
    "average_process_score",
]
