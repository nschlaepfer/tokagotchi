"""Orchestrator modules for Opus-driven self-improvement loops."""


def __getattr__(name: str):
    if name == "BudgetExhaustedError":
        from src.orchestrator.budget_tracker import BudgetExhaustedError
        return BudgetExhaustedError
    if name == "BudgetTracker":
        from src.orchestrator.budget_tracker import BudgetTracker
        return BudgetTracker
    if name == "ExperimentGit":
        from src.orchestrator.experiment_git import ExperimentGit
        return ExperimentGit
    if name == "MasterLoop":
        from src.orchestrator.master_loop import MasterLoop
        return MasterLoop
    if name == "OpusClient":
        from src.orchestrator.opus_client import OpusClient
        return OpusClient
    if name == "OpusResponse":
        from src.orchestrator.opus_client import OpusResponse
        return OpusResponse
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BudgetExhaustedError",
    "BudgetTracker",
    "ExperimentGit",
    "MasterLoop",
    "OpusClient",
    "OpusResponse",
]
