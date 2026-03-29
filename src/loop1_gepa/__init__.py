"""Loop 1: Guided Evolution of Prompt Architectures (GEPA).

Evolves PromptGenome configurations using Opus-driven mutation,
multi-objective Pareto selection, and arena-based evaluation.

Heavy imports (evaluate_genome, GEPAEngine, propose_mutation) are
lazy to avoid dragging in Docker/arena dependencies for lightweight
uses like ParetoTracker or prompt_genome utilities.
"""

from src.loop1_gepa.pareto_tracker import ParetoTracker
from src.loop1_gepa.prompt_genome import (
    create_seed_genome,
    crossover,
    dict_to_genome,
    genome_to_dict,
    load_population,
    save_population,
)


def __getattr__(name: str):
    """Lazy imports for heavy modules that pull in Docker/arena deps."""
    if name == "evaluate_genome":
        from src.loop1_gepa.evaluator import evaluate_genome
        return evaluate_genome
    if name == "GEPAEngine":
        from src.loop1_gepa.evolution_engine import GEPAEngine
        return GEPAEngine
    if name in ("MutationOperator", "propose_mutation"):
        from src.loop1_gepa import mutation_operators
        return getattr(mutation_operators, name)
    if name == "DspyGEPAEngine":
        from src.loop1_gepa.dspy_gepa_engine import DspyGEPAEngine
        return DspyGEPAEngine
    raise AttributeError(f"module 'src.loop1_gepa' has no attribute {name!r}")


__all__ = [
    "DspyGEPAEngine",
    "GEPAEngine",
    "MutationOperator",
    "ParetoTracker",
    "create_seed_genome",
    "crossover",
    "dict_to_genome",
    "evaluate_genome",
    "genome_to_dict",
    "load_population",
    "propose_mutation",
    "save_population",
]
