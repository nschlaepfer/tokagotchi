"""Loop 1: Guided Evolution of Prompt Architectures (GEPA).

Evolves PromptGenome configurations using Opus-driven mutation,
multi-objective Pareto selection, and arena-based evaluation.
"""

from src.loop1_gepa.evaluator import evaluate_genome
from src.loop1_gepa.evolution_engine import GEPAEngine
from src.loop1_gepa.mutation_operators import MutationOperator, propose_mutation
from src.loop1_gepa.pareto_tracker import ParetoTracker
from src.loop1_gepa.prompt_genome import (
    create_seed_genome,
    crossover,
    dict_to_genome,
    genome_to_dict,
    load_population,
    save_population,
)

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
