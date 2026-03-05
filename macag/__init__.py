"""MACAG: mechanistic circuit evidence-allocation games."""

from macag.graph import CircuitGraph, NodeId
from macag.scoring import (
    CallbackInterventionScorer,
    InterventionScorer,
    ReplacementModelInterventionScorer,
    ScoringOracle,
    ToyAdditiveInterventionScorer,
)
from macag.factories.replacement_model import (
    OracleFactoryOutput,
    create_replacement_model_oracle,
    create_replacement_model_scorer,
)

__all__ = [
    "CircuitGraph",
    "NodeId",
    "InterventionScorer",
    "CallbackInterventionScorer",
    "ReplacementModelInterventionScorer",
    "ScoringOracle",
    "ToyAdditiveInterventionScorer",
    "OracleFactoryOutput",
    "create_replacement_model_oracle",
    "create_replacement_model_scorer",
]
