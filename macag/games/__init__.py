"""Game solvers for MACAG."""

from macag.games.game1_min_faithful import EvidenceSetResult, prefilter_candidates, solve_game1
from macag.games.game2_contrastive import ContrastiveEvidenceResult, solve_game2

__all__ = [
    "EvidenceSetResult",
    "ContrastiveEvidenceResult",
    "prefilter_candidates",
    "solve_game1",
    "solve_game2",
]
