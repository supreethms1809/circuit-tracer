"""B1 — singleton-gain and random-k floors (opt-in, published baselines untouched).

- ``singleton``: rank candidates by their singleton coalition value v({n}) under
  the same capped v the games optimize. This is the Game 1 prefilter order as a
  standalone selector: the "does greedy interaction modeling beat independent
  singleton ranking?" floor. Costs |C| coalition evaluations.
- ``random``: seeded uniform shuffle of the candidate pool. The chance floor:
  any method must beat a random size-k set. Costs zero oracle calls.

Neither modifies the published baselines (influence/EAP/ACDC/Shapley); they are
additional floors selected via ``--methods`` (not in the default set).
"""

from __future__ import annotations

import random
from typing import Sequence

from macag.baselines.common import SelectionResult, coalition_value, ranking_from_scores
from macag.graph import NodeId
from macag.scoring import ScoringOracle, TargetId
from macag.utils.metrics import dedupe_preserve_order


def select_top_singleton(
    oracle: ScoringOracle,
    target: TargetId,
    candidates: Sequence[NodeId],
    alpha: float = 0.5,
    cap_sufficiency: bool = True,
    cap_necessity: bool = True,
) -> SelectionResult:
    """Rank candidates by singleton coalition value v({n}), best first."""
    pool = dedupe_preserve_order(candidates)
    if not pool:
        raise ValueError("Singleton ranking needs a non-empty candidate pool.")
    scores = {
        node: coalition_value(
            oracle,
            target,
            {node},
            alpha,
            cap_sufficiency=cap_sufficiency,
            cap_necessity=cap_necessity,
        )
        for node in pool
    }
    return SelectionResult(
        method="singleton",
        ranking=ranking_from_scores(scores),
        scores=dict(scores),
        params={
            "alpha": alpha,
            "cap_sufficiency": cap_sufficiency,
            "cap_necessity": cap_necessity,
        },
    )


def select_random(
    candidates: Sequence[NodeId],
    seed: int = 0,
) -> SelectionResult:
    """Seeded uniform shuffle of the candidate pool (chance floor)."""
    pool = dedupe_preserve_order(candidates)
    if not pool:
        raise ValueError("Random ranking needs a non-empty candidate pool.")
    rng = random.Random(seed)
    ranking = rng.sample(list(pool), len(pool))
    return SelectionResult(
        method="random",
        ranking=ranking,
        scores=None,
        params={"seed": seed},
    )
