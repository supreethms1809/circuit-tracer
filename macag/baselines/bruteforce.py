"""B3.2 — exact best size-k subset by exhaustive search (greedy optimality gap).

On small candidate pools (prefilter to ~12-15 first) this brute-forces the
best size-k coalition under the same v(S), giving the empirical optimality gap
that backs the (1-1/e)/non-submodularity discussion in macag.md §3.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from typing import Any, Sequence

from macag.baselines.common import coalition_value
from macag.graph import NodeId
from macag.scoring import ScoringOracle, TargetId
from macag.utils.metrics import dedupe_preserve_order


@dataclass
class BruteForceResult:
    k: int
    best_set: list[NodeId]
    best_value: float
    evaluations: int
    ties: int
    params: dict[str, Any] = field(default_factory=dict)


def best_subset_bruteforce(
    oracle: ScoringOracle,
    target: TargetId,
    candidates: Sequence[NodeId],
    k: int,
    alpha: float = 0.5,
    max_evaluations: int = 100_000,
) -> BruteForceResult:
    """Exhaustively evaluate every size-k subset and return the v-maximizer.

    Deterministic: candidates are sorted, combinations enumerate
    lexicographically, and the first maximizer wins ties (tie count reported).
    Refuses to run when C(n, k) exceeds ``max_evaluations`` — prefilter the
    pool instead of silently sampling.
    """
    pool = sorted(dedupe_preserve_order(candidates), key=str)
    if k <= 0 or k > len(pool):
        raise ValueError(f"k must be in [1, {len(pool)}], got {k}.")
    total = comb(len(pool), k)
    if total > max_evaluations:
        raise ValueError(
            f"C({len(pool)}, {k}) = {total} subsets exceeds max_evaluations="
            f"{max_evaluations}; prefilter the candidate pool (macag.md B3.2 "
            "recommends ~12-15 candidates) or raise the cap explicitly."
        )

    best_set: tuple[NodeId, ...] | None = None
    best_value = float("-inf")
    ties = 0
    evaluations = 0
    for subset in combinations(pool, k):
        value = coalition_value(oracle, target, set(subset), alpha)
        evaluations += 1
        if value > best_value:
            best_value = value
            best_set = subset
            ties = 0
        elif value == best_value:
            ties += 1

    assert best_set is not None  # k >= 1 and pool non-empty
    return BruteForceResult(
        k=k,
        best_set=list(best_set),
        best_value=best_value,
        evaluations=evaluations,
        ties=ties,
        params={"alpha": alpha, "pool_size": len(pool)},
    )
