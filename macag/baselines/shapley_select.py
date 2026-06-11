"""B2.2 — Monte-Carlo Shapley (and Banzhaf) over the MACAG intervention oracle.

This is the gold-standard credit reference of macag.md §3.6 / §A.4: the value
function is the SAME coalitional v(S) the games optimize — the alpha-mixed
faithfulness over keep_only/remove ablations through the oracle — NOT the
spline-CLT reconstruction game in attribution/shapley.py, which measures a
different v on a different object and must not be wrapped here.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Sequence

from macag.baselines.common import SelectionResult, coalition_value, ranking_from_scores
from macag.graph import NodeId
from macag.scoring import ScoringOracle, TargetId
from macag.utils.metrics import dedupe_preserve_order

LOGGER = logging.getLogger(__name__)


@dataclass
class ShapleyEstimate:
    """Per-node credit estimates over the MACAG coalitional game."""

    values: dict[NodeId, float]
    std_errors: dict[NodeId, float]
    samples_per_node: int
    estimator: str
    seed: int
    base_value: float
    grand_value: float
    # sum(values) - (grand_value - base_value); 0 in expectation for Shapley
    # (efficiency axiom), a finite-sample diagnostic here.
    efficiency_gap: float
    params: dict[str, object] = field(default_factory=dict)

    def ranking(self) -> list[NodeId]:
        return ranking_from_scores(self.values)


def _finalize(
    sums: dict[NodeId, float],
    sumsqs: dict[NodeId, float],
    pool: list[NodeId],
    count: int,
    estimator: str,
    seed: int,
    base_value: float,
    grand_value: float,
    params: dict[str, object],
) -> ShapleyEstimate:
    values: dict[NodeId, float] = {}
    std_errors: dict[NodeId, float] = {}
    for node in pool:
        mean = sums[node] / count
        variance = max(0.0, sumsqs[node] / count - mean * mean)
        values[node] = mean
        std_errors[node] = math.sqrt(variance / count)
    efficiency_gap = sum(values.values()) - (grand_value - base_value)
    return ShapleyEstimate(
        values=values,
        std_errors=std_errors,
        samples_per_node=count,
        estimator=estimator,
        seed=seed,
        base_value=base_value,
        grand_value=grand_value,
        efficiency_gap=efficiency_gap,
        params=params,
    )


def estimate_shapley(
    oracle: ScoringOracle,
    target: TargetId,
    candidates: Sequence[NodeId],
    alpha: float = 0.5,
    permutations: int = 64,
    seed: int = 0,
    antithetic: bool = True,
    progress: bool = False,
) -> ShapleyEstimate:
    """Permutation-sampling Shapley estimator with optional antithetic pairing.

    Each permutation walks the pool front-to-back, charging every node its
    marginal v-gain at the coalition built so far: |C| coalition evaluations
    per permutation, each costing at most two fresh oracle calls (keep_only and
    remove; all/empty are cached after the first). Antithetic pairing runs the
    reversed permutation alongside, which cancels order noise for near-additive
    v. Deterministic for a fixed seed.
    """
    if permutations <= 0:
        raise ValueError("permutations must be positive.")
    pool = dedupe_preserve_order(candidates)
    if not pool:
        raise ValueError("Shapley estimation needs a non-empty candidate pool.")

    rng = random.Random(seed)
    sums: dict[NodeId, float] = {node: 0.0 for node in pool}
    sumsqs: dict[NodeId, float] = {node: 0.0 for node in pool}
    base_value = coalition_value(oracle, target, set(), alpha)

    runs = 0
    while runs < permutations:
        permutation = rng.sample(pool, len(pool))
        batch = [permutation]
        if antithetic and runs + 1 < permutations:
            batch.append(list(reversed(permutation)))
        for ordering in batch:
            previous = base_value
            coalition: set[NodeId] = set()
            for node in ordering:
                coalition.add(node)
                value = coalition_value(oracle, target, coalition, alpha)
                marginal = value - previous
                sums[node] += marginal
                sumsqs[node] += marginal * marginal
                previous = value
            runs += 1
        if progress and runs % 8 == 0:
            LOGGER.info("Shapley: %d/%d permutations", runs, permutations)

    grand_value = coalition_value(oracle, target, set(pool), alpha)
    return _finalize(
        sums,
        sumsqs,
        pool,
        runs,
        estimator="shapley",
        seed=seed,
        base_value=base_value,
        grand_value=grand_value,
        params={"alpha": alpha, "permutations": runs, "antithetic": antithetic},
    )


def estimate_banzhaf(
    oracle: ScoringOracle,
    target: TargetId,
    candidates: Sequence[NodeId],
    alpha: float = 0.5,
    samples: int = 64,
    seed: int = 0,
    progress: bool = False,
) -> ShapleyEstimate:
    """Monte-Carlo Banzhaf: marginals against uniformly random coalitions.

    Per sample, draws S by independent fair coin flips and charges every node
    v(S + node) - v(S - node). Less order-sensitive than Shapley for strongly
    interacting features (macag.md §3.6); the natural second gold reference.
    """
    if samples <= 0:
        raise ValueError("samples must be positive.")
    pool = dedupe_preserve_order(candidates)
    if not pool:
        raise ValueError("Banzhaf estimation needs a non-empty candidate pool.")

    rng = random.Random(seed)
    sums: dict[NodeId, float] = {node: 0.0 for node in pool}
    sumsqs: dict[NodeId, float] = {node: 0.0 for node in pool}
    base_value = coalition_value(oracle, target, set(), alpha)

    for run in range(1, samples + 1):
        sample = {node for node in pool if rng.random() < 0.5}
        for node in pool:
            with_node = sample | {node}
            without_node = sample - {node}
            marginal = coalition_value(oracle, target, with_node, alpha) - coalition_value(
                oracle, target, without_node, alpha
            )
            sums[node] += marginal
            sumsqs[node] += marginal * marginal
        if progress and run % 8 == 0:
            LOGGER.info("Banzhaf: %d/%d samples", run, samples)

    grand_value = coalition_value(oracle, target, set(pool), alpha)
    return _finalize(
        sums,
        sumsqs,
        pool,
        samples,
        estimator="banzhaf",
        seed=seed,
        base_value=base_value,
        grand_value=grand_value,
        params={"alpha": alpha, "samples": samples},
    )


def select_top_shapley(
    oracle: ScoringOracle,
    target: TargetId,
    candidates: Sequence[NodeId],
    alpha: float = 0.5,
    permutations: int = 64,
    seed: int = 0,
    antithetic: bool = True,
    estimator: str = "shapley",
    progress: bool = False,
) -> SelectionResult:
    """Rank candidates by estimated Shapley (or Banzhaf) value, best first."""
    if estimator == "shapley":
        estimate = estimate_shapley(
            oracle,
            target,
            candidates,
            alpha=alpha,
            permutations=permutations,
            seed=seed,
            antithetic=antithetic,
            progress=progress,
        )
    elif estimator == "banzhaf":
        estimate = estimate_banzhaf(
            oracle, target, candidates, alpha=alpha, samples=permutations, seed=seed, progress=progress
        )
    else:
        raise ValueError("estimator must be 'shapley' or 'banzhaf'.")

    return SelectionResult(
        method=estimator,
        ranking=estimate.ranking(),
        scores=dict(estimate.values),
        params=dict(estimate.params) | {"seed": seed},
        extras={
            "std_errors": {str(node): err for node, err in estimate.std_errors.items()},
            "samples_per_node": estimate.samples_per_node,
            "base_value": estimate.base_value,
            "grand_value": estimate.grand_value,
            "efficiency_gap": estimate.efficiency_gap,
        },
    )
