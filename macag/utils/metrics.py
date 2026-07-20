"""Faithfulness and utility metrics for MACAG solvers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from macag.graph import NodeId
from macag.scoring import ScoringOracle, TargetId

LOGGER = logging.getLogger(__name__)


# Numerical guard for the normalized-metric denominator (recoverable range).
_RANGE_EPS = 1e-9


@dataclass(frozen=True)
class FaithfulnessMetrics:
    all_score: float
    empty_score: float
    keep_only_score: float
    remove_score: float
    sufficiency: float
    necessity: float
    faithfulness_delta: float
    # Error-node-aware normalization (C1). The "recoverable range" is the gap
    # between the full-circuit score and the all-features-ablated score. Because
    # error nodes are (by default) never ablated, ``empty_score`` carries an
    # error floor; dividing by the recoverable range yields metrics that stay
    # well-conditioned even when that floor dominates the absolute scores.
    recoverable_range: float
    sufficiency_normalized: float
    necessity_normalized: float
    faithfulness_delta_normalized: float


def compute_faithfulness_metrics(
    oracle: ScoringOracle,
    target: TargetId,
    nodes: set[NodeId],
    alpha: float,
) -> FaithfulnessMetrics:
    all_score = oracle.all(target)
    empty_score = oracle.empty(target)
    keep_only_score = oracle.keep_only(nodes, target)
    remove_score = oracle.remove(nodes, target)

    sufficiency = keep_only_score - empty_score
    necessity = all_score - remove_score
    faithfulness_delta = alpha * sufficiency + (1.0 - alpha) * necessity

    recoverable_range = all_score - empty_score
    if abs(recoverable_range) < _RANGE_EPS:
        sufficiency_normalized = 0.0
        necessity_normalized = 0.0
    else:
        sufficiency_normalized = sufficiency / recoverable_range
        necessity_normalized = necessity / recoverable_range
    faithfulness_delta_normalized = (
        alpha * sufficiency_normalized + (1.0 - alpha) * necessity_normalized
    )
    return FaithfulnessMetrics(
        all_score=all_score,
        empty_score=empty_score,
        keep_only_score=keep_only_score,
        remove_score=remove_score,
        sufficiency=sufficiency,
        necessity=necessity,
        faithfulness_delta=faithfulness_delta,
        recoverable_range=recoverable_range,
        sufficiency_normalized=sufficiency_normalized,
        necessity_normalized=necessity_normalized,
        faithfulness_delta_normalized=faithfulness_delta_normalized,
    )


def game1_utility(faithfulness_delta: float, size: int, lam: float) -> float:
    return faithfulness_delta - lam * size


def game2_utility(
    faithfulness_delta: float,
    size: int,
    overlap_weight: float,
    lam: float,
    beta: float,
) -> float:
    """Game 2 utility with a (possibly fractional) overlap penalty.

    ``overlap_weight`` is the hard overlap count ``|E ∩ E_other|`` under ABR, or
    the expected overlap ``sum_{n in E} p(n)`` against an empirical mixture of
    opponent sets under fictitious play.
    """
    return faithfulness_delta - lam * size - beta * overlap_weight


def overlap_rate(set_a: set[NodeId], set_b: set[NodeId]) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def sparsity(selected_size: int, total_size: int) -> float:
    if total_size == 0:
        return 0.0
    if selected_size > total_size:
        LOGGER.warning(
            "selected_size (%d) > total_size (%d) in sparsity calculation",
            selected_size,
            total_size,
        )
    return 1.0 - (selected_size / total_size)


def dedupe_preserve_order(sequence: Sequence[NodeId]) -> list[NodeId]:
    """Remove duplicates from a sequence while preserving insertion order."""
    seen: set[NodeId] = set()
    deduped: list[NodeId] = []
    for item in sequence:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def metrics_to_dict(metrics: FaithfulnessMetrics) -> dict[str, float]:
    return {
        "all": metrics.all_score,
        "empty": metrics.empty_score,
        "keep_only": metrics.keep_only_score,
        "remove": metrics.remove_score,
        "sufficiency": metrics.sufficiency,
        "necessity": metrics.necessity,
        "faithfulness": metrics.faithfulness_delta,
        # Error-node-aware normalized view (C1).
        "error_floor": metrics.empty_score,
        "recoverable_range": metrics.recoverable_range,
        "sufficiency_normalized": metrics.sufficiency_normalized,
        "necessity_normalized": metrics.necessity_normalized,
        "faithfulness_normalized": metrics.faithfulness_delta_normalized,
    }
