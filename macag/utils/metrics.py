"""Faithfulness and utility metrics for MACAG solvers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from macag.graph import NodeId
from macag.scoring import ScoringOracle, TargetId

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaithfulnessMetrics:
    all_score: float
    empty_score: float
    keep_only_score: float
    remove_score: float
    sufficiency: float
    necessity: float
    faithfulness_delta: float


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
    return FaithfulnessMetrics(
        all_score=all_score,
        empty_score=empty_score,
        keep_only_score=keep_only_score,
        remove_score=remove_score,
        sufficiency=sufficiency,
        necessity=necessity,
        faithfulness_delta=faithfulness_delta,
    )


def game1_utility(faithfulness_delta: float, size: int, lam: float) -> float:
    return faithfulness_delta - lam * size


def game2_utility(
    faithfulness_delta: float,
    size: int,
    overlap_size: int,
    lam: float,
    beta: float,
) -> float:
    return faithfulness_delta - lam * size - beta * overlap_size


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
    }
