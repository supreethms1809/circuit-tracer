"""Faithfulness and utility metrics for MACAG solvers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from macag.graph import NodeId
from macag.scoring import ScoringOracle, TargetId
from macag.nvtx import nvtx_range

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
    # ``sufficiency``/``necessity`` are the values mixed into faithfulness.
    # When the caps are set (default), sufficiency cannot exceed the clean gap
    # and necessity cannot exceed the empty-set floor: keep-only overshoot and
    # below-empty removal are measurement artifacts, not recovery. The uncapped
    # gaps, the capped sufficiency, and both overshoot magnitudes
    # (``keep_excess`` = max(0, keep-all); ``keep_distance`` = |keep-all|;
    # ``remove_below_empty`` = max(0, empty-remove)) are always recorded, and
    # ``is_degenerate`` flags R <= 0 cells where capped sufficiency cannot
    # discriminate (the empty set already sits above the clean gap).
    sufficiency_uncapped: float = 0.0
    sufficiency_capped: float = 0.0
    necessity_uncapped: float = 0.0
    keep_excess: float = 0.0
    keep_distance: float = 0.0
    remove_below_empty: float = 0.0
    cap_sufficiency: bool = True
    cap_necessity: bool = True
    is_degenerate: bool = False


def compute_faithfulness_metrics(
    oracle: ScoringOracle,
    target: TargetId,
    nodes: set[NodeId],
    alpha: float,
    cap_sufficiency: bool = True,
    cap_necessity: bool = True,
) -> FaithfulnessMetrics:
    with nvtx_range("game1.faithfulness"):
        all_score = oracle.all(target)
        empty_score = oracle.empty(target)
        keep_only_score = oracle.keep_only(nodes, target)
        remove_score = oracle.remove(nodes, target)

        sufficiency_uncapped = keep_only_score - empty_score
        necessity_uncapped = all_score - remove_score
        # min(keep, clean) - empty. Once keep-only reaches the clean score,
        # further increases do not raise sufficiency.
        sufficiency_capped = min(keep_only_score, all_score) - empty_score
        sufficiency = sufficiency_capped if cap_sufficiency else sufficiency_uncapped
        if cap_necessity:
            necessity = all_score - max(remove_score, empty_score)
        else:
            necessity = necessity_uncapped
        faithfulness_delta = alpha * sufficiency + (1.0 - alpha) * necessity

        recoverable_range = all_score - empty_score
        keep_excess = max(0.0, keep_only_score - all_score)
        keep_distance = abs(keep_only_score - all_score)
        remove_below_empty = max(0.0, empty_score - remove_score)
        is_degenerate = recoverable_range <= 0.0
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
            sufficiency_uncapped=sufficiency_uncapped,
            sufficiency_capped=sufficiency_capped,
            necessity_uncapped=necessity_uncapped,
            keep_excess=keep_excess,
            keep_distance=keep_distance,
            remove_below_empty=remove_below_empty,
            cap_sufficiency=cap_sufficiency,
            cap_necessity=cap_necessity,
            is_degenerate=is_degenerate,
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


def metrics_to_dict(metrics: FaithfulnessMetrics) -> dict[str, Any]:
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
        # Capped-faithfulness diagnostics: uncapped values + overshoot
        # magnitudes so "Game 1 has higher necessity" can be read next to N/R.
        "sufficiency_uncapped": metrics.sufficiency_uncapped,
        "sufficiency_capped": metrics.sufficiency_capped,
        "necessity_uncapped": metrics.necessity_uncapped,
        "keep_excess": metrics.keep_excess,
        "keep_distance": metrics.keep_distance,
        "remove_below_empty": metrics.remove_below_empty,
        "is_degenerate": metrics.is_degenerate,
        "cap_sufficiency": metrics.cap_sufficiency,
        "cap_necessity": metrics.cap_necessity,
    }
