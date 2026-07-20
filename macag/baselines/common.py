"""Shared value function, result container, and rank statistics for baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from macag.graph import NodeId
from macag.scoring import ScoringOracle, TargetId
from macag.utils.metrics import compute_faithfulness_metrics


def coalition_value(
    oracle: ScoringOracle,
    target: TargetId,
    nodes: set[NodeId],
    alpha: float,
) -> float:
    """The characteristic function v(S) of the underlying coalitional game.

    v(S) = alpha * (keep_only(S) - empty) + (1 - alpha) * (all - remove(S)),
    i.e. the alpha-mixed faithfulness_delta the games optimize (macag.md §3.0).
    Every baseline selects or ranks under this same v so only the selection
    rule differs across methods.
    """
    metrics = compute_faithfulness_metrics(oracle=oracle, target=target, nodes=set(nodes), alpha=alpha)
    return metrics.faithfulness_delta


@dataclass
class SelectionResult:
    """A baseline's output: a best-first ranking whose k-prefix is its evidence at budget k."""

    method: str
    ranking: list[NodeId]
    scores: dict[NodeId, float] | None
    params: dict[str, Any]
    extras: dict[str, Any] = field(default_factory=dict)

    def evidence_at(self, k: int) -> set[NodeId]:
        if k < 0:
            raise ValueError("k must be non-negative.")
        return set(self.ranking[:k])


def ranking_from_scores(scores: Mapping[NodeId, float]) -> list[NodeId]:
    """Deterministic best-first ranking: score descending, then node string."""
    return sorted(scores, key=lambda node: (-float(scores[node]), str(node)))


def jaccard(set_a: set[NodeId], set_b: set[NodeId]) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def precision_at_k(ranking: Sequence[NodeId], gold_ranking: Sequence[NodeId], k: int) -> float:
    """Fraction of the method's top-k that appears in the gold top-k."""
    if k <= 0:
        return 0.0
    top = set(ranking[:k])
    gold = set(gold_ranking[:k])
    if not top:
        return 0.0
    return len(top & gold) / k


def _average_ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks with ties assigned the average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for idx in order[i : j + 1]:
            ranks[idx] = average
        i = j + 1
    return ranks


def spearman_rank_correlation(
    scores_a: Mapping[NodeId, float],
    scores_b: Mapping[NodeId, float],
) -> float | None:
    """Spearman rho over the keys both score maps share (average-rank ties).

    Returns None when fewer than two common keys exist or either side is
    constant (rho undefined).
    """
    common = sorted(set(scores_a) & set(scores_b), key=str)
    if len(common) < 2:
        return None
    ranks_a = _average_ranks([float(scores_a[node]) for node in common])
    ranks_b = _average_ranks([float(scores_b[node]) for node in common])
    n = len(common)
    mean_a = sum(ranks_a) / n
    mean_b = sum(ranks_b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ranks_a, ranks_b))
    var_a = sum((x - mean_a) ** 2 for x in ranks_a)
    var_b = sum((y - mean_b) ** 2 for y in ranks_b)
    if var_a <= 0.0 or var_b <= 0.0:
        return None
    return cov / (var_a * var_b) ** 0.5
