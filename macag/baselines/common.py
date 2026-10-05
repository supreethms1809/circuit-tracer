"""Shared value function, result container, and rank statistics for baselines."""

from __future__ import annotations

import math
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
    cap_sufficiency: bool = True,
    cap_necessity: bool = True,
) -> float:
    """The characteristic function v(S) of the underlying coalitional game.

    v(S) = alpha * (keep_only(S) - empty) + (1 - alpha) * (all - remove(S)),
    i.e. the alpha-mixed faithfulness_delta the games optimize (macag.md §3.0).
    Sufficiency is keep_only(S) - empty, or min(keep_only(S), all) - empty
    when ``cap_sufficiency`` is set, so a set is not rewarded for driving the
    logit gap past the clean model. Every baseline selects or ranks under
    this same v so only the selection rule differs across methods.

    Caps (A2/A2', default on) bound both terms by the recoverable range:
    sufficiency uses min(keep, all) - empty, necessity uses all - max(remove,
    empty). All callers (Game 1, Game 2, Shapley/Banzhaf gold, prefix
    evaluation) must pass the same flags so the gold ranks under the v Game 1
    optimizes [G 1.3].
    """
    metrics = compute_faithfulness_metrics(
        oracle=oracle,
        target=target,
        nodes=set(nodes),
        alpha=alpha,
        cap_sufficiency=cap_sufficiency,
        cap_necessity=cap_necessity,
    )
    value = float(metrics.faithfulness_delta)
    if not math.isfinite(value):
        raise ValueError(f"Coalition value must be finite, got {value!r}.")
    return value


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


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def ranking_from_scores(
    scores: Mapping[NodeId, float],
    *,
    tie_tol: float = 1e-12,
) -> list[NodeId]:
    """Deterministic best-first ranking with finite checks and tolerant ties."""
    if tie_tol < 0:
        raise ValueError("tie_tol must be non-negative.")
    parsed = {node: float(score) for node, score in scores.items()}
    nonfinite = {str(node): score for node, score in parsed.items() if not math.isfinite(score)}
    if nonfinite:
        raise ValueError(f"Ranking scores must be finite, got {nonfinite}.")

    ordered = sorted(parsed, key=lambda node: (-parsed[node], str(node)))
    ranking: list[NodeId] = []
    index = 0
    while index < len(ordered):
        anchor = parsed[ordered[index]]
        end = index + 1
        while end < len(ordered) and _close(parsed[ordered[end]], anchor, tie_tol):
            end += 1
        ranking.extend(sorted(ordered[index:end], key=str))
        index = end
    return ranking


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


def tie_aware_precision_at_k(
    ranking: Sequence[NodeId],
    gold_scores: Mapping[NodeId, float],
    k: int,
    *,
    tie_tol: float = 1e-12,
) -> float:
    """Expected precision@k when a gold score tie crosses the k boundary."""
    if k <= 0:
        return 0.0
    gold_order = ranking_from_scores(gold_scores, tie_tol=tie_tol)
    if not gold_order:
        return 0.0
    boundary_index = min(k, len(gold_order)) - 1
    boundary = float(gold_scores[gold_order[boundary_index]])
    above = {
        node
        for node, score in gold_scores.items()
        if float(score) > boundary and not _close(float(score), boundary, tie_tol)
    }
    tied = {
        node
        for node, score in gold_scores.items()
        if _close(float(score), boundary, tie_tol)
    }
    remaining = max(0, min(k, len(gold_order)) - len(above))
    tied_credit = (remaining / len(tied)) if tied else 0.0
    selected = set(ranking[:k])
    credit = sum(1.0 for node in selected if node in above)
    credit += sum(tied_credit for node in selected if node in tied)
    return credit / k


def precision_at_k_uncertainty_bounds(
    ranking: Sequence[NodeId],
    gold_scores: Mapping[NodeId, float],
    gold_std_errors: Mapping[NodeId, float],
    k: int,
    *,
    z: float = 1.96,
) -> tuple[float, float]:
    """Conservative precision@k bounds from per-node normal confidence intervals."""
    if k <= 0:
        return 0.0, 0.0
    nodes = list(gold_scores)
    intervals: dict[NodeId, tuple[float, float]] = {}
    for node in nodes:
        score = float(gold_scores[node])
        se = float(gold_std_errors.get(node, float("nan")))
        if not math.isfinite(score):
            raise ValueError(f"Gold score for {node!r} must be finite.")
        if not math.isfinite(se) or se < 0:
            # Undefined one-draw uncertainty: no node can be certified, and all
            # nodes remain possible at the boundary.
            intervals[node] = (float("-inf"), float("inf"))
        else:
            intervals[node] = (score - z * se, score + z * se)

    lowers = sorted((lo for lo, _ in intervals.values()), reverse=True)
    kth_lower = lowers[min(k, len(lowers)) - 1]
    possible = {node for node, (_, hi) in intervals.items() if hi >= kth_lower}

    uppers = sorted((hi for _, hi in intervals.values()), reverse=True)
    next_upper = uppers[k] if k < len(uppers) else float("-inf")
    definite = {node for node, (lo, _) in intervals.items() if lo > next_upper}

    selected = set(ranking[:k])
    return len(selected & definite) / k, min(k, len(selected & possible)) / k


def _average_ranks(values: Sequence[float], tie_tol: float = 1e-12) -> list[float]:
    """1-based ranks with ties assigned the average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and _close(
            float(values[order[j + 1]]), float(values[order[i]]), tie_tol
        ):
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
    values_a = [float(scores_a[node]) for node in common]
    values_b = [float(scores_b[node]) for node in common]
    if not all(math.isfinite(value) for value in values_a + values_b):
        raise ValueError("Spearman scores must be finite.")
    ranks_a = _average_ranks(values_a)
    ranks_b = _average_ranks(values_b)
    n = len(common)
    mean_a = sum(ranks_a) / n
    mean_b = sum(ranks_b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ranks_a, ranks_b))
    var_a = sum((x - mean_a) ** 2 for x in ranks_a)
    var_b = sum((y - mean_b) ** 2 for y in ranks_b)
    if var_a <= 0.0 or var_b <= 0.0:
        return None
    return cov / (var_a * var_b) ** 0.5
