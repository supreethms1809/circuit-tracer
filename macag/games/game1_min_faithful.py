"""Game 1: minimal faithful evidence set via greedy hill-climb."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Sequence

from macag.graph import CircuitGraph, NodeId, grow_connected_frontier
from macag.scoring import ScoringOracle, TargetId
from macag.utils.metrics import (
    FaithfulnessMetrics,
    compute_faithfulness_metrics,
    dedupe_preserve_order,
    game1_utility,
    sparsity,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

LOGGER = logging.getLogger(__name__)

CandidatePrefilter = Callable[
    [CircuitGraph, ScoringOracle, TargetId, Sequence[NodeId], float, float, int],
    list[NodeId],
]


def _sort_key(node: NodeId) -> str:
    return str(node)


def prefilter_candidates(
    graph: CircuitGraph,
    oracle: ScoringOracle,
    target: TargetId,
    candidates: Sequence[NodeId],
    alpha: float,
    lam: float,
    top_k: int,
    connected: bool = False,
) -> list[NodeId]:
    """Rank candidates by singleton gain and keep the top-k.

    The ranking is always computed so the output is deterministic regardless of
    `top_k` (L3). When `connected` is set, the retained pool is grown as a
    connected frontier instead of a raw rank truncation so the connected greedy is
    not handed a disconnected pool (L2).
    """
    if top_k <= 0:
        return []

    ranking: list[tuple[float, NodeId]] = []
    for node in candidates:
        if not graph.has_node(node):
            continue
        singleton = {node}
        metrics = compute_faithfulness_metrics(oracle=oracle, target=target, nodes=singleton, alpha=alpha)
        utility = game1_utility(metrics.faithfulness_delta, size=1, lam=lam)
        ranking.append((utility, node))

    ranking.sort(key=lambda item: (-item[0], _sort_key(item[1])))
    ranked_nodes = [node for _, node in ranking]
    if top_k >= len(ranked_nodes):
        return ranked_nodes
    if connected:
        return grow_connected_frontier(graph, ranked_nodes, top_k)
    return ranked_nodes[:top_k]


@dataclass
class EvidenceSetResult:
    evidence: set[NodeId]
    induced_subgraph: CircuitGraph
    metrics: FaithfulnessMetrics
    utility: float
    selected_order: list[NodeId]
    iterations: int
    candidate_count: int
    total_candidates: int
    params: dict[str, Any]
    oracle_calls: int
    cache_hits: int
    cache_size: int
    sparsity: float


def solve_game1(
    graph: CircuitGraph,
    oracle: ScoringOracle,
    target: TargetId,
    candidates: Sequence[NodeId] | None = None,
    alpha: float = 0.5,
    lam: float = 0.01,
    budget: int | None = None,
    faithfulness_eps: float | None = None,
    stop_metric: str = "normalized",
    prefilter_top_k: int | None = None,
    prefilter_fn: CandidatePrefilter | None = None,
    connected: bool = False,
    min_gain: float = 0.0,
    progress: bool = True,
    log_every: int = 50,
) -> EvidenceSetResult:
    """Greedy hill-climb solver for Game 1.

    `stop_metric` controls how `faithfulness_eps` is interpreted:
      * "normalized" (default): stop when faithfulness_delta_normalized >= 1 - eps.
        This divides by recoverable_range (all - empty), which is the correct
        error-floor-aware target with frozen attention but goes DEGENERATE when
        the range collapses toward zero/negative (e.g. unfrozen attention),
        producing spurious early/late stops.
      * "raw_relative": denominator-free diminishing-returns stop. Stop before
        adding a node whose marginal raw faithfulness gain is < eps * (the first
        feature's gain). Stable regardless of the error floor, so it is the
        correct choice when recoverable_range is unreliable (unfrozen attention).
    """
    if stop_metric not in ("normalized", "raw_relative"):
        raise ValueError("stop_metric must be 'normalized' or 'raw_relative'.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    if lam < 0.0:
        raise ValueError("lam must be non-negative.")
    if budget is not None and budget < 0:
        raise ValueError("budget must be non-negative when provided.")
    if min_gain < 0.0:
        raise ValueError("min_gain must be non-negative.")

    candidate_pool = dedupe_preserve_order(candidates if candidates is not None else graph.nodes())
    candidate_pool = [node for node in candidate_pool if graph.has_node(node)]
    # Full candidate count before any prefilter, so reported sparsity reflects the
    # true graph rather than the (possibly much smaller) prefiltered pool (I4).
    total_candidates = len(candidate_pool)
    if progress:
        LOGGER.info(
            "Game1 start: candidates=%d budget=%s alpha=%.3f lambda=%.4f",
            len(candidate_pool),
            budget,
            alpha,
            lam,
        )

    if prefilter_top_k is not None:
        if prefilter_fn:
            candidate_pool = prefilter_fn(
                graph, oracle, target, candidate_pool, alpha, lam, prefilter_top_k
            )
        else:
            candidate_pool = prefilter_candidates(
                graph, oracle, target, candidate_pool, alpha, lam, prefilter_top_k, connected
            )

    utility_cache: dict[frozenset[NodeId], float] = {}
    metric_cache: dict[frozenset[NodeId], FaithfulnessMetrics] = {}

    def evaluate(nodes: set[NodeId]) -> tuple[float, FaithfulnessMetrics]:
        key = frozenset(nodes)
        if key in utility_cache:
            return utility_cache[key], metric_cache[key]
        metrics = compute_faithfulness_metrics(oracle=oracle, target=target, nodes=nodes, alpha=alpha)
        utility = game1_utility(faithfulness_delta=metrics.faithfulness_delta, size=len(nodes), lam=lam)
        utility_cache[key] = utility
        metric_cache[key] = metrics
        return utility, metrics

    selected: set[NodeId] = set()
    selected_order: list[NodeId] = []
    iterations = 0
    first_gain: float | None = None  # raw marginal gain of the first added node

    while True:
        if budget is not None and len(selected) >= budget:
            break

        current_utility, _ = evaluate(selected)
        best_node: NodeId | None = None
        best_gain = min_gain

        iterator: Sequence[NodeId] | Any = candidate_pool
        if progress and tqdm is not None:
            iterator = tqdm(
                candidate_pool,
                desc=f"Game1 sweep |E|={len(selected)}",
                leave=False,
            )

        for idx, node in enumerate(iterator, start=1):
            if node in selected:
                continue
            trial = set(selected)
            trial.add(node)
            if connected and len(trial) > 1 and not graph.connected_through(trial):
                continue
            trial_utility, _ = evaluate(trial)
            gain = trial_utility - current_utility
            if gain > best_gain:
                best_gain = gain
                best_node = node
            elif gain == best_gain and best_node is not None and _sort_key(node) < _sort_key(best_node):
                best_node = node
            if progress and tqdm is None and log_every > 0 and idx % log_every == 0:
                LOGGER.info("Game1 evaluated %d/%d candidates", idx, len(candidate_pool))

        if best_node is None:
            if progress:
                LOGGER.info("Game1 no improving candidate found (|E|=%d)", len(selected))
            break

        # Denominator-free diminishing-returns stop (before adding): the best
        # available feature contributes less than `eps` of the top feature's
        # raw marginal gain. Stable when recoverable_range is unreliable.
        if (
            faithfulness_eps is not None
            and stop_metric == "raw_relative"
            and first_gain is not None
            and first_gain > 0.0
            and best_gain < faithfulness_eps * first_gain
        ):
            if progress:
                LOGGER.info(
                    "Game1 raw_relative stop: best_gain=%.6f < eps*first_gain=%.6f (|E|=%d)",
                    best_gain,
                    faithfulness_eps * first_gain,
                    len(selected),
                )
            break

        selected.add(best_node)
        selected_order.append(best_node)
        iterations += 1
        if first_gain is None:
            first_gain = best_gain
        if progress:
            LOGGER.info("Game1 added node=%s gain=%.6f |E|=%d", best_node, best_gain, len(selected))

        if faithfulness_eps is not None and stop_metric == "normalized":
            _, metrics = evaluate(selected)
            # Stop on the SAME alpha-mixed objective the solver optimizes, in its
            # normalized (error-floor-aware) form (C2). faithfulness_delta_normalized
            # is in [0, 1]; reaching >= 1 - eps means the evidence recovers all but
            # `eps` of the achievable faithfulness. The previous condition tested only
            # the raw sufficiency gap, which is inconsistent when alpha != 1.
            if metrics.faithfulness_delta_normalized >= 1.0 - faithfulness_eps:
                if progress:
                    LOGGER.info(
                        "Game1 reached faithfulness_eps=%.6f (normalized delta=%.6f)",
                        faithfulness_eps,
                        metrics.faithfulness_delta_normalized,
                    )
                break

    final_utility, final_metrics = evaluate(selected)
    stats = oracle.cache_stats()
    if progress:
        LOGGER.info(
            "Game1 finished: iterations=%d oracle_calls=%d cache_hits=%d",
            iterations,
            stats["oracle_calls"],
            stats["cache_hits"],
        )
    return EvidenceSetResult(
        evidence=selected,
        induced_subgraph=graph.subgraph(selected),
        metrics=final_metrics,
        utility=final_utility,
        selected_order=selected_order,
        iterations=iterations,
        candidate_count=len(candidate_pool),
        total_candidates=total_candidates,
        params={
            "alpha": alpha,
            "lambda": lam,
            "budget": budget,
            "faithfulness_eps": faithfulness_eps,
            "prefilter_top_k": prefilter_top_k,
            "connected": connected,
            "min_gain": min_gain,
        },
        oracle_calls=stats["oracle_calls"],
        cache_hits=stats["cache_hits"],
        cache_size=stats["cache_size"],
        sparsity=sparsity(selected_size=len(selected), total_size=max(1, total_candidates)),
    )
