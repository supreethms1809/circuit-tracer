"""Game 1: minimal faithful evidence set via greedy hill-climb."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Sequence

from macag.graph import CircuitGraph, NodeId
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
) -> list[NodeId]:
    """Rank candidates by singleton gain and keep the top-k."""
    if top_k <= 0:
        return []
    if top_k >= len(candidates):
        return list(candidates)

    ranking: list[tuple[float, NodeId]] = []
    for node in candidates:
        if not graph.has_node(node):
            continue
        singleton = {node}
        metrics = compute_faithfulness_metrics(oracle=oracle, target=target, nodes=singleton, alpha=alpha)
        utility = game1_utility(metrics.faithfulness_delta, size=1, lam=lam)
        ranking.append((utility, node))

    ranking.sort(key=lambda item: (-item[0], _sort_key(item[1])))
    return [node for _, node in ranking[:top_k]]


@dataclass
class EvidenceSetResult:
    evidence: set[NodeId]
    induced_subgraph: CircuitGraph
    metrics: FaithfulnessMetrics
    utility: float
    selected_order: list[NodeId]
    iterations: int
    candidate_count: int
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
    prefilter_top_k: int | None = None,
    prefilter_fn: CandidatePrefilter | None = None,
    connected: bool = False,
    min_gain: float = 0.0,
    progress: bool = True,
    log_every: int = 50,
) -> EvidenceSetResult:
    """Greedy hill-climb solver for Game 1."""
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
                graph, oracle, target, candidate_pool, alpha, lam, prefilter_top_k
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
            if connected and len(trial) > 1 and not graph.is_weakly_connected(trial):
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

        selected.add(best_node)
        selected_order.append(best_node)
        iterations += 1
        if progress:
            LOGGER.info("Game1 added node=%s gain=%.6f |E|=%d", best_node, best_gain, len(selected))

        if faithfulness_eps is not None:
            _, metrics = evaluate(selected)
            if (metrics.all_score - metrics.keep_only_score) <= faithfulness_eps:
                if progress:
                    LOGGER.info("Game1 reached faithfulness_eps=%.6f", faithfulness_eps)
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
        sparsity=sparsity(selected_size=len(selected), total_size=max(1, len(candidate_pool))),
    )
