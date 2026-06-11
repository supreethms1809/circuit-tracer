"""B2.4 — ACDC ported to the CLT node set: top-down threshold pruning.

ACDC (Conmy et al., 2023) prunes a circuit from the output backwards, removing
a component whenever its removal changes the metric by less than a threshold
tau. The port keeps the selection RULE but applies it to the same candidate
node set and the same coalitional v(S) the games use, isolating search
direction (prune-down vs grow-up) from everything else (macag.md §A.1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from macag.baselines.common import coalition_value
from macag.graph import CircuitGraph, NodeId
from macag.scoring import ScoringOracle, TargetId
from macag.utils.metrics import dedupe_preserve_order

LOGGER = logging.getLogger(__name__)


@dataclass
class ACDCPruneResult:
    tau: float
    kept: list[NodeId]
    removed_order: list[NodeId]
    value: float
    decisions: list[dict[str, Any]]
    params: dict[str, Any] = field(default_factory=dict)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _topdown_order(graph: CircuitGraph, candidates: Sequence[NodeId]) -> list[NodeId]:
    """Output-side first: descending layer, then descending ctx_idx, then node string."""

    def key(node: NodeId) -> tuple[int, int, str]:
        metadata = graph.metadata(node) if graph.has_node(node) else {}
        layer = _to_int(metadata.get("layer"), default=-1)
        ctx = _to_int(metadata.get("ctx_idx"), default=-1)
        return (-layer, -ctx, str(node))

    return sorted(candidates, key=key)


def acdc_prune(
    graph: CircuitGraph,
    oracle: ScoringOracle,
    target: TargetId,
    candidates: Sequence[NodeId],
    tau: float,
    alpha: float = 0.5,
    order: str = "top_down",
    progress: bool = False,
) -> ACDCPruneResult:
    """Single top-down sweep: drop a node when v(E) - v(E - node) < tau.

    A node whose removal barely hurts — or helps — the coalition value is
    pruned; larger tau prunes more aggressively. One pass in a deterministic
    order (``top_down`` per node layer/ctx metadata, or ``given`` to keep the
    candidate order), mirroring ACDC's single output-to-input traversal.
    """
    if order not in ("top_down", "given"):
        raise ValueError("order must be 'top_down' or 'given'.")
    pool = [node for node in dedupe_preserve_order(candidates) if graph.has_node(node)]
    if not pool:
        raise ValueError("ACDC pruning needs a non-empty candidate pool present in the graph.")
    ordered = _topdown_order(graph, pool) if order == "top_down" else list(pool)

    kept = set(pool)
    current_value = coalition_value(oracle, target, kept, alpha)
    removed_order: list[NodeId] = []
    decisions: list[dict[str, Any]] = []

    for node in ordered:
        trial = kept - {node}
        trial_value = coalition_value(oracle, target, trial, alpha)
        degradation = current_value - trial_value
        pruned = degradation < tau
        decisions.append(
            {"node": str(node), "degradation": degradation, "pruned": pruned}
        )
        if pruned:
            kept = trial
            current_value = trial_value
            removed_order.append(node)
        if progress:
            LOGGER.info(
                "ACDC tau=%.4g node=%s degradation=%.6f pruned=%s |E|=%d",
                tau,
                node,
                degradation,
                pruned,
                len(kept),
            )

    return ACDCPruneResult(
        tau=tau,
        kept=sorted(kept, key=str),
        removed_order=removed_order,
        value=current_value,
        decisions=decisions,
        params={"alpha": alpha, "order": order, "tau": tau},
    )


def acdc_tau_sweep(
    graph: CircuitGraph,
    oracle: ScoringOracle,
    target: TargetId,
    candidates: Sequence[NodeId],
    taus: Sequence[float],
    alpha: float = 0.5,
    order: str = "top_down",
    progress: bool = False,
) -> list[ACDCPruneResult]:
    """Run the prune at each tau (ascending) to trace a size/faithfulness curve.

    Shares the oracle cache across taus — the full-set and single-removal
    scores repeat — so the sweep costs little more than the largest single run.
    """
    if not taus:
        raise ValueError("Provide at least one tau.")
    results = []
    for tau in sorted(set(float(t) for t in taus)):
        results.append(
            acdc_prune(
                graph,
                oracle,
                target,
                candidates,
                tau=tau,
                alpha=alpha,
                order=order,
                progress=progress,
            )
        )
    return results
