"""Ported ACDC node τ-prune (stable method id: ``acdc``, alias ``acdc_ported``).

Keeps Conmy et al. 2023's selection *rule* (top-down; drop if Δmetric < τ) but
applies it to CLT feature nodes under MACAG coalitional ``v`` (zero-ablation),
not native edges with corrupted patching / KL(G‖H). For the native-component
track see ``macag.baselines.acdc_native`` (``acdc_native``).

Naming map: ``macag/docs/baseline_method_map.md``.

Tau units: raw v units (logit-gap points by default), NOT normalized
fractions. The degradation test is ``v(E) - v(E - node) < tau`` on the capped
coalition value, so a tau of 0.05 means 0.05 raw faithfulness points —
calibrate per score_kind (KL-scale v needs much smaller taus). Prefer
``--acdc-target-k`` (bisect to a budget-matched size) over hand-picked taus
when comparing against fixed-budget methods.
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


class ACDCBudgetUnreachableError(ValueError):
    """No τ-pruned set with size ≤ ``target_k`` was found under the search budget."""


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
    cap_sufficiency: bool = True,
    cap_necessity: bool = True,
) -> ACDCPruneResult:
    """Single top-down sweep: drop a node when v(E) - v(E - node) < tau.

    A node whose removal barely hurts — or helps — the coalition value is
    pruned; larger tau prunes more aggressively. One pass in a deterministic
    order (``top_down`` per node layer/ctx metadata, or ``given`` to keep the
    candidate order), mirroring ACDC's single output-to-input traversal.

    ``tau`` is in raw v units (logit-gap points by default), not normalized
    fractions — see the module docstring.
    """
    if order not in ("top_down", "given"):
        raise ValueError("order must be 'top_down' or 'given'.")
    pool = [node for node in dedupe_preserve_order(candidates) if graph.has_node(node)]
    if not pool:
        raise ValueError("ACDC pruning needs a non-empty candidate pool present in the graph.")
    ordered = _topdown_order(graph, pool) if order == "top_down" else list(pool)

    kept = set(pool)
    current_value = coalition_value(
        oracle, target, kept, alpha,
        cap_sufficiency=cap_sufficiency, cap_necessity=cap_necessity,
    )
    removed_order: list[NodeId] = []
    decisions: list[dict[str, Any]] = []

    for node in ordered:
        trial = kept - {node}
        trial_value = coalition_value(
            oracle, target, trial, alpha,
            cap_sufficiency=cap_sufficiency, cap_necessity=cap_necessity,
        )
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
        params={
            "alpha": alpha, "order": order, "tau": tau,
            "cap_sufficiency": cap_sufficiency, "cap_necessity": cap_necessity,
        },
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
    cap_sufficiency: bool = True,
    cap_necessity: bool = True,
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
                cap_sufficiency=cap_sufficiency,
                cap_necessity=cap_necessity,
            )
        )
    return results


def acdc_target_size(
    graph: CircuitGraph,
    oracle: ScoringOracle,
    target: TargetId,
    candidates: Sequence[NodeId],
    target_k: int,
    *,
    alpha: float = 0.5,
    order: str = "top_down",
    max_iters: int = 24,
    tau_lo: float | None = None,
    tau_hi: float | None = None,
    seed_results: Sequence[ACDCPruneResult] | None = None,
    progress: bool = False,
    cap_sufficiency: bool = True,
    cap_necessity: bool = True,
) -> ACDCPruneResult:
    """Find a tau whose pruned set is as close as possible to ``target_k`` **without exceeding it**.

    Kept size is **not** monotone in tau under path-dependent pruning, so this
    does not bisect. It evaluates:
    1. any provided ``seed_results`` (e.g. the fixed tau sweep),
    2. a keep-biased run at tau=0,
    3. every unique degradation observed on those runs (the critical thresholds
       where a prune decision can flip),
    4. optional ``tau_lo`` / ``tau_hi`` endpoints,
    5. up to ``max_iters`` additional midpoints between neighboring evaluated
       taus that still miss ``target_k``,
    6. one aggressive high-tau probe if every evaluated set still exceeds
       ``target_k`` (so the empty / heavily pruned regime is reachable).

    Returns the best result among sets with ``|S| ≤ target_k``, ranked by
    ``(target_k - size, -value)`` — prefer closest under the budget, then higher
    coalition value. Oversized sets are never returned (fair vs hard-budget
    selectors like Game 1). Raises ``ACDCBudgetUnreachableError`` if no
    ≤``target_k`` set is found. ``params["exact"]`` records whether ``|S|``
    hit ``target_k`` exactly.
    """
    if target_k < 1:
        raise ValueError("target_k must be >= 1.")
    pool = [node for node in dedupe_preserve_order(candidates) if graph.has_node(node)]
    if not pool:
        raise ValueError("ACDC pruning needs a non-empty candidate pool present in the graph.")

    def run(tau: float) -> ACDCPruneResult:
        return acdc_prune(
            graph,
            oracle,
            target,
            pool,
            tau=tau,
            alpha=alpha,
            order=order,
            progress=progress,
            cap_sufficiency=cap_sufficiency,
            cap_necessity=cap_necessity,
        )

    def feasible(result: ACDCPruneResult) -> bool:
        return len(result.kept) <= target_k

    def rank(result: ACDCPruneResult) -> tuple[int, float]:
        # Closer to target_k from below wins; ties break on higher coalition value.
        return (target_k - len(result.kept), -result.value)

    evaluated: dict[float, ACDCPruneResult] = {}
    for prior in seed_results or ():
        evaluated.setdefault(float(prior.tau), prior)
    if 0.0 not in evaluated:
        evaluated[0.0] = run(0.0)

    # Critical thresholds are the degradations observed on evaluated paths —
    # pruning decisions flip when tau crosses a node's degradation.
    pending: list[float] = []
    if tau_lo is not None:
        pending.append(float(tau_lo))
    if tau_hi is not None:
        pending.append(float(tau_hi))
    for result in list(evaluated.values()):
        for decision in result.decisions:
            pending.append(float(decision["degradation"]))
            # Slightly above the degradation so the "pruned = degradation < tau"
            # branch flips for that node on a re-run.
            pending.append(float(decision["degradation"]) + 1e-9)

    extra_runs = 0
    while pending and extra_runs < max_iters:
        tau = pending.pop(0)
        if any(abs(tau - seen) <= 1e-15 for seen in evaluated):
            continue
        evaluated[tau] = run(tau)
        extra_runs += 1
        if len(evaluated[tau].kept) == target_k:
            break
        for decision in evaluated[tau].decisions:
            deg = float(decision["degradation"])
            for candidate_tau in (deg, deg + 1e-9):
                if all(abs(candidate_tau - seen) > 1e-15 for seen in evaluated):
                    pending.append(candidate_tau)

    # If still missing exact size, probe midpoints between neighboring taus.
    if all(len(result.kept) != target_k for result in evaluated.values()):
        ordered_taus = sorted(evaluated)
        for left, right in zip(ordered_taus, ordered_taus[1:]):
            if extra_runs >= max_iters:
                break
            if right - left <= 1e-12:
                continue
            mid = (left + right) / 2.0
            if any(abs(mid - seen) <= 1e-15 for seen in evaluated):
                continue
            evaluated[mid] = run(mid)
            extra_runs += 1
            if len(evaluated[mid].kept) == target_k:
                break

    # Guarantee an under-budget candidate when every evaluated set still overshoots.
    if not any(feasible(result) for result in evaluated.values()):
        max_deg = max(
            (
                float(decision["degradation"])
                for result in evaluated.values()
                for decision in result.decisions
            ),
            default=0.0,
        )
        aggressive_tau = max_deg + 1.0
        if all(abs(aggressive_tau - seen) > 1e-15 for seen in evaluated):
            evaluated[aggressive_tau] = run(aggressive_tau)
            extra_runs += 1

    under_budget = [result for result in evaluated.values() if feasible(result)]
    if not under_budget:
        sizes = sorted({len(result.kept) for result in evaluated.values()})
        raise ACDCBudgetUnreachableError(
            f"No ACDC set with size ≤ {target_k} among {len(evaluated)} evaluated "
            f"taus (observed sizes={sizes})."
        )

    best = min(under_budget, key=rank)
    achieved = len(best.kept)
    best.params = dict(
        best.params,
        target_k=target_k,
        achieved_k=achieved,
        bisection_iters=extra_runs,  # kept key name for JSON compatibility
        search_evals=len(evaluated),
        exact=achieved == target_k,
        budget_capped=True,
    )
    if achieved != target_k:
        LOGGER.warning(
            "acdc_target_size: target_k=%d unreachable among %d evaluated taus; "
            "returning nearest under-budget size %d (tau=%.4g)",
            target_k,
            len(evaluated),
            achieved,
            best.tau,
        )
    return best
