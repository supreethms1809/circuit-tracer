"""B2.3 — EAP / attribution-patching node scores derived from graph edges.

The attribution graph's edge weights ARE first-order gradient-times-activation
scores, so a node's EAP importance for the output is its total signed path
effect on the target logit (minus the foil logit when one is given). This is
the "cheap variant" sanctioned by macag.md B2.3: derive the node score directly
from the graph instead of re-running backward passes through the model. Zero
oracle calls — the whole point of the comparison is that MACAG pays real
interventions where EAP pays one local-linear read-off.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from macag.baselines.common import SelectionResult
from macag.graph import NodeId
from macag.utils.metrics import dedupe_preserve_order

LOGGER = logging.getLogger(__name__)

_LOGIT_FEATURE_TYPE = "logit"


def _logit_seed_weights(
    nodes: Sequence[Mapping[str, Any]],
    target_match: str | None,
    foil_match: str | None,
) -> dict[NodeId, float]:
    """Seed weights on logit nodes: +1 target, -1 foil.

    Target selection: `target_match` as a case-insensitive substring of the
    logit node's `clerp`; when omitted, the graph's own `is_target_logit` flag.
    """
    seeds: dict[NodeId, float] = {}
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        if str(node.get("feature_type", "")).strip().lower() != _LOGIT_FEATURE_TYPE:
            continue
        node_id = node.get("node_id", node.get("id"))
        if node_id is None:
            continue
        clerp = str(node.get("clerp", ""))
        is_target = (
            target_match.lower() in clerp.lower()
            if target_match is not None
            else bool(node.get("is_target_logit"))
        )
        is_foil = foil_match is not None and foil_match.lower() in clerp.lower()
        if is_target and is_foil:
            raise ValueError(
                f"Logit node {node_id} ({clerp!r}) matches both the target and the foil "
                "pattern; disambiguate --eap-target-match / --eap-foil-match."
            )
        if is_target:
            seeds[node_id] = 1.0
        elif is_foil:
            seeds[node_id] = -1.0

    if not any(weight > 0 for weight in seeds.values()):
        raise ValueError(
            "No target logit seed found: no logit node matched "
            f"target_match={target_match!r} and none carries is_target_logit=True."
        )
    if foil_match is not None and not any(weight < 0 for weight in seeds.values()):
        raise ValueError(f"No logit node clerp matched foil_match={foil_match!r}.")
    return seeds


def compute_eap_node_scores(
    payload: Mapping[str, Any],
    target_match: str | None = None,
    foil_match: str | None = None,
    tol: float = 1e-12,
    max_sweeps: int | None = None,
) -> tuple[dict[NodeId, float], dict[str, Any]]:
    """Total signed path effect of every node on the seeded logit difference.

    Solves effect(n) = seed(n) + sum_{n->m} weight(n,m) * effect(m) by Jacobi
    sweeps; on a DAG this converges in at most depth+1 sweeps. Returns the
    effect map plus an info dict (seeds, sweeps, converged).
    """
    nodes_raw = payload.get("nodes", [])
    links_raw = payload.get("links", payload.get("edges", []))
    seeds = _logit_seed_weights(nodes_raw, target_match=target_match, foil_match=foil_match)

    node_ids: list[NodeId] = []
    for node in nodes_raw:
        if isinstance(node, Mapping):
            node_id = node.get("node_id", node.get("id"))
            if node_id is not None:
                node_ids.append(node_id)
        else:
            node_ids.append(node)

    out_edges: dict[NodeId, list[tuple[NodeId, float]]] = {}
    for link in links_raw:
        if not isinstance(link, Mapping):
            raise ValueError("EAP scores need weighted links; got a bare edge tuple.")
        source = link.get("source")
        target = link.get("target")
        weight = link.get("weight")
        if weight is None:
            raise ValueError(
                "Graph links carry no 'weight'; the EAP baseline needs the "
                "attribution-weighted graph JSON (circuit-tracer export)."
            )
        out_edges.setdefault(source, []).append((target, float(weight)))

    effects: dict[NodeId, float] = {node: seeds.get(node, 0.0) for node in node_ids}
    sweep_cap = max_sweeps if max_sweeps is not None else max(len(node_ids), 8)
    converged = False
    sweeps = 0
    for sweeps in range(1, sweep_cap + 1):
        max_delta = 0.0
        updated: dict[NodeId, float] = {}
        for node in node_ids:
            value = seeds.get(node, 0.0)
            for downstream, weight in out_edges.get(node, ()):  # absent downstream -> 0
                value += weight * effects.get(downstream, 0.0)
            updated[node] = value
            delta = abs(value - effects[node])
            if delta > max_delta:
                max_delta = delta
        effects = updated
        if max_delta <= tol:
            converged = True
            break
    if not converged:
        LOGGER.warning(
            "EAP propagation did not converge in %d sweeps (cyclic graph?); "
            "scores are the last iterate.",
            sweep_cap,
        )

    info = {
        "seeds": {str(node): weight for node, weight in sorted(seeds.items(), key=lambda kv: str(kv[0]))},
        "sweeps": sweeps,
        "converged": converged,
    }
    return effects, info


def select_top_eap(
    payload: Mapping[str, Any],
    candidates: Sequence[NodeId],
    target_match: str | None = None,
    foil_match: str | None = None,
    use_absolute: bool = True,
) -> SelectionResult:
    """Rank candidates by their EAP node score, best first."""
    effects, info = compute_eap_node_scores(
        payload, target_match=target_match, foil_match=foil_match
    )
    pool = dedupe_preserve_order(candidates)
    scores = {
        node: (abs(effects[node]) if use_absolute else effects[node])
        for node in pool
        if node in effects
    }
    if not scores:
        raise ValueError("No candidate node appears in the graph payload for EAP scoring.")
    dropped = [node for node in pool if node not in effects]
    if dropped:
        LOGGER.warning("%d candidate(s) missing from the graph payload are ranked last.", len(dropped))

    ranking = sorted(scores, key=lambda node: (-scores[node], str(node)))
    ranking.extend(sorted(dropped, key=str))
    return SelectionResult(
        method="eap",
        ranking=ranking,
        scores=scores,
        params={
            "target_match": target_match,
            "foil_match": foil_match,
            "use_absolute": use_absolute,
        },
        extras=info,
    )
