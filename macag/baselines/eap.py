"""Graph path-effect baseline (stable method id: ``eap``, alias ``eap_graph``).

NOT Syed/Nanda attribution patching. This cheap variant sums signed path
effects on the exported attribution-graph link weights (Jacobi / influence-
family). For the paper formula ``(a_corr − a_clean)·∇L`` see
``macag.baselines.eap_syed`` (method id ``eap_syed``).

Naming map: ``macag/docs/baseline_method_map.md``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping, Sequence

from macag.baselines.common import SelectionResult
from macag.graph import NodeId
from macag.utils.metrics import dedupe_preserve_order

LOGGER = logging.getLogger(__name__)

_LOGIT_FEATURE_TYPE = "logit"
# circuit-tracer logit clerps look like: Output " A" (p=0.512)
_LOGIT_TOKEN_RE = re.compile(r'Output\s+"([^"]*)"', re.IGNORECASE)


class EAPUnavailableError(ValueError):
    """Raised when the exported graph cannot represent the requested logit objective."""


def _clerp_logit_token(clerp: str) -> str | None:
    """Extract the quoted logit token from a circuit-tracer clerp string."""
    match = _LOGIT_TOKEN_RE.search(clerp)
    if match is not None:
        return match.group(1)
    # Fallback: first quoted span, if any.
    generic = re.search(r'"([^"]*)"', clerp)
    return generic.group(1) if generic is not None else None


def _token_match_variants(token: str) -> set[str]:
    """Tokenizer-aware equality variants (leading space / strip); case-sensitive."""
    raw = str(token)
    stripped = raw.strip()
    variants = {raw, stripped, raw.lstrip(), f" {stripped}" if stripped else raw}
    return {variant for variant in variants if variant}


def _clerp_matches_token(clerp: str, match: str) -> bool:
    """True when ``match`` equals the logit token in ``clerp`` (not a raw substring).

    Substring matching against the full clerp is unsafe: short tokens like
    ``"A"`` / ``"1"`` hit unrelated words (``cellular``) or probability text
    (``(p=0.010)``). Matching is case-sensitive so ``" D"`` ≠ ``" d"``.
    """
    extracted = _clerp_logit_token(clerp)
    if extracted is None:
        return False
    return bool(_token_match_variants(match) & _token_match_variants(extracted))


def _logit_seed_weights(
    nodes: Sequence[Mapping[str, Any]],
    target_match: str | None,
    foil_match: str | None,
) -> dict[NodeId, float]:
    """Seed weights on logit nodes: +1 target, -1 foil.

    Target selection: `target_match` equals the quoted logit token in `clerp`
    (tokenizer leading-space variants allowed). When omitted, the graph's own
    `is_target_logit` flag is used. Foil uses the same token equality; if no
    exported logit matches the foil (common when the foil token is outside the
    graph's top-k logits), foil seeding is skipped with a warning.
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
            _clerp_matches_token(clerp, target_match)
            if target_match is not None
            else bool(node.get("is_target_logit"))
        )
        is_foil = foil_match is not None and _clerp_matches_token(clerp, foil_match)
        if is_target and is_foil:
            raise EAPUnavailableError(
                f"Logit node {node_id} ({clerp!r}) matches both the target and the foil "
                "pattern; disambiguate --eap-target-match / --eap-foil-match."
            )
        if is_target:
            seeds[node_id] = 1.0
        elif is_foil:
            seeds[node_id] = -1.0

    if not any(weight > 0 for weight in seeds.values()):
        raise EAPUnavailableError(
            "No target logit seed found: no logit node matched "
            f"target_match={target_match!r} and none carries is_target_logit=True."
        )
    if foil_match is not None and not any(weight < 0 for weight in seeds.values()):
        raise EAPUnavailableError(
            "No exported logit node matched "
            f"foil_match={foil_match!r}; target-only seeding would optimize a "
            "different objective than the target-minus-foil evaluation oracle."
        )
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
