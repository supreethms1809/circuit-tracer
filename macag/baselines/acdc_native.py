"""Native-component ACDC (method id: acdc_native).

Applies the same top-down τ-prune **selection rule** as ``acdc_prune`` (ported
Conmy rule) to TransformerLens native components — attention heads ``a{l}.h{h}``
and MLPs ``m{l}`` — via ``HookedComponentInterventionScorer``.

This is the behavioral / native-granularity track:
- Same search direction as Conmy (prune-down if Δmetric < τ)
- Native universe (heads/MLPs), not CLT features
- Still **node**-level (not ArthurConmy edge-level Algorithm 1)
- Default ablation is zero (component scorer); corrupted patching is a follow-up

See ``macag/docs/baseline_method_map.md``.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from macag.baselines.acdc_prune import ACDCPruneResult, acdc_prune, acdc_target_size, acdc_tau_sweep
from macag.graph import CircuitGraph, NodeId
from macag.scoring import ScoringOracle
from macag.scoring_components import (
    HookedComponentInterventionScorer,
    component_universe,
)
from macag.utils.metrics import dedupe_preserve_order

LOGGER = logging.getLogger(__name__)


def build_native_component_graph(n_layers: int, n_heads: int) -> CircuitGraph:
    """Synthetic graph metadata so ``acdc_prune`` can order by layer."""
    nodes: list[NodeId] = []
    metadata: dict[NodeId, dict[str, Any]] = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            node = f"a{layer}.h{head}"
            nodes.append(node)
            metadata[node] = {
                "feature_type": "attention_head",
                "layer": str(layer),
                "ctx_idx": 0,
                "head": head,
            }
        mlp = f"m{layer}"
        nodes.append(mlp)
        metadata[mlp] = {
            "feature_type": "mlp",
            "layer": str(layer),
            "ctx_idx": 0,
        }
    return CircuitGraph(nodes=nodes, node_metadata=metadata)


def build_native_component_oracle(
    model: Any,
    *,
    prompt: str,
    target_to_logit_idx: Mapping[Any, int],
    score_kind: str = "logit_gap",
    foil_by_target: Mapping[Any, Any] | None = None,
    default_foil: Any | None = None,
    candidates: Sequence[NodeId] | None = None,
) -> tuple[ScoringOracle, CircuitGraph, list[NodeId]]:
    """Oracle + graph + candidate list over the model's native components."""
    cfg = model.cfg
    if not getattr(cfg, "use_attn_result", False):
        # Required by HookedComponentInterventionScorer head hooks.
        model.cfg.use_attn_result = True
        LOGGER.info("acdc_native: set model.cfg.use_attn_result=True for head ablation hooks.")

    n_layers = int(cfg.n_layers)
    n_heads = int(cfg.n_heads)
    universe = component_universe(n_layers, n_heads)
    if candidates is None:
        pool = universe
    else:
        allowed = set(universe)
        pool = [node for node in dedupe_preserve_order(candidates) if node in allowed]
        if not pool:
            raise ValueError(
                "acdc_native candidates have no overlap with component universe "
                f"(n_layers={n_layers}, n_heads={n_heads})."
            )

    scorer = HookedComponentInterventionScorer(
        model=model,
        prompt=prompt,
        target_to_logit_idx=target_to_logit_idx,
        score_kind=score_kind,  # type: ignore[arg-type]
        foil_by_target=foil_by_target,
        default_foil=default_foil,
        node_universe=set(pool),
    )
    oracle = ScoringOracle(backend=scorer, cache_enabled=True)
    graph = build_native_component_graph(n_layers, n_heads)
    return oracle, graph, list(pool)


def run_acdc_native(
    model: Any,
    *,
    prompt: str,
    target: Any,
    target_to_logit_idx: Mapping[Any, int],
    taus: Sequence[float],
    alpha: float = 0.5,
    order: str = "top_down",
    score_kind: str = "logit_gap",
    foil_by_target: Mapping[Any, Any] | None = None,
    default_foil: Any | None = None,
    candidates: Sequence[NodeId] | None = None,
    target_k: int | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """τ-sweep (and optional budget-matched) ACDC on native components."""
    oracle, graph, pool = build_native_component_oracle(
        model,
        prompt=prompt,
        target_to_logit_idx=target_to_logit_idx,
        score_kind=score_kind,
        foil_by_target=foil_by_target,
        default_foil=default_foil,
        candidates=candidates,
    )
    sweep = acdc_tau_sweep(
        graph,
        oracle,
        target,
        pool,
        taus=taus,
        alpha=alpha,
        order=order,
        progress=progress,
    )
    output: dict[str, Any] = {
        "params": {
            "universe": "native_components",
            "n_candidates": len(pool),
            "alpha": alpha,
            "order": order,
            "score_kind": score_kind,
            "note": (
                "Conmy-style top-down τ-prune on heads/MLPs (node-level); "
                "not ArthurConmy edge ACDC."
            ),
        },
        "sweep": [
            {
                "tau": result.tau,
                "size": len(result.kept),
                "kept": sorted(result.kept, key=str),
                "value": result.value,
                "removed_order": [str(node) for node in result.removed_order],
            }
            for result in sweep
        ],
    }

    best_by_size: dict[int, dict[str, Any]] = {}
    for result in sweep:
        size = len(result.kept)
        entry = {
            "evidence": sorted(result.kept, key=str),
            "scores": {"faithfulness": result.value},
            "tau": result.tau,
        }
        prior = best_by_size.get(size)
        if prior is None or float(result.value) > float(prior["scores"]["faithfulness"]):
            best_by_size[size] = entry
    output["best_by_size"] = {str(size): best_by_size[size] for size in sorted(best_by_size)}

    if target_k is not None:
        matched = acdc_target_size(
            graph,
            oracle,
            target,
            pool,
            target_k=target_k,
            alpha=alpha,
            order=order,
            seed_results=sweep,
            progress=progress,
        )
        output["matched_k"] = {
            "target_k": target_k,
            "achieved_k": len(matched.kept),
            "exact": bool(matched.params.get("exact")),
            "tau": matched.tau,
            "value": matched.value,
            "kept": sorted(matched.kept, key=str),
            "evidence": sorted(matched.kept, key=str),
            "params": matched.params,
        }

    stats = oracle.cache_stats()
    output["selection_stats"] = {
        "oracle_calls": stats["oracle_calls"],
        "cache_hits": stats["cache_hits"],
    }
    return output


__all__ = [
    "ACDCPruneResult",
    "build_native_component_graph",
    "build_native_component_oracle",
    "run_acdc_native",
    "acdc_prune",
    "acdc_tau_sweep",
]
