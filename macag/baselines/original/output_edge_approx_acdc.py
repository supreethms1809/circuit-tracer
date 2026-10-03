"""Original-style ACDC on native output edges (``acdc_edge``).

Top-down τ-prune (Conmy et al. 2023 Algorithm 1 *selection rule*) on the
component output-edge graph, with **corrupted activation patching** (not zero
ablation) and a scalar task metric (default logit-gap; KL available).

This is an in-repo reimplementation of the published *rule* on the output-edge
universe. It is **not** a wrap of the ArthurConmy repo and **not** pairwise
path-edge ACDC. Do not Jaccard against CLT Game 1.

See ``macag/docs/baseline_original_track.md``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import torch

from macag.baselines.original.edges import OutputEdge, build_output_edges, topdown_order
from macag.baselines.original.evaluate import (
    build_corrupt_cache,
    run_with_patched_edges,
    score_edge_circuit,
)
from macag.baselines.original.metrics import (
    kl_full_vs_circuit,
    logit_gap_from_logits,
    resolve_token_index,
    tokenize_pair,
)
from macag.baselines.original.types import OriginalCircuitResult, unavailable_result

LOGGER = logging.getLogger(__name__)


def _metric_value(
    logits: torch.Tensor,
    *,
    metric: str,
    full_logits: torch.Tensor,
    target_idx: int,
    foil_idx: int | None,
) -> float:
    if metric == "logit_gap":
        return logit_gap_from_logits(logits, target_idx, foil_idx)
    if metric == "kl":
        # Larger KL = worse circuit; degradation = metric(current) - metric(trial)
        # uses this so pruning when Δ_kl is small (removing edge barely changes KL).
        return kl_full_vs_circuit(full_logits, logits)
    raise ValueError(f"Unknown ACDC metric {metric!r}")


def acdc_edge_prune(
    model: Any,
    clean_tokens: torch.Tensor,
    corrupt_cache: dict[str, torch.Tensor],
    full_logits: torch.Tensor,
    edges: Sequence[OutputEdge],
    *,
    tau: float,
    metric: str,
    target_idx: int,
    foil_idx: int | None,
) -> tuple[list[str], list[dict[str, Any]], int]:
    """Single top-down pass; drop edge when metric degradation < tau.

    For ``logit_gap``, degradation = current - trial (drop if removing barely
    hurts). For ``kl``, degradation = trial - current (drop if removing barely
    increases KL).
    """
    ordered = topdown_order(edges)
    kept = {edge.edge_id for edge in edges}
    forwards = 0

    current_logits = run_with_patched_edges(
        model,
        clean_tokens,
        corrupt_cache,
        all_edges=edges,
        kept_edge_ids=kept,
    )
    forwards += 1
    current = _metric_value(
        current_logits,
        metric=metric,
        full_logits=full_logits,
        target_idx=target_idx,
        foil_idx=foil_idx,
    )

    decisions: list[dict[str, Any]] = []
    for edge in ordered:
        trial_kept = kept - {edge.edge_id}
        trial_logits = run_with_patched_edges(
            model,
            clean_tokens,
            corrupt_cache,
            all_edges=edges,
            kept_edge_ids=trial_kept,
        )
        forwards += 1
        trial = _metric_value(
            trial_logits,
            metric=metric,
            full_logits=full_logits,
            target_idx=target_idx,
            foil_idx=foil_idx,
        )
        if metric == "logit_gap":
            degradation = current - trial
        else:
            degradation = trial - current
        pruned = degradation < tau
        decisions.append(
            {
                "edge": edge.edge_id,
                "degradation": degradation,
                "pruned": pruned,
            }
        )
        if pruned:
            kept = trial_kept
            current = trial
    return sorted(kept), decisions, forwards


def run_acdc_edge(
    model: Any,
    *,
    clean_prompt: str,
    corrupted_prompt: str,
    target_token: str,
    foil_token: str | None,
    taus: Sequence[float],
    metric: str = "logit_gap",
    target_k: int | None = None,
    target_idx: int | None = None,
    foil_idx: int | None = None,
) -> dict[str, Any]:
    """τ-sweep (+ optional hard budget-matched) original-style edge ACDC.

    Returns a method block with ``sweep``, ``best_by_size``, and optional
    ``matched_k`` (size ≤ target_k only).
    """
    if metric not in ("logit_gap", "kl"):
        raise ValueError("metric must be 'logit_gap' or 'kl'")
    if not getattr(model.cfg, "use_attn_result", False):
        model.cfg.use_attn_result = True
        LOGGER.info("acdc_edge: set use_attn_result=True")

    try:
        clean_tokens, corrupt_tokens = tokenize_pair(model, clean_prompt, corrupted_prompt)
    except ValueError as exc:
        return unavailable_result("acdc_edge", str(exc)).to_dict()

    if target_idx is None:
        target_idx = resolve_token_index(model, target_token)
    if foil_token is not None and foil_idx is None:
        foil_idx = resolve_token_index(model, foil_token)
    if metric == "logit_gap" and foil_idx is None:
        return unavailable_result(
            "acdc_edge", "logit_gap metric requires a foil token/index"
        ).to_dict()

    edges = build_output_edges(model)
    t0 = time.perf_counter()
    with torch.inference_mode():
        full_logits = model(clean_tokens)
    corrupt_cache = build_corrupt_cache(model, corrupt_tokens, edges)

    sweep: list[dict[str, Any]] = []
    total_forwards = 1  # full clean
    for tau in sorted({float(t) for t in taus}):
        kept, decisions, forwards = acdc_edge_prune(
            model,
            clean_tokens,
            corrupt_cache,
            full_logits,
            edges,
            tau=tau,
            metric=metric,
            target_idx=target_idx,
            foil_idx=foil_idx,
        )
        total_forwards += forwards
        scores = score_edge_circuit(
            model,
            clean_tokens,
            corrupt_cache,
            full_logits,
            all_edges=edges,
            kept_edge_ids=set(kept),
            target_idx=target_idx,
            foil_idx=foil_idx,
        )
        total_forwards += 1
        sweep.append(
            {
                "tau": tau,
                "size": len(kept),
                "edges_kept": kept,
                "scores": scores,
                "n_decisions": len(decisions),
            }
        )

    best_by_size: dict[str, dict[str, Any]] = {}
    for entry in sweep:
        size = str(entry["size"])
        prior = best_by_size.get(size)
        # Prefer higher logit_gap; for KL prefer lower.
        if prior is None:
            best_by_size[size] = {
                "edges_kept": entry["edges_kept"],
                "scores": entry["scores"],
                "tau": entry["tau"],
            }
        else:
            if metric == "logit_gap":
                better = entry["scores"]["logit_gap"] > prior["scores"]["logit_gap"]
            else:
                better = entry["scores"]["kl"] < prior["scores"]["kl"]
            if better:
                best_by_size[size] = {
                    "edges_kept": entry["edges_kept"],
                    "scores": entry["scores"],
                    "tau": entry["tau"],
                }

    matched_k = None
    if target_k is not None:
        if target_k < 1:
            raise ValueError("target_k must be >= 1")
        # Prefer exact, else nearest under budget by |size-k| then metric.
        candidates = []
        for entry in sweep:
            size = int(entry["size"])
            if size <= target_k:
                candidates.append(entry)
        if not candidates:
            matched_k = {
                "status": "unavailable",
                "target_k": target_k,
                "achieved_k": None,
                "exact": False,
                "budget_capped": True,
                "reason": f"No acdc_edge set with size ≤ {target_k} in τ sweep.",
            }
        else:

            def rank(entry: dict[str, Any]) -> tuple[int, float]:
                size = int(entry["size"])
                if metric == "logit_gap":
                    return (target_k - size, -float(entry["scores"]["logit_gap"]))
                return (target_k - size, float(entry["scores"]["kl"]))

            best = min(candidates, key=rank)
            matched_k = {
                "status": "ok",
                "target_k": target_k,
                "achieved_k": int(best["size"]),
                "exact": int(best["size"]) == target_k,
                "budget_capped": True,
                "tau": best["tau"],
                "edges_kept": best["edges_kept"],
                "scores": best["scores"],
            }

    # Headline result: matched_k if present else best logit_gap / lowest KL in sweep.
    if matched_k and matched_k.get("status") == "ok":
        headline_edges = list(matched_k["edges_kept"])
        headline_scores = dict(matched_k["scores"])
        headline_size = int(matched_k["achieved_k"])
    elif sweep:
        if metric == "logit_gap":
            best = max(sweep, key=lambda e: e["scores"]["logit_gap"])
        else:
            best = min(sweep, key=lambda e: e["scores"]["kl"])
        headline_edges = list(best["edges_kept"])
        headline_scores = dict(best["scores"])
        headline_size = int(best["size"])
    else:
        return unavailable_result("acdc_edge", "empty tau sweep").to_dict()

    elapsed = time.perf_counter() - t0
    result = OriginalCircuitResult(
        method="acdc_edge",
        edges_kept=headline_edges,
        size=headline_size,
        scores=headline_scores,
        selection_stats={
            "model_forwards": total_forwards,
            "model_backwards": 0,
            "wall_s": round(elapsed, 3),
        },
        params={
            "taus": [float(t) for t in taus],
            "metric": metric,
            "n_edges": len(edges),
            "edge_universe": "component_output_to_resid",
            "ablation": "corrupted_patch",
            "note": (
                "Conmy top-down τ-prune on native output edges with corrupted "
                "patching. In-repo reimplementation — not ArthurConmy edge DAG."
            ),
        },
    )
    payload = result.to_dict()
    payload["sweep"] = sweep
    payload["best_by_size"] = best_by_size
    if matched_k is not None:
        payload["matched_k"] = matched_k
    return payload
