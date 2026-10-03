"""True Conmy et al. ACDC via UFO-101 ``auto-circuit``.

Wraps ``auto_circuit.prune_algos.ACDC.acdc_prune_scores`` — the maintained
implementation of Algorithm 1 (ArthurConmy's repo is frozen and points here).

Operates on the factorized residual-stream edge graph with corrupted
(resample) patching and KL faithfulness. Requires ``pip install auto-circuit``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

from macag.baselines.original.autocircuit_bridge import (
    build_prompt_loader,
    circuit_logits_for_edges,
    full_model_logits,
    make_patchable_model,
    score_circuit_logits,
    top_edges_from_prune_scores,
)
from macag.baselines.original.tl_compat import enable_autocircuit_tl3_compat
from macag.baselines.original.types import OriginalCircuitResult, unavailable_result

LOGGER = logging.getLogger(__name__)


def run_acdc_edge(
    model: Any,
    *,
    clean_prompt: str,
    corrupted_prompt: str,
    target_token: str,
    foil_token: str | None,
    taus: Sequence[float] | None = None,
    metric: str = "kl",
    target_k: int | None = None,
    target_idx: int | None = None,
    foil_idx: int | None = None,
) -> dict[str, Any]:
    """Run AutoCircuit ACDC; optionally take a hard budget-matched top-k circuit.

    ``taus`` is accepted for CLI compatibility. AutoCircuit sweeps an internal
    tao grid (``tao_bases`` × ``tao_exps``); when ``taus`` is provided we map it
    onto ``tao_bases``/``tao_exps`` approximately by using those float values
    directly as the tao list via a thin wrapper around ``acdc_prune_scores``.
    """
    enable_autocircuit_tl3_compat()
    try:
        from auto_circuit.prune_algos.ACDC import acdc_prune_scores
    except ImportError as exc:
        return unavailable_result(
            "acdc_edge",
            f"auto-circuit not importable ({exc}). Install with: pip install auto-circuit",
        ).to_dict()

    from macag.baselines.original.metrics import resolve_token_index

    if metric not in ("kl", "logit_gap", "kl_div", "mse"):
        raise ValueError("metric must be 'kl'/'kl_div' or 'mse' (AutoCircuit ACDC targets)")
    faith_target = "mse" if metric == "mse" else "kl_div"

    if target_idx is None:
        target_idx = resolve_token_index(model, target_token)
    if foil_token is not None and foil_idx is None:
        foil_idx = resolve_token_index(model, foil_token)

    try:
        loader, _seq_len = build_prompt_loader(
            model,
            clean_prompt=clean_prompt,
            corrupted_prompt=corrupted_prompt,
            target_idx=target_idx,
            foil_idx=foil_idx,
        )
    except ValueError as exc:
        return unavailable_result("acdc_edge", str(exc)).to_dict()

    t0 = time.perf_counter()
    patch_model = make_patchable_model(model)

    # AutoCircuit's public API takes tao_exps/tao_bases. For an explicit tau list,
    # run with a custom grid by patching through the same function kwargs.
    if taus:
        # Represent each tau as base*10**exp ≈ tau by using tao_bases=[tau] and
        # tao_exps=[0] after scaling: pass bases as the raw thresholds via
        # tao_bases=list(taus), tao_exps=[0].
        prune_scores = acdc_prune_scores(
            patch_model,
            loader,
            None,
            tao_exps=[0],
            tao_bases=[float(t) for t in taus],
            faithfulness_target=faith_target,  # type: ignore[arg-type]
        )
        tau_list = [float(t) for t in taus]
    else:
        prune_scores = acdc_prune_scores(
            patch_model,
            loader,
            None,
            faithfulness_target=faith_target,  # type: ignore[arg-type]
        )
        tau_list = []

    # Headline circuit: budget-matched top-k if requested, else all unpruned (score=inf).
    if target_k is not None:
        if target_k < 1:
            raise ValueError("target_k must be >= 1")
        k = min(int(target_k), len(patch_model.edges))
        kept = top_edges_from_prune_scores(patch_model, prune_scores, k)
        exact = len(kept) == int(target_k)
        matched_k: dict[str, Any] = {
            "status": "ok",
            "target_k": int(target_k),
            "achieved_k": len(kept),
            "exact": exact,
            "budget_capped": True,
        }
    else:
        # Edges that survived all τ (score still +inf).
        kept = []
        for edge in patch_model.edges:
            score = float(edge.prune_score(prune_scores).item())
            if score == float("inf"):
                kept.append(str(edge))
        kept = sorted(kept)
        matched_k = {
            "status": "ok",
            "target_k": None,
            "achieved_k": len(kept),
            "exact": True,
            "budget_capped": False,
            "note": "All edges with prune_score=+inf after ACDC τ sweep",
        }
        k = max(len(kept), 1)

    clean_tokens = model.to_tokens(clean_prompt)
    full_logits = full_model_logits(patch_model, clean_tokens)
    if not kept:
        return unavailable_result(
            "acdc_edge",
            "ACDC left no edges above threshold; cannot score an empty circuit",
        ).to_dict()
    # Exact edge-set eval: ACDC scores are tied at +inf/τ, so auto-circuit's
    # threshold keep would expand to the full graph and report KL≈0.
    circuit_logits, realized_k = circuit_logits_for_edges(patch_model, loader, kept)
    scores = score_circuit_logits(
        full_logits, circuit_logits, target_idx=target_idx, foil_idx=foil_idx
    )
    scores["realized_edge_count"] = realized_k
    elapsed = time.perf_counter() - t0

    result = OriginalCircuitResult(
        method="acdc_edge",
        edges_kept=kept,
        size=len(kept),
        scores=scores,
        selection_stats={
            "model_forwards": None,  # AutoCircuit ACDC is O(|E|·|τ|); wall_s is authoritative
            "model_backwards": 0,
            "wall_s": round(elapsed, 3),
            "n_graph_edges": len(patch_model.edges),
        },
        params={
            "taus": tau_list,
            "metric": faith_target,
            "n_edges": len(patch_model.edges),
            "edge_universe": "autocircuit_factorized_qkv",
            "ablation": "resample_corrupt",
            "implementation": "auto_circuit.prune_algos.ACDC.acdc_prune_scores",
            "note": (
                "True Conmy et al. 2023 ACDC via UFO-101 auto-circuit "
                "(ArthurConmy repo is frozen and redirects here)."
            ),
        },
    )
    payload = result.to_dict()
    payload["matched_k"] = matched_k | {
        "edges_kept": kept,
        "scores": scores,
    }
    # Optional size curve via a few k checkpoints for analysis.
    sweep = []
    for size in sorted({1, 4, 8, 16, 32, len(kept)}):
        if size < 1 or size > len(patch_model.edges):
            continue
        edges_k = top_edges_from_prune_scores(patch_model, prune_scores, size)
        logits_k, realized_k = circuit_logits_for_edges(patch_model, loader, edges_k)
        sc = score_circuit_logits(
            full_logits, logits_k, target_idx=target_idx, foil_idx=foil_idx
        )
        sc["realized_edge_count"] = realized_k
        sweep.append({"size": size, "edges_kept": edges_k, "scores": sc})
    payload["sweep"] = sweep
    payload["best_by_size"] = {
        str(entry["size"]): {
            "edges_kept": entry["edges_kept"],
            "scores": entry["scores"],
        }
        for entry in sweep
    }
    return payload
