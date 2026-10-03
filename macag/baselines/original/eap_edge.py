"""True Syed / Nanda Edge Attribution Patching via UFO-101 ``auto-circuit``.

Wraps
``auto_circuit.prune_algos.edge_attribution_patching.edge_attribution_patching_prune_scores``,
which AutoCircuit documents as an exact replication of Syed et al. 2023 / their
repo implementation, on the **factorized residual-stream edge graph**
(including Q/K/V destinations).

Requires: ``pip install auto-circuit`` and TransformerLens ≥ 2 (TL 3 supported
via :mod:`macag.baselines.original.tl_compat`).
"""

from __future__ import annotations

import logging
import time
from typing import Any

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

# Syed-style recovery curve sizes (exact edge sets, not threshold bands).
_DEFAULT_SWEEP_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def _eap_sweep_sizes(n_edges: int, extra: int | None = None) -> list[int]:
    sizes = {s for s in _DEFAULT_SWEEP_SIZES if 1 <= s <= n_edges}
    sizes.add(n_edges)
    if extra is not None and 1 <= extra <= n_edges:
        sizes.add(int(extra))
    return sorted(sizes)


def run_eap_edge(
    model: Any,
    *,
    clean_prompt: str,
    corrupted_prompt: str,
    target_token: str,
    foil_token: str | None,
    k: int | None = None,
    target_idx: int | None = None,
    foil_idx: int | None = None,
) -> dict[str, Any]:
    """Syed AtP ranking + TREE_PATCH evaluation.

    ``k is None`` or ``k <= 0``: native mode — full ranking + faithfulness curve
    (no fixed budget). Headline circuit = all edges ranked by |AtP| (trivial
    upper bound); use ``sweep`` for the paper-style recovery curve.

    ``k >= 1``: also report an exact top-``k`` circuit (budget-matched mode).
    """
    enable_autocircuit_tl3_compat()
    try:
        from auto_circuit.prune_algos.edge_attribution_patching import (
            edge_attribution_patching_prune_scores,
        )
    except ImportError as exc:
        return unavailable_result(
            "eap_edge",
            f"auto-circuit not importable ({exc}). Install with: pip install auto-circuit",
            k=k,
        ).to_dict()

    from macag.baselines.original.metrics import resolve_token_index

    budget_capped = k is not None and int(k) >= 1
    if budget_capped and int(k) < 1:
        raise ValueError("k must be >= 1 when budget-capping")

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
        return unavailable_result("eap_edge", str(exc), k=k).to_dict()

    t0 = time.perf_counter()
    patch_model = make_patchable_model(model)
    prune_scores = edge_attribution_patching_prune_scores(
        patch_model, loader, None, answer_diff=foil_idx is not None
    )
    n_edges = len(patch_model.edges)
    ranked = sorted(
        (
            (float(edge.prune_score(prune_scores).abs().item()), str(edge))
            for edge in patch_model.edges
        ),
        key=lambda item: (-item[0], item[1]),
    )
    ranking = [name for _, name in ranked]
    edge_scores = {name: score for score, name in ranked}

    clean_tokens = model.to_tokens(clean_prompt)
    full_logits = full_model_logits(patch_model, clean_tokens)

    # Headline: budget top-k if requested; else no single circuit (see sweep).
    if budget_capped:
        kept = top_edges_from_prune_scores(patch_model, prune_scores, int(k))
        mode = "budget_topk"
        circuit_logits, realized_k = circuit_logits_for_edges(patch_model, loader, kept)
        scores = score_circuit_logits(
            full_logits, circuit_logits, target_idx=target_idx, foil_idx=foil_idx
        )
        scores["realized_edge_count"] = realized_k
    else:
        kept = []
        mode = "native_ranking"
        scores = {
            "logit_gap": None,
            "kl": None,
            "note": "Native Syed AtP has no fixed circuit size; see sweep.",
        }

    sweep: list[dict[str, Any]] = []
    for size in _eap_sweep_sizes(n_edges, extra=int(k) if budget_capped else None):
        edges_k = ranking[:size]
        logits_k, realized = circuit_logits_for_edges(patch_model, loader, edges_k)
        sc = score_circuit_logits(
            full_logits, logits_k, target_idx=target_idx, foil_idx=foil_idx
        )
        sc["realized_edge_count"] = realized
        sweep.append({"size": size, "edges_kept": edges_k, "scores": sc})

    elapsed = time.perf_counter() - t0
    result = OriginalCircuitResult(
        method="eap_edge",
        edges_kept=kept,
        size=len(kept),
        scores=scores,
        selection_stats={
            "model_forwards": (1 if budget_capped else 0) + len(sweep),
            "model_backwards": 1,
            "wall_s": round(elapsed, 3),
            "n_graph_edges": n_edges,
        },
        params={
            "k": int(k) if budget_capped else None,
            "mode": mode,
            "budget_capped": budget_capped,
            "n_edges": n_edges,
            "estimator": "syed_edge_attribution_patching",
            "edge_universe": "autocircuit_factorized_qkv",
            "implementation": "auto_circuit.prune_algos.edge_attribution_patching",
            "note": (
                "True Syed et al. 2023 Edge Attribution Patching via UFO-101 "
                "auto-circuit (factorized residual edges including Q/K/V). "
                "Native mode reports the |AtP| ranking + size→faithfulness sweep."
            ),
        },
        ranking=ranking,
        # Full score map is huge on Gemma/Llama; keep top-4k for inspection.
        edge_scores={name: edge_scores[name] for name in ranking[: min(4096, n_edges)]},
    )
    payload = result.to_dict()
    payload["sweep"] = sweep
    payload["best_by_size"] = {
        str(entry["size"]): {
            "edges_kept": entry["edges_kept"],
            "scores": entry["scores"],
        }
        for entry in sweep
    }
    return payload
