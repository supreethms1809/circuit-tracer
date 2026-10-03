"""Syed / Nanda attribution patching on native output edges (``eap_edge``).

Applies

    Δ_e L ≈ (e_corr − e_clean)ᵀ ∇_{e_clean} L(clean)

to each head ``hook_result`` slice and each ``hook_mlp_out``, then ranks by
``|Δ|`` and keeps the top-k edges. This is the published AtP *formula* on the
native writer→residual edge universe (not CLT features, not graph path-effect).

See ``macag/docs/baseline_original_track.md``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from macag.baselines.original.edges import OutputEdge, build_output_edges
from macag.baselines.original.evaluate import build_corrupt_cache, score_edge_circuit
from macag.baselines.original.metrics import logit_gap_tensor, tokenize_pair
from macag.baselines.original.types import OriginalCircuitResult, unavailable_result

LOGGER = logging.getLogger(__name__)


def _cache_clean_with_grad(
    model: Any,
    clean_tokens: torch.Tensor,
    edges: list[OutputEdge],
    target_idx: int,
    foil_idx: int | None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """Clean activations + grads w.r.t. logit-gap at each edge hook."""
    names = sorted({edge.hook_name for edge in edges})
    saved: dict[str, torch.Tensor] = {}

    def _retain(acts: torch.Tensor, hook: Any) -> torch.Tensor:
        acts.retain_grad()
        saved[hook.name] = acts
        return acts

    hooks = [(name, _retain) for name in names]
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.enable_grad():
            # TransformerLens: hooks(fwd_hooks=...); positional also binds fwd_hooks.
            try:
                hook_cm = model.hooks(fwd_hooks=hooks)
            except TypeError:
                hook_cm = model.hooks(hooks)
            with hook_cm:
                logits = model(clean_tokens)
            metric = logit_gap_tensor(logits, target_idx, foil_idx)
            if not metric.requires_grad:
                raise RuntimeError("eap_edge: logit-gap has no grad_fn on clean forward.")
            model.zero_grad(set_to_none=True)
            for tensor in saved.values():
                if tensor.grad is not None:
                    tensor.grad = None
            metric.backward()
    finally:
        if was_training:
            model.train()

    acts_out: dict[str, torch.Tensor] = {}
    grads_out: dict[str, torch.Tensor] = {}
    for name in names:
        tensor = saved.get(name)
        if tensor is None or tensor.grad is None:
            raise RuntimeError(f"eap_edge: missing activation/grad at {name}")
        acts_out[name] = tensor.detach()
        grads_out[name] = tensor.grad.detach()
    return acts_out, grads_out, logits.detach()


def _edge_atp_score(
    edge: OutputEdge,
    clean_acts: dict[str, torch.Tensor],
    corrupt_acts: dict[str, torch.Tensor],
    grads: dict[str, torch.Tensor],
) -> float:
    clean = clean_acts[edge.hook_name]
    corrupt = corrupt_acts[edge.hook_name]
    grad = grads[edge.hook_name]
    if edge.kind == "head":
        assert edge.head is not None
        delta = corrupt[:, :, edge.head, :] - clean[:, :, edge.head, :]
        g = grad[:, :, edge.head, :]
        return float((delta * g).sum().item())
    delta = corrupt - clean
    return float((delta * grad).sum().item())


def run_eap_edge(
    model: Any,
    *,
    clean_prompt: str,
    corrupted_prompt: str,
    target_token: str,
    foil_token: str | None,
    k: int,
    target_idx: int | None = None,
    foil_idx: int | None = None,
) -> OriginalCircuitResult:
    """Top-k native output edges by |AtP| score; evaluate under corrupt patching."""
    from macag.baselines.original.metrics import resolve_token_index

    if k < 1:
        raise ValueError("k must be >= 1")
    if not getattr(model.cfg, "use_attn_result", False):
        model.cfg.use_attn_result = True
        LOGGER.info("eap_edge: set use_attn_result=True")

    try:
        clean_tokens, corrupt_tokens = tokenize_pair(model, clean_prompt, corrupted_prompt)
    except ValueError as exc:
        return unavailable_result("eap_edge", str(exc), k=k)

    if target_idx is None:
        target_idx = resolve_token_index(model, target_token)
    if foil_token is not None and foil_idx is None:
        foil_idx = resolve_token_index(model, foil_token)

    edges = build_output_edges(model)
    t0 = time.perf_counter()
    clean_acts, grads, full_logits = _cache_clean_with_grad(
        model, clean_tokens, edges, target_idx, foil_idx
    )
    corrupt_cache = build_corrupt_cache(model, corrupt_tokens, edges)
    scores = {
        edge.edge_id: _edge_atp_score(edge, clean_acts, corrupt_cache, grads) for edge in edges
    }
    ranking = sorted(scores, key=lambda eid: abs(scores[eid]), reverse=True)
    kept = ranking[: min(k, len(ranking))]
    circuit_scores = score_edge_circuit(
        model,
        clean_tokens,
        corrupt_cache,
        full_logits,
        all_edges=edges,
        kept_edge_ids=set(kept),
        target_idx=target_idx,
        foil_idx=foil_idx,
    )
    elapsed = time.perf_counter() - t0
    return OriginalCircuitResult(
        method="eap_edge",
        edges_kept=kept,
        size=len(kept),
        scores=circuit_scores,
        selection_stats={
            "model_forwards": 3,
            "model_backwards": 1,
            "wall_s": round(elapsed, 3),
        },
        params={
            "k": k,
            "n_edges": len(edges),
            "estimator": "attribution_patching",
            "edge_universe": "component_output_to_resid",
            "metric_selection": "logit_gap_atp",
            "note": (
                "Syed/Nanda AtP on native head/MLP output edges. "
                "Not CLT eap_syed; not pairwise path edges."
            ),
        },
        ranking=ranking,
        edge_scores={eid: scores[eid] for eid in ranking},
    )
