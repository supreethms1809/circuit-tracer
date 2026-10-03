"""Syed / Nanda attribution patching on CLT feature nodes (method id: eap_syed).

Implements the scoring formula from Syed et al. 2023 (Edge Attribution Patching),
which uses Nanda's attribution-patching first-order approximation:

    Δ_e L ≈ (e_corr − e_clean)ᵀ ∇_{e_clean} L(clean)

applied to each MACAG candidate **feature node** (layer, position, feature_idx),
then ranking by |Δ|. This is the apples-to-apples “real EAP” on the same
candidate universe as Game 1.

It is **not** native-edge EAP over attention/MLP edges (Syed’s original graph),
and it is **not** ``macag.baselines.eap`` (graph path-effect). See
``macag/docs/baseline_method_map.md``.

Gradient path
-------------
``ReplacementModel.feature_intervention`` is decorated with ``@torch.no_grad`` and
routes deltas through a sparse ``nonzero`` gather, so leaf-Parameter interventions
cannot backprop into L. On a real ReplacementModel we therefore estimate
``∂L/∂a`` by contracting each feature's decoder write direction(s) with the
residual-stream gradient at ``feature_output_hook`` (the same linearized residual
path AttributionContext uses). A leaf-Parameter + ``feature_intervention`` path
is kept only for lightweight fakes in unit tests.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Mapping, Sequence

import torch

from macag.baselines.common import SelectionResult
from macag.graph import NodeId
from macag.utils.metrics import dedupe_preserve_order

LOGGER = logging.getLogger(__name__)


def _as_intervention_triple(
    spec: tuple[Any, ...] | Sequence[Any],
) -> tuple[int, int, int]:
    if len(spec) < 3:
        raise ValueError(f"Intervention spec needs (layer, pos, feat); got {spec!r}.")
    layer, pos, feat = int(spec[0]), int(spec[1]), int(spec[2])
    return layer, pos, feat


def _logit_gap(logits: torch.Tensor, target_idx: int, foil_idx: int | None) -> torch.Tensor:
    """Scalar metric L on the final position (batch size 1)."""
    if logits.ndim == 3:
        last = logits[0, -1]
    elif logits.ndim == 2:
        last = logits[-1]
    else:
        raise ValueError(f"Unexpected logits shape {tuple(logits.shape)}.")
    if foil_idx is None:
        return last[target_idx]
    return last[target_idx] - last[foil_idx]


def _supports_decoder_atp(model: Any) -> bool:
    """True for ReplacementModel-like objects with decoder vectors + hook API."""
    return (
        hasattr(model, "transcoders")
        and hasattr(model, "cfg")
        and hasattr(model, "feature_output_hook")
        and hasattr(model, "hooks")
        and callable(getattr(getattr(model, "transcoders", None), "_get_decoder_vectors", None))
    )


def _residual_grads_at_feature_output(
    model: Any,
    tokens: torch.Tensor,
    target_logit_idx: int,
    foil_logit_idx: int | None,
) -> torch.Tensor:
    """``∂L/∂r`` at each layer's feature-output hook; shape ``(n_layers, n_pos, d_model)``."""
    n_layers = int(model.cfg.n_layers)
    saved: dict[int, torch.Tensor] = {}

    def _save(acts: torch.Tensor, hook: Any, *, layer: int) -> torch.Tensor:
        acts.retain_grad()
        saved[layer] = acts
        return acts

    hooks = [
        (f"blocks.{layer}.{model.feature_output_hook}", partial(_save, layer=layer))
        for layer in range(n_layers)
    ]

    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.enable_grad():
            with model.hooks(hooks):  # type: ignore[attr-defined]
                logits = model(tokens)
            metric = _logit_gap(logits, target_logit_idx, foil_logit_idx)
            if not metric.requires_grad:
                raise RuntimeError(
                    "eap_syed: logit-gap metric has no grad_fn. ReplacementModel "
                    "should enable grads on embeddings via hook_embed; check model setup."
                )
            if hasattr(model, "zero_grad"):
                model.zero_grad(set_to_none=True)
            for tensor in saved.values():
                if tensor.grad is not None:
                    tensor.grad = None
            metric.backward()
    finally:
        if was_training and hasattr(model, "train"):
            model.train()

    grads: list[torch.Tensor] = []
    for layer in range(n_layers):
        tensor = saved.get(layer)
        if tensor is None or tensor.grad is None:
            raise RuntimeError(
                f"eap_syed: missing residual grad at layer {layer} "
                f"(feature_output_hook={model.feature_output_hook!r})."
            )
        # Hook activations are typically (batch, pos, d_model). Collapse the
        # batch axis so callers can index [layer, pos] safely. ReplacementModel
        # attribution always runs batch size 1.
        grad = tensor.grad.detach()
        if grad.ndim == 3:
            if int(grad.shape[0]) != 1:
                raise ValueError(
                    "eap_syed expects residual grads with batch size 1, "
                    f"got shape {tuple(grad.shape)} at layer {layer}."
                )
            grad = grad[0]
        elif grad.ndim != 2:
            raise ValueError(
                "eap_syed residual grads must be (pos, d_model) or "
                f"(batch, pos, d_model); got {tuple(grad.shape)} at layer {layer}."
            )
        grads.append(grad)
    return torch.stack(grads, dim=0)


def _decoder_grad_contract(
    model: Any,
    *,
    layer: int,
    pos: int,
    feat: int,
    resid_grads: torch.Tensor,
) -> float:
    """``W_dec(feature) · ∂L/∂r`` summed over CLT write sites (or single-layer write)."""
    feat_ids = torch.tensor([feat], device=resid_grads.device, dtype=torch.long)
    W = model.transcoders._get_decoder_vectors(layer, feat_ids)
    W = W.to(device=resid_grads.device, dtype=resid_grads.dtype)

    if W.ndim == 2:
        # Single-layer transcoder: [1, d_model] writes at ``layer``.
        return float(torch.dot(W[0], resid_grads[layer, pos]).item())
    if W.ndim == 3:
        # Cross-layer: [1, n_remaining, d_model] → layers [layer, ..., n_layers).
        n_remaining = int(W.shape[1])
        total = W.new_zeros(())
        for offset in range(n_remaining):
            write_layer = layer + offset
            total = total + torch.dot(W[0, offset], resid_grads[write_layer, pos])
        return float(total.item())
    raise ValueError(f"Unexpected decoder vector shape {tuple(W.shape)}.")


def _scores_via_decoder_atp(
    model: Any,
    *,
    tokens: torch.Tensor,
    a_clean: torch.Tensor,
    a_corr: torch.Tensor,
    pool: Sequence[NodeId],
    node_to_intervention: Mapping[NodeId, tuple[Any, ...] | Sequence[Any]],
    target_logit_idx: int,
    foil_logit_idx: int | None,
    freeze_attention: bool,
) -> tuple[dict[NodeId, float], dict[str, Any]]:
    # Normalized to (n_layers, n_pos, d_model); a_* are (n_layers, n_pos, n_feat).
    resid_grads = _residual_grads_at_feature_output(
        model, tokens, target_logit_idx, foil_logit_idx
    )
    if resid_grads.ndim != 3:
        raise ValueError(
            "eap_syed expected residual grads of shape (n_layers, n_pos, d_model), "
            f"got {tuple(resid_grads.shape)}."
        )
    scores: dict[NodeId, float] = {}
    for node in pool:
        layer, pos, feat = _as_intervention_triple(node_to_intervention[node])
        if not (
            0 <= layer < a_clean.shape[0]
            and 0 <= pos < a_clean.shape[1]
            and 0 <= feat < a_clean.shape[2]
            and 0 <= layer < resid_grads.shape[0]
            and 0 <= pos < resid_grads.shape[1]
        ):
            raise ValueError(
                f"Node {node!r} maps to (layer={layer}, pos={pos}, feat={feat}) outside "
                f"activation/grad shapes {tuple(a_clean.shape)} / {tuple(resid_grads.shape)}."
            )
        delta = float(a_corr[layer, pos, feat].item() - a_clean[layer, pos, feat].item())
        grad_a = _decoder_grad_contract(
            model, layer=layer, pos=pos, feat=feat, resid_grads=resid_grads
        )
        scores[node] = delta * grad_a

    info = {
        "estimator": "syed_attribution_patching",
        "grad_path": "decoder_dot_residual_grad",
        "n_candidates": len(pool),
        "missing_grad_count": 0,
        "freeze_attention": freeze_attention,
        "freeze_attention_note": (
            "ReplacementModel permanently stops grads through attention/LN scales; "
            "freeze_attention is recorded for API parity with feature_intervention."
        ),
        "target_logit_idx": int(target_logit_idx),
        "foil_logit_idx": None if foil_logit_idx is None else int(foil_logit_idx),
        "seq_len": int(tokens.numel()),
    }
    return scores, info


def _scores_via_leaf_interventions(
    model: Any,
    *,
    tokens: torch.Tensor,
    a_clean: torch.Tensor,
    a_corr: torch.Tensor,
    pool: Sequence[NodeId],
    node_to_intervention: Mapping[NodeId, tuple[Any, ...] | Sequence[Any]],
    target_logit_idx: int,
    foil_logit_idx: int | None,
    freeze_attention: bool,
) -> tuple[dict[NodeId, float], dict[str, Any]]:
    """Test-only path: Parameter leaves through a differentiable ``feature_intervention``."""
    device = a_clean.device
    dtype = a_clean.dtype
    params: dict[NodeId, torch.nn.Parameter] = {}
    interventions: list[tuple[int, int, int, torch.Tensor]] = []
    deltas: dict[NodeId, float] = {}

    for node in pool:
        layer, pos, feat = _as_intervention_triple(node_to_intervention[node])
        if not (0 <= layer < a_clean.shape[0] and 0 <= pos < a_clean.shape[1]):
            raise ValueError(
                f"Node {node!r} maps to (layer={layer}, pos={pos}) outside activation "
                f"shape {tuple(a_clean.shape)}."
            )
        a_c = float(a_clean[layer, pos, feat].item())
        a_k = float(a_corr[layer, pos, feat].item())
        deltas[node] = a_k - a_c
        leaf = torch.nn.Parameter(
            torch.tensor(a_c, device=device, dtype=dtype),
            requires_grad=True,
        )
        params[node] = leaf
        interventions.append((layer, pos, feat, leaf))

    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        logits, _ = model.feature_intervention(
            tokens,
            interventions,
            freeze_attention=freeze_attention,
            return_activations=False,
        )
        metric = _logit_gap(logits, target_logit_idx, foil_logit_idx)
        if not metric.requires_grad:
            raise RuntimeError(
                "eap_syed leaf path: feature_intervention returned logits without grad_fn. "
                "On ReplacementModel use the decoder×residual path instead "
                "(feature_intervention is @torch.no_grad)."
            )
        if hasattr(model, "zero_grad"):
            model.zero_grad(set_to_none=True)
        for param in params.values():
            if param.grad is not None:
                param.grad = None
        metric.backward()
    finally:
        if was_training and hasattr(model, "train"):
            model.train()

    scores: dict[NodeId, float] = {}
    missing_grad = 0
    for node in pool:
        grad = params[node].grad
        if grad is None:
            missing_grad += 1
            scores[node] = 0.0
            continue
        scores[node] = float(deltas[node] * float(grad.detach().item()))

    info = {
        "estimator": "syed_attribution_patching",
        "grad_path": "leaf_feature_intervention",
        "n_candidates": len(pool),
        "missing_grad_count": missing_grad,
        "freeze_attention": freeze_attention,
        "target_logit_idx": int(target_logit_idx),
        "foil_logit_idx": None if foil_logit_idx is None else int(foil_logit_idx),
        "seq_len": int(tokens.numel()),
    }
    if missing_grad:
        LOGGER.warning(
            "eap_syed: %d/%d candidates had no gradient; scored as 0.",
            missing_grad,
            len(pool),
        )
    return scores, info


def compute_syed_eap_node_scores(
    model: Any,
    *,
    prompt: str,
    corrupted_prompt: str,
    node_to_intervention: Mapping[NodeId, tuple[Any, ...] | Sequence[Any]],
    candidates: Sequence[NodeId],
    target_logit_idx: int,
    foil_logit_idx: int | None = None,
    freeze_attention: bool = True,
) -> tuple[dict[NodeId, float], dict[str, Any]]:
    """Attribution-patching scores for candidate feature nodes (Syed Eq. 2–3).

    Requires a ``ReplacementModel`` with ``get_activations`` / ``ensure_tokenized``
    and either decoder vectors (production path) or a differentiable
    ``feature_intervention`` (unit-test fakes).
    """
    pool = [node for node in dedupe_preserve_order(candidates) if node in node_to_intervention]
    if not pool:
        raise ValueError("eap_syed needs candidates present in node_to_intervention.")

    clean_tokens = model.ensure_tokenized(prompt)
    corr_tokens = model.ensure_tokenized(corrupted_prompt)
    if int(clean_tokens.numel()) != int(corr_tokens.numel()):
        raise ValueError(
            "eap_syed requires clean and corrupted prompts to tokenize to the same "
            f"length (got {int(clean_tokens.numel())} vs {int(corr_tokens.numel())})."
        )

    with torch.inference_mode():
        _logits_clean, a_clean = model.get_activations(clean_tokens, sparse=False)
        _logits_corr, a_corr = model.get_activations(corr_tokens, sparse=False)
    if a_clean.shape != a_corr.shape:
        raise ValueError(
            f"Activation shape mismatch: clean {tuple(a_clean.shape)} vs "
            f"corrupted {tuple(a_corr.shape)}."
        )

    common = dict(
        tokens=clean_tokens,
        a_clean=a_clean,
        a_corr=a_corr,
        pool=pool,
        node_to_intervention=node_to_intervention,
        target_logit_idx=target_logit_idx,
        foil_logit_idx=foil_logit_idx,
        freeze_attention=freeze_attention,
    )
    if _supports_decoder_atp(model):
        return _scores_via_decoder_atp(model, **common)
    return _scores_via_leaf_interventions(model, **common)


def select_top_eap_syed(
    model: Any,
    *,
    prompt: str,
    corrupted_prompt: str,
    node_to_intervention: Mapping[NodeId, tuple[Any, ...] | Sequence[Any]],
    candidates: Sequence[NodeId],
    target_logit_idx: int,
    foil_logit_idx: int | None = None,
    freeze_attention: bool = True,
    use_absolute: bool = True,
) -> SelectionResult:
    """Rank candidates by Syed/Nanda attribution-patching score, best first."""
    scores, info = compute_syed_eap_node_scores(
        model,
        prompt=prompt,
        corrupted_prompt=corrupted_prompt,
        node_to_intervention=node_to_intervention,
        candidates=candidates,
        target_logit_idx=target_logit_idx,
        foil_logit_idx=foil_logit_idx,
        freeze_attention=freeze_attention,
    )
    rank_scores = {
        node: (abs(score) if use_absolute else score) for node, score in scores.items()
    }
    ranking = sorted(rank_scores, key=lambda node: (-rank_scores[node], str(node)))
    return SelectionResult(
        method="eap_syed",
        ranking=ranking,
        scores=rank_scores,
        params={
            "use_absolute": use_absolute,
            "freeze_attention": freeze_attention,
            "formula": "(a_corr - a_clean) * dL/da_clean",
            "grad_path": info.get("grad_path"),
        },
        extras={**info, "signed_scores": {str(node): scores[node] for node in ranking}},
    )
