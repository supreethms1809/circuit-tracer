"""Bridge helpers for UFO-101 ``auto-circuit`` (true ACDC / Syed EAP)."""

from __future__ import annotations

import logging
from itertools import count
from typing import Any, Sequence, Set

import torch
import torch.nn.functional as F

from macag.baselines.original.tl_compat import enable_autocircuit_tl3_compat

LOGGER = logging.getLogger(__name__)

_GQA_DEST_PATCHED = False


def prepare_hooked_transformer_for_autocircuit(model: Any) -> Any:
    """Set TL flags required by auto-circuit's factorized edge graph."""
    cfg = model.cfg
    cfg.use_attn_result = True
    if not getattr(cfg, "attn_only", False):
        cfg.use_hook_mlp_in = True
    cfg.use_split_qkv_input = True
    # Dest-side residual edges also require attn_in when separate_qkv is False;
    # with separate_qkv=True, split qkv input is the required flag (set above).
    return model


def _n_kv_heads(cfg: Any) -> int:
    n_heads = int(cfg.n_heads)
    n_kv = getattr(cfg, "n_key_value_heads", None)
    if n_kv is None:
        return n_heads
    return int(n_kv)


def factorized_dest_nodes_gqa(model: Any, separate_qkv: bool) -> Set[Any]:
    """GQA-aware replacement for auto-circuit ``factorized_dest_nodes``.

    Upstream builds K/V dests for every Q head index, but TransformerLens
    ``hook_{k,v}_input`` tensors have shape ``[..., n_key_value_heads, ...]``
    under GQA (Gemma-2, Llama-3). Indexing / patching with ``head_idx >= n_kv``
    raises ``IndexError``. Q destinations still use ``n_heads``.
    """
    from auto_circuit.types import DestNode

    cfg = model.cfg
    if separate_qkv:
        assert cfg.use_split_qkv_input
    else:
        assert cfg.use_attn_in
    if not cfg.attn_only:
        assert cfg.use_hook_mlp_in

    n_heads = int(cfg.n_heads)
    n_kv = _n_kv_heads(cfg)
    layers = count(1)
    nodes: Set[Any] = set()
    for block_idx in range(int(cfg.n_layers)):
        layer = next(layers)
        if separate_qkv:
            for head_idx in range(n_heads):
                nodes.add(
                    DestNode(
                        name=f"A{block_idx}.{head_idx}.Q",
                        module_name=f"blocks.{block_idx}.hook_q_input",
                        layer=layer,
                        head_dim=2,
                        head_idx=head_idx,
                        weight=f"blocks.{block_idx}.attn.W_Q",
                        weight_head_dim=0,
                    )
                )
            for head_idx in range(n_kv):
                for letter in ("K", "V"):
                    nodes.add(
                        DestNode(
                            name=f"A{block_idx}.{head_idx}.{letter}",
                            module_name=(
                                f"blocks.{block_idx}.hook_{letter.lower()}_input"
                            ),
                            layer=layer,
                            head_dim=2,
                            head_idx=head_idx,
                            weight=f"blocks.{block_idx}.attn.W_{letter}",
                            weight_head_dim=0,
                        )
                    )
        else:
            # Shared attn_in is still indexed by n_heads in TL.
            for head_idx in range(n_heads):
                nodes.add(
                    DestNode(
                        name=f"A{block_idx}.{head_idx}",
                        module_name=f"blocks.{block_idx}.hook_attn_in",
                        layer=layer,
                        head_dim=2,
                        head_idx=head_idx,
                        weight=f"blocks.{block_idx}.attn.W_QKV",
                        weight_head_dim=0,
                    )
                )
        if not cfg.attn_only:
            nodes.add(
                DestNode(
                    name=f"MLP {block_idx}",
                    module_name=f"blocks.{block_idx}.hook_mlp_in",
                    layer=layer if cfg.parallel_attn_mlp else next(layers),
                    weight=f"blocks.{block_idx}.mlp.W_in",
                )
            )
    nodes.add(
        DestNode(
            name="Resid End",
            module_name=f"blocks.{int(cfg.n_layers) - 1}.hook_resid_post",
            layer=next(layers),
            weight="unembed.W_U",
        )
    )
    return nodes


def enable_autocircuit_gqa_dest_patch() -> None:
    """Monkeypatch auto-circuit dest-node construction for GQA models."""
    global _GQA_DEST_PATCHED
    if _GQA_DEST_PATCHED:
        return
    enable_autocircuit_tl3_compat()
    import auto_circuit.model_utils.transformer_lens_utils as tl_utils

    tl_utils.factorized_dest_nodes = factorized_dest_nodes_gqa  # type: ignore[assignment]
    _GQA_DEST_PATCHED = True
    LOGGER.info(
        "Patched auto_circuit factorized_dest_nodes for GQA (K/V use n_key_value_heads)."
    )

def build_prompt_loader(
    model: Any,
    *,
    clean_prompt: str,
    corrupted_prompt: str,
    target_idx: int,
    foil_idx: int | None,
    batch_size: int = 1,
) -> tuple[Any, int]:
    """Build a one-prompt (or batched) ``PromptDataLoader`` for auto-circuit."""
    enable_autocircuit_tl3_compat()
    from auto_circuit.data import PromptDataLoader, PromptDataset

    clean = model.to_tokens(clean_prompt)
    corrupt = model.to_tokens(corrupted_prompt)
    if int(clean.shape[-1]) != int(corrupt.shape[-1]):
        raise ValueError(
            "clean/corrupted prompts must tokenize to the same length "
            f"(got {int(clean.shape[-1])} vs {int(corrupt.shape[-1])})."
        )
    seq_len = int(clean.shape[-1])
    answers = [torch.tensor([int(target_idx)], device=clean.device)]
    if foil_idx is None:
        # auto-circuit answer_diff needs a wrong answer; fall back to a different id.
        foil = (int(target_idx) + 1) % int(model.cfg.d_vocab)
        LOGGER.warning("No foil token; using vocab id %d as wrong answer for AtP.", foil)
        wrong = [torch.tensor([foil], device=clean.device)]
    else:
        wrong = [torch.tensor([int(foil_idx)], device=clean.device)]

    dataset = PromptDataset(clean, corrupt, answers, wrong)
    # drop_last=True inside PromptDataLoader → need len(dataset) >= batch_size.
    if len(dataset) < batch_size:
        raise ValueError(f"Need at least {batch_size} prompts; got {len(dataset)}.")
    loader = PromptDataLoader(
        dataset,
        seq_len=seq_len,
        diverge_idx=0,
        batch_size=batch_size,
    )
    return loader, seq_len


def make_patchable_model(model: Any, *, device: torch.device | None = None) -> Any:
    """Wrap a HookedTransformer as auto-circuit ``PatchableModel`` (factorized edges)."""
    enable_autocircuit_tl3_compat()
    enable_autocircuit_gqa_dest_patch()
    from auto_circuit.utils.graph_utils import patchable_model

    prepare_hooked_transformer_for_autocircuit(model)
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    # Keep activations/masks in float32: auto-circuit einsum rejects bf16/float mixes.
    try:
        param = next(model.parameters())
        if param.dtype != torch.float32:
            LOGGER.warning(
                "Casting model %s -> float32 for auto-circuit compatibility",
                param.dtype,
            )
            model = model.to(dtype=torch.float32)
    except StopIteration:
        pass
    return patchable_model(
        model,
        factorized=True,
        slice_output="last_seq",
        separate_qkv=True,
        device=device,
    )


def top_edges_from_prune_scores(
    model: Any, prune_scores: dict[str, torch.Tensor], k: int
) -> list[str]:
    """Return the top-|score| ``k`` edge name strings from a PruneScores map.

    Deterministic among ties: sort by ``(-|score|, edge_name)`` then take ``k``.
    Does **not** expand tied score bands (unlike auto-circuit's threshold keep).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    k = min(k, len(model.edges))
    ranked: list[tuple[float, str]] = []
    for edge in model.edges:
        score = float(edge.prune_score(prune_scores).abs().item())
        ranked.append((score, str(edge)))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in ranked[:k]]


def binary_prune_scores_for_edges(
    model: Any, edge_names: Sequence[str]
) -> dict[str, torch.Tensor]:
    """PruneScores with 1.0 on ``edge_names`` and 0.0 elsewhere (exact circuit)."""
    wanted = set(edge_names)
    ps = model.new_prune_scores(init_val=0.0)
    for edge in model.edges:
        if str(edge) in wanted or getattr(edge, "name", None) in wanted:
            ps[edge.dest.module_name][edge.patch_idx] = 1.0
    return ps


def circuit_logits_at_k(
    model: Any,
    loader: Any,
    prune_scores: dict[str, torch.Tensor],
    k: int,
    *,
    patch_type: Any | None = None,
) -> tuple[torch.Tensor, int]:
    """Run a patched circuit of size ``k``; return (logits, realized_edge_count).

    Warning: auto-circuit thresholds on score magnitude, so tied scores (ACDC's
    ``+inf`` / ``τ`` bands) can realize far more than ``k`` edges. Prefer
    :func:`circuit_logits_for_edges` for budget-matched evaluation.
    """
    enable_autocircuit_tl3_compat()
    from auto_circuit.prune import run_circuits
    from auto_circuit.types import AblationType, PatchType

    if patch_type is None:
        patch_type = PatchType.TREE_PATCH
    outs = run_circuits(
        model,
        loader,
        test_edge_counts=[k],
        prune_scores=prune_scores,
        patch_type=patch_type,
        ablation_type=AblationType.RESAMPLE,
    )
    if not outs:
        raise RuntimeError("run_circuits returned no outputs")
    if k in outs and outs[k]:
        realized = k
        batch_outs = outs[k]
    else:
        realized = min(outs.keys(), key=lambda n: (abs(int(n) - k), int(n)))
        batch_outs = outs[realized]
        if not batch_outs:
            raise RuntimeError(
                f"run_circuits produced empty batch map at realized k={realized} "
                f"(requested {k}; keys={sorted(outs)})"
            )
    return next(iter(batch_outs.values())), int(realized)


def circuit_logits_for_edges(
    model: Any,
    loader: Any,
    edge_names: Sequence[str],
    *,
    patch_type: Any | None = None,
) -> tuple[torch.Tensor, int]:
    """Patched logits for an **exact** edge set (no threshold-band expansion)."""
    enable_autocircuit_tl3_compat()
    from auto_circuit.types import PatchType

    if patch_type is None:
        patch_type = PatchType.TREE_PATCH
    if not edge_names and patch_type == PatchType.TREE_PATCH:
        # Empty keep-only: patch every edge (binary scores all zero, k=0).
        ps = model.new_prune_scores(init_val=0.0)
        return circuit_logits_at_k(model, loader, ps, 0, patch_type=patch_type)
    if not edge_names:
        raise ValueError("edge_names must be non-empty for EDGE_PATCH remove")
    ps = binary_prune_scores_for_edges(model, edge_names)
    logits, realized = circuit_logits_at_k(
        model, loader, ps, len(edge_names), patch_type=patch_type
    )
    if realized != len(edge_names):
        LOGGER.warning(
            "Exact-edge circuit realized %d edges; requested %d (patch_type=%s)",
            realized,
            len(edge_names),
            patch_type,
        )
    return logits, realized


def logit_gap_faithfulness_for_edges(
    model: Any,
    loader: Any,
    edge_names: Sequence[str],
    *,
    target_idx: int,
    foil_idx: int | None,
    alpha: float = 0.5,
    clean_tokens: Any | None = None,
) -> dict[str, Any]:
    """Logit-gap S/N/F for an exact edge circuit (AutoCircuit TREE/EDGE patch).

    Definitions (resample ablation on corrupted activations):
    - ``all``: clean forward, no patching
    - ``empty``: TREE_PATCH with zero edges kept (all edges resampled)
    - ``keep_only``: TREE_PATCH keeping exactly ``edge_names``
    - ``remove``: EDGE_PATCH ablating exactly ``edge_names``
    """
    enable_autocircuit_tl3_compat()
    from auto_circuit.types import PatchType

    from macag.baselines.original.metrics import logit_gap_from_logits

    if clean_tokens is None:
        batch = next(iter(loader))
        clean_tokens = batch.clean

    full_logits = full_model_logits(model, clean_tokens)
    s_all = logit_gap_from_logits(full_logits, target_idx, foil_idx)

    empty_logits, _ = circuit_logits_for_edges(
        model, loader, [], patch_type=PatchType.TREE_PATCH
    )
    s_empty = logit_gap_from_logits(empty_logits, target_idx, foil_idx)

    if edge_names:
        keep_logits, keep_realized = circuit_logits_for_edges(
            model, loader, edge_names, patch_type=PatchType.TREE_PATCH
        )
        rem_logits, rem_realized = circuit_logits_for_edges(
            model, loader, edge_names, patch_type=PatchType.EDGE_PATCH
        )
    else:
        keep_logits, keep_realized = empty_logits, 0
        rem_logits, rem_realized = full_logits, 0
    s_keep = logit_gap_from_logits(keep_logits, target_idx, foil_idx)
    s_remove = logit_gap_from_logits(rem_logits, target_idx, foil_idx)

    sufficiency = s_keep - s_empty
    necessity = s_all - s_remove
    faithfulness = float(alpha) * sufficiency + (1.0 - float(alpha)) * necessity
    return {
        "size": len(edge_names),
        "alpha": float(alpha),
        "scores": {
            "all": s_all,
            "empty": s_empty,
            "keep_only": s_keep,
            "remove": s_remove,
            "sufficiency": sufficiency,
            "necessity": necessity,
            "faithfulness": faithfulness,
        },
        "realized_keep_edges": keep_realized,
        "realized_remove_edges": rem_realized,
        "edge_universe": "autocircuit_factorized_qkv",
        "ablation": "resample_corrupt",
    }


def score_circuit_logits(
    full_logits: torch.Tensor,
    circuit_logits: torch.Tensor,
    *,
    target_idx: int,
    foil_idx: int | None,
) -> dict[str, float]:
    """Logit-gap + KL(full ‖ circuit) on the last-token distribution."""
    if full_logits.ndim == 3:
        full_last = full_logits[0, -1]
    elif full_logits.ndim == 2:
        # auto-circuit slice_output=last_seq → [batch, vocab]
        full_last = full_logits[0]
    else:
        raise ValueError(f"Unexpected full logits shape {tuple(full_logits.shape)}")

    if circuit_logits.ndim == 3:
        circ_last = circuit_logits[0, -1]
    elif circuit_logits.ndim == 2:
        circ_last = circuit_logits[0]
    else:
        raise ValueError(f"Unexpected circuit logits shape {tuple(circuit_logits.shape)}")

    if foil_idx is None:
        gap = float(circ_last[target_idx].detach().item())
    else:
        gap = float((circ_last[target_idx] - circ_last[foil_idx]).detach().item())

    full_log_probs = F.log_softmax(full_last.float(), dim=-1)
    circ_log_probs = F.log_softmax(circ_last.float(), dim=-1)
    full_probs = full_log_probs.exp()
    kl = float(torch.sum(full_probs * (full_log_probs - circ_log_probs)).item())
    return {"logit_gap": gap, "kl": kl}


def full_model_logits(model: Any, clean_tokens: torch.Tensor) -> torch.Tensor:
    """Clean forward through the wrapped / underlying model."""
    wrapped = getattr(model, "wrapped_model", model)
    with torch.inference_mode():
        logits = wrapped(clean_tokens)
    # Match last-token slice used by patchable models.
    if logits.ndim == 3:
        return logits[:, -1, :]
    return logits
