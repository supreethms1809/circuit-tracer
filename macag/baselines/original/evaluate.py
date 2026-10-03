"""Evaluate a kept set of output edges under corrupted patching."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from macag.baselines.original.edges import OutputEdge, edges_by_id
from macag.baselines.original.metrics import kl_full_vs_circuit, logit_gap_from_logits


def _patch_hooks(
    edges_to_patch: Sequence[OutputEdge],
    corrupt_cache: dict[str, torch.Tensor],
) -> list[tuple[str, Any]]:
    """Replace clean activations with corrupted ones for the listed edges."""
    heads_by_layer: dict[int, list[int]] = {}
    mlp_layers: list[int] = []
    for edge in edges_to_patch:
        if edge.kind == "head" and edge.head is not None:
            heads_by_layer.setdefault(edge.layer, []).append(edge.head)
        else:
            mlp_layers.append(edge.layer)

    hooks: list[tuple[str, Any]] = []

    for layer, heads in sorted(heads_by_layer.items()):
        hook_name = f"blocks.{layer}.attn.hook_result"
        corrupt = corrupt_cache[hook_name]
        heads_sorted = sorted(heads)

        def make_head_hook(corr: torch.Tensor, head_list: list[int]):
            def hook(value: torch.Tensor, hook: Any = None) -> torch.Tensor:
                # value: [batch, pos, head, d_head]
                value[:, :, head_list, :] = corr[:, :, head_list, :]
                return value

            return hook

        hooks.append((hook_name, make_head_hook(corrupt, heads_sorted)))

    for layer in sorted(set(mlp_layers)):
        hook_name = f"blocks.{layer}.hook_mlp_out"
        corrupt = corrupt_cache[hook_name]

        def make_mlp_hook(corr: torch.Tensor):
            def hook(value: torch.Tensor, hook: Any = None) -> torch.Tensor:
                value[:] = corr
                return value

            return hook

        hooks.append((hook_name, make_mlp_hook(corrupt)))

    return hooks


def run_with_patched_edges(
    model: Any,
    clean_tokens: torch.Tensor,
    corrupt_cache: dict[str, torch.Tensor],
    *,
    all_edges: Sequence[OutputEdge],
    kept_edge_ids: set[str],
) -> torch.Tensor:
    """Forward on clean tokens while corrupt-patching every edge not in ``kept``."""
    by_id = edges_by_id(all_edges)
    to_patch = [by_id[eid] for eid in by_id if eid not in kept_edge_ids]
    hooks = _patch_hooks(to_patch, corrupt_cache)
    with torch.inference_mode():
        return model.run_with_hooks(clean_tokens, fwd_hooks=hooks)


def score_edge_circuit(
    model: Any,
    clean_tokens: torch.Tensor,
    corrupt_cache: dict[str, torch.Tensor],
    full_clean_logits: torch.Tensor,
    *,
    all_edges: Sequence[OutputEdge],
    kept_edge_ids: set[str],
    target_idx: int,
    foil_idx: int | None,
) -> dict[str, float]:
    """Logit-gap and KL(full ‖ circuit) for a kept edge set."""
    circuit_logits = run_with_patched_edges(
        model,
        clean_tokens,
        corrupt_cache,
        all_edges=all_edges,
        kept_edge_ids=kept_edge_ids,
    )
    return {
        "logit_gap": logit_gap_from_logits(circuit_logits, target_idx, foil_idx),
        "kl": kl_full_vs_circuit(full_clean_logits, circuit_logits),
    }


def build_corrupt_cache(model: Any, corrupt_tokens: torch.Tensor, edges: Sequence[OutputEdge]) -> dict[str, torch.Tensor]:
    """Cache corrupted activations at every output-edge hook."""
    names = sorted({edge.hook_name for edge in edges})
    logits, cache = model.run_with_cache(corrupt_tokens, names_filter=names)
    del logits
    return {name: cache[name].detach() for name in names}
