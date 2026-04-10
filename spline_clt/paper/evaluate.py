"""Evaluation helpers used by the config-driven paper runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from attribution.causal import build_attribution_graph
from attribution.graph import create_graph_from_attribution
from circuit_tracer.graph import compute_graph_scores
from circuit_tracer.utils.create_graph_files import create_graph_files
from macag.factories.replacement_model import resolve_target_to_logit_idx
from spline_clt.kan_transcoder import KANCrossLayerTranscoder
from spline_clt.seed import seed_everything
from spline_clt.training.data import ActivationDataset


@dataclass
class PromptCache:
    prompt: str
    token_ids: torch.Tensor
    token_strings: list[str]
    mlp_inputs: torch.Tensor
    mlp_outputs: torch.Tensor
    logits: torch.Tensor
    final_resid_pre_ln: torch.Tensor


def load_language_model(model_name: str, device: torch.device):
    """Load a TransformerLens model for prompt-level evaluation."""
    from transformer_lens import HookedTransformer

    lm = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    lm.eval()
    return lm


@torch.no_grad()
def collect_prompt_cache(
    lm: Any,
    prompt: str,
    n_layers: int,
    feature_input_hook: str,
    feature_output_hook: str,
) -> PromptCache:
    """Collect the activations needed for prompt-level evaluation."""
    token_ids_batch = lm.to_tokens(prompt)
    token_ids = token_ids_batch.squeeze(0)
    token_strings = lm.to_str_tokens(prompt)
    hook_names_in = [f"blocks.{i}.{feature_input_hook}" for i in range(n_layers)]
    hook_names_out = [f"blocks.{i}.{feature_output_hook}" for i in range(n_layers)]
    last_layer = n_layers - 1
    resid_post_hook = f"blocks.{last_layer}.hook_resid_post"
    names_filter = hook_names_in + hook_names_out + [resid_post_hook]
    logits, cache = lm.run_with_cache(token_ids_batch, names_filter=names_filter)
    mlp_inputs = torch.stack([cache[name].squeeze(0) for name in hook_names_in])
    mlp_outputs = torch.stack([cache[name].squeeze(0) for name in hook_names_out])
    final_resid_pre_ln = cache[resid_post_hook].squeeze(0)
    return PromptCache(
        prompt=prompt,
        token_ids=token_ids,
        token_strings=token_strings,
        mlp_inputs=mlp_inputs,
        mlp_outputs=mlp_outputs,
        logits=logits.squeeze(0),
        final_resid_pre_ln=final_resid_pre_ln,
    )


def select_logit_tokens(
    logits: torch.Tensor,
    max_n_logits: int,
    desired_logit_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the top output tokens covering the requested probability mass."""
    final_logits = logits[-1].float()
    probs = torch.softmax(final_logits, dim=-1)
    sorted_probs, sorted_idx = probs.sort(descending=True)
    threshold = torch.tensor(desired_logit_prob, device=sorted_probs.device)
    count = int(torch.searchsorted(torch.cumsum(sorted_probs, dim=0), threshold).item()) + 1
    count = max(1, min(max_n_logits, count))
    return sorted_idx[:count], sorted_probs[:count]


@torch.no_grad()
def evaluate_reconstruction_samples(
    model: KANCrossLayerTranscoder,
    dataset: ActivationDataset,
    device: torch.device,
    dtype: torch.dtype,
    sample_indices: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate reconstruction metrics on a deterministic subset of samples."""
    model.eval()
    mse_per_layer = [0.0] * model.n_layers
    cos_total = 0.0
    rel_total = 0.0
    active_total = 0.0
    density_total = 0.0
    records: list[dict[str, Any]] = []

    for sample_index in sample_indices:
        sample = dataset[sample_index]
        x_in = sample["mlp_inputs"].to(device=device, dtype=dtype)
        y_true = sample["mlp_outputs"].to(device=device, dtype=dtype)
        activations = model.encode(x_in)
        y_hat = model.decode_dense(activations, input_acts=x_in)

        layer_mse = []
        for layer_id in range(model.n_layers):
            mse = ((y_hat[layer_id].float() - y_true[layer_id].float()) ** 2).mean().item()
            mse_per_layer[layer_id] += mse
            layer_mse.append(mse)

        cosine_similarity = F.cosine_similarity(
            y_hat.reshape(-1, model.d_model).float(),
            y_true.reshape(-1, model.d_model).float(),
            dim=-1,
        ).mean().item()
        relative_error = (
            (y_hat.float() - y_true.float()).norm()
            / y_true.float().norm().clamp(min=1e-8)
        ).item()
        active = (activations > 0).float()
        active_per_pos = active.sum(dim=-1).mean().item()
        activation_density = active.mean().item()

        cos_total += cosine_similarity
        rel_total += relative_error
        active_total += active_per_pos
        density_total += activation_density
        records.append(
            {
                "record_type": "reconstruction_sample",
                "sample_index": sample_index,
                "mse_total": sum(layer_mse) / len(layer_mse),
                "cosine_similarity": cosine_similarity,
                "relative_error": relative_error,
                "active_per_pos": active_per_pos,
                "activation_density": activation_density,
            }
        )

    n_samples = max(1, len(sample_indices))
    summary = {
        "mse_total": sum(mse_per_layer) / (n_samples * model.n_layers),
        "mse_per_layer": [value / n_samples for value in mse_per_layer],
        "cosine_similarity": cos_total / n_samples,
        "relative_error": rel_total / n_samples,
        "active_per_pos": active_total / n_samples,
        "activation_density": density_total / n_samples,
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        "n_samples": len(sample_indices),
    }
    return summary, records


@torch.no_grad()
def evaluate_prompt_replacement(
    model: KANCrossLayerTranscoder,
    prompt_cache: PromptCache,
    lm: Any,
) -> dict[str, float]:
    """Compute prompt-level replacement fidelity metrics."""
    features = model.encode(prompt_cache.mlp_inputs).to_sparse()
    reconstruction = model.decode(features, input_acts=prompt_cache.mlp_inputs)
    residual_adjustment = (reconstruction - prompt_cache.mlp_outputs).sum(dim=0).float()
    adjusted_resid = prompt_cache.final_resid_pre_ln.float() + residual_adjustment
    normed = lm.ln_final(adjusted_resid)
    replacement_logits = normed @ lm.W_U.float()
    if getattr(lm, "b_U", None) is not None:
        replacement_logits = replacement_logits + lm.b_U.float()

    original_logits = prompt_cache.logits.float()
    original_top1 = original_logits.argmax(dim=-1)
    replacement_top1 = replacement_logits.argmax(dim=-1)
    top1_match_rate = float((original_top1 == replacement_top1).float().mean().item())

    original_probs = torch.softmax(original_logits, dim=-1)
    replacement_log_probs = torch.log_softmax(replacement_logits, dim=-1)
    kl_divergence = float(
        F.kl_div(replacement_log_probs, original_probs, reduction="batchmean").item()
    )
    return {
        "top1_match_rate": top1_match_rate,
        "kl_divergence": kl_divergence,
    }


def build_prompt_graph(
    model: KANCrossLayerTranscoder,
    lm: Any,
    prompt_cache: PromptCache,
    graph_dir: Path,
    graph_slug: str,
    scan_id: str,
    max_features: int,
    max_n_logits: int,
    desired_logit_prob: float,
    node_threshold: float,
    edge_threshold: float,
) -> dict[str, Any]:
    """Build and persist a circuit-tracer graph for one prompt."""
    # Select logit tokens first so we can compute feature→logit edges
    logit_tokens, logit_probabilities = select_logit_tokens(
        logits=prompt_cache.logits,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
    )
    attribution = build_attribution_graph(
        model=model,
        x_in=prompt_cache.mlp_inputs,
        max_features=max_features,
        W_U=lm.W_U,
        logit_token_ids=logit_tokens,
        y_true=prompt_cache.mlp_outputs,
    )
    # Pass content tokens only (exclude BOS at position 0) so that
    # Graph.n_pos matches the number of error/embedding node positions
    # in the adjacency matrix.
    content_tokens = prompt_cache.token_ids[1:].cpu()
    graph = create_graph_from_attribution(
        attribution_result=attribution,
        input_string=prompt_cache.prompt,
        input_tokens=content_tokens,
        logit_tokens=logit_tokens.cpu(),
        logit_probabilities=logit_probabilities.cpu(),
        cfg=lm.cfg,
        scan=scan_id,
    )
    graph_replacement_score, graph_completeness_score = compute_graph_scores(graph)

    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_pt_path = graph_dir / f"{graph_slug}.pt"
    graph_json_path = graph_dir / f"{graph_slug}.json"
    graph.to_pt(str(graph_pt_path))
    create_graph_files(
        graph_or_path=graph,
        slug=graph_slug,
        output_path=graph_dir,
        scan=scan_id,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )
    node_summary = summarize_graph_json(graph_json_path)
    return {
        "active_feature_count": int(len(attribution["active_features"])),
        "graph_replacement_score": graph_replacement_score,
        "graph_completeness_score": graph_completeness_score,
        "graph_pt_path": str(graph_pt_path),
        "graph_json_path": str(graph_json_path),
        "graph_slug": graph_slug,
        **node_summary,
    }


def summarize_graph_json(graph_json_path: Path) -> dict[str, Any]:
    """Summarize retained node counts by type from a pruned graph JSON payload."""
    payload = json.loads(graph_json_path.read_text())
    counts: dict[str, int] = {}
    for node in payload.get("nodes", []):
        if not isinstance(node, dict):
            continue
        feature_type = str(node.get("feature_type", "")).strip().lower()
        counts[feature_type] = counts.get(feature_type, 0) + 1

    retained_feature_node_count = counts.get("cross layer transcoder", 0)
    retained_error_node_count = counts.get("mlp reconstruction error", 0)
    retained_embedding_node_count = counts.get("embedding", 0)
    retained_logit_node_count = counts.get("logit", 0)
    retained_total_node_count = sum(counts.values())
    explainer_node_count = retained_feature_node_count + retained_error_node_count
    retained_error_node_fraction = (
        retained_error_node_count / explainer_node_count
        if explainer_node_count
        else float("nan")
    )
    return {
        "retained_feature_node_count": retained_feature_node_count,
        "retained_error_node_count": retained_error_node_count,
        "retained_embedding_node_count": retained_embedding_node_count,
        "retained_logit_node_count": retained_logit_node_count,
        "retained_total_node_count": retained_total_node_count,
        "retained_error_node_fraction": retained_error_node_fraction,
    }


def load_ranked_feature_nodes(graph_json_path: Path, top_k: int) -> list[str]:
    """Return the top-k graph feature node IDs ranked by influence."""
    payload = json.loads(graph_json_path.read_text())
    nodes = payload.get("nodes", [])
    feature_nodes = [
        node for node in nodes
        if node.get("feature_type") == "cross layer transcoder" and node.get("node_id")
    ]
    feature_nodes.sort(
        key=lambda node: (
            -(float(node.get("influence") or 0.0)),
            -(float(node.get("activation") or 0.0)),
            str(node["node_id"]),
        )
    )
    return [str(node["node_id"]) for node in feature_nodes[:top_k]]


def build_logit_gap_direction(
    tokenizer: Any,
    unembed: torch.Tensor,
    target_token: str,
    foil_token: str,
) -> tuple[int, int, torch.Tensor]:
    """Construct a logit-gap direction vector for Shapley analysis."""
    target_map = resolve_target_to_logit_idx(
        tokenizer=tokenizer,
        target_token_by_label={"y": target_token, "y_foil": foil_token},
        strict_single_token=False,
    )
    target_idx = target_map["y"]
    foil_idx = target_map["y_foil"]
    direction = unembed[:, target_idx].float() - unembed[:, foil_idx].float()
    return target_idx, foil_idx, direction


def jaccard_overlap(items_a: list[str], items_b: list[str]) -> float:
    """Compute the Jaccard overlap between two ranked sets."""
    set_a = set(items_a)
    set_b = set(items_b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def deterministic_sample_indices(total: int, count: int, seed: int) -> list[int]:
    """Sample a deterministic subset of dataset indices."""
    seed_everything(seed)
    count = min(total, count)
    return torch.randperm(total)[:count].tolist()


def feature_node_id(layer_id: int, position: int, feature_id: int) -> str:
    """Construct the canonical circuit-tracer node id for one feature."""
    return f"{layer_id}_{feature_id}_{position}"


def parse_feature_node_id(node_id: str) -> tuple[int, int, int] | None:
    """Parse a circuit-tracer feature node id into (layer, position, feature_id)."""
    parts = str(node_id).split("_")
    if len(parts) != 3:
        return None
    try:
        layer_id = int(parts[0])
        feature_id = int(parts[1])
        position = int(parts[2])
    except ValueError:
        return None
    return layer_id, position, feature_id


def replacement_logits_from_reconstruction(
    prompt_cache: PromptCache,
    reconstruction: torch.Tensor,
    lm: Any,
) -> torch.Tensor:
    """Project reconstructed MLP outputs back to token logits."""
    residual_adjustment = (reconstruction - prompt_cache.mlp_outputs).sum(dim=0).float()
    adjusted_resid = prompt_cache.final_resid_pre_ln.float() + residual_adjustment
    normed = lm.ln_final(adjusted_resid)
    replacement_logits = normed @ lm.W_U.float()
    if getattr(lm, "b_U", None) is not None:
        replacement_logits = replacement_logits + lm.b_U.float()
    return replacement_logits


@torch.no_grad()
def replacement_logit_gap_from_subset(
    model: KANCrossLayerTranscoder,
    prompt_cache: PromptCache,
    lm: Any,
    target_idx: int,
    foil_idx: int,
    selected_node_ids: list[str] | None,
) -> float:
    """Evaluate the final-position target-foil logit gap for a selected feature subset."""
    baseline_activations = model.encode(prompt_cache.mlp_inputs)
    if selected_node_ids is not None:
        masked_activations = torch.zeros_like(baseline_activations)
        for node_id in selected_node_ids:
            triple = parse_feature_node_id(node_id)
            if triple is None:
                continue
            layer_id, position, feature_id = triple
            if (
                0 <= layer_id < masked_activations.shape[0]
                and 0 <= position < masked_activations.shape[1]
                and 0 <= feature_id < masked_activations.shape[2]
            ):
                masked_activations[layer_id, position, feature_id] = baseline_activations[
                    layer_id, position, feature_id
                ]
    else:
        masked_activations = baseline_activations

    reconstruction = model.decode_dense(masked_activations, input_acts=prompt_cache.mlp_inputs)
    replacement_logits = replacement_logits_from_reconstruction(
        prompt_cache=prompt_cache,
        reconstruction=reconstruction,
        lm=lm,
    )
    final_logits = replacement_logits[-1].float()
    return float((final_logits[target_idx] - final_logits[foil_idx]).item())
