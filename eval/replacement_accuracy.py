"""Replacement model accuracy evaluation for KAN-CLT.

Measures how well the KAN-CLT replacement model matches the original model's outputs.
Key metrics:
    - Top-1 match rate: fraction of tokens where replacement model's top prediction matches original
    - Per-layer reconstruction MSE
    - KL divergence between output distributions
"""

import torch
from torch.nn import functional as F
from tqdm import tqdm

from kan_clt.kan_transcoder import KANCrossLayerTranscoder


@torch.no_grad()
def evaluate_reconstruction(
    model: KANCrossLayerTranscoder,
    mlp_inputs: torch.Tensor,
    mlp_outputs: torch.Tensor,
) -> dict[str, float | list[float]]:
    """Evaluate MLP output reconstruction quality.

    Args:
        model: Trained KAN-CLT.
        mlp_inputs: Residual stream inputs, shape (n_samples, n_layers, seq_len, d_model).
        mlp_outputs: True MLP outputs, shape (n_samples, n_layers, seq_len, d_model).

    Returns:
        Dict with:
            mse_total: Average MSE across all layers.
            mse_per_layer: List of per-layer MSE values.
            cosine_similarity: Average cosine similarity.
            relative_error: ||y_hat - y|| / ||y||.
    """
    n_samples, n_layers, seq_len, d_model = mlp_inputs.shape

    mse_per_layer = [0.0] * n_layers
    cos_sim_total = 0.0
    rel_error_total = 0.0
    n_total = 0

    for i in range(n_samples):
        x_in = mlp_inputs[i].permute(1, 0, 2)  # (seq_len, n_layers, d_model) → wrong
        # Actually: (n_layers, seq_len, d_model) — already correct after indexing
        x_in = mlp_inputs[i]  # (n_layers, seq_len, d_model)
        y_true = mlp_outputs[i]  # (n_layers, seq_len, d_model)

        features = model.encode(x_in).to_sparse()
        y_hat = model.decode(features, input_acts=x_in)

        for l in range(n_layers):
            mse = ((y_hat[l] - y_true[l]) ** 2).mean().item()
            mse_per_layer[l] += mse

        # Cosine similarity (flattened)
        cos_sim = F.cosine_similarity(
            y_hat.reshape(-1, d_model), y_true.reshape(-1, d_model), dim=-1
        ).mean().item()
        cos_sim_total += cos_sim

        # Relative error
        rel = (y_hat - y_true).norm() / y_true.norm().clamp(min=1e-8)
        rel_error_total += rel.item()

        n_total += 1

    mse_per_layer = [m / n_total for m in mse_per_layer]

    return {
        "mse_total": sum(mse_per_layer) / n_layers,
        "mse_per_layer": mse_per_layer,
        "cosine_similarity": cos_sim_total / n_total,
        "relative_error": rel_error_total / n_total,
    }


@torch.no_grad()
def evaluate_replacement_accuracy(
    original_model,
    kan_clt: KANCrossLayerTranscoder,
    prompts: list[str],
    device: torch.device | None = None,
) -> dict[str, float]:
    """Evaluate top-1 accuracy of KAN-CLT replacement model vs original.

    Requires a TransformerLens HookedTransformer as the original model.

    Args:
        original_model: HookedTransformer original model.
        kan_clt: Trained KAN-CLT.
        prompts: List of text prompts to evaluate on.
        device: Device to run on.

    Returns:
        Dict with top1_match_rate, kl_divergence, and per-position accuracy.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    top1_matches = 0
    total_positions = 0
    kl_div_total = 0.0

    for prompt in tqdm(prompts, desc="Evaluating replacement accuracy"):
        tokens = original_model.tokenizer(
            prompt, return_tensors="pt"
        ).input_ids.to(device)

        # Original model logits
        original_logits = original_model(tokens).squeeze(0)  # (seq_len, vocab)

        # Collect MLP inputs/outputs via hooks
        hook_names_in = [
            f"blocks.{i}.{kan_clt.feature_input_hook}"
            for i in range(kan_clt.n_layers)
        ]
        hook_names_out = [
            f"blocks.{i}.{kan_clt.feature_output_hook}"
            for i in range(kan_clt.n_layers)
        ]

        _, cache = original_model.run_with_cache(
            tokens, names_filter=hook_names_in + hook_names_out
        )

        mlp_in = torch.stack(
            [cache[name].squeeze(0) for name in hook_names_in]
        )  # (n_layers, seq_len, d_model)
        mlp_out = torch.stack(
            [cache[name].squeeze(0) for name in hook_names_out]
        )

        # KAN-CLT reconstruction
        features = kan_clt.encode(mlp_in).to_sparse()
        recon = kan_clt.decode(features, input_acts=mlp_in)

        # Compute replacement logits by replacing MLP outputs
        # Simple approximation: adjust final residual by reconstruction difference
        residual_adjustment = (recon - mlp_out).sum(dim=0)  # (seq_len, d_model)

        # Get final residual stream from original model
        _, final_cache = original_model.run_with_cache(
            tokens, names_filter=["ln_final.hook_normalized"]
        )
        final_resid = final_cache["ln_final.hook_normalized"].squeeze(0)

        # Adjusted logits
        replacement_logits = (final_resid + residual_adjustment) @ original_model.W_U + original_model.b_U

        # Top-1 match
        original_top1 = original_logits.argmax(dim=-1)
        replacement_top1 = replacement_logits.argmax(dim=-1)
        top1_matches += (original_top1 == replacement_top1).sum().item()
        total_positions += len(original_top1)

        # KL divergence
        original_probs = F.softmax(original_logits, dim=-1)
        replacement_log_probs = F.log_softmax(replacement_logits, dim=-1)
        kl = F.kl_div(replacement_log_probs, original_probs, reduction="batchmean")
        kl_div_total += kl.item()

    n_prompts = len(prompts)
    return {
        "top1_match_rate": top1_matches / max(1, total_positions),
        "kl_divergence": kl_div_total / max(1, n_prompts),
    }


@torch.no_grad()
def evaluate_sparsity(
    model: KANCrossLayerTranscoder,
    mlp_inputs: torch.Tensor,
) -> dict[str, float | list[float]]:
    """Evaluate feature sparsity statistics.

    Args:
        model: Trained KAN-CLT.
        mlp_inputs: Residual stream inputs, shape (n_samples, n_layers, seq_len, d_model).

    Returns:
        Dict with average_active_per_pos, activation_density, per-layer stats.
    """
    n_samples = mlp_inputs.shape[0]
    active_counts = []
    densities = []

    for i in range(n_samples):
        x_in = mlp_inputs[i]
        features = model.encode(x_in)  # (n_layers, seq_len, d_transcoder)
        active = (features > 0).float()

        active_per_pos = active.sum(dim=-1).mean().item()
        density = active.mean().item()

        active_counts.append(active_per_pos)
        densities.append(density)

    return {
        "average_active_per_pos": sum(active_counts) / n_samples,
        "activation_density": sum(densities) / n_samples,
    }
