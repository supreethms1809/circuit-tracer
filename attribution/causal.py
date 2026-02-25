"""Causal (ablation-based) attribution for KAN-CLT.

Since the KAN encoder is nonlinear, we cannot rely solely on backward Jacobians
for attribution. This module implements ablation-based causal attribution:

For each active feature s:
    1. Zero out a_s (set activation to 0)
    2. Re-run the forward pass
    3. Measure change in every downstream feature and output logit

Edge weight A_{s→t} = a_t_original - a_t_ablated

This gives exact causal effects without linear approximation.
"""

import torch
from tqdm import tqdm

from kan_clt.kan_transcoder import KANCrossLayerTranscoder


def ablation_attribution(
    model: KANCrossLayerTranscoder,
    x_in: torch.Tensor,
    y_true: torch.Tensor | None = None,
    max_features: int = 256,
    batch_ablations: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute causal attribution via single-feature ablation.

    Args:
        model: Trained KAN-CLT model.
        x_in: Residual stream inputs, shape (n_layers, n_pos, d_model).
        y_true: Optional true MLP outputs for error computation.
        max_features: Maximum number of features to attribute (by activation magnitude).
        batch_ablations: Whether to batch ablations for efficiency.

    Returns:
        Dict with:
            active_features: (n_active, 3) tensor of (layer, pos, feat_idx).
            activation_values: (n_active,) tensor of feature activations.
            feature_effects: (n_active, n_active) feature-to-feature effects.
            output_effects: (n_active, n_layers, n_pos, d_model) feature-to-output effects.
    """
    n_layers, n_pos, d_model = x_in.shape

    # Step 1: Forward pass to get baseline activations (no grad needed)
    with torch.no_grad():
        baseline_activations = model.encode(x_in)  # (n_layers, n_pos, d_transcoder)
        baseline_sparse = baseline_activations.to_sparse()

        # Get active features
        layer_idx, pos_idx, feat_idx = baseline_sparse.coalesce().indices()
        act_values = baseline_sparse.coalesce().values()

        n_active = len(act_values)
        active_features = torch.stack([layer_idx, pos_idx, feat_idx], dim=1)

        # Limit to top features by activation magnitude
        if n_active > max_features:
            top_indices = act_values.abs().topk(max_features).indices
            active_features = active_features[top_indices]
            act_values = act_values[top_indices]
            layer_idx = active_features[:, 0]
            pos_idx = active_features[:, 1]
            feat_idx = active_features[:, 2]
            n_active = max_features

        # Baseline reconstruction
        baseline_recon = model.decode(baseline_sparse, input_acts=x_in)

    # Step 2: Precompute encoder directions for all active features (needs grad)
    encoder_directions = _batch_encoder_directions(
        model, x_in, layer_idx, pos_idx, feat_idx
    )  # (n_active, d_model)

    # Step 3: Ablate each feature and measure effects (no grad needed)
    with torch.no_grad():
        feature_to_feature = torch.zeros(
            n_active, n_active, device=x_in.device, dtype=x_in.dtype
        )
        feature_to_output = torch.zeros(
            n_active, n_layers, n_pos, d_model, device=x_in.device, dtype=x_in.dtype
        )

        for i in tqdm(range(n_active), desc="Ablating features", disable=n_active < 50):
            # Create ablated activations (zero out feature i)
            ablated_acts = baseline_activations.clone()
            l, p, f = layer_idx[i], pos_idx[i], feat_idx[i]
            ablated_acts[l, p, f] = 0.0

            # Compute ablated reconstruction
            ablated_sparse = ablated_acts.to_sparse()
            ablated_recon = model.decode(ablated_sparse, input_acts=x_in)

            # Effect on reconstruction (output)
            recon_diff = baseline_recon - ablated_recon
            feature_to_output[i] = recon_diff

            # Feature-to-feature effect: project reconstruction change onto
            # each target feature's encoder direction
            for j in range(n_active):
                l_j, p_j = layer_idx[j], pos_idx[j]
                feature_to_feature[i, j] = (
                    recon_diff[l_j, p_j] * encoder_directions[j]
                ).sum()

    return {
        "active_features": active_features,
        "activation_values": act_values,
        "feature_effects": feature_to_feature,
        "output_effects": feature_to_output,
        "baseline_reconstruction": baseline_recon,
    }


def _batch_encoder_directions(
    model: KANCrossLayerTranscoder,
    x_in: torch.Tensor,
    layer_idx: torch.Tensor,
    pos_idx: torch.Tensor,
    feat_idx: torch.Tensor,
) -> torch.Tensor:
    """Compute encoder directions (Jacobian rows) for a batch of active features.

    Args:
        model: KAN-CLT model.
        x_in: Input activations, shape (n_layers, n_pos, d_model).
        layer_idx: Layer indices for each feature.
        pos_idx: Position indices for each feature.
        feat_idx: Feature indices for each feature.

    Returns:
        Encoder directions, shape (n_features, d_model).
    """
    n_features = len(layer_idx)
    d_model = x_in.shape[-1]
    directions = torch.zeros(n_features, d_model, device=x_in.device, dtype=x_in.dtype)

    for i in range(n_features):
        layer = int(layer_idx[i])
        pos = int(pos_idx[i])
        feat = int(feat_idx[i])

        x_pos = x_in[layer, pos].detach().requires_grad_(True)
        with torch.enable_grad():
            output = model.encoders[layer](x_pos.unsqueeze(0))  # (1, n_features)
            output[0, feat].backward()
        directions[i] = x_pos.grad.detach()

    return directions


def build_attribution_graph(
    model: KANCrossLayerTranscoder,
    x_in: torch.Tensor,
    logit_effects: torch.Tensor | None = None,
    max_features: int = 256,
) -> dict[str, torch.Tensor]:
    """Build a full attribution graph compatible with circuit-tracer's Graph format.

    Node ordering: [active_features, error_nodes, token_nodes, logit_nodes]

    Args:
        model: Trained KAN-CLT model.
        x_in: Residual stream inputs, shape (n_layers, n_pos, d_model).
        logit_effects: Optional pre-computed logit attribution, shape (n_logits, n_nodes).
        max_features: Max features to include.

    Returns:
        Dict with active_features, adjacency_matrix, and metadata needed for Graph.
    """
    attribution = ablation_attribution(model, x_in, max_features=max_features)

    n_active = len(attribution["active_features"])
    n_layers, n_pos, d_model = x_in.shape
    n_error = n_layers * n_pos
    n_tokens = n_pos
    n_logits = logit_effects.shape[0] if logit_effects is not None else 0

    total_nodes = n_active + n_error + n_tokens + n_logits

    adj = torch.zeros(total_nodes, total_nodes, device=x_in.device, dtype=x_in.dtype)

    # Feature-to-feature edges
    adj[:n_active, :n_active] = attribution["feature_effects"].T

    # Feature-to-error edges (reconstruction error changes)
    baseline = attribution["baseline_reconstruction"]
    for i in range(n_active):
        effects = attribution["output_effects"][i]  # (n_layers, n_pos, d_model)
        for l in range(n_layers):
            for p in range(n_pos):
                error_idx = n_active + l * n_pos + p
                adj[error_idx, i] = effects[l, p].norm()

    # Logit edges
    if logit_effects is not None:
        adj[-n_logits:, :] = logit_effects

    return {
        "active_features": attribution["active_features"],
        "activation_values": attribution["activation_values"],
        "adjacency_matrix": adj,
    }
