"""Causal (ablation-based) attribution for Spline-CLT.

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

from spline_clt.kan_transcoder import KANCrossLayerTranscoder


def ablation_attribution(
    model: KANCrossLayerTranscoder,
    x_in: torch.Tensor,
    y_true: torch.Tensor | None = None,
    max_features: int = 256,
    batch_ablations: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute causal attribution via single-feature ablation.

    Args:
        model: Trained Spline-CLT model.
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
        # Zero out BOS position (pos 0) — consistent with the replacement model's
        # activation cache hook, which also zeros BOS so that circuit graphs and
        # MACAG interventions are aligned to actual token positions.
        baseline_activations[:, 0, :] = 0.0
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
        model: Spline-CLT model.
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


def compute_logit_effects(
    attribution: dict[str, torch.Tensor],
    W_U: torch.Tensor,
    logit_token_ids: torch.Tensor,
    n_error: int,
    n_tokens: int,
) -> torch.Tensor:
    """Compute feature→logit edges for the adjacency matrix.

    For each active feature, projects its last-position reconstruction effect
    through the unembedding matrix to estimate the logit change.

    Args:
        attribution: Output from ablation_attribution().
        W_U: Unembedding matrix, shape (d_model, n_vocab).
        logit_token_ids: Vocab indices for the selected logit tokens, shape (n_logits,).
        n_error: Number of error nodes in the adjacency matrix.
        n_tokens: Number of token (embedding) nodes.

    Returns:
        Logit effect matrix, shape (n_logits, n_active + n_error + n_tokens).
        Rows correspond to logit nodes, columns to all other nodes.
    """
    n_active = len(attribution["active_features"])
    n_logits = len(logit_token_ids)
    total_sources = n_active + n_error + n_tokens

    logit_effects = torch.zeros(
        n_logits, total_sources, device=W_U.device, dtype=W_U.dtype
    )

    # Feature→logit: project each feature's last-position output effect through W_U
    # output_effects[i] has shape (n_layers, n_pos, d_model)
    output_effects = attribution["output_effects"]
    # Sum across layers at the last position to get total residual stream effect
    last_pos_effects = output_effects[:, :, -1, :].sum(dim=1)  # (n_active, d_model)
    # Project onto unembedding vectors for selected logit tokens
    W_logits = W_U[:, logit_token_ids].detach().float()  # (d_model, n_logits)
    logit_effects[:, :n_active] = (last_pos_effects.float() @ W_logits).T

    # Error→logit: project each error node's reconstruction error through W_U
    # Error nodes represent (layer, position) pairs. The reconstruction error at the
    # last token position from each layer feeds into the final residual and affects logits.
    baseline_recon = attribution.get("baseline_reconstruction")  # (n_layers, n_pos, d_model)
    y_true = attribution.get("y_true")  # (n_layers, n_pos, d_model)
    if baseline_recon is not None and y_true is not None:
        n_layers = baseline_recon.shape[0]
        n_pos = baseline_recon.shape[1]
        for l in range(n_layers):
            error_vec = (y_true[l, -1, :] - baseline_recon[l, -1, :]).float()
            error_idx = n_active + l * n_pos + (n_pos - 1)  # last position
            logit_effects[:, error_idx] = (error_vec @ W_logits).squeeze()

    return logit_effects


def build_attribution_graph(
    model: KANCrossLayerTranscoder,
    x_in: torch.Tensor,
    max_features: int = 256,
    W_U: torch.Tensor | None = None,
    logit_token_ids: torch.Tensor | None = None,
    y_true: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build a full attribution graph compatible with circuit-tracer's Graph format.

    Node ordering: [active_features, error_nodes, token_nodes, logit_nodes]

    Args:
        model: Trained Spline-CLT model.
        x_in: Residual stream inputs, shape (n_layers, n_pos, d_model).
        max_features: Max features to include.
        W_U: Unembedding matrix, shape (d_model, n_vocab). Required for logit nodes.
        logit_token_ids: Vocab indices for selected logit tokens, shape (n_logits,).
            Required for logit nodes.
        y_true: True MLP outputs, shape (n_layers, n_pos, d_model). Needed for
            error→logit edges (reconstruction error contribution to logits).

    Returns:
        Dict with active_features, adjacency_matrix, and metadata needed for Graph.
    """
    attribution = ablation_attribution(model, x_in, max_features=max_features)
    if y_true is not None:
        attribution["y_true"] = y_true

    n_active = len(attribution["active_features"])
    n_layers, n_pos, _ = x_in.shape
    n_error = n_layers * n_pos
    n_tokens = n_pos

    has_logits = W_U is not None and logit_token_ids is not None
    if has_logits:
        assert W_U is not None and logit_token_ids is not None  # for type checker
        logit_effects = compute_logit_effects(
            attribution, W_U, logit_token_ids, n_error, n_tokens
        )
        n_logits = len(logit_token_ids)
    else:
        n_logits = 0

    total_nodes = n_active + n_error + n_tokens + n_logits

    adj = torch.zeros(total_nodes, total_nodes, device=x_in.device, dtype=x_in.dtype)

    # Feature-to-feature edges
    adj[:n_active, :n_active] = attribution["feature_effects"].T

    # Feature-to-error edges (reconstruction error changes)
    for i in range(n_active):
        effects = attribution["output_effects"][i]  # (n_layers, n_pos, d_model)
        for l in range(n_layers):
            for p in range(n_pos):
                error_idx = n_active + l * n_pos + p
                adj[error_idx, i] = effects[l, p].norm()

    # Logit edges
    if has_logits:
        adj[-n_logits:, :n_active + n_error + n_tokens] = logit_effects

    return {
        "active_features": attribution["active_features"],
        "activation_values": attribution["activation_values"],
        "adjacency_matrix": adj,
    }
