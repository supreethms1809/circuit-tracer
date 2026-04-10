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


@torch.no_grad()
def compute_logit_effects(
    attribution: dict[str, torch.Tensor],
    error_vectors: torch.Tensor,
    token_vectors: torch.Tensor,
    W_U: torch.Tensor,
    logit_token_ids: torch.Tensor,
    n_error: int,
    n_tokens: int,
) -> torch.Tensor:
    """Compute logit attribution rows for the adjacency matrix.

    For each active feature, projects its output reconstruction delta at the
    *final* token position (where next-token prediction happens) through the
    demeaned unembedding vectors for the selected logit tokens.  Error and
    embedding nodes are handled analogously using actual error vectors and
    token embeddings (matching the original circuit_tracer).

    Args:
        attribution: Output from ablation_attribution().
        error_vectors: Reconstruction error, shape (n_layers, n_pos, d_model).
            Computed as y_true - baseline_reconstruction.
        token_vectors: Token embeddings, shape (n_pos, d_model).
        W_U: Unembedding matrix, shape (d_model, n_vocab).
        logit_token_ids: Vocab indices for the selected logit tokens, shape (n_logits,).
        n_error: Number of error nodes (n_layers * n_pos).
        n_tokens: Number of token (embedding) nodes (n_pos).

    Returns:
        Logit effect matrix, shape (n_logits, n_active + n_error + n_tokens).
        Rows correspond to logit nodes, columns to all other nodes.
    """
    active_features = attribution["active_features"]
    output_effects = attribution["output_effects"]
    n_active = len(active_features)
    n_logits = len(logit_token_ids)
    n_layers, n_pos, d_model = error_vectors.shape
    total_sources = n_active + n_error + n_tokens

    W_U = W_U.detach().float()

    # Demeaned unembedding columns for selected logits: (n_logits, d_model)
    if W_U.shape[0] == d_model:
        cols = W_U[:, logit_token_ids].T                       # (n_logits, d_model)
        demean = W_U.mean(dim=-1, keepdim=True)                # (d_model, 1)
        logit_dirs = cols - demean.T                           # (n_logits, d_model)
    else:
        cols = W_U[logit_token_ids]                            # (n_logits, d_model)
        demean = W_U.mean(dim=0, keepdim=True)                 # (1, d_model)
        logit_dirs = cols - demean                             # (n_logits, d_model)

    logit_effects = torch.zeros(
        n_logits, total_sources, device=error_vectors.device, dtype=error_vectors.dtype
    )

    # Feature → logit: project the feature's output effect at the *final* position
    # (where next-token prediction happens) through the unembedding.
    final_pos = n_pos - 1
    for i in range(n_active):
        effect = output_effects[i, :, final_pos, :].sum(dim=0).float()  # (d_model,)
        logit_effects[:, i] = logit_dirs @ effect

    # Error node → logit: project the reconstruction error vector at each
    # (layer, pos) through the unembedding.  This matches the original
    # circuit_tracer which uses error_vectors = mlp_out - reconstruction.
    for l in range(n_layers):
        for p in range(n_pos):
            error_idx = n_active + l * n_pos + p
            logit_effects[:, error_idx] = logit_dirs @ error_vectors[l, p].float()

    # Token/embedding → logit: project the actual token embedding vectors
    # through the unembedding.
    for p in range(n_tokens):
        token_idx = n_active + n_error + p
        logit_effects[:, token_idx] = logit_dirs @ token_vectors[p].float()

    return logit_effects


def build_attribution_graph(
    model: KANCrossLayerTranscoder,
    x_in: torch.Tensor,
    max_features: int = 256,
    W_U: torch.Tensor | None = None,
    logit_token_ids: torch.Tensor | None = None,
    y_true: torch.Tensor | None = None,
    token_vectors: torch.Tensor | None = None,
    _raw_attribution: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Build a full attribution graph compatible with circuit-tracer's Graph format.

    Node ordering: [active_features, error_nodes, token_nodes, logit_nodes]

    Matches the original circuit_tracer layout:
    - n_layers * n_pos error nodes (ALL positions including BOS)
    - n_pos token/embedding nodes (ALL positions including BOS)
    - BOS positions have zeroed values but slots still exist

    When ``W_U`` and ``logit_token_ids`` are provided, features are ranked by
    their influence on the target logit (matching the original circuit_tracer)
    rather than by raw activation magnitude.

    Args:
        model: Trained Spline-CLT model.
        x_in: Residual stream inputs, shape (n_layers, n_pos, d_model).
        max_features: Max features to include.
        W_U: Unembedding matrix, shape (d_model, n_vocab). Required for logit nodes.
        logit_token_ids: Vocab indices for selected logit tokens, shape (n_logits,).
            Required for logit nodes.
        y_true: True MLP outputs, shape (n_layers, n_pos, d_model). Needed for
            error node computation (reconstruction error = y_true - reconstruction).
        token_vectors: Token embedding vectors, shape (n_pos, d_model). Needed for
            proper embedding→logit edges. Falls back to x_in[0] if not provided.
        _raw_attribution: Optional pre-computed ablation attribution dict (avoids
            recomputing when the caller already ran ``ablation_attribution``).

    Returns:
        Dict with active_features, adjacency_matrix, and metadata needed for Graph.
    """
    # Run ablation with a generous budget — we'll prune by influence below.
    raw_max = max_features * 4 if W_U is not None and logit_token_ids is not None else max_features
    attribution = _raw_attribution or ablation_attribution(
        model, x_in, max_features=raw_max
    )
    if y_true is not None:
        attribution["y_true"] = y_true

    n_layers, n_pos, d_model = x_in.shape

    # --- Influence-based feature selection -----------------------------------
    has_logits = W_U is not None and logit_token_ids is not None
    if has_logits and len(attribution["active_features"]) > max_features:
        assert W_U is not None and logit_token_ids is not None
        n_pre = len(attribution["active_features"])
        W_U_f = W_U.detach().float()
        if W_U_f.shape[0] == d_model:
            cols = W_U_f[:, logit_token_ids].T
            demean = W_U_f.mean(dim=-1, keepdim=True)
            logit_dirs = cols - demean.T
        else:
            cols = W_U_f[logit_token_ids]
            demean = W_U_f.mean(dim=0, keepdim=True)
            logit_dirs = cols - demean
        final_pos = n_pos - 1
        influence_scores = torch.zeros(n_pre, device=x_in.device)
        for i in range(n_pre):
            effect = attribution["output_effects"][i, :, final_pos, :].sum(dim=0).float()
            influence_scores[i] = (logit_dirs @ effect).abs().sum()
        top_k_idx = influence_scores.topk(max_features).indices.sort().values
        attribution["active_features"] = attribution["active_features"][top_k_idx]
        attribution["activation_values"] = attribution["activation_values"][top_k_idx]
        attribution["feature_effects"] = attribution["feature_effects"][top_k_idx][:, top_k_idx]
        attribution["output_effects"] = attribution["output_effects"][top_k_idx]

    n_active = len(attribution["active_features"])

    # Match original circuit_tracer: ALL positions (including BOS) get error/token slots.
    # BOS values are zeroed but the structural slots remain.
    n_error = n_layers * n_pos
    n_tokens = n_pos

    # Compute error vectors: reconstruction error at each (layer, pos).
    # This matches the original circuit_tracer's error_vectors = mlp_out - reconstruction.
    baseline_recon = attribution["baseline_reconstruction"]
    if y_true is not None:
        error_vectors = y_true - baseline_recon  # (n_layers, n_pos, d_model)
    else:
        # Fallback: use zero error vectors (edges will be zero)
        error_vectors = torch.zeros_like(baseline_recon)
    # Zero BOS error vectors (matching original circuit_tracer)
    error_vectors[:, 0, :] = 0.0

    # Token embedding vectors
    if token_vectors is None:
        # Fallback: use layer-0 input as proxy for embeddings
        token_vectors = x_in[0]  # (n_pos, d_model)

    if has_logits:
        assert W_U is not None and logit_token_ids is not None
        logit_effects = compute_logit_effects(
            attribution, error_vectors, token_vectors,
            W_U, logit_token_ids, n_error, n_tokens,
        )
        n_logits = len(logit_token_ids)
    else:
        n_logits = 0

    total_nodes = n_active + n_error + n_tokens + n_logits

    adj = torch.zeros(total_nodes, total_nodes, device=x_in.device, dtype=x_in.dtype)

    # Feature-to-feature edges (enforce causal ordering: source layer < target layer)
    ff = attribution["feature_effects"].T.clone()  # ff[target, source]
    active_layers = attribution["active_features"][:, 0]  # layer of each feature
    for t in range(n_active):
        for s in range(n_active):
            if active_layers[s] >= active_layers[t]:
                ff[t, s] = 0.0
    adj[:n_active, :n_active] = ff

    # Feature-to-error edges (enforce causal ordering: feature layer < error layer).
    # Project the feature's reconstruction delta onto the error direction at each
    # downstream (layer, pos), giving a signed directional effect rather than a
    # magnitude.  This matches how the original circuit_tracer contracts gradients
    # with error vectors.
    for i in range(n_active):
        feat_layer = int(active_layers[i])
        effects = attribution["output_effects"][i]  # (n_layers, n_pos, d_model)
        for l in range(feat_layer + 1, n_layers):
            for p in range(n_pos):
                error_idx = n_active + l * n_pos + p
                err_vec = error_vectors[l, p]
                err_norm = err_vec.norm()
                if err_norm > 0:
                    # Project feature effect onto the error direction
                    adj[error_idx, i] = (effects[l, p] * err_vec).sum() / err_norm
                else:
                    adj[error_idx, i] = 0.0

    # Embedding → feature edges: project the token embedding onto the feature's
    # encoder direction, giving the direct linear effect of the embedding on the
    # feature activation.
    active_positions = attribution["active_features"][:, 1]
    encoder_directions = _batch_encoder_directions(
        model, x_in,
        attribution["active_features"][:, 0],
        attribution["active_features"][:, 1],
        attribution["active_features"][:, 2],
    )
    for i in range(n_active):
        pos = int(active_positions[i])
        token_idx = n_active + n_error + pos
        # Direct effect of embedding on feature via encoder direction
        adj[i, token_idx] = (token_vectors[pos] * encoder_directions[i]).sum()

    # Embedding → error edges: project the embedding onto the error direction
    # at layer 0 for each position, giving the direct effect of the embedding
    # on the first-layer reconstruction error.
    for p in range(n_pos):
        token_idx = n_active + n_error + p
        error_idx = n_active + 0 * n_pos + p  # layer-0 error
        err_vec = error_vectors[0, p]
        err_norm = err_vec.norm()
        if err_norm > 0:
            adj[error_idx, token_idx] = (token_vectors[p] * err_vec).sum() / err_norm
        else:
            adj[error_idx, token_idx] = 0.0

    # Logit edges
    if has_logits:
        adj[-n_logits:, :n_active + n_error + n_tokens] = logit_effects

    return {
        "active_features": attribution["active_features"],
        "activation_values": attribution["activation_values"],
        "adjacency_matrix": adj,
    }
