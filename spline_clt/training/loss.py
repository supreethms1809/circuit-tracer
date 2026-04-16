"""Loss functions for Spline-CLT training.

Implements the same training objective as Anthropic's CLT:
    L_total = L_MSE + L_sparsity

Where:
    L_MSE = Σ_l ||y_hat^l - y^l||^2      (MLP output reconstruction)
    L_sparsity = λ Σ tanh(c · ||W_dec_i|| · a_i)  (sparsity regularization)
"""

import torch

from spline_clt.kan_transcoder import KANCrossLayerTranscoder


def reconstruction_loss(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """MSE reconstruction loss between predicted and true MLP outputs.

    Args:
        y_hat: Predicted outputs, shape (n_layers, n_pos, d_model).
        y_true: True MLP outputs, shape (n_layers, n_pos, d_model).

    Returns:
        Scalar MSE loss averaged across all dimensions.
    """
    return ((y_hat.float() - y_true.float()) ** 2).mean()


def paper_style_reconstruction_sparsity_metrics(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    activations: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Scalars for comparing runs to CLT reporting (e.g. attribution-graph papers).

    Anthropic quote *normalized mean reconstruction error* and *L0* (features
    active per input token) without always specifying the exact formula in-line.
    We log several unambiguous quantities so you can align with their appendix
    or reproduce their definition once pinned down.

    Args:
        y_hat: Predicted MLP outputs, (n_layers, n_pos, d_model).
        y_true: Ground-truth MLP outputs, same shape.
        activations: Post–JumpReLU features, (n_layers, n_pos, d_transcoder).

    Returns:
        Flat dict suitable for tqdm / W&B (all plain floats).
    """
    yf = y_hat.detach().float()
    y = y_true.detach().float()
    n_elem = y.numel()
    mse_mean = ((yf - y) ** 2).mean()
    diff = yf - y
    rel_fro = diff.norm() / (y.norm() + eps)
    # NMSE: ratio of mean squared error to mean squared target (scale-invariant).
    mean_y2 = (y**2).mean().clamp_min(eps)
    nmse_mean = mse_mean / mean_y2

    active = (activations > 0).to(torch.float32)
    # Total active latents summed over all layers at each position, then mean over tokens.
    l0_per_token = active.sum(dim=(0, 2)).mean()

    return {
        "reconstruction/mse_sum": float((mse_mean * n_elem).item()),
        "reconstruction/rel_fro_error": float(rel_fro.item()),
        "reconstruction/nmse_mean": float(nmse_mean.item()),
        "stats/l0_active_features_per_token": float(l0_per_token.item()),
    }


def sparsity_loss(
    activations: torch.Tensor,
    decoder_norms: list[torch.Tensor],
    lambda_: float = 0.05,
    c: float = 1.0,
) -> torch.Tensor:
    """Sparsity regularization loss matching Anthropic's formulation.

    L_sparsity = λ Σ tanh(c · ||W_dec_i|| · a_i)

    This encourages features to be sparse (few active per token) while accounting
    for the magnitude of decoder vectors (larger decoders penalized more).

    Args:
        activations: Feature activations, shape (n_layers, n_pos, d_transcoder).
            Should be post-activation (after JumpReLU/ReLU).
        decoder_norms: List of decoder weight norms per layer. Each element has
            shape (d_transcoder,) giving the L2 norm of each feature's decoder.
        lambda_: Sparsity coefficient.
        c: Scaling factor inside tanh.

    Returns:
        Scalar sparsity loss.
    """
    total = torch.tensor(0.0, device=activations.device, dtype=torch.float32)

    for layer_id in range(activations.shape[0]):
        # decoder_norms[layer_id]: (d_transcoder,)
        # activations[layer_id]: (n_pos, d_transcoder)
        weighted = c * decoder_norms[layer_id].unsqueeze(0) * activations[layer_id].float()
        total = total + torch.tanh(weighted).sum()

    return lambda_ * total / activations.shape[1]  # normalize by n_pos


def compute_decoder_norms(model: KANCrossLayerTranscoder) -> list[torch.Tensor]:
    """Compute L2 norms of decoder vectors for each feature.

    For cross-layer decoders, uses the maximum norm across target layers.

    Args:
        model: The Spline-CLT model.

    Returns:
        List of tensors, one per layer, each of shape (d_transcoder,).
    """
    norms = []
    for layer_id in range(model.n_layers):
        # W_dec[layer_id]: (d_transcoder, n_remaining_layers, d_model)
        w = model.W_dec[layer_id]
        # Norm across d_model, max across target layers
        layer_norms = w.norm(dim=-1).max(dim=-1).values  # (d_transcoder,)
        norms.append(layer_norms.detach())
    return norms


def total_loss(
    model: KANCrossLayerTranscoder,
    x_in: torch.Tensor,
    y_true: torch.Tensor,
    lambda_sparsity: float = 0.05,
    c_sparsity: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute total training loss.

    Args:
        model: The Spline-CLT model.
        x_in: Residual stream inputs, shape (n_layers, n_pos, d_model).
        y_true: True MLP outputs, shape (n_layers, n_pos, d_model).
        lambda_sparsity: Sparsity loss coefficient.
        c_sparsity: Sparsity scaling factor.

    Returns:
        Tuple of (total_loss, metrics_dict) where metrics_dict contains
        individual loss components for logging.
    """
    # Forward pass — use dense decode to avoid OOM from materializing
    # (n_active * n_layers * d_model) during early training when sparsity is low
    activations = model.encode(x_in)
    y_hat = model.decode_dense(activations, input_acts=x_in)

    # Reconstruction loss
    l_recon = reconstruction_loss(y_hat, y_true)

    # Sparsity loss
    dec_norms = compute_decoder_norms(model)
    l_sparse = sparsity_loss(activations, dec_norms, lambda_sparsity, c_sparsity)

    # KAN regularization: L1 on spline weights (KAN encoder only).
    # We do NOT use KANLinear.regularization_loss() because its entropy branch
    # computes p * log(p) which evaluates to 0 * -inf = NaN when spline_weight is all
    # zeros (which happens after update_grid when the old grid didn't cover the data).
    # Even with regularize_entropy=0.0, Python's 0.0 * nan = nan in IEEE 754.
    # Linear encoder has no spline weights, so this term is skipped.
    l_kan_reg = torch.tensor(0.0, device=x_in.device, dtype=torch.float32)
    if model.encoder_type == "kan":
        for encoder in model.encoders:
            l_kan_reg = l_kan_reg + encoder.kan_linear.spline_weight.abs().mean()
        l_kan_reg = 0.01 * l_kan_reg

    l_total = l_recon + l_sparse + l_kan_reg

    paper_metrics = paper_style_reconstruction_sparsity_metrics(y_hat, y_true, activations)
    # Mean over (layer, pos) of active count in that layer at that position (not paper L0).
    active_per_layer_pos = (activations > 0).float().sum(dim=-1).mean().item()

    metrics = {
        "loss/total": l_total.item(),
        "loss/reconstruction": l_recon.item(),
        "loss/sparsity": l_sparse.item(),
        "loss/kan_regularization": l_kan_reg.item(),
        "stats/active_features_per_pos": active_per_layer_pos,
        **paper_metrics,
    }

    return l_total, metrics
