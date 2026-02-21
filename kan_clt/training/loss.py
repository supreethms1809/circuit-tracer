"""Loss functions for KAN-CLT training.

Implements the same training objective as Anthropic's CLT:
    L_total = L_MSE + L_sparsity

Where:
    L_MSE = Σ_l ||y_hat^l - y^l||^2      (MLP output reconstruction)
    L_sparsity = λ Σ tanh(c · ||W_dec_i|| · a_i)  (sparsity regularization)
"""

import torch

from kan_clt.kan_transcoder import KANCrossLayerTranscoder


def reconstruction_loss(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """MSE reconstruction loss between predicted and true MLP outputs.

    Args:
        y_hat: Predicted outputs, shape (n_layers, n_pos, d_model).
        y_true: True MLP outputs, shape (n_layers, n_pos, d_model).

    Returns:
        Scalar MSE loss summed across layers.
    """
    return ((y_hat - y_true) ** 2).sum(dim=-1).mean()


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
    total = torch.tensor(0.0, device=activations.device, dtype=activations.dtype)

    for layer_id in range(activations.shape[0]):
        # decoder_norms[layer_id]: (d_transcoder,)
        # activations[layer_id]: (n_pos, d_transcoder)
        weighted = c * decoder_norms[layer_id].unsqueeze(0) * activations[layer_id]
        total = total + torch.tanh(weighted).sum()

    return lambda_ * total / activations.shape[1]  # normalize by n_pos


def compute_decoder_norms(model: KANCrossLayerTranscoder) -> list[torch.Tensor]:
    """Compute L2 norms of decoder vectors for each feature.

    For cross-layer decoders, uses the maximum norm across target layers.

    Args:
        model: The KAN-CLT model.

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
        model: The KAN-CLT model.
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

    # KAN regularization (L1 on spline weights + entropy)
    l_kan_reg = torch.tensor(0.0, device=x_in.device)
    for encoder in model.encoders:
        l_kan_reg = l_kan_reg + encoder.kan_linear.regularization_loss(
            regularize_activation=0.01, regularize_entropy=0.01
        )

    l_total = l_recon + l_sparse + l_kan_reg

    metrics = {
        "loss/total": l_total.item(),
        "loss/reconstruction": l_recon.item(),
        "loss/sparsity": l_sparse.item(),
        "loss/kan_regularization": l_kan_reg.item(),
        "stats/active_features_per_pos": (activations > 0).float().sum(dim=-1).mean().item(),
    }

    return l_total, metrics
