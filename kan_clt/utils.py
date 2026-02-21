"""Utility functions for KAN-CLT."""

import torch


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_breakdown(model: torch.nn.Module) -> dict[str, int]:
    """Count parameters broken down by component."""
    breakdown = {}
    for name, param in model.named_parameters():
        component = name.split(".")[0]
        if component not in breakdown:
            breakdown[component] = 0
        breakdown[component] += param.numel()
    return breakdown


def compare_parameter_counts(
    kan_clt: torch.nn.Module,
    d_model: int,
    d_transcoder: int,
    n_layers: int,
) -> dict[str, int]:
    """Compare KAN-CLT parameter count against equivalent linear CLT.

    Args:
        kan_clt: KAN-CLT model.
        d_model: Model dimension.
        d_transcoder: Number of features.
        n_layers: Number of layers.

    Returns:
        Dict with kan_params, linear_params, ratio.
    """
    kan_params = count_parameters(kan_clt)

    # Linear CLT encoder params: n_layers * d_transcoder * d_model (W_enc)
    #                           + n_layers * d_transcoder (b_enc)
    linear_enc_params = n_layers * (d_transcoder * d_model + d_transcoder)

    # Decoder params are the same for both
    linear_dec_params = sum(
        d_transcoder * (n_layers - i) * d_model for i in range(n_layers)
    )

    linear_total = linear_enc_params + linear_dec_params + n_layers * d_model  # b_dec

    return {
        "kan_params": kan_params,
        "linear_params": linear_total,
        "ratio": kan_params / linear_total,
    }
