"""Linear encoder module — baseline replacement for KANEncoder.

Implements the standard CLT encoder: pre_activations = W_enc @ x (bias-free
since b_enc is held in the transcoder). Provides the same interface as
KANEncoder so the two can be swapped transparently in KANCrossLayerTranscoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearEncoder(nn.Module):
    """Linear encoder for cross-layer transcoder features.

    Computes feature pre-activations as W_enc @ x — exactly the standard CLT
    encoder. Used as a baseline to compare against KANEncoder.

    Args:
        d_model: Dimension of the residual stream (e.g., 768 for GPT-2).
        n_features: Number of transcoder features to extract.
    """

    def __init__(self, d_model: int, n_features: int):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.W_enc = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.kaiming_uniform_(self.W_enc)

    @property
    def n_parameters(self) -> int:
        return self.W_enc.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute feature pre-activations.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Feature pre-activations of shape (..., n_features).
        """
        return F.linear(x.to(self.W_enc.dtype), self.W_enc).to(x.dtype)

    def get_encoder_vectors(
        self, x: torch.Tensor, active_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return W_enc rows for active features (exact, no Jacobian needed).

        Args:
            x: Input tensor of shape (n_pos, d_model). Unused for linear encoder.
            active_mask: Boolean or sparse mask of active features.

        Returns:
            Encoder vectors of shape (n_active, d_model).
        """
        if active_mask.is_sparse:
            _, feat_idx = active_mask.coalesce().indices()
        else:
            _, feat_idx = active_mask.nonzero(as_tuple=True)

        if len(feat_idx) == 0:
            return torch.zeros(0, self.d_model, device=x.device, dtype=x.dtype)

        return self.W_enc[feat_idx].to(x.dtype)

    def get_encoder_vectors_fast(
        self, x: torch.Tensor, active_mask: torch.Tensor
    ) -> torch.Tensor:
        """Same as get_encoder_vectors — linear encoder has exact directions."""
        return self.get_encoder_vectors(x, active_mask)

    def update_grid(self, x: torch.Tensor) -> None:
        """No-op — linear encoder has no adaptive grid."""
        pass
