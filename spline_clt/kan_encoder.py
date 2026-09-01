"""KAN encoder module wrapping efficient-kan's KANLinear for nonlinear feature detection.

Replaces the linear encoder (W_enc @ x + b_enc) in Anthropic's CLT with a KAN layer
that applies B-spline basis expansion before linear combination. The nonlinearity is in
how pre-activations are computed from the residual stream, not in the decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficient_kan import KANLinear


def unwrap_encoder_module(enc: nn.Module) -> nn.Module:
    """Return the underlying ``KANEncoder``, unwrapping an inner FSDP if present."""
    inner = getattr(enc, "_fsdp_wrapped_module", None)
    if inner is not None:
        return inner
    module = getattr(enc, "module", None)
    if module is not None and module is not enc and hasattr(module, "kan_linear"):
        return module
    return enc


def kan_linear_from_encoder(enc: nn.Module) -> KANLinear:
    """Access ``KANLinear`` through a possibly FSDP-wrapped encoder."""
    return unwrap_encoder_module(enc).kan_linear


class KANEncoder(nn.Module):
    """KAN-based encoder for cross-layer transcoder features.

    Uses B-spline basis expansion (via efficient-kan) to compute feature pre-activations
    from residual stream activations. Supports Jacobian computation for attribution.

    The forward pass computes:
        base_output = base_weight @ base_activation(x)
        spline_output = spline_weight @ B_spline_basis(x)
        output = base_output + spline_output

    Args:
        d_model: Dimension of the residual stream (e.g., 768 for GPT-2).
        n_features: Number of transcoder features to extract.
        grid_size: Number of grid intervals for B-spline basis. Default 5.
        spline_order: Order of B-spline basis functions. Default 3.
        grid_range: Range of the input grid. Default [-1, 1].
        base_activation: Activation applied to input before base linear map.
        scale_base: Init scale for ``base_weight`` (efficient-kan). Default 1.0.
        scale_spline: Init scale for spline path / ``spline_scaler``. Default 1.0.
    """

    def __init__(
        self,
        d_model: int,
        n_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: list[float] | None = None,
        base_activation: type[nn.Module] = nn.SiLU,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_base = scale_base
        self.scale_spline = scale_spline

        if grid_range is None:
            grid_range = [-1.0, 1.0]

        self.kan_linear = KANLinear(
            in_features=d_model,
            out_features=n_features,
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation=base_activation,
            grid_range=grid_range,
            scale_base=scale_base,
            scale_spline=scale_spline,
        )

    @property
    def basis_expansion_dim(self) -> int:
        """Effective dimensionality after B-spline basis expansion."""
        return self.d_model * (self.grid_size + self.spline_order)

    @property
    def n_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def forward_split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(base, spline)`` pre-activations matching ``forward`` summands.

        Used by BaseJump: gate on ``base``, magnitude from ``base + spline``.
        """
        kl = self.kan_linear
        original_shape = x.shape
        xf = x.float().reshape(-1, self.d_model)
        base = F.linear(kl.base_activation(xf), kl.base_weight)
        spline = F.linear(
            kl.b_splines(xf).view(xf.size(0), -1),
            kl.scaled_spline_weight.view(kl.out_features, -1),
        )
        out_shape = (*original_shape[:-1], self.n_features)
        return base.reshape(*out_shape).to(x.dtype), spline.reshape(*out_shape).to(x.dtype)

    def forward(self, x: torch.Tensor, return_split: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute feature pre-activations from residual stream activations.

        KANLinear parameters are kept in float32 for numerical stability (see
        KANCrossLayerTranscoder.__init__). Input is cast up to float32 for the
        B-spline computation and the result is cast back to the caller's dtype.

        Under nested FSDP the encoder is wrapped such that only ``forward`` /
        ``__call__`` triggers unshard hooks. Callers that need ``(base, spline)``
        for BaseJump must pass ``return_split=True`` here rather than invoking
        ``forward_split`` on the FSDP module (that would bypass unshard).

        Args:
            x: Input tensor of shape (..., d_model).
            return_split: If True, return ``(base, spline)`` pre-activations
                instead of their sum.

        Returns:
            Feature pre-activations of shape (..., n_features), or a
            ``(base, spline)`` tuple when ``return_split`` is True.
        """
        if return_split:
            return self.forward_split(x)
        return self.kan_linear(x.float()).to(x.dtype)

    def get_encoder_vectors(
        self, x: torch.Tensor, active_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute Jacobian rows for active features only (for attribution).

        For each active feature f at each position, computes the gradient of
        output[f] w.r.t. input x, giving the local linear encoder direction.
        This is the KAN equivalent of returning W_enc[feat_idx] in a linear encoder.

        Args:
            x: Input tensor of shape (n_pos, d_model).
            active_mask: Boolean mask or sparse tensor indicating active features.
                If sparse, shape (n_pos, n_features) with nonzero entries at active positions.

        Returns:
            Encoder vectors of shape (n_active, d_model) — one Jacobian row per active feature.
        """
        if active_mask.is_sparse:
            pos_idx, feat_idx = active_mask.coalesce().indices()
        else:
            pos_idx, feat_idx = active_mask.nonzero(as_tuple=True)

        if len(pos_idx) == 0:
            return torch.zeros(0, self.d_model, device=x.device, dtype=x.dtype)

        # Compute Jacobian rows via backward passes for each active feature.
        # We batch by unique positions to avoid redundant forward passes.
        # Cast to float32 to match KANLinear parameters (kept in float32
        # for numerical stability, see KANCrossLayerTranscoder.__init__).
        x_requires_grad = x.detach().float().requires_grad_(True)
        output = self.kan_linear(x_requires_grad)  # (n_pos, n_features)

        encoder_vecs = torch.zeros(
            len(pos_idx), self.d_model, device=x.device, dtype=x.dtype
        )

        # Group by unique positions for efficiency
        unique_positions = pos_idx.unique()
        for pos in unique_positions:
            mask = pos_idx == pos
            features_at_pos = feat_idx[mask]
            indices_at_pos = mask.nonzero(as_tuple=True)[0]

            for local_idx, (out_idx, f_idx) in enumerate(
                zip(indices_at_pos, features_at_pos)
            ):
                # Compute gradient of this specific feature output w.r.t. input
                if x_requires_grad.grad is not None:
                    x_requires_grad.grad.zero_()
                output[pos, f_idx].backward(retain_graph=True)
                encoder_vecs[out_idx] = x_requires_grad.grad[pos].to(x.dtype).clone()

        return encoder_vecs

    def get_encoder_vectors_fast(
        self, x: torch.Tensor, active_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute Jacobian rows using batched forward-mode AD (faster for many features).

        Uses torch.func.jacrev with vmap for efficient batched Jacobian computation.
        Falls back to get_encoder_vectors if torch.func is unavailable.

        Args:
            x: Input tensor of shape (n_pos, d_model).
            active_mask: Sparse tensor of shape (n_pos, n_features).

        Returns:
            Encoder vectors of shape (n_active, d_model).
        """
        if active_mask.is_sparse:
            pos_idx, feat_idx = active_mask.coalesce().indices()
        else:
            pos_idx, feat_idx = active_mask.nonzero(as_tuple=True)

        if len(pos_idx) == 0:
            return torch.zeros(0, self.d_model, device=x.device, dtype=x.dtype)

        # Use vmap + jacrev for batched Jacobian computation.
        # Cast to float32 to match KANLinear parameters.
        x_f32 = x.float()

        def single_input_forward(x_single: torch.Tensor) -> torch.Tensor:
            return self.kan_linear(x_single.unsqueeze(0)).squeeze(0)

        try:
            jacrev_fn = torch.func.jacrev(single_input_forward)

            encoder_vecs = []
            unique_positions = pos_idx.unique()
            for pos in unique_positions:
                # Full Jacobian at this position: (n_features, d_model)
                jac = jacrev_fn(x_f32[pos])
                mask = pos_idx == pos
                features_at_pos = feat_idx[mask]
                encoder_vecs.append(jac[features_at_pos].to(x.dtype))

            return torch.cat(encoder_vecs, dim=0)

        except (AttributeError, RuntimeError):
            # Fallback to backward-based computation
            return self.get_encoder_vectors(x, active_mask)

    def update_grid(self, x: torch.Tensor) -> None:
        """Update the B-spline grid based on observed activations.

        Should be called periodically during training to adapt the grid to the
        data distribution.  For wide models the efficient-kan lstsq inside
        curve2coeff allocates O(batch * in_features * out_features) floats,
        which can exceed GPU memory.

        This implementation never calls ``.cpu()`` / ``.to(device)`` on
        ``self.kan_linear`` itself — that would tear an inner-FSDP module off
        its flat-parameter buffers. Instead it fits on a detached CPU clone and
        copies the updated ``state_dict`` back onto the live parameters/buffers
        (which must already be fully materialized, e.g. under
        ``FSDP.summon_full_params(..., writeback=True)``).

        Args:
            x: Input tensor of shape (batch, d_model).
        """
        if x.dim() > 2:
            x = x.reshape(-1, self.d_model)

        x_cpu = x.detach().float().contiguous().cpu()
        kl = self.kan_linear
        # Side-channel CPU module: same hyperparameters, current weights.
        # grid_range is overwritten by load_state_dict (includes ``grid`` buffer).
        cpu_kl = KANLinear(
            in_features=kl.in_features,
            out_features=kl.out_features,
            grid_size=kl.grid_size,
            spline_order=kl.spline_order,
            scale_base=kl.scale_base,
            scale_spline=kl.scale_spline,
            enable_standalone_scale_spline=kl.enable_standalone_scale_spline,
            base_activation=type(kl.base_activation),
            grid_eps=kl.grid_eps,
        )
        cpu_kl.load_state_dict(
            {k: v.detach().cpu().clone() for k, v in kl.state_dict().items()}
        )
        cpu_kl.update_grid(x_cpu)
        updated = cpu_kl.state_dict()
        for name, tensor in kl.state_dict().items():
            tensor.copy_(updated[name].to(device=tensor.device, dtype=tensor.dtype))
