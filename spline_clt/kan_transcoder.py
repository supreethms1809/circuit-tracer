"""Spline Cross-Layer Transcoder — replaces linear encoder in CLT with KAN encoder.

Architecture:
    a^l = JumpReLU(KAN_enc^l(x^l))          # KAN encoder (nonlinear)
    y_hat^l = Σ W_dec^(l'→l) · a^(l')       # linear decoder (same as standard CLT)

The decoder MUST remain linear — features need clean directions in residual stream
space for activation patching and steering to work.

This module mirrors the interface of circuit_tracer.transcoder.CrossLayerTranscoder
so it can be used as a drop-in replacement in the attribution pipeline.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
from torch.utils.checkpoint import checkpoint

from circuit_tracer.transcoder.activation_functions import JumpReLU, jumprelu
from spline_clt.kan_encoder import KANEncoder
from spline_clt.linear_encoder import LinearEncoder


class KANCrossLayerTranscoder(nn.Module):
    """Cross-layer transcoder with pluggable encoders and linear decoders.

    Each layer has one encoder (KAN or linear) that reads from the residual
    stream and produces feature pre-activations. JumpReLU is applied for
    sparsity. Each feature writes to all subsequent layers via linear decoder
    matrices.

    Interface is compatible with circuit_tracer.transcoder.CrossLayerTranscoder
    so it can be used with the existing ReplacementModel and AttributionContext.

    Args:
        n_layers: Number of transformer layers.
        d_transcoder: Number of features per layer.
        d_model: Dimension of the residual stream.
        encoder_type: "kan" (default) or "linear" (baseline comparison).
        grid_size: KAN grid size for B-spline basis (ignored for linear).
        spline_order: KAN spline order (ignored for linear).
        activation_function: "jump_relu", "base_jump", or "relu".
            ``base_jump`` gates on the KAN base score and takes magnitude from
            ``relu(base + spline)`` (requires ``encoder_type="kan"``).
        skip_connection: Whether to include a learned skip connection.
        feature_input_hook: Hook point where features read from.
        feature_output_hook: Hook point where features write to.
        scan: Optional identifier for feature visualization.
    """

    def __init__(
        self,
        n_layers: int,
        d_transcoder: int,
        d_model: int,
        encoder_type: str = "kan",
        grid_size: int = 5,
        spline_order: int = 3,
        activation_function: str = "jump_relu",
        skip_connection: bool = False,
        feature_input_hook: str = "hook_resid_mid",
        feature_output_hook: str = "hook_mlp_out",
        scan: str | list[str] | None = None,
        threshold_init: float = 0.001,
        jumprelu_bandwidth: float = 0.001,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if encoder_type not in ("kan", "linear"):
            raise ValueError(f"encoder_type must be 'kan' or 'linear', got {encoder_type!r}")
        if activation_function not in ("jump_relu", "base_jump", "relu"):
            raise ValueError(
                f"Invalid activation function: {activation_function!r}"
            )
        if activation_function == "base_jump" and encoder_type != "kan":
            raise ValueError("activation_function='base_jump' requires encoder_type='kan'")

        self.n_layers = n_layers
        self.d_transcoder = d_transcoder
        self.d_model = d_model
        self.encoder_type = encoder_type
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.activation_function_name = activation_function
        self.feature_input_hook = feature_input_hook
        self.feature_output_hook = feature_output_hook
        self.skip_connection = skip_connection
        self.scan = scan

        # Encoders — one per layer.
        # KAN encoders are kept in float32 regardless of training dtype:
        #   KANLinear stores a `grid` buffer (B-spline knot positions); bfloat16
        #   causes adjacent knots to quantize identically → degenerate lstsq in
        #   update_grid → NaN spline weights. KANEncoder.forward() casts inputs
        #   to float32 and outputs back to model dtype.
        # Linear encoders follow the model dtype normally.
        if encoder_type == "kan":
            self.encoders = nn.ModuleList([
                KANEncoder(
                    d_model=d_model,
                    n_features=d_transcoder,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                )
                for _ in range(n_layers)
            ])
            self.encoders.to(device=device)  # device only; float32 dtype preserved
        else:
            self.encoders = nn.ModuleList([
                LinearEncoder(d_model=d_model, n_features=d_transcoder)
                for _ in range(n_layers)
            ])
            self.encoders.to(device=device, dtype=dtype)

        # Per-layer input normalization (standardization) statistics.
        # Raw GPT-2 residual-stream activations have wildly different per-layer
        # scales and heavy outliers (e.g. std ~10 with abs-max in the thousands).
        # Feeding them raw makes the B-spline grid_range=[-1, 1] meaningless and
        # lets the adaptive grid get stretched by outliers, collapsing spline
        # resolution where the data actually lives. We standardize the encoder
        # input per layer/dimension before the encoder so the spline operates in
        # a well-conditioned coordinate system. Buffers default to a no-op
        # (mean=0, std=1) so untrained models and legacy checkpoints behave
        # exactly as before; ``set_input_normalization`` populates them from data.
        self.register_buffer(
            "enc_input_mean", torch.zeros(n_layers, d_model, device=device, dtype=dtype)
        )
        self.register_buffer(
            "enc_input_std", torch.ones(n_layers, d_model, device=device, dtype=dtype)
        )

        # Encoder biases (applied after KAN forward, before activation)
        self.b_enc = nn.Parameter(
            torch.zeros(n_layers, d_transcoder, device=device, dtype=dtype)
        )

        # Decoder biases
        self.b_dec = nn.Parameter(
            torch.zeros(n_layers, d_model, device=device, dtype=dtype)
        )

        # Activation function
        if activation_function in ("jump_relu", "base_jump"):
            # LOG-SPACE, NON-NEGATIVE JumpReLU threshold. The module's `.threshold`
            # parameter stores log θ; the effective gate is θ = exp(log θ) > 0,
            # applied in apply_activation_function(). This guarantees the gate
            # x > θ passes only x > 0, so post-JumpReLU features are ≥ 0 and the
            # sparsity penalty λ·Σ tanh(c·‖W_dec‖·a) can never invert to a reward.
            #
            # An UNCONSTRAINED threshold (the old parametrization) could drift < 0:
            # once it did, negative pre-activations passed the gate as negative
            # activations, flipped the sparsity term negative, and drove total loss
            # below zero — a self-reinforcing collapse (linear arms hit 14–32% of
            # features with θ < 0). exp() bounds θ > 0 by construction; as the
            # optimizer pushes log θ → −∞ the feature asymptotes to a plain ReLU
            # (θ → 0⁺) instead of inverting. threshold_init stays the *effective*
            # init (stored as its log); it must be > 0.
            #
            # WARNING: never call self.activation_function.forward()/(...) — the
            # stock JumpReLU.forward gates on the raw stored value, which is now
            # log θ. Only apply_activation_function() (which exponentiates) is safe.
            # bandwidth MUST be on the scale of θ. The upstream default is 2,
            # which smears the STE window over (-1, 1) and, together with the
            # extra θ factor from the log parametrization, drove dL/d(log θ) to
            # ~1e-11 — below Adam's eps floor, freezing θ at its init and making
            # JumpReLU a no-op (the linear arms then never sparsified).
            #
            # base_jump: gate on KAN base score; magnitude = relu(base + spline).
            self.activation_function = JumpReLU(
                torch.full(
                    (n_layers, 1, d_transcoder),
                    math.log(float(threshold_init)),
                    device=device,
                    dtype=dtype,
                ),
                bandwidth=float(jumprelu_bandwidth),
            )
        elif activation_function == "relu":
            self.activation_function = nn.functional.relu
        else:
            raise ValueError(f"Invalid activation function: {activation_function}")

        # Cross-layer decoder weights: W_dec[i] has shape (d_transcoder, n_layers - i, d_model)
        # Feature at layer i can write to layers i, i+1, ..., n_layers-1
        self.W_dec = nn.ParameterList([
            nn.Parameter(
                torch.zeros(
                    d_transcoder, n_layers - i, d_model,
                    device=device, dtype=dtype,
                )
            )
            for i in range(n_layers)
        ])

        # Optional skip connection
        if skip_connection:
            self.W_skip = nn.Parameter(
                torch.zeros(n_layers, d_model, d_model, device=device, dtype=dtype)
            )
        else:
            self.W_skip = None

        self._init_decoder_weights()

    def _init_decoder_weights(self) -> None:
        """Initialize decoder weights with small random values.

        ``kaiming_uniform_`` normalizes by ``fan_in = d_model``, so ‖w_dec‖ ≈ √2
        for *every* base model no matter how large its MLP outputs are. Encoder
        inputs are normalized (``enc_input_std``) but the targets are not, so the
        initial ‖ŷ‖/‖y‖ ratio is decided entirely by the base model's output
        scale. See :meth:`scale_decoder_per_target_layer` and
        ``spline_clt.training.train.calibrate_decoder_scale_from_data``.
        """
        for w_dec in self.W_dec:
            nn.init.kaiming_uniform_(w_dec.view(-1, self.d_model))

    @torch.no_grad()
    def scale_decoder_per_target_layer(self, scales: torch.Tensor) -> None:
        """Multiply every decoder block that writes to layer ``l`` by ``scales[l]``.

        ``W_dec[source_l]`` has shape ``(d_transcoder, n_layers - source_l, d_model)``
        where index ``j`` writes to target layer ``source_l + j``. Scaling along
        that axis therefore scales ``y_hat[l]`` by exactly ``scales[l]``, whichever
        source layers happen to contribute to it.

        Args:
            scales: Per-target-layer multipliers, shape ``(n_layers,)``.
        """
        if scales.shape != (self.n_layers,):
            raise ValueError(
                f"scales must have shape ({self.n_layers},), "
                f"got {tuple(scales.shape)}."
            )
        for source_l in range(self.n_layers):
            w_dec = self.W_dec[source_l]
            block = scales[source_l:].to(device=w_dec.device, dtype=w_dec.dtype)
            w_dec.data.mul_(block[None, :, None])

    def to(self, *args, **kwargs) -> "KANCrossLayerTranscoder":
        """Move model to device/dtype, keeping KAN encoders in float32.

        KANLinear stores a grid buffer used in B-spline interpolation.
        Moving it to bfloat16 causes adjacent knot positions to collapse
        (bfloat16 resolution ~0.01), making the lstsq in update_grid
        degenerate and producing NaN spline weights. All other parameters
        (decoder, biases, thresholds) follow the requested dtype normally.
        """
        super().to(*args, **kwargs)
        if self.encoder_type == "kan":
            for enc in self.encoders:
                enc.kan_linear.to(dtype=torch.float32)
        return self

    @property
    def device(self) -> torch.device:
        return self.b_enc.device

    @property
    def dtype(self) -> torch.dtype:
        return self.b_enc.dtype

    @property
    def effective_threshold(self) -> torch.Tensor:
        """Non-negative JumpReLU gates ``θ = exp(log θ)``.

        The ``JumpReLU.threshold`` parameter stores log-thresholds in memory.
        On-disk ``threshold_{i}`` tensors are the literal effective values (see
        ``to_safetensors`` / ``load_spline_clt``). Shape: ``(n_layers, d_transcoder)``.
        """
        if not isinstance(self.activation_function, JumpReLU):
            raise AttributeError(
                "effective_threshold is only defined for jump_relu/base_jump activation"
            )
        return self.activation_function.threshold.squeeze(1).exp()

    def _normalize_input(self, x: torch.Tensor, layer_id: int) -> torch.Tensor:
        """Standardize a single layer's encoder input by stored mean/std.

        Computed in float32 (then cast back) so large per-dim means/outliers do
        not lose precision under bfloat16. With the default buffers (mean=0,
        std=1) this is an exact no-op.

        Args:
            x: Input tensor of shape (..., d_model).
            layer_id: Which layer's statistics to use.
        """
        mean = self.enc_input_mean[layer_id].float()
        std = self.enc_input_std[layer_id].float()
        return ((x.float() - mean) / std).to(x.dtype)

    @torch.no_grad()
    def set_input_normalization(
        self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-3
    ) -> None:
        """Populate the per-layer encoder-input normalization buffers.

        Args:
            mean: Per-layer per-dim means, shape (n_layers, d_model).
            std: Per-layer per-dim standard deviations, shape (n_layers, d_model).
                Values below ``eps`` are replaced by 1.0 to avoid amplifying
                (near-)constant dimensions.
        """
        mean = mean.to(self.enc_input_mean.device, self.enc_input_mean.dtype)
        std = std.to(self.enc_input_std.device, self.enc_input_std.dtype)
        std = torch.where(std < eps, torch.ones_like(std), std)
        self.enc_input_mean.copy_(mean)
        self.enc_input_std.copy_(std)

    def _layer_threshold(self, layer_id: int) -> torch.Tensor:
        """Effective JumpReLU threshold θ = exp(log θ) for one layer."""
        log_threshold = self.activation_function.threshold[layer_id].squeeze(0).clone()
        return log_threshold.exp()

    def apply_base_jump(
        self, layer_id: int, score: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """BaseJump: STE mask from ``score``, magnitude from ``relu(value)``.

        Matches ``spline_sae`` BaseJump: gate on the KAN base path so the spline
        can contribute magnitude without needing to also clear the threshold.
        """
        if not isinstance(self.activation_function, JumpReLU):
            raise RuntimeError("apply_base_jump requires a JumpReLU threshold module")
        threshold = self._layer_threshold(layer_id)
        bandwidth = float(self.activation_function.bandwidth)
        # Straight-through estimator on the hard gate (same as SAE JumpReLU.gated).
        hard = (score > threshold).to(dtype=score.dtype)
        soft = ((score - threshold) / bandwidth + 0.5).clamp(0.0, 1.0)
        mask = hard + soft - soft.detach()
        return mask * torch.relu(value)

    def apply_activation_function(
        self, layer_id: int, features: torch.Tensor
    ) -> torch.Tensor:
        """Apply the sparsity-inducing activation function.

        Args:
            layer_id: Which layer's activation function to use.
            features: Pre-activation values of shape (..., d_transcoder).
                For ``base_jump``, this is the gate score (base + bias); prefer
                ``apply_base_jump`` when score and value differ.

        Returns:
            Activated features (same shape).
        """
        if self.activation_function_name == "base_jump":
            # Score == value when only one tensor is provided (legacy path).
            return self.apply_base_jump(layer_id, features, features)
        if isinstance(self.activation_function, JumpReLU):
            # `.threshold` holds LOG-thresholds; the effective gate is
            # θ = exp(log θ) > 0, so activations are non-negative and the sparsity
            # penalty cannot invert (see __init__). The custom jumprelu autograd
            # function gives θ its surrogate gradient, which flows back through
            # exp() to the log-parameter. .clone() breaks FSDP's flat-param view
            # chain; .exp() is a differentiable forward op preserving that flow.
            threshold = self._layer_threshold(layer_id)
            return jumprelu.apply(features, threshold, self.activation_function.bandwidth)
        else:
            return self.activation_function(features)

    def encode_layer(
        self, x: torch.Tensor, layer_id: int, apply_activation_function: bool = True
    ) -> torch.Tensor:
        """Encode residual stream activations at a single layer.

        Args:
            x: Input tensor of shape (..., d_model).
            layer_id: Which layer to encode.
            apply_activation_function: Whether to apply JumpReLU/BaseJump/ReLU.
                When False under ``base_jump``, returns the **gate score**
                (base + bias) used for threshold calibration — not base+spline.

        Returns:
            Feature activations of shape (..., d_transcoder).
        """
        # Under FSDP, parameters are backed by a flat sharded buffer. index_select
        # + squeeze returns a view (ViewBackward0) of that buffer; FSDP modifies
        # the buffer in-place during reshard, which trips autograd's view-inplace
        # guard. .clone() breaks the view chain so the backward graph holds a
        # self-contained tensor, not a view of the FSDP flat param.
        x = self._normalize_input(x, layer_id)
        bias = self.b_enc[layer_id].clone()

        if self.activation_function_name == "base_jump":
            # Must go through encoder __call__/forward so nested FSDP unshards.
            # Direct forward_split() on an FSDP module bypasses unshard hooks.
            split_out = self.encoders[layer_id](x, return_split=True)
            base, spline = split_out
            score = base + bias
            if not apply_activation_function:
                return score
            value = score + spline
            return self.apply_base_jump(layer_id, score, value)

        features = self.encoders[layer_id](x) + bias
        if not apply_activation_function:
            return features
        return self.apply_activation_function(layer_id, features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode residual stream activations at all layers.

        During training with KAN encoders, each layer is gradient-checkpointed.
        BaseJump materializes ``scaled_spline_weight`` (≈0.42 GiB fp32/layer at
        Gemma-2-2B d_t=6144) for backward; without checkpointing that is held for
        *all* layers at once (~11 GiB) on top of FSDP's full-decoder unshard
        (~9.26 GiB bf16 triangular W_dec). Peak then sits at ~94 GiB of 142 GiB
        and the first post-collect backward OOMs after allocator fragmentation.

        Args:
            x: Input tensor of shape (n_layers, batch, d_model).

        Returns:
            Feature activations of shape (n_layers, batch, d_transcoder).
        """
        layer_features = []
        use_ckpt = (
            self.training
            and self.encoder_type == "kan"
            and torch.is_grad_enabled()
        )
        for layer_id in range(self.n_layers):
            x_layer = x[layer_id]
            if use_ckpt:
                # use_reentrant=False is required with FSDP + non-tensor args
                # (layer_id) and avoids the reentrant autograd edge cases.
                features = checkpoint(
                    self.encode_layer,
                    x_layer,
                    layer_id,
                    True,
                    use_reentrant=False,
                )
            else:
                features = self.encode_layer(x_layer, layer_id)
            layer_features.append(features)
        return torch.stack(layer_features)

    def encode_sparse(
        self, x: torch.Tensor, zero_positions: slice = slice(0, 1)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to sparse activations with Jacobian-based encoder vectors.

        Processes layers sequentially and converts to sparse format immediately
        for memory efficiency. Encoder vectors are computed as Jacobian rows of
        the KAN encoder (local linear approximation) for attribution.

        Args:
            x: Input tensor of shape (n_layers, n_pos, d_model).
            zero_positions: Positions to zero out (e.g., BOS token).

        Returns:
            sparse_features: Sparse tensor of shape (n_layers, n_pos, d_transcoder).
            active_encoders: Jacobian-based encoder vectors for active features,
                shape (total_active, d_model).
        """
        sparse_layers = []
        encoder_vectors = []

        for layer_id in range(self.n_layers):
            # Encoder vectors are computed against the *normalized* input (the
            # space the encoder actually sees). x_norm is reused for the
            # Jacobian below; encode_layer re-normalizes internally.
            x_norm = self._normalize_input(x[layer_id], layer_id)
            layer_features = self.encode_layer(x[layer_id], layer_id)
            layer_features[zero_positions] = 0

            sparse_layer = layer_features.to_sparse()
            sparse_layers.append(sparse_layer)

            # Compute Jacobian-based encoder vectors for active features
            if sparse_layer._nnz() > 0:
                active_mask = sparse_layer
                try:
                    enc_vecs = self.encoders[layer_id].get_encoder_vectors_fast(
                        x_norm, active_mask
                    )
                except RuntimeError:
                    enc_vecs = self.encoders[layer_id].get_encoder_vectors(
                        x_norm, active_mask
                    )
                # Chain rule: the encoder acts on x_norm = (x_raw - mean)/std, so
                # d(feature)/d(x_raw) = d(feature)/d(x_norm) * (1/std). Rescale so
                # the returned directions live in raw residual-stream coordinates
                # (what the attribution pipeline expects). For the linear encoder
                # this turns W_enc rows into the equivalent raw-space direction.
                inv_std = (1.0 / self.enc_input_std[layer_id]).to(enc_vecs.dtype)
                encoder_vectors.append(enc_vecs * inv_std)

        sparse_features = torch.stack(sparse_layers).coalesce()

        if encoder_vectors:
            active_encoders = torch.cat(encoder_vectors, dim=0)
        else:
            active_encoders = torch.zeros(0, self.d_model, device=x.device, dtype=x.dtype)

        return sparse_features, active_encoders

    def _get_decoder_vectors(
        self, layer_id: int, feat_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Get decoder weight vectors for a specific layer.

        Args:
            layer_id: Source layer index.
            feat_ids: Optional feature indices to select. If None, return all.

        Returns:
            Decoder vectors of shape (n_feats, n_remaining_layers, d_model).
        """
        if feat_ids is not None:
            return self.W_dec[layer_id][feat_ids]
        return self.W_dec[layer_id]

    def select_decoder_vectors(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select and scale decoder vectors for active features.

        Mirrors CrossLayerTranscoder.select_decoder_vectors exactly.

        Args:
            features: Sparse tensor of shape (n_layers, n_pos, d_transcoder).

        Returns:
            pos_ids: Position indices for each decoder vector.
            layer_ids: Target layer indices for each decoder vector.
            feat_ids: Feature indices.
            decoder_vectors: Scaled decoder vectors (activation * W_dec).
            encoder_mapping: Maps each decoder vector back to its source encoder index.
        """
        if not features.is_sparse:
            features = features.to_sparse()

        layer_idx, pos_idx, feat_idx = features.indices()
        activations = features.values()
        n_layers = features.shape[0]
        device = features.device

        pos_ids = []
        layer_ids = []
        feat_ids = []
        decoder_vectors = []
        encoder_mapping = []
        st = 0

        for layer_id in range(n_layers):
            current_layer = layer_idx == layer_id
            if not current_layer.any():
                continue

            current_layer_features = feat_idx[current_layer]
            unique_feats, inv = current_layer_features.unique(return_inverse=True)

            unique_decoders = self._get_decoder_vectors(layer_id, unique_feats)
            scaled_decoders = (
                unique_decoders[inv] * activations[current_layer, None, None]
            )
            decoder_vectors.append(scaled_decoders.reshape(-1, self.d_model))

            n_output_layers = self.n_layers - layer_id
            pos_ids.append(
                pos_idx[current_layer].repeat_interleave(n_output_layers)
            )
            feat_ids.append(
                current_layer_features.repeat_interleave(n_output_layers)
            )
            layer_ids.append(
                torch.arange(layer_id, self.n_layers, device=device).repeat(
                    len(current_layer_features)
                )
            )

            source_ids = torch.arange(len(current_layer_features), device=device) + st
            st += len(current_layer_features)
            encoder_mapping.append(
                torch.repeat_interleave(source_ids, n_output_layers)
            )

        pos_ids = torch.cat(pos_ids, dim=0)
        layer_ids = torch.cat(layer_ids, dim=0)
        feat_ids = torch.cat(feat_ids, dim=0)
        decoder_vectors = torch.cat(decoder_vectors, dim=0)
        encoder_mapping = torch.cat(encoder_mapping, dim=0)

        return pos_ids, layer_ids, feat_ids, decoder_vectors, encoder_mapping

    def compute_reconstruction(
        self,
        pos_ids: torch.Tensor,
        layer_ids: torch.Tensor,
        decoder_vectors: torch.Tensor,
        input_acts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct MLP outputs from decoder vectors.

        Args:
            pos_ids: Position indices for each decoder vector.
            layer_ids: Target layer indices for each decoder vector.
            decoder_vectors: Scaled decoder vectors.
            input_acts: Optional input activations for skip connection.

        Returns:
            Reconstructed MLP outputs of shape (n_layers, n_pos, d_model).
        """
        # Use input_acts.shape[1] when available: pos_ids.max()+1 can undercount
        # if ablation removes the only active feature at the last token position,
        # causing a shape mismatch between baseline and ablated reconstructions.
        if input_acts is not None:
            n_pos = input_acts.shape[1]
        elif pos_ids.numel() > 0:
            n_pos = int(pos_ids.max().item()) + 1
        else:
            n_pos = 0
        flat_idx = layer_ids * n_pos + pos_ids
        recon = torch.zeros(
            n_pos * self.n_layers,
            self.d_model,
            device=decoder_vectors.device,
            dtype=decoder_vectors.dtype,
        ).index_add_(0, flat_idx, decoder_vectors)
        recon = recon.reshape(self.n_layers, n_pos, self.d_model) + self.b_dec[:, None]

        if self.W_skip is not None:
            assert input_acts is not None, (
                "Transcoder has skip connection but no input_acts were provided"
            )
            recon = recon + input_acts @ self.W_skip

        return recon

    def decode(
        self, features: torch.Tensor, input_acts: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Decode features to reconstructed MLP outputs.

        Args:
            features: Sparse feature activations.
            input_acts: Optional input activations for skip connection.

        Returns:
            Reconstructed MLP outputs of shape (n_layers, n_pos, d_model).
        """
        pos_ids, layer_ids, feat_ids, decoder_vectors, _ = (
            self.select_decoder_vectors(features)
        )
        return self.compute_reconstruction(
            pos_ids, layer_ids, decoder_vectors, input_acts
        )

    def decode_dense(
        self, activations: torch.Tensor, input_acts: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Decode dense feature activations to reconstructed MLP outputs.

        Memory-efficient alternative to decode() for use during training when
        features are not yet sparse (JumpReLU threshold near zero). Uses einsum
        instead of materializing the full (n_active, n_layers, d_model) tensor.

        Args:
            activations: Dense feature activations of shape (n_layers, n_pos, d_transcoder).
            input_acts: Optional input activations for skip connection.

        Returns:
            Reconstructed MLP outputs of shape (n_layers, n_pos, d_model).
        """
        w_dtype = self.b_dec.dtype
        activations = activations.to(w_dtype)
        if input_acts is not None:
            input_acts = input_acts.to(w_dtype)

        n_layers, n_pos, _ = activations.shape
        y_hat = self.b_dec[:, None, :].expand(n_layers, n_pos, self.d_model).clone()

        for source_l in range(n_layers):
            # activations[source_l]: (n_pos, d_transcoder)
            # W_dec[source_l]:       (d_transcoder, n_remaining, d_model)
            # result:                (n_remaining, n_pos, d_model)
            contrib = torch.einsum(
                "pf,fld->lpd", activations[source_l], self.W_dec[source_l]
            )
            y_hat[source_l:] += contrib

        if self.W_skip is not None:
            assert input_acts is not None, (
                "Transcoder has skip connection but no input_acts were provided"
            )
            y_hat = y_hat + input_acts @ self.W_skip

        return y_hat

    def compute_skip(self, layer_id: int, inputs: torch.Tensor) -> torch.Tensor:
        """Compute skip connection output for a layer.

        Args:
            layer_id: Layer index.
            inputs: Input activations.

        Returns:
            Skip connection output.
        """
        if self.W_skip is not None:
            return inputs @ self.W_skip[layer_id]
        raise ValueError("Transcoder has no skip connection")

    def forward(
        self, x_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """FSDP-compatible forward: encode + decode_dense + decoder norms.

        FSDP only unshards flat parameters during module.forward() hook calls.
        Calling encode() or decode_dense() directly bypasses these hooks and
        sees sharded/flat params. This method exists so model_for_train(x_in)
        triggers FSDP's unshard/reshard cycle with all FSDP-managed parameter
        accesses (b_enc, W_dec, threshold) inside the guarded window.

        Uses decode_dense (not the sparse decode path) to avoid OOM early in
        training when JumpReLU threshold is near zero and all features are active.
        Under the replicated-KAN FSDP path, encoder spline weights live in
        ``ignored_modules``; under ``shard_kan_encoders``, each encoder is its
        own nested FSDP unit. Either way ``compute_losses`` reaches them via
        ``kan_linear_from_encoder``.

        Returns:
            activations: (n_layers, n_pos, d_transcoder)
            y_hat:       (n_layers, n_pos, d_model)
            dec_norms:   list of (d_transcoder,) tensors, one per source layer
        """
        activations = self.encode(x_in)
        y_hat = self.decode_dense(activations, input_acts=x_in)
        dec_norms = [
            self.W_dec[l].norm(dim=-1).max(dim=-1).values.detach()
            for l in range(self.n_layers)
        ]
        return activations, y_hat, dec_norms

    @torch.no_grad()
    def spline_contribution_fraction(
        self, x: torch.Tensor, max_pos: int | None = 512
    ) -> float:
        """Diagnostic: mean over layers of ||spline_output|| / ||encoder_output||.

        Measures how much the B-spline path moves the encoder pre-activation
        relative to the linear/SiLU base path. ~0 means the KAN is behaving like
        its linear base (the nonlinearity is unused — the thesis-critical failure
        mode); larger means the spline is doing real work. Computed on the
        normalized input (the space the encoder sees), pre-bias/pre-activation.

        Returns NaN for the linear encoder (no spline path). Subsamples positions
        to ``max_pos`` so it is cheap enough to call at every log step.

        Args:
            x: Input of shape (n_layers, n_pos, d_model).
            max_pos: Cap on positions used for the estimate (None = all).
        """
        if self.encoder_type != "kan":
            return float("nan")
        import torch.nn.functional as F

        fracs: list[float] = []
        for layer_id in range(self.n_layers):
            xb = self._normalize_input(x[layer_id], layer_id).float()
            if max_pos is not None and xb.shape[0] > max_pos:
                xb = xb[:max_pos]
            kl = self.encoders[layer_id].kan_linear
            base = F.linear(kl.base_activation(xb), kl.base_weight)
            spline = F.linear(
                kl.b_splines(xb).view(xb.size(0), -1),
                kl.scaled_spline_weight.view(kl.out_features, -1),
            )
            denom = (base + spline).norm()
            if denom > 0:
                fracs.append((spline.norm() / denom).item())
        return float(sum(fracs) / len(fracs)) if fracs else float("nan")

    def compute_attribution_components(
        self, inputs: torch.Tensor, zero_positions: slice = slice(0, 1)
    ) -> dict[str, torch.Tensor]:
        """Extract active features and encoder/decoder vectors for attribution.

        Returns the same dict format as CrossLayerTranscoder.compute_attribution_components
        so it can be used with the existing AttributionContext.

        The key difference: encoder_vecs are Jacobian rows (local linear approximation)
        instead of W_enc rows.

        Args:
            inputs: Input tensor of shape (n_layers, n_pos, d_model).
            zero_positions: Positions to zero out.

        Returns:
            Dict with keys:
                activation_matrix: Sparse activation matrix.
                reconstruction: Reconstructed outputs.
                encoder_vecs: Jacobian-based encoder vectors for active features.
                decoder_vecs: Scaled decoder vectors for active features.
                encoder_to_decoder_map: Mapping from encoder to decoder indices.
                decoder_locations: (layer_ids, pos_ids) for decoder vectors.
        """
        features, encoder_vectors = self.encode_sparse(
            inputs, zero_positions=zero_positions
        )
        pos_ids, layer_ids, feat_ids, decoder_vectors, encoder_to_decoder_map = (
            self.select_decoder_vectors(features)
        )
        reconstruction = self.compute_reconstruction(
            pos_ids, layer_ids, decoder_vectors, inputs
        )

        return {
            "activation_matrix": features,
            "reconstruction": reconstruction,
            "encoder_vecs": encoder_vectors,
            "decoder_vecs": decoder_vectors,
            "encoder_to_decoder_map": encoder_to_decoder_map,
            "decoder_locations": torch.stack((layer_ids, pos_ids)),
        }

    def to_safetensors(self, save_path: str) -> None:
        """Save Spline-CLT to safetensors format.

        Saves encoder state dicts, decoder weights, biases, and thresholds.

        Args:
            save_path: Directory path for safetensors files.
        """
        os.makedirs(save_path, exist_ok=True)

        has_threshold = isinstance(self.activation_function, JumpReLU)

        for i in range(self.n_layers):
            enc_dict = {}
            if self.encoder_type == "kan":
                # Save full KAN encoder state dict
                enc_state = self.encoders[i].kan_linear.state_dict()
                enc_dict = {f"encoder_{i}.{k}": v.cpu() for k, v in enc_state.items()}
            else:
                # Save linear encoder weight matrix
                enc_dict[f"encoder_{i}.W_enc"] = self.encoders[i].W_enc.cpu()

            enc_dict[f"b_enc_{i}"] = self.b_enc[i].cpu()
            enc_dict[f"b_dec_{i}"] = self.b_dec[i].cpu()
            enc_dict[f"enc_input_mean_{i}"] = self.enc_input_mean[i].cpu()
            enc_dict[f"enc_input_std_{i}"] = self.enc_input_std[i].cpu()

            if has_threshold:
                # `.threshold` holds log θ in memory; persist the *effective*
                # θ = exp(log θ) so the on-disk `threshold_{i}` stays the literal
                # non-negative gate value (format-compatible with the standard-CLT
                # convention and with any downstream inspection). load_spline_clt
                # takes log() to restore the in-memory log-parametrization.
                enc_dict[f"threshold_{i}"] = (
                    self.activation_function.threshold[i].squeeze(0).exp().cpu()
                )

            save_file(enc_dict, os.path.join(save_path, f"encoder_{i}.safetensors"))

            # Save decoder
            dec_dict = {f"W_dec_{i}": self.W_dec[i].cpu()}
            save_file(dec_dict, os.path.join(save_path, f"W_dec_{i}.safetensors"))

        # Save metadata (include encoder_type so load_spline_clt can reconstruct correctly)
        metadata = {
            "n_layers": torch.tensor(self.n_layers),
            "d_transcoder": torch.tensor(self.d_transcoder),
            "d_model": torch.tensor(self.d_model),
            "grid_size": torch.tensor(self.grid_size),
            "spline_order": torch.tensor(self.spline_order),
            # encoder_type stored as 0=kan, 1=linear
            "encoder_type_linear": torch.tensor(self.encoder_type == "linear"),
            # activation: 0=jump_relu/relu (inferred from thresholds), 1=base_jump
            "activation_base_jump": torch.tensor(
                self.activation_function_name == "base_jump"
            ),
            "scale_base": torch.tensor(float(self.scale_base)),
            "scale_spline": torch.tensor(float(self.scale_spline)),
        }
        save_file(metadata, os.path.join(save_path, "metadata.safetensors"))


def load_spline_clt(
    clt_path: str,
    feature_input_hook: str = "hook_resid_mid",
    feature_output_hook: str = "hook_mlp_out",
    scan: str | list[str] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> KANCrossLayerTranscoder:
    """Load a Spline-CLT from safetensors files.

    Args:
        clt_path: Path to directory containing saved safetensors files.
        feature_input_hook: Hook point where features read from.
        feature_output_hook: Hook point where features write to.
        scan: Optional identifier for feature visualization.
        device: Device to load tensors to.
        dtype: Data type for loaded tensors.

    Returns:
        Loaded KANCrossLayerTranscoder instance.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load metadata
    from safetensors import safe_open

    meta_path = os.path.join(clt_path, "metadata.safetensors")
    with safe_open(meta_path, framework="pt") as f:
        n_layers = f.get_tensor("n_layers").item()
        d_transcoder = f.get_tensor("d_transcoder").item()
        d_model = f.get_tensor("d_model").item()
        grid_size = f.get_tensor("grid_size").item()
        spline_order = f.get_tensor("spline_order").item()
        # encoder_type_linear key was added after initial checkpoints; default to KAN
        keys = set(f.keys())
        is_linear = (
            f.get_tensor("encoder_type_linear").item()
            if "encoder_type_linear" in keys
            else False
        )
        is_base_jump = (
            bool(f.get_tensor("activation_base_jump").item())
            if "activation_base_jump" in keys
            else False
        )
        scale_base = (
            float(f.get_tensor("scale_base").item()) if "scale_base" in keys else 1.0
        )
        scale_spline = (
            float(f.get_tensor("scale_spline").item()) if "scale_spline" in keys else 1.0
        )
    encoder_type = "linear" if is_linear else "kan"

    # Detect activation function from first encoder file + metadata flag
    enc_path = os.path.join(clt_path, "encoder_0.safetensors")
    with safe_open(enc_path, framework="pt") as f:
        has_threshold = "threshold_0" in f.keys()

    if is_base_jump:
        act_fn = "base_jump"
    elif has_threshold:
        act_fn = "jump_relu"
    else:
        act_fn = "relu"

    instance = KANCrossLayerTranscoder(
        n_layers=n_layers,
        d_transcoder=d_transcoder,
        d_model=d_model,
        encoder_type=encoder_type,
        grid_size=grid_size,
        spline_order=spline_order,
        activation_function=act_fn,
        feature_input_hook=feature_input_hook,
        feature_output_hook=feature_output_hook,
        scan=scan,
        device=device,
        dtype=dtype,
        scale_base=scale_base,
        scale_spline=scale_spline,
    )

    # Load encoder weights and biases
    for i in range(n_layers):
        enc_file = os.path.join(clt_path, f"encoder_{i}.safetensors")
        enc_data = load_file(enc_file, device=str(device))

        prefix = f"encoder_{i}."
        if encoder_type == "kan":
            kan_state = {
                k[len(prefix):]: v.to(dtype=dtype)
                for k, v in enc_data.items()
                if k.startswith(prefix)
            }
            instance.encoders[i].kan_linear.load_state_dict(kan_state)
        else:
            # Linear encoder: restore W_enc directly
            instance.encoders[i].W_enc.data.copy_(
                enc_data[f"encoder_{i}.W_enc"].to(dtype=dtype)
            )

        instance.b_enc.data[i] = enc_data[f"b_enc_{i}"].to(dtype=dtype)
        instance.b_dec.data[i] = enc_data[f"b_dec_{i}"].to(dtype=dtype)

        # Input-normalization buffers were added later; default (mean=0, std=1)
        # leaves pre-normalization checkpoints behaving exactly as before.
        if f"enc_input_mean_{i}" in enc_data:
            instance.enc_input_mean.data[i] = enc_data[f"enc_input_mean_{i}"].to(dtype=dtype)
        if f"enc_input_std_{i}" in enc_data:
            instance.enc_input_std.data[i] = enc_data[f"enc_input_std_{i}"].to(dtype=dtype)

        if has_threshold:
            # On disk `threshold_{i}` is the literal effective θ ≥ 0; the module
            # stores log θ (see to_safetensors / __init__). clamp_min guards
            # against log(0) from any degenerate/legacy θ == 0 (fresh runs never
            # write 0, since θ = exp(·) > 0).
            instance.activation_function.threshold.data[i] = (
                enc_data[f"threshold_{i}"].clamp_min(1e-12).log().unsqueeze(0).to(dtype=dtype)
            )

        # Load decoder
        dec_file = os.path.join(clt_path, f"W_dec_{i}.safetensors")
        dec_data = load_file(dec_file, device=str(device))
        instance.W_dec[i].data.copy_(dec_data[f"W_dec_{i}"].to(dtype=dtype))

    return instance
