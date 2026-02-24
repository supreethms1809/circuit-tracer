"""KAN Cross-Layer Transcoder — replaces linear encoder in CLT with KAN encoder.

Architecture:
    a^l = JumpReLU(KAN_enc^l(x^l))          # KAN encoder (nonlinear)
    y_hat^l = Σ W_dec^(l'→l) · a^(l')       # linear decoder (same as standard CLT)

The decoder MUST remain linear — features need clean directions in residual stream
space for activation patching and steering to work.

This module mirrors the interface of circuit_tracer.transcoder.CrossLayerTranscoder
so it can be used as a drop-in replacement in the attribution pipeline.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

from circuit_tracer.transcoder.activation_functions import JumpReLU
from kan_clt.kan_encoder import KANEncoder
from kan_clt.linear_encoder import LinearEncoder


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
        activation_function: "jump_relu" or "relu".
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
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if encoder_type not in ("kan", "linear"):
            raise ValueError(f"encoder_type must be 'kan' or 'linear', got {encoder_type!r}")

        self.n_layers = n_layers
        self.d_transcoder = d_transcoder
        self.d_model = d_model
        self.encoder_type = encoder_type
        self.grid_size = grid_size
        self.spline_order = spline_order
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

        # Encoder biases (applied after KAN forward, before activation)
        self.b_enc = nn.Parameter(
            torch.zeros(n_layers, d_transcoder, device=device, dtype=dtype)
        )

        # Decoder biases
        self.b_dec = nn.Parameter(
            torch.zeros(n_layers, d_model, device=device, dtype=dtype)
        )

        # Activation function
        if activation_function == "jump_relu":
            self.activation_function = JumpReLU(
                torch.zeros(n_layers, 1, d_transcoder, device=device, dtype=dtype)
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
        """Initialize decoder weights with small random values."""
        for w_dec in self.W_dec:
            nn.init.kaiming_uniform_(w_dec.view(-1, self.d_model))

    @property
    def device(self) -> torch.device:
        return self.b_enc.device

    @property
    def dtype(self) -> torch.dtype:
        return self.b_enc.dtype

    def apply_activation_function(
        self, layer_id: int, features: torch.Tensor
    ) -> torch.Tensor:
        """Apply the sparsity-inducing activation function.

        Args:
            layer_id: Which layer's activation function to use.
            features: Pre-activation values of shape (..., d_transcoder).

        Returns:
            Activated features (same shape).
        """
        if isinstance(self.activation_function, JumpReLU):
            mask = features > self.activation_function.threshold[layer_id]
            return features * mask
        else:
            return self.activation_function(features)

    def encode_layer(
        self, x: torch.Tensor, layer_id: int, apply_activation_function: bool = True
    ) -> torch.Tensor:
        """Encode residual stream activations at a single layer.

        Args:
            x: Input tensor of shape (..., d_model).
            layer_id: Which layer to encode.
            apply_activation_function: Whether to apply JumpReLU/ReLU.

        Returns:
            Feature activations of shape (..., d_transcoder).
        """
        features = self.encoders[layer_id](x) + self.b_enc[layer_id]

        if not apply_activation_function:
            return features

        return self.apply_activation_function(layer_id, features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode residual stream activations at all layers.

        Args:
            x: Input tensor of shape (n_layers, batch, d_model).

        Returns:
            Feature activations of shape (n_layers, batch, d_transcoder).
        """
        layer_features = []
        for layer_id in range(self.n_layers):
            features = self.encode_layer(x[layer_id], layer_id)
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
            layer_features = self.encode_layer(x[layer_id], layer_id)
            layer_features[zero_positions] = 0

            sparse_layer = layer_features.to_sparse()
            sparse_layers.append(sparse_layer)

            # Compute Jacobian-based encoder vectors for active features
            if sparse_layer._nnz() > 0:
                active_mask = sparse_layer
                try:
                    enc_vecs = self.encoders[layer_id].get_encoder_vectors_fast(
                        x[layer_id], active_mask
                    )
                except RuntimeError:
                    enc_vecs = self.encoders[layer_id].get_encoder_vectors(
                        x[layer_id], active_mask
                    )
                encoder_vectors.append(enc_vecs)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode → decode.

        Args:
            x: Input tensor of shape (n_layers, batch, d_model).

        Returns:
            Reconstructed MLP outputs of shape (n_layers, batch, d_model).
        """
        features = self.encode(x).to_sparse()
        decoded = self.decode(features)

        if self.W_skip is not None:
            skip = x @ self.W_skip
            decoded = decoded + skip

        return decoded

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
        """Save KAN-CLT to safetensors format.

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

            if has_threshold:
                enc_dict[f"threshold_{i}"] = (
                    self.activation_function.threshold[i].squeeze(0).cpu()
                )

            save_file(enc_dict, os.path.join(save_path, f"encoder_{i}.safetensors"))

            # Save decoder
            dec_dict = {f"W_dec_{i}": self.W_dec[i].cpu()}
            save_file(dec_dict, os.path.join(save_path, f"W_dec_{i}.safetensors"))

        # Save metadata (include encoder_type so load_kan_clt can reconstruct correctly)
        metadata = {
            "n_layers": torch.tensor(self.n_layers),
            "d_transcoder": torch.tensor(self.d_transcoder),
            "d_model": torch.tensor(self.d_model),
            "grid_size": torch.tensor(self.grid_size),
            "spline_order": torch.tensor(self.spline_order),
            # encoder_type stored as 0=kan, 1=linear
            "encoder_type_linear": torch.tensor(self.encoder_type == "linear"),
        }
        save_file(metadata, os.path.join(save_path, "metadata.safetensors"))


def load_kan_clt(
    clt_path: str,
    feature_input_hook: str = "hook_resid_mid",
    feature_output_hook: str = "hook_mlp_out",
    scan: str | list[str] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> KANCrossLayerTranscoder:
    """Load a KAN-CLT from safetensors files.

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
        keys = f.keys()
        is_linear = (
            f.get_tensor("encoder_type_linear").item()
            if "encoder_type_linear" in keys
            else False
        )
    encoder_type = "linear" if is_linear else "kan"

    # Detect activation function from first encoder file
    enc_path = os.path.join(clt_path, "encoder_0.safetensors")
    with safe_open(enc_path, framework="pt") as f:
        has_threshold = "threshold_0" in f.keys()

    act_fn = "jump_relu" if has_threshold else "relu"

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

        if has_threshold:
            instance.activation_function.threshold.data[i] = (
                enc_data[f"threshold_{i}"].unsqueeze(0).to(dtype=dtype)
            )

        # Load decoder
        dec_file = os.path.join(clt_path, f"W_dec_{i}.safetensors")
        dec_data = load_file(dec_file, device=str(device))
        instance.W_dec[i].data.copy_(dec_data[f"W_dec_{i}"].to(dtype=dtype))

    return instance
