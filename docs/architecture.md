# Architecture

This document explains how Spline-CLT works, from the encoder through the decoder to the full cross-layer transcoder.

## Overview

A Cross-Layer Transcoder (CLT) decomposes MLP outputs in a transformer into sparse, interpretable features. The architecture has three stages:

```
Residual Stream → Encoder → JumpReLU → Decoder → Reconstructed MLP Output
```

**Standard CLT** (Anthropic's approach):
```
a = JumpReLU(W_enc · x + b_enc)        # linear encoder
y_hat = W_dec · a + b_dec               # linear decoder
```

**Spline-CLT** (this project):
```
a = JumpReLU(KAN(x) + b_enc)           # KAN encoder (nonlinear)
y_hat = W_dec · a + b_dec               # linear decoder (unchanged)
```

The decoder stays linear because features need clean directions in residual stream space for activation patching and steering to work.

## KAN Encoder

**File**: `spline_clt/kan_encoder.py`

The KAN encoder wraps the efficient-kan library's `KANLinear` layer, which implements Kolmogorov-Arnold Networks using B-spline basis functions.

### How B-Spline Encoding Works

A standard linear encoder computes:
```
output = W · x      where W is (n_features, d_model)
```

A KAN encoder first expands each input dimension into B-spline basis functions, then linearly combines them:

```
Step 1: Basis expansion
  For each input dimension i (of d_model=768):
    Compute (grid_size + spline_order) = 8 basis function values
    → 768 inputs become 768 × 8 = 6144 basis features

Step 2: Linear combination
  output = W_spline · basis_features + W_base · silu(x)
  where W_spline is (n_features, 6144) and W_base is (n_features, 768)
```

The B-spline functions are defined by a grid of knot points. With `grid_size=5` and `spline_order=3` (cubic), each input dimension gets 8 basis functions that tile the input range `[-1, 1]`.

### Numerical Precision

KAN encoders are kept in float32 regardless of the training dtype. B-spline computation requires float32 because:
- Grid knot points in bfloat16 can quantize to identical values
- This causes degenerate basis functions where multiple knots collapse to the same position
- The encoder casts inputs to float32, computes, then casts back to the training dtype

### Encoder Vectors (Jacobian)

For attribution, we need encoder "directions" — how each feature relates to the input space. For a linear encoder, these are simply the rows of W_enc. For a KAN encoder, directions are input-dependent.

`KANEncoder.get_encoder_vectors(x, active_mask)` computes:
```
encoder_vector[f] = d(output[f]) / d(input)    evaluated at x
```

This Jacobian row gives the local linear approximation at each input point. Two implementations exist:
- `get_encoder_vectors()`: backward-mode AD, groups by unique positions
- `get_encoder_vectors_fast()`: forward-mode AD via `torch.func.jacrev` + `vmap`, faster when many features are active

## Linear Encoder (Baseline)

**File**: `spline_clt/linear_encoder.py`

Drop-in replacement for `KANEncoder` that computes `F.linear(x, W_enc)`. Same interface:
- `forward(x)` → feature pre-activations
- `get_encoder_vectors(x, active_mask)` → selected rows of W_enc (no Jacobian needed)
- `update_grid(x)` → no-op (no grid to update)

## Cross-Layer Transcoder

**File**: `spline_clt/kan_transcoder.py`

### Cross-Layer Structure

In a transformer, information flows through residual stream connections across layers. A feature detected at layer 3 might influence MLP outputs at layers 4, 5, ..., 11. The cross-layer transcoder models this:

```
Feature at layer l reads from:   residual stream at layer l (one encoder)
Feature at layer l writes to:    MLP outputs at layers l, l+1, ..., n_layers-1 (multiple decoders)
```

Concretely:
- **Encoders**: One per layer. `encoders[l]` maps `(d_model,) → (d_transcoder,)`.
- **Decoder weights**: `W_dec[l]` has shape `(d_transcoder, n_layers - l, d_model)`. Features at layer `l` have separate decoder vectors for each downstream layer.
- **Biases**: `b_enc` of shape `(n_layers, d_transcoder)` and `b_dec` of shape `(n_layers, d_model)`.

### JumpReLU Activation

JumpReLU applies a learned threshold per feature:
```
JumpReLU(x, θ) = x * (x > θ)
```

Where θ is a learnable parameter of shape `(n_layers, 1, d_transcoder)`. Features with pre-activations below the threshold are zeroed out, providing sparsity. This is different from standard ReLU (which uses θ=0) — the learned threshold allows the model to control the activation density of each feature.

### Forward Pass

The full forward pass:

```python
# 1. Encode: residual stream → sparse features
for each layer l:
    pre_act = encoder[l](x[l]) + b_enc[l]      # KAN or linear
    features[l] = JumpReLU(pre_act, threshold[l])

# 2. Decode: sparse features → reconstructed MLP outputs
for each layer l:
    y_hat[l] = b_dec[l]
    for each source layer l' <= l:
        y_hat[l] += features[l'] @ W_dec[l'][:, l - l', :]
```

### Sparse vs Dense Decoding

Two decoding paths exist:

- **`decode_dense()`**: Used during training. Operates on dense activation tensors. Uses `einsum` for efficient cross-layer matmul without materializing per-feature decoder contributions.

- **`decode()` / `select_decoder_vectors()`**: Used during attribution. Converts activations to sparse format first, then computes `activation_value × decoder_vector` for each active feature. Returns indexed decoder vectors compatible with circuit-tracer's graph format.

### Save/Load Format

Models are saved as safetensors files:
```
checkpoint_dir/
├── metadata.safetensors          # Config: n_layers, d_model, d_transcoder, encoder_type, etc.
├── encoder_layer_0.safetensors   # KAN: grid, base_weight, spline_weight, etc.
├── encoder_layer_1.safetensors   # Linear: W_enc matrix
├── ...
├── decoder_layer_0.safetensors   # W_dec[0]: (d_transcoder, n_remaining, d_model)
├── decoder_layer_1.safetensors
├── ...
├── b_enc.safetensors             # (n_layers, d_transcoder)
├── b_dec.safetensors             # (n_layers, d_model)
└── jump_relu_thresholds.safetensors  # (n_layers, 1, d_transcoder) if JumpReLU
```

`load_spline_clt(path)` auto-detects encoder type and activation function from saved metadata and reconstructs the model.

## Parameter Counts

At GPT-2 small dimensions (d_model=768, d_transcoder=4096, 12 layers):

| Component | Spline-CLT | Linear CLT |
|-----------|---------|------------|
| Encoder (per layer) | ~25M params | ~3.1M params |
| Encoder (all layers) | ~300M params | ~37M params |
| Decoder (all layers) | identical | identical |
| Bias (all) | identical | identical |
| **Ratio (encoder)** | **~8x more** | **baseline** |

The parameter increase comes from the B-spline basis expansion: each of 768 input dimensions produces 8 basis features (6144 total), and the linear combination `(n_features, 6144)` is larger than `(n_features, 768)`.

The `spline_clt/utils.py` module provides `count_parameters()`, `count_parameters_breakdown()`, and `compare_parameter_counts()` for inspecting these.
