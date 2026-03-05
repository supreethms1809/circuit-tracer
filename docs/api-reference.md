# API Reference

## spline_clt.kan_encoder

### `KANEncoder(nn.Module)`

B-spline KAN encoder that computes nonlinear feature pre-activations from residual stream activations.

```python
KANEncoder(
    d_model: int,                          # residual stream dimension (768 for GPT-2)
    n_features: int,                       # number of output features
    grid_size: int = 5,                    # B-spline grid intervals
    spline_order: int = 3,                 # B-spline order (3 = cubic)
    grid_range: list[float] | None = None, # input grid bounds (default [-1, 1])
    base_activation: type[nn.Module] = nn.SiLU  # activation for base linear map
)
```

**Properties:**
- `basis_expansion_dim → int` — `d_model × (grid_size + spline_order)`
- `n_parameters → int` — total trainable parameter count

**Methods:**

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `forward(x)` | `(..., d_model)` | `(..., n_features)` | Compute feature pre-activations |
| `get_encoder_vectors(x, active_mask)` | `(n_pos, d_model)`, mask | `(n_active, d_model)` | Jacobian rows via backward AD |
| `get_encoder_vectors_fast(x, active_mask)` | same | same | Jacobian rows via forward AD (faster) |
| `update_grid(x)` | `(batch, d_model)` | None | Adapt B-spline knots to data |

---

## spline_clt.linear_encoder

### `LinearEncoder(nn.Module)`

Linear baseline encoder. Drop-in replacement for `KANEncoder`.

```python
LinearEncoder(
    d_model: int,     # residual stream dimension
    n_features: int,  # number of output features
)
```

**Methods:**

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `forward(x)` | `(..., d_model)` | `(..., n_features)` | `F.linear(x, W_enc)` |
| `get_encoder_vectors(x, active_mask)` | `(n_pos, d_model)`, mask | `(n_active, d_model)` | Selected rows of `W_enc` |
| `get_encoder_vectors_fast(x, active_mask)` | same | same | Same as above (exact for linear) |
| `update_grid(x)` | any | None | No-op |

---

## spline_clt.kan_transcoder

### `KANCrossLayerTranscoder(nn.Module)`

Full cross-layer transcoder with KAN or linear encoder, JumpReLU activation, and cross-layer linear decoder.

```python
KANCrossLayerTranscoder(
    n_layers: int,                          # transformer layers (12 for GPT-2)
    d_transcoder: int,                      # features per layer (e.g. 4096)
    d_model: int,                           # residual stream dimension (768)
    encoder_type: str = "kan",              # "kan" or "linear"
    grid_size: int = 5,                     # KAN grid size
    spline_order: int = 3,                  # KAN spline order
    activation_function: str = "jump_relu", # "jump_relu" or "relu"
    skip_connection: bool = False,          # learned skip connection
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
)
```

**Encoding methods:**

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `encode_layer(x, layer_id)` | `(..., d_model)`, int | `(..., d_transcoder)` | Single-layer encode + activate |
| `encode(x)` | `(n_layers, batch, d_model)` | `(n_layers, batch, d_transcoder)` | Encode all layers |
| `encode_sparse(x)` | `(n_layers, n_pos, d_model)` | sparse features, encoder vectors | Encode + sparsify + Jacobians |

**Decoding methods:**

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `decode_dense(activations)` | `(n_layers, n_pos, d_transcoder)` | `(n_layers, n_pos, d_model)` | Dense cross-layer decode (training) |
| `decode(features)` | sparse features | `(n_layers, n_pos, d_model)` | Sparse cross-layer decode |
| `select_decoder_vectors(features)` | sparse features | pos_ids, layer_ids, feat_ids, vectors, mapping | Extract indexed decoder vectors |

**Full pipeline:**

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `forward(x)` | `(n_layers, batch, d_model)` | `(n_layers, batch, d_model)` | Encode → sparsify → decode |
| `compute_attribution_components(inputs)` | `(n_layers, n_pos, d_model)` | dict | Encoder/decoder vectors for attribution |

**Save/Load:**

| Method | Description |
|--------|-------------|
| `to_safetensors(save_path)` | Save model to safetensors directory |
| `load_spline_clt(save_path, ...)` | Module-level function: load from safetensors |

---

## spline_clt.training.data

### `ActivationDataset(Dataset)`

PyTorch dataset of paired MLP input/output activations.

```python
ActivationDataset(
    mlp_inputs: torch.Tensor,   # (n_samples, n_layers, seq_len, d_model)
    mlp_outputs: torch.Tensor,  # same shape
)
```

| Method | Description |
|--------|-------------|
| `__getitem__(idx) → dict` | Returns `{"mlp_inputs": ..., "mlp_outputs": ...}` |
| `save(path)` | Save tensors to directory |
| `load(path, max_samples=None)` | Class method: mmap-load, slice, clone to RAM |

### `collect_activations(config: DataConfig) → ActivationDataset`

Collect activations from a transformer model.

```python
DataConfig(
    model_name: str = "gpt2",
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    n_tokens: int = 10_000_000,
    seq_len: int = 128,
    batch_size: int = 32,
    save_dir: str = "data/activations",
    device: str = "cuda",
)
```

---

## spline_clt.training.loss

| Function | Signature | Description |
|----------|-----------|-------------|
| `reconstruction_loss(y_hat, y_true)` | `Tensor, Tensor → Tensor` | MSE loss |
| `sparsity_loss(activations, decoder_norms, lambda_, c)` | `Tensor, list, float, float → Tensor` | Weighted tanh sparsity |
| `compute_decoder_norms(model)` | `KANCrossLayerTranscoder → list[Tensor]` | Max L2 norm per feature |
| `total_loss(model, x_in, y_true, lambda_sparsity, c_sparsity)` | `... → (Tensor, dict)` | Combined loss + metrics |

---

## spline_clt.training.train

### `TrainConfig`

```python
TrainConfig(
    # Architecture
    n_layers: int = 12, d_model: int = 768, d_transcoder: int = 4096,
    encoder_type: str = "kan", grid_size: int = 5, spline_order: int = 3,
    # Training
    learning_rate: float = 1e-4, warmup_steps: int = 1000,
    total_steps: int = 50_000, batch_size: int = 4,
    lambda_sparsity: float = 0.05, c_sparsity: float = 1.0, grad_clip: float = 1.0,
    # KAN grid
    update_grid_every: int = 10_000, update_grid_from: int = 2000,
    # Checkpointing
    log_every: int = 100, eval_every: int = 5000, save_every: int = 5000,
    checkpoint_dir: str = "checkpoints", run_name: str = "spline_clt",
    # Data
    data_dir: str = "data/activations", device: str = "cuda",
    dtype: str = "float32", val_fraction: float = 0.05,
)
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `train(config, dataset=None)` | `TrainConfig, ActivationDataset? → KANCrossLayerTranscoder` | Full training loop |
| `evaluate(model, val_dataset, config, device, dtype)` | `... → float` | Validation reconstruction loss |
| `load_config(path)` | `str → TrainConfig` | Load from YAML |

---

## attribution.causal

| Function | Description |
|----------|-------------|
| `ablation_attribution(model, x_in, max_features=256)` | Single-feature ablation. Returns `active_features`, `activation_values`, `feature_effects`, `output_effects`. |
| `build_attribution_graph(model, x_in, max_features=256)` | Ablation + graph construction. Returns `active_features`, `adjacency_matrix`. |

---

## attribution.shapley

| Function | Description |
|----------|-------------|
| `shapley_attribution(model, x_in, target="reconstruction", n_samples=256, max_features=64, antithetic=True)` | Monte Carlo Shapley values. Returns `shapley_values`, `active_features`, `activation_values`. |
| `shapley_logit_attribution(model, x_in, logit_target, n_samples=128, max_features=64)` | Shapley for specific logit direction. |

---

## attribution.graph

| Function | Description |
|----------|-------------|
| `create_graph_from_attribution(attribution_result, input_string, input_tokens, logit_tokens, logit_probabilities, cfg)` | Convert attribution dict to `circuit_tracer.graph.Graph`. |

---

## eval.replacement_accuracy

| Function | Returns |
|----------|---------|
| `evaluate_reconstruction(model, mlp_inputs, mlp_outputs)` | `{mse_total, mse_per_layer, cosine_similarity, relative_error}` |
| `evaluate_replacement_accuracy(original_model, spline_clt, prompts)` | `{top1_match_rate, kl_divergence}` |
| `evaluate_sparsity(model, mlp_inputs)` | `{average_active_per_pos, activation_density}` |

---

## eval.monosemanticity

### `FeatureReport`

```python
FeatureReport(
    layer: int,
    feature_id: int,
    activation_frequency: float,
    mean_activation: float,
    max_activation: float,
    gini_coefficient: float,
    top_examples: list[FeatureExample],
)
```

| Function | Description |
|----------|-------------|
| `gini_coefficient(values)` | Compute Gini index of a distribution |
| `collect_max_activating_examples(model, dataset, top_n_features=200, top_k_examples=10, n_samples=500)` | Two-pass analysis → `list[FeatureReport]` |
| `save_reports(reports, path)` | Serialize to JSON |
| `print_summary(reports, top_n=20)` | Print human-readable table |

---

## spline_clt.utils

| Function | Description |
|----------|-------------|
| `count_parameters(model)` | Total trainable parameters |
| `count_parameters_breakdown(model)` | Parameters grouped by component name |
| `compare_parameter_counts(spline_clt, d_model, d_transcoder, n_layers)` | KAN vs linear parameter comparison |
