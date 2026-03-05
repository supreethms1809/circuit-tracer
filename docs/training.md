# Training

This document covers the data collection pipeline, training loop, and loss functions.

## Data Collection

**File**: `spline_clt/training/data.py`

Training data consists of paired MLP inputs and outputs collected from GPT-2 small. The idea is to train the transcoder to reconstruct what each MLP layer does, given the residual stream.

### What Gets Collected

For each token position in each sequence:
- **MLP inputs**: residual stream activations *before* the MLP at each layer (`hook_resid_mid`)
- **MLP outputs**: the MLP's output at each layer (`hook_mlp_out`)

Shape: `(n_samples, n_layers=12, seq_len=128, d_model=768)`

### Collection Process

```
1. Load GPT-2 small via TransformerLens
2. Load wikitext-2 dataset from HuggingFace
3. Tokenize texts into sequences of length 128
4. Run model with hooks, capture activations at every layer
5. Store as bfloat16 tensors on disk
```

Key details:
- Sequences shorter than 128 tokens are discarded
- Activations are stored in bfloat16 to halve storage (float32 would be ~80GB)
- Tensors are preallocated and filled in-place to avoid 2x peak RAM from list concatenation
- Total dataset: ~8,500 sequences, ~40GB on disk

### Loading Data

`ActivationDataset.load(path, max_samples=3000)` uses a memory-mapped loading strategy:

```
1. Memory-map the 40GB files (no RAM used yet)
2. Slice the first 3000 samples
3. Clone into contiguous RAM (~14GB)
```

This avoids loading the entire dataset. Loading all 40GB causes OOM or SIGBUS errors on systems where the OS reclaims mmap pages under memory pressure.

### Running Data Collection

```bash
conda run -n ct python experiments/train_spline_clt.py \
    --collect-data --model gpt2 --device cuda
```

This takes ~30 minutes on a GPU. Output goes to `data/activations/`.

## Loss Functions

**File**: `spline_clt/training/loss.py`

The training objective has three components:

### 1. Reconstruction Loss (MSE)

```
L_recon = mean((y_hat - y_true)^2)
```

Measures how well the transcoder reconstructs the true MLP outputs across all layers and positions. Both tensors are cast to float32 for numerical stability.

### 2. Sparsity Loss

```
L_sparsity = λ × mean(Σ tanh(c × ||W_dec_i|| × a_i))
```

Matches Anthropic's CLT formulation. Encourages sparse feature activations:
- `a_i`: activation of feature i (post-JumpReLU, always ≥ 0)
- `||W_dec_i||`: L2 norm of feature i's decoder vector (max across target layers)
- `c`: scaling constant (default 1.0)
- `λ`: sparsity coefficient (default 0.05)

The tanh saturates at 1 for strongly active features, so the penalty grows sub-linearly. Features with large decoder norms are penalized more — this encourages the model to use many small features rather than a few large ones.

Decoder norms use the **max** across target layers (a feature at layer 3 has separate decoder vectors for layers 3-11; the largest one determines the penalty).

### 3. KAN Regularization

```
L_kan_reg = 0.01 × mean(|spline_weight|)
```

L1 penalty on the B-spline coefficients. Only applies to KAN encoders (linear encoders skip this). Prevents overfitting of spline shapes to grid artifacts.

**Important**: This does NOT use `KANLinear.regularization_loss()` from the efficient-kan library. That function computes an entropy term `p * log(p)` which produces NaN when spline weights are exactly zero (after grid updates), because `0 * log(0)` = `0 * (-inf)` = NaN in floating point.

### Total Loss

```
L_total = L_recon + L_sparsity + L_kan_reg
```

The loss function returns both the scalar loss and a metrics dictionary:
```python
{
    "loss/total": ...,
    "loss/reconstruction": ...,
    "loss/sparsity": ...,
    "loss/kan_regularization": ...,
    "stats/active_features_per_pos": ...    # mean # of nonzero features per position
}
```

## Training Loop

**File**: `spline_clt/training/train.py`

### Configuration

Training is configured via a `TrainConfig` dataclass, typically loaded from a YAML file.

**Spline-CLT config** (`experiments/configs/gpt2_small.yaml`):
```yaml
n_layers: 12
d_model: 768
d_transcoder: 4096
encoder_type: kan         # "kan" or "linear"
grid_size: 5
spline_order: 3

learning_rate: 1.0e-4
warmup_steps: 1000
total_steps: 50000
batch_size: 8
lambda_sparsity: 0.05
c_sparsity: 1.0
grad_clip: 1.0

update_grid_every: 10000  # KAN-specific: adapt B-spline knots
update_grid_from: 2000    # Don't update grid during early training

data_dir: data/activations
device: cuda
dtype: bfloat16
```

The **linear baseline** config (`experiments/configs/gpt2_small_linear_baseline.yaml`) is identical except `encoder_type: linear` and grid updates disabled.

### Training Steps

Each training step:

```
1. Fetch batch: (batch_size, n_layers, seq_len, d_model)
   Reshape to:  (n_layers, batch_size × seq_len, d_model)

2. Update learning rate (warmup + cosine decay)

3. Forward pass:
   - Encode: KAN/linear encoder → add bias → JumpReLU → sparse features
   - Decode: cross-layer linear decoder → reconstructed MLP outputs
   - Compute loss: reconstruction + sparsity + KAN regularization

4. Backward pass:
   - loss.backward()
   - Clip gradients to norm 1.0
   - optimizer.step()

5. Periodic operations:
   - Every 100 steps: log metrics
   - Every 10,000 steps (after step 2000): update KAN grid knots
   - Every 5,000 steps: evaluate on validation set, save checkpoint
```

### Learning Rate Schedule

Linear warmup followed by cosine decay:

```
Steps 0–1000:      LR ramps from 0 to 1e-4 (linear)
Steps 1000–50000:  LR decays from 1e-4 to 0 (cosine)
```

### Grid Updates (KAN Only)

Every 10,000 steps (starting from step 2,000), the B-spline grid knots are updated to better cover the actual data distribution. This calls `KANLinear.update_grid()` which repositions knot points based on quantiles of the input data.

Grid updates are skipped during early training because:
- The model hasn't learned meaningful features yet
- Updating the grid changes the function the encoder computes, which can destabilize training

### Validation

Every 5,000 steps, the model is evaluated on a held-out 5% validation split. Only reconstruction loss is measured (no sparsity or regularization). If the validation loss improves, the model is saved as the "best" checkpoint.

### Decode Strategy

Training uses `decode_dense()` instead of `decode()`. Dense decoding operates on the full activation tensor with einsum, while sparse decoding first converts to sparse format. Dense is more memory-efficient during early training when features are not yet sparse (many nonzero activations would make the sparse representation larger than the dense one).

## Running Training

```bash
# Train Spline-CLT
conda run -n ct python experiments/train_spline_clt.py \
    --config experiments/configs/gpt2_small.yaml

# Train linear baseline
conda run -n ct python experiments/train_spline_clt.py \
    --config experiments/configs/gpt2_small_linear_baseline.yaml

# Override specific config values via CLI
conda run -n ct python experiments/train_spline_clt.py \
    --config experiments/configs/gpt2_small.yaml \
    --learning-rate 5e-5 \
    --total-steps 100000 \
    --device cuda
```

### Outputs

Training produces checkpoints in the configured directory:
```
checkpoints/gpt2_small/
├── spline_clt_gpt2_best/        # Best validation loss
│   ├── metadata.safetensors
│   ├── encoder_layer_0.safetensors
│   ├── ...
│   └── b_dec.safetensors
├── spline_clt_gpt2_step_5000/   # Periodic checkpoints
├── spline_clt_gpt2_step_10000/
└── spline_clt_gpt2_final/       # End of training
```
