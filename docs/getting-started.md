# Getting Started

This guide walks through setup, data collection, training, and evaluation.

## Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- ~50GB disk space for activation data
- ~16GB GPU memory (training) or ~8GB (evaluation only)
- Conda environment named `ct`

## Installation

```bash
# Clone the repository
git clone https://github.com/supreethms1809/circuit-tracer.git
cd circuit-tracer

# Create conda environment
conda create -n ct python=3.10
conda activate ct

# Install dependencies
pip install torch torchvision  # with CUDA support
pip install transformer-lens
pip install datasets           # HuggingFace datasets
pip install safetensors
pip install pyyaml
pip install efficient-kan       # B-spline KAN implementation

# Install circuit-tracer (the upstream library)
pip install -e .
```

## Verify Installation

Run the test suite to confirm everything works:

```bash
conda run -n ct pytest tests/test_kan_encoder.py tests/test_kan_transcoder.py \
    tests/test_attribution.py tests/test_shapley.py -v
```

All 53 tests should pass.

## Step 1: Collect Activation Data

Collect MLP input/output activations from GPT-2 small:

```bash
conda run -n ct python experiments/train_spline_clt.py \
    --collect-data --model gpt2 --device cuda
```

This runs GPT-2 on the wikitext-2 dataset and saves activations to `data/activations/`. Takes ~30 minutes on a GPU. Output is ~40GB.

**What gets collected**: For each token position in ~8,500 sequences (128 tokens each), the residual stream activations before each MLP (`hook_resid_mid`) and the MLP output (`hook_mlp_out`) across all 12 layers.

## Step 2: Train Models

### Train Spline-CLT

```bash
conda run -n ct python experiments/train_spline_clt.py \
    --config experiments/configs/gpt2_small.yaml
```

This trains for 50,000 steps with:
- KAN encoder (B-spline, grid_size=5, spline_order=3)
- 4096 features per layer
- JumpReLU activation with learned thresholds
- Adam optimizer, lr=1e-4 with warmup + cosine decay

Checkpoints are saved to `checkpoints/gpt2_small/`.

### Train Linear Baseline

```bash
conda run -n ct python experiments/train_spline_clt.py \
    --config experiments/configs/gpt2_small_linear_baseline.yaml
```

Same architecture and hyperparameters, but with a standard linear encoder instead of KAN. This provides a controlled comparison. Checkpoints go to `checkpoints/gpt2_small_linear/`.

### Training Time

On an NVIDIA GH200 or A100:
- Spline-CLT: ~10-15 hours (50K steps)
- Linear CLT: ~3-5 hours (50K steps, simpler encoder)

## Step 3: Evaluate

### Quick Comparison

Compare reconstruction quality between KAN and linear:

```bash
conda run -n ct python experiments/compare_models.py \
    --kan-checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
    --linear-checkpoint checkpoints/gpt2_small_linear/linear_clt_gpt2_best \
    --data-dir data/activations \
    --n-samples 200
```

Prints a side-by-side table with MSE, cosine similarity, active features, and parameter counts.

### Circuit Tracing (Single Prompt)

Trace a circuit for a specific prompt:

```bash
conda run -n ct python experiments/run_circuit.py \
    --checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
    --prompt "The Eiffel Tower is located in" \
    --max-features 64 \
    --output results/circuits/eiffel.pt
```

This outputs a ranked table of the top features involved in processing this prompt, along with their layers, positions, and activation values.

Add `--shapley` for game-theoretic attribution (slower but handles feature interactions):

```bash
conda run -n ct python experiments/run_circuit.py \
    --checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
    --prompt "The Eiffel Tower is located in" \
    --shapley --shapley-samples 128
```

### Spline Analysis (KAN Only)

Visualize what nonlinear functions the KAN encoder learned:

```bash
conda run -n ct python experiments/analyze_splines.py \
    --checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
    --n-features 20 \
    --output-dir results/splines
```

Produces CSV files and PNG plots showing the B-spline transfer functions for the top 20 features.

### Full Evaluation Pipeline

Run all evaluation stages at once:

```bash
conda run -n ct python experiments/run_pipeline.py \
    --kan-checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
    --linear-checkpoint checkpoints/gpt2_small_linear/linear_clt_gpt2_best \
    --data-dir data/activations \
    --output-dir results/eval_run1 \
    --shapley
```

This runs:
1. **Reconstruction evaluation** — MSE, cosine similarity, sparsity
2. **Circuit tracing** — 7 benchmark prompts (IOI, arithmetic, factual recall)
3. **Spline analysis** — B-spline shapes for top features (KAN only)
4. **Monosemanticity** — Gini coefficients and max-activating examples

Results are saved to `results/eval_run1/` with a `report.md` summary.

To skip slow stages:
```bash
conda run -n ct python experiments/run_pipeline.py \
    --kan-checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
    --data-dir data/activations \
    --skip-circuits --skip-monosemanticity --no-plot
```

## Workflow Summary

```
collect data → train Spline-CLT → train linear baseline → evaluate
     ↓              ↓                    ↓                  ↓
data/activations/  checkpoints/gpt2_small/  checkpoints/gpt2_small_linear/  results/
```

## Troubleshooting

### Out of Memory (OOM) during data loading
Use `max_samples` to limit how many sequences are loaded:
```python
dataset = ActivationDataset.load("data/activations", max_samples=1000)
```
Default is 3000 (~14GB RAM). Reduce if needed.

### SIGBUS during data loading (WSL2)
This happens when the OS reclaims memory-mapped pages. The mmap+clone loading pattern should prevent this. If it still occurs, reduce `max_samples`.

### NaN loss during training
Check if `KANLinear.regularization_loss()` is being called somewhere — it has a known NaN bug with zero spline weights. The training code uses `spline_weight.abs().mean()` instead.

### Slow training
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Use bfloat16 dtype in config for faster training (default in YAML configs)
- KAN encoder runs in float32 internally regardless of training dtype (required for B-spline precision)
