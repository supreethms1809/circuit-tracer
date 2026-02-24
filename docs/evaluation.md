# Evaluation

This document covers how to evaluate trained KAN-CLT models: reconstruction quality, replacement accuracy, sparsity, monosemanticity, and spline analysis.

## Reconstruction Quality

**File**: `eval/replacement_accuracy.py`

Measures how well the transcoder reproduces the original MLP outputs.

### Metrics

| Metric | What it measures | Good values |
|--------|-----------------|-------------|
| **MSE (total)** | Average squared error across all layers/positions | Lower is better |
| **MSE (per layer)** | Breakdown by transformer layer | Identifies weak layers |
| **Cosine similarity** | Directional agreement between predicted and true outputs | Close to 1.0 |
| **Relative error** | Frobenius norm ratio: \|\|y_hat - y_true\|\| / \|\|y_true\|\| | Close to 0.0 |

### How Reconstruction is Measured

```python
from eval.replacement_accuracy import evaluate_reconstruction

metrics = evaluate_reconstruction(model, mlp_inputs, mlp_outputs)
# mlp_inputs:  (n_samples, n_layers, seq_len, d_model)
# mlp_outputs: (n_samples, n_layers, seq_len, d_model)

# Returns:
#   mse_total: float
#   mse_per_layer: list[float]
#   cosine_similarity: float
#   relative_error: float
```

For each sample, the model encodes activations, decodes to reconstructed MLP outputs, and compares against ground truth.

## Replacement Accuracy

Measures whether replacing MLPs with the transcoder preserves the model's predictions.

### Metrics

| Metric | What it measures |
|--------|-----------------|
| **Top-1 match rate** | Fraction of positions where predicted token matches original |
| **KL divergence** | How much the output distribution changes |

### How Replacement is Measured

```python
from eval.replacement_accuracy import evaluate_replacement_accuracy

metrics = evaluate_replacement_accuracy(
    original_model,  # HookedTransformer (GPT-2)
    kan_clt,         # trained transcoder
    prompts,         # list of text strings
)

# Returns:
#   top1_match_rate: float (0 to 1)
#   kl_divergence: float (lower is better)
```

For each prompt:
1. Run the original model to get logits and MLP activations
2. Run the transcoder to get reconstructed MLP outputs
3. Compute adjusted logits using the reconstruction
4. Compare adjusted vs original predictions

## Sparsity

Measures how sparse the feature activations are.

```python
from eval.replacement_accuracy import evaluate_sparsity

metrics = evaluate_sparsity(model, mlp_inputs)

# Returns:
#   average_active_per_pos: float (mean # of active features per position)
#   activation_density: float (fraction of features that are nonzero)
```

Good sparsity means each position activates only a small fraction of features. Typical targets: 50-200 active features per position out of 4096 total (1-5% density).

## Monosemanticity

**File**: `eval/monosemanticity.py`

Measures whether individual features respond to coherent, interpretable concepts rather than being polysemantic (responding to many unrelated things).

### Gini Coefficient

The Gini coefficient measures how concentrated a feature's activations are across examples:
- **Gini ≈ 0**: Feature activates uniformly (polysemantic)
- **Gini ≈ 1**: Feature activates on very few examples (monosemantic)

Higher Gini coefficients suggest features are more interpretable.

### Max-Activating Examples

For each feature, the analysis collects the top-K examples (token positions) where the feature activates most strongly. These examples can be manually inspected to assess whether the feature captures a coherent concept.

### How Monosemanticity is Measured

```python
from eval.monosemanticity import collect_max_activating_examples, print_summary

reports = collect_max_activating_examples(
    model,
    dataset,             # ActivationDataset
    top_n_features=200,  # analyze the 200 most frequent features
    top_k_examples=10,   # keep 10 strongest activations per feature
    n_samples=500,       # scan 500 dataset samples
)

print_summary(reports, top_n=20)
```

Each `FeatureReport` contains:
- `layer`, `feature_id`: which feature
- `activation_frequency`: fraction of positions where feature is active
- `mean_activation`, `max_activation`: activation statistics
- `gini_coefficient`: sparsity of activation distribution
- `top_examples`: list of `FeatureExample` with sample index, position, activation value

### Two-Pass Algorithm

The analysis runs in two passes for efficiency:

1. **Frequency scan**: iterate all samples, count activation frequency and compute mean/max per feature. Identify the top-N features by frequency.

2. **Example collection**: iterate again, maintain a heap of top-K examples for each selected feature. Collect all activations for Gini computation.

### Saving Results

```python
from eval.monosemanticity import save_reports

save_reports(reports, "results/monosemanticity/kan_features.json")
```

Saves a JSON file with layer, feature_id, statistics, and top examples for each feature.

## Spline Analysis (KAN Only)

**File**: `experiments/analyze_splines.py`

Extracts and visualizes the learned B-spline transfer functions in the KAN encoder. This shows what nonlinear transformations the model learned for each feature.

### What Gets Extracted

For each top feature:
1. Identify the 3 input dimensions with the largest base weights (most important inputs)
2. Probe the KAN encoder along each dimension: sweep input from -5 to +5
3. Record the transfer function: output value vs. input value

This produces curves showing how the encoder transforms each input dimension into the feature's pre-activation.

### Running Spline Analysis

```bash
conda run -n ct python experiments/analyze_splines.py \
    --checkpoint checkpoints/gpt2_small/kan_clt_gpt2_best \
    --n-features 20 \
    --output-dir results/splines
```

### Outputs

```
results/splines/
├── feature_summary.csv          # all features: layer, feat_id, frequency, mean, max
├── layer0_feat42.csv            # spline curves: t, dim_3, dim_8, dim_15
├── layer0_feat42.png            # matplotlib plot
├── layer2_feat128.csv
├── layer2_feat128.png
└── ...
```

Each CSV has columns `[t, dim_0, dim_1, dim_2]` where `t` is the input value and `dim_N` is the output along that input dimension. Plots show three curves (one per top dimension) overlaid.

### Interpretation

- **Linear curves**: The KAN feature is behaving like a linear encoder for this dimension
- **Threshold curves**: Step-function-like response indicates a learned threshold
- **Nonlinear curves**: Polynomial, peaked, or other nonlinear shapes indicate features that couldn't be captured by a linear encoder

## Full Evaluation Pipeline

**File**: `experiments/run_pipeline.py`

Runs all evaluation stages end-to-end and produces a summary report.

### Stages

| Stage | What it does | Output |
|-------|-------------|--------|
| **1. Reconstruction** | MSE, cosine sim, sparsity for each model | `reconstruction_metrics.json` |
| **2. Circuits** | Circuit tracing on 7 benchmark prompts | `circuit_summary.json` + `.pt` files |
| **3. Splines** | B-spline transfer functions (KAN only) | CSV + PNG in `splines/` |
| **4. Monosemanticity** | Gini coefficients + max-activating examples | JSON in `monosemanticity/` |

### Benchmark Prompts (Stage 2)

The pipeline traces circuits on these prompts:
- **IOI** (Indirect Object Identification): "John gave Mary the book. She gave it to"
- **Arithmetic**: "25 + 37 =", "The answer to 12 times 8 is"
- **Factual recall**: "The Eiffel Tower is located in", "The capital of Japan is"
- Additional reasoning prompts

### Running the Pipeline

```bash
# Full pipeline with both models
conda run -n ct python experiments/run_pipeline.py \
    --kan-checkpoint checkpoints/gpt2_small/kan_clt_gpt2_best \
    --linear-checkpoint checkpoints/gpt2_small_linear/linear_clt_gpt2_best \
    --data-dir data/activations \
    --output-dir results/eval_run1 \
    --shapley

# KAN-CLT only, skip slow stages
conda run -n ct python experiments/run_pipeline.py \
    --kan-checkpoint checkpoints/gpt2_small/kan_clt_gpt2_best \
    --data-dir data/activations \
    --skip-circuits --skip-monosemanticity

# Quick comparison without full pipeline
conda run -n ct python experiments/compare_models.py \
    --kan-checkpoint checkpoints/gpt2_small/kan_clt_gpt2_best \
    --linear-checkpoint checkpoints/gpt2_small_linear/linear_clt_gpt2_best \
    --data-dir data/activations \
    --n-samples 200
```

### Pipeline Output

```
results/eval_run1/
├── report.md                    # Markdown summary with tables
├── reconstruction_metrics.json
├── circuit_summary.json
├── circuits/
│   ├── kan_ioi_john_mary.pt
│   ├── linear_ioi_john_mary.pt
│   └── ...
├── splines/
│   ├── feature_summary.csv
│   ├── layer0_feat42.csv
│   └── ...
└── monosemanticity/
    ├── kan_features.json
    └── linear_features.json
```

The `report.md` summarizes all stages in a format suitable for inclusion in a research paper.
