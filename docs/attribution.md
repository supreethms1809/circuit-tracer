# Attribution Methods

This document explains how Spline-CLT computes feature-to-feature and feature-to-output attribution for circuit tracing.

## Why New Attribution Methods?

Standard circuit tracing (Anthropic's approach) computes attribution using the encoder weight matrix directly:
```
effect(feature_s → feature_t) = a_s × (W_dec_s · W_enc_t)
```

This works because the linear encoder gives each feature a fixed direction in activation space. With a KAN encoder, feature directions are **input-dependent** (the Jacobian changes at every input point). The linear shortcut breaks down, so we need attribution methods that don't assume encoder linearity.

Spline-CLT provides two methods:
1. **Causal ablation**: exact causal effects via single-feature knockout
2. **Shapley values**: game-theoretic credit assignment via permutation sampling

## Causal Ablation

**File**: `attribution/causal.py`

### How It Works

For each active feature, zero it out and measure how everything downstream changes:

```
For each active feature s:
    1. Set a_s = 0 (ablate feature s)
    2. Re-run the decoder to get the ablated reconstruction
    3. For each downstream feature t:
        effect(s → t) = (reconstruction_change) · (encoder_direction_t)
    4. For output:
        effect(s → output) = reconstruction_change at each position/layer
```

The encoder direction for feature t is the Jacobian row `d(encoder_output[t]) / d(input)` evaluated at the current input. For linear encoders, this is just the weight matrix row.

### API

```python
from attribution.causal import ablation_attribution, build_attribution_graph

# Basic ablation: get feature-to-feature effects
result = ablation_attribution(
    model,              # trained KANCrossLayerTranscoder
    x_in,               # (n_layers, n_pos, d_model) residual stream
    max_features=256,    # limit to top N features by activation
)

# result contains:
#   active_features:  (n_active, 3) tensor of [layer, pos, feat_idx]
#   activation_values: (n_active,) magnitudes
#   feature_effects:   (n_active, n_active) causal effect matrix
#   output_effects:    (n_active, n_layers, n_pos, d_model)
#   baseline_reconstruction: full reconstruction without ablation

# Build circuit-tracer compatible graph
graph_result = build_attribution_graph(
    model, x_in, max_features=256
)
# Returns adjacency matrix with nodes: [features | error | token | logit]
```

### Complexity

O(n_active) forward passes through the decoder. Each ablation requires one decoder pass to compute the counterfactual reconstruction.

## Shapley Attribution

**File**: `attribution/shapley.py`

### Why Shapley Values?

Causal ablation measures what happens when you remove a single feature in isolation. But features can have **interaction effects** — removing feature A alone might have little effect, but removing both A and B together causes a large change. Shapley values from cooperative game theory handle this correctly by averaging a feature's marginal contribution across all possible coalitions.

### How It Works

The Shapley value of feature s is:
```
φ_s = E_π [ V(S ∪ {s}) - V(S) ]
```

Where:
- π is a random permutation of features
- S is the set of features before s in the permutation
- V(S) is the value function (how well subset S reconstructs the output)

Since exact computation requires evaluating all 2^n subsets, we use **Monte Carlo estimation** with permutation sampling.

### Algorithm

```
1. Identify top N active features by activation magnitude
2. For M permutation samples:
    a. Generate random permutation of N features
    b. Walk through the permutation, adding one feature at a time:
       - Before adding feature s: compute V(S) = ||y_full - y_current||²
       - After adding feature s:  compute V(S ∪ {s})
       - Marginal contribution of s: V(S ∪ {s}) - V(S)
    c. (If antithetic) Repeat with reversed permutation
3. Average marginal contributions across all samples
```

### Antithetic Sampling

With `antithetic=True` (default), each permutation is paired with its reverse. This doubles the effective sample count without additional decoder evaluations and reduces variance because the forward and reverse orderings sample complementary coalitions.

### API

```python
from attribution.shapley import shapley_attribution, shapley_logit_attribution

# Reconstruction-target Shapley values
result = shapley_attribution(
    model,               # trained KANCrossLayerTranscoder
    x_in,                # (n_layers, n_pos, d_model)
    target="reconstruction",  # or "feature" for feature-to-feature
    n_samples=256,       # Monte Carlo samples
    max_features=64,     # top N features
    antithetic=True,     # paired permutations for variance reduction
)

# result contains:
#   active_features:  (n_active, 3)
#   activation_values: (n_active,)
#   shapley_values:    (n_active,) contribution scores
#   full_output:       reconstruction with all features
#   empty_output:      reconstruction with no features

# Feature-to-feature Shapley (expensive, limited to n_active ≤ 32)
result = shapley_attribution(
    model, x_in,
    target="feature",
    n_samples=64,
    max_features=32,
)
# result["feature_shapley"]: (n_active, n_active) matrix

# Logit-direction Shapley (for token prediction circuits)
result = shapley_logit_attribution(
    model, x_in,
    logit_target=W_U[:, token_id],  # (d_model,) direction
    n_samples=128,
    max_features=64,
)
# result["shapley_values"]: (n_active,) contribution to target logit
```

### Complexity

O(n_active × n_samples) decoder evaluations for reconstruction Shapley. Feature-to-feature Shapley is O(n_active² × n_samples), which is why it's limited to 32 features.

## Circuit Graph Construction

**File**: `attribution/graph.py`

### Connecting to circuit-tracer

The circuit-tracer library provides graph pruning, visualization, and evaluation tools. Spline-CLT attribution results are converted to the `circuit_tracer.graph.Graph` format via:

```python
from attribution.graph import create_graph_from_attribution

graph = create_graph_from_attribution(
    attribution_result,      # from ablation or Shapley
    input_string="The Eiffel Tower is located in",
    input_tokens=token_ids,  # (n_pos,)
    logit_tokens=top_k_ids,  # top K predicted token IDs
    logit_probabilities=top_k_probs,
    cfg=model.cfg,           # HookedTransformerConfig
)

# graph is a circuit_tracer.graph.Graph instance
# Can be used with circuit-tracer's pruning and visualization
```

### Node Types in the Graph

The adjacency matrix has this node ordering:
```
[active features | error nodes | token nodes | logit nodes]
```

- **Feature nodes**: Each active feature (layer, position, feature_id)
- **Error nodes**: Capture unexplained variance at each position
- **Token nodes**: Input token embeddings
- **Logit nodes**: Output predictions

Edge weights come from the attribution method (causal effects or Shapley values).

## Choosing an Attribution Method

| Method | Speed | Handles Interactions | Theory |
|--------|-------|---------------------|--------|
| Causal ablation | Fast (O(n_active)) | No (single-feature only) | Causal |
| Shapley values | Slow (O(n_active × n_samples)) | Yes (all orders) | Game-theoretic |

**Use causal ablation** when:
- You need fast attribution for many prompts
- Feature interactions are not the primary concern
- You want exact single-feature causal effects

**Use Shapley values** when:
- You suspect important feature interactions
- You need theoretically principled credit assignment
- You can afford the computational cost

Both methods produce circuit-tracer compatible graphs for downstream analysis.

## End-to-End Circuit Tracing

The `experiments/run_circuit.py` script ties everything together:

```bash
# Causal ablation (fast)
conda run -n ct python experiments/run_circuit.py \
    --checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
    --prompt "The Eiffel Tower is located in" \
    --max-features 64

# With Shapley attribution (slow)
conda run -n ct python experiments/run_circuit.py \
    --checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
    --prompt "The Eiffel Tower is located in" \
    --shapley --shapley-samples 128
```

This outputs:
1. A ranked table of top features by activation magnitude
2. Top feature-to-feature edges by causal effect
3. Optional Shapley value ranking
4. An optional `.pt` file with the full attribution result
