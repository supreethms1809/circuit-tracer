# Metrics Reference

A complete reference of every metric used in the Spline-CLT project — what it measures, how it's computed, and what values to aim for.

---

## Table of Contents

1. [Training Loss Metrics](#1-training-loss-metrics)
2. [Reconstruction Quality Metrics](#2-reconstruction-quality-metrics)
3. [Replacement Accuracy Metrics](#3-replacement-accuracy-metrics)
4. [Sparsity Metrics](#4-sparsity-metrics)
5. [Feature Interpretability Metrics](#5-feature-interpretability-metrics)
6. [Attribution Metrics](#6-attribution-metrics)
7. [Spline Analysis Metrics](#7-spline-analysis-metrics)
8. [Summary Table](#8-summary-table)

---

## 1. Training Loss Metrics

These metrics are computed at every training step and logged periodically. Defined in `spline_clt/training/loss.py` and logged in `spline_clt/training/train.py`.

### 1.1 Total Loss (`loss/total`)

| Property | Value |
|----------|-------|
| **What it measures** | Overall training objective — the single scalar being minimized |
| **Formula** | `L_total = L_reconstruction + L_sparsity + L_kan_regularization` |
| **Range** | [0, ∞) |
| **Ideal value** | As low as possible; typical converged values are 0.001–0.01 |
| **Computed in** | `total_loss()` in `spline_clt/training/loss.py` |

The total loss is the sum of three components described below. It balances reconstruction fidelity against sparsity and regularization.

---

### 1.2 Reconstruction Loss (`loss/reconstruction`)

| Property | Value |
|----------|-------|
| **What it measures** | How well the transcoder reproduces the true MLP outputs across all layers |
| **Formula** | `L_recon = mean((y_hat - y_true)²)` |
| **Inputs** | `y_hat`: predicted MLP output `(n_layers, n_pos, d_model)`, `y_true`: ground truth MLP output (same shape) |
| **Range** | [0, ∞) |
| **Ideal value** | As close to 0 as possible. Values < 0.005 indicate good reconstruction |
| **Computed in** | `reconstruction_loss()` in `spline_clt/training/loss.py` |

This is the mean squared error averaged over all layers, all token positions, and all residual stream dimensions. Both tensors are cast to float32 before computation for numerical stability.

**Interpretation:**
- Decreasing reconstruction loss means the transcoder is learning to approximate the MLP
- If this plateaus at a high value, the model capacity (d_transcoder) may be too small
- Per-layer breakdown (see [Section 2.2](#22-per-layer-mse-mse_per_layer)) reveals which layers are hardest to reconstruct

---

### 1.3 Sparsity Loss (`loss/sparsity`)

| Property | Value |
|----------|-------|
| **What it measures** | Penalty for having too many active features per position |
| **Formula** | `L_sparsity = λ × (1/n_pos) × Σ_layers Σ_pos Σ_features tanh(c × ‖W_dec_i‖ × a_i)` |
| **Parameters** | `λ` (lambda): sparsity coefficient (default 0.05), `c`: scaling constant (default 1.0) |
| **Inputs** | `activations`: post-JumpReLU features `(n_layers, n_pos, d_transcoder)`, `decoder_norms`: L2 norm of decoder vectors per feature |
| **Range** | [0, ∞) |
| **Ideal value** | Depends on desired sparsity level; typically 0.01–0.1 at convergence |
| **Computed in** | `sparsity_loss()` in `spline_clt/training/loss.py` |

**How it works:**
- `a_i` is the activation of feature i (post-JumpReLU, always ≥ 0)
- `‖W_dec_i‖` is the max L2 norm of feature i's decoder vector across all target layers (computed by `compute_decoder_norms()`)
- `tanh(c × ‖W_dec_i‖ × a_i)` saturates at 1.0 for strongly active features
- The tanh provides sub-linear growth — very active features aren't penalized infinitely more than moderately active ones

**Interpretation:**
- Higher sparsity loss → more features are active (less sparse)
- The weighting by decoder norm means features with large decoder vectors (large downstream impact) are penalized more heavily
- Increasing `λ` produces sparser representations at the cost of reconstruction quality
- The decoder norms use max across target layers: for a feature at layer 3 with decoders to layers 3–11, the largest decoder norm determines the penalty

---

### 1.4 KAN Regularization Loss (`loss/kan_regularization`)

| Property | Value |
|----------|-------|
| **What it measures** | Complexity of the learned B-spline functions in KAN encoders |
| **Formula** | `L_kan_reg = 0.01 × (1/n_layers) × Σ_layers mean(|spline_weight|)` |
| **Inputs** | `spline_weight` parameter from each `KANLinear` layer — shape `(n_features, d_model × (grid_size + spline_order))` |
| **Range** | [0, ∞) |
| **Ideal value** | Small but nonzero; typically 0.001–0.01. Zero means splines collapsed to linear |
| **Applies to** | KAN encoders only. Skipped (returns 0.0) for linear encoders |
| **Computed in** | `total_loss()` in `spline_clt/training/loss.py` |

**Why this specific formulation:**
- The efficient-kan library provides `KANLinear.regularization_loss()`, but it computes an entropy term `p × log(p)` which produces NaN when spline weights are exactly zero (because `0 × log(0) = 0 × -∞ = NaN` in IEEE 754 floating point)
- After grid updates, some spline weights can become exactly zero, triggering this bug
- The simple L1 penalty `|spline_weight|.mean()` avoids this entirely

**Interpretation:**
- Encourages spline weights to be small, preventing overfitting to B-spline grid artifacts
- Too strong → splines collapse to near-linear functions (defeating the purpose of KAN)
- Too weak → splines can develop erratic shapes that don't generalize
- The 0.01 coefficient was chosen to be small enough that it doesn't dominate reconstruction

---

### 1.5 Active Features Per Position (`stats/active_features_per_pos`)

| Property | Value |
|----------|-------|
| **What it measures** | Average number of features with nonzero activation at each token position |
| **Formula** | `mean(Σ_features 𝟙[a_i > 0])` averaged over all layers and positions |
| **Inputs** | `activations`: post-JumpReLU features `(n_layers, n_pos, d_transcoder)` |
| **Range** | [0, d_transcoder] (i.e., [0, 4096] with default config) |
| **Ideal value** | 50–200 (1–5% of features). Depends on task |
| **Computed in** | `total_loss()` in `spline_clt/training/loss.py` |

**Interpretation:**
- This is the primary sparsity indicator during training
- Too few active features → poor reconstruction (not enough capacity)
- Too many active features → features are not selective, harder to interpret
- Anthropic's CLTs typically operate at 50–200 active features out of thousands
- Early in training, this may be high (before JumpReLU thresholds are learned); it should decrease as sparsity loss takes effect

---

### 1.6 Learning Rate (`lr`)

| Property | Value |
|----------|-------|
| **What it measures** | Current optimizer learning rate |
| **Formula** | Warmup: `lr_base × step / warmup_steps`. Cosine: `lr_base × 0.5 × (1 + cos(π × progress))` |
| **Range** | [0, learning_rate] (default [0, 1e-4]) |
| **Schedule** | Steps 0–1000: linear warmup. Steps 1000–50000: cosine decay to 0 |
| **Computed in** | `get_lr()` in `spline_clt/training/train.py` |

---

### 1.7 Validation Loss (`val_loss`)

| Property | Value |
|----------|-------|
| **What it measures** | Reconstruction loss on held-out data (no sparsity/regularization) |
| **Formula** | `mean((y_hat - y_true)²)` averaged over all validation batches |
| **Range** | [0, ∞) |
| **Ideal value** | Close to training reconstruction loss. Large gap → overfitting |
| **Computed in** | `evaluate()` in `spline_clt/training/train.py` |

Only reconstruction loss is measured during validation — sparsity and KAN regularization are training-only signals. The best checkpoint is saved when validation loss reaches a new minimum.

---

## 2. Reconstruction Quality Metrics

These metrics evaluate how well a trained transcoder approximates the original MLP outputs. Computed in `eval/replacement_accuracy.py` and `experiments/compare_models.py`.

### 2.1 Total MSE (`mse_total`)

| Property | Value |
|----------|-------|
| **What it measures** | Average reconstruction error across all layers |
| **Formula** | `(1/n_layers) × Σ_l mean((y_hat_l - y_true_l)²)` |
| **Inputs** | Predicted and true MLP outputs over multiple evaluation samples |
| **Range** | [0, ∞) |
| **Ideal value** | < 0.005 for good reconstruction; < 0.001 for excellent |
| **Computed in** | `evaluate_reconstruction()` in `eval/replacement_accuracy.py` |

This is the average of per-layer MSE values, giving a single number for overall reconstruction quality. Computed over `n_samples` evaluation sequences (default 200).

---

### 2.2 Per-Layer MSE (`mse_per_layer`)

| Property | Value |
|----------|-------|
| **What it measures** | Reconstruction error at each individual transformer layer |
| **Formula** | `mean((y_hat_l - y_true_l)²)` for each layer l |
| **Output shape** | List of n_layers floats (12 for GPT-2 small) |
| **Ideal value** | Roughly uniform across layers; spikes indicate problematic layers |
| **Computed in** | `evaluate_reconstruction()` in `eval/replacement_accuracy.py` |

**Interpretation:**
- Early layers (0–3) are often easier to reconstruct (simpler MLP functions)
- Later layers (8–11) may have higher MSE due to more complex computations
- A layer with disproportionately high MSE may need more features (higher d_transcoder) or better encoder capacity

---

### 2.3 Cosine Similarity (`cosine_similarity`)

| Property | Value |
|----------|-------|
| **What it measures** | Directional agreement between predicted and true MLP output vectors |
| **Formula** | `mean(cos_sim(y_hat.flatten(), y_true.flatten()))` where `cos_sim(a, b) = (a · b) / (‖a‖ × ‖b‖)` |
| **Inputs** | Predicted and true outputs reshaped to `(-1, d_model)` — each row is one position's output vector |
| **Range** | [-1, 1] |
| **Ideal value** | > 0.95. Value of 1.0 means perfect directional match |
| **Computed in** | `evaluate_reconstruction()` in `eval/replacement_accuracy.py` |

**Interpretation:**
- Cosine similarity captures directional accuracy independent of magnitude
- A model can have high cosine similarity but moderate MSE if the magnitudes are slightly off
- More informative than MSE for understanding whether the transcoder preserves the "direction" of computation
- Values below 0.9 suggest the reconstruction is qualitatively different from the original

---

### 2.4 Relative Error (`relative_error`)

| Property | Value |
|----------|-------|
| **What it measures** | Reconstruction error normalized by the magnitude of the true output |
| **Formula** | `‖y_hat - y_true‖_F / max(‖y_true‖_F, 1e-8)` |
| **Inputs** | Frobenius norms of the error and true output tensors |
| **Range** | [0, ∞) |
| **Ideal value** | < 0.1 (10% relative error). < 0.05 is excellent |
| **Computed in** | `evaluate_reconstruction()` in `eval/replacement_accuracy.py` |

**Interpretation:**
- Scale-independent measure: a relative error of 0.05 means the reconstruction error is 5% of the signal magnitude
- More meaningful than raw MSE when comparing across layers or models with different activation scales
- The `max(..., 1e-8)` clamp prevents division by zero for near-silent positions

---

## 3. Replacement Accuracy Metrics

These metrics measure whether replacing the original MLPs with the transcoder preserves the language model's behavior. Computed in `eval/replacement_accuracy.py`.

### 3.1 Top-1 Match Rate (`top1_match_rate`)

| Property | Value |
|----------|-------|
| **What it measures** | Fraction of token positions where the replacement model predicts the same top token as the original |
| **Formula** | `(1/n_positions) × Σ_pos 𝟙[argmax(logits_original) = argmax(logits_replaced)]` |
| **Inputs** | Original model logits and replacement model logits for each position in each prompt |
| **Range** | [0, 1] |
| **Ideal value** | > 0.90. Value of 1.0 means the replacement model always predicts the same top token |
| **Computed in** | `evaluate_replacement_accuracy()` in `eval/replacement_accuracy.py` |

**How the replacement works:**
1. Run the original GPT-2 model, collect logits and MLP activations at every layer
2. Run the transcoder to reconstruct MLP outputs
3. Compute the residual adjustment: `Σ_layers (reconstructed_mlp - true_mlp)`
4. Adjusted logits = `(final_residual + adjustment) @ W_U + b_U`
5. Compare argmax of original vs adjusted logits

**Interpretation:**
- This is the coarsest measure of fidelity — does the replacement change the model's predictions?
- High match rate (>95%) means the transcoder faithfully preserves model behavior
- Low match rate (<80%) means the reconstruction errors are large enough to change predictions
- Important caveat: a model can have high match rate but still have significant probability distribution differences (see KL divergence below)

---

### 3.2 KL Divergence (`kl_divergence`)

| Property | Value |
|----------|-------|
| **What it measures** | How much the replacement model's output probability distribution differs from the original |
| **Formula** | `KL(P_replaced ‖ P_original) = Σ_tokens P_replaced(t) × log(P_replaced(t) / P_original(t))` |
| **Inputs** | Softmax probability distributions over vocabulary for original and replacement models |
| **Range** | [0, ∞) |
| **Ideal value** | < 0.1 nats. Value of 0.0 means identical distributions |
| **Computed in** | `evaluate_replacement_accuracy()` in `eval/replacement_accuracy.py` |

**Interpretation:**
- More sensitive than top-1 match rate — captures differences in the full probability distribution
- KL < 0.01: essentially identical distributions
- KL 0.01–0.1: minor distribution shifts, unlikely to affect downstream behavior
- KL 0.1–1.0: noticeable differences, model behavior may change in some cases
- KL > 1.0: substantial distribution shift, replacement is not faithful
- Averaged across all positions in all evaluation prompts

---

## 4. Sparsity Metrics

These metrics quantify how sparse the feature activations are. Computed in `eval/replacement_accuracy.py` and `experiments/compare_models.py`.

### 4.1 Average Active Features Per Position (`average_active_per_pos`)

| Property | Value |
|----------|-------|
| **What it measures** | Mean number of features with nonzero activation at each (layer, position) |
| **Formula** | `mean over all (layer, pos) of: Σ_features 𝟙[a_i > 0]` |
| **Inputs** | Post-JumpReLU activations `(n_layers, n_pos, d_transcoder)` |
| **Range** | [0, d_transcoder] |
| **Ideal value** | 50–200 out of 4096 features (1–5% density) |
| **Computed in** | `evaluate_sparsity()` in `eval/replacement_accuracy.py` |

**Interpretation:**
- Core sparsity measure — how many features does the model "use" at each position?
- Fewer active features → easier to interpret individual circuits
- Too few → reconstruction suffers (not enough capacity)
- This is the same quantity as `stats/active_features_per_pos` from training, but computed on evaluation data

---

### 4.2 Activation Density (`activation_density`)

| Property | Value |
|----------|-------|
| **What it measures** | Fraction of all (layer, position, feature) entries that are nonzero |
| **Formula** | `mean(𝟙[activations > 0])` over the entire activation tensor |
| **Inputs** | Post-JumpReLU activations `(n_layers, n_pos, d_transcoder)` |
| **Range** | [0, 1] |
| **Ideal value** | 0.01–0.05 (1–5%) |
| **Computed in** | `evaluate_sparsity()` in `eval/replacement_accuracy.py` |

**Relationship to active features per position:**
```
activation_density ≈ average_active_per_pos / d_transcoder
```

**Interpretation:**
- Global sparsity measure across the entire evaluation set
- Density of 0.02 means 2% of features are active on average — good sparsity
- Density above 0.10 suggests the JumpReLU thresholds need to be higher or `λ_sparsity` should increase

---

### 4.3 Parameter Count (`n_params`)

| Property | Value |
|----------|-------|
| **What it measures** | Total number of trainable parameters in the model |
| **Formula** | `Σ p.numel() for p in model.parameters() if p.requires_grad` |
| **Range** | Positive integer |
| **Spline-CLT typical value** | ~300M encoder + shared decoder |
| **Linear CLT typical value** | ~37M encoder + shared decoder |
| **Computed in** | `evaluate_model()` in `experiments/compare_models.py` |

Used for fair comparison — Spline-CLT has ~8x more encoder parameters than linear CLT at the same d_transcoder, due to B-spline basis expansion.

---

## 5. Feature Interpretability Metrics

These metrics assess whether individual features capture coherent, interpretable concepts. Computed in `eval/monosemanticity.py`.

### 5.1 Gini Coefficient (`gini_coefficient`)

| Property | Value |
|----------|-------|
| **What it measures** | How concentrated a feature's activations are across examples — a proxy for monosemanticity |
| **Formula** | Sort activations: `v_1 ≤ v_2 ≤ ... ≤ v_n`. Then: `G = (2 × Σ_i (i × v_i)) / (n × Σ v_i) - (n + 1) / n` |
| **Inputs** | All activation values for one feature across all evaluated positions |
| **Range** | [0, 1] |
| **Ideal value** | > 0.8 for monosemantic features |
| **Computed in** | `gini_coefficient()` in `eval/monosemanticity.py` |

**Interpretation:**
- **G ≈ 0**: Feature activates uniformly across all examples → polysemantic (responds to many unrelated concepts)
- **G ≈ 0.5**: Moderate concentration → may respond to a few related concepts
- **G ≈ 1**: Feature activates on very few specific examples → monosemantic (responds to one concept)
- A high Gini coefficient is necessary but not sufficient for interpretability — the feature could be monosemantic but respond to an uninterpretable pattern

**Aggregate statistics:**
- `mean_gini`: Average Gini across top features. Higher is better.
- `high_gini_count`: Number of features with Gini > 0.8. More is better.

---

### 5.2 Activation Frequency (`activation_frequency`)

| Property | Value |
|----------|-------|
| **What it measures** | Fraction of token positions where a feature is active (nonzero) |
| **Formula** | `(# positions where a_f > 0) / (total # positions evaluated)` |
| **Inputs** | Post-JumpReLU activations for one feature across all samples |
| **Range** | [0, 1] |
| **Ideal value** | 0.001–0.05 (0.1%–5%). Features that fire on everything are not useful |
| **Computed in** | `collect_max_activating_examples()` Pass 1 in `eval/monosemanticity.py` |

**Interpretation:**
- Very low frequency (< 0.001) → feature may be dead or extremely specific
- Moderate frequency (0.001–0.05) → likely captures a meaningful, selective concept
- High frequency (> 0.10) → probably polysemantic or captures a generic pattern (e.g., "any noun")
- Features are ranked by activation frequency to identify the most commonly used ones for analysis

---

### 5.3 Mean Activation (`mean_activation`)

| Property | Value |
|----------|-------|
| **What it measures** | Average activation magnitude across all positions (including zeros) |
| **Formula** | `(Σ_pos a_f(pos)) / n_positions` |
| **Range** | [0, ∞) |
| **Ideal value** | Context-dependent; used for relative comparison between features |
| **Computed in** | `collect_max_activating_examples()` Pass 1 in `eval/monosemanticity.py` |

**Interpretation:**
- Low mean with high max → feature is selective (fires rarely but strongly)
- High mean → feature contributes a steady background signal
- Mean / max ratio indicates how "peaky" the activation distribution is

---

### 5.4 Max Activation (`max_activation`)

| Property | Value |
|----------|-------|
| **What it measures** | Strongest activation observed for a feature across all evaluated data |
| **Formula** | `max_pos a_f(pos)` |
| **Range** | [0, ∞) |
| **Ideal value** | Context-dependent; higher values indicate the feature has strong "trigger" examples |
| **Computed in** | `collect_max_activating_examples()` Pass 1 in `eval/monosemanticity.py` |

**Interpretation:**
- Used to identify "interesting" features that respond strongly to specific inputs
- Features with very low max activation may be effectively dead
- The max-activating examples (stored in `top_examples`) are the primary tool for manual interpretability assessment

---

## 6. Attribution Metrics

These metrics quantify feature contributions and interactions during circuit tracing. Computed in `attribution/causal.py` and `attribution/shapley.py`.

### 6.1 Causal Feature Effect (`feature_effects`)

| Property | Value |
|----------|-------|
| **What it measures** | The causal effect of ablating feature s on feature t's activation |
| **Formula** | `effect(s → t) = (baseline_recon - ablated_recon)[layer_t, pos_t] · encoder_direction_t` |
| **Inputs** | Baseline reconstruction, ablated reconstruction (with feature s zeroed), encoder Jacobian for feature t |
| **Output shape** | `(n_active, n_active)` matrix |
| **Range** | (-∞, ∞); positive = feature s supports feature t, negative = feature s suppresses feature t |
| **Computed in** | `ablation_attribution()` in `attribution/causal.py` |

**Interpretation:**
- Large positive value → feature s strongly activates feature t (excitatory connection)
- Large negative value → feature s suppresses feature t (inhibitory connection)
- Near zero → feature s has negligible causal effect on feature t
- The encoder direction for feature t is the Jacobian row `d(encoder_output[t])/d(input)` for KAN, or the weight matrix row for linear encoders

---

### 6.2 Output Effect (`output_effects`)

| Property | Value |
|----------|-------|
| **What it measures** | How much ablating a feature changes the reconstructed MLP output at every (layer, position) |
| **Formula** | `output_effect(s) = baseline_reconstruction - ablated_reconstruction(s)` |
| **Output shape** | `(n_active, n_layers, n_pos, d_model)` |
| **Range** | (-∞, ∞) per dimension |
| **Computed in** | `ablation_attribution()` in `attribution/causal.py` |

**Interpretation:**
- Shows the full spatial pattern of each feature's causal influence
- Large output effects at positions far from the feature's own position indicate cross-position influence (via cross-layer decoder)
- The norm of the output effect vector indicates overall influence magnitude

---

### 6.3 Shapley Value (`shapley_values`)

| Property | Value |
|----------|-------|
| **What it measures** | Game-theoretic fair contribution of each feature to the reconstruction quality |
| **Formula** | `φ_s = (1/M) × Σ_permutations [V(S ∪ {s}) - V(S)]` where `V(S) = ‖y_full - y_S‖²` is the reconstruction MSE with feature subset S |
| **Inputs** | Model, residual stream inputs, n_samples permutations |
| **Output shape** | `(n_active,)` |
| **Range** | (-∞, ∞); typically positive (features improve reconstruction). Sum ≈ V(all) - V(∅) |
| **Ideal interpretation** | Shapley values sum to the total reconstruction improvement |
| **Computed in** | `shapley_attribution()` in `attribution/shapley.py` |

**Interpretation:**
- Positive Shapley value → feature contributes to better reconstruction (MSE reduction)
- Negative Shapley value → feature's presence *worsens* reconstruction (possible interference)
- Larger magnitude → greater individual contribution
- Unlike causal ablation, Shapley values correctly account for feature interactions (synergies, redundancies)
- The efficiency property guarantees: `Σ φ_s = V({all features}) - V(∅)`

**Variance considerations:**
- Monte Carlo estimates have sampling noise; increase `n_samples` for more precise values
- With antithetic sampling (default), each permutation pair reduces variance by sampling complementary coalitions
- Standard error decreases as `O(1/√n_samples)`

---

### 6.4 Feature-to-Feature Shapley (`feature_shapley`)

| Property | Value |
|----------|-------|
| **What it measures** | Shapley value of source feature s for target feature t's activation level |
| **Formula** | For each target t: `φ_s^t = (1/M) × Σ_perms [a_t(S ∪ {s}) - a_t(S)]` where `a_t(S)` is t's activation when only sources in S are active |
| **Output shape** | `(n_active, n_active)` — `[s, t]` is Shapley value of s for t |
| **Range** | (-∞, ∞) |
| **Constraint** | Only computed when `n_active ≤ 32` (computational cost is O(n_active² × n_samples)) |
| **Computed in** | `_feature_to_feature_shapley()` in `attribution/shapley.py` |

**Interpretation:**
- Reveals how features causally support or inhibit each other, accounting for all interaction orders
- More expensive but more principled than single-feature ablation for understanding feature interactions

---

### 6.5 Logit Shapley Value

| Property | Value |
|----------|-------|
| **What it measures** | Feature's contribution to a specific output token's logit |
| **Formula** | Same as Shapley value but value function is `V(S) = y_hat_S[-1, -1, :] · logit_target` where `logit_target` is the unembedding direction for the target token |
| **Inputs** | Logit target direction `(d_model,)`, typically `W_U[:, token_id]` |
| **Output shape** | `(n_active,)` |
| **Range** | (-∞, ∞) |
| **Computed in** | `shapley_logit_attribution()` in `attribution/shapley.py` |

**Interpretation:**
- Positive → feature promotes the target token
- Negative → feature suppresses the target token
- Useful for understanding token prediction circuits (e.g., "which features cause the model to predict 'Paris' after 'The capital of France is'?")

---

### 6.6 Adjacency Matrix (Circuit Graph)

| Property | Value |
|----------|-------|
| **What it measures** | Weighted directed graph of feature-to-feature and feature-to-output connections |
| **Output shape** | `(total_nodes, total_nodes)` sparse matrix |
| **Node ordering** | `[active_features | error_nodes | token_nodes | logit_nodes]` |
| **Edge weights** | Causal effects or Shapley values |
| **Computed in** | `build_attribution_graph()` in `attribution/causal.py` |

**Edge types:**
- **Feature → feature**: Causal effect of source on target (from `feature_effects`)
- **Feature → error**: L2 norm of unexplained reconstruction change
- **Error/token → logit**: From pre-computed logit effects (optional)

The graph is compatible with `circuit_tracer.graph.Graph` for use with the upstream library's pruning and visualization tools.

---

## 7. Spline Analysis Metrics

These metrics describe the learned B-spline transfer functions in KAN encoders. Computed in `experiments/analyze_splines.py`. Only applicable to Spline-CLT models.

### 7.1 Top Input Dimensions

| Property | Value |
|----------|-------|
| **What it measures** | Which residual stream dimensions most influence a given feature |
| **Formula** | `top_dims = base_weight[feature_id].abs().topk(k).indices` |
| **Inputs** | KANLinear base_weight matrix `(n_features, d_model)` |
| **Output** | List of k dimension indices (default k=3) |
| **Computed in** | `top_input_dims()` in `experiments/analyze_splines.py` |

**Interpretation:**
- Identifies which input dimensions the feature pays most attention to
- The transfer function along these dimensions reveals the nonlinear pattern

### 7.2 Spline Transfer Function

| Property | Value |
|----------|-------|
| **What it measures** | The effective input-output function of the KAN encoder along one input dimension |
| **Formula** | `curve(t) = encoder(e_j × t)[feature_id]` for `t ∈ [-5, +5]`, where `e_j` is the j-th standard basis vector |
| **Inputs** | Unit vector along one dimension, sweep parameter t |
| **Output** | Array of (t, output) pairs — 200 points per curve |
| **Computed in** | `extract_spline_curve()` in `experiments/analyze_splines.py` |

**Interpretation:**
- **Linear curve** (straight line) → KAN encoder behaves like a linear encoder for this feature/dimension; no benefit from KAN
- **Threshold/step curve** → feature detects when a dimension exceeds a value; a nonlinear boundary that linear encoders cannot capture cleanly
- **Peaked curve** → feature responds to a specific range of input values; band-pass behavior
- **Polynomial/curved** → smooth nonlinear relationship that captures activation patterns missed by linear encoding

---

## 8. Summary Table

| Metric | Category | Range | Ideal | Key Insight |
|--------|----------|-------|-------|-------------|
| **Total loss** | Training | [0, ∞) | < 0.01 | Overall training objective |
| **Reconstruction loss** | Training | [0, ∞) | < 0.005 | Fidelity of MLP approximation |
| **Sparsity loss** | Training | [0, ∞) | 0.01–0.1 | Feature activation penalty |
| **KAN regularization** | Training | [0, ∞) | 0.001–0.01 | B-spline complexity penalty |
| **Active features/pos** | Training/Eval | [0, 4096] | 50–200 | Sparsity level |
| **Validation loss** | Training | [0, ∞) | ≈ train recon loss | Overfitting detector |
| **MSE (total)** | Reconstruction | [0, ∞) | < 0.005 | Overall reconstruction quality |
| **MSE (per-layer)** | Reconstruction | [0, ∞) | Uniform across layers | Per-layer quality |
| **Cosine similarity** | Reconstruction | [-1, 1] | > 0.95 | Directional agreement |
| **Relative error** | Reconstruction | [0, ∞) | < 0.05 | Scale-normalized error |
| **Top-1 match rate** | Replacement | [0, 1] | > 0.90 | Token prediction fidelity |
| **KL divergence** | Replacement | [0, ∞) | < 0.1 | Distribution shift |
| **Activation density** | Sparsity | [0, 1] | 0.01–0.05 | Global sparsity |
| **Gini coefficient** | Interpretability | [0, 1] | > 0.8 | Feature selectivity |
| **Activation frequency** | Interpretability | [0, 1] | 0.001–0.05 | Feature usage rate |
| **Mean activation** | Interpretability | [0, ∞) | Context-dependent | Background signal level |
| **Max activation** | Interpretability | [0, ∞) | Context-dependent | Peak response strength |
| **Causal feature effect** | Attribution | (-∞, ∞) | Large magnitude = important | Feature-to-feature causation |
| **Output effect** | Attribution | (-∞, ∞) | Large norm = influential | Feature-to-output causation |
| **Shapley value** | Attribution | (-∞, ∞) | Positive = helpful | Fair contribution score |
| **Feature Shapley** | Attribution | (-∞, ∞) | Positive = supportive | Pairwise interaction score |
| **Logit Shapley** | Attribution | (-∞, ∞) | Positive = promotes token | Token prediction contribution |
| **Parameter count** | Model | Positive int | Lower at same quality | Efficiency comparison |
