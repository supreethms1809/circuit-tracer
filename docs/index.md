# KAN Circuit Tracer Documentation

KAN Circuit Tracer (KAN-CLT) replaces the linear encoder in Anthropic's Cross-Layer Transcoder with a KAN (Kolmogorov-Arnold Network) encoder, enabling nonlinear feature detection while preserving circuit tracing capability.

## What This Project Does

Anthropic's circuit tracing pipeline decomposes language model computations into interpretable features. Their Cross-Layer Transcoder (CLT) uses a **linear encoder** to detect features, followed by JumpReLU activation for sparsity, and a linear decoder to reconstruct MLP outputs. This linearity assumption means features must be defined by linear boundaries in activation space.

KAN-CLT replaces the linear encoder with a **B-spline-based KAN encoder** that can learn nonlinear feature boundaries. The decoder remains linear (required for activation patching and steering). Attribution is handled by causal ablation and game-theoretic Shapley values, which work correctly with nonlinear encoders without falling back to linearization.

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | How KAN-CLT works: encoder, decoder, activation, cross-layer structure |
| [Training](training.md) | Data collection, training loop, loss functions, hyperparameters |
| [Attribution](attribution.md) | Causal ablation, Shapley values, circuit graph construction |
| [Evaluation](evaluation.md) | Reconstruction accuracy, monosemanticity, spline analysis |
| [Getting Started](getting-started.md) | Setup, data collection, training, and evaluation walkthrough |
| [Metrics Reference](metrics.md) | Every metric: definitions, formulas, ideal values |
| [API Reference](api-reference.md) | Classes, methods, and function signatures |

## Project Structure

```
circuit-tracer/
├── kan_clt/                         # Core library
│   ├── kan_encoder.py               # KAN encoder (B-spline nonlinear)
│   ├── linear_encoder.py            # Linear encoder baseline
│   ├── kan_transcoder.py            # Full cross-layer transcoder
│   ├── utils.py                     # Parameter counting utilities
│   └── training/
│       ├── train.py                 # Training loop
│       ├── data.py                  # Activation dataset collection
│       └── loss.py                  # Loss functions
├── attribution/
│   ├── causal.py                    # Ablation-based attribution
│   ├── shapley.py                   # Monte Carlo Shapley attribution
│   └── graph.py                     # Circuit-tracer graph adapter
├── eval/
│   ├── replacement_accuracy.py      # Reconstruction quality metrics
│   └── monosemanticity.py           # Feature interpretability scoring
├── experiments/
│   ├── configs/                     # YAML training configs
│   ├── train_kan_clt.py             # Training entry point
│   ├── run_pipeline.py              # Full evaluation pipeline
│   ├── compare_models.py            # KAN vs linear comparison
│   ├── analyze_splines.py           # Spline shape visualization
│   └── run_circuit.py              # Single-prompt circuit tracing
└── tests/                           # 53 tests across 4 test files
```

## Key Concepts

- **KAN Encoder**: Uses B-spline basis functions to compute nonlinear feature pre-activations from the residual stream. Each input dimension is expanded into `grid_size + spline_order` basis functions before linear combination.

- **Cross-Layer Structure**: Each feature reads from one transformer layer but writes to all subsequent layers via separate decoder matrices. This captures how information flows across layers.

- **JumpReLU**: A ReLU variant with a learned threshold per feature. Provides the sparsity mechanism — only features with pre-activations above the threshold are active.

- **Jacobian Encoder Vectors**: Since the KAN encoder is nonlinear, encoder directions are input-dependent. The Jacobian `d(output)/d(input)` at each input point gives a local linear approximation for compatibility with existing attribution methods.

- **Causal Attribution**: Zero out one feature at a time, measure the change in downstream features and model output. Exact causal effect, no linearity assumptions.

- **Shapley Attribution**: Game-theoretic credit assignment via Monte Carlo permutation sampling. Correctly handles nonlinear feature interactions.
