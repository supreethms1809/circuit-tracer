# KAN Circuit Tracer

## Project Goal
Build a KAN-based Cross-Layer Transcoder (KAN-CLT) that replaces the linear encoder in Anthropic's circuit tracing pipeline with a KAN encoder, enabling nonlinear feature detection while preserving circuit tracing capability. This is a PhD research project exploring whether nonlinear feature boundaries improve mechanistic interpretability.

## Core Research Hypothesis
Anthropic's CLT assumes linear feature boundaries (linear encoder → JumpReLU). This misses features defined by nonlinear activation patterns. A KAN encoder can capture these, and game-theoretic attribution (Shapley values) handles the resulting nonlinear feature interactions without falling back to linearization.

## Architecture

### Standard CLT (Anthropic's approach)
```
a^l = JumpReLU(W_enc^l · x^l)           # linear encoder
y_hat^l = Σ W_dec^(l'→l) · a^(l')       # linear decoder, cross-layer
```

### KAN-CLT (our approach)
```
a^l = JumpReLU(KAN_enc^l(x^l))          # KAN encoder (nonlinear)
y_hat^l = Σ W_dec^(l'→l) · a^(l')       # linear decoder stays the same
```

The decoder MUST remain linear — features need clean directions in residual stream space for activation patching and steering to work.

### Attribution Difference
- Anthropic: edge weight = a_s · w_{s→t} (exact, because encoder is linear)
- Ours: feature interactions are nonlinear → use causal interventions or Shapley values instead of backward Jacobians
- Jacobian-based encoder directions (local linear approximation) are available for compatibility with the existing attribution pipeline

## Key Dependencies
- `circuit-tracer` (forked to github.com/supreethms1809/circuit-tracer) — provides ReplacementModel, graph pruning, visualization
- `efficient-kan` (github.com/Blealtan/efficient-kan) — fast B-spline KAN implementation
- `transformer-lens` — model hooking (already in circuit-tracer)
- Target model: GPT-2 small (standard mechinterp testbed, 12 layers, 768-dim residual stream)

## Project Structure
```
circuit-tracer/                          # fork of safety-research/circuit-tracer
├── CLAUDE.md                            # this file
├── TASKS.md                             # phased implementation checklist
├── GH200_SETUP.md                       # GH200 node setup guide
├── kan_clt/
│   ├── kan_encoder.py                   # KAN encoder wrapping efficient-kan
│   ├── kan_transcoder.py                # Full KAN-CLT module (CLT-compatible interface)
│   ├── utils.py                         # Parameter counting, comparison utilities
│   └── training/
│       ├── train.py                     # Training loop with Adam + cosine decay
│       ├── data.py                      # Activation dataset collection from GPT-2
│       └── loss.py                      # Reconstruction + sparsity losses
├── attribution/
│   ├── causal.py                        # Ablation-based causal attribution
│   └── graph.py                         # Graph adapter for circuit-tracer format
├── eval/
│   └── replacement_accuracy.py          # Top-1 match, KL divergence, sparsity stats
├── experiments/
│   ├── configs/gpt2_small.yaml          # Default training config
│   └── train_kan_clt.py                 # Main training entry point
├── circuit_tracer/                      # upstream circuit-tracer library (unmodified)
└── tests/
    ├── test_kan_encoder.py              # 14 tests
    ├── test_kan_transcoder.py           # 19 tests
    └── test_attribution.py              # 6 tests
```

## Important Technical Notes

1. **efficient-kan reformulation**: A KAN layer with grid_size G and spline_order k expands each input dimension to (G+k) basis activations, then linearly combines them. So for 768 inputs with G=5, k=3, you get 768×8=6144 basis features → linear map to num_features.

2. **JumpReLU**: Anthropic uses JumpReLU (ReLU with a learned threshold) as the activation on feature pre-activations. Keep this — it provides the sparsity mechanism. The KAN nonlinearity is BEFORE this, in how the pre-activation is computed from the residual stream.

3. **Cross-layer structure**: Each feature reads from ONE layer but writes to ALL subsequent layers via separate decoder weight matrices. This is critical for the attribution graph structure.

4. **Jacobian-based encoder vectors**: For attribution, KANEncoder.get_encoder_vectors() computes d(output[f])/d(input) at each input point. This gives local linear encoder directions that plug into the existing AttributionContext. Verified correct via finite difference tests.

5. **Training objective**: Same as Anthropic's CLT:
   - L_MSE = Σ_l ||y_hat^l - y^l||^2
   - L_sparsity = λ Σ tanh(c · ||W_dec_i|| · a_i)
   - L_kan_reg = KAN L1 + entropy regularization on spline weights
   - Total = L_MSE + L_sparsity + L_kan_reg

6. **Parameter ratio**: KAN-CLT has ~2.2x more parameters than linear CLT at matched dimensions due to B-spline basis expansion. This is expected and tractable.

## Code Style
- PyTorch throughout
- Type hints on function signatures
- Docstrings on public methods
- Config via dataclasses or simple YAML, not argparse spaghetti
- Tests for core components (encoder, transcoder, attribution)

## Common Commands
```bash
# Run all KAN-CLT tests (39 tests)
pytest tests/test_kan_encoder.py tests/test_kan_transcoder.py tests/test_attribution.py -v

# Collect activations from GPT-2
python experiments/train_kan_clt.py --collect-data --model gpt2 --device cuda

# Train KAN-CLT
python experiments/train_kan_clt.py --config experiments/configs/gpt2_small.yaml

# Run all tests including upstream circuit-tracer
pytest tests/ -v
```

## Current Status
- Phases 1-2 complete: KAN encoder, KAN transcoder, training pipeline
- Phase 4.1 complete: Ablation-based causal attribution with Jacobian encoder directions
- 39/39 tests passing, 0 regressions on upstream circuit-tracer tests
- Next: Phase 3 (train on GPT-2 activations on GH200), Phase 4.2 (Shapley values)
