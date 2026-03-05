# KAN Circuit Tracer

## Project Goal
Build a KAN-based Cross-Layer Transcoder (KAN-CLT) that replaces the linear encoder in Anthropic's circuit tracing pipeline with a KAN encoder, enabling nonlinear feature detection while preserving circuit tracing capability. This is a PhD research project exploring whether nonlinear feature boundaries improve mechanistic interpretability.

## Core Research Hypothesis
Anthropic's CLT assumes linear feature boundaries (linear encoder → JumpReLU). This misses features defined by nonlinear activation patterns. A KAN encoder can capture these, and game-theoretic attribution (Shapley values) handles the resulting nonlinear feature interactions without falling back to linearization.

## End-to-End Deliverables
This repository now supports two linked outputs from the same linear or KAN-CLT checkpoint:
1. **Circuit-tracer graph + MACAG**: build a graph, then run evidence-allocation games and dashboard/report generation.
2. **Exact Shapley attribution**: compute nonlinear feature attributions directly from the checkpointed CLT on the same prompt.

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

### Attribution Methods
- **Causal ablation** (`attribution/causal.py`): zero out one feature at a time, measure downstream change. Exact causal effect, O(n_active) forward passes.
- **Shapley values** (`attribution/shapley.py`): Monte Carlo permutation sampling with antithetic pairs. Game-theoretically sound for nonlinear interactions. O(n_active × n_samples) forward passes.
- **Jacobian encoder vectors**: local linear approximation of KAN encoder direction at each input point. Available for compatibility with existing circuit-tracer attribution pipeline.

## Key Dependencies
- `circuit-tracer` (forked to github.com/supreethms1809/circuit-tracer) — provides ReplacementModel, graph pruning, visualization
- `efficient-kan` (github.com/Blealtan/efficient-kan) — fast B-spline KAN implementation
- `macag` — graph-first allocation games and intervention validation over circuit-tracer graphs
- `transformer-lens` — model hooking for activation collection
- Target model: GPT-2 small (12 layers, 768-dim residual stream)

## Project Structure
```
circuit-tracer/                          # fork of safety-research/circuit-tracer
├── CLAUDE.md                            # this file
├── TASKS.md                             # phased implementation checklist
├── GH200_SETUP.md                       # GH200 node setup guide
├── kan_clt/
│   ├── kan_encoder.py                   # KAN encoder wrapping efficient-kan
│   ├── kan_transcoder.py                # Full KAN-CLT module (encoder_type="kan"|"linear")
│   ├── linear_encoder.py                # Linear encoder baseline (same interface as KANEncoder)
│   ├── utils.py                         # Parameter counting, comparison utilities
│   └── training/
│       ├── train.py                     # Training loop with Adam + cosine decay
│       ├── data.py                      # Activation dataset collection (mmap + clone)
│       └── loss.py                      # Reconstruction + sparsity + KAN reg losses
├── attribution/
│   ├── causal.py                        # Ablation-based causal attribution
│   ├── shapley.py                       # Monte Carlo Shapley attribution
│   └── graph.py                         # Graph adapter for circuit-tracer format
├── eval/
│   ├── replacement_accuracy.py          # Top-1 match, KL divergence, sparsity stats
│   └── monosemanticity.py               # Gini coefficient, max-activating examples
├── experiments/
│   ├── configs/
│   │   ├── gpt2_small.yaml              # KAN-CLT training config
│   │   └── gpt2_small_linear_baseline.yaml  # Matched linear CLT baseline
│   ├── train_kan_clt.py                 # Main training entry point
│   ├── run_pipeline.py                  # ← Full evaluation pipeline (run this)
│   ├── compare_models.py                # KAN-CLT vs linear CLT evaluation table
│   ├── analyze_splines.py               # Spline shape extraction and visualization
│   └── run_circuit.py                   # Single-prompt circuit tracing
├── macag/
│   ├── cli/                            # run_macag, annotate_graph, suggest_supernodes
│   ├── factories/                      # ReplacementModel-backed MACAG scorer builders
│   ├── games/                          # Game 1 / Game 2 solvers
│   ├── graph.py                        # circuit-tracer graph wrapper
│   ├── scoring.py                      # Scoring oracles + intervention adapters
│   ├── utils/                          # Metrics + supernode helpers
│   └── README.md                       # MACAG usage notes
├── circuit_tracer/                      # core circuit-tracer library
└── tests/
    ├── test_kan_encoder.py              # 14 tests
    ├── test_kan_transcoder.py           # 20 tests (includes linear encoder save/load)
    ├── test_attribution.py              # 6 tests
    └── test_shapley.py                  # 13 tests
```

MACAG is a first-party top-level package in this repository. Use `macag/...`
from the repo root. The current CLI entrypoint is `python -m macag.cli.run_macag`,
and real circuit interventions should be wired through
`macag.factories.replacement_model:create_replacement_model_oracle`.

`macag.factories.replacement_model` can build a scorer from either:
- a hub `transcoder_set`, or
- a local checkpoint path, including both standard CLT and KAN-CLT checkpoints.

## MACAG Integration

MACAG is the graph-first downstream analysis layer. It does not replace
`circuit_tracer` graph generation; it consumes a circuit graph JSON, runs
evidence-allocation games over the feature nodes, and can write an annotated
graph back for the existing `circuit_tracer` frontend.

### MACAG Code Map
- `macag/cli/run_macag.py`: main CLI for Game 1 (`game1`) and Game 2 (`game2`)
- `macag/cli/annotate_graph.py`: merges MACAG outputs back into a graph JSON for UI inspection
- `macag/cli/suggest_supernodes.py`: auto-generates candidate supernodes from graph metrics
- `macag/factories/replacement_model.py`: bridge from MACAG scoring into `circuit_tracer.ReplacementModel`
- `macag/graph.py`: circuit graph wrapper used by the game solvers
- `macag/scoring.py`: scoring oracles and intervention backend adapters
- `macag/games/`: optimization logic for Game 1 / Game 2
- `macag/utils/`: metrics and supernode helper utilities

### Current Integration Path
1. Generate or load a graph JSON from `circuit_tracer`.
2. Run `python -m macag.cli.run_macag ...` on that graph.
3. For real interventions, pass
   `--oracle-factory macag.factories.replacement_model:create_replacement_model_oracle`
   plus an oracle kwargs JSON file.
4. Use `python -m macag.cli.annotate_graph ...` to write an annotated graph JSON.
5. Open the annotated graph in the existing `circuit_tracer` server/frontend.

### ReplacementModel Wiring
- `macag.factories.replacement_model.create_replacement_model_oracle` is the canonical
  integration entrypoint.
- It supports hub-backed loading via `transcoder_set`.
- It also supports local checkpoints via `local_clt_path`.
- `local_clt_path` auto-detects KAN-CLT vs standard CLT:
  - if the checkpoint directory contains `metadata.safetensors`, it loads through
    `kan_clt.kan_transcoder.load_kan_clt`
  - otherwise it loads through
    `circuit_tracer.transcoder.cross_layer_transcoder.load_clt`
- Candidate interventions are restricted to feature nodes present in the graph JSON
  (default `feature_type == "cross layer transcoder"`), so MACAG and the scorer stay
  aligned to the traced circuit.

### Operational Notes
- `macag/cache` and `macag/output` are runtime artifacts, not source code.
- `macag/docs` and `macag/examples` are reference material; the active Python package is
  the top-level `macag/` package shown above.
- If you are committing or pushing changes, avoid accidentally staging large cached model
  artifacts from `macag/cache`.

## Important Technical Notes

1. **efficient-kan reformulation**: A KAN layer with grid_size G and spline_order k expands each input dimension to (G+k) basis activations, then linearly combines them. So for 768 inputs with G=5, k=3, you get 768×8=6144 basis features → linear map to num_features.

2. **JumpReLU**: Anthropic uses JumpReLU (ReLU with a learned threshold) as the activation on feature pre-activations. Keep this — it provides the sparsity mechanism. The KAN nonlinearity is BEFORE this, in how the pre-activation is computed from the residual stream.

3. **Cross-layer structure**: Each feature reads from ONE layer but writes to ALL subsequent layers via separate decoder weight matrices. This is critical for the attribution graph structure.

4. **Jacobian-based encoder vectors**: For attribution, KANEncoder.get_encoder_vectors() computes d(output[f])/d(input) at each input point. This gives local linear encoder directions that plug into the existing AttributionContext.

5. **Training objective**:
   - L_MSE = Σ_l ||y_hat^l - y^l||^2
   - L_sparsity = λ Σ tanh(c · ||W_dec_i|| · a_i)
   - L_kan_reg = 0.01 × spline_weight.abs().mean()  (KAN only — DO NOT use KANLinear.regularization_loss(), it produces NaN when spline_weight=0 via 0*log(0))
   - Total = L_MSE + L_sparsity + L_kan_reg

6. **Parameter ratio**: KAN-CLT has ~10x more encoder parameters than linear CLT at matched d_transcoder (due to B-spline basis expansion). Decoder is identical.

7. **Data loading**: Dataset is ~40GB total. Use `ActivationDataset.load(path, max_samples=3000)` which mmap-slices then clones into RAM (~14GB). Avoid loading the full dataset — causes OOM or SIGBUS on WSL2.

8. **encoder_type**: `KANCrossLayerTranscoder(encoder_type="kan"|"linear")`. Both use the same interface. `to_safetensors`/`load_kan_clt` handle both correctly.

## Code Style
- PyTorch throughout
- Type hints on function signatures
- Docstrings on public methods
- Config via dataclasses or YAML
- Tests for all core components

## Common Commands

```bash
# --- Testing ---
# Run all KAN-CLT tests (53 tests)
conda run -n ct pytest tests/test_kan_encoder.py tests/test_kan_transcoder.py \
    tests/test_attribution.py tests/test_shapley.py -v

# --- Full evaluation pipeline (run this once training is complete) ---
conda run -n ct python experiments/run_pipeline.py \
    --kan-checkpoint    checkpoints/gpt2_small/kan_clt_gpt2_best \
    --linear-checkpoint checkpoints/gpt2_small_linear/linear_clt_gpt2_best \
    --data-dir data/activations \
    --output-dir results/eval_run1 \
    --shapley \
    --no-plot           # omit for matplotlib PNG plots

# KAN-CLT only, skip slow stages:
conda run -n ct python experiments/run_pipeline.py \
    --kan-checkpoint checkpoints/gpt2_small/kan_clt_gpt2_best \
    --data-dir data/activations \
    --skip-circuits --skip-monosemanticity

# Run everything including upstream circuit-tracer tests
conda run -n ct pytest tests/ -v

# --- Data collection ---
# Collect GPT-2 small activations (~40GB, takes ~30 min on GPU)
conda run -n ct python experiments/train_kan_clt.py \
    --collect-data --model gpt2 --device cuda

# --- Training ---
# Train KAN-CLT
conda run -n ct python experiments/train_kan_clt.py \
    --config experiments/configs/gpt2_small.yaml

# Train linear CLT baseline
conda run -n ct python experiments/train_kan_clt.py \
    --config experiments/configs/gpt2_small_linear_baseline.yaml

# --- Evaluation ---
# Compare KAN-CLT vs linear CLT (needs both checkpoints)
conda run -n ct python experiments/compare_models.py \
    --kan-checkpoint checkpoints/gpt2_small/kan_clt_gpt2_best \
    --linear-checkpoint checkpoints/gpt2_small_linear/linear_clt_gpt2_best \
    --data-dir data/activations \
    --n-samples 200

# --- Circuit tracing ---
# End-to-end circuit trace for a prompt
conda run -n ct python experiments/run_circuit.py \
    --checkpoint checkpoints/gpt2_small/kan_clt_gpt2_best \
    --prompt "The Eiffel Tower is located in" \
    --model gpt2 \
    --max-features 64 \
    --output results/circuits/eiffel.pt

# Same with Shapley attribution (slower)
conda run -n ct python experiments/run_circuit.py \
    --checkpoint checkpoints/gpt2_small/kan_clt_gpt2_best \
    --prompt "The Eiffel Tower is located in" \
    --shapley --shapley-samples 128

# --- Spline analysis (KAN encoder only) ---
conda run -n ct python experiments/analyze_splines.py \
    --checkpoint checkpoints/gpt2_small/kan_clt_gpt2_best \
    --n-features 20 \
    --output-dir results/splines
```

## Current Status
- **53/53 tests passing**
- Phases 1-4 complete: KAN encoder, KAN transcoder, training pipeline, linear baseline, causal attribution, Shapley attribution
- Phase 4.3 verified: `attribution/graph.py` API matches `circuit_tracer.graph.Graph` exactly
- Training running on GH200 (separate machine) — checkpoints not yet available locally
- Next: Phase 5 evaluation once checkpoints are ready (compare_models.py, analyze_splines.py, run_circuit.py are all ready to run)
