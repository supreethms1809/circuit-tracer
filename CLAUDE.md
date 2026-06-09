# Spline Circuit Tracer

## Project Goal
Build a Spline-CLT that replaces the linear encoder in Anthropic's circuit tracing pipeline with a spline/KAN encoder, enabling nonlinear feature detection while preserving circuit tracing capability. This is a PhD research project centered on whether nonlinear feature boundaries improve mechanistic interpretability and downstream circuit faithfulness.

## Core Research Hypothesis
Anthropic's CLT assumes linear feature boundaries (linear encoder → JumpReLU). This misses features defined by nonlinear activation patterns. A KAN encoder can capture these, and game-theoretic attribution (Shapley values) handles the resulting nonlinear feature interactions without falling back to linearization.

## End-to-End Deliverables
This repository now supports three linked outputs from the same linear or Spline-CLT checkpoint:
1. **Config-driven paper evaluation**: a conference runner that expands dataset collection, training, evaluation, MACAG, aggregation, and report generation from one JSON suite.
2. **Circuit-tracer graph + MACAG**: build a graph, then run evidence-allocation games and dashboard/report generation.
3. **Exact Shapley attribution**: compute nonlinear feature attributions directly from the checkpointed CLT on the same prompt.

## Architecture

### Standard CLT (Anthropic's approach)
```
a^l = JumpReLU(W_enc^l · x^l)           # linear encoder
y_hat^l = Σ W_dec^(l'→l) · a^(l')       # linear decoder, cross-layer
```

### Spline-CLT (our approach)
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
├── spline_clt/
│   ├── kan_encoder.py                   # KAN encoder wrapping efficient-kan
│   ├── kan_transcoder.py                # Full Spline-CLT module (encoder_type="kan"|"linear")
│   ├── linear_encoder.py                # Linear encoder baseline (same interface as KANEncoder)
│   ├── paper_eval.py                    # CLI entrypoint for paper-eval
│   ├── seed.py                          # Shared deterministic seeding helpers
│   ├── paper/
│   │   ├── config.py                    # JSON + OmegaConf suite loading and validation
│   │   ├── evaluate.py                  # Prompt/graph evaluation helpers
│   │   ├── reporting.py                 # Suite aggregation + report/table generation
│   │   └── runner.py                    # End-to-end paper suite runner
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
│   │   ├── gpt2_small.yaml              # Spline-CLT training config
│   │   └── gpt2_small_linear_baseline.yaml  # Matched linear CLT baseline
│   ├── paper_configs/
│   │   ├── base/                        # Shared paper defaults
│   │   ├── models/                      # Spline + linear paper variants
│   │   ├── benchmarks/                  # Curated benchmark manifests
│   │   ├── macag/                       # MACAG defaults
│   │   └── suites/                      # Runnable paper presets
│   ├── train_spline_clt.py                 # Main training entry point
│   ├── run_pipeline.py                  # ← Full evaluation pipeline (run this)
│   ├── compare_models.py                # Spline-CLT vs linear CLT evaluation table
│   ├── analyze_splines.py               # Spline shape extraction and visualization
│   └── run_circuit.py                   # Single-prompt circuit tracing
├── docs/
│   └── paper-evaluation.md              # How to run paper-eval and read outputs
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
    ├── test_kan_encoder.py
    ├── test_kan_transcoder.py
    ├── test_attribution.py
    ├── test_graph.py
    ├── test_shapley.py
    ├── test_paper_config.py             # Paper config validation
    ├── test_paper_reporting.py          # Suite aggregation/reporting
    ├── test_paper_runner.py             # Paper runner dry-run + smoke
    ├── test_replacement_model_alignment.py  # BOS / intervention alignment regressions
    └── test_macag_annotate_graph.py     # MACAG annotation schema compatibility
```

## Conference Runner

The canonical NeurIPS interface is now `paper-eval` (`spline_clt/paper_eval.py`). Required: `--suite`. Common: `--dry-run`, `--validate-only`. Operational extras: `--re-evaluate` (drop eval/MACAG artifacts, keep checkpoints), `--worker-id` / `--num-workers`, `--stages` to intersect with the suite’s enabled stages.

All conference runs are defined by JSON config under `experiments/paper_configs/` and merged with OmegaConf defaults. For the **final campaign**, do not replace suite-chosen seeds or hyperparameters with ad hoc CLI overrides; use `--stages` / worker sharding only for infrastructure.

Metric equations and methodology comparisons live in `docs/metric_definitions.md` and `docs/methodology_comparison.md`.

### Runnable Paper Suites
- `experiments/paper_configs/suites/neurips_core_gpt2.json`
- `experiments/paper_configs/suites/neurips_high_gpt2.json`
- `experiments/paper_configs/suites/neurips_high_gpt2_pythia160m.json`
- `experiments/paper_configs/suites/macag_case_studies.json`
- `experiments/paper_configs/suites/smoke_trial_gpt2.json`

### Paper Runner Outputs
Each suite writes:
- `resolved_config.json`
- `manifest.json`
- `per_example_metrics.jsonl`
- `aggregate_metrics.json`
- `tables.csv`
- `report.md`
- `figures/`

### Current Evaluation Coverage
The paper runner now reports:
- reconstruction: `mse_total`, `cosine_similarity`, `relative_error`, `active_per_pos`, `activation_density`
- replacement fidelity: `top1_match_rate`, `kl_divergence`
- circuit faithfulness: `keep_only_gap_ratio`, `gap_drop_ratio`, `shapley_causal_jaccard`
- graph quality / error reliance: `graph_replacement_score`, `graph_completeness_score`, `retained_feature_node_count`, `retained_error_node_count`, `retained_error_node_fraction`
- monosemanticity: mean/median Gini and threshold fractions
- MACAG Game 1 / Game 2 aggregate metrics and stability

MACAG is a first-party top-level package in this repository. Use `macag/...`
from the repo root. The current CLI entrypoint is `python -m macag.cli.run_macag`,
and real circuit interventions should be wired through
`macag.factories.replacement_model:create_replacement_model_oracle`.

`macag.factories.replacement_model` can build a scorer from either:
- a hub `transcoder_set`, or
- a local checkpoint path, including both standard CLT and Spline-CLT checkpoints.

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
- `local_clt_path` auto-detects Spline-CLT vs standard CLT:
  - if the checkpoint directory contains `metadata.safetensors`, it loads through
    `spline_clt.kan_transcoder.load_spline_clt`
  - otherwise it loads through
    `circuit_tracer.transcoder.cross_layer_transcoder.load_clt`
- Candidate interventions are restricted to feature nodes present in the graph JSON
  (default `feature_type == "cross layer transcoder"`), so MACAG and the scorer stay
  aligned to the traced circuit.

### Recent Integration Fixes
- `kan_clt` was fully renamed to `spline_clt` in tracked code and entrypoints.
- Graph export now uses the correct `selected_features` index convention expected by `circuit_tracer.Graph`.
- ReplacementModel intervention buffers now use the same BOS-prepended tokenization path as graph/prompt evaluation, fixing prompt-length mismatches in MACAG.
- MACAG annotation now tolerates both legacy Game 1 evidence lists and dict-form evidence payloads.
- Paper aggregation now exposes explicit retained error-node metrics instead of hiding them behind circuit graphs only.
- MACAG `connected` constraint no longer routes connectivity through logit/embedding hub nodes (it was vacuous before: every pruned graph is one weak component through the logits). Error nodes remain valid intermediates.
- Game 1 `raw_relative` stop now compares λ-free faithfulness gains, not λ-penalized utility gains, so the sparsity penalty cannot distort the eps-relative test.
- Empty ablation universes (candidate/graph node-ID mismatches) now raise instead of silently degenerating to `empty == all`; `score_remove` is intersected with the universe, symmetric with `score_keep_only`.
- Solver oracle stats are per-solve (reset at solver entry); Game 2 reports `best_iteration` (0 = initial empty allocation won); subgraphs serialize in sorted order; `CircuitGraph.from_dict` prefers `node_id` over `id` to match the intervention loader.

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

6. **Parameter ratio**: Spline-CLT has ~10x more encoder parameters than linear CLT at matched d_transcoder (due to B-spline basis expansion). Decoder is identical.

7. **Data loading**: Dataset is ~40GB total. Two modes:
   - `ActivationDataset.load(path)` (no `max_samples`) returns a **streaming**
     dataset that keeps the mmap open and copies one sample per `__getitem__`
     call. RAM cost is O(batch). This is the training path — it sees every
     collected sequence.
   - `ActivationDataset.load(path, max_samples=N)` copies an `N`-sequence slice
     into contiguous RAM (~14 GB for N=3000 on GPT-2 small). Used by paper
     evaluation / analysis scripts that want a bounded deterministic subset.
   Avoid loading the full dataset with a large explicit `max_samples` — causes
   OOM or SIGBUS on WSL2.

8. **encoder_type**: `KANCrossLayerTranscoder(encoder_type="kan"|"linear")`. Both use the same interface. `to_safetensors`/`load_spline_clt` handle both correctly.

9. **Determinism**: seed all train/val splits, dataset shuffling, evaluation subsets, spline sampling, monosemanticity sampling, Shapley sampling, and bootstrap reporting. The paper runner is intended to be exactly reproducible from the suite JSON.

10. **Graph metrics**: retained error nodes are expected and are now reported explicitly. A zero `gap_drop_ratio` does not mean error nodes are absent; it means removing the selected top-k feature nodes did not move the target-foil gap on that prompt.

## Code Style
- PyTorch throughout
- Type hints on function signatures
- Docstrings on public methods
- Config via dataclasses or YAML
- Tests for all core components

## Common Commands

```bash
# --- Paper runner ---
# Validate a conference suite
conda run -n ct python -m spline_clt.paper_eval \
    --suite experiments/paper_configs/suites/neurips_core_gpt2.json \
    --validate-only

# Expand the job matrix without running
conda run -n ct python -m spline_clt.paper_eval \
    --suite experiments/paper_configs/suites/neurips_core_gpt2.json \
    --dry-run

# End-to-end smoke validation of the full paper pipeline
conda run -n ct python -m spline_clt.paper_eval \
    --suite experiments/paper_configs/suites/smoke_trial_gpt2.json

# Core NeurIPS GPT-2 suite
conda run -n ct python -m spline_clt.paper_eval \
    --suite experiments/paper_configs/suites/neurips_core_gpt2.json

# --- Testing ---
# Focused regression suite for paper runner + integrations
conda run -n ct pytest \
    tests/test_graph.py \
    tests/test_attribution.py \
    tests/test_paper_config.py \
    tests/test_paper_reporting.py \
    tests/test_paper_runner.py \
    tests/test_replacement_model_alignment.py \
    tests/test_macag_annotate_graph.py -q

# Run core Spline-CLT tests
conda run -n ct pytest tests/test_kan_encoder.py tests/test_kan_transcoder.py \
    tests/test_attribution.py tests/test_shapley.py -v

# --- Full evaluation pipeline (run this once training is complete) ---
conda run -n ct python experiments/run_pipeline.py \
    --kan-checkpoint    checkpoints/gpt2_small/spline_clt_gpt2_best \
    --linear-checkpoint checkpoints/gpt2_small_linear/linear_clt_gpt2_best \
    --data-dir data/activations \
    --output-dir results/eval_run1 \
    --shapley \
    --no-plot           # omit for matplotlib PNG plots

# Spline-CLT only, skip slow stages:
conda run -n ct python experiments/run_pipeline.py \
    --kan-checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
    --data-dir data/activations \
    --skip-circuits --skip-monosemanticity

# Run everything including upstream circuit-tracer tests
conda run -n ct pytest tests/ -v

# --- Data collection ---
# Collect GPT-2 small activations (~40GB, takes ~30 min on GPU)
conda run -n ct python experiments/train_spline_clt.py \
    --collect-data --model gpt2 --device cuda

# --- Training ---
# Train Spline-CLT
conda run -n ct python experiments/train_spline_clt.py \
    --config experiments/configs/gpt2_small.yaml

# Train linear CLT baseline
conda run -n ct python experiments/train_spline_clt.py \
    --config experiments/configs/gpt2_small_linear_baseline.yaml

# --- Evaluation ---
# Compare Spline-CLT vs linear CLT (needs both checkpoints)
conda run -n ct python experiments/compare_models.py \
    --kan-checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
    --linear-checkpoint checkpoints/gpt2_small_linear/linear_clt_gpt2_best \
    --data-dir data/activations \
    --n-samples 200

# --- Circuit tracing ---
# End-to-end circuit trace for a prompt
conda run -n ct python experiments/run_circuit.py \
    --checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
    --prompt "The Eiffel Tower is located in" \
    --model gpt2 \
    --max-features 64 \
    --output results/circuits/eiffel.pt

# Same with Shapley attribution (slower)
conda run -n ct python experiments/run_circuit.py \
    --checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
    --prompt "The Eiffel Tower is located in" \
    --shapley --shapley-samples 128

# --- Spline analysis (KAN encoder only) ---
conda run -n ct python experiments/analyze_splines.py \
    --checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
    --n-features 20 \
    --output-dir results/splines
```

## Current Status
- `spline_clt` package rename is complete in tracked code.
- Config-driven paper runner is implemented and documented.
- NeurIPS suite configs exist for core/high GPT-2 runs, high-budget Pythia runs, MACAG case studies, and a CPU smoke trial.
- The smoke trial now runs end to end: dataset collection, training, prompt evaluation, graph export, MACAG, aggregation, and report generation.
- Recent regressions fixed: graph selected-feature indexing, BOS alignment for ReplacementModel interventions, MACAG annotation schema compatibility, and explicit error-node reporting.
- Focused paper/integration regression suite is currently passing locally.
- The next scientific step is to run `neurips_core_gpt2.json` on GH200-class hardware and use the resulting aggregate tables to decide whether the spline claim clears the conference bar.
