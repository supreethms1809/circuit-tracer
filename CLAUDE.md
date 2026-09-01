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
│   ├── cli/                            # run_macag, run_baselines, annotate_graph, suggest_supernodes
│   ├── baselines/                      # Phase-2 baseline selectors (influence, EAP, Shapley/Banzhaf-gold, ACDC, bruteforce)
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
    ├── test_macag_annotate_graph.py     # MACAG annotation schema compatibility
    └── test_macag_baselines.py          # Baseline selectors + run_baselines harness (B2.0-B2.4, B3.2)
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
- `macag/cli/run_baselines.py`: head-to-head baseline harness (roadmap B2.0) — runs every selector on the same candidates + oracle, emits per-k evidence/scores, per-method oracle costs, precision@k/Jaccard vs Shapley-gold, faithfulness AUC, and Spearman linearity diagnostics
- `macag/baselines/`: baseline selectors over the same coalitional v(S) — `influence.py` (B2.1 top-k influence), `shapley_select.py` (B2.2 MC Shapley + Banzhaf over the MACAG oracle; NOT a wrapper of `attribution/shapley.py`), `eap.py` (B2.3 graph-derived EAP node scores), `acdc_prune.py` (B2.4 ported top-down τ-pruning), `bruteforce.py` (B3.2 exact best size-k subset / greedy optimality gap)
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
   - L_NMSE = Σ_l ||y_hat^l - y^l||² / Σ_l ||y^l||²   (scale-free; **not** Anthropic's raw L_MSE)
   - L_sparsity = λ Σ tanh(c · ||W_dec_i|| · a_i) / n_pos
   - L_kan_reg = `lambda_kan_reg` × spline_weight.abs().mean()  (KAN only — DO NOT use KANLinear.regularization_loss(), it produces NaN when spline_weight=0 via 0*log(0))
   - Total = L_NMSE + L_sparsity + L_kan_reg

   The reconstruction term is **normalized** by Σ||y||². With a raw `.mean()` MSE its
   magnitude tracks the base model's activation scale — `mean(y²)` is 0.75 for
   gpt2-large, 7.88 for gpt2-small, 24.67 for qwen3-0.6b — while L_sparsity is a
   per-token sum, so one λ meant three different things. That killed gpt2-large
   (L0 91816 → 0.9, ŷ ≈ 0) while qwen3-0.6b trained fine. L_NMSE = 1.0 at the
   zero-output baseline for every model.

   NMSE removes the **activation-scale dependence** and the collapse mode. It does
   **not** mean one λ yields the same L0 on every model — λ is still set per model to
   hit a target L0, as `λ_eff = λ · mean(y²)` is what the old objective saw:

   | model | mean(y²) | λ | λ·mean(y²) | resulting L0 |
   |---|---|---|---|---|
   | gpt2-small | 7.88 | 6.3e-4 | 5.0e-3 | ~104 |
   | gpt2-large | 0.75 | 6.3e-4 | 4.7e-4 | ~216 (probe 7809) |
   | qwen3-0.6b | 24.67 | **2.027e-4** | 5.0e-3 | ~105–137 |

   A shared 6.3e-4 made qwen's λ_eff 1.55e-2 — **3.1× too strong** — driving it to
   L0 29 (~1 feature/layer/token) at rel_fro 0.825. Fixed in job 7815.

   Within a model, λ **must** be identical for `encoder_type="kan"` and `"linear"` —
   a differing λ across arms confounds the spline-vs-linear comparison.

   **Loss values are not comparable across this change** (they shrink by mean(y²)),
   and `val_loss`/`best_val_loss` are in NMSE units too. Compare `rel_fro`,
   `nmse_mean`, and `L0`. See `logs/v2_collapse_evidence/FINDINGS.md`.

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

11. **TF32 is required, not optional.** The KAN encoder is deliberately fp32; without
    `torch.set_float32_matmul_precision("high")` its GEMMs fall back to CUDA-core SIMT
    SGEMM (profiled: 45.6 vs 141.3 TFLOPS on GH200). `TrainConfig.tf32=True` handles
    this; `_tf32_disabled()` restores true fp32 for `update_grid`'s lstsq refit.

12. **Diagnostic metrics cost a device sync.** `compute_losses(..., compute_metrics=False)`
    on non-log steps. The L0 mask must stay on-device — a `.cpu()` + `(>0).float()` over
    the activation tensor costs ~3.2 s/step single-threaded (torchrun sets `OMP_NUM_THREADS=1`).

13. **`collection_chunk_n_tokens` is bounded by /lscratch (894 GB), not by `n_tokens`.**
    gpt2-large activations are 180 KB/token, so 4M tokens ≈ 687 GiB per chunk. Exceeding
    the disk kills the job with SIGBUS (`exitcode: -7`), not ENOSPC.

14. **Do not enable DataLoader workers for the mmap activation datasets.** `num_workers>0`
    with `pin_memory=True` OOM-killed a gpt2-large job at MaxRSS 1.17 TB / 1.24 TiB: the
    mmap page cache is charged to the step's cgroup and page-locked buffers can't be
    reclaimed against it. Keep `num_workers=0`.

15. **Decoder init scale is per-model; set `decoder_init_strategy: "data_scaled"` on any
    new base model.** `_init_decoder_weights` uses `kaiming_uniform_` with
    `fan_in = d_model`, so ‖w_dec‖ ≈ √2 on every model regardless of its MLP-output
    scale. Encoder *inputs* are normalized (`enc_input_std`); the *targets* are not.
    So initial ‖ŷ‖ is model-independent while ‖y‖ is not:

    | model | mean(y²) | initial per-layer FVU | outcome |
    |---|---|---|---|
    | gpt2-small | 9.58 | 1.7 | trains (dip to l0_min 0.56, recovers) |
    | qwen3-0.6b | 24.67 | 7.6 | trains (dip to l0_min 0.17, recovers) |
    | llama-3.2-1B | **0.087** | **105** | **dies** (l0_min → 0.0 at step 480) |
    | gemma3-1b | 2804 | ≈1 (ŷ ≈ 0) | inverted failure, still unfixed |

    gpt2/qwen land near ‖ŷ‖/‖y‖ ≈ 1 **by luck**, not design. When ŷ starts ~10× hot the
    fastest descent is to shrink it toward zero, dragging preactivations under θ; JumpReLU's
    `(x > θ)` gate then gives zero gradient and the read-layer is permanently dead.

    `"data_scaled"` runs `calibrate_decoder_scale_from_data` on cold start (after input
    normalization and threshold calibration, before the FSDP wrap), measuring ‖y_l‖ and
    ‖ŷ_l‖ on a seeded sample and rescaling each target layer so ‖ŷ_l‖ ≈ ‖y_l‖. Resulting
    per-layer FVU ≈ 2 on any model. Default `"kaiming"` preserves old behaviour.

    **This is not a λ problem** — diagnose it with `threshold_mean` (frozen if the sparsity
    term is inert) and step-0 `reconstruction/rel_fro_per_layer_mean` (>> 1.5 means hot init),
    not by tuning `lambda_sparsity`. `recon_layer_energy_beta` is also the wrong lever: llama's
    layer-energy dispersion (278×, L1 48.7% + L15 41.3%) is the same *shape* as gpt2's
    (110×, L2 48.0% + L11 37.6%), so it cannot be what distinguishes them.

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
    tests/test_macag_annotate_graph.py \
    tests/test_macag_baselines.py -q

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
