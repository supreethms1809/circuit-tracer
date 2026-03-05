# Experiments and Results Template

Use this document to run, record, and report MACAG experiments for your chapter.

## 1. Core Research Questions

- RQ1: Can Game 1 recover sparse evidence while preserving target behavior?
- RQ2: Can Game 2 separate shared vs target-unique vs foil-unique evidence?
- RQ3: How sensitive are outcomes to \(\alpha,\lambda,\beta\), budget, and candidate filtering?
- RQ4: What is the runtime/caching profile of intervention-based optimization?

## 2. Minimum Experiment Matrix

For each prompt:

- Game 1: vary \(\lambda\in\{0.001,0.002,0.005,0.01\}\), budget \(\in\{10,20,30\}\).
- Game 2: vary \(\beta\in\{0,0.1,0.25,0.5\}\) with fixed \(\lambda\), then vary \(\lambda\).
- Compare full candidates vs prefiltered candidates.

## 3. Reference Run Commands

## Full Game 2 (CPU, no candidate prefilter)

```bash
cd /path/to/circuit-tracer

PYTHONPATH=. python -m macag.cli.run_macag game2 \
  --graph-json /path/to/circuit-tracer/macag/output/macag-g426k-denver-20260214-191122.json \
  --target y \
  --foil y_foil \
  --input-id colorado_vs_wyoming_full_cpu \
  --alpha 0.6 \
  --lam 0.002 \
  --beta 0.25 \
  --abr-iters 12 \
  --budget 30 \
  --min-gain 1e-5 \
  --oracle-factory macag.factories.replacement_model:create_replacement_model_oracle \
  --oracle-kwargs-json '{"model_name":"google/gemma-2-2b","transcoder_set":"mntss/clt-gemma-2-2b-426k","prompt":"The capital of the state containing Denver is","graph_json":"/path/to/circuit-tracer/macag/output/macag-g426k-denver-20260214-191122.json","backend":"transformerlens","score_kind":"logit_gap","target_token_by_label":{"y":" Colorado","y_foil":" Wyoming"},"foil_by_target":{"y":"y_foil","y_foil":"y"},"freeze_attention":true,"model_kwargs":{"dtype":"bf16"}}' \
  --output-json /path/to/circuit-tracer/macag/output/macag_game2_426k_full_colorado_vs_wyoming_cpu.json
```

## Annotate for visualization

```bash
PYTHONPATH=. python -m macag.cli.annotate_graph \
  --graph-json /path/to/circuit-tracer/macag/output/macag-g426k-denver-20260214-191122.json \
  --macag-result-json /path/to/circuit-tracer/macag/output/macag_game2_426k_full_colorado_vs_wyoming_cpu.json \
  --output-json /path/to/circuit-tracer/macag/output/macag-g426k-denver-20260214-191122_macag_game2_full_colorado_vs_wyoming_cpu.json \
  --replace-existing-supernodes \
  --replace-existing-pins
```

## 4. Metrics to Report

From result JSON:

- Set size metrics:
  - \(|E_y|\), \(|E_{foil}|\), \(|shared|\), \(|unique_y|\), \(|unique_{foil}|\)
- Faithfulness:
  - `all`, `empty`, `keep_only`, `remove`, `faithfulness`
- Utilities:
  - `utility_target`, `utility_foil`
- Efficiency:
  - `oracle_calls`, `cache_hits`, `cache_size`, `iterations`, `converged`

## 5. Table Templates

## Table A: Main Game 2 Results

| Input ID | alpha | lambda | beta | |E_y| | |E_foil| | |shared| | overlap_rate | faithfulness_y | faithfulness_foil | oracle_calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

## Table B: Hyperparameter Sensitivity

| beta | |shared| | |unique_y| | |unique_foil| | utility_target | utility_foil |
|---:|---:|---:|---:|---:|---:|
| 0.00 | ... | ... | ... | ... | ... |
| 0.10 | ... | ... | ... | ... | ... |
| 0.25 | ... | ... | ... | ... | ... |
| 0.50 | ... | ... | ... | ... | ... |

## Table C: Runtime Profile

| Configuration | candidate_count | abr_iters | budget | oracle_calls | cache_hits | wall_time_sec |
|---|---:|---:|---:|---:|---:|---:|
| full | ... | ... | ... | ... | ... | ... |
| prefilter | ... | ... | ... | ... | ... | ... |

## 6. Figure Plan

- Figure 1: Annotated graph with `shared`, `unique_y`, `unique_foil`.
- Figure 2: \(\beta\) vs overlap curve.
- Figure 3: sparsity-faithfulness tradeoff (\(\lambda\) sweep).
- Figure 4: oracle calls vs candidate count.

## 7. Result Writing Template

"For prompt [X], Game 2 selected \(|E_y|=[a]\) and \(|E_{foil}|=[b]\), with shared evidence size [c]. As \(\beta\) increased from [u] to [v], overlap decreased from [m] to [n], indicating that the overlap penalty can drive a cleaner contrastive decomposition. Faithfulness remained [stable/degraded] with target delta [d], while oracle calls increased from [p] to [q], reflecting the computational cost of broader search."

## 8. Reproducibility Checklist

- Save full command line for each run.
- Save oracle kwargs JSON (or inline JSON string in appendix).
- Save output JSON and annotated graph JSON.
- Record git commit hash and date.
- Record hardware/device and dtype.

## 9. Framework-Evolution Notes (for Proposal)

When writing results in the proposal, separate:

- current empirical evidence from this MACAG codebase,
- planned architectural changes described in:
  - `FRAMEWORK_EVOLUTION_PLAN.md`
  - `IMPLEMENTATION_ROADMAP_POST_PRELIMS.md`

Recommended sentence:
"All quantitative results in this section are produced with the current standalone MACAG implementation; planned CDEA-aligned architectural refactoring is future engineering work and is not assumed in reported metrics."
