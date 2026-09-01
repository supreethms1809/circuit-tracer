# v3 capacity (d_transcoder) sweep — result paths

GPT-2 small, 1B tokens, seed 101. Six capacity points per arm. Within a point the
two arms are **parameter-matched**, not d_t-matched: a spline feature costs
2.1996x a linear feature (`152088` vs `69144` params per unit of d_t at
n_layers=12, d_model=768, grid_size=5, spline_order=3), so the linear arm needs
~2.20x the d_t to hold total parameters fixed.

Every run holds **lambda_sparsity=0.005, lr=5e-05, warmup=2900, total_steps=29000,
batch_size=128** fixed. `d_transcoder` is the only variable.

`OUT` below = `/cluster/ai4wy/gscratch/ssuresh/results/paper`

Per-run metrics live at:
`<suite_dir>/runs/<variant>/seed_101/training_records.jsonl`
(plus `aggregate_metrics.json` / `per_example_metrics.jsonl` once the eval stage runs)

## Spline arm

| spline d_t | params | job | suite dir (under `OUT`) | variant |
|---|---|---|---|---|
| 768   | 116.8M | 8422 | `paper_v3_dt_sweep_gpt2_small_spline_dt768`   | `spline_dt768_gpt2_small_pv3`   |
| 1536  | 233.6M | 8423 | `paper_v3_dt_sweep_gpt2_small_spline_dt1536`  | `spline_dt1536_gpt2_small_pv3`  |
| 3072  | 467.2M | 8424 | `paper_v3_dt_sweep_gpt2_small_spline_dt3072`  | `spline_dt3072_gpt2_small_pv3`  |
| 5568  | 846.8M | 8412 | `paper_v3_rebuttal_gpt2_small_spline_pm` †    | `spline_param_match_gpt2_small_pv3` |
| 6144  | 934.4M | 8426 | `paper_v3_dt_sweep_gpt2_small_spline_dt6144`  | `spline_dt6144_gpt2_small_pv3`  |
| 12288 | 1868.9M| 8415 | `paper_v3_rebuttal_gpt2_small_spline_fm` †    | `spline_feature_match_gpt2_small_pv3` |

## Linear arm (parameter-matched to the spline point on the same row)

| linear d_t | pairs with spline | params | delta vs spline | job | suite dir (under `OUT`) | variant |
|---|---|---|---|---|---|---|
| 1664  | 768   | 115.1M | -1.50% | 8428 | `paper_v3_dt_sweep_gpt2_small_linear_dt1664`  | `linear_dt1664_gpt2_small_pv3`  |
| 3392  | 1536  | 234.5M | +0.40% | 8429 | `paper_v3_dt_sweep_gpt2_small_linear_dt3392`  | `linear_dt3392_gpt2_small_pv3`  |
| 6784  | 3072  | 469.1M | +0.40% | 8430 | `paper_v3_dt_sweep_gpt2_small_linear_dt6784`  | `linear_dt6784_gpt2_small_pv3`  |
| 12288 | 5568  | 849.6M | +0.33% | 8411 | `paper_v3_rebuttal_gpt2_small_linear_fm` †    | `linear_feature_match_gpt2_small_pv3` |
| 13504 | 6144  | 933.7M | -0.08% | 8431 | `paper_v3_dt_sweep_gpt2_small_linear_dt13504` | `linear_dt13504_gpt2_small_pv3` |
| 27008 | 12288 | 1867.5M| -0.08% | 8413 | `paper_v3_rebuttal_gpt2_small_linear_pm` †    | `linear_param_match_gpt2_small_pv3` |

† **Sourced from the rebuttal suites, not the sweep suites.** These four points were
already running with training configs verified byte-identical (ignoring `run_name`)
to what the sweep would have produced, so dedicated sweep jobs for them were
cancelled (8425, 8427) or never launched. The sweep configs for the two spline
points still exist at
`experiments/paper_configs_v3/models/gpt2_small_spline_dt{5568,12288}.json` and can
be re-run if self-contained provenance is later preferred.

## Reading the curve

Plot `reconstruction/nmse_mean` (or `rel_fro_error`) against
`stats/l0_active_features_per_token` per arm to get a rate-distortion frontier, and
against total parameters for the capacity-scaling view. Comparing arms at a single
lambda is **not** a matched-sparsity comparison — as of this launch the arms sat at
L0 219 (spline) vs 49 (linear) at similar NMSE, which is why the frontier is needed.
For a directly matched-sparsity comparison, vary lambda instead: see
`experiments/paper_configs_v3/suites/probe_lambda_sweep_{,spline_}gpt2_small.json`.

Note `stats/l0_active_features_per_token` is summed over all 12 layers; divide by
n_layers for per-layer-per-token L0 (`stats/active_features_per_pos`).
