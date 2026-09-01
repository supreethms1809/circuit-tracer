# v3 rebuttal checkpoint paths

Checkpoint locations for the 16 currently-tracked v3 rebuttal/dt-sweep training
jobs, for later use with the Ravel evaluation and autointerp scoring passes.
**Paths only** — no metrics captured here yet.

`OUT` below = `/cluster/ai4wy/gscratch/ssuresh/results/paper`

Status as of 2026-07-25 (~19:00 into the batch): 7 of 16 jobs complete (8440, 8444,
8448, 8449, 8450, 8451, 8453). All 4 dt_sweep linear arms and 3 of 4 dt_sweep spline
arms are done; only dt3072 (8446) and dt6144 (8447) are still training. rebuttal_gpt2_small
linear_fm (8440) is done; spline_fm/spline_pm/linear_pm still running. qwen3 and gemma3
(167-chunk/58000-step lineages) are both still far from done.

Ravel eval + autointerp for the 12 plateaued gpt2-small checkpoints (8 dt_sweep + 4
rebuttal_gpt2_small fm/pm) have been launched separately — see "Ravel eval / autointerp
launch" section below.
- `COMPLETE` — training finished, checkpoint is final (per-run eval stage already ran)
- `RUNNING` — training still in progress; path below is the best-so-far checkpoint
  and **will keep changing** until the job finishes. Re-check before using for
  eval/autointerp.

Every checkpoint path is a directory (safetensors shards inside), suitable for
`local_clt_path` in `macag.factories.replacement_model` or the paper-eval loader.

## dt_sweep_gpt2_small (d_transcoder capacity sweep, gpt2-small)

| job  | arm    | d_t   | status   | checkpoint path |
|------|--------|-------|----------|------------------|
| 8444 | spline | 768   | COMPLETE | `OUT/paper_v3_dt_sweep_gpt2_small_spline_dt768/runs/spline_dt768_gpt2_small_pv3/seed_101/checkpoints/spline_dt768_gpt2_small_pv3_seed101_best` |
| 8453 | spline | 1536  | COMPLETE | `OUT/paper_v3_dt_sweep_gpt2_small_spline_dt1536/runs/spline_dt1536_gpt2_small_pv3/seed_101/checkpoints/spline_dt1536_gpt2_small_pv3_seed101_best` |
| 8446 | spline | 3072  | RUNNING  | `OUT/paper_v3_dt_sweep_gpt2_small_spline_dt3072/runs/spline_dt3072_gpt2_small_pv3/seed_101/checkpoints/spline_dt3072_gpt2_small_pv3_seed101_best` |
| 8447 | spline | 6144  | RUNNING  | `OUT/paper_v3_dt_sweep_gpt2_small_spline_dt6144/runs/spline_dt6144_gpt2_small_pv3/seed_101/checkpoints/spline_dt6144_gpt2_small_pv3_seed101_best` |
| 8448 | linear | 1664  | COMPLETE | `OUT/paper_v3_dt_sweep_gpt2_small_linear_dt1664/runs/linear_dt1664_gpt2_small_pv3/seed_101/checkpoints/linear_dt1664_gpt2_small_pv3_seed101_best` |
| 8449 | linear | 3392  | COMPLETE | `OUT/paper_v3_dt_sweep_gpt2_small_linear_dt3392/runs/linear_dt3392_gpt2_small_pv3/seed_101/checkpoints/linear_dt3392_gpt2_small_pv3_seed101_best` |
| 8450 | linear | 6784  | COMPLETE | `OUT/paper_v3_dt_sweep_gpt2_small_linear_dt6784/runs/linear_dt6784_gpt2_small_pv3/seed_101/checkpoints/linear_dt6784_gpt2_small_pv3_seed101_best` |
| 8451 | linear | 13504 | COMPLETE | `OUT/paper_v3_dt_sweep_gpt2_small_linear_dt13504/runs/linear_dt13504_gpt2_small_pv3/seed_101/checkpoints/linear_dt13504_gpt2_small_pv3_seed101_best` |

## rebuttal_gpt2_small (feature-match / param-match pairs, gpt2-small)

| job  | arm    | match type   | d_t   | status  | checkpoint path |
|------|--------|--------------|-------|---------|------------------|
| 8441 | spline | param-match  | 5568  | RUNNING | `OUT/paper_v3_rebuttal_gpt2_small_spline_pm/runs/spline_param_match_gpt2_small_pv3/seed_101/checkpoints/spline_pm_gpt2_small_pv3_seed101_best` |
| 8442 | linear | param-match  | 27008 | RUNNING | `OUT/paper_v3_rebuttal_gpt2_small_linear_pm/runs/linear_param_match_gpt2_small_pv3/seed_101/checkpoints/linear_pm_gpt2_small_pv3_seed101_best` |
| 8443 | spline | feature-match| 12288 | RUNNING | `OUT/paper_v3_rebuttal_gpt2_small_spline_fm/runs/spline_feature_match_gpt2_small_pv3/seed_101/checkpoints/spline_fm_gpt2_small_pv3_seed101_best` |
| 8440 | linear | feature-match| 12288 | COMPLETE | `OUT/paper_v3_rebuttal_gpt2_small_linear_fm/runs/linear_feature_match_gpt2_small_pv3/seed_101/checkpoints/linear_fm_gpt2_small_pv3_seed101_best` |

## rebuttal_qwen3 (qwen3-0.6b, spline vs linear)

| job  | arm    | d_t  | status  | checkpoint path |
|------|--------|------|---------|------------------|
| 8437 | spline | 5184 | RUNNING | `OUT/paper_v3_rebuttal_qwen3_spline/runs/spline_param_match_qwen3_06b_pv3/seed_101/checkpoints/spline_pm_qwen3_06b_pv3_seed101_best` |
| 8436 | linear | 8192 | RUNNING | `OUT/paper_v3_rebuttal_qwen3_linear/runs/linear_d8192_qwen3_06b_pv3/seed_101/checkpoints/linear_d8192_qwen3_06b_pv3_seed101_best` |

## rebuttal_gemma3 (gemma3-1b, spline vs linear)

| job  | arm    | d_t  | status  | checkpoint path |
|------|--------|------|---------|------------------|
| 8452 | spline | 5056 | RUNNING | `OUT/paper_v3_rebuttal_gemma3_spline/runs/spline_param_match_gemma3_1b_pv3/seed_101/checkpoints/spline_pm_gemma3_1b_pv3_seed101_best` |
| 8454 | linear | 8192 | RUNNING | `OUT/paper_v3_rebuttal_gemma3_linear/runs/linear_d8192_gemma3_1b_pv3/seed_101/checkpoints/linear_d8192_gemma3_1b_pv3_seed101_best` |

## reduced-capacity hub-comparison spline training

These one-seed spline arms deliberately use fewer features and fewer CLT
parameters than the published hub CLTs used by the RAVEL anchors below. They
were submitted on 2026-07-25 and are currently running. Gemma2 jobs 8522–8524
OOMed or were cancelled at d_t=8192 (static KAN+Adam ~138 GiB on 4 GPUs).
Job 8525 uses d_t=6144 on 4 nodes / 8 GPUs with CPU offload off
(~104 GiB estimated static, ~42 GiB headroom).

| job | arm | base model | d_t | nodes × GPUs | status | expected checkpoint path |
|-----|-----|------------|-----|--------------|--------|--------------------------|
| 8521 | llama32 spline reduced | `meta-llama/Llama-3.2-1B` | 12288 | 2 × 2 | RUNNING | `OUT/paper_v3_rebuttal_llama32_spline/runs/spline_dt12288_llama32_1b_pv3/seed_101/checkpoints/spline_dt12288_llama32_1b_pv3_seed101_best` |
| 8525 | gemma2 spline reduced | `google/gemma-2-2b` | 6144 | 4 × 2 | RUNNING | `OUT/paper_v3_rebuttal_gemma2_spline/runs/spline_dt6144_gemma2_2b_pv3/seed_101/checkpoints/spline_dt6144_gemma2_2b_pv3_seed101_best` |

## Ravel eval / autointerp launch (2026-07-25)

Ravel eval and autointerp were launched for the 12 checkpoints that had plateaued
in val loss at the time (all 8 dt_sweep arms + all 4 rebuttal_gpt2_small fm/pm
arms), regardless of whether the training job itself had finished yet — the
`_best` checkpoint was already representative in every plateaued case.

**Ravel eval**: 10 suite configs under
`experiments/paper_configs/suites/paper_v3_ravel_*.json` (8 dt_sweep + 2 fm/pm).
First pass was jobs 8466–8475 (top-1 only). **Re-run with top-5/top-10 fidelity**
(merged from `rebuttal-eval-path-edits` `87fda1e` into live tree): jobs
**8496–8505** with `RE_EVALUATE=1` (explicit `ct` python). Linear dt_sweep
arms **8500–8503** already finished with `top5`/`top10` in aggregates; remaining
spline/fm/pm arms still running. Graph naming **8506** depends on those.
Hub anchors (local HF snapshot + tokenizer + offline `create_graph_files`):
**8518** llama, **8519** gemma (supersede failed 8479–8517). Output under
`OUT/ravel_eval_suite_v3_<name>/`.

**Autointerp (REQ-4, random alive sample)**: needs a vLLM judge endpoint.
`scripts/slurm/launch_vllm.sbatch` (job 8455, Qwen2.5-72B-Instruct, `ai4wy-2`
partition, 2 GPUs) came up on `ai4wy-213:8192` and wrote
`results/rebuttal/vllm_endpoint.json`. Job 8476
(`scripts/slurm/rebuttal_autointerp_plateaued.sbatch`, all 12 checkpoints,
sequential, N_FEATURES=200) is running against that endpoint. Output under
`results/rebuttal/autointerp_<label>/`.

**RAVEL graph naming autointerp**: after RAVEL graphs exist, job submitted via
`scripts/slurm/rebuttal_autointerp_ravel_graphs.sbatch` (depends on remaining
ravel jobs). Targets unique feature IDs from each arm's
`evaluation/graphs/*.json`, scores them, and writes named copies to
`evaluation/graphs_autointerp/` plus `results/rebuttal/autointerp_ravel_<label>/`.

**Published hub CLT RAVEL anchors** (current jobs **8518/8519**): same 600-prompt
RAVEL replacement-fidelity protocol against HF-cached `mntss` CLTs, via
`python -m rebuttal_eval.ravel_hub_eval` (not paper-eval — hub CLTs are
`load_clt` format and have no val activation cache). Loader keeps the hub model
*name* for TransformerLens config mapping but feeds weights + tokenizer from
the local snapshot (`HF_HUB_OFFLINE=1`); `create_graph_files` resolves the
tokenizer via `snapshot_download(..., local_files_only=True)`.

| job  | suite | base model | CLT | output |
|------|-------|------------|-----|--------|
| 8518 | `paper_v3_ravel_hub_llama32.json` | `meta-llama/Llama-3.2-1B` | `mntss/clt-llama-3.2-1b-524k` | `OUT/ravel_eval_suite_v3_hub_llama32/` |
| 8519 | `paper_v3_ravel_hub_gemma2.json` | `google/gemma-2-2b` | `mntss/clt-gemma-2-2b-426k` | `OUT/ravel_eval_suite_v3_hub_gemma2/` |

Not launched: `mntss/clt-gemma-2-2b-2.5M` (160G / d_t=98304 — attribution cost
prohibitive for a 600-prompt pass; use only if a scale-matched anchor is needed).

## Notes

- All 16 paths above were verified to exist on disk as of 2026-07-25 (each job has
  already saved at least one best-checkpoint, even the ones still training).
- Excluded on purpose: `paper_v3_gpt2_small` and `paper_v3_qwen3_06b` (older core
  suite runs, not part of this rebuttal/dt-sweep batch) and everything under
  `paper_v2_*` / `paper_gpt2_*` (pre-v3 legacy runs).
- Suite root for each run (parent of `runs/`) also holds `resolved_config.json` and
  `manifest.json` if config provenance is needed alongside the checkpoint.
