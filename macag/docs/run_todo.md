# MACAG Result Collection — MIB + InterpBench, 3 seeds

Updated 2026-07-04. Results are collected on **two benchmarks only**:

1. **MIB** (mib-bench circuit-localization prompts, gemma2 cells → both gemma CLTs)
2. **InterpBench** (exact ground-truth circuits, node-level AUROC/precision)

Everything else in `macag/data/` is an internal diagnostic set, **not** a paper
result: `acdc_benchmark_prompts.json` (own test set), `nonlinear_benchmark_prompts.json`
(hand-written Spline-CLT stress prompts), and the two-hop generalization
manifest. Their drivers (`scripts/run_macag_acdc*.sh`, `run_macag_pipeline.sh`)
remain available for ad-hoc debugging.

All campaign configuration lives **inside** the two benchmark scripts
(`scripts/run_mib_benchmark.sh`, `scripts/run_interpbench_benchmark.sh`):
3-seed loop (`SEEDS="0 1 2"`), fast-pass baselines + deferred Shapley-gold
(same seed, first 50 prompts/task), KL rescore, analyzer CSVs, stats, curves,
gold-circuit scoring, and cross-seed aggregation. Both scripts are safe to
re-run after a crash — completed work is skipped.

## 0. One-time setup

```bash
conda activate ct

# MIB repo + HF datasets (only needed to build the prompt JSON):
[[ -d external/MIB-circuit-track ]] || bash external/setup_mib.sh

# Build the MIB prompt JSON at benchmark size. Validation splits:
# ioi 10,000 | mcqa 50 | arc_easy 570. mcqa + arc_easy run in full; ioi is
# capped at 500 (uncapped = ~2 GPU-years across seeds). One flag to change.
python experiments/build_mib_benchmark_prompts.py \
  --models gemma2 --tasks ioi mcqa arc_easy --split validation \
  --task-limit ioi=500

python -c "import json; d=json.load(open('macag/data/mib_benchmark_prompts.json')); \
print({k: len(v) for k, v in d['tasks'].items()})"
# expect: {'ioi': 500, 'mcqa': 50, 'arc_easy': 570}
```

## 1. The two commands (sequentially — they share the GPU)

```bash
# InterpBench first (hours):
nohup scripts/run_interpbench_benchmark.sh > results/interpbench_campaign.log 2>&1 &

# then MIB (the long one — see cost note below):
nohup scripts/run_mib_benchmark.sh > results/mib_campaign.log 2>&1 &
```

That's it. Outputs:

- `results/interpbench_macag_seed{0,1,2}/interpbench_macag.csv` (+ `run.log`)
- `results/macag_mib_seed{0,1,2}/` — per-prompt game outputs plus `summary.csv`,
  `baselines.csv` (gold `*_shapley` columns for the first 50/task),
  `abr_vs_fp.csv`, `frozen_vs_unfrozen{,_agg}.csv`, `gold_circuits.csv`,
  `bootstrap_wilcoxon.{md,csv}`, `curves/`
- `results/macag_mib_seeds/` — cross-seed `*_allseeds.csv` +
  `*_seed_summary.csv` (mean/std/n per (clt, task); paper tables read these)

## Cost note (measured pilot throughput: 426k pool 3.3 cells/h, 2.5M 1.5 cells/h)

The MIB campaign is 1,120 prompts x 2 CLTs x 3 seeds ≈ **45 GPU-days per seed**
on one GH200. To spread one seed across nodes (shared filesystem), run
`SEEDS=<one seed> scripts/run_mib_benchmark.sh` per node with disjoint
`CLTS=gemma2-426k` / `CLTS=gemma2-2.5M`, or shrink the ioi/arc_easy caps in
step 0 (e.g. `--task-limit ioi=100 --task-limit arc_easy=200` ≈ 14 days/seed).

## Failure triage

- MIB shard failures: `FAIL` lines in `results/macag_mib_seed<S>/status.*.txt`;
  read the matching `<clt>-<slug>.log`. CUDA OOM → re-run the same campaign
  command (resumable), optionally with `WORKERS_GEMMA2_426K=6`.
- Orphaned GPU workers after Ctrl+C: `scripts/macag_kill_sweep.sh <root>`.
- InterpBench failures: `results/interpbench_macag_seed<S>/run.log`; delete
  that seed's dir and re-run the campaign script.
- A missing MIB aggregate CSV with all cells `OK`: re-run
  `ANALYZE_ONLY=1 JSON=macag/data/mib_benchmark_prompts.json \
   OUTROOT=results/macag_mib_seed<S> FREEZE_MODE=both scripts/run_macag_mib.sh`
  and read its stderr (analyzer failures are logged, not fatal).
