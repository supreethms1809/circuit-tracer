# MACAG Result Collection — TMLR-250 Pass A (v3)

Updated 2026-08-04. Paper evaluation uses **250 unique prompts** under protocol **v3 Pass A**:

| Knob | Value |
| --- | --- |
| Prompts | MIB IOI 100 + MCQA 50 + ARC-Easy 50 + InterpBench IOI 50 |
| Prefilter | **off** |
| Connected | **off** (`--no-connected`) |
| Selection utility | **`logit_gap` only** (not dual `kl_divergence` selection) |
| KL | **rescore** on saved evidence (`macag_kl_faithfulness.json`) |
| Game 1 | dual-freeze (`freeze_mode=both`), budget 8 |
| Game 2 | **abr + fp** (kept) |
| Baselines (Pass A) | `influence,eap,game1` — **ACDC and Shapley deferred** |
| Seeds | **0** first; expand to 0 1 2 after coverage |
| Fleet | **4 nodes / 8 GPUs per CLT** (Slurm array `0-3`), requeue OK |

Fresh outputs go to `macag_mib_tmlr250v3_*`. Prior roots (`macag_mib_tmlr250v2_*`,
`macag_mib_h200`, …) are left aside.

## Why Pass A

Per-prompt IOI timings showed dual `score_kinds` + ACDC pushing gemma prompts to
~16 h. Pass A drops KL-as-selection (~2×) and ACDC (~0.7–2.5 h/prompt) while
keeping Game 1 + Game 2. Expected serial cost ~**3–6 h/prompt** depending on CLT;
with 4 nodes / 8 GPUs, seed0 × 200 is ~1–2× 24h requeues per CLT.

## 0. Prompt manifest (already wired)

`macag/data/mib_benchmark_prompts.json` is the TMLR subset
(ioi=100 / mcqa=50 / arc_easy=50 per `mib_model`). Full export:
`macag/data/mib_benchmark_prompts_full_v1.json`.

```bash
conda activate ct
python experiments/build_mib_benchmark_prompts.py \
  --models gemma2 llama3 --tasks ioi mcqa arc_easy --split validation \
  --task-limit ioi=100 --task-limit mcqa=50 --task-limit arc_easy=50
```

InterpBench default is `LIMIT=50` in `scripts/run_interpbench_benchmark.sh`.

## 1. Cluster launch (Pass A)

```bash
# cancel prior TMLR jobs if needed, submit 3× 4-node MIB arrays + InterpBench
scripts/slurm/submit_all_tmlr250.sh

# or without cancelling anything already running
scripts/slurm/submit_all_tmlr250.sh --no-cancel
```

Individual CLT arrays (each expands to 4 nodes):

```bash
sbatch scripts/slurm/submit_macag_mib_h200.sh       # gemma2-426k → tmlr250v3_h200
sbatch scripts/slurm/submit_macag_mib_gemma25m.sh   # gemma2-2.5M → tmlr250v3_gemma25m
sbatch scripts/slurm/submit_macag_mib_llama.sh      # llama32-524k → tmlr250v3_llama
sbatch scripts/slurm/submit_interpbench.sh
```

Array shards set `RUN_ANALYSIS=0`. After all tasks for a root finish (or on
requeue once coverage is complete), aggregate once:

```bash
RESULTS_ROOT=/gscratch/$USER/macag_mib_tmlr250v3_h200 \
  RUN_ANALYSIS=1 SEEDS=0 scripts/run_mib_benchmark.sh
```

Requeue is safe: completed Game1 / Game2 solvers / baselines JSONs are skipped.

## 2. Pass B — deferred ACDC

Same graphs and Pass A baselines; merge ACDC into `macag_baselines.json`:

```bash
# full 6 taus + matched-k (budget 8)
scripts/run_macag_acdc_pass.sh /gscratch/$USER/macag_mib_tmlr250v3_h200/macag_mib_seed0

# thinner tau sweep if needed
ACDC_TAUS="0.01,0.1,0.5" \
  scripts/run_macag_acdc_pass.sh /gscratch/$USER/macag_mib_tmlr250v3_h200/macag_mib_seed0
```

Then refresh CSVs (`analyze_macag_baselines.py`, bootstrap, curves).

Shapley-gold remains optional via `GOLD_PER_TASK=50 scripts/run_macag_shapley_pass.sh <root>`.

## 3. Local / interactive

```bash
RESULTS_ROOT=/gscratch/$USER/macag_mib_tmlr250v3_h200 \
  SEEDS=0 SCORE_KINDS=logit_gap \
  BASELINE_METHODS=influence,eap,game1 GOLD_PER_TASK=0 \
  scripts/run_mib_benchmark.sh
```

## Outputs

- `/gscratch/$USER/macag_mib_tmlr250v3_*/macag_mib_seed0/` — per-prompt
  `logit_gap/{macag_game1,macag_game2_*,macag_baselines,macag_kl_faithfulness}.json`
- After Pass B: `methods.acdc` merged into baselines
- Aggregate CSVs after `RUN_ANALYSIS=1`

## Failure triage

- MIB shard failures: `FAIL` lines in `$RESULTS_ROOT/macag_mib_seed<S>/status.*.txt`
- Orphaned GPU workers: `scripts/macag_kill_sweep.sh <root>`
- Stale GPU slot locks (multi-node scoped under `.gpu_slots/<nodename>/`): cleared
  automatically at campaign start per node
- InterpBench failures: `$RESULTS_ROOT/interpbench_macag_seed<S>/run.log`
