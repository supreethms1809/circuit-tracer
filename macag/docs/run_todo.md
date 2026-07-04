# MACAG Full Rerun Playbook — 3 seeds, full-size benchmarks

Rewritten 2026-07-04. This replaces the old status-dependent backlog: it is the
exact, from-scratch command sequence to collect **every** MACAG paper result as
**three seeded replicates** on **full-size prompt sets**. No command below
depends on any previously collected root.

## Conventions (read once)

- Run everything from the repo root in the `ct` env (`conda activate ct`).
- One sweep at a time per GPU; the `*_parallel.sh` drivers self-shard
  (426k=8 / 2.5M=3 / llama=4 workers, CLT groups sequential).
- **Seed semantics**: `SEED ∈ {0,1,2}`. The seed controls the only explicitly
  stochastic stages — MC-Shapley/Banzhaf gold-baseline sampling
  (`SHAPLEY_SEED`, plumbed through `run_macag_pipeline.sh --shapley-seed`) and
  the InterpBench Shapley estimator (`--seed`). Graph attribution, Game 1/2,
  influence/EAP/ACDC, and KL rescore have no RNG; cross-seed spread in those
  numbers measures CUDA/bfloat16 nondeterminism, which the replicates capture
  because each seed root is rebuilt from scratch (fresh graphs included).
- **Roots**: `results/<sweep>_seed${SEED}`. Never point a seeded run at an old
  root — the resume logic would skip "done" cells from the previous campaign.
- **Two-pass split**: every sweep runs the fast baselines first
  (`BASELINE_METHODS="influence,eap,game1,acdc"`), then a deferred
  Shapley-gold pass (`run_macag_shapley_pass.sh`) **with the same seed**.
  Until that pass merges, `baselines.csv` has empty `*_shapley` columns and
  the vs-gold rows are skipped.
- All sweeps are resumable within a root: completed `(CLT, prompt)` cells are
  skipped on restart, so a crashed worker just needs the same command again.
- OOM guard (one 426k arc_easy cell died of a 25.7 GiB attribution-backward
  spike in the pilot): keep `PYTORCH_ALLOC_CONF=expandable_segments:True`
  exported (step 0.1); if OOMs recur, relaunch with `WORKERS_GEMMA2_426K=6`.

## 0. One-time setup

```bash
# 0.1 env + allocator guard (put in the shell profile for long campaigns)
conda activate ct
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 0.2 MIB repo + datasets (needed only to rebuild the prompt JSON)
[[ -d external/MIB-circuit-track ]] || bash external/setup_mib.sh

# 0.3 Rebuild the MIB prompt JSON at full benchmark size.
# Validation split sizes: ioi 10,000 | mcqa 50 | arc_easy 570.
# mcqa + arc_easy run IN FULL. ioi is capped at 500 because the full 10,000
# (x2 CLTs x3 seeds = 60,000 pipeline cells, ~2 GPU-years at measured
# throughput) is not tractable; 500 is 50x the pilot. To change the cap, edit
# the one --task-limit value; to run truly uncapped, delete it.
python experiments/build_mib_benchmark_prompts.py \
  --models gemma2 --tasks ioi mcqa arc_easy --split validation \
  --task-limit ioi=500

# 0.4 verify counts before burning GPU time
python -c "import json; d=json.load(open('macag/data/mib_benchmark_prompts.json')); \
print({k: len(v) for k, v in d['tasks'].items()})"
# expect: {'ioi': 500, 'mcqa': 50, 'arc_easy': 570}
```

The ACDC (`acdc_benchmark_prompts.json`: IOI 10 + greater_than 8 + docstring 3)
and nonlinear (`nonlinear_benchmark_prompts.json`: 4 tasks x 5) manifests are
hand-curated — they have no larger "full" version and are already run whole
(`LIMIT=0` is the sweep default; the old 10/task MIB cap lived only in the
prompt JSON build, which 0.3 removes).

## Cost budget (measured on the 10/task pilot, one GH200 96 GB, fast pass)

Throughput: 426k pool ≈ 3.3 cells/h, 2.5M pool ≈ 1.5 cells/h (a cell = one
(CLT, prompt) full pipeline: graph + dual Game 1 + Game 2 abr/fp + fast
baselines + KL).

| Sweep (per seed)               | cells                | est. wall (1 GPU) |
|--------------------------------|----------------------|-------------------|
| MIB (500+50+570, 2 gemma CLTs) | 2,240                | ~45 days          |
| ACDC bench (21 x 3 CLTs)       | 63                   | ~1.5 days         |
| Nonlinear (20 x 3 CLTs)        | 60                   | ~1.5 days         |
| Shapley-gold pass (MIB 50/task, ACDC+nonlinear full) | 423 | ~7–10 days |

The MIB sweep dominates: **~50 days/seed, x3 seeds ≈ 5 GPU-months on one
node.** Plan for either (a) sharding one seed across nodes with a shared
filesystem — run the same command per node with `CLTS=...` split, or
`NUM_WORKERS`/`WORKER_ID` partitioning — or (b) shrinking `--task-limit`
(e.g. `ioi=100` + `--task-limit arc_easy=200` → ~14 days/seed). Decide before
step 0.3; all seeds must use the same JSON.

## 1. Per-seed campaign (run to completion for SEED=0, then 1, then 2)

```bash
export SEED=0        # then 1, then 2
```

### 1.1 MIB sweep (fast pass)

```bash
mkdir -p results/macag_mib_seed${SEED}
SHAPLEY_SEED=$SEED BASELINE_METHODS="influence,eap,game1,acdc" \
OUTROOT=results/macag_mib_seed${SEED} \
  nohup scripts/run_macag_mib_parallel.sh \
  > results/macag_mib_seed${SEED}/parallel.log 2>&1 &
```

The launcher aggregates automatically at the end (KL rescore + the four
analyzers). Check `results/macag_mib_seed${SEED}/status.*.txt` for `FAIL`
lines; rerun the same command to fill crashed cells, then re-aggregate:

```bash
ANALYZE_ONLY=1 JSON=macag/data/mib_benchmark_prompts.json \
OUTROOT=results/macag_mib_seed${SEED} FREEZE_MODE=both scripts/run_macag_mib.sh
```

### 1.2 ACDC-benchmark sweep (budget-matched ACDC, answer_span for greater_than)

```bash
mkdir -p results/macag_acdc_seed${SEED}
SHAPLEY_SEED=$SEED BASELINE_METHODS="influence,eap,game1,acdc" ACDC_TARGET_K=-1 \
OUTROOT=results/macag_acdc_seed${SEED} \
  nohup scripts/run_macag_acdc_parallel.sh \
  > results/macag_acdc_seed${SEED}/parallel.log 2>&1 &
# re-aggregate after fixing FAILs:
ANALYZE_ONLY=1 ACDC_TARGET_K=-1 OUTROOT=results/macag_acdc_seed${SEED} \
  FREEZE_MODE=both scripts/run_macag_acdc.sh
```

### 1.3 Nonlinear-benchmark sweep (same driver, different JSON)

```bash
mkdir -p results/macag_nonlinear_seed${SEED}
SHAPLEY_SEED=$SEED BASELINE_METHODS="influence,eap,game1,acdc" ACDC_TARGET_K=-1 \
JSON=macag/data/nonlinear_benchmark_prompts.json \
OUTROOT=results/macag_nonlinear_seed${SEED} \
  nohup scripts/run_macag_acdc_parallel.sh \
  > results/macag_nonlinear_seed${SEED}/parallel.log 2>&1 &
# re-aggregate:
ANALYZE_ONLY=1 ACDC_TARGET_K=-1 JSON=macag/data/nonlinear_benchmark_prompts.json \
  OUTROOT=results/macag_nonlinear_seed${SEED} FREEZE_MODE=both scripts/run_macag_acdc.sh
```

### 1.4 Shapley-gold pass (same seed; idempotent, shardable)

```bash
# ACDC + nonlinear roots: all prompts.
SHAPLEY_SEED=$SEED scripts/run_macag_shapley_pass.sh results/macag_acdc_seed${SEED}
SHAPLEY_SEED=$SEED scripts/run_macag_shapley_pass.sh results/macag_nonlinear_seed${SEED}

# MIB root: first 50 prompts per task (gold on the full 1,120 would cost more
# than the sweep itself). Widen the regex to widen gold coverage.
SHAPLEY_SEED=$SEED SLUG_REGEX='_(ioi|mcqa|arc_easy)_00[0-4][0-9]$' \
  scripts/run_macag_shapley_pass.sh results/macag_mib_seed${SEED}
# (shard with CLTS=... NUM_WORKERS=... WORKER_ID=... as usual)
```

### 1.5 Re-aggregate + alt-foil + stats + curves (CPU except alt-foil)

```bash
for ROOT in results/macag_mib_seed${SEED} results/macag_acdc_seed${SEED} \
            results/macag_nonlinear_seed${SEED}; do
  case "$ROOT" in
    *mib*)       BENCH=macag/data/mib_benchmark_prompts.json;;
    *nonlinear*) BENCH=macag/data/nonlinear_benchmark_prompts.json;;
    *)           BENCH=macag/data/acdc_benchmark_prompts.json;;
  esac
  # refresh baselines.csv with the merged shapley columns
  python experiments/analyze_macag_baselines.py --root "$ROOT" --bench "$BENCH" \
    --csv "$ROOT/baselines.csv"
  python scripts/macag_bootstrap_wilcoxon.py --root "$ROOT"
  python experiments/plot_faithfulness_curves.py --root "$ROOT" --bench "$BENCH"
done

# alt-foil rescore (GPU; ACDC root only — MIB/nonlinear manifests need
# metadata.alt_incorrect_token added first, same convention as ACDC):
python -m macag.cli.rescore_altfoil --root results/macag_acdc_seed${SEED} \
  --bench macag/data/acdc_benchmark_prompts.json --progress
```

### 1.6 Gold-circuit validation

```bash
# (layer, token-role) IOI scoring on the ACDC root (S1/IO metadata):
python experiments/analyze_gold_circuits.py --root results/macag_acdc_seed${SEED} \
  --bench macag/data/acdc_benchmark_prompts.json \
  --task indirect_object_identification --include-baselines

# MIB IOI root (heuristic role fallback):
python experiments/analyze_gold_circuits.py --root results/macag_mib_seed${SEED} \
  --bench macag/data/mib_benchmark_prompts.json --task ioi --include-baselines

# InterpBench exact ground truth (node-level AUROC; seeded Shapley):
python experiments/run_interpbench_macag.py --limit 500 --device cuda \
  --shapley-permutations 64 --budget 4 --seed $SEED \
  --out-dir results/interpbench_macag_seed${SEED}
```

### 1.7 Per-seed done criteria (check before moving to the next seed)

- `status.*.txt` files in all three roots contain no `FAIL` lines.
- Each root has non-empty `summary.csv`, `baselines.csv`, `abr_vs_fp.csv`,
  `frozen_vs_unfrozen.csv` + `frozen_vs_unfrozen_agg.csv` (the pilot silently
  lost `summary.csv` to a tolerated analyzer failure — check all five).
- `baselines.csv` has populated `*_shapley` columns for every ACDC/nonlinear
  row and for the first-50 MIB rows.
- `bootstrap_wilcoxon.{md,csv}` and `curves/{curves,auc}.csv` exist per root.

## 2. One-off controls (seed-independent; run once, alongside any seed)

```bash
# 2.1 Ablation-convention control: ACDC bench under corrupted-prompt (patch)
# ablation. Only the Game-1 range-flip diagnostic is read from this root, so
# skip baselines. Compare frozen_vs_unfrozen_agg.csv against the zero-ablation
# roots.
ABLATION_MODE=corrupted SKIP_BASELINES=1 OUTROOT=results/macag_acdc_corrupted \
  nohup scripts/run_macag_acdc_parallel.sh \
  > results/macag_acdc_corrupted/parallel.log 2>&1 &

# 2.2 Two-hop case study (narrative figures; SHAPLEY_SEED left at 0):
for TSET in "gemma2-426k mntss/clt-gemma-2-2b-426k google/gemma-2-2b" \
            "gemma2-2.5M mntss/clt-gemma-2-2b-2.5M google/gemma-2-2b" \
            "llama32-524k mntss/clt-llama-3.2-1b-524k meta-llama/Llama-3.2-1B"; do
  set -- $TSET; TAG=$1; SET=$2; MODEL=$3
  python - <<'PY' | while IFS=$'\t' read -r SLUG PROMPT TARGET FOIL; do
import json
d = json.load(open('experiments/macag_generalization_prompts.json'))
for p in d['prompts']:
    print(f"{p['slug']}\t{d['template'].replace('{CITY}', p['city'])}\t{p['target']}\t{p['foil']}")
PY
    scripts/run_macag_pipeline.sh --prompt "$PROMPT" --target "$TARGET" --foil "$FOIL" \
      --model "$MODEL" --transcoder-set "$SET" --slug "$SLUG" \
      --outdir "results/macag_twohop/$TAG/$SLUG" --device cuda \
      --freeze-mode both --solvers "abr fp"
  done
done
```

## 3. Cross-seed aggregation (CPU, minutes; after all three seeds finish)

```bash
for SWEEP in macag_mib macag_acdc macag_nonlinear; do
  python scripts/macag_combine_seeds.py \
    results/${SWEEP}_seed0 results/${SWEEP}_seed1 results/${SWEEP}_seed2 \
    --out results/${SWEEP}_seeds
done
```

Outputs per sweep: `<csv>_allseeds.csv` (row-level with a `seed` column) and
`<csv>_seed_summary.csv` (per-(clt, task) mean/std/n across seeds) for every
analyzer CSV present. Paper tables read the `_seed_summary` files; the
`_allseeds` files feed any additional significance testing.

## Order of execution (one GPU)

Seed 0: 1.2 → 1.3 → 1.1 (longest last, so ACDC/nonlinear numbers land early)
→ 1.4 → 1.5 → 1.6 → checklist 1.7. Then 2.1 + 2.2 while seed 1's MIB sweep
runs. Repeat 1.x for seeds 1 and 2 → step 3.

## Failure triage

- `FAIL` in `status.*`: read `OUTROOT/<clt>-<slug>.log`. CUDA OOM → relaunch
  the sweep command unchanged (resumable) with `WORKERS_GEMMA2_426K=6`.
- Orphaned GPU workers after Ctrl+C: `scripts/macag_kill_sweep.sh <OUTROOT>`.
- A missing aggregate CSV with runs all `OK`: rerun the matching
  `ANALYZE_ONLY=1 ...` command from 1.1–1.3 and read its stderr — analyzer
  failures are tolerated (logged, not fatal) during auto-aggregation.
