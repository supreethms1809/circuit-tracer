#!/usr/bin/env bash
# One-command MIB benchmark campaign: three seeded replicates, end to end.
#
#   nohup scripts/run_mib_benchmark.sh > results/mib_campaign.log 2>&1 &
#
# Per seed (sequentially): fast-pass parallel sweep (influence/eap/game1/acdc
# + Game 1 dual-freeze + Game 2 abr/fp + KL rescore + analyzer CSVs) into
# results/macag_mib_seed<SEED>, then the Shapley-gold pass (first
# GOLD_PER_TASK prompts per task, same seed), then baselines-CSV refresh,
# bootstrap/Wilcoxon stats, faithfulness curves, and gold-circuit IOI scoring.
# After all seeds: cross-seed aggregation into results/macag_mib_seeds.
#
# Safe to re-run after a crash: every stage skips completed work.
#
# Knobs (defaults are the campaign configuration — override only for debugging):
#   SEEDS="0 1 2"          seed list
#   GOLD_PER_TASK=50       prompts per task that get the MC-Shapley gold baseline
#   BASELINE_METHODS       fast-pass selector list
#   ACDC_TARGET_K=-1       budget-matched ACDC (bisect tau to |E|=budget); "" disables
#   ALLOW_SMALL_JSON=1     run even if the prompt JSON looks pilot-sized
# Sweep-level knobs (WORKERS_*, STAGGER, DEVICE, ...) pass through to
# run_macag_mib_parallel.sh as usual.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-.}:."
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

SEEDS="${SEEDS:-0 1 2}"
JSON="${JSON:-macag/data/mib_benchmark_prompts.json}"
BASELINE_METHODS="${BASELINE_METHODS:-influence,eap,game1,acdc}"
ACDC_TARGET_K="${ACDC_TARGET_K--1}"
GOLD_PER_TASK="${GOLD_PER_TASK:-50}"
FREEZE_MODE="${FREEZE_MODE:-both}"
COMBINED_OUT="${COMBINED_OUT:-results/macag_mib_seeds}"

# --- prompt JSON sanity: refuse to burn weeks of GPU on the 10/task pilot ---
if [[ ! -f "$JSON" ]]; then
  echo "ERROR: $JSON missing. Build it first (full mcqa+arc_easy, ioi capped):" >&2
  echo "  python experiments/build_mib_benchmark_prompts.py \\" >&2
  echo "    --models gemma2 --tasks ioi mcqa arc_easy --split validation --task-limit ioi=500" >&2
  exit 2
fi
n_ioi=$(JSON="$JSON" python -c "import json,os; print(len(json.load(open(os.environ['JSON']))['tasks'].get('ioi', [])))")
if (( n_ioi <= 10 )) && [[ "${ALLOW_SMALL_JSON:-0}" != "1" ]]; then
  echo "ERROR: $JSON has only $n_ioi ioi prompts (pilot-sized)." >&2
  echo "Rebuild it at benchmark size (see header) or set ALLOW_SMALL_JSON=1 for a smoke run." >&2
  exit 2
fi

# First-N-per-task filter for the gold pass (slugs end in a 4-digit index).
if [[ "$GOLD_PER_TASK" == "all" ]]; then
  gold_regex=""   # no filter: gold baseline on every prompt (expensive)
elif [[ "$GOLD_PER_TASK" =~ ^[1-9]0$|^100$ ]]; then
  hi=$(( GOLD_PER_TASK / 10 - 1 ))
  gold_regex="_(ioi|mcqa|arc_easy)_00[0-${hi}][0-9]\$"
else
  echo "ERROR: GOLD_PER_TASK=$GOLD_PER_TASK must be a multiple of 10 in 10..100, or 'all'." >&2
  exit 2
fi

step() { echo ""; echo ">>> [$(date '+%F %T')] $*"; }
run_soft() { step "\$ $*"; "$@" || echo ">>> WARNING: step failed (campaign continues): $*" >&2; }

roots=()
for SEED in $SEEDS; do
  OUTROOT="results/macag_mib_seed${SEED}"
  roots+=("$OUTROOT")
  mkdir -p "$OUTROOT"
  step "===== MIB campaign seed $SEED -> $OUTROOT ====="

  step "fast-pass sweep (blocks until all shards + aggregation finish)"
  if ! SHAPLEY_SEED="$SEED" BASELINE_METHODS="$BASELINE_METHODS" \
       ACDC_TARGET_K="$ACDC_TARGET_K" \
       JSON="$JSON" OUTROOT="$OUTROOT" FREEZE_MODE="$FREEZE_MODE" \
       scripts/run_macag_mib_parallel.sh > "$OUTROOT/parallel.log" 2>&1; then
    echo ">>> WARNING: sweep reported shard failures — check $OUTROOT/status.*.txt," >&2
    echo ">>> fix and re-run this script (completed cells are skipped)." >&2
  fi

  step "Shapley-gold pass (first $GOLD_PER_TASK/task, seed $SEED)"
  run_soft env SHAPLEY_SEED="$SEED" SLUG_REGEX="$gold_regex" \
    scripts/run_macag_shapley_pass.sh "$OUTROOT"

  step "post-processing (CSV refresh, stats, curves, gold circuits)"
  run_soft python experiments/analyze_macag_baselines.py \
    --root "$OUTROOT" --bench "$JSON" --csv "$OUTROOT/baselines.csv"
  run_soft python scripts/macag_bootstrap_wilcoxon.py --root "$OUTROOT"
  run_soft python experiments/plot_faithfulness_curves.py --root "$OUTROOT" --bench "$JSON"
  run_soft python experiments/analyze_gold_circuits.py --root "$OUTROOT" \
    --bench "$JSON" --task ioi --include-baselines
done

step "cross-seed aggregation -> $COMBINED_OUT"
run_soft python scripts/macag_combine_seeds.py "${roots[@]}" --out "$COMBINED_OUT"

step "MIB campaign done. Per-seed roots: ${roots[*]} | combined: $COMBINED_OUT"
echo ">>> Done-criteria: no FAIL in <root>/status.*.txt; summary/baselines/abr_vs_fp/"
echo ">>> frozen_vs_unfrozen(+_agg) CSVs present; *_shapley columns populated for the"
echo ">>> first $GOLD_PER_TASK prompts/task; gold_circuits.csv + bootstrap_wilcoxon.{md,csv} + curves/ per root."
