#!/usr/bin/env bash
# One-command MIB benchmark campaign: seeded replicates, end to end.
#
# TMLR-250 Pass A (v3) defaults: ioi=100 / mcqa=50 / arc_easy=50 per mib_model,
# logit_gap selection only, Game 1 dual-freeze + Game 2 abr/fp, baselines
# influence/eap/game1 (ACDC + Shapley deferred). Prefer the cluster launcher:
#   scripts/slurm/submit_all_tmlr250.sh
#
# Per seed (sequentially): fast-pass parallel sweep + KL rescore + analyzer
# CSVs into $RESULTS_ROOT/macag_mib_seed<SEED>. Optional Shapley-gold pass when
# GOLD_PER_TASK>0. After all seeds: cross-seed aggregation.
#
# Safe to re-run after a crash: every stage skips completed work.
#
# Knobs (Pass A defaults — override only for debugging / Pass B):
#   SEEDS="0"              seed list (expand to "0 1 2" after Pass A coverage)
#   SCORE_KINDS=logit_gap  selection utilities (KL is rescore-only)
#   BASELINE_METHODS       fast-pass selector list (no acdc/shapley in Pass A)
#   GOLD_PER_TASK=0        skip Shapley-gold; set 50 (or all) to enable
#   ACDC_TARGET_K=         unused unless acdc is in BASELINE_METHODS; -1 = budget
#   ALLOW_SMALL_JSON=1     run even if the prompt JSON looks pilot-sized
#   RESULTS_ROOT=results   base dir for per-seed roots + cross-seed aggregation
# Sweep-level knobs (WORKERS_*, NODE_RANK, TOTAL_WORKERS, STAGGER, ...) pass
# through to run_macag_mib_parallel.sh as usual.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-.}:."
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
# shellcheck source=scripts/macag_parallel_common.sh
source scripts/macag_parallel_common.sh

SEEDS="${SEEDS:-0}"
JSON="${JSON:-macag/data/mib_benchmark_prompts.json}"
export SCORE_KINDS="${SCORE_KINDS:-logit_gap}"
BASELINE_METHODS="${BASELINE_METHODS:-influence,eap,game1}"
ACDC_TARGET_K="${ACDC_TARGET_K-}"
GOLD_PER_TASK="${GOLD_PER_TASK:-0}"
FREEZE_MODE="${FREEZE_MODE:-both}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
COMBINED_OUT="${COMBINED_OUT:-$RESULTS_ROOT/macag_mib_seeds}"

# Belt-and-suspenders: whenever this orchestrator exits (normal completion,
# a signal, or an `-u` unbound-var abort), sweep for anything still matching
# our own worker patterns under $RESULTS_ROOT. In the normal case this is a
# harmless no-op (the per-CLT worker pools already reaped everything via
# `wait`); it only does real work if a shard's cleanup trap was bypassed.
trap 'macag_kill_macag_procs "$RESULTS_ROOT" >/dev/null 2>&1 || true' EXIT

# --- prompt JSON sanity: refuse to burn weeks of GPU on the 10/task pilot ---
if [[ ! -f "$JSON" ]]; then
  echo "ERROR: $JSON missing. Build the TMLR-scale MIB subset first:" >&2
  echo "  python experiments/build_mib_benchmark_prompts.py \\" >&2
  echo "    --models gemma2 llama3 --tasks ioi mcqa arc_easy --split validation \\" >&2
  echo "    --task-limit ioi=100 --task-limit mcqa=50 --task-limit arc_easy=50" >&2
  echo "  (or slice macag/data/mib_benchmark_prompts_full_v1.json to those caps)" >&2
  exit 2
fi
n_ioi=$(JSON="$JSON" python -c "import json,os; print(len(json.load(open(os.environ['JSON']))['tasks'].get('ioi', [])))")
# Refuse the old 10/task pilot. TMLR default is 100 ioi prompts per mib_model
# (200 rows when gemma2+llama3 are both present).
if (( n_ioi <= 10 )) && [[ "${ALLOW_SMALL_JSON:-0}" != "1" ]]; then
  echo "ERROR: $JSON has only $n_ioi ioi prompts (pilot-sized)." >&2
  echo "Rebuild at TMLR size (ioi=100 / mcqa=50 / arc_easy=50) or set ALLOW_SMALL_JSON=1." >&2
  exit 2
fi

# First-N-per-task filter for the gold pass (slugs end in a 4-digit index).
# GOLD_PER_TASK=0 skips the Shapley-gold pass entirely (Pass A default).
gold_regex=""
skip_shapley=0
if [[ "$GOLD_PER_TASK" == "0" ]]; then
  skip_shapley=1
elif [[ "$GOLD_PER_TASK" == "all" ]]; then
  gold_regex=""   # no filter: gold baseline on every prompt (expensive)
elif [[ "$GOLD_PER_TASK" =~ ^[1-9]0$|^100$ ]]; then
  hi=$(( GOLD_PER_TASK / 10 - 1 ))
  gold_regex="_(ioi|mcqa|arc_easy)_00[0-${hi}][0-9]\$"
else
  echo "ERROR: GOLD_PER_TASK=$GOLD_PER_TASK must be 0, a multiple of 10 in 10..100, or 'all'." >&2
  exit 2
fi

step() { echo ""; echo ">>> [$(date '+%F %T')] $*"; }
run_soft() { step "\$ $*"; "$@" || echo ">>> WARNING: step failed (campaign continues): $*" >&2; }

macag_clear_stale_gpu_slots "$RESULTS_ROOT"

roots=()
for SEED in $SEEDS; do
  OUTROOT="$RESULTS_ROOT/macag_mib_seed${SEED}"
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

  if (( skip_shapley )); then
    step "Shapley-gold pass skipped (GOLD_PER_TASK=0; Pass A / deferred like ACDC)"
  else
    step "Shapley-gold pass (first $GOLD_PER_TASK/task, seed $SEED)"
    run_soft env SHAPLEY_SEED="$SEED" SLUG_REGEX="$gold_regex" \
      scripts/run_macag_shapley_pass.sh "$OUTROOT"
  fi

  # Multi-node array shards set RUN_ANALYSIS=0; only aggregate when this process
  # owns analysis (single-node or explicit RUN_ANALYSIS=1).
  if [[ "${RUN_ANALYSIS:-1}" != "0" ]]; then
    step "post-processing (CSV refresh, stats, curves, gold circuits)"
    run_soft python experiments/analyze_macag_baselines.py \
      --root "$OUTROOT" --bench "$JSON" --csv "$OUTROOT/baselines.csv"
    run_soft python scripts/macag_bootstrap_wilcoxon.py --root "$OUTROOT"
    run_soft python experiments/plot_faithfulness_curves.py --root "$OUTROOT" --bench "$JSON"
    run_soft python experiments/analyze_gold_circuits.py --root "$OUTROOT" \
      --bench "$JSON" --task ioi --include-baselines
  else
    step "post-processing skipped (RUN_ANALYSIS=0; re-run after all array nodes finish)"
  fi
done

if [[ "${RUN_ANALYSIS:-1}" != "0" ]]; then
  step "cross-seed aggregation -> $COMBINED_OUT"
  run_soft python scripts/macag_combine_seeds.py "${roots[@]}" --out "$COMBINED_OUT"
fi

step "MIB campaign done. Per-seed roots: ${roots[*]} | combined: $COMBINED_OUT"
echo ">>> Done-criteria: no FAIL in <root>/status.*.txt; summary/baselines/abr_vs_fp/"
echo ">>> frozen_vs_unfrozen(+_agg) CSVs present. Pass A: SCORE_KINDS=$SCORE_KINDS"
echo ">>> BASELINE_METHODS=$BASELINE_METHODS GOLD_PER_TASK=$GOLD_PER_TASK"
