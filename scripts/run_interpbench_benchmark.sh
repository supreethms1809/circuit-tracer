#!/usr/bin/env bash
# InterpBench campaign: free Game1 selection (no hard budget) + gold-in-set metrics.
#
# Protocol (v5):
#   - score_kinds=logit_gap only
#   - BUDGET=0 (uncapped): grow while faithfulness/necessity utility improves
#   - FILL_BUDGET=0
#   - final_set = E* ∩ gold; leave-one-out importance inside E*
#
# Knobs: SEEDS LIMIT BUDGET ALPHA LAM FILL_BUDGET SCORE_KINDS RESULTS_ROOT DEVICE
#        SHAPLEY_PERMUTATIONS FAITHFULNESS_EPS
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-.}:."
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

SEEDS="${SEEDS:-0 1 2}"
LIMIT="${LIMIT:-50}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
DEVICE="${DEVICE:-cuda}"
SHAPLEY_PERMUTATIONS="${SHAPLEY_PERMUTATIONS:-64}"
BUDGET="${BUDGET:-0}"
ALPHA="${ALPHA:-0.5}"
LAM="${LAM:-0.0}"
FILL_BUDGET="${FILL_BUDGET:-0}"
SCORE_KINDS="${SCORE_KINDS:-logit_gap}"
FAITHFULNESS_EPS="${FAITHFULNESS_EPS:-}"

mkdir -p "$RESULTS_ROOT"
for SEED in $SEEDS; do
  OUT="$RESULTS_ROOT/interpbench_macag_seed${SEED}"
  if [[ -f "$OUT/interpbench_macag.csv" ]]; then
    echo ">>> [seed $SEED] $OUT already has results — skipping (delete the dir to redo)"
    continue
  fi
  mkdir -p "$OUT"
  echo ">>> [$(date '+%F %T')] InterpBench seed $SEED -> $OUT (limit=$LIMIT budget=${BUDGET:-uncapped} alpha=$ALPHA lam=$LAM fill_budget=$FILL_BUDGET kinds=$SCORE_KINDS)"
  fill_args=(--no-fill-budget)
  [[ "$FILL_BUDGET" == "1" || "$FILL_BUDGET" == "true" ]] && fill_args=(--fill-budget)
  eps_args=()
  [[ -n "$FAITHFULNESS_EPS" ]] && eps_args=(--faithfulness-eps "$FAITHFULNESS_EPS")
  if ! python experiments/run_interpbench_macag.py \
      --limit "$LIMIT" --device "$DEVICE" \
      --shapley-permutations "$SHAPLEY_PERMUTATIONS" --budget "$BUDGET" \
      --alpha "$ALPHA" --lam "$LAM" "${fill_args[@]}" "${eps_args[@]}" \
      --score-kinds "$SCORE_KINDS" \
      --seed "$SEED" --out-dir "$OUT" > "$OUT/run.log" 2>&1; then
    echo ">>> WARNING: seed $SEED failed — see $OUT/run.log (campaign continues)" >&2
  fi
done

echo ">>> InterpBench campaign done. Roots: $RESULTS_ROOT/interpbench_macag_seed{${SEEDS// /,}}"
