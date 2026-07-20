#!/usr/bin/env bash
# One-command InterpBench benchmark campaign: three seeded replicates.
#
#   nohup scripts/run_interpbench_benchmark.sh > results/interpbench_campaign.log 2>&1 &
#
# Runs experiments/run_interpbench_macag.py (node-level AUROC / precision
# against InterpBench's exact ground-truth circuits) once per seed into
# results/interpbench_macag_seed<SEED>. Seeds whose root already has the
# output CSV are skipped, so re-running after a crash is safe.
#
# Knobs (defaults are the campaign configuration):
#   SEEDS="0 1 2"             seed list (seeds Shapley sampling + bootstrap)
#   LIMIT=500                 IOI prompts (0 = all; matches the MIB ioi cap)
#   DEVICE=cuda
#   SHAPLEY_PERMUTATIONS=64
#   BUDGET=4                  Game 1 budget (|gold circuit| = 4)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-.}:."
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

SEEDS="${SEEDS:-0 1 2}"
LIMIT="${LIMIT:-500}"
DEVICE="${DEVICE:-cuda}"
SHAPLEY_PERMUTATIONS="${SHAPLEY_PERMUTATIONS:-64}"
BUDGET="${BUDGET:-4}"

for SEED in $SEEDS; do
  OUT="results/interpbench_macag_seed${SEED}"
  if [[ -f "$OUT/interpbench_macag.csv" ]]; then
    echo ">>> [seed $SEED] $OUT already has results — skipping (delete the dir to redo)"
    continue
  fi
  mkdir -p "$OUT"
  echo ">>> [$(date '+%F %T')] InterpBench seed $SEED -> $OUT (limit=$LIMIT)"
  if ! python experiments/run_interpbench_macag.py \
      --limit "$LIMIT" --device "$DEVICE" \
      --shapley-permutations "$SHAPLEY_PERMUTATIONS" --budget "$BUDGET" \
      --seed "$SEED" --out-dir "$OUT" > "$OUT/run.log" 2>&1; then
    echo ">>> WARNING: seed $SEED failed — see $OUT/run.log (campaign continues)" >&2
  fi
done

echo ">>> InterpBench campaign done. Roots: results/interpbench_macag_seed{${SEEDS// /,}}"
