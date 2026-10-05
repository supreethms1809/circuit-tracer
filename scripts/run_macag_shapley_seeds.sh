#!/usr/bin/env bash
# S3: Shapley rankings at seeds 0, 1, 2, scored on both freeze legs.
# Writes macag_shapley_s<seed>.json next to the other baseline outputs.
set -euo pipefail

GRAPH="${1:?graph json}"
KWARGS="${2:?oracle kwargs json}"
GAME1="${3:?game1 json}"
OUTDIR="${4:?output directory}"
BUDGET="${BUDGET:-128}"
SEED0="${SHAPLEY_SEED0:-0}"

mkdir -p "$OUTDIR"
for offset in 0 1 2; do
  seed=$((SEED0 + offset))
  python -m macag.cli.run_baselines \
    --graph-json "$GRAPH" \
    --target y \
    --input-id "${SLUG:-shapley}" \
    --oracle-factory macag.factories.replacement_model:create_replacement_model_oracle \
    --oracle-kwargs-file "$KWARGS" \
    --budget "$BUDGET" \
    --methods shapley \
    --shapley-permutations "${SHAPLEY_PERMUTATIONS:-64}" \
    --shapley-seed "$seed" \
    --legs both \
    --acdc-target-from-game1 "$GAME1" \
    --cap-sufficiency --cap-necessity \
    --output-json "$OUTDIR/macag_shapley_s${seed}.json"
done
