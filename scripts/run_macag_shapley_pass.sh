#!/usr/bin/env bash
# Deferred Shapley-gold pass over an existing MACAG sweep root.
#
# Use after a fast sweep that skipped the expensive gold baseline
# (BASELINE_METHODS="influence,eap,game1,acdc"): for every completed run whose
# macag_baselines.json lacks a shapley block, run the MC-Shapley selector on the
# stored graph + oracle kwargs, then merge it back and rebuild the comparison
# block (macag.cli.merge_baselines — pure JSON, no extra oracle cost).
#
# Usage:
#   scripts/run_macag_shapley_pass.sh results/macag_acdc
#   CLTS="gemma2-426k" NUM_WORKERS=2 WORKER_ID=0 scripts/run_macag_shapley_pass.sh results/macag_mib
#
# Env knobs: SHAPLEY_PERMUTATIONS (64), SHAPLEY_SEED (0), BUDGET (8), ALPHA (0.5),
# LAM (0.02), KL_RESCORE (1 = refresh KL blocks for the merged shapley set),
# SLUG_REGEX (extended regex; only process run dirs whose slug matches, e.g.
#   SLUG_REGEX='_(ioi|mcqa|arc_easy)_00[0-4][0-9]$'   # first 50 prompts per MIB task
# — use it to bound the gold-baseline cost on large sweep roots).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-.}:."

ROOT="${1:?usage: run_macag_shapley_pass.sh <sweep_root>}"
SHAPLEY_PERMUTATIONS="${SHAPLEY_PERMUTATIONS:-64}"
SHAPLEY_SEED="${SHAPLEY_SEED:-0}"
BUDGET="${BUDGET:-8}"
ALPHA="${ALPHA:-0.5}"
LAM="${LAM:-0.02}"
CLTS="${CLTS:-}"
SLUG_REGEX="${SLUG_REGEX:-}"
NUM_WORKERS="${NUM_WORKERS:-1}"
WORKER_ID="${WORKER_ID:-0}"
KL_RESCORE="${KL_RESCORE:-1}"

STATUS="$ROOT/status.shapley.w${WORKER_ID}.txt"; : > "$STATUS"
echo ">>> Shapley pass over $ROOT | perms=$SHAPLEY_PERMUTATIONS seed=$SHAPLEY_SEED | worker=$WORKER_ID/$NUM_WORKERS"

CELL=0
for clt_dir in "$ROOT"/*/; do
  tag=$(basename "$clt_dir")
  [[ -n "$CLTS" && " $CLTS " != *" $tag "* ]] && continue
  for run_dir in "$clt_dir"*/; do
    [[ -d "$run_dir" ]] || continue
    slug=$(basename "$run_dir")
    if [[ -n "$SLUG_REGEX" ]] && ! grep -Eq "$SLUG_REGEX" <<<"$slug"; then continue; fi
    main_json="$run_dir/macag_baselines.json"
    kwargs="$run_dir/oracle_kwargs.json"
    graph="$run_dir/graphs/$slug.json"
    [[ -f "$main_json" && -f "$kwargs" && -f "$graph" ]] || continue

    cell=$CELL; CELL=$((CELL + 1))
    (( cell % NUM_WORKERS == WORKER_ID )) || continue

    if python - "$main_json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if (d.get("methods") or {}).get("shapley", {}).get("results") else 1)
PY
    then
      echo "SKIP-done $tag/$slug" | tee -a "$STATUS"; continue
    fi

    sidecar="$run_dir/macag_baselines_shapley.json"
    echo ">>> [$tag/$slug] shapley (${SHAPLEY_PERMUTATIONS} perms)"
    if python -m macag.cli.run_baselines \
        --graph-json "$graph" --target y --input-id "$slug" \
        --oracle-factory macag.factories.replacement_model:create_replacement_model_oracle \
        --oracle-kwargs-file "$kwargs" \
        --budget "$BUDGET" --alpha "$ALPHA" --lam "$LAM" \
        --methods shapley --shapley-permutations "$SHAPLEY_PERMUTATIONS" \
        --shapley-seed "$SHAPLEY_SEED" --no-progress \
        --output-json "$sidecar" \
      && python -m macag.cli.merge_baselines --main "$main_json" --extra "$sidecar"; then
      if [[ "$KL_RESCORE" == "1" ]]; then
        # Refresh embedded KL blocks so the merged shapley set gets kl_faith too.
        python -m macag.cli.rescore_kl --run-dir "$run_dir" --force >/dev/null \
          || echo ">>> (KL refresh failed for $tag/$slug; merge itself succeeded)"
      fi
      echo "OK   $tag/$slug" | tee -a "$STATUS"
    else
      echo "FAIL $tag/$slug" | tee -a "$STATUS"
    fi
  done
done

echo ""; echo ">>> Shapley pass done (worker $WORKER_ID/$NUM_WORKERS):"
sort "$STATUS" | uniq -c | sort -rn | head
echo ">>> Re-run the aggregate analyzers (analyze_macag_baselines.py, bootstrap_wilcoxon, curves) to refresh CSVs."
