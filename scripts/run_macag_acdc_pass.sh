#!/usr/bin/env bash
# Deferred ACDC Pass B over an existing Pass A MACAG sweep root.
#
# After Pass A (BASELINE_METHODS=influence,eap,game1), run ACDC alone on each
# completed logit_gap run, merge into macag_baselines.json, and optionally
# refresh KL faithfulness for the ACDC sets.
#
# Usage:
#   scripts/run_macag_acdc_pass.sh /gscratch/$USER/macag_mib_tmlr250v3_h200/macag_mib_seed0
#   ACDC_TAUS="0.01,0.1,0.5" scripts/run_macag_acdc_pass.sh <root>
#   CLTS="gemma2-426k" NUM_WORKERS=4 WORKER_ID=0 scripts/run_macag_acdc_pass.sh <root>
#
# Env knobs:
#   ACDC_TAUS          default: full 6 taus (0.001,0.01,0.05,0.1,0.2,0.5)
#   ACDC_TARGET_K      default -1 (= budget-matched bisection); "" disables
#   ACDC_ORDER         top_down (default) | given
#   BUDGET ALPHA LAM   Game1-matched eval (defaults 8 / 0.5 / 0.02)
#   SCORE_KIND         subdirectory to process (default logit_gap)
#   KL_RESCORE         1 = refresh KL blocks after merge (default 1)
#   CLTS SLUG_REGEX NUM_WORKERS WORKER_ID  sharding filters
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-.}:."

ROOT="${1:?usage: run_macag_acdc_pass.sh <sweep_root>}"
ACDC_TAUS="${ACDC_TAUS:-0.001,0.01,0.05,0.1,0.2,0.5}"
ACDC_TARGET_K="${ACDC_TARGET_K--1}"
ACDC_ORDER="${ACDC_ORDER:-top_down}"
BUDGET="${BUDGET:-8}"
ALPHA="${ALPHA:-0.5}"
LAM="${LAM:-0.02}"
SCORE_KIND="${SCORE_KIND:-logit_gap}"
CLTS="${CLTS:-}"
SLUG_REGEX="${SLUG_REGEX:-}"
NUM_WORKERS="${NUM_WORKERS:-1}"
WORKER_ID="${WORKER_ID:-0}"
KL_RESCORE="${KL_RESCORE:-1}"
CONNECTED="${CONNECTED:-0}"

STATUS="$ROOT/status.acdc.w${WORKER_ID}.txt"; : > "$STATUS"
echo ">>> ACDC Pass B over $ROOT | taus=$ACDC_TAUS target_k=${ACDC_TARGET_K:-off} | worker=$WORKER_ID/$NUM_WORKERS"
echo ">>> SCORE_KIND=$SCORE_KIND budget=$BUDGET"

CELL=0
for clt_dir in "$ROOT"/*/; do
  tag=$(basename "$clt_dir")
  [[ "$tag" == .* ]] && continue
  [[ -n "$CLTS" && " $CLTS " != *" $tag "* ]] && continue
  for run_dir in "$clt_dir"*/; do
    [[ -d "$run_dir" ]] || continue
    slug=$(basename "$run_dir")
    if [[ -n "$SLUG_REGEX" ]] && ! grep -Eq "$SLUG_REGEX" <<<"$slug"; then continue; fi

    # Prefer Pass A kind subdirectory; fall back to legacy top-level / symlink.
    kind_dir="$run_dir/$SCORE_KIND"
    if [[ -d "$kind_dir" ]]; then
      main_json="$kind_dir/macag_baselines.json"
      kwargs="$kind_dir/oracle_kwargs.json"
      work_dir="$kind_dir"
    else
      main_json="$run_dir/macag_baselines.json"
      kwargs="$run_dir/oracle_kwargs.json"
      work_dir="$run_dir"
    fi
    graph="$run_dir/graphs/$slug.json"
    [[ -f "$main_json" && -f "$kwargs" && -f "$graph" ]] || continue
    # Require Game 1 so Pass A finished selection before ACDC.
    [[ -f "$work_dir/macag_game1.json" || -f "$run_dir/macag_game1.json" ]] || continue

    cell=$CELL; CELL=$((CELL + 1))
    (( cell % NUM_WORKERS == WORKER_ID )) || continue

    if python - "$main_json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
acdc = (d.get("methods") or {}).get("acdc") or {}
# Done when matched_k or a non-empty sweep is present.
sys.exit(0 if (acdc.get("matched_k") or acdc.get("sweep")) else 1)
PY
    then
      echo "SKIP-done $tag/$slug" | tee -a "$STATUS"; continue
    fi

    sidecar="$work_dir/macag_baselines_acdc.json"
    echo ">>> [$tag/$slug] ACDC (taus=$ACDC_TAUS target_k=${ACDC_TARGET_K:-off})"
    cand_file="$work_dir/candidates_from_main.json"
    python - "$main_json" "$cand_file" <<'PY'
import json, sys
main = json.load(open(sys.argv[1]))
cands = main.get("candidates") or []
json.dump(cands, open(sys.argv[2], "w"))
print(len(cands))
PY
    bl_cmd=(
      python -m macag.cli.run_baselines
      --graph-json "$graph" --target y --input-id "$slug"
      --oracle-factory macag.factories.replacement_model:create_replacement_model_oracle
      --oracle-kwargs-file "$kwargs"
      --candidates-file "$cand_file"
      --budget "$BUDGET" --alpha "$ALPHA" --lam "$LAM"
      --methods acdc
      --acdc-taus "$ACDC_TAUS"
      --acdc-order "$ACDC_ORDER"
      --output-json "$sidecar"
    )
    if [[ "${CONNECTED}" != "1" && "${CONNECTED}" != "true" ]]; then
      bl_cmd+=(--no-connected)
    fi
    if [[ -n "${ACDC_TARGET_K}" ]]; then
      bl_cmd+=(--acdc-target-k "$ACDC_TARGET_K")
    fi

    if "${bl_cmd[@]}" \
      && python -m macag.cli.merge_baselines --main "$main_json" --extra "$sidecar"; then
      # Keep top-level symlink in sync when Pass A used kind subdirs.
      if [[ -L "$run_dir/macag_baselines.json" || ! -e "$run_dir/macag_baselines.json" ]]; then
        ln -sfn "$SCORE_KIND/macag_baselines.json" "$run_dir/macag_baselines.json" 2>/dev/null || true
      fi
      if [[ "$KL_RESCORE" == "1" ]]; then
        python -m macag.cli.rescore_kl --run-dir "$work_dir" --force >/dev/null \
          || echo ">>> (KL refresh failed for $tag/$slug; merge itself succeeded)"
      fi
      echo "OK   $tag/$slug" | tee -a "$STATUS"
    else
      echo "FAIL $tag/$slug" | tee -a "$STATUS"
    fi
  done
done

echo ""; echo ">>> ACDC Pass B done (worker $WORKER_ID/$NUM_WORKERS):"
sort "$STATUS" | uniq -c | sort -rn | head
echo ">>> Re-run analyze_macag_baselines.py / bootstrap_wilcoxon / curves to refresh CSVs."
