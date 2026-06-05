#!/usr/bin/env bash
# Unfrozen-attention Game 1 (raw_relative stop) on the ACDC benchmark prompts,
# across all 3 CLTs. Reuses the frozen attribution graphs + oracle kwargs from
# results/macag_acdc/<tag>/<slug>/ (freeze_attention is scoring-time, so the
# graph is unchanged). raw_relative stop is required because the ACDC/IOI frozen
# runs have negative recoverable_range (the normalized stop would be degenerate).
#
# Tests the attention-mediation hypothesis: gemma IOI was 10/10 reconstruction
# failures frozen (answer in attention); unfreezing should recruit features and
# flip recoverable_range positive.
#
# Output: results/macag_acdc_unfrozen/<tag>/<slug>/macag_game1.json
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUDGET="${BUDGET:-20}"
PREFILTER="${PREFILTER:-30}"
EPS="${EPS:-0.1}"
SRCROOT="${SRCROOT:-results/macag_acdc}"
OUTROOT="${OUTROOT:-results/macag_acdc_unfrozen}"
OF="macag.factories.replacement_model:create_replacement_model_oracle"
TAGS=(gemma2-426k gemma2-2.5M llama32-524k)

mkdir -p "$OUTROOT"
STATUS="$OUTROOT/status.txt"; : > "$STATUS"
echo ">>> ACDC unfrozen Game1 | budget=$BUDGET prefilter=$PREFILTER stop=raw_relative src=$SRCROOT"

for tag in "${TAGS[@]}"; do
  for d in "$SRCROOT/$tag"/*/; do
    [[ -d "$d" ]] || continue
    slug=$(basename "$d")
    graph="$d/graphs/$slug.json"
    kw_in="$d/oracle_kwargs.json"
    [[ -f "$graph" && -f "$kw_in" ]] || { echo "SKIP-missing $tag/$slug" | tee -a "$STATUS"; continue; }
    out="$OUTROOT/$tag/$slug"
    res="$out/macag_game1.json"
    [[ -f "$res" ]] && { echo "SKIP-done $tag/$slug" | tee -a "$STATUS"; continue; }
    mkdir -p "$out/graphs"
    ln -sf "$(readlink -f "$graph")" "$out/graphs/$slug.json"
    KW_IN="$kw_in" KW_OUT="$out/oracle_kwargs_unfrozen.json" python - <<'PY'
import json, os
k = json.load(open(os.environ["KW_IN"]))
k["freeze_attention"] = False
json.dump(k, open(os.environ["KW_OUT"], "w"), indent=2)
PY
    echo ">>> [$tag/$slug] unfrozen game1 (raw_relative) ..."
    if python -m macag.cli.run_macag game1 \
        --graph-json "$graph" --target y \
        --oracle-factory "$OF" --oracle-kwargs-file "$out/oracle_kwargs_unfrozen.json" \
        --prefilter-top-k "$PREFILTER" --budget "$BUDGET" \
        --alpha 0.5 --lam 0.02 --faithfulness-eps "$EPS" --stop-metric raw_relative \
        --output-json "$res" > "$out/game1.log" 2>&1; then
      echo "OK   $tag/$slug" | tee -a "$STATUS"
    else
      echo "FAIL $tag/$slug  (see $out/game1.log)" | tee -a "$STATUS"
    fi
  done
done
echo ">>> DONE. Status:"; sort "$STATUS" | sed -E 's#/[a-z0-9_]+##' | sort | uniq -c
