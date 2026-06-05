#!/usr/bin/env bash
# Unfrozen-attention Game 1 sweep at high budget, across all runnable CLTs.
#
# Reuses the attribution graphs + oracle kwargs already produced by the frozen
# runs (freeze_attention is a scoring-time arg, so the graph is unchanged), flips
# freeze_attention -> false, and re-runs Game 1 with a larger budget so the
# minimal faithful set can actually converge instead of hitting the cap.
#
# Output: results/macag_unfrozen/<clt_tag>/<slug>/macag_game1.json (+ graph symlink)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUDGET="${BUDGET:-20}"
PREFILTER="${PREFILTER:-30}"
EPS="${EPS:-0.1}"
STOP_METRIC="${STOP_METRIC:-normalized}"
OUTROOT="${OUTROOT:-results/macag_unfrozen}"
OF="macag.factories.replacement_model:create_replacement_model_oracle"

# clt_tag -> source dir holding <slug>/{graphs/<slug>.json, oracle_kwargs.json}
TAGS=(gemma2-426k gemma2-2.5M llama32-524k)
SRC=(results/macag_sweep results/macag_clt_compare/gemma2-2.5M results/macag_clt_compare/llama32-524k)
SLUGS=(dallas-austin houston-austin chicago-springfield miami-tallahassee \
       detroit-lansing portland-salem cleveland-columbus philadelphia-harrisburg)

mkdir -p "$OUTROOT"
STATUS="$OUTROOT/status.txt"; : > "$STATUS"
echo ">>> Unfrozen Game1 sweep | budget=$BUDGET prefilter=$PREFILTER eps=$EPS stop=$STOP_METRIC outroot=$OUTROOT"

for i in "${!TAGS[@]}"; do
  tag="${TAGS[$i]}"; src="${SRC[$i]}"
  for slug in "${SLUGS[@]}"; do
    graph="$src/$slug/graphs/$slug.json"
    kw_in="$src/$slug/oracle_kwargs.json"
    out="$OUTROOT/$tag/$slug"
    res="$out/macag_game1.json"
    if [[ ! -f "$graph" || ! -f "$kw_in" ]]; then
      echo "SKIP-missing $tag/$slug" | tee -a "$STATUS"; continue
    fi
    if [[ -f "$res" ]]; then
      echo "SKIP-done    $tag/$slug" | tee -a "$STATUS"; continue
    fi
    mkdir -p "$out/graphs"
    ln -sf "$(readlink -f "$graph")" "$out/graphs/$slug.json"
    # unfrozen kwargs
    KW_IN="$kw_in" KW_OUT="$out/oracle_kwargs_unfrozen.json" python - <<'PY'
import json, os
k = json.load(open(os.environ["KW_IN"]))
k["freeze_attention"] = False
json.dump(k, open(os.environ["KW_OUT"], "w"), indent=2)
PY
    echo ">>> [$tag/$slug] unfrozen game1 ..."
    if python -m macag.cli.run_macag game1 \
        --graph-json "$graph" --target y \
        --oracle-factory "$OF" \
        --oracle-kwargs-file "$out/oracle_kwargs_unfrozen.json" \
        --prefilter-top-k "$PREFILTER" --budget "$BUDGET" \
        --alpha 0.5 --lam 0.02 --faithfulness-eps "$EPS" --stop-metric "$STOP_METRIC" \
        --output-json "$res" > "$out/game1.log" 2>&1; then
      echo "OK   $tag/$slug" | tee -a "$STATUS"
    else
      echo "FAIL $tag/$slug  (see $out/game1.log)" | tee -a "$STATUS"
    fi
  done
done
echo ">>> DONE. Status:"; cat "$STATUS"
