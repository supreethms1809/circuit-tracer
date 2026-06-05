#!/usr/bin/env bash
# Unfrozen-attention Game 2 (contrastive) across all runnable CLTs.
#
# Reuses the graphs + unfrozen oracle kwargs already produced by the Game 1
# unfrozen run (results/macag_unfrozen_raw/<tag>/<slug>/), so no re-attribution
# and no new kwargs. Params are matched to the FROZEN Game 2 runs (budget 8,
# prefilter 20, beta 0.2, abr-iters 4, alpha 0.5, lam 0.02) so the only
# difference vs the frozen baseline is freeze_attention=false.
#
# Game 2 has no faithfulness_eps stop (it selects on raw game2_utility via ABR),
# so the Game-1 stop_metric fix does not apply here.
#
# Output: results/macag_unfrozen_raw/<tag>/<slug>/macag_game2.json
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUDGET="${BUDGET:-8}"
PREFILTER="${PREFILTER:-20}"
BETA="${BETA:-0.2}"
ABR_ITERS="${ABR_ITERS:-4}"
OUTROOT="${OUTROOT:-results/macag_unfrozen_raw}"
OF="macag.factories.replacement_model:create_replacement_model_oracle"

TAGS=(gemma2-426k gemma2-2.5M llama32-524k)
SLUGS=(dallas-austin houston-austin chicago-springfield miami-tallahassee \
       detroit-lansing portland-salem cleveland-columbus philadelphia-harrisburg)

STATUS="$OUTROOT/status_game2.txt"; : > "$STATUS"
echo ">>> Unfrozen Game2 sweep | budget=$BUDGET prefilter=$PREFILTER beta=$BETA abr=$ABR_ITERS outroot=$OUTROOT"

for tag in "${TAGS[@]}"; do
  for slug in "${SLUGS[@]}"; do
    out="$OUTROOT/$tag/$slug"
    graph="$out/graphs/$slug.json"
    kw="$out/oracle_kwargs_unfrozen.json"
    res="$out/macag_game2.json"
    if [[ ! -f "$graph" || ! -f "$kw" ]]; then
      echo "SKIP-missing $tag/$slug" | tee -a "$STATUS"; continue
    fi
    if [[ -f "$res" ]]; then
      echo "SKIP-done    $tag/$slug" | tee -a "$STATUS"; continue
    fi
    echo ">>> [$tag/$slug] unfrozen game2 ..."
    if python -m macag.cli.run_macag game2 \
        --graph-json "$graph" --target y --foil y_foil \
        --oracle-factory "$OF" --oracle-kwargs-file "$kw" \
        --prefilter-top-k "$PREFILTER" --budget "$BUDGET" \
        --beta "$BETA" --abr-iters "$ABR_ITERS" \
        --alpha 0.5 --lam 0.02 \
        --output-json "$res" > "$out/game2.log" 2>&1; then
      echo "OK   $tag/$slug" | tee -a "$STATUS"
    else
      echo "FAIL $tag/$slug  (see $out/game2.log)" | tee -a "$STATUS"
    fi
  done
done
echo ">>> DONE. Status:"; cat "$STATUS"
