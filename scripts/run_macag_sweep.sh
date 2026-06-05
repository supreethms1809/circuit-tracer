#!/usr/bin/env bash
# Generalization sweep: run the MACAG pipeline over a manifest of prompts that
# share one template, so the two-hop (city -> state -> capital) circuit can be
# compared across facts.
#
# Usage:
#   scripts/run_macag_sweep.sh [MANIFEST] [SWEEP_OUTDIR] [DEVICE]
#
# Defaults:
#   MANIFEST     = experiments/macag_generalization_prompts.json
#   SWEEP_OUTDIR = results/macag_sweep
#   DEVICE       = cuda
#
# Each prompt writes results/<SWEEP_OUTDIR>/<slug>/ exactly like a single
# run_macag_pipeline.sh invocation. Failures are logged and skipped so one bad
# prompt does not abort the whole sweep. After it finishes, analyze with:
#   python experiments/analyze_macag_sweep.py --sweep-dir <SWEEP_OUTDIR>
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="${1:-experiments/macag_generalization_prompts.json}"
SWEEP_OUTDIR="${2:-results/macag_sweep}"
DEVICE="${3:-cuda}"

[[ -f "$MANIFEST" ]] || { echo "ERROR: manifest not found: $MANIFEST" >&2; exit 1; }
mkdir -p "$SWEEP_OUTDIR"

MODEL=$(python -c "import json,sys;print(json.load(open('$MANIFEST'))['model'])")
TSET=$(python -c "import json,sys;print(json.load(open('$MANIFEST'))['transcoder_set'])")
TEMPLATE=$(python -c "import json,sys;print(json.load(open('$MANIFEST'))['template'])")
N=$(python -c "import json,sys;print(len(json.load(open('$MANIFEST'))['prompts']))")

echo ">>> MACAG generalization sweep"
echo "    manifest : $MANIFEST  ($N prompts)"
echo "    model    : $MODEL"
echo "    clt      : $TSET"
echo "    template : $TEMPLATE"
echo "    outdir   : $SWEEP_OUTDIR"
echo "    device   : $DEVICE"

SUMMARY="$SWEEP_OUTDIR/sweep_status.txt"
: > "$SUMMARY"

for i in $(seq 0 $((N-1))); do
  read -r SLUG CITY TARGET FOIL < <(python - "$MANIFEST" "$i" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["prompts"][int(sys.argv[2])]
print(m["slug"], m["city"], m["target"], m["foil"])
PY
)
  PROMPT="${TEMPLATE/\{CITY\}/$CITY}"
  OUTDIR="$SWEEP_OUTDIR/$SLUG"
  LOG="$SWEEP_OUTDIR/${SLUG}.log"
  echo ""
  echo ">>> [$((i+1))/$N] $SLUG | city=$CITY target=$TARGET foil=$FOIL"
  echo "    prompt: $PROMPT"
  echo "    log   : $LOG"

  if scripts/run_macag_pipeline.sh \
        --prompt "$PROMPT" \
        --target "$TARGET" \
        --foil "$FOIL" \
        --model "$MODEL" \
        --transcoder-set "$TSET" \
        --slug "$SLUG" \
        --outdir "$OUTDIR" \
        --device "$DEVICE" \
        >"$LOG" 2>&1; then
    echo "OK   $SLUG" | tee -a "$SUMMARY"
  else
    echo "FAIL $SLUG  (see $LOG)" | tee -a "$SUMMARY"
  fi
done

echo ""
echo ">>> Sweep done. Status:"
cat "$SUMMARY"
echo ""
echo ">>> Analyze with:"
echo "    python experiments/analyze_macag_sweep.py --sweep-dir $SWEEP_OUTDIR"
