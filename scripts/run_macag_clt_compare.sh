#!/usr/bin/env bash
# Multi-CLT comparison driver.
#
# Runs the same prompt sweep (scripts/run_macag_sweep.sh) once per cross-layer
# transcoder listed in experiments/macag_clt_compare.json, then runs the
# cross-CLT comparison analyzer. CLTs with "run": false are reused in place
# (e.g. the already-computed gemma-426k baseline).
#
# Usage:
#   scripts/run_macag_clt_compare.sh [CONFIG]
# Default CONFIG = experiments/macag_clt_compare.json
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${1:-experiments/macag_clt_compare.json}"
[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }

PROMPTS=$(python -c "import json;print(json.load(open('$CONFIG'))['prompts_manifest'])")
DEVICE=$(python -c "import json;print(json.load(open('$CONFIG')).get('device','cuda'))")
N=$(python -c "import json;print(len(json.load(open('$CONFIG'))['clts']))")

echo ">>> MACAG multi-CLT comparison"
echo "    config   : $CONFIG"
echo "    prompts  : $PROMPTS"
echo "    device   : $DEVICE"
echo "    CLTs     : $N"

STATUS="results/macag_clt_compare/clt_status.txt"
mkdir -p results/macag_clt_compare
: > "$STATUS"

for i in $(seq 0 $((N-1))); do
  read -r TAG MODEL TSET SWEEPDIR RUN < <(python - "$CONFIG" "$i" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))["clts"][int(sys.argv[2])]
print(c["tag"], c["model"], c["transcoder_set"], c["sweep_dir"], str(c.get("run", True)).lower())
PY
)
  if [[ "$RUN" != "true" ]]; then
    echo ">>> [$((i+1))/$N] $TAG  -> reuse existing $SWEEPDIR (run=false)"
    echo "REUSE $TAG  $SWEEPDIR" >> "$STATUS"
    continue
  fi

  echo ""
  echo ">>> [$((i+1))/$N] $TAG | model=$MODEL clt=$TSET"
  echo "    sweep_dir: $SWEEPDIR"
  mkdir -p "$SWEEPDIR"

  # Per-CLT manifest = shared prompts with this CLT's model/transcoder_set.
  MANI="$SWEEPDIR/_manifest.json"
  PROMPTS="$PROMPTS" MODEL="$MODEL" TSET="$TSET" MANI="$MANI" python - <<'PY'
import json, os
m = json.load(open(os.environ["PROMPTS"]))
m["model"] = os.environ["MODEL"]
m["transcoder_set"] = os.environ["TSET"]
json.dump(m, open(os.environ["MANI"], "w"), indent=2)
PY

  CLT_LOG="$SWEEPDIR/sweep_driver.log"
  if scripts/run_macag_sweep.sh "$MANI" "$SWEEPDIR" "$DEVICE" >"$CLT_LOG" 2>&1; then
    NOK=$(grep -c '^OK' "$SWEEPDIR/sweep_status.txt" 2>/dev/null || echo 0)
    echo "DONE  $TAG  ($NOK prompts OK)  $SWEEPDIR" | tee -a "$STATUS"
  else
    echo "FAIL  $TAG  (see $CLT_LOG)" | tee -a "$STATUS"
  fi
done

echo ""
echo ">>> All CLTs processed. Status:"
cat "$STATUS"

echo ""
echo ">>> Running cross-CLT comparison ..."
python experiments/analyze_clt_comparison.py \
    --config "$CONFIG" \
    --csv results/macag_clt_compare/comparison.csv
