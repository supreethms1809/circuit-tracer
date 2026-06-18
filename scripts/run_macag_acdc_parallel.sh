#!/usr/bin/env bash
# Parallel launcher for the MACAG ACDC sweep on a single GPU.
#
# Spawns NUM_WORKERS copies of scripts/run_macag_acdc.sh that round-robin the
# (CLT, prompt) cells, waits for all of them, then runs the analyzers once over
# the combined output (ANALYZE_ONLY). Oracle calls underutilize a GH200, so a
# few concurrent shards overlap well; the stagger offsets attribution VRAM peaks.
#
# Usage:
#   scripts/run_macag_acdc_parallel.sh                 # 3 shards, ACDC prompts
#   NUM_WORKERS=4 scripts/run_macag_acdc_parallel.sh   # 4 shards
#   JSON=macag/data/nonlinear_benchmark_prompts.json \
#   OUTROOT=results/macag_nonlinear \
#       scripts/run_macag_acdc_parallel.sh             # nonlinear set
#
# Any run_macag_acdc.sh env var is forwarded to every shard, e.g.:
#   CLTS, TASKS, LIMIT, DEVICE, FREEZE_MODE, SOLVERS, SKIP_BASELINES,
#   SHAPLEY_PERMUTATIONS, BASELINE_METHODS
#
# Knobs specific to this launcher:
#   NUM_WORKERS   number of concurrent shards (default 3)
#   STAGGER       seconds to wait between shard launches (default 45)
#   JSON, OUTROOT prompt set / output root (forwarded; shown here for clarity)
#   RUN_ANALYSIS  set 0 to skip the final aggregate step
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NUM_WORKERS="${NUM_WORKERS:-8}"
STAGGER="${STAGGER:-45}"
JSON="${JSON:-macag/data/acdc_benchmark_prompts.json}"
OUTROOT="${OUTROOT:-results/macag_acdc}"
FREEZE_MODE="${FREEZE_MODE:-both}"

if ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || (( NUM_WORKERS < 1 )); then
  echo "ERROR: NUM_WORKERS must be a positive integer (got '$NUM_WORKERS')" >&2
  exit 2
fi

mkdir -p "$OUTROOT"
echo ">>> Parallel MACAG sweep | workers=$NUM_WORKERS stagger=${STAGGER}s json=$JSON outroot=$OUTROOT"

pids=()
# Kill all shards if the launcher is interrupted.
cleanup() {
  echo ">>> Interrupted — terminating ${#pids[@]} shard(s) ..."
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup INT TERM

for (( w = 0; w < NUM_WORKERS; w++ )); do
  out_log="$OUTROOT/shard.$w.out"
  echo ">>> launching worker $w/$NUM_WORKERS -> $out_log"
  NUM_WORKERS="$NUM_WORKERS" WORKER_ID="$w" \
  JSON="$JSON" OUTROOT="$OUTROOT" FREEZE_MODE="$FREEZE_MODE" \
  RUN_ANALYSIS=0 \
    nohup scripts/run_macag_acdc.sh > "$out_log" 2>&1 &
  pids+=("$!")
  # Stagger every launch except the last, to offset concurrent attribution peaks.
  if (( w < NUM_WORKERS - 1 )); then sleep "$STAGGER"; fi
done

echo ">>> All $NUM_WORKERS shards launched (pids: ${pids[*]}). Waiting ..."
echo ">>> Tip: tail -f $OUTROOT/shard.*.out   |   watch -n2 nvidia-smi"

fail=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    echo ">>> worker $idx finished OK"
  else
    echo ">>> worker $idx FAILED (exit $?) — see $OUTROOT/shard.$idx.out" >&2
    fail=1
  fi
done
trap - INT TERM

echo ""; echo ">>> Combined status across shards:"
cat "$OUTROOT"/status.w*.txt 2>/dev/null | sort | uniq -c | sort -rn | head

if [[ "$fail" -ne 0 ]]; then
  echo ">>> WARNING: at least one shard failed; aggregating available results anyway." >&2
fi

# --- aggregate once, over the combined tree --------------------------------
if [[ "${RUN_ANALYSIS:-1}" != "0" ]]; then
  echo ""; echo ">>> Aggregating (ANALYZE_ONLY) ..."
  ANALYZE_ONLY=1 RUN_ANALYSIS=1 \
  JSON="$JSON" OUTROOT="$OUTROOT" FREEZE_MODE="$FREEZE_MODE" \
    scripts/run_macag_acdc.sh
fi

echo ""; echo ">>> Parallel sweep complete. Output under $OUTROOT/"
exit "$fail"
