#!/usr/bin/env bash
# run_local_multigpu.sh — Run paper-eval across N GPUs on a single machine.
#
# Usage:
#   bash scripts/run_local_multigpu.sh \
#       --suite <path/to/suite.json> \
#       [--num-gpus 8] \
#       [--conda-env ct] \
#       [--log-dir logs/multigpu]
#
# Phases:
#   Phase 1 (GPU 0):  collect dataset
#   Phase 2 (N GPUs): parallel train+evaluate+macag workers
#   Phase 3 (CPU):    report aggregation

set -euo pipefail

# --- Defaults ---
SUITE=""
NUM_GPUS=8
CONDA_ENV="${CONDA_ENV:-ct}"
LOG_DIR="logs/multigpu"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite)      SUITE="$2";      shift 2 ;;
        --num-gpus)   NUM_GPUS="$2";   shift 2 ;;
        --conda-env)  CONDA_ENV="$2";  shift 2 ;;
        --log-dir)    LOG_DIR="$2";    shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$SUITE" ]]; then
    echo "Error: --suite is required" >&2
    exit 1
fi

SUITE="$(realpath "$SUITE")"
mkdir -p "${LOG_DIR}"

# --- Count train jobs, cap workers at actual job count ---
echo "Counting train jobs..."
NUM_JOBS=$(
    conda run -n "${CONDA_ENV}" python -m spline_clt.paper_eval \
        --suite "${SUITE}" --dry-run 2>/dev/null \
    | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(sum(1 for j in d.get('jobs', []) if j.get('stage') == 'train'))
"
)

ACTUAL_WORKERS=$(( NUM_JOBS < NUM_GPUS ? NUM_JOBS : NUM_GPUS ))
echo "Train jobs: ${NUM_JOBS}, workers: ${ACTUAL_WORKERS} (of ${NUM_GPUS} GPUs available)"

# --- Phase 1: collect dataset (GPU 0) ---
echo ""
echo "=== Phase 1: collect dataset (GPU 0) ==="
CUDA_VISIBLE_DEVICES=0 conda run -n "${CONDA_ENV}" python -m spline_clt.paper_eval \
    --suite "${SUITE}" \
    --stages collect \
    --worker-id 0 \
    --num-workers 1 \
    2>&1 | tee "${LOG_DIR}/collect.log"
echo "Phase 1 complete."

# --- Phase 2: parallel train+evaluate+macag ---
echo ""
echo "=== Phase 2: train+evaluate+macag (${ACTUAL_WORKERS} workers) ==="
PIDS=()
for (( i=0; i<ACTUAL_WORKERS; i++ )); do
    GPU_ID=$(( i % NUM_GPUS ))
    LOG_FILE="${LOG_DIR}/worker_${i}.log"
    echo "  Starting worker ${i} on GPU ${GPU_ID} → ${LOG_FILE}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" conda run -n "${CONDA_ENV}" python -m spline_clt.paper_eval \
        --suite "${SUITE}" \
        --stages train evaluate macag \
        --worker-id "${i}" \
        --num-workers "${ACTUAL_WORKERS}" \
        > "${LOG_FILE}" 2>&1 &
    PIDS+=($!)
done

# Wait for all workers; collect exit codes.
FAILED=0
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    if wait "${PID}"; then
        echo "  Worker ${i} (PID ${PID}) finished OK"
    else
        echo "  Worker ${i} (PID ${PID}) FAILED (exit $?)" >&2
        FAILED=$(( FAILED + 1 ))
    fi
done

if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo "Error: ${FAILED} worker(s) failed. Skipping report." >&2
    exit 1
fi
echo "Phase 2 complete."

# --- Phase 3: report (CPU only) ---
echo ""
echo "=== Phase 3: generate report ==="
conda run -n "${CONDA_ENV}" python -m spline_clt.paper_eval \
    --suite "${SUITE}" \
    --stages report \
    --worker-id 0 \
    --num-workers 1 \
    2>&1 | tee "${LOG_DIR}/report.log"
echo "Phase 3 complete."

echo ""
echo "Done. Logs: ${LOG_DIR}/"
