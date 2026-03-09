#!/usr/bin/env bash
# submit_paper_eval.sh — PBS Pro 3-phase orchestration for paper-eval
#
# Usage:
#   bash scripts/pbs/submit_paper_eval.sh \
#       --suite <path/to/suite.json> \
#       [--max-workers 32] \
#       [--conda-env ct] \
#       [--dry-run]
#
# Phases:
#   Phase 1 (1 GPU):  collect dataset
#   Phase 2 (N GPUs): array of train+evaluate+macag workers (afterok Phase 1)
#   Phase 3 (0 GPU):  report aggregation (afterok all Phase 2 tasks)

set -euo pipefail

# --- Defaults ---
SUITE=""
MAX_WORKERS=32
CONDA_ENV="${CONDA_ENV:-ct}"
DRY_RUN=false
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite)        SUITE="$2";        shift 2 ;;
        --max-workers)  MAX_WORKERS="$2";  shift 2 ;;
        --conda-env)    CONDA_ENV="$2";    shift 2 ;;
        --dry-run)      DRY_RUN=true;      shift   ;;
        *) echo "Unknown option: $1" >&2; exit 1   ;;
    esac
done

if [[ -z "$SUITE" ]]; then
    echo "Error: --suite is required" >&2
    exit 1
fi

SUITE="$(realpath "$SUITE")"

# --- Count trainable jobs via dry-run ---
echo "Counting train jobs in suite: $SUITE"
NUM_JOBS=$(
    conda run -n "${CONDA_ENV}" python -m spline_clt.paper_eval \
        --suite "${SUITE}" --dry-run 2>/dev/null \
    | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(sum(1 for j in d.get('jobs', []) if j.get('stage') == 'train'))
"
)

if [[ -z "$NUM_JOBS" || "$NUM_JOBS" -eq 0 ]]; then
    echo "Error: could not determine number of train jobs from dry-run" >&2
    exit 1
fi

ACTUAL_WORKERS=$(( NUM_JOBS < MAX_WORKERS ? NUM_JOBS : MAX_WORKERS ))
ARRAY_MAX=$(( ACTUAL_WORKERS - 1 ))

echo "Train jobs:     ${NUM_JOBS}"
echo "Max workers:    ${MAX_WORKERS}"
echo "Actual workers: ${ACTUAL_WORKERS}"
echo "Array range:    0-${ARRAY_MAX}"
echo "Conda env:      ${CONDA_ENV}"
echo "Repo root:      ${REPO_ROOT}"

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "[dry-run] Would submit:"
    echo "  Phase 1: qsub -v SUITE_PATH=${SUITE},CONDA_ENV=${CONDA_ENV},REPO_ROOT=${REPO_ROOT} 01_collect_dataset.sh"
    echo "  Phase 2: qsub -J 0-${ARRAY_MAX} -W depend=afterok:<JOB1> -v ...,NUM_WORKERS=${ACTUAL_WORKERS} 02_train_eval_array.sh"
    echo "  Phase 3: qsub -W depend=afterok:<JOB2>[] -v ... 03_generate_report.sh"
    exit 0
fi

# Ensure log dir exists relative to REPO_ROOT
mkdir -p "${REPO_ROOT}/logs/pbs"

PBS_SCRIPTS_DIR="${REPO_ROOT}/scripts/pbs"

# --- Phase 1: collect dataset ---
JOB1=$(qsub \
    -v "SUITE_PATH=${SUITE},CONDA_ENV=${CONDA_ENV},REPO_ROOT=${REPO_ROOT}" \
    "${PBS_SCRIPTS_DIR}/01_collect_dataset.sh")
echo "Submitted Phase 1 (collect): ${JOB1}"

# --- Phase 2: train+evaluate+macag array ---
JOB2=$(qsub \
    -J "0-${ARRAY_MAX}" \
    -W "depend=afterok:${JOB1}" \
    -v "SUITE_PATH=${SUITE},CONDA_ENV=${CONDA_ENV},REPO_ROOT=${REPO_ROOT},NUM_WORKERS=${ACTUAL_WORKERS}" \
    "${PBS_SCRIPTS_DIR}/02_train_eval_array.sh")
echo "Submitted Phase 2 (train+eval, ${ACTUAL_WORKERS} workers): ${JOB2}"

# --- Phase 3: report (waits for ALL array tasks) ---
JOB3=$(qsub \
    -W "depend=afterok:${JOB2}[]" \
    -v "SUITE_PATH=${SUITE},CONDA_ENV=${CONDA_ENV},REPO_ROOT=${REPO_ROOT}" \
    "${PBS_SCRIPTS_DIR}/03_generate_report.sh")
echo "Submitted Phase 3 (report): ${JOB3}"

echo ""
echo "Monitor with:"
echo "  qstat -u \$USER"
echo "  qstat -t ${JOB2}   # array task status"
