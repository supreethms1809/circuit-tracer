#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J rebuttal-suite-eval
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p ai4wy-2
#SBATCH -t 24:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/rebuttal_suite_%j.out
#SBATCH -e logs/slurm/rebuttal_suite_%j.err
#SBATCH --signal=B:TERM@120

# Run one rebuttal evaluation suite (eval + report stages only) on a single
# GPU. Suite name is the sbatch argument; defaults to the v3 RAVEL pair.
#
# Submit (args: suite name, worker id, num workers). With 2 variants x 1 seed
# and 2 workers, each job evaluates exactly one encoder arm:
#   sbatch scripts/slurm/rebuttal_suite_eval.sbatch paper_gpt2_small_ravel_v3_fm 0 2
#   sbatch scripts/slurm/rebuttal_suite_eval.sbatch paper_gpt2_small_ravel_v3_fm 1 2

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

module load arcc miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    if [[ "$(ps -o user= -p "${pid}" 2>/dev/null)" == "${USER}" ]]; then
        echo "reaping stray GPU pid ${pid}"
        kill -9 "${pid}" || true
    fi
done

SUITE_NAME="${1:-paper_gpt2_small_ravel_v3_fm}"
WORKER_ID="${2:-0}"
NUM_WORKERS="${3:-1}"
SUITE="experiments/paper_configs/suites/${SUITE_NAME}.json"
if [[ ! -f "${SUITE}" ]]; then
    echo "suite config not found: ${SUITE}" >&2
    exit 1
fi

echo "[$(date -u +%FT%TZ)] paper_eval suite ${SUITE_NAME} (worker ${WORKER_ID}/${NUM_WORKERS})"
python -m spline_clt.paper_eval --suite "${SUITE}" \
    --worker-id "${WORKER_ID}" --num-workers "${NUM_WORKERS}"
echo "[$(date -u +%FT%TZ)] suite ${SUITE_NAME} worker ${WORKER_ID} complete"
