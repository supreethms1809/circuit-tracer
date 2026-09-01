#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J ravel-eval
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p gp-1
#SBATCH -t 24:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/ravel_eval_%j.out
#SBATCH -e logs/slurm/ravel_eval_%j.err
#SBATCH --signal=B:TERM@120

# Single-GPU RAVEL benchmark eval on gp-1 (gp101–gp112, 1×H200).
# stages.train/macag are false in these configs.
#
# Submit:  sbatch scripts/slurm/ravel_eval_run.sbatch <suite.json>

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: sbatch $0 SUITE_JSON_PATH" >&2
    exit 2
fi
SUITE="$1"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

module load arcc miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct
# Explicit interpreter — `conda activate` can leave a non-ct python on ai4wy-1.
PYTHON="${CONDA_PREFIX:-}/bin/python"
if [[ ! -x "${PYTHON}" ]] || ! "${PYTHON}" -c "import torch" 2>/dev/null; then
    PYTHON=/home/ssuresh/.conda/envs/ct/bin/python
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    if [[ "$(ps -o user= -p "${pid}" 2>/dev/null)" == "${USER}" ]]; then
        echo "reaping stray GPU pid ${pid}"
        kill -9 "${pid}" || true
    fi
done

EXTRA=()
if [[ "${RE_EVALUATE:-0}" == "1" ]]; then
    EXTRA+=(--re-evaluate)
fi

echo "[$(date -u +%FT%TZ)] ravel eval: ${SUITE} python=${PYTHON} extra=${EXTRA[*]:-none}"
"${PYTHON}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"${PYTHON}" -m spline_clt.paper_eval --suite "${SUITE}" "${EXTRA[@]}"
echo "[$(date -u +%FT%TZ)] ravel eval complete: ${SUITE}"
