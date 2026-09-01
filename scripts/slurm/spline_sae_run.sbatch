#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J spline-sae
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p ai4wy-1
#SBATCH -t 24:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/spline_sae_%j.out
#SBATCH -e logs/slurm/spline_sae_%j.err
#SBATCH --signal=B:TERM@120

# 1-GPU Spline-SAE baseline / analog training.
# Submit: sbatch scripts/slurm/spline_sae_run.sbatch <config.yaml>

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: sbatch $0 CONFIG_YAML" >&2
    exit 2
fi
CFG="$1"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm" /gscratch/ssuresh/results/spline_sae

module load arcc miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct
PYTHON="${CONDA_PREFIX:-}/bin/python"
if [[ ! -x "${PYTHON}" ]] || ! "${PYTHON}" -c "import torch" 2>/dev/null; then
    PYTHON=/home/ssuresh/.conda/envs/ct/bin/python
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/gscratch/ssuresh}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

# Reap stray GPU procs from prior jobs on this node.
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true); do
    if [[ "$(ps -o user= -p "${pid}" 2>/dev/null)" == "${USER}" ]]; then
        echo "reaping stray GPU pid ${pid}"
        kill -9 "${pid}" || true
    fi
done

echo "[$(date -u +%FT%TZ)] spline_sae train cfg=${CFG} python=${PYTHON}"
"${PYTHON}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"${PYTHON}" -m spline_sae.train --config "${CFG}"
echo "[$(date -u +%FT%TZ)] spline_sae complete cfg=${CFG}"
