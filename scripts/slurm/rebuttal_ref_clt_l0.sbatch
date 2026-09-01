#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J rebuttal-ref-clt-l0
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p ai4wy-2
#SBATCH -t 02:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/rebuttal_ref_clt_l0_%j.out
#SBATCH -e logs/slurm/rebuttal_ref_clt_l0_%j.err
#SBATCH --signal=B:TERM@60

# Measure L0 of published linear CLTs (mntss llama + gemma) on wikitext.
# Gives the sparsity target for a healthy linear baseline (HANDOFF §D.1).
#
# Submit:  sbatch scripts/slurm/rebuttal_ref_clt_l0.sbatch
# Optional env: SKIP_GEMMA=1  N_SEQ=64  MAX_LEN=128

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm" "${REPO_ROOT}/results/rebuttal/ref_clt_l0"

module load arcc miniconda3
# Prefer the user env explicitly — `conda activate` can silently miss on
# some compute-node module stacks, leaving a bare python without numpy.
CT_PYTHON="${CT_PYTHON:-${HOME}/.conda/envs/ct/bin/python}"
if [[ ! -x "${CT_PYTHON}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ct
    CT_PYTHON="$(command -v python)"
fi
echo "using python: ${CT_PYTHON} ($("${CT_PYTHON}" -c 'import sys; print(sys.executable)'))"
"${CT_PYTHON}" -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/gscratch/ssuresh}"
# Keep hub online-capable: transformers tokenizers ping model_info even with a
# warm cache. The Python side resolves repos to local snapshot paths so no
# downloads occur; unset OFFLINE so that ping is not hard-failed.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Reap stray GPU processes belonging to this user.
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true); do
    if [[ "$(ps -o user= -p "${pid}" 2>/dev/null)" == "${USER}" ]]; then
        echo "reaping stray GPU pid ${pid}"
        kill -9 "${pid}" || true
    fi
done

N_SEQ="${N_SEQ:-64}"
MAX_LEN="${MAX_LEN:-128}"
OUT="${REPO_ROOT}/results/rebuttal/ref_clt_l0"

EXTRA=()
if [[ "${SKIP_GEMMA:-0}" == "1" ]]; then
    EXTRA+=(--skip-gemma)
fi

echo "[$(date -u +%FT%TZ)] ref CLT L0: n_seq=${N_SEQ} max_len=${MAX_LEN} skip_gemma=${SKIP_GEMMA:-0}"

"${CT_PYTHON}" -m rebuttal_eval.measure_ref_clt_l0 \
    --out-dir "${OUT}" \
    --n-sequences "${N_SEQ}" \
    --max-seq-len "${MAX_LEN}" \
    --seed 101 \
    --device cuda \
    --dtype bfloat16 \
    "${EXTRA[@]}"

echo "[$(date -u +%FT%TZ)] done"
