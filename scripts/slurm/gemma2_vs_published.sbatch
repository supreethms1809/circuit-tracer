#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J gemma2-vs-pub
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p gp-2
#SBATCH -t 04:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/gemma2_vs_published_%j.out
#SBATCH -e logs/slurm/gemma2_vs_published_%j.err
#SBATCH --signal=B:TERM@60

# Compare trained Gemma-2 Spline-CLT (dt6144 best) to published mntss linear
# CLT (d_t=16384) on the same val activation mmap.
#
# Submit: sbatch scripts/slurm/gemma2_vs_published.sbatch

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

module load arcc miniconda3
CT_PYTHON="${CT_PYTHON:-${HOME}/.conda/envs/ct/bin/python}"
if [[ ! -x "${CT_PYTHON}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ct
    CT_PYTHON="$(command -v python)"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/gscratch/ssuresh}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true); do
    if [[ "$(ps -o user= -p "${pid}" 2>/dev/null)" == "${USER}" ]]; then
        echo "reaping stray GPU pid ${pid}"
        kill -9 "${pid}" || true
    fi
done

CKPT="${SPLINE_CKPT:-/gscratch/ssuresh/results/paper_r5b_b3/paper_r5b_b3_gemma2_2b_spline_dt6144_basejump/runs/gemma2_2b_spline_dt6144_r5b_b3_bj/seed_101/checkpoints/gemma2_2b_spline_dt6144_r5b_b3_basejump_seed101_best}"
ACT="${ACT_DIR:-/gscratch/ssuresh/shared/activations/paper_r5b_b3/gemma2_2b_spline_dt6144_basejump/val/google_gemma-2-2b}"
OUT="${OUT_DIR:-${REPO_ROOT}/results/rebuttal/gemma2_vs_published}"
N="${N_SAMPLES:-256}"

echo "[$(date -u +%FT%TZ)] compare spline=${CKPT}"
echo "[$(date -u +%FT%TZ)] activations=${ACT}"

"${CT_PYTHON}" -m rebuttal_eval.compare_gemma2_published \
    --spline-checkpoint "${CKPT}" \
    --activation-dir "${ACT}" \
    --out-dir "${OUT}" \
    --n-samples "${N}" \
    --seed 101 \
    --device cuda \
    --label gemma2_dt6144_best_vs_mntss426k \
    --allow-hub-download

# Also emit the full §4.1–4.3 reconstruction report for the spline alone.
"${CT_PYTHON}" -m rebuttal_eval.check_reconstruction \
    --checkpoint "${CKPT}" \
    --activation-dir "${ACT}" \
    --out-dir "${OUT}/spline_check_reconstruction" \
    --n-samples "${N}" \
    --seed 101 \
    --device cuda \
    --label gemma2_dt6144_best

echo "[$(date -u +%FT%TZ)] done"
