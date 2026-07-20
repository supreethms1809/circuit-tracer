#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J rescore-val
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p ai4wy-2
#SBATCH -t 00:30:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/rescore_val_%j.out
#SBATCH -e logs/slurm/rescore_val_%j.err
#SBATCH --signal=B:TERM@60

# Re-score _best and _final on the raw vs de-duplicated val holdout. Read-only.
# Submit: sbatch scripts/slurm/rescore_clean_val.sbatch

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

module load arcc miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "[$(date -u +%FT%TZ)] rescore-val on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "nvidia-smi failed"

# Stage checkpoints (and clean val) to node-local /lscratch so slow/contended
# shared-FS reads can't stall load_spline_clt (the earlier 15-min "loading" hangs).
SRC="/cluster/ai4wy/gscratch/ssuresh/results/paper/paper_gpt2_small_spline_1b_v2/runs/spline_feature_match_gpt2_small_1b_v2/seed_101/checkpoints"
STAGE="/lscratch/rescore_${SLURM_JOB_ID}"
mkdir -p "${STAGE}/val_clean/gpt2"
echo "[stage] copying checkpoints + clean val to ${STAGE} ..."
cp -r "${SRC}/spline_fm_gpt2_small_1b_v2_seed101_best"  "${STAGE}/best"
cp -r "${SRC}/spline_fm_gpt2_small_1b_v2_seed101_final" "${STAGE}/final"
cp /gscratch/ssuresh/shared/activations/gpt2_2b_v2/val_clean/gpt2/*.npy "${STAGE}/val_clean/gpt2/"
echo "[stage] done ($(date -u +%FT%TZ))"

# Load models on CPU (the efficient-kan curve2coeff lstsq deadlocks on the GH200
# GPU at construction), then the script moves them to GPU for eval.
export PYTHONUNBUFFERED=1
python -u scripts/rescore_clean_val.py \
    --ckpt best="${STAGE}/best" final="${STAGE}/final" \
    --val  clean="${STAGE}/val_clean/gpt2"

rm -rf "${STAGE}" 2>/dev/null || true
echo "[$(date -u +%FT%TZ)] rescore-val complete"
