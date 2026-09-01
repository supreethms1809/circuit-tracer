#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J rebuttal-autointerp
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p ai4wy-2
#SBATCH -t 12:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/rebuttal_autointerp_%j.out
#SBATCH -e logs/slurm/rebuttal_autointerp_%j.err
#SBATCH --signal=B:TERM@120

# REQ-4 auto-interp: collect (GPU) -> explain -> score (vLLM endpoint) for
# the v3 spline FM and linear FM checkpoints. Requires a running vLLM server
# advertised via ENDPOINT_FILE (scripts/slurm/launch_vllm.sbatch). Collection
# runs first, so the server only needs to be up by the explain phase; the
# client then polls the endpoint file for up to 30 minutes.
#
# Submit:  sbatch scripts/slurm/rebuttal_autointerp.sbatch
# Env: N_FEATURES (default 200), ENDPOINT_FILE, LLM_MODEL

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

R=/gscratch/ssuresh/results/paper
V3=$R/paper_v3_gpt2_small/runs
ENDPOINT_FILE="${ENDPOINT_FILE:-${REPO_ROOT}/results/rebuttal/vllm_endpoint.json}"
N_FEATURES="${N_FEATURES:-200}"
LLM_MODEL="${LLM_MODEL:-}"
OUT="${REPO_ROOT}/results/rebuttal"

run_autointerp () {
    local label="$1" ckpt="$2"
    echo "[$(date -u +%FT%TZ)] autointerp ${label} (n_features=${N_FEATURES})"
    python -m rebuttal_eval.autointerp.run \
        --checkpoint "${ckpt}" \
        --model gpt2 \
        --out-dir "${OUT}/autointerp_${label}" \
        --backend openai_compat \
        --endpoint-file "${ENDPOINT_FILE}" \
        --llm-model "${LLM_MODEL}" \
        --n-features "${N_FEATURES}" \
        --label "${label}"
}

run_autointerp spline_fm_v3 \
    "$V3/spline_feature_match_gpt2_small_pv3/seed_101/checkpoints/spline_fm_gpt2_small_pv3_seed101_best"
run_autointerp linear_fm_v3 \
    "$V3/linear_feature_match_gpt2_small_pv3/seed_101/checkpoints/linear_fm_gpt2_small_pv3_seed101_best"

echo "[$(date -u +%FT%TZ)] autointerp complete"
