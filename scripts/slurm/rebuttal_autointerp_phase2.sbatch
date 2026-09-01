#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J rebuttal-autointerp-p2
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p ai4wy-2
#SBATCH -t 12:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/rebuttal_autointerp_p2_%j.out
#SBATCH -e logs/slurm/rebuttal_autointerp_p2_%j.err
#SBATCH --signal=B:TERM@120

# REQ-4 auto-interp phase 2: the six checkpoints not covered by the first run
# (gpt2-small v3 param-match pair, gpt2-large v2 pair, qwen3-0.6b v2 pair).
# Requires the vLLM server (scripts/slurm/launch_vllm.sbatch) to be up by the
# time each collection phase ends. All phases resume from JSONL caches.
#
# Submit:  sbatch scripts/slurm/rebuttal_autointerp_phase2.sbatch

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

module load arcc miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/gscratch/ssuresh}"
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
OUT="${REPO_ROOT}/results/rebuttal"

run_autointerp () {
    local label="$1" base_model="$2" ckpt="$3"
    echo "[$(date -u +%FT%TZ)] autointerp ${label} (${base_model}, n_features=${N_FEATURES})"
    python -m rebuttal_eval.autointerp.run \
        --checkpoint "${ckpt}" \
        --model "${base_model}" \
        --out-dir "${OUT}/autointerp_${label}" \
        --backend openai_compat \
        --endpoint-file "${ENDPOINT_FILE}" \
        --llm-model "" \
        --n-features "${N_FEATURES}" \
        --label "${label}"
}

# gpt2-small v3 param-match pair
run_autointerp spline_pm_v3 gpt2 \
    "$V3/spline_param_match_gpt2_small_pv3/seed_101/checkpoints/spline_pm_gpt2_small_pv3_seed101_best"
run_autointerp linear_pm_v3 gpt2 \
    "$V3/linear_param_match_gpt2_small_pv3/seed_101/checkpoints/linear_pm_gpt2_small_pv3_seed101_best"

# gpt2-large v2 pair
run_autointerp spline_fm_gpt2l gpt2-large \
    "$R/paper_v2_gpt2_large/runs/spline_feature_match_gpt2_large_pv2/seed_101/checkpoints/spline_fm_gpt2_large_pv2_seed101_best"
run_autointerp linear_fm_gpt2l gpt2-large \
    "$R/paper_v2_gpt2_large/runs/linear_feature_match_gpt2_large_pv2/seed_101/checkpoints/linear_fm_gpt2_large_pv2_seed101_best"

# qwen3-0.6b v2 pair
run_autointerp spline_fm_qwen "Qwen/Qwen3-0.6B" \
    "$R/paper_v2_qwen3_06b/runs/spline_feature_match_qwen3_06b_pv2/seed_101/checkpoints/spline_fm_qwen3_06b_pv2_seed101_best"
run_autointerp linear_fm_qwen "Qwen/Qwen3-0.6B" \
    "$R/paper_v2_qwen3_06b/runs/linear_feature_match_qwen3_06b_pv2/seed_101/checkpoints/linear_fm_qwen3_06b_pv2_seed101_best"

echo "[$(date -u +%FT%TZ)] autointerp phase 2 complete"
