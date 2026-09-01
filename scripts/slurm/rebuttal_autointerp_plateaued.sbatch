#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J rebuttal-autointerp-plat
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p ai4wy-2
#SBATCH -t 24:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/rebuttal_autointerp_plateaued_%j.out
#SBATCH -e logs/slurm/rebuttal_autointerp_plateaued_%j.err
#SBATCH --signal=B:TERM@120

# Auto-interp: collect (GPU) -> explain -> score (vLLM endpoint) for the
# 12 currently-plateaued v3 checkpoints (8 dt_sweep arms + rebuttal_gpt2_small
# fm/pm spline+linear pairs). Requires a running vLLM server advertised via
# ENDPOINT_FILE (scripts/slurm/launch_vllm.sbatch).
#
# Submit:  sbatch scripts/slurm/rebuttal_autointerp_plateaued.sbatch
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
ENDPOINT_FILE="${ENDPOINT_FILE:-${REPO_ROOT}/results/rebuttal/vllm_endpoint.json}"
N_FEATURES="${N_FEATURES:-200}"
LLM_MODEL="${LLM_MODEL:-}"
OUT="${REPO_ROOT}/results/rebuttal"

run_autointerp () {
    local label="$1" ckpt="$2"
    echo "[$(date -u +%FT%TZ)] autointerp ${label} (n_features=${N_FEATURES}) -> ${OUT}/autointerp_${label}"
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

# --- dt_sweep spline arms ---
run_autointerp dtsweep_spline_dt768 \
    "$R/paper_v3_dt_sweep_gpt2_small_spline_dt768/runs/spline_dt768_gpt2_small_pv3/seed_101/checkpoints/spline_dt768_gpt2_small_pv3_seed101_best"
run_autointerp dtsweep_spline_dt1536 \
    "$R/paper_v3_dt_sweep_gpt2_small_spline_dt1536/runs/spline_dt1536_gpt2_small_pv3/seed_101/checkpoints/spline_dt1536_gpt2_small_pv3_seed101_best"
run_autointerp dtsweep_spline_dt3072 \
    "$R/paper_v3_dt_sweep_gpt2_small_spline_dt3072/runs/spline_dt3072_gpt2_small_pv3/seed_101/checkpoints/spline_dt3072_gpt2_small_pv3_seed101_best"
run_autointerp dtsweep_spline_dt6144 \
    "$R/paper_v3_dt_sweep_gpt2_small_spline_dt6144/runs/spline_dt6144_gpt2_small_pv3/seed_101/checkpoints/spline_dt6144_gpt2_small_pv3_seed101_best"

# --- dt_sweep linear arms ---
run_autointerp dtsweep_linear_dt1664 \
    "$R/paper_v3_dt_sweep_gpt2_small_linear_dt1664/runs/linear_dt1664_gpt2_small_pv3/seed_101/checkpoints/linear_dt1664_gpt2_small_pv3_seed101_best"
run_autointerp dtsweep_linear_dt3392 \
    "$R/paper_v3_dt_sweep_gpt2_small_linear_dt3392/runs/linear_dt3392_gpt2_small_pv3/seed_101/checkpoints/linear_dt3392_gpt2_small_pv3_seed101_best"
run_autointerp dtsweep_linear_dt6784 \
    "$R/paper_v3_dt_sweep_gpt2_small_linear_dt6784/runs/linear_dt6784_gpt2_small_pv3/seed_101/checkpoints/linear_dt6784_gpt2_small_pv3_seed101_best"
run_autointerp dtsweep_linear_dt13504 \
    "$R/paper_v3_dt_sweep_gpt2_small_linear_dt13504/runs/linear_dt13504_gpt2_small_pv3/seed_101/checkpoints/linear_dt13504_gpt2_small_pv3_seed101_best"

# --- rebuttal_gpt2_small feature-match pair ---
run_autointerp rebuttal_spline_fm \
    "$R/paper_v3_rebuttal_gpt2_small_spline_fm/runs/spline_feature_match_gpt2_small_pv3/seed_101/checkpoints/spline_fm_gpt2_small_pv3_seed101_best"
run_autointerp rebuttal_linear_fm \
    "$R/paper_v3_rebuttal_gpt2_small_linear_fm/runs/linear_feature_match_gpt2_small_pv3/seed_101/checkpoints/linear_fm_gpt2_small_pv3_seed101_best"

# --- rebuttal_gpt2_small param-match pair ---
run_autointerp rebuttal_spline_pm \
    "$R/paper_v3_rebuttal_gpt2_small_spline_pm/runs/spline_param_match_gpt2_small_pv3/seed_101/checkpoints/spline_pm_gpt2_small_pv3_seed101_best"
run_autointerp rebuttal_linear_pm \
    "$R/paper_v3_rebuttal_gpt2_small_linear_pm/runs/linear_param_match_gpt2_small_pv3/seed_101/checkpoints/linear_pm_gpt2_small_pv3_seed101_best"

echo "[$(date -u +%FT%TZ)] autointerp (plateaued batch) complete"
