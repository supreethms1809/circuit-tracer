#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J r2-ravel-interp-llama
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p ai4wy-1
#SBATCH -t 20:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/rebuttal_autointerp_ravel_r2_llama32_%j.out
#SBATCH -e logs/slurm/rebuttal_autointerp_ravel_r2_llama32_%j.err
#SBATCH --signal=B:TERM@120

# Graph-targeted autointerp for the r2 llama32_1b_spline_dt8192 ravel eval
# suite: extract unique feature IDs from evaluation/graphs/*.json,
# collect->explain->score on exactly those IDs, write explanations into
# annotated graph clerps. Client-side only needs 1 GPU (ai4wy-1) -- talks to
# the remote vLLM endpoint over HTTP.
#
# Prerequisites:
#   - r2 llama32_1b_spline_dt8192 ravel eval suite finished under
#     /gscratch/ssuresh/results/paper_rebuttal/ravel_eval_suite_r2_llama32_1b_spline_dt8192
#   - vLLM endpoint at ENDPOINT_FILE (scripts/slurm/launch_vllm.sbatch)
#
# Env: ENDPOINT_FILE, LLM_MODEL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

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
# Do NOT force HF_HUB_OFFLINE here: transformers' tokenizer loader
# unconditionally calls is_base_mistral() -> model_info() as part of a
# regex-patch compatibility check, and offline mode makes that raise
# OfflineModeIsEnabled instead of falling back to cache (job 8806, 2026-07-28).
# This is a single job, not a fan-out of parallel hub-eval jobs, so the
# rate-limit concern that motivated offline mode elsewhere doesn't apply.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "[$(date -u +%FT%TZ)] python=${PYTHON}"
"${PYTHON}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    if [[ "$(ps -o user= -p "${pid}" 2>/dev/null)" == "${USER}" ]]; then
        echo "reaping stray GPU pid ${pid}"
        kill -9 "${pid}" || true
    fi
done

R=/gscratch/ssuresh/results/paper_rebuttal
ENDPOINT_FILE="${ENDPOINT_FILE:-${REPO_ROOT}/results/rebuttal/vllm_endpoint.json}"
LLM_MODEL="${LLM_MODEL:-}"
OUT="${R}"

run_name_graphs () {
    local label="$1" graphs_dir="$2" ckpt="$3" model="$4"
    local out_dir="${OUT}/autointerp_ravel_${label}"
    local n_graphs
    n_graphs=$(find "${graphs_dir}" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
    if [[ "${n_graphs}" -lt 1 ]]; then
        echo "[$(date -u +%FT%TZ)] SKIP ${label}: no graphs in ${graphs_dir}" >&2
        return 0
    fi
    if [[ ! -d "${ckpt}" ]]; then
        echo "[$(date -u +%FT%TZ)] SKIP ${label}: checkpoint missing ${ckpt}" >&2
        return 0
    fi
    echo "[$(date -u +%FT%TZ)] name_graphs ${label} (n_graphs=${n_graphs}) -> ${out_dir}"
    if ! "${PYTHON}" -m rebuttal_eval.autointerp.name_graphs \
        --graphs-dir "${graphs_dir}" \
        --checkpoint "${ckpt}" \
        --model "${model}" \
        --out-dir "${out_dir}" \
        --backend openai_compat \
        --endpoint-file "${ENDPOINT_FILE}" \
        --llm-model "${LLM_MODEL}" \
        --label "${label}" \
        --include-scores-in-clerp; then
        echo "[$(date -u +%FT%TZ)] FAILED ${label} (continuing)" >&2
        return 0
    fi

    local mirror
    mirror="$(dirname "${graphs_dir}")/graphs_autointerp"
    mkdir -p "${mirror}"
    cp -f "${out_dir}/annotated_graphs/"*.json "${mirror}/" 2>/dev/null || true
    if [[ -f "${out_dir}/annotated_graphs/clerp_writeback_summary.json" ]]; then
        cp -f "${out_dir}/annotated_graphs/clerp_writeback_summary.json" "${mirror}/"
    fi
    echo "[$(date -u +%FT%TZ)] mirrored annotated graphs -> ${mirror}"
}

run_name_graphs r2_llama32_1b_spline_dt8192 \
    "$R/ravel_eval_suite_r2_llama32_1b_spline_dt8192/runs/r2_llama32_1b_spline_dt8192/seed_101/evaluation/graphs" \
    "$R/paper_r2_llama32_1b_spline_dt8192/runs/llama32_1b_spline_dt8192_r2/seed_101/checkpoints/llama32_1b_spline_dt8192_r2_seed101_best" \
    "meta-llama/Llama-3.2-1B"

echo "[$(date -u +%FT%TZ)] r2 llama32_1b ravel graph autointerp complete"
