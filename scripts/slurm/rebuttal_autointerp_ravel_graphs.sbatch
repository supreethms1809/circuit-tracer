#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J ravel-name-graphs
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p ai4wy-2
#SBATCH -t 24:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/rebuttal_autointerp_ravel_graphs_%j.out
#SBATCH -e logs/slurm/rebuttal_autointerp_ravel_graphs_%j.err
#SBATCH --signal=B:TERM@120

# Graph-targeted autointerp for RAVEL eval suites: extract unique feature IDs
# from evaluation/graphs/*.json, collect→explain→score on exactly those IDs
# (wikitext-2 val+test corpus), write explanations into annotated graph clerps.
#
# Prerequisites:
#   - RAVEL paper-eval suites finished under $R/ravel_eval_suite_v3_*
#   - vLLM endpoint at ENDPOINT_FILE (scripts/slurm/launch_vllm.sbatch)
#
# Submit (wait for remaining ravel jobs):
#   sbatch --dependency=afterok:8469,8474,8475 \
#       scripts/slurm/rebuttal_autointerp_ravel_graphs.sbatch
#
# Env: ENDPOINT_FILE, LLM_MODEL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

module load arcc miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct
# Explicit interpreter — `conda activate` can leave a non-ct python on some nodes.
PYTHON="${CONDA_PREFIX:-}/bin/python"
if [[ ! -x "${PYTHON}" ]] || ! "${PYTHON}" -c "import torch" 2>/dev/null; then
    PYTHON=/home/ssuresh/.conda/envs/ct/bin/python
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "[$(date -u +%FT%TZ)] python=${PYTHON}"
"${PYTHON}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    if [[ "$(ps -o user= -p "${pid}" 2>/dev/null)" == "${USER}" ]]; then
        echo "reaping stray GPU pid ${pid}"
        kill -9 "${pid}" || true
    fi
done

R=/gscratch/ssuresh/results/paper
ENDPOINT_FILE="${ENDPOINT_FILE:-${REPO_ROOT}/results/rebuttal/vllm_endpoint.json}"
LLM_MODEL="${LLM_MODEL:-}"
OUT="${REPO_ROOT}/results/rebuttal"

run_name_graphs () {
    local label="$1" graphs_dir="$2" ckpt="$3"
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
        --model gpt2 \
        --out-dir "${out_dir}" \
        --backend openai_compat \
        --endpoint-file "${ENDPOINT_FILE}" \
        --llm-model "${LLM_MODEL}" \
        --label "${label}" \
        --include-scores-in-clerp; then
        echo "[$(date -u +%FT%TZ)] FAILED ${label} (continuing)" >&2
        return 0
    fi

    # Mirror annotated graphs next to the originals for easy UI loading.
    local mirror
    mirror="$(dirname "${graphs_dir}")/graphs_autointerp"
    mkdir -p "${mirror}"
    cp -f "${out_dir}/annotated_graphs/"*.json "${mirror}/" 2>/dev/null || true
    if [[ -f "${out_dir}/annotated_graphs/clerp_writeback_summary.json" ]]; then
        cp -f "${out_dir}/annotated_graphs/clerp_writeback_summary.json" "${mirror}/"
    fi
    echo "[$(date -u +%FT%TZ)] mirrored annotated graphs -> ${mirror}"
}

# --- dt_sweep spline ---
run_name_graphs dtsweep_spline_dt768 \
    "$R/ravel_eval_suite_v3_dtsweep_spline_dt768/runs/spline_dtsweep_gpt2_small/seed_101/evaluation/graphs" \
    "$R/paper_v3_dt_sweep_gpt2_small_spline_dt768/runs/spline_dt768_gpt2_small_pv3/seed_101/checkpoints/spline_dt768_gpt2_small_pv3_seed101_best"

run_name_graphs dtsweep_spline_dt1536 \
    "$R/ravel_eval_suite_v3_dtsweep_spline_dt1536/runs/spline_dtsweep_gpt2_small/seed_101/evaluation/graphs" \
    "$R/paper_v3_dt_sweep_gpt2_small_spline_dt1536/runs/spline_dt1536_gpt2_small_pv3/seed_101/checkpoints/spline_dt1536_gpt2_small_pv3_seed101_best"

run_name_graphs dtsweep_spline_dt3072 \
    "$R/ravel_eval_suite_v3_dtsweep_spline_dt3072/runs/spline_dtsweep_gpt2_small/seed_101/evaluation/graphs" \
    "$R/paper_v3_dt_sweep_gpt2_small_spline_dt3072/runs/spline_dt3072_gpt2_small_pv3/seed_101/checkpoints/spline_dt3072_gpt2_small_pv3_seed101_best"

run_name_graphs dtsweep_spline_dt6144 \
    "$R/ravel_eval_suite_v3_dtsweep_spline_dt6144/runs/spline_dtsweep_gpt2_small/seed_101/evaluation/graphs" \
    "$R/paper_v3_dt_sweep_gpt2_small_spline_dt6144/runs/spline_dt6144_gpt2_small_pv3/seed_101/checkpoints/spline_dt6144_gpt2_small_pv3_seed101_best"

# --- dt_sweep linear ---
run_name_graphs dtsweep_linear_dt1664 \
    "$R/ravel_eval_suite_v3_dtsweep_linear_dt1664/runs/linear_dtsweep_gpt2_small/seed_101/evaluation/graphs" \
    "$R/paper_v3_dt_sweep_gpt2_small_linear_dt1664/runs/linear_dt1664_gpt2_small_pv3/seed_101/checkpoints/linear_dt1664_gpt2_small_pv3_seed101_best"

run_name_graphs dtsweep_linear_dt3392 \
    "$R/ravel_eval_suite_v3_dtsweep_linear_dt3392/runs/linear_dtsweep_gpt2_small/seed_101/evaluation/graphs" \
    "$R/paper_v3_dt_sweep_gpt2_small_linear_dt3392/runs/linear_dt3392_gpt2_small_pv3/seed_101/checkpoints/linear_dt3392_gpt2_small_pv3_seed101_best"

run_name_graphs dtsweep_linear_dt6784 \
    "$R/ravel_eval_suite_v3_dtsweep_linear_dt6784/runs/linear_dtsweep_gpt2_small/seed_101/evaluation/graphs" \
    "$R/paper_v3_dt_sweep_gpt2_small_linear_dt6784/runs/linear_dt6784_gpt2_small_pv3/seed_101/checkpoints/linear_dt6784_gpt2_small_pv3_seed101_best"

run_name_graphs dtsweep_linear_dt13504 \
    "$R/ravel_eval_suite_v3_dtsweep_linear_dt13504/runs/linear_dtsweep_gpt2_small/seed_101/evaluation/graphs" \
    "$R/paper_v3_dt_sweep_gpt2_small_linear_dt13504/runs/linear_dt13504_gpt2_small_pv3/seed_101/checkpoints/linear_dt13504_gpt2_small_pv3_seed101_best"

# --- rebuttal_gpt2_small feature-match pair ---
run_name_graphs rebuttal_spline_fm \
    "$R/ravel_eval_suite_v3_rebuttal_fm/runs/spline_feature_match_gpt2_small/seed_101/evaluation/graphs" \
    "$R/paper_v3_rebuttal_gpt2_small_spline_fm/runs/spline_feature_match_gpt2_small_pv3/seed_101/checkpoints/spline_fm_gpt2_small_pv3_seed101_best"

run_name_graphs rebuttal_linear_fm \
    "$R/ravel_eval_suite_v3_rebuttal_fm/runs/linear_feature_match_gpt2_small/seed_101/evaluation/graphs" \
    "$R/paper_v3_rebuttal_gpt2_small_linear_fm/runs/linear_feature_match_gpt2_small_pv3/seed_101/checkpoints/linear_fm_gpt2_small_pv3_seed101_best"

# --- rebuttal_gpt2_small param-match pair ---
run_name_graphs rebuttal_spline_pm \
    "$R/ravel_eval_suite_v3_rebuttal_pm/runs/spline_param_match_gpt2_small/seed_101/evaluation/graphs" \
    "$R/paper_v3_rebuttal_gpt2_small_spline_pm/runs/spline_param_match_gpt2_small_pv3/seed_101/checkpoints/spline_pm_gpt2_small_pv3_seed101_best"

run_name_graphs rebuttal_linear_pm \
    "$R/ravel_eval_suite_v3_rebuttal_pm/runs/linear_param_match_gpt2_small/seed_101/evaluation/graphs" \
    "$R/paper_v3_rebuttal_gpt2_small_linear_pm/runs/linear_param_match_gpt2_small_pv3/seed_101/checkpoints/linear_pm_gpt2_small_pv3_seed101_best"

echo "[$(date -u +%FT%TZ)] ravel graph autointerp complete"
