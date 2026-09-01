#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J rebuttal-compute-bench
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p ai4wy-2
#SBATCH -t 08:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/rebuttal_compute_%j.out
#SBATCH -e logs/slurm/rebuttal_compute_%j.err
#SBATCH --signal=B:TERM@60

# REQ-7 + REQ-8 compute benchmarks: transcoder inference latency/memory
# (inference_bench), attribution cost vs d_t (attr_scaling), and the
# fp32-vs-bf16 inference ablation (dtype_ablation). Read-only over
# checkpoints; writes only to results/rebuttal/.
#
# Submit:  sbatch scripts/slurm/rebuttal_compute_bench.sbatch

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
SWEEP=$R/paper_v2_gpt2_small_dt_sweep/runs
V3=$R/paper_v3_gpt2_small/runs
ACT_S=/gscratch/ssuresh/shared/activations/paper_v3/gpt2_small_1b/val/gpt2
ACT_L=/gscratch/ssuresh/shared/activations/paper_v2/gpt2_large/val/gpt2-large
ACT_Q=/gscratch/ssuresh/shared/activations/paper_v2/qwen3_06b/val/Qwen_Qwen3-0.6B
OUT="${REPO_ROOT}/results/rebuttal"

SPLINE_SWEEP=(
    "$SWEEP/spline_dt768_gpt2_small_pv2/seed_101/checkpoints/spline_dt768_gpt2_small_pv2_seed101_best"
    "$SWEEP/spline_dt1536_gpt2_small_pv2/seed_101/checkpoints/spline_dt1536_gpt2_small_pv2_seed101_best"
    "$SWEEP/spline_dt3072_gpt2_small_pv2/seed_101/checkpoints/spline_dt3072_gpt2_small_pv2_seed101_best"
    "$SWEEP/spline_dt5568_gpt2_small_pv2/seed_101/checkpoints/spline_dt5568_gpt2_small_pv2_seed101_best"
    "$SWEEP/spline_dt6144_gpt2_small_pv2/seed_101/checkpoints/spline_dt6144_gpt2_small_pv2_seed101_best"
    "$SWEEP/spline_dt12288_gpt2_small_pv2/seed_101/checkpoints/spline_dt12288_gpt2_small_pv2_seed101_best"
)
V3_MAIN=(
    "$V3/spline_feature_match_gpt2_small_pv3/seed_101/checkpoints/spline_fm_gpt2_small_pv3_seed101_best"
    "$V3/spline_param_match_gpt2_small_pv3/seed_101/checkpoints/spline_pm_gpt2_small_pv3_seed101_best"
    "$V3/linear_feature_match_gpt2_small_pv3/seed_101/checkpoints/linear_fm_gpt2_small_pv3_seed101_best"
    "$V3/linear_param_match_gpt2_small_pv3/seed_101/checkpoints/linear_pm_gpt2_small_pv3_seed101_best"
)

echo "[$(date -u +%FT%TZ)] inference_bench gpt2-small (dt sweep + v3 variants)"
python -m rebuttal_eval.inference_bench \
    --checkpoints "${SPLINE_SWEEP[@]}" "${V3_MAIN[@]}" \
    --activation-dir "${ACT_S}" \
    --out-dir "${OUT}/inference_bench" --label gpt2_small

echo "[$(date -u +%FT%TZ)] inference_bench gpt2-large"
python -m rebuttal_eval.inference_bench \
    --checkpoints \
    "$R/paper_v2_gpt2_large/runs/spline_feature_match_gpt2_large_pv2/seed_101/checkpoints/spline_fm_gpt2_large_pv2_seed101_best" \
    "$R/paper_v2_gpt2_large/runs/linear_feature_match_gpt2_large_pv2/seed_101/checkpoints/linear_fm_gpt2_large_pv2_seed101_best" \
    --activation-dir "${ACT_L}" \
    --out-dir "${OUT}/inference_bench" --label gpt2_large

echo "[$(date -u +%FT%TZ)] inference_bench qwen3-0.6b"
python -m rebuttal_eval.inference_bench \
    --checkpoints \
    "$R/paper_v2_qwen3_06b/runs/spline_feature_match_qwen3_06b_pv2/seed_101/checkpoints/spline_fm_qwen3_06b_pv2_seed101_best" \
    "$R/paper_v2_qwen3_06b/runs/linear_feature_match_qwen3_06b_pv2/seed_101/checkpoints/linear_fm_qwen3_06b_pv2_seed101_best" \
    --activation-dir "${ACT_Q}" \
    --out-dir "${OUT}/inference_bench" --label qwen3_06b

echo "[$(date -u +%FT%TZ)] dtype_ablation (v3 spline FM + linear FM)"
python -m rebuttal_eval.dtype_ablation \
    --checkpoint "$V3/spline_feature_match_gpt2_small_pv3/seed_101/checkpoints/spline_fm_gpt2_small_pv3_seed101_best" \
    --activation-dir "${ACT_S}" \
    --out-dir "${OUT}/dtype_ablation" --label spline_fm_v3
python -m rebuttal_eval.dtype_ablation \
    --checkpoint "$V3/linear_feature_match_gpt2_small_pv3/seed_101/checkpoints/linear_fm_gpt2_small_pv3_seed101_best" \
    --activation-dir "${ACT_S}" \
    --out-dir "${OUT}/dtype_ablation" --label linear_fm_v3

echo "[$(date -u +%FT%TZ)] attr_scaling (dt sweep + v3 variants, 5 prompts)"
python -m rebuttal_eval.attr_scaling \
    --checkpoints "${SPLINE_SWEEP[@]}" "${V3_MAIN[@]}" \
    --model gpt2 --n-prompts 5 \
    --out-dir "${OUT}/attr_scaling" --label gpt2_small

echo "[$(date -u +%FT%TZ)] compute benchmarks complete"
