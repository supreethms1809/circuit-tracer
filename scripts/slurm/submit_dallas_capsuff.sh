#!/usr/bin/env bash
#SBATCH --job-name=dal-capsuff
#SBATCH -A uwyo-0002
#SBATCH --partition=gp-2
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#
# Dallas–Austin only. Unpruned graph (node_threshold=1.0) already on gscratch.
# Game 1: logit-gap, alpha=0.5, no prefilter, dual-freeze, sufficiency capped
# at the clean score. Then frozen influence, feature AtP, and matched-k ACDC
# under that same v. Does not overwrite the earlier prefilter / KL artifacts.
#
#   sbatch scripts/slurm/submit_dallas_capsuff.sh
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

export PYTHONPATH="${PYTHONPATH:-.}:."
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"

export SLUG="${SLUG:-dallas-austin}"
export CLT_TAG="${CLT_TAG:-llama32-524k}"
export OUTDIR="${OUTDIR:-/gscratch/${USER}/macag_dallas_austin_llama/${CLT_TAG}/${SLUG}}"
export SCORE_KIND=logit_gap
export ALPHA=0.5
export LAM=0.02
export EPS=0.1
export PREFILTER_TOP_K=off
export FREEZE_MODE=both
export CONNECTED=0
export CAP_SUFFICIENCY=1
export OUTPUT_TAG=capsuff

LOG="$OUTDIR/logs/ablation.g1_capsuff_full.out"
mkdir -p "$OUTDIR/logs"
exec > >(tee -a "$LOG") 2>&1

echo ">>> job=${SLURM_JOB_ID:-none} node=${SLURMD_NODENAME:-none}"
echo ">>> Dallas capped-sufficiency rerun (one prompt, existing 1.0 graph)"
nvidia-smi -L || true

scripts/run_macag_dallas_game1_prefilter.sh

G1="$OUTDIR/logit_gap/macag_game1_capsuff.json"
[[ -f "$G1" ]] || { echo "ERROR: missing $G1" >&2; exit 2; }

K_FROZEN=$(python - "$G1" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
print(len(payload["frozen"]["evidence"]["E_star"]))
PY
)
echo ">>> frozen |E*|=$K_FROZEN"
if [[ "$K_FROZEN" -lt 1 ]]; then
  echo "ERROR: frozen evidence is empty; not matching baselines to k=0" >&2
  exit 2
fi

# Prefix curves must cover Game 1's natural size.
if [[ "$K_FROZEN" -gt 32 ]]; then
  export REPORT_BUDGET="$K_FROZEN"
else
  export REPORT_BUDGET=32
fi
export ACDC_TARGET_K="$K_FROZEN"
export ACDC_TAUS="${ACDC_TAUS:-0.001,0.01,0.05,0.1,0.2,0.5}"

for method in influence eap_syed acdc; do
  echo ">>> baseline $method budget=$REPORT_BUDGET acdc_target_k=$ACDC_TARGET_K"
  METHOD="$method" scripts/run_macag_dallas_llama_method.sh
done

echo ">>> capped Dallas rerun complete"
