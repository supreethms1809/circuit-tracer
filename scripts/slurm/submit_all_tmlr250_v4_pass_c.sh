#!/usr/bin/env bash
# TMLR-250 v4 Pass C — original edge EAP/ACDC (Track B) sidecars.
#
# For every completed Pass A run dir lacking macag_original_baselines.json:
# run run_original_baselines (native output-edge AtP + corrupt-patch tau
# ACDC) into a sidecar. Never merges into feature macag_baselines.json
# (different ID universe; no Jaccard). Report F/S/N and KL for edge circuits;
# no Jaccard vs Game 1 (see run_todo_v4.md §3).
#
#   sbatch scripts/slurm/submit_all_tmlr250_v4_pass_c.sh
#   RANK=0 WORLD=1 ROOTS=/path/to/macag_mib_seed0 bash scripts/slurm/submit_all_tmlr250_v4_pass_c.sh  # manual
#SBATCH --job-name=macag_v4_passc
#SBATCH -A uwyo-0002
#SBATCH --partition=gp-2
#SBATCH --array=0-7
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
mkdir -p logs/slurm

module load miniconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate ct 2>/dev/null || true
export PATH="${CONDA_PREFIX:-/opt/conda}/bin:${PATH}"
hash -r
export PYTHONPATH="${PYTHONPATH:-.}:."

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  RANK="$SLURM_ARRAY_TASK_ID"
  WORLD="${SLURM_ARRAY_TASK_COUNT:-1}"
else
  RANK="${RANK:-0}"
  WORLD="${WORLD:-1}"
fi

# --- Pass C knobs (native edge circuits; budget 0 = no budget matching) ---
ROOTS="${ROOTS:-/gscratch/${USER}/macag_mib_tmlr250v4_h200/macag_mib_seed0 /gscratch/${USER}/macag_mib_tmlr250v4_gemma25m/macag_mib_seed0 /gscratch/${USER}/macag_mib_tmlr250v4_llama/macag_mib_seed0}"
SCORE_KIND="${SCORE_KIND:-logit_gap}"
METHODS="${METHODS:-eap_edge,acdc_edge}"
EDGE_BUDGET="${EDGE_BUDGET:-0}"
ACDC_EDGE_TAUS="${ACDC_EDGE_TAUS:-default}"
ACDC_EDGE_METRIC="${ACDC_EDGE_METRIC:-kl}"
ACDC_EDGE_TARGET_K="${ACDC_EDGE_TARGET_K:-0}"

echo ">>> v4 Pass C | rank=$RANK/$WORLD methods=$METHODS"
nvidia-smi -L || true

CELL=0
for ROOT in $ROOTS; do
  [[ -d "$ROOT" ]] || { echo ">>> no root $ROOT; skipping"; continue; }
  STATUS="$ROOT/status.passc.w${RANK}.txt"; : > "$STATUS"
  for clt_dir in "$ROOT"/*/; do
    tag=$(basename "$clt_dir")
    [[ "$tag" == .* ]] && continue
    for run_dir in "$clt_dir"*/; do
      [[ -d "$run_dir" ]] || continue
      slug=$(basename "$run_dir")
      kind_dir="$run_dir/$SCORE_KIND"
      [[ -d "$kind_dir" ]] || continue
      kwargs="$kind_dir/oracle_kwargs.json"
      out="$kind_dir/macag_original_baselines.json"
      [[ -f "$kwargs" ]] || continue

      cell=$CELL; CELL=$((CELL + 1))
      (( cell % WORLD == RANK )) || continue

      if [[ -f "$out" ]]; then
        echo "SKIP-done $tag/$slug" | tee -a "$STATUS"; continue
      fi

      echo ">>> [$tag/$slug] original $METHODS -> $out"
      if python -m macag.cli.run_original_baselines \
        --oracle-kwargs-file "$kwargs" \
        --input-id "$slug" \
        --budget "$EDGE_BUDGET" \
        --methods "$METHODS" \
        --acdc-taus "$ACDC_EDGE_TAUS" \
        --acdc-metric "$ACDC_EDGE_METRIC" \
        --acdc-target-k "$ACDC_EDGE_TARGET_K" \
        --output-json "$out"; then
        echo "OK   $tag/$slug" | tee -a "$STATUS"
      else
        echo "FAIL $tag/$slug" | tee -a "$STATUS"
      fi
    done
  done
done

echo ">>> Pass C rank $RANK done."
