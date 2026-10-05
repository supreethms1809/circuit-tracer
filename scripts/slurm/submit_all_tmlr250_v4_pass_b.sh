#!/usr/bin/env bash
# TMLR-250 v4 Pass B — deferred eap_syed + ported ACDC, merged into Pass A files.
#
# For every completed Pass A run dir lacking these methods: run
# run_baselines --methods eap_syed,acdc on the stored graph + oracle kwargs
# (REPORT_BUDGET=128 curve length), merge back with merge_baselines
# (--adopt-extra-identity), and refresh embedded KL blocks. Same pattern as
# scripts/run_macag_shapley_pass.sh.
#
# Uses REPORT_BUDGET=128 for curve length; do NOT interpret ACDC as budget-8
# matched unless ACDC_TARGET_K is set explicitly (see run_todo_v4.md §2).
#
#   sbatch scripts/slurm/submit_all_tmlr250_v4_pass_b.sh
#   RANK=0 WORLD=1 ROOTS=/path/to/macag_mib_seed0 bash scripts/slurm/submit_all_tmlr250_v4_pass_b.sh  # manual
#SBATCH --job-name=macag_v4_passb
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

# --- Pass B knobs ---
ROOTS="${ROOTS:-/gscratch/${USER}/macag_mib_tmlr250v4_h200/macag_mib_seed0 /gscratch/${USER}/macag_mib_tmlr250v4_gemma25m/macag_mib_seed0 /gscratch/${USER}/macag_mib_tmlr250v4_llama/macag_mib_seed0}"
SCORE_KIND="${SCORE_KIND:-logit_gap}"
METHODS="${METHODS:-eap_syed,acdc}"
BUDGET="${BUDGET:-${REPORT_BUDGET:-128}}"
ALPHA="${ALPHA:-0.5}"
LAM="${LAM:-0.02}"
ACDC_TAUS="${ACDC_TAUS:-0.001,0.01,0.05,0.1,0.2,0.5}"
ACDC_TARGET_K="${ACDC_TARGET_K:-}"   # empty = native tau sweep (v4 default); -1 = match --budget
KL_RESCORE="${KL_RESCORE:-1}"
CONNECTED="${CONNECTED:-0}"
CAP_SUFFICIENCY="${CAP_SUFFICIENCY:-1}"
CAP_NECESSITY="${CAP_NECESSITY:-1}"

echo ">>> v4 Pass B | rank=$RANK/$WORLD methods=$METHODS budget=$BUDGET acdc_target_k=${ACDC_TARGET_K:-<native>}"
nvidia-smi -L || true

CELL=0
for ROOT in $ROOTS; do
  [[ -d "$ROOT" ]] || { echo ">>> no root $ROOT; skipping"; continue; }
  STATUS="$ROOT/status.passb.w${RANK}.txt"; : > "$STATUS"
  for clt_dir in "$ROOT"/*/; do
    tag=$(basename "$clt_dir")
    [[ "$tag" == .* ]] && continue
    for run_dir in "$clt_dir"*/; do
      [[ -d "$run_dir" ]] || continue
      slug=$(basename "$run_dir")
      kind_dir="$run_dir/$SCORE_KIND"
      [[ -d "$kind_dir" ]] || continue
      main_json="$kind_dir/macag_baselines.json"
      kwargs="$kind_dir/oracle_kwargs.json"
      graph="$run_dir/graphs/$slug.json"
      [[ -f "$main_json" && -f "$kwargs" && -f "$graph" ]] || continue

      cell=$CELL; CELL=$((CELL + 1))
      (( cell % WORLD == RANK )) || continue

      if python - "$main_json" <<'PY'
import json, sys
methods = (json.load(open(sys.argv[1])).get("methods") or {})
syed = methods.get("eap_syed") or {}
acdc = methods.get("acdc") or {}
ok = bool(syed.get("results")) and bool(acdc.get("best_by_size") or acdc.get("matched_k"))
sys.exit(0 if ok else 1)
PY
      then
        echo "SKIP-done $tag/$slug" | tee -a "$STATUS"; continue
      fi

      echo ">>> [$tag/$slug] pass B ($METHODS)"
      cand_file="$kind_dir/candidates_from_main.json"
      python - "$main_json" "$cand_file" <<'PY'
import json, sys
main = json.load(open(sys.argv[1]))
json.dump(main.get("candidates") or [], open(sys.argv[2], "w"))
PY
      sidecar="$kind_dir/macag_baselines_passb.json"
      bl_cmd=(
        python -m macag.cli.run_baselines
        --graph-json "$graph" --target y --input-id "$slug"
        --oracle-factory macag.factories.replacement_model:create_replacement_model_oracle
        --oracle-kwargs-file "$kwargs"
        --candidates-file "$cand_file"
        --budget "$BUDGET" --alpha "$ALPHA" --lam "$LAM"
        --methods "$METHODS"
        --acdc-taus "$ACDC_TAUS"
        --no-progress
        --output-json "$sidecar"
      )
      [[ -n "$ACDC_TARGET_K" ]] && bl_cmd+=(--acdc-target-k "$ACDC_TARGET_K")
      [[ "$CONNECTED" != "1" && "$CONNECTED" != "true" ]] && bl_cmd+=(--no-connected)
      [[ "$CAP_SUFFICIENCY" == "1" ]] || bl_cmd+=(--no-cap-sufficiency)
      [[ "$CAP_NECESSITY" == "1" ]] || bl_cmd+=(--no-cap-necessity)
      if "${bl_cmd[@]}" \
        && python -m macag.cli.merge_baselines \
          --main "$main_json" --extra "$sidecar" --adopt-extra-identity; then
        if [[ "$KL_RESCORE" == "1" ]]; then
          python -m macag.cli.rescore_kl --run-dir "$kind_dir" --force >/dev/null \
            || echo ">>> (KL refresh failed for $tag/$slug; merge itself succeeded)"
        fi
        echo "OK   $tag/$slug" | tee -a "$STATUS"
      else
        echo "FAIL $tag/$slug" | tee -a "$STATUS"
      fi
    done
  done
done

echo ">>> Pass B rank $RANK done."
