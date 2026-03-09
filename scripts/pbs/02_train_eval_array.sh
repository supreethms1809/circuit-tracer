#!/usr/bin/env bash
#PBS -N paper-train-eval
#PBS -J 0-31
#PBS -l select=1:ncpus=8:ngpus=1:mem=48gb
#PBS -l walltime=24:00:00
#PBS -o logs/pbs/train_eval_${PBS_ARRAY_ID}_${PBS_ARRAY_INDEX}.out
#PBS -e logs/pbs/train_eval_${PBS_ARRAY_ID}_${PBS_ARRAY_INDEX}.err

# NOTE: Override -J at submit time via:
#   qsub -J 0-${ACTUAL_WORKERS-1} scripts/pbs/02_train_eval_array.sh
# PBS Pro automatically sets CUDA_VISIBLE_DEVICES based on the allocated GPU.

set -euo pipefail

cd "${PBS_O_WORKDIR}"

echo "[$(date -u +%FT%TZ)] Starting train+eval worker ${PBS_ARRAY_INDEX}/${NUM_WORKERS}"
echo "  SUITE_PATH:          ${SUITE_PATH}"
echo "  CONDA_ENV:           ${CONDA_ENV}"
echo "  PBS_ARRAY_INDEX:     ${PBS_ARRAY_INDEX}"
echo "  NUM_WORKERS:         ${NUM_WORKERS}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

conda run -n "${CONDA_ENV}" python -m spline_clt.paper_eval \
    --suite "${SUITE_PATH}" \
    --stages train evaluate macag \
    --worker-id "${PBS_ARRAY_INDEX}" \
    --num-workers "${NUM_WORKERS}"

echo "[$(date -u +%FT%TZ)] Worker ${PBS_ARRAY_INDEX} complete"
