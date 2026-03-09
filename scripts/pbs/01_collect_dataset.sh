#!/usr/bin/env bash
#PBS -N paper-collect
#PBS -l select=1:ncpus=16:ngpus=1:mem=64gb
#PBS -l walltime=02:00:00
#PBS -o logs/pbs/collect_${PBS_JOBID}.out
#PBS -e logs/pbs/collect_${PBS_JOBID}.err

set -euo pipefail

cd "${PBS_O_WORKDIR}"

echo "[$(date -u +%FT%TZ)] Starting dataset collection"
echo "  SUITE_PATH:  ${SUITE_PATH}"
echo "  CONDA_ENV:   ${CONDA_ENV}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

conda run -n "${CONDA_ENV}" python -m spline_clt.paper_eval \
    --suite "${SUITE_PATH}" \
    --stages collect \
    --worker-id 0 \
    --num-workers 1

echo "[$(date -u +%FT%TZ)] Dataset collection complete"
