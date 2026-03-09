#!/usr/bin/env bash
#PBS -N paper-report
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=01:00:00
#PBS -o logs/pbs/report_${PBS_JOBID}.out
#PBS -e logs/pbs/report_${PBS_JOBID}.err

set -euo pipefail

cd "${PBS_O_WORKDIR}"

echo "[$(date -u +%FT%TZ)] Starting report generation"
echo "  SUITE_PATH:  ${SUITE_PATH}"
echo "  CONDA_ENV:   ${CONDA_ENV}"

conda run -n "${CONDA_ENV}" python -m spline_clt.paper_eval \
    --suite "${SUITE_PATH}" \
    --stages report \
    --worker-id 0 \
    --num-workers 1

echo "[$(date -u +%FT%TZ)] Report generation complete"
