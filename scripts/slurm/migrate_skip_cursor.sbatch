#!/usr/bin/env bash
# One-time, SINGLE-PROCESS migration to derive the item-level corpus cursor
# (corpus_skip_items) for a chunked-offline run whose checkpoint only has the
# legacy window counter (corpus_skip_sequences).
#
# Run this ONCE before resubmitting the training job. It does NOT use torchrun
# (so there is no second NCCL rank that could time out on the prefix walk) and
# never runs the model forward — it only tokenizes text to count windows.
#
# Usage:
#   sbatch scripts/slurm/migrate_skip_cursor.sbatch \
#     /path/to/<run>_training_state.pt \
#     /path/to/<suite>/resolved_config.json
#
#SBATCH -A uwyo-0002
#SBATCH -J migrate-skip-cursor
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p ai4wy-2
#SBATCH -t 03:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/migrate_skip_cursor_%j.out
#SBATCH -e logs/slurm/migrate_skip_cursor_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

STATE_PATH="${1:?usage: sbatch migrate_skip_cursor.sbatch <training_state.pt> <resolved_config.json>}"
RESOLVED_CONFIG="${2:?usage: sbatch migrate_skip_cursor.sbatch <training_state.pt> <resolved_config.json>}"

module load arcc miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct

echo "[$(date -u +%FT%TZ)] Migrating skip cursor"
echo "  state:           ${STATE_PATH}"
echo "  resolved-config: ${RESOLVED_CONFIG}"

python -m spline_clt.training.migrate_skip_cursor \
    --state "${STATE_PATH}" \
    --resolved-config "${RESOLVED_CONFIG}"

echo "[$(date -u +%FT%TZ)] Migration complete"
