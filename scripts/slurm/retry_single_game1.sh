#!/usr/bin/env bash
# One-prompt Game 1 retry on a dedicated 1-GPU node.
# Defaults to the OOM-at-load 426k prompt (mib_gemma2_mcqa_0044); override via env.
# Reuses the existing graph (--skip-attribute) and runs the game1 stage only,
# with the same protocol as PAPER_STAGE=game1 in submit_macag_final.sh.
#
#   sbatch scripts/slurm/retry_single_game1.sh
#   SLUG=mib_gemma2_ioi_0132 CLT_TAG=gemma2-2.5M sbatch scripts/slurm/retry_single_game1.sh
#
#SBATCH --job-name=paper_final_game1_retry
#SBATCH -A uwyo-0002
#SBATCH --partition=gp-1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

CLT_TAG="${CLT_TAG:-gemma2-426k}"
SLUG="${SLUG:-mib_gemma2_mcqa_0044}"
case "$CLT_TAG" in
  gemma2-426k)  MODEL="google/gemma-2-2b"; TSET="mntss/clt-gemma-2-2b-426k"; MIB_MODEL="gemma2"; TASK="mcqa";;
  gemma2-2.5M)  MODEL="google/gemma-2-2b"; TSET="mntss/clt-gemma-2-2b-2.5M";  MIB_MODEL="gemma2"; TASK="mcqa";;
  llama32-524k) MODEL="meta-llama/Llama-3.2-1B"; TSET="mntss/clt-llama-3.2-1b-524k"; MIB_MODEL="llama3"; TASK="${TASK:-mcqa}";;
  *) echo "ERROR: unknown CLT_TAG=$CLT_TAG" >&2; exit 2;;
esac

OUTROOT="${OUTROOT:-/gscratch/${USER}/paper_final/macag}"
OUTDIR="$OUTROOT/$CLT_TAG/$SLUG"
PROMPT="${PROMPT:-$(JSON="${JSON:-macag/data/mib_benchmark_prompts.json}" python - "$TASK" "$SLUG" <<'PY'
import json, os, sys
d = json.load(open(os.environ.get("JSON", "macag/data/mib_benchmark_prompts.json")))
task, slug = sys.argv[1], sys.argv[2]
for it in d["tasks"][task]:
    if it.get("id") == slug:
        sys.stdout.write(it.get("clean_prompt", ""))
        break
PY
)}"
TARGET="${TARGET:-$(JSON="${JSON:-macag/data/mib_benchmark_prompts.json}" python - "$TASK" "$SLUG" <<'PY'
import json, os, sys
d = json.load(open(os.environ.get("JSON", "macag/data/mib_benchmark_prompts.json")))
task, slug = sys.argv[1], sys.argv[2]
for it in d["tasks"][task]:
    if it.get("id") == slug:
        sys.stdout.write(it.get("correct_token", ""))
        break
PY
)}"
FOIL="${FOIL:-$(JSON="${JSON:-macag/data/mib_benchmark_prompts.json}" python - "$TASK" "$SLUG" <<'PY'
import json, os, sys
d = json.load(open(os.environ.get("JSON", "macag/data/mib_benchmark_prompts.json")))
task, slug = sys.argv[1], sys.argv[2]
for it in d["tasks"][task]:
    if it.get("id") == slug:
        sys.stdout.write(it.get("incorrect_token", ""))
        break
PY
)}"

echo ">>> retry game1 | $CLT_TAG/$SLUG target=[$TARGET] foil=[$FOIL] -> $OUTDIR"
echo ">>> prompt: $PROMPT"

LOG="$OUTROOT/$CLT_TAG-$SLUG.log"
{
echo ">>> RETRY $(date -u '+%Y-%m-%d %H:%M:%S UTC') job=${SLURM_JOB_ID:-local} game1-only, 1 GPU"
if scripts/run_macag_pipeline.sh \
  --prompt "$PROMPT" --target "$TARGET" --foil "$FOIL" \
  --model "$MODEL" --transcoder-set "$TSET" \
  --slug "$SLUG" --outdir "$OUTDIR" --device cuda \
  --freeze-mode both --solvers "abr" \
  --score-kinds "logit_gap" \
  --report-budget 128 \
  --node-threshold 0.9 --edge-threshold 1.0 \
  --max-feature-nodes 1000000 \
  --skip-attribute --skip-game2 --skip-baselines \
  --abr-iters 8 --no-connected; then
  echo "OK   $CLT_TAG/$SLUG ($TASK/$MIB_MODEL) [retry]" | tee -a "$OUTROOT/status.retry.txt"
else
  echo "FAIL $CLT_TAG/$SLUG ($TASK/$MIB_MODEL)  (see $LOG) [retry]" | tee -a "$OUTROOT/status.retry.txt"
  exit 1
fi
} 2>&1 | tee -a "$LOG"
