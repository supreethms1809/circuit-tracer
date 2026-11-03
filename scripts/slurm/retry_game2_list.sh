#!/usr/bin/env bash
# Sequential Game 2 retry over a list of prompts on ONE 1-GPU node (gp-1).
# Reuses graph + game1 (--skip-attribute --skip-game1). Per-prompt FAILs are
# logged; the job exits 0 so dependent chained jobs still run.
#
#   SLUGS="mib_llama3_arc_easy_0317 mib_llama3_arc_easy_0338" CLT_TAG=llama32-524k \
#     sbatch --job-name=paper_final_game2_retry_list scripts/slurm/retry_game2_list.sh
#
#SBATCH --job-name=paper_final_game2_retry_list
#SBATCH -A uwyo-0002
#SBATCH --partition=gp-1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

export GAME2_BETAS="${GAME2_BETAS:-0 0.2}"
export GAME2_SCORE_KIND="${GAME2_SCORE_KIND:-negative_loss}"
export GAME2_FREEZE_ATTENTION="${GAME2_FREEZE_ATTENTION:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

CLT_TAG="${CLT_TAG:-llama32-524k}"
SLUGS="${SLUGS:?set SLUG list}"
case "$CLT_TAG" in
  gemma2-426k)  MODEL="google/gemma-2-2b"; TSET="mntss/clt-gemma-2-2b-426k"; MIB_MODEL="gemma2";;
  gemma2-2.5M)  MODEL="google/gemma-2-2b"; TSET="mntss/clt-gemma-2-2b-2.5M";  MIB_MODEL="gemma2";;
  llama32-524k) MODEL="meta-llama/Llama-3.2-1B"; TSET="mntss/clt-llama-3.2-1b-524k"; MIB_MODEL="llama3";;
  *) echo "ERROR: unknown CLT_TAG=$CLT_TAG" >&2; exit 2;;
esac

JSON="${JSON:-macag/data/mib_benchmark_prompts.json}"
OUTROOT="${OUTROOT:-/gscratch/${USER}/paper_final/macag}"
FAILED=0
DONE=0

for SLUG in $SLUGS; do
  rest="${SLUG#mib_${MIB_MODEL}_}"
  TASK="${rest%_[0-9]*}"
  PROMPT="$(JSON="$JSON" python - "$TASK" "$SLUG" <<'PY'
import json, os, sys
d = json.load(open(os.environ["JSON"]))
task, slug = sys.argv[1], sys.argv[2]
for it in d["tasks"].get(task, []):
    if it.get("id") == slug:
        sys.stdout.write(it.get("clean_prompt", ""))
        break
PY
)"
  TARGET="$(JSON="$JSON" python - "$TASK" "$SLUG" <<'PY'
import json, os, sys
d = json.load(open(os.environ["JSON"]))
task, slug = sys.argv[1], sys.argv[2]
for it in d["tasks"].get(task, []):
    if it.get("id") == slug:
        sys.stdout.write(it.get("correct_token", ""))
        break
PY
)"
  FOIL="$(JSON="$JSON" python - "$TASK" "$SLUG" <<'PY'
import json, os, sys
d = json.load(open(os.environ["JSON"]))
task, slug = sys.argv[1], sys.argv[2]
for it in d["tasks"].get(task, []):
    if it.get("id") == slug:
        sys.stdout.write(it.get("incorrect_token", ""))
        break
PY
)"
  if [[ -z "$PROMPT" || -z "$TARGET" || -z "$FOIL" ]]; then
    echo "FAIL $CLT_TAG/$SLUG lookup failed (task=$TASK)" | tee -a "$OUTROOT/status.retry.txt"
    FAILED=$((FAILED + 1)); continue
  fi
  OUTDIR="$OUTROOT/$CLT_TAG/$SLUG"
  LOG="$OUTROOT/$CLT_TAG-$SLUG.log"
  echo ">>> [retry game2 $(date -u '+%H:%M')] $CLT_TAG/$SLUG ($TASK)"
  {
  echo ">>> RETRY game2 $(date -u '+%Y-%m-%d %H:%M:%S UTC') job=${SLURM_JOB_ID:-local} sequential 1-GPU"
  if scripts/run_macag_pipeline.sh \
    --prompt "$PROMPT" --target "$TARGET" --foil "$FOIL" \
    --model "$MODEL" --transcoder-set "$TSET" \
    --slug "$SLUG" --outdir "$OUTDIR" --device cuda \
    --freeze-mode both --solvers "abr" \
    --score-kinds "logit_gap" \
    --report-budget 128 \
    --node-threshold 0.9 --edge-threshold 1.0 \
    --max-feature-nodes 1000000 \
    --skip-attribute --skip-game1 --skip-baselines \
    --abr-iters 8 --no-connected; then
    echo "OK   $CLT_TAG/$SLUG ($TASK/$MIB_MODEL) [retry game2]" | tee -a "$OUTROOT/status.retry.txt"
    DONE=$((DONE + 1))
  else
    echo "FAIL $CLT_TAG/$SLUG ($TASK/$MIB_MODEL)  (see $LOG) [retry game2]" | tee -a "$OUTROOT/status.retry.txt"
    FAILED=$((FAILED + 1))
  fi
  } 2>&1 | tee -a "$LOG"
done

echo ">>> retry game2 list done: OK=$DONE FAIL=$FAILED"
exit 0
