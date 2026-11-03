#!/usr/bin/env bash
# Sequential Game 1 retry over a list of prompts on ONE 1-GPU node (gp-1).
# Prompts run strictly one after another (no GPU contention). Per-prompt FAILs
# are recorded and the loop continues; the job exits 0 so a dependent chained
# job still runs. Check status.retry.txt / per-prompt logs afterwards.
#
#   SLUGS="mib_gemma2_ioi_0071 mib_gemma2_mcqa_0017" CLT_TAG=gemma2-2.5M \
#     sbatch --job-name=paper_final_retry_list1 scripts/slurm/retry_game1_list.sh
#
#SBATCH --job-name=paper_final_game1_retry_list
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

CLT_TAG="${CLT_TAG:-gemma2-2.5M}"
SLUGS="${SLUGS:?set SLUG list, e.g. SLUGS=\"mib_gemma2_ioi_0071 ...\"}"
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
  # slug mib_<mibmodel>_<task>_<id> -> task (handles arc_easy-style underscores)
  rest="${SLUG#mib_${MIB_MODEL}_}"
  TASK="${rest%_[0-9]*}"
  # NOTE: three separate extractions — clean_prompt is multi-line, so a single
  # print-and-split (sed -n 'Np') truncates PROMPT and shifts TARGET/FOIL into
  # prompt lines (caused IndexError retries on 2026-10-08).
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
  echo ">>> [retry $(date -u '+%H:%M')] $CLT_TAG/$SLUG ($TASK) target=[$TARGET] foil=[$FOIL]"
  {
  echo ">>> RETRY $(date -u '+%Y-%m-%d %H:%M:%S UTC') job=${SLURM_JOB_ID:-local} game1-only, sequential 1-GPU"
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
    DONE=$((DONE + 1))
  else
    echo "FAIL $CLT_TAG/$SLUG ($TASK/$MIB_MODEL)  (see $LOG) [retry]" | tee -a "$OUTROOT/status.retry.txt"
    FAILED=$((FAILED + 1))
  fi
  } 2>&1 | tee -a "$LOG"
done

echo ">>> retry list done: OK=$DONE FAIL=$FAILED"
exit 0
