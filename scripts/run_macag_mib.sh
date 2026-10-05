#!/usr/bin/env bash
# Run the full MACAG pipeline over MIB-bench prompts (macag/data/mib_benchmark_prompts.json).
# Reuses the same game outputs and analyzers as the ACDC sweep; only the prompt source
# and per-model CLT routing differ (gemma2 MIB prompts -> gemma2 CLTs, llama3 MIB
# prompts -> the llama32-524k CLT; llama3 here means Llama-3.2-1B, not the MIB
# leaderboard's Llama-3.1-8B -- see build_mib_benchmark_prompts.py note).
#
# Build the prompt JSON first (once, or when changing split/limit):
#   conda run -n ct python experiments/build_mib_benchmark_prompts.py \
#     --models gemma2 --tasks ioi mcqa arc_easy --split validation --limit-per-task 10
#   conda run -n ct python experiments/build_mib_benchmark_prompts.py \
#     --models llama3 --tasks ioi mcqa arithmetic_addition arithmetic_subtraction \
#         arc_easy arc_challenge --split validation --limit-per-task 10
#
# Usage:
#   scripts/run_macag_mib.sh
#   LIMIT=2 scripts/run_macag_mib.sh                    # smoke
#   TASKS="ioi" MIB_MODELS="gemma2" scripts/run_macag_mib.sh
#   MIB_MODELS="llama3" scripts/run_macag_mib.sh        # llama3 only
#
# Output: results/macag_mib/<clt_tag>/<slug>/ + summary CSVs from existing analyzers.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/macag_parallel_common.sh
source scripts/macag_parallel_common.sh

JSON="${JSON:-macag/data/mib_benchmark_prompts.json}"
OUTROOT="${OUTROOT:-results/macag_mib}"
DEVICE="${DEVICE:-cuda}"
SOLVERS="${SOLVERS-abr fp}"
FREEZE_MODE="${FREEZE_MODE:-both}"
STOP_METRIC="${STOP_METRIC:-raw_relative}"
SCORE_KINDS="${SCORE_KINDS:-logit_gap}"
CONNECTED="${CONNECTED:-0}"
PREFILTER_TOP_K="${PREFILTER_TOP_K:-}"
GAME2_PREFILTER_TOP_K="${GAME2_PREFILTER_TOP_K:-}"
TASKS="${TASKS:-}"
LIMIT="${LIMIT:-0}"
CLTS="${CLTS:-}"
MIB_MODELS="${MIB_MODELS:-gemma2}"
SKIP_TASKS="${SKIP_TASKS:-}"
SKIP_GAME1="${SKIP_GAME1:-0}"
SKIP_BASELINES="${SKIP_BASELINES:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"
WORKER_ID="${WORKER_ID:-0}"
ANALYZE_ONLY="${ANALYZE_ONLY:-0}"
SHARD_TAG="${SHARD_TAG:-}"

# clt_tag  model  transcoder_set  mib_models (space-separated)
CLT_TAGS=(gemma2-426k gemma2-2.5M llama32-524k)
CLT_MODEL=(google/gemma-2-2b google/gemma-2-2b meta-llama/Llama-3.2-1B)
CLT_TSET=(mntss/clt-gemma-2-2b-426k mntss/clt-gemma-2-2b-2.5M mntss/clt-llama-3.2-1b-524k)
CLT_MIB_MODELS=(gemma2 gemma2 llama3)

if [[ ! -f "$JSON" ]]; then
  echo "ERROR: $JSON not found. Build it with:" >&2
  echo "  conda run -n ct python experiments/build_mib_benchmark_prompts.py --models gemma2" >&2
  exit 2
fi

mkdir -p "$OUTROOT"
STATUS="$OUTROOT/status.${SHARD_TAG:+${SHARD_TAG}.}w${WORKER_ID}.txt"; : > "$STATUS"
echo ">>> MACAG MIB sweep | json=$JSON tasks=${TASKS:-ALL} mib_models=$MIB_MODELS limit=$LIMIT"
echo ">>> game1 freeze_mode=$FREEZE_MODE | out=$OUTROOT | worker=$WORKER_ID/$NUM_WORKERS"

emit_prompts() {
  JSON="$JSON" TASKS="$TASKS" LIMIT="$LIMIT" SKIP_TASKS="$SKIP_TASKS" MIB_MODELS="$MIB_MODELS" python - <<'PY'
import json, os
d = json.load(open(os.environ["JSON"]))
want_tasks = set(os.environ.get("TASKS", "").split()) or None
skip_tasks = set(os.environ.get("SKIP_TASKS", "").split())
want_models = set(os.environ.get("MIB_MODELS", "").split()) or None
limit = int(os.environ.get("LIMIT", "0"))
for task, items in d["tasks"].items():
    if (want_tasks and task not in want_tasks) or task in skip_tasks:
        continue
    for i, it in enumerate(items):
        if want_models and it.get("mib_model") not in want_models:
            continue
        if limit and i >= limit:
            break
        print(f"{task}\t{i}\t{it.get('mib_model', '')}")
PY
}

get_field() {
  JSON="$JSON" python - "$1" "$2" "$3" <<'PY'
import json, os, sys
d = json.load(open(os.environ["JSON"]))
task, idx, field = sys.argv[1], int(sys.argv[2]), sys.argv[3]
val = d["tasks"][task][idx].get(field, "")
if val is None:
    val = ""
sys.stdout.write(str(val))
PY
}

clt_matches_mib_model() {
  local tag="$1" mib_model="$2"
  local ci mib_list
  for ci in "${!CLT_TAGS[@]}"; do
    [[ "${CLT_TAGS[$ci]}" == "$tag" ]] || continue
    mib_list="${CLT_MIB_MODELS[$ci]}"
    [[ -z "$mib_list" ]] && return 1
    [[ " $mib_list " == *" $mib_model "* ]] && return 0
    return 1
  done
  return 1
}

CELL=0
if [[ "$ANALYZE_ONLY" != "1" ]]; then
for ci in "${!CLT_TAGS[@]}"; do
  tag="${CLT_TAGS[$ci]}"; model="${CLT_MODEL[$ci]}"; tset="${CLT_TSET[$ci]}"
  if [[ -n "$CLTS" && " $CLTS " != *" $tag "* ]]; then continue; fi
  echo ""; echo ">>> ===== CLT $tag ($model | $tset) ====="
  while IFS=$'\t' read -r task idx mib_model; do
    [[ -z "$task" ]] && continue
    if ! clt_matches_mib_model "$tag" "$mib_model"; then
      continue
    fi
    cell=$CELL; CELL=$((CELL + 1))
    (( cell % NUM_WORKERS == WORKER_ID )) || continue
    slug=$(get_field "$task" "$idx" id)
    prompt=$(get_field "$task" "$idx" clean_prompt)
    target=$(get_field "$task" "$idx" correct_token)
    foil=$(get_field "$task" "$idx" incorrect_token)
    corrupted=$(get_field "$task" "$idx" corrupted_prompt 2>/dev/null || true)
    out="$OUTROOT/$tag/$slug"
    echo ">>> [$tag/$slug] ($task/$mib_model) target=$target foil=$foil"
    mkdir -p "$out"
    # Resume: require every SCORE_KIND subdirectory to have finished game1,
    # each configured Game2 solver, and baselines (unless skipped).
    # Artifact-based: completed stages inside run_macag_pipeline.sh are also
    # skipped individually when their JSON already exists.
    g1_done=1; g2_done=1; bl_done=1
    for kind in $SCORE_KINDS; do
      [[ "$SKIP_GAME1" == "1" || -f "$out/$kind/macag_game1.json" ]] || g1_done=0
      if [[ -n "${SOLVERS// }" ]]; then
        for solver in $SOLVERS; do
          [[ -f "$out/$kind/macag_game2_${solver}.json" ]] || g2_done=0
        done
      fi
      [[ "$SKIP_BASELINES" == "1" || -f "$out/$kind/macag_baselines.json" ]] || bl_done=0
    done
    resume_args=()
    graph_path="$out/graphs/${slug}.json"
    if [[ -f "$graph_path" ]]; then
      if NODE_THRESHOLD="${NODE_THRESHOLD:-1.0}" python - "$graph_path" <<'PY'
import json, os, sys
want = float(os.environ["NODE_THRESHOLD"])
meta = (json.load(open(sys.argv[1])).get("metadata") or {})
have = meta.get("node_threshold")
if have is None:
    raise SystemExit(1)
raise SystemExit(0 if abs(float(have) - want) < 1e-9 else 1)
PY
      then
        resume_args+=(--skip-attribute)
      else
        echo ">>> graph node_threshold mismatch vs requested ${NODE_THRESHOLD:-1.0}; rebuilding $graph_path"
      fi
    fi
    skip_baselines=()
    [[ "$SKIP_BASELINES" == "1" ]] && skip_baselines=(--skip-baselines)
    skip_game1=()
    [[ "$SKIP_GAME1" == "1" ]] && skip_game1=(--skip-game1)
    if [[ "$g1_done" -eq 1 && "$g2_done" -eq 1 && "$bl_done" -eq 1 ]]; then
      echo "SKIP-done $tag/$slug" | tee -a "$STATUS"; continue
    fi
    pipeline_args=(
      --prompt "$prompt" --target "$target" --foil "$foil"
      --model "$model" --transcoder-set "$tset"
      --slug "$slug" --outdir "$out" --device "$DEVICE"
      --freeze-mode "$FREEZE_MODE" --solvers "$SOLVERS"
      --score-kinds "$SCORE_KINDS"
      --budget "${BUDGET:-none}"
      --report-budget "${REPORT_BUDGET:-128}"
      --node-threshold "${NODE_THRESHOLD:-1.0}"
      --edge-threshold "${EDGE_THRESHOLD:-1.0}"
      --max-feature-nodes "${MAX_FEATURE_NODES:-1000000}"
      "${resume_args[@]}"
      "${skip_game1[@]}"
      "${skip_baselines[@]}"
    )
    [[ -n "${corrupted:-}" ]] && pipeline_args+=(--corrupted-prompt "$corrupted")
    if [[ "${CONNECTED}" == "1" || "${CONNECTED}" == "true" ]]; then
      pipeline_args+=(--connected)
    else
      pipeline_args+=(--no-connected)
    fi
    [[ -n "${PREFILTER_TOP_K}" && "${PREFILTER_TOP_K}" != "0" ]] && \
      pipeline_args+=(--prefilter-top-k "$PREFILTER_TOP_K")
    [[ -n "${GAME2_PREFILTER_TOP_K}" && "${GAME2_PREFILTER_TOP_K}" != "0" && "${GAME2_PREFILTER_TOP_K}" != "off" && "${GAME2_PREFILTER_TOP_K}" != "none" ]] && \
      pipeline_args+=(--game2-prefilter-top-k "$GAME2_PREFILTER_TOP_K")
    # A1: separate one-sided Game 2 oracle (empty = legacy shared kwargs).
    [[ -n "${GAME2_SCORE_KIND:-}" ]] && pipeline_args+=(--game2-score-kind "$GAME2_SCORE_KIND")
    [[ -n "${SHAPLEY_PERMUTATIONS:-}" ]] && pipeline_args+=(--shapley-permutations "$SHAPLEY_PERMUTATIONS")
    [[ -n "${BASELINE_METHODS:-}" ]] && pipeline_args+=(--baseline-methods "$BASELINE_METHODS")
    if [[ "$FREEZE_MODE" != "both" ]]; then
      pipeline_args+=(--stop-metric "$STOP_METRIC")
    fi
    # Dual freeze writes macag_game1.json only after BOTH legs. Harvest the
    # previous slug log into a Game 1 ckpt BEFORE truncating it, otherwise a
    # 24h kill restarts frozen greedy from |E|=0.
    slug_log="$OUTROOT/$tag-$slug.log"
    if [[ "$SKIP_GAME1" != "1" && -f "$slug_log" ]]; then
      for kind in $SCORE_KINDS; do
        if [[ -f "$out/$kind/macag_game1.json" ]]; then
          continue
        fi
        mkdir -p "$out/$kind"
        python - "$out/$kind/macag_game1.ckpt.json" "$slug_log" <<'PY'
import sys
from pathlib import Path
from macag.games.game1_min_faithful import (
    game1_checkpoint_progress,
    maybe_write_harvested_game1_checkpoint,
)
dest, log = sys.argv[1], sys.argv[2]
before = Path(dest)
old_key = None
if before.is_file():
    import json
    old_key = game1_checkpoint_progress(json.loads(before.read_text()))
written = maybe_write_harvested_game1_checkpoint(dest, [log])
if written is None:
    print(">>> no Game 1 log harvest")
    raise SystemExit(0)
new_key = game1_checkpoint_progress(written)
kept = old_key is not None and old_key >= new_key
action = "keep existing" if kept else "harvested"
print(
    f">>> {action} Game 1 ckpt frozen_done={bool(written.get('frozen_done'))} "
    f"leg={written.get('leg')} |frozen|={len(written.get('frozen_order') or [])} "
    f"|E|={len(written.get('selected_order') or [])} -> {dest}"
)
PY
      done
    fi
    gpu_slot=""
    gpu_slot="$(macag_acquire_gpu_slot "$OUTROOT")" || exit 1
    pipeline_rc=0
    if SCORE_KINDS="$SCORE_KINDS" CONNECTED="$CONNECTED" PREFILTER_TOP_K="${PREFILTER_TOP_K:-}" \
         GAME2_PREFILTER_TOP_K="${GAME2_PREFILTER_TOP_K:-}" \
         scripts/run_macag_pipeline.sh "${pipeline_args[@]}" \
          > "$OUTROOT/$tag-$slug.log" 2>&1; then
      :
    else
      pipeline_rc=1
    fi
    macag_release_gpu_slot "$gpu_slot"
    if [[ "$pipeline_rc" -eq 0 ]]; then
      echo "OK   $tag/$slug ($task/$mib_model)" | tee -a "$STATUS"
    else
      echo "FAIL $tag/$slug ($task/$mib_model)  (see $OUTROOT/$tag-$slug.log)" | tee -a "$STATUS"
    fi
  done < <(emit_prompts)
done

echo ""; echo ">>> DONE (worker $WORKER_ID/$NUM_WORKERS). Status summary:"
sort "$STATUS" | uniq -c | sort -rn | head
fi

run_analysis_now=0
if [[ "${RUN_ANALYSIS:-1}" != "0" ]]; then
  if [[ "$ANALYZE_ONLY" == "1" || "$NUM_WORKERS" -eq 1 ]]; then
    run_analysis_now=1
  fi
fi
if [[ "$run_analysis_now" -eq 0 && "$ANALYZE_ONLY" != "1" && "$NUM_WORKERS" -gt 1 ]]; then
  echo ""; echo ">>> Sharded run: skipping auto-analysis. After ALL shards finish, aggregate with:"
  echo ">>>   ANALYZE_ONLY=1 JSON=\"$JSON\" OUTROOT=\"$OUTROOT\" FREEZE_MODE=\"$FREEZE_MODE\" scripts/run_macag_mib.sh"
  echo ">>> (includes KL rescore -> summary/baselines CSVs with kl_faith columns)"
fi
if [[ "$run_analysis_now" -eq 1 ]]; then
  echo ""; echo ">>> Running analysis scripts (existing MACAG metrics) ..."
  run_analysis() { echo ">>> \$ $*"; "$@" || echo ">>> (analysis step failed: $*)"; }

  echo ""; echo ">>> ===== analysis: $OUTROOT ====="
  if [[ "${KL_RESCORE:-1}" != "0" ]]; then
    run_analysis python -m macag.cli.rescore_kl --root "$OUTROOT" --progress
  fi
  run_analysis python experiments/analyze_macag_acdc.py \
    --root "$OUTROOT" --bench "$JSON" --csv "$OUTROOT/summary.csv"
  run_analysis python experiments/analyze_game2_abr_vs_fp.py \
    --root "$OUTROOT" --bench "$JSON" --csv "$OUTROOT/abr_vs_fp.csv"
  run_analysis python experiments/analyze_macag_baselines.py \
    --root "$OUTROOT" --bench "$JSON" --csv "$OUTROOT/baselines.csv"
  if [[ "$FREEZE_MODE" == "both" ]]; then
    run_analysis python experiments/analyze_acdc_frozen_vs_unfrozen.py \
      --root "$OUTROOT" --bench "$JSON" \
      --csv "$OUTROOT/frozen_vs_unfrozen.csv"
  fi
  echo ""; echo ">>> Analysis complete. CSVs under $OUTROOT/"
  echo ">>> KL columns: summary.csv (kl_faith), baselines.csv (kl_faith_*); per-run macag_kl_faithfulness.json"
fi
