#!/usr/bin/env bash
# Run the full MACAG pipeline (attribute -> game1 -> game2 -> baselines -> annotate)
# over the ACDC benchmark prompts (macag/data/acdc_benchmark_prompts.json) for all
# three runnable CLTs. These are NEW prompts, so each needs a fresh attribution graph.
#
# This is the single entry point for MACAG benchmark runs. It produces per-prompt
# game outputs, head-to-head baseline comparisons (influence/eap/shapley/game1/acdc),
# and aggregate CSVs for MACAG metrics, ABR-vs-FP, baselines, and frozen-vs-unfrozen.
#
# Game 1 uses --freeze-mode both (parameter-matched frozen+unfrozen legs with
# attention_mediation diagnostic in macag_game1.json). Game 2 and baselines run
# under the default frozen oracle (freeze_attention=true).
#
# Usage:
#   scripts/run_macag_acdc.sh                # all tasks, all 3 CLTs
#   TASKS="indirect_object_identification" scripts/run_macag_acdc.sh   # one task
#   LIMIT=1 scripts/run_macag_acdc.sh        # first N prompts per task (smoke)
#
# Game 2 runs BOTH solvers per prompt: ABR and fictitious play (fp), writing
# macag_game2_abr.json and macag_game2_fp.json (plus a canonical macag_game2.json
# copy). Override with SOLVERS="abr" to run just one.
#
# Override Game 1 protocol with FREEZE_MODE=frozen|unfrozen|both (default both).
# To use the nonlinear stress prompts instead of the ACDC tasks:
#   JSON=macag/data/nonlinear_benchmark_prompts.json scripts/run_macag_acdc.sh
#
# Output: results/macag_acdc/<clt_tag>/<slug>/ (game1 dual, game2_{abr,fp},
#         macag_baselines.json, graph)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JSON="${JSON:-macag/data/acdc_benchmark_prompts.json}"
OUTROOT="${OUTROOT:-results/macag_acdc}"
DEVICE="${DEVICE:-cuda}"
SOLVERS="${SOLVERS:-abr fp}"   # Game 2 solvers, space-separated
FREEZE_MODE="${FREEZE_MODE:-both}"   # Game 1: frozen|unfrozen|both
# stop-metric only applies when FREEZE_MODE is frozen or unfrozen (both forces raw_relative)
STOP_METRIC="${STOP_METRIC:-raw_relative}"
TASKS="${TASKS:-}"     # space-separated task filter; empty = all tasks
LIMIT="${LIMIT:-0}"    # >0 = first N prompts per task (for smoke tests)
CLTS="${CLTS:-}"       # space-separated clt-tag filter; empty = all 3 CLTs
# greater_than is excluded by default: its target/foil (e.g. "42"/"40") share a
# first token, so first-token logit_gap scoring gives a degenerate gap of 0.
SKIP_TASKS="${SKIP_TASKS:-greater_than}"
SKIP_BASELINES="${SKIP_BASELINES:-0}"
# Grid sharding for single-GPU concurrency: launch NUM_WORKERS copies (each with a
# distinct WORKER_ID in 0..NUM_WORKERS-1) and they round-robin the (CLT,prompt)
# cells. Oracle calls underutilize a GH200, so 2-3 concurrent shards overlap well.
NUM_WORKERS="${NUM_WORKERS:-1}"
WORKER_ID="${WORKER_ID:-0}"
# ANALYZE_ONLY=1 skips the sweep and only runs the analyzers (use to aggregate
# once all shards have joined).
ANALYZE_ONLY="${ANALYZE_ONLY:-0}"

# clt_tag  model  transcoder_set
CLT_TAGS=(gemma2-426k gemma2-2.5M llama32-524k)
CLT_MODEL=(google/gemma-2-2b google/gemma-2-2b meta-llama/Llama-3.2-1B)
CLT_TSET=(mntss/clt-gemma-2-2b-426k mntss/clt-gemma-2-2b-2.5M mntss/clt-llama-3.2-1b-524k)

mkdir -p "$OUTROOT"
# Per-worker status file so concurrent shards do not truncate each other's log.
STATUS="$OUTROOT/status.w${WORKER_ID}.txt"; : > "$STATUS"
echo ">>> MACAG ACDC sweep | json=$JSON tasks=${TASKS:-ALL} limit=$LIMIT device=$DEVICE"
echo ">>> game1 freeze_mode=$FREEZE_MODE | out=$OUTROOT | worker=$WORKER_ID/$NUM_WORKERS"

# Emit "task<TAB>idx" per prompt. Only task+idx are line-safe; the prompt itself
# may contain newlines (docstring task), so it is fetched separately by index.
emit_prompts() {
  JSON="$JSON" TASKS="$TASKS" LIMIT="$LIMIT" SKIP_TASKS="$SKIP_TASKS" python - <<'PY'
import json, os
d = json.load(open(os.environ["JSON"]))
want = set(os.environ.get("TASKS", "").split()) or None
skip = set(os.environ.get("SKIP_TASKS", "").split())
limit = int(os.environ.get("LIMIT", "0"))
for task, items in d["tasks"].items():
    if (want and task not in want) or task in skip:
        continue
    for i in range(len(items)):
        if limit and i >= limit:
            break
        print(f"{task}\t{i}")
PY
}

# get_field <task> <idx> <field> -> raw value (newline-safe; no trailing newline).
get_field() {
  JSON="$JSON" python - "$1" "$2" "$3" <<'PY'
import json, os, sys
d = json.load(open(os.environ["JSON"]))
task, idx, field = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.stdout.write(d["tasks"][task][idx][field])
PY
}

CELL=0   # global (CLT,prompt) counter for round-robin sharding across workers
if [[ "$ANALYZE_ONLY" != "1" ]]; then
for ci in "${!CLT_TAGS[@]}"; do
  tag="${CLT_TAGS[$ci]}"; model="${CLT_MODEL[$ci]}"; tset="${CLT_TSET[$ci]}"
  if [[ -n "$CLTS" && " $CLTS " != *" $tag "* ]]; then continue; fi
  echo ""; echo ">>> ===== CLT $tag ($model | $tset) ====="
  while IFS=$'\t' read -r task idx; do
    [[ -z "$task" ]] && continue
    # Round-robin: each worker owns cells where (cell % NUM_WORKERS == WORKER_ID).
    cell=$CELL; CELL=$((CELL + 1))
    (( cell % NUM_WORKERS == WORKER_ID )) || continue
    slug=$(get_field "$task" "$idx" id)
    prompt=$(get_field "$task" "$idx" clean_prompt)
    target=$(get_field "$task" "$idx" correct_token)
    foil=$(get_field "$task" "$idx" incorrect_token)
    out="$OUTROOT/$tag/$slug"
    echo ">>> [$tag/$slug] ($task) target=$target foil=$foil"
    skip_baselines=()
    [[ "$SKIP_BASELINES" == "1" ]] && skip_baselines=(--skip-baselines)
    g1_done=1
    [[ -f "$out/macag_game1.json" ]] || g1_done=0
    g2_done=1
    [[ -f "$out/macag_game2_abr.json" && -f "$out/macag_game2_fp.json" ]] || g2_done=0
    bl_done=1
    [[ "$SKIP_BASELINES" == "1" || -f "$out/macag_baselines.json" ]] || bl_done=0
    if [[ "$g1_done" -eq 1 && "$g2_done" -eq 1 && "$bl_done" -eq 1 ]]; then
      echo "SKIP-done $tag/$slug" | tee -a "$STATUS"; continue
    fi
    mkdir -p "$out"
    pipeline_args=(
      --prompt "$prompt" --target "$target" --foil "$foil"
      --model "$model" --transcoder-set "$tset"
      --slug "$slug" --outdir "$out" --device "$DEVICE"
      --freeze-mode "$FREEZE_MODE" --solvers "$SOLVERS"
      "${skip_baselines[@]}"
    )
    # Optional cost knobs: Shapley is the dominant baseline cost. Lower
    # SHAPLEY_PERMUTATIONS or trim BASELINE_METHODS to speed up the sweep.
    [[ -n "${SHAPLEY_PERMUTATIONS:-}" ]] && pipeline_args+=(--shapley-permutations "$SHAPLEY_PERMUTATIONS")
    [[ -n "${BASELINE_METHODS:-}" ]] && pipeline_args+=(--baseline-methods "$BASELINE_METHODS")
    if [[ "$FREEZE_MODE" != "both" ]]; then
      pipeline_args+=(--stop-metric "$STOP_METRIC")
    fi
    if scripts/run_macag_pipeline.sh "${pipeline_args[@]}" \
          > "$OUTROOT/$tag-$slug.log" 2>&1; then
      echo "OK   $tag/$slug ($task)" | tee -a "$STATUS"
    else
      echo "FAIL $tag/$slug ($task)  (see $OUTROOT/$tag-$slug.log)" | tee -a "$STATUS"
    fi
  done < <(emit_prompts)
done

echo ""; echo ">>> DONE (worker $WORKER_ID/$NUM_WORKERS). Status summary:"
sort "$STATUS" | uniq -c | sort -rn | head
fi  # end ANALYZE_ONLY guard

# --- auto-analysis once all games are finished -----------------------------
# Runs when single-worker (default) or in ANALYZE_ONLY mode. Sharded runs skip
# it (each shard finishes on partial data) and print the join command instead.
# Set RUN_ANALYSIS=0 to skip entirely. Analysis failures do not affect exit code.
run_analysis_now=0
if [[ "${RUN_ANALYSIS:-1}" != "0" ]]; then
  if [[ "$ANALYZE_ONLY" == "1" || "$NUM_WORKERS" -eq 1 ]]; then
    run_analysis_now=1
  fi
fi
if [[ "$run_analysis_now" -eq 0 && "$ANALYZE_ONLY" != "1" && "$NUM_WORKERS" -gt 1 ]]; then
  echo ""; echo ">>> Sharded run: skipping auto-analysis. After ALL shards finish, aggregate with:"
  echo ">>>   ANALYZE_ONLY=1 JSON=\"$JSON\" OUTROOT=\"$OUTROOT\" FREEZE_MODE=\"$FREEZE_MODE\" scripts/run_macag_acdc.sh"
fi
if [[ "$run_analysis_now" -eq 1 ]]; then
  echo ""; echo ">>> Running analysis scripts ..."
  run_analysis() { echo ">>> \$ $*"; "$@" || echo ">>> (analysis step failed: $*)"; }

  echo ""; echo ">>> ===== analysis: $OUTROOT ====="
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
fi
