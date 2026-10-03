#!/usr/bin/env bash
#
# End-to-end MACAG pipeline over a linear-CLT circuit-tracer graph:
#   attribute (build graph) -> game1 + game2 + baselines -> annotate -> (optional) serve
#
# Assumes the conda env is already activated (uses `python` directly).
# Run from the repo root, or it will cd there based on this script's location.
#
# Usage:
#   scripts/run_macag_pipeline.sh \
#     --prompt "Fact: The capital of the state containing Dallas is" \
#     --target " Austin" --foil " Texas"
#
# Override any default via --flag value (see DEFAULTS below). Common ones:
#   --model --transcoder-set --slug --outdir --device --dtype
#   --prefilter-top-k --game2-prefilter-top-k --budget --report-budget --beta --abr-iters --solvers --fp-tol --alpha --lam --eps
#   --stop-metric raw_relative|normalized   Game 1 stop (ignored when --freeze-mode both)
#   --freeze-mode frozen|unfrozen|both   Game 1 attention protocol (default both)
#   --freeze-select frozen|unfrozen|both annotate dual Game 1 legs (default both)
#   --max-feature-nodes --batch-size --node-threshold --edge-threshold
#   --freeze-attention true|false   scoring-time attention freeze (default true)
#   --score-kinds "logit_gap kl_divergence"  selection utilities (default both; TMLR)
#   --score-kind KIND   single utility override (logit_gap|kl_divergence|answer_span|...)
#   --budget none|N     Game1/Game2 size cap; "none" = natural |E| (TMLR v4 default)
#   --report-budget N   Ranking-baseline F-vs-k curve length (default 128)
#   --no-connected      default; pass --connected to require evidence connectivity
#   --prefilter-top-k N optional singleton prefilter for Game 1 (default: off for TMLR)
#   --game2-prefilter-top-k N Game 2 only (does not change Game 1)
#   --graph PATH       use this graph instead of <outdir>/graphs/<slug>.json
#   --skip-attribute   reuse an existing graph at <outdir>/graphs/<slug>.json
#   --skip-game1       reuse <outdir>/macag_game1.json
#   --skip-game2       reuse <outdir>/macag_game2_<solver>.json for each solver in --solvers
#                      (or pass empty --solvers / omit Game 2 entirely: Pass A)
#   --skip-baselines   skip the head-to-head baseline harness (influence/eap/shapley/game1/acdc)
#   --no-parallel-agents  Game 2: sequential y-then-foil instead of within-round 2-way parallel
#   --skip-kl-rescore  skip post-hoc KL faithfulness on saved evidence sets
#   --baseline-methods influence   (TMLR v4 Pass A; no duplicate game1; no shapley)
#   --shapley-permutations 64
#   --shapley-seed 0     MC Shapley/Banzhaf sampling seed (or SHAPLEY_SEED env)
#   --serve            launch the visualization server at the end
#   --port             server port (default 8041)
set -euo pipefail

# --- locate repo root (parent of this script's dir) -------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-.}:."

# --- DEFAULTS ---------------------------------------------------------------
PROMPT="Fact: The capital of the state containing Dallas is"
TARGET=" Austin"
FOIL=" Texas"
MODEL="google/gemma-2-2b"
TRANSCODER_SET="mntss/clt-gemma-2-2b-426k"
SLUG="dallas-austin"
OUTDIR="results/macag_demo"
DEVICE="cuda"          # cuda | cpu  (mps is unsupported: safetensors lazy-decoder)
DTYPE="bfloat16"
# graph (attribution) params
# Graph export: TMLR v4 default is unpruned (full influence mass). Override to
# 0.8/0.98 only for explicit pruned-graph sensitivity runs.
MAX_FEATURE_NODES="${MAX_FEATURE_NODES:-1000000}"
BATCH_SIZE=256
NODE_THRESHOLD="${NODE_THRESHOLD:-1.0}"
EDGE_THRESHOLD="${EDGE_THRESHOLD:-1.0}"
# game params — TMLR protocol: no singleton prefilter, no connectivity constraint
# (both were distorting short-prompt evidence sets). Empty PREFILTER_TOP_K =
# disabled. CONNECTED=0 -> --no-connected.
PREFILTER_TOP_K="${PREFILTER_TOP_K:-}"
# Game 2-only pool. Empty = inherit PREFILTER_TOP_K (usually off). Pass A v4
# sets 50 so ABR+FP can finish on unpruned graphs without prefiltering Game 1.
GAME2_PREFILTER_TOP_K="${GAME2_PREFILTER_TOP_K:-}"
CONNECTED="${CONNECTED:-0}"
# BUDGET empty / "none" / "null" => omit --budget for Game1/Game2 (natural size).
# REPORT_BUDGET is the finite k used only for ranking-baseline F-vs-k curves;
# matched-k vs Game1 |E*| is computed at analysis time from those rankings.
BUDGET="${BUDGET:-none}"
REPORT_BUDGET="${REPORT_BUDGET:-128}"
BETA=0.2
ABR_ITERS=4
ALPHA=0.5
LAM=0.02
EPS=0.1
# Selection utilities to run on the same graph (compare head-to-head).
# Pass A / TMLR v3 default: logit_gap only (KL via post-hoc rescore).
SCORE_KINDS="${SCORE_KINDS:-logit_gap}"
# Legacy single-kind override still works; if SCORE_KIND is set explicitly and
# SCORE_KINDS was left at default, prefer the single kind for back-compat
# one-off runs that export SCORE_KIND=answer_span etc.
if [[ -n "${SCORE_KIND:-}" && "${SCORE_KINDS}" == "logit_gap" && "${SCORE_KIND}" != "logit_gap" ]]; then
  SCORE_KINDS="$SCORE_KIND"
fi
# Ablation values: zero (default) | mean (per-feature mean over clean-prompt
# positions) | corrupted (patch-style value from CORRUPTED_PROMPT at the same
# position — the ACDC convention; requires CORRUPTED_PROMPT).
ABLATION_MODE="${ABLATION_MODE:-zero}"
CORRUPTED_PROMPT="${CORRUPTED_PROMPT:-}"
# Game 1 stop metric for --faithfulness-eps when freeze_mode is frozen/unfrozen.
# Ignored when --freeze-mode both (raw_relative is forced on both legs).
STOP_METRIC=raw_relative
# Game 1 attention protocol. 'both' runs matched frozen+unfrozen legs in one
# invocation and emits attention_mediation (recommended for ACDC benchmarks).
FREEZE_MODE=both
FREEZE_SELECT=both
# Game 2 solver(s) to run, space-separated. Each is run independently from the
# same candidate universe, writing macag_game2_<solver>.json. Choices: abr fp.
# Empty string = skip Game 2 (Pass A). Pass A2 sets SOLVERS="abr fp".
SOLVERS="abr fp"
FP_TOL=1e-3
# Within-round Jacobi parallel (y || foil, barrier after each round).
PARALLEL_AGENTS="${PARALLEL_AGENTS:-1}"
# candidate universe: empty = full feature-node universe (recommended on CUDA).
# Set to a path (.json list or text) to restrict, e.g. for CPU tractability.
CANDIDATES_FILE=""
SKIP_ATTRIBUTE=0
SKIP_GAME1=0
SKIP_GAME2=0
SKIP_BASELINES=0
BASELINE_METHODS="${BASELINE_METHODS:-influence}"
SHAPLEY_PERMUTATIONS=64
# Seed for the MC Shapley/Banzhaf gold baseline (the only stochastic stage of
# the pipeline). Env-overridable so sweep drivers can run seeded replicates:
#   SHAPLEY_SEED=1 scripts/run_macag_acdc_parallel.sh ...
SHAPLEY_SEED="${SHAPLEY_SEED:-0}"
# Post-hoc KL faithfulness on saved evidence sets (set 0 to skip extra GPU work).
# When the selection utility is already kl_divergence this is a no-op check.
KL_RESCORE="${KL_RESCORE:-1}"
SKIP_KL_RESCORE=0
# freeze attention+embeddings+error nodes during scoring (scoring-time only;
# the attribution graph is identical either way). true = frozen baseline.
FREEZE_ATTENTION=true
# optional explicit graph path; empty = $OUTDIR/graphs/$SLUG.json. Set this with
# --skip-attribute to reuse a graph built elsewhere (e.g. the frozen pass).
GRAPH_OVERRIDE=""
SERVE=0
PORT=8041

# --- parse --flag value -----------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt) PROMPT="$2"; shift 2;;
    --target) TARGET="$2"; shift 2;;
    --foil) FOIL="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --transcoder-set) TRANSCODER_SET="$2"; shift 2;;
    --slug) SLUG="$2"; shift 2;;
    --outdir) OUTDIR="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --dtype) DTYPE="$2"; shift 2;;
    --max-feature-nodes) MAX_FEATURE_NODES="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --node-threshold) NODE_THRESHOLD="$2"; shift 2;;
    --edge-threshold) EDGE_THRESHOLD="$2"; shift 2;;
    --prefilter-top-k) PREFILTER_TOP_K="$2"; shift 2;;
    --game2-prefilter-top-k) GAME2_PREFILTER_TOP_K="$2"; shift 2;;
    --connected) CONNECTED=1; shift;;
    --no-connected) CONNECTED=0; shift;;
    --budget) BUDGET="$2"; shift 2;;
    --report-budget) REPORT_BUDGET="$2"; shift 2;;
    --beta) BETA="$2"; shift 2;;
    --abr-iters) ABR_ITERS="$2"; shift 2;;
    --solvers) SOLVERS="$2"; shift 2;;
    --fp-tol) FP_TOL="$2"; shift 2;;
    --parallel-agents) PARALLEL_AGENTS=1; shift;;
    --no-parallel-agents) PARALLEL_AGENTS=0; shift;;
    --alpha) ALPHA="$2"; shift 2;;
    --lam) LAM="$2"; shift 2;;
    --eps) EPS="$2"; shift 2;;
    --stop-metric) STOP_METRIC="$2"; shift 2;;
    --score-kind) SCORE_KINDS="$2"; SCORE_KIND="$2"; shift 2;;
    --score-kinds) SCORE_KINDS="$2"; shift 2;;
    --ablation-mode) ABLATION_MODE="$2"; shift 2;;
    --corrupted-prompt) CORRUPTED_PROMPT="$2"; shift 2;;
    --freeze-mode) FREEZE_MODE="$2"; shift 2;;
    --freeze-select) FREEZE_SELECT="$2"; shift 2;;
    --candidates-file) CANDIDATES_FILE="$2"; shift 2;;
    --freeze-attention) FREEZE_ATTENTION="$2"; shift 2;;
    --graph) GRAPH_OVERRIDE="$2"; shift 2;;
    --skip-attribute) SKIP_ATTRIBUTE=1; shift;;
    --skip-game1) SKIP_GAME1=1; shift;;
    --skip-game2) SKIP_GAME2=1; shift;;
    --skip-baselines) SKIP_BASELINES=1; shift;;
    --skip-kl-rescore) SKIP_KL_RESCORE=1; KL_RESCORE=0; shift;;
    --baseline-methods) BASELINE_METHODS="$2"; shift 2;;
    --shapley-permutations) SHAPLEY_PERMUTATIONS="$2"; shift 2;;
    --shapley-seed) SHAPLEY_SEED="$2"; shift 2;;
    --serve) SERVE=1; shift;;
    --port) PORT="$2"; shift 2;;
    -h|--help) sed -n '2,40p' "$0"; exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

GRAPH="$OUTDIR/graphs/$SLUG.json"
[[ -n "$GRAPH_OVERRIDE" ]] && GRAPH="$GRAPH_OVERRIDE"
mkdir -p "$OUTDIR/graphs"

PREFILTER_ARGS=()
if [[ -n "${PREFILTER_TOP_K}" && "${PREFILTER_TOP_K}" != "0" && "${PREFILTER_TOP_K}" != "off" && "${PREFILTER_TOP_K}" != "none" ]]; then
  PREFILTER_ARGS=(--prefilter-top-k "$PREFILTER_TOP_K")
fi
GAME2_PREFILTER_ARGS=()
if [[ -n "${GAME2_PREFILTER_TOP_K}" && "${GAME2_PREFILTER_TOP_K}" != "0" && "${GAME2_PREFILTER_TOP_K}" != "off" && "${GAME2_PREFILTER_TOP_K}" != "none" ]]; then
  GAME2_PREFILTER_ARGS=(--prefilter-top-k "$GAME2_PREFILTER_TOP_K")
else
  GAME2_PREFILTER_ARGS=("${PREFILTER_ARGS[@]}")
fi
if [[ "${CONNECTED}" == "1" || "${CONNECTED}" == "true" ]]; then
  CONNECTED_ARGS=(--connected)
else
  CONNECTED_ARGS=(--no-connected)
fi

CAND_ARG=()
[[ -n "$CANDIDATES_FILE" ]] && CAND_ARG=(--candidates-file "$CANDIDATES_FILE")

echo ">>> MACAG pipeline | model=$MODEL clt=$TRANSCODER_SET device=$DEVICE"
echo ">>> prompt: $PROMPT"
echo ">>> target=$TARGET  foil=$FOIL  -> $OUTDIR"
echo ">>> score_kinds=[$SCORE_KINDS] prefilter=${PREFILTER_TOP_K:-off} game2_prefilter=${GAME2_PREFILTER_TOP_K:-${PREFILTER_TOP_K:-off}} connected=${CONNECTED}"

macag_nsys_this_worker() {
  [[ "${NSYS_PROFILE:-0}" == "1" ]] || return 1
  local want="${NSYS_WORKER_IDS:-}"
  [[ -z "$want" ]] && return 0
  local wid="${WORKER_ID:-0}"
  [[ ",${want}," == *",${wid},"* ]]
}

run_game1_python() {
  if ! macag_nsys_this_worker; then
    python -m macag.cli.run_macag game1 "$@"
    return
  fi
  local nsys_bin="${NSYS_BIN:-nsys}"
  if [[ ! -x "$nsys_bin" ]] && ! command -v "$nsys_bin" >/dev/null 2>&1; then
    echo "ERROR: NSYS_PROFILE=1 but nsys not found (NSYS_BIN=$nsys_bin)" >&2
    return 2
  fi
  local out_dir="${NSYS_OUTDIR:-$OUTDIR/nsys}"
  mkdir -p "$out_dir"
  local out="$out_dir/game1.${SLUG}.w${WORKER_ID:-0}"
  local delay="${NSYS_DELAY:-90}"
  local duration="${NSYS_DURATION:-180}"
  echo ">>> nsys profile delay=${delay}s duration=${duration}s kill=none -> ${out}.nsys-rep"
  "$nsys_bin" profile \
    --trace=cuda,nvtx,osrt,cublas,cudnn \
    --sample=none \
    --cpuctxsw=none \
    --cuda-memory-usage=true \
    --delay="$delay" \
    --duration="$duration" \
    --kill=none \
    --force-overwrite=true \
    --stats=true \
    --output="$out" \
    python -m macag.cli.run_macag game1 "$@"
}

# --- 1) attribution graph ---------------------------------------------------
if [[ "$SKIP_ATTRIBUTE" -eq 0 ]]; then
  echo ">>> [1/6] Building attribution graph ..."
  python -m circuit_tracer attribute \
    -m "$MODEL" \
    -t "$TRANSCODER_SET" \
    -p "$PROMPT" \
    --dtype "$DTYPE" \
    --batch_size "$BATCH_SIZE" \
    --max_feature_nodes "$MAX_FEATURE_NODES" \
    --slug "$SLUG" \
    --graph_file_dir "$OUTDIR/graphs" \
    --node_threshold "$NODE_THRESHOLD" --edge_threshold "$EDGE_THRESHOLD" \
    --verbose
else
  echo ">>> [1/6] Skipping attribution; reusing $GRAPH"
  [[ -f "$GRAPH" ]] || { echo "ERROR: $GRAPH not found"; exit 1; }
fi

ORACLE_FACTORY="macag.factories.replacement_model:create_replacement_model_oracle"
G1_STOP_ARGS=()
if [[ "$FREEZE_MODE" != "both" ]]; then
  G1_STOP_ARGS=(--stop-metric "$STOP_METRIC")
fi

write_oracle_kwargs() {
  local kwargs_path="$1"
  local score_kind="$2"
  PROMPT="$PROMPT" TARGET="$TARGET" FOIL="$FOIL" MODEL="$MODEL" \
  TRANSCODER_SET="$TRANSCODER_SET" GRAPH="$GRAPH" DEVICE="$DEVICE" DTYPE="$DTYPE" \
  FREEZE_ATTENTION="$FREEZE_ATTENTION" SCORE_KIND="$score_kind" \
  ABLATION_MODE="$ABLATION_MODE" CORRUPTED_PROMPT="$CORRUPTED_PROMPT" \
  KWARGS="$kwargs_path" python - <<'PY'
import json, os
freeze = os.environ.get("FREEZE_ATTENTION", "true").strip().lower() in ("1", "true", "yes", "y")
kw = {
    "model_name": os.environ["MODEL"],
    "transcoder_set": os.environ["TRANSCODER_SET"],
    "prompt": os.environ["PROMPT"],
    "graph_json": os.environ["GRAPH"],
    "backend": "transformerlens",
    "score_kind": os.environ.get("SCORE_KIND", "logit_gap"),
    "strict_single_token": False,
    "freeze_attention": freeze,
    "target_token_by_label": {"y": os.environ["TARGET"], "y_foil": os.environ["FOIL"]},
    "foil_by_target": {"y": "y_foil", "y_foil": "y"},
    "model_kwargs": {"dtype": os.environ["DTYPE"], "device": os.environ["DEVICE"]},
}
ablation_mode = os.environ.get("ABLATION_MODE", "zero")
if ablation_mode != "zero":
    kw["ablation_mode"] = ablation_mode
# Always persist corrupted_prompt when provided (eap_syed / edge originals need it
# even under zero ablation).
corrupted = os.environ.get("CORRUPTED_PROMPT", "")
if corrupted:
    kw["corrupted_prompt"] = corrupted
with open(os.environ["KWARGS"], "w") as f:
    json.dump(kw, f, indent=2)
print("wrote", os.environ["KWARGS"])
PY
}

# Resolve game vs baseline budgets.
GAME_BUDGET_ARGS=()
BASELINE_BUDGET="$REPORT_BUDGET"
case "${BUDGET}" in
  ""|none|None|NONE|null|NULL)
    echo ">>> Game1/Game2: unbudgeted (natural |E|); baselines report_budget=$BASELINE_BUDGET"
    ;;
  *)
    GAME_BUDGET_ARGS=(--budget "$BUDGET")
    # If caller only set BUDGET (legacy), use it for baseline curves too unless
    # REPORT_BUDGET was explicitly exported to something else.
    if [[ "${REPORT_BUDGET}" == "128" && "$BUDGET" != "128" ]]; then
      BASELINE_BUDGET="$BUDGET"
    fi
    echo ">>> Game1/Game2 budget=$BUDGET; baselines report_budget=$BASELINE_BUDGET"
    ;;
esac

PRIMARY_KIND=""
for SCORE_KIND in $SCORE_KINDS; do
  KIND_DIR="$OUTDIR/$SCORE_KIND"
  mkdir -p "$KIND_DIR"
  [[ -z "$PRIMARY_KIND" ]] && PRIMARY_KIND="$SCORE_KIND"
  KWARGS="$KIND_DIR/oracle_kwargs.json"
  G1_OUT="$KIND_DIR/macag_game1.json"
  G2_OUT="$KIND_DIR/macag_game2.json"
  BASELINES_OUT="$KIND_DIR/macag_baselines.json"
  MACAG_SLUG="macag-${SLUG}-${SCORE_KIND}"
  ANNOTATED="$OUTDIR/graphs/${MACAG_SLUG}.json"

  echo ">>> ===== score_kind=$SCORE_KIND -> $KIND_DIR ====="
  echo ">>> [2/6] Writing oracle kwargs (score_kind=$SCORE_KIND, freeze_attention=$FREEZE_ATTENTION)"
  write_oracle_kwargs "$KWARGS" "$SCORE_KIND"

  if [[ "$SKIP_GAME1" -eq 0 ]]; then
    if [[ -f "$G1_OUT" ]]; then
      echo ">>> [3/6] Skipping Game 1; reusing $G1_OUT"
    else
      echo ">>> [3/6] Game 1 | freeze_mode=$FREEZE_MODE ${G1_STOP_ARGS[*]} ${CONNECTED_ARGS[*]} ${PREFILTER_ARGS[*]} ${GAME_BUDGET_ARGS[*]:-unbudgeted}"
      run_game1_python \
        --graph-json "$GRAPH" --target y --input-id "$SLUG" \
        --oracle-factory "$ORACLE_FACTORY" \
        --oracle-kwargs-file "$KWARGS" \
        "${CAND_ARG[@]}" \
        "${PREFILTER_ARGS[@]}" "${CONNECTED_ARGS[@]}" \
        "${GAME_BUDGET_ARGS[@]}" \
        --alpha "$ALPHA" --lam "$LAM" \
        --faithfulness-eps "$EPS" \
        --freeze-mode "$FREEZE_MODE" \
        "${G1_STOP_ARGS[@]}" \
        --checkpoint-json "$KIND_DIR/macag_game1.ckpt.json" \
        --output-json "$G1_OUT"
    fi
  else
    if [[ -f "$G1_OUT" ]]; then
      echo ">>> [3/6] Skipping Game 1; reusing $G1_OUT"
    elif [[ "$SKIP_GAME2" -eq 1 || -z "${SOLVERS// }" ]]; then
      echo "ERROR: $G1_OUT not found (needed when Game 2 is also skipped)" >&2
      exit 1
    else
      echo ">>> [3/6] Skipping Game 1 (no $G1_OUT; Pass A2 Game 2 will still run)"
    fi
  fi

  echo ">>> [4/6] Game 2 | solvers: ${SOLVERS:-<skipped>} prefilter=${GAME2_PREFILTER_TOP_K:-${PREFILTER_TOP_K:-off}}"
  if [[ "$SKIP_GAME2" -eq 1 || -z "${SOLVERS// }" ]]; then
    echo ">>>      Skipping Game 2 (--skip-game2 or empty --solvers)"
    # Prefer an existing abr/fp artifact if present (resume / partial runs).
    if [[ -f "$KIND_DIR/macag_game2_abr.json" ]]; then
      cp "$KIND_DIR/macag_game2_abr.json" "$G2_OUT"
    elif [[ -f "$KIND_DIR/macag_game2_fp.json" ]]; then
      cp "$KIND_DIR/macag_game2_fp.json" "$G2_OUT"
    elif [[ -f "$G2_OUT" ]]; then
      :
    else
      # Annotator needs a JSON; mirror Game 1 evidence into the Game 2 schema keys.
      python - "$G1_OUT" "$G2_OUT" <<'PY'
import json, sys
g1 = json.load(open(sys.argv[1]))
payload = {
    "input_id": g1.get("input_id"),
    "target": g1.get("target"),
    "foil": g1.get("foil"),
    "game": "game2_skipped",
    "params": g1.get("params") or {},
    "evidence": {"E_y": [], "E_foil": [], "shared": [], "unique_y": [], "unique_foil": []},
    "note": "Game 2 skipped; placeholder for annotate_graph",
}
# Dual-freeze Game 1: lift frozen E_star into E_y for a non-empty annotate.
leg = g1.get("frozen") or g1.get("unfrozen") or g1
ev = (leg.get("evidence") or {}) if isinstance(leg, dict) else {}
estar = ev.get("E_star") or ev.get("E_y") or []
if estar:
    payload["evidence"]["E_y"] = list(estar)
    payload["evidence"]["unique_y"] = list(estar)
json.dump(payload, open(sys.argv[2], "w"), indent=2)
print("wrote placeholder", sys.argv[2])
PY
    fi
  else
    for solver in $SOLVERS; do
      g2_out="$KIND_DIR/macag_game2_${solver}.json"
      if [[ -f "$g2_out" ]]; then
        echo ">>>      solver=$solver -> $g2_out (already present)"
        continue
      fi
      echo ">>>      solver=$solver -> $g2_out"
      g2_cmd=(
        python -m macag.cli.run_macag game2
        --graph-json "$GRAPH" --target y --foil y_foil --input-id "$SLUG"
        --oracle-factory "$ORACLE_FACTORY"
        --oracle-kwargs-file "$KWARGS"
      )
      ((${#CAND_ARG[@]})) && g2_cmd+=("${CAND_ARG[@]}")
      ((${#GAME2_PREFILTER_ARGS[@]})) && g2_cmd+=("${GAME2_PREFILTER_ARGS[@]}")
      ((${#CONNECTED_ARGS[@]})) && g2_cmd+=("${CONNECTED_ARGS[@]}")
      g2_cmd+=(
        "${GAME_BUDGET_ARGS[@]}"
        --beta "$BETA" --abr-iters "$ABR_ITERS"
        --solver "$solver" --fp-tol "$FP_TOL"
        --alpha "$ALPHA" --lam "$LAM"
        --checkpoint-json "$KIND_DIR/macag_game2_${solver}.ckpt.json"
        --output-json "$g2_out"
      )
      if [[ "${PARALLEL_AGENTS}" == "0" || "${PARALLEL_AGENTS}" == "false" || "${PARALLEL_AGENTS}" == "no" ]]; then
        g2_cmd+=(--no-parallel-agents)
      fi
      "${g2_cmd[@]}"
    done
    if [[ -f "$KIND_DIR/macag_game2_abr.json" ]]; then
      cp "$KIND_DIR/macag_game2_abr.json" "$G2_OUT"
    else
      first_solver="${SOLVERS%% *}"
      cp "$KIND_DIR/macag_game2_${first_solver}.json" "$G2_OUT"
    fi
  fi

  if [[ "$SKIP_BASELINES" -eq 0 ]]; then
    if [[ -f "$BASELINES_OUT" ]]; then
      echo ">>> [5/6] Skipping baselines; reusing $BASELINES_OUT"
    else
      echo ">>> [5/6] Baselines (methods=$BASELINE_METHODS, report_budget=$BASELINE_BUDGET) -> $BASELINES_OUT"
      acdc_target_args=()
      [[ -n "${ACDC_TARGET_K:-}" ]] && acdc_target_args=(--acdc-target-k "$ACDC_TARGET_K")
      bl_cmd=(
        python -m macag.cli.run_baselines
        --graph-json "$GRAPH" --target y --input-id "$SLUG"
        --oracle-factory "$ORACLE_FACTORY"
        --oracle-kwargs-file "$KWARGS"
      )
      ((${#CAND_ARG[@]})) && bl_cmd+=("${CAND_ARG[@]}")
      ((${#PREFILTER_ARGS[@]})) && bl_cmd+=("${PREFILTER_ARGS[@]}")
      if [[ "${CONNECTED}" != "1" && "${CONNECTED}" != "true" ]]; then
        bl_cmd+=(--no-connected)
      fi
      bl_cmd+=(
        --budget "$BASELINE_BUDGET" --alpha "$ALPHA" --lam "$LAM"
        --methods "$BASELINE_METHODS"
        --shapley-permutations "$SHAPLEY_PERMUTATIONS"
        --shapley-seed "$SHAPLEY_SEED"
        "${acdc_target_args[@]}"
        --output-json "$BASELINES_OUT"
      )
      "${bl_cmd[@]}"
    fi
  else
    echo ">>> [5/6] Skipping baseline harness"
  fi

  echo ">>> [6/6] Annotating graph -> $ANNOTATED"
  python -m macag.cli.annotate_graph \
    --graph-json "$GRAPH" \
    --macag-result-json "$G2_OUT" --label-prefix "MACAG:g2" \
    --output-json "$ANNOTATED"
  python -m macag.cli.annotate_graph \
    --graph-json "$ANNOTATED" \
    --macag-result-json "$G1_OUT" --label-prefix "MACAG:g1" \
    --freeze-select "$FREEZE_SELECT" \
    --output-json "$ANNOTATED"

  if [[ "$KL_RESCORE" == "1" && "$SKIP_KL_RESCORE" -eq 0 && "$SCORE_KIND" != "kl_divergence" ]]; then
    echo ">>> KL rescore (eval metric) -> $KIND_DIR/macag_kl_faithfulness.json"
    python -m macag.cli.rescore_kl --run-dir "$KIND_DIR" --progress || \
      echo ">>> (KL rescore failed; selection results are unchanged)"
  fi
done

if [[ -n "$PRIMARY_KIND" && -d "$OUTDIR/$PRIMARY_KIND" ]]; then
  for f in macag_game1.json macag_game2.json macag_baselines.json oracle_kwargs.json \
           macag_game2_abr.json macag_game2_fp.json macag_kl_faithfulness.json; do
    if [[ -f "$OUTDIR/$PRIMARY_KIND/$f" && ! -e "$OUTDIR/$f" ]]; then
      ln -sfn "$PRIMARY_KIND/$f" "$OUTDIR/$f"
    fi
  done
fi

echo ">>> Done."
echo "    graph      : $GRAPH"
echo "    score_kinds: $SCORE_KINDS"
for SCORE_KIND in $SCORE_KINDS; do
  echo "    $SCORE_KIND/game1 : $OUTDIR/$SCORE_KIND/macag_game1.json"
done

if [[ "$SERVE" -eq 1 ]]; then
  echo ">>> Serving $OUTDIR/graphs on port $PORT (Ctrl-C to stop) ..."
  python -m circuit_tracer start-server --graph_file_dir "$OUTDIR/graphs" --port "$PORT"
else
  echo ">>> To visualize:"
  echo "    python -m circuit_tracer start-server --graph_file_dir $OUTDIR/graphs --port $PORT"
fi
