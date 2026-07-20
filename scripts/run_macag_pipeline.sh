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
#   --prefilter-top-k --budget --beta --abr-iters --solvers --fp-tol --alpha --lam --eps
#   --stop-metric raw_relative|normalized   Game 1 stop (ignored when --freeze-mode both)
#   --freeze-mode frozen|unfrozen|both   Game 1 attention protocol (default both)
#   --freeze-select frozen|unfrozen|both annotate dual Game 1 legs (default both)
#   --max-feature-nodes --batch-size --node-threshold --edge-threshold
#   --freeze-attention true|false   scoring-time attention freeze (default true)
#   --score-kind KIND  oracle score: logit_gap (default) | answer_span | logit | prob | negative_loss
#   --graph PATH       use this graph instead of <outdir>/graphs/<slug>.json
#   --skip-attribute   reuse an existing graph at <outdir>/graphs/<slug>.json
#   --skip-game1       reuse <outdir>/macag_game1.json
#   --skip-game2       reuse <outdir>/macag_game2_<solver>.json for each solver in --solvers
#   --skip-baselines   skip the head-to-head baseline harness (influence/eap/shapley/game1/acdc)
#   --skip-kl-rescore  skip post-hoc KL faithfulness on saved evidence sets
#   --baseline-methods influence,eap,shapley,game1,acdc
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
MAX_FEATURE_NODES=7500
BATCH_SIZE=256
NODE_THRESHOLD=0.8
EDGE_THRESHOLD=0.98
# game params
PREFILTER_TOP_K=20
BUDGET=8
BETA=0.2
ABR_ITERS=4
ALPHA=0.5
LAM=0.02
EPS=0.1
# Oracle score kind: logit_gap (default) | answer_span (full-span log-prob gap;
# use when target/foil share a first token, e.g. greater_than) | logit | prob |
# negative_loss. Env-overridable: SCORE_KIND=answer_span scripts/run_macag_pipeline.sh ...
SCORE_KIND="${SCORE_KIND:-logit_gap}"
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
# same prefiltered universe, writing macag_game2_<solver>.json. Choices: abr fp.
SOLVERS="abr fp"
FP_TOL=1e-3
# candidate universe: empty = full feature-node universe (recommended on CUDA).
# Set to a path (.json list or text) to restrict, e.g. for CPU tractability.
CANDIDATES_FILE=""
SKIP_ATTRIBUTE=0
SKIP_GAME1=0
SKIP_GAME2=0
SKIP_BASELINES=0
BASELINE_METHODS="influence,eap,shapley,game1,acdc"
SHAPLEY_PERMUTATIONS=64
# Seed for the MC Shapley/Banzhaf gold baseline (the only stochastic stage of
# the pipeline). Env-overridable so sweep drivers can run seeded replicates:
#   SHAPLEY_SEED=1 scripts/run_macag_acdc_parallel.sh ...
SHAPLEY_SEED="${SHAPLEY_SEED:-0}"
# Post-hoc KL faithfulness on saved evidence sets (set 0 to skip extra GPU work).
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
    --budget) BUDGET="$2"; shift 2;;
    --beta) BETA="$2"; shift 2;;
    --abr-iters) ABR_ITERS="$2"; shift 2;;
    --solvers) SOLVERS="$2"; shift 2;;
    --fp-tol) FP_TOL="$2"; shift 2;;
    --alpha) ALPHA="$2"; shift 2;;
    --lam) LAM="$2"; shift 2;;
    --eps) EPS="$2"; shift 2;;
    --stop-metric) STOP_METRIC="$2"; shift 2;;
    --score-kind) SCORE_KIND="$2"; shift 2;;
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
KWARGS="$OUTDIR/oracle_kwargs.json"
G1_OUT="$OUTDIR/macag_game1.json"
G2_OUT="$OUTDIR/macag_game2.json"
BASELINES_OUT="$OUTDIR/macag_baselines.json"
MACAG_SLUG="macag-${SLUG}"
ANNOTATED="$OUTDIR/graphs/${MACAG_SLUG}.json"
mkdir -p "$OUTDIR/graphs"

CAND_ARG=()
[[ -n "$CANDIDATES_FILE" ]] && CAND_ARG=(--candidates-file "$CANDIDATES_FILE")

echo ">>> MACAG pipeline | model=$MODEL clt=$TRANSCODER_SET device=$DEVICE"
echo ">>> prompt: $PROMPT"
echo ">>> target=$TARGET  foil=$FOIL  -> $OUTDIR"

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

# --- 2) oracle kwargs (JSON written safely via python) ----------------------
echo ">>> [2/6] Writing oracle kwargs -> $KWARGS (freeze_attention=$FREEZE_ATTENTION)"
PROMPT="$PROMPT" TARGET="$TARGET" FOIL="$FOIL" MODEL="$MODEL" \
TRANSCODER_SET="$TRANSCODER_SET" GRAPH="$GRAPH" DEVICE="$DEVICE" DTYPE="$DTYPE" \
FREEZE_ATTENTION="$FREEZE_ATTENTION" SCORE_KIND="$SCORE_KIND" \
ABLATION_MODE="$ABLATION_MODE" CORRUPTED_PROMPT="$CORRUPTED_PROMPT" \
KWARGS="$KWARGS" python - <<'PY'
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
    corrupted = os.environ.get("CORRUPTED_PROMPT", "")
    if corrupted:
        kw["corrupted_prompt"] = corrupted
with open(os.environ["KWARGS"], "w") as f:
    json.dump(kw, f, indent=2)
print("wrote", os.environ["KWARGS"])
PY

ORACLE_FACTORY="macag.factories.replacement_model:create_replacement_model_oracle"

# --- 3) Game 1: minimal faithful set ---------------------------------------
G1_STOP_ARGS=()
if [[ "$FREEZE_MODE" != "both" ]]; then
  G1_STOP_ARGS=(--stop-metric "$STOP_METRIC")
fi
if [[ "$SKIP_GAME1" -eq 0 ]]; then
  echo ">>> [3/6] Game 1 (minimal faithful set) | freeze_mode=$FREEZE_MODE ${G1_STOP_ARGS[*]} ..."
  python -m macag.cli.run_macag game1 \
    --graph-json "$GRAPH" --target y --input-id "$SLUG" \
    --oracle-factory "$ORACLE_FACTORY" \
    --oracle-kwargs-file "$KWARGS" \
    "${CAND_ARG[@]}" \
    --prefilter-top-k "$PREFILTER_TOP_K" --budget "$BUDGET" \
    --alpha "$ALPHA" --lam "$LAM" \
    --faithfulness-eps "$EPS" \
    --freeze-mode "$FREEZE_MODE" \
    "${G1_STOP_ARGS[@]}" \
    --output-json "$G1_OUT"
else
  echo ">>> [3/6] Skipping Game 1; reusing $G1_OUT"
  [[ -f "$G1_OUT" ]] || { echo "ERROR: $G1_OUT not found"; exit 1; }
fi

# --- 4) Game 2: contrastive allocation (one run per solver) ----------------
echo ">>> [4/6] Game 2 (contrastive target vs foil) | solvers: $SOLVERS"
for solver in $SOLVERS; do
  g2_out="$OUTDIR/macag_game2_${solver}.json"
  if [[ "$SKIP_GAME2" -eq 1 && -f "$g2_out" ]]; then
    echo ">>>      solver=$solver -> $g2_out (reuse)"
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
  g2_cmd+=(
    --prefilter-top-k "$PREFILTER_TOP_K" --budget "$BUDGET"
    --beta "$BETA" --abr-iters "$ABR_ITERS"
    --solver "$solver" --fp-tol "$FP_TOL"
    --alpha "$ALPHA" --lam "$LAM"
    --output-json "$g2_out"
  )
  "${g2_cmd[@]}"
done
# Backward-compat: keep macag_game2.json as the canonical (ABR if present, else
# the first solver run) for the annotator and existing analyzers.
if [[ -f "$OUTDIR/macag_game2_abr.json" ]]; then
  cp "$OUTDIR/macag_game2_abr.json" "$G2_OUT"
else
  first_solver="${SOLVERS%% *}"
  cp "$OUTDIR/macag_game2_${first_solver}.json" "$G2_OUT"
fi

# --- 5) baseline head-to-head (same graph + oracle, matched budget) --------
if [[ "$SKIP_BASELINES" -eq 0 ]]; then
  echo ">>> [5/6] Baselines (methods=$BASELINE_METHODS, budget=$BUDGET, shapley_seed=$SHAPLEY_SEED) -> $BASELINES_OUT"
  # ACDC_TARGET_K: opt-in budget-matched ACDC (-1 = match --budget). Adds a
  # tau bisection on the warm cache (up to ~24 extra prune sweeps), so it is
  # off by default; enable for the paper head-to-head runs.
  acdc_target_args=()
  [[ -n "${ACDC_TARGET_K:-}" ]] && acdc_target_args=(--acdc-target-k "$ACDC_TARGET_K")
  python -m macag.cli.run_baselines \
    --graph-json "$GRAPH" --target y --input-id "$SLUG" \
    --oracle-factory "$ORACLE_FACTORY" \
    --oracle-kwargs-file "$KWARGS" \
    "${CAND_ARG[@]}" \
    --budget "$BUDGET" --alpha "$ALPHA" --lam "$LAM" \
    --prefilter-top-k "$PREFILTER_TOP_K" \
    --methods "$BASELINE_METHODS" \
    --shapley-permutations "$SHAPLEY_PERMUTATIONS" \
    --shapley-seed "$SHAPLEY_SEED" \
    "${acdc_target_args[@]}" \
    --output-json "$BASELINES_OUT"
else
  echo ">>> [5/6] Skipping baseline harness"
fi

# --- 6) annotate graph with both games' evidence -------------------------
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

if [[ "$KL_RESCORE" == "1" && "$SKIP_KL_RESCORE" -eq 0 ]]; then
  echo ">>> KL rescore (independent faithfulness metric) -> $OUTDIR/macag_kl_faithfulness.json"
  python -m macag.cli.rescore_kl --run-dir "$OUTDIR" --progress || \
    echo ">>> (KL rescore failed; logit-gap results are unchanged)"
fi

echo ">>> Done."
echo "    graph     : $GRAPH"
echo "    game1     : $G1_OUT"
for solver in $SOLVERS; do
  echo "    game2/$solver : $OUTDIR/macag_game2_${solver}.json"
done
echo "    game2     : $G2_OUT  (canonical copy for annotators/analyzers)"
[[ "$SKIP_BASELINES" -eq 0 ]] && echo "    baselines : $BASELINES_OUT"
echo "    annotated : $ANNOTATED  (select slug '$MACAG_SLUG' in the UI)"

# --- optional: serve --------------------------------------------------------
if [[ "$SERVE" -eq 1 ]]; then
  echo ">>> Serving $OUTDIR/graphs on port $PORT (Ctrl-C to stop) ..."
  python -m circuit_tracer start-server --graph_file_dir "$OUTDIR/graphs" --port "$PORT"
else
  echo ">>> To visualize:"
  echo "    python -m circuit_tracer start-server --graph_file_dir $OUTDIR/graphs --port $PORT"
fi
