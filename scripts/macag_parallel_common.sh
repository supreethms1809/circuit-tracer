# Shared per-CLT parallel launcher for MACAG benchmark sweeps.
# Source from run_macag_*_parallel.sh after setting RUNNER_SCRIPT.
#
# Default worker counts (override per tag):
#   WORKERS_GEMMA2_426K=8   smaller Gemma CLT
#   WORKERS_GEMMA2_2_5M=2   larger Gemma CLT
#   WORKERS_LLAMA32_524K=2  Llama CLT
#
# CLT groups run sequentially (never 8+2+2 concurrent on one GPU). Within each
# group, workers round-robin that CLT's prompts only (via CLTS=<tag>).
#
# Set NUM_WORKERS to force the same count for every CLT (legacy global mode).
#
# Multi-GPU: workers within a CLT group are round-robined across the GPUs
# visible to this script (CUDA_VISIBLE_DEVICES if set, else all GPUs reported
# by nvidia-smi, else a single device). Set NUM_GPUS to override the count
# explicitly (assumes device indices 0..NUM_GPUS-1). Each worker gets its own
# CUDA_VISIBLE_DEVICES so "cuda" inside that worker maps to exactly one
# physical GPU. On a single-GPU host this degenerates to the old behavior.
#
# GPU memory is NOT shared across worker processes — each shard loads its own
# copy of the base LM + CLT. nvidia-smi per-process MiB often sums to less than
# total used-memory (driver pools, fragmentation, freed-but-unreclaimed caches).
# MAX_WORKERS_PER_GPU caps how many shards run GPU work on the same device at
# once (slot semaphore in run_macag_mib.sh); extra shards wait rather than OOM.
# Set MAX_WORKERS_PER_GPU=0 to disable the cap (legacy: all shards run together).
#
# Interrupt handling: workers are started in their own session (setsid) so
# Ctrl+C can kill the whole tree (shard -> pipeline -> python). Do not wrap
# workers in nohup; it leaves GPU children behind. To clean up orphans manually:
#   scripts/macag_kill_sweep.sh [OUTROOT]

macag_gpu_list() {
  if [[ -n "${NUM_GPUS:-}" ]]; then
    seq 0 $(( NUM_GPUS - 1 ))
    return
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES"
    return
  fi
  # nvidia-smi prints its "couldn't communicate with the NVIDIA driver" error
  # to STDOUT (not stderr) with a nonzero exit code when no GPU/driver is
  # present, so gate on the exit status + non-empty output, not line count.
  local out
  if command -v nvidia-smi >/dev/null 2>&1 \
      && out="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)" \
      && [[ -n "$out" ]]; then
    printf '%s\n' "$out"
    return
  fi
  echo 0
}

macag_kill_tree() {
  local pid="$1"
  local sig="${2:-TERM}"
  [[ -z "$pid" || "$pid" -le 1 ]] && return 0
  local child
  while IFS= read -r child; do
    [[ -n "$child" ]] && macag_kill_tree "$child" "$sig"
  done < <(pgrep -P "$pid" 2>/dev/null || true)
  kill -"$sig" "$pid" 2>/dev/null || true
}

macag_kill_shard() {
  local pid="$1"
  local sig="${2:-TERM}"
  # setsid workers: negative PID kills the whole session/process group.
  kill -"$sig" -- -"$pid" 2>/dev/null || macag_kill_tree "$pid" "$sig"
}

macag_kill_macag_procs() {
  local user="${USER:-$(whoami)}"
  local outroot="${1:-}"
  local -a patterns=(
    "scripts/run_macag_mib"
    "scripts/run_macag_acdc"
    "scripts/run_macag_pipeline"
    "python -m macag.cli"
  )
  local pat
  for pat in "${patterns[@]}"; do
    if [[ -n "$outroot" ]]; then
      pkill -u "$user" -TERM -f "${pat}.*${outroot}" 2>/dev/null || true
    else
      pkill -u "$user" -TERM -f "$pat" 2>/dev/null || true
    fi
  done
  sleep 2
  for pat in "${patterns[@]}"; do
    if [[ -n "$outroot" ]]; then
      pkill -u "$user" -KILL -f "${pat}.*${outroot}" 2>/dev/null || true
    else
      pkill -u "$user" -KILL -f "$pat" 2>/dev/null || true
    fi
  done
}

macag_max_workers_per_gpu() {
  # Empirical anchor: 8 concurrent gemma2-426k shards on one 96GB GPU.
  local cap="${MAX_WORKERS_PER_GPU:-8}"
  if [[ "$cap" == "0" ]]; then
    echo 0
    return
  fi
  if ! [[ "$cap" =~ ^[0-9]+$ ]] || (( cap < 1 )); then
    echo "ERROR: MAX_WORKERS_PER_GPU must be a positive integer or 0 (got '$cap')" >&2
    return 1
  fi
  echo "$cap"
}

# Acquire a per-GPU slot before launching GPU-heavy work. Returns the slot path
# on stdout; caller must pass it to macag_release_gpu_slot on exit.
# Slot dirs are node-scoped so multi-node array jobs sharing RESULTS_ROOT do not
# contend on the same mkdir locks (each node remaps physical GPUs to 0..N-1).
macag_acquire_gpu_slot() {
  local outroot="${1:-${OUTROOT:-}}"
  local gpu="${CUDA_VISIBLE_DEVICES:-0}"
  local max
  max="$(macag_max_workers_per_gpu)" || return 1
  if (( max == 0 )); then
    echo ""
    return 0
  fi
  local node="${SLURMD_NODENAME:-${NODE_RANK:-local}}"
  local lockdir="$outroot/.gpu_slots/${node}/gpu${gpu}"
  mkdir -p "$lockdir"
  local waited=0
  local poll="${GPU_SLOT_POLL_SEC:-10}"
  while true; do
    local slot
    for (( slot = 0; slot < max; slot++ )); do
      local path="$lockdir/slot_${slot}"
      if mkdir "$path" 2>/dev/null; then
        if (( waited > 0 )); then
          echo ">>> gpu=$gpu acquired slot $slot after ${waited}s wait" >&2
        fi
        echo "$path"
        return 0
      fi
    done
    if (( waited == 0 )); then
      echo ">>> gpu=$gpu saturated ($max slots); waiting for a free slot ..." >&2
    fi
    sleep "$poll"
    waited=$(( waited + poll ))
  done
}

macag_release_gpu_slot() {
  local slot_path="${1:-}"
  [[ -z "$slot_path" ]] && return 0
  rmdir "$slot_path" 2>/dev/null || true
}

# Slot locks are mkdir-based and only released by macag_release_gpu_slot on a
# clean exit. A run killed outright -- SLURM time limit (SIGKILL after KillWait),
# scancel, node failure -- leaves its slot dirs behind. The leak accumulates
# across restarts: each killed run strands up to MAX_WORKERS_PER_GPU dirs, and
# once a GPU's slots are all stranded every worker blocks forever in
# macag_acquire_gpu_slot waiting on a holder that no longer exists.
#
# Multi-node array jobs share RESULTS_ROOT but use node-scoped slot dirs
# (.gpu_slots/<nodename>/...). Only clear *this* node's slots at campaign
# start so we do not steal locks from siblings still running.
macag_clear_stale_gpu_slots() {
  local outroot="${1:-${RESULTS_ROOT:-}}"
  [[ -z "$outroot" || ! -d "$outroot" ]] && return 0
  local node="${SLURMD_NODENAME:-${NODE_RANK:-local}}"
  local n=0 slot
  local scope="$outroot/.gpu_slots/${node}"
  if [[ -d "$scope" ]]; then
    while IFS= read -r slot; do
      rmdir "$slot" 2>/dev/null && n=$(( n + 1 ))
    done < <(find "$scope" -type d -path '*/gpu*/slot_*' 2>/dev/null)
  fi
  # Legacy unscoped slots (pre-multi-node): clear only when single-node.
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" && -z "${NODE_RANK:-}" ]]; then
    while IFS= read -r slot; do
      rmdir "$slot" 2>/dev/null && n=$(( n + 1 ))
    done < <(find "$outroot" -type d -path '*/.gpu_slots/gpu*/slot_*' 2>/dev/null)
  fi
  (( n > 0 )) && echo ">>> released $n stale GPU slot lock(s) under $outroot (node=$node; leaked by a killed run)"
  return 0
}

macag_workers_for_clt() {
  local tag="$1"
  if [[ -n "${NUM_WORKERS:-}" ]]; then
    echo "$NUM_WORKERS"
    return
  fi
  case "$tag" in
    gemma2-426k)  echo "${WORKERS_GEMMA2_426K:-8}" ;;
    gemma2-2.5M)  echo "${WORKERS_GEMMA2_2_5M:-2}" ;;
    llama32-524k) echo "${WORKERS_LLAMA32_524K:-2}" ;;
    *) echo "ERROR: unknown CLT tag '$tag' for worker lookup" >&2; return 1 ;;
  esac
}

macag_run_parallel_sweep() {
  local runner="$1"
  local fail=0
  local clt_tags=(gemma2-426k gemma2-2.5M llama32-524k)

  if [[ -z "$runner" || ! -f "$runner" ]]; then
    echo "ERROR: RUNNER_SCRIPT must point to an existing sweep script" >&2
    return 2
  fi

  mkdir -p "$OUTROOT"

  if [[ -n "${NUM_WORKERS:-}" ]]; then
    echo ">>> Parallel MACAG | legacy NUM_WORKERS=$NUM_WORKERS (all CLTs share one pool)"
    echo ">>> Tip: unset NUM_WORKERS to use per-CLT defaults (426k=8, 2.5M/llama=2)"
    macag_launch_worker_pool "$runner" "" "$NUM_WORKERS" || fail=1
  else
    echo ">>> Parallel MACAG | per-CLT workers: 426k=${WORKERS_GEMMA2_426K:-8} 2.5M=${WORKERS_GEMMA2_2_5M:-2} llama=${WORKERS_LLAMA32_524K:-2}"
    for tag in "${clt_tags[@]}"; do
      if [[ -n "${CLTS:-}" && " $CLTS " != *" $tag "* ]]; then
        continue
      fi
      local nw
      nw="$(macag_workers_for_clt "$tag")" || return 1
      echo ""; echo ">>> ===== CLT group $tag | workers=$nw ====="
      macag_launch_worker_pool "$runner" "$tag" "$nw" || fail=1
    done
  fi

  echo ""; echo ">>> Combined status across shards:"
  cat "$OUTROOT"/status.*.w*.txt 2>/dev/null | sort | uniq -c | sort -rn | head

  if [[ "$fail" -ne 0 ]]; then
    echo ">>> WARNING: at least one shard failed; aggregating available results anyway." >&2
  fi

  if [[ "${RUN_ANALYSIS:-1}" != "0" ]]; then
    echo ""; echo ">>> Aggregating (ANALYZE_ONLY) ..."
    ANALYZE_ONLY=1 RUN_ANALYSIS=1 KL_RESCORE="${KL_RESCORE:-1}" \
    JSON="$JSON" OUTROOT="$OUTROOT" FREEZE_MODE="$FREEZE_MODE" \
      "$runner"
  fi

  echo ""; echo ">>> Parallel sweep complete. Output under $OUTROOT/"
  return "$fail"
}

macag_launch_worker_pool() {
  local runner="$1"
  local clt_tag="$2"
  local nw="$3"

  if ! [[ "$nw" =~ ^[0-9]+$ ]] || (( nw < 1 )); then
    echo "ERROR: worker count must be a positive integer (got '$nw' for ${clt_tag:-ALL})" >&2
    return 2
  fi

  # Multi-node array: this node launches `nw` local shards whose global IDs are
  # NODE_RANK * nw + local_id, with NUM_WORKERS=TOTAL_WORKERS for prompt modulo.
  # Single-node default: TOTAL_WORKERS unset → global == local (unchanged).
  local node_rank="${NODE_RANK:-0}"
  local total_workers="${TOTAL_WORKERS:-$nw}"
  if ! [[ "$node_rank" =~ ^[0-9]+$ ]]; then
    echo "ERROR: NODE_RANK must be a non-negative integer (got '$node_rank')" >&2
    return 2
  fi
  if ! [[ "$total_workers" =~ ^[0-9]+$ ]] || (( total_workers < 1 )); then
    echo "ERROR: TOTAL_WORKERS must be a positive integer (got '$total_workers')" >&2
    return 2
  fi
  local worker_offset=$(( node_rank * nw ))
  if (( worker_offset + nw > total_workers )); then
    echo "ERROR: NODE_RANK=$node_rank with $nw local workers exceeds TOTAL_WORKERS=$total_workers" >&2
    return 2
  fi
  if (( total_workers != nw )); then
    echo ">>> multi-node shard: NODE_RANK=$node_rank local_workers=$nw TOTAL_WORKERS=$total_workers offset=$worker_offset"
  fi

  local shard_prefix="shard"
  local status_tag=""
  if [[ -n "$clt_tag" ]]; then
    shard_prefix="shard.${clt_tag}"
    status_tag="$clt_tag"
  fi

  local -a gpu_list
  mapfile -t gpu_list < <(macag_gpu_list)
  local ngpu="${#gpu_list[@]}"
  (( ngpu < 1 )) && ngpu=1

  local max_per_gpu=0
  if ! max_per_gpu="$(macag_max_workers_per_gpu)"; then
    return 1
  fi
  if (( max_per_gpu > 0 )); then
    local max_total=$(( ngpu * max_per_gpu ))
    if (( nw > max_total )); then
      echo ">>> NOTE: $nw shard(s) on $ngpu GPU(s) with MAX_WORKERS_PER_GPU=$max_per_gpu"
      echo ">>>       -> at most $max_per_gpu concurrent per GPU ($max_total total);"
      echo ">>>       extra shards wait on per-GPU slots (see run_macag_mib.sh)."
    else
      local per_gpu=$(( (nw + ngpu - 1) / ngpu ))
      if (( per_gpu > max_per_gpu )); then
        echo ">>> NOTE: ~$per_gpu shard(s)/GPU exceeds MAX_WORKERS_PER_GPU=$max_per_gpu;"
        echo ">>>       shards will queue on per-GPU slots during GPU-heavy stages."
      fi
    fi
  fi

  local pids=()
  cleanup() {
    echo ">>> Interrupted — terminating ${#pids[@]} shard(s) (${clt_tag:-ALL}) ..."
    for pid in "${pids[@]}"; do macag_kill_shard "$pid" TERM; done
    sleep 3
    for pid in "${pids[@]}"; do macag_kill_shard "$pid" KILL; done
    macag_kill_macag_procs "$OUTROOT"
  }
  trap cleanup INT TERM

  # Launch one worker per GPU at a time (a "round"), then pause once per
  # round instead of once per worker -- e.g. with 2 GPUs: gpu0+gpu1 launch
  # back-to-back, pause STAGGER, next gpu0+gpu1 launch back-to-back, pause,
  # ... This keeps simultaneous model-loading spikes bounded (still <= ngpu
  # workers starting up at once) while cutting the total ramp-up time by
  # ~ngpu versus staggering every single worker individually.
  for (( round_start = 0; round_start < nw; round_start += ngpu )); do
    local round_end=$(( round_start + ngpu - 1 ))
    (( round_end >= nw )) && round_end=$(( nw - 1 ))
    for (( w = round_start; w <= round_end; w++ )); do
      local global_w=$(( worker_offset + w ))
      local gpu_id="${gpu_list[$(( w % ngpu ))]}"
      local out_log="$OUTROOT/${shard_prefix}.$global_w.out"
      echo ">>> launching ${clt_tag:-ALL} worker $global_w/$total_workers (local $w/$nw) -> gpu=$gpu_id -> $out_log"
      local -a env_args=(
        PATH="${PATH}"
        CONDA_PREFIX="${CONDA_PREFIX:-}"
        CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
        CONDA_PYTHON_EXE="${CONDA_PYTHON_EXE:-}"
        VIRTUAL_ENV="${VIRTUAL_ENV:-}"
        NUM_WORKERS="$total_workers"
        WORKER_ID="$global_w"
        NODE_RANK="$node_rank"
        TOTAL_WORKERS="$total_workers"
        SLURMD_NODENAME="${SLURMD_NODENAME:-}"
        SHARD_TAG="$status_tag"
        JSON="$JSON"
        OUTROOT="$OUTROOT"
        FREEZE_MODE="$FREEZE_MODE"
        KL_RESCORE="${KL_RESCORE:-1}"
        SCORE_KINDS="${SCORE_KINDS:-logit_gap}"
        GAME2_SCORE_KIND="${GAME2_SCORE_KIND:-}"
        CONNECTED="${CONNECTED:-0}"
        PREFILTER_TOP_K="${PREFILTER_TOP_K:-}"
        GAME2_PREFILTER_TOP_K="${GAME2_PREFILTER_TOP_K:-}"
        BASELINE_METHODS="${BASELINE_METHODS:-}"
        SOLVERS="${SOLVERS-abr fp}"
        SKIP_GAME1="${SKIP_GAME1:-0}"
        SKIP_BASELINES="${SKIP_BASELINES:-0}"
        TASKS="${TASKS:-}"
        LIMIT="${LIMIT:-0}"
        SKIP_TASKS="${SKIP_TASKS:-}"
        NSYS_PROFILE="${NSYS_PROFILE:-0}"
        NSYS_BIN="${NSYS_BIN:-}"
        NSYS_OUTDIR="${NSYS_OUTDIR:-}"
        NSYS_DELAY="${NSYS_DELAY:-}"
        NSYS_DURATION="${NSYS_DURATION:-}"
        NSYS_WORKER_IDS="${NSYS_WORKER_IDS:-}"
        MACAG_NVTX="${MACAG_NVTX:-}"
        PARALLEL_AGENTS="${PARALLEL_AGENTS:-1}"
        RUN_ANALYSIS=0
        CUDA_VISIBLE_DEVICES="$gpu_id"
      )
      [[ -n "$clt_tag" ]] && env_args+=(CLTS="$clt_tag")
      setsid env "${env_args[@]}" "$runner" > "$out_log" 2>&1 &
      pids+=("$!")
    done
    if (( round_end < nw - 1 )); then sleep "${STAGGER:-45}"; fi
  done

  echo ">>> ${clt_tag:-ALL}: $nw local shard(s) launched (pids: ${pids[*]}). Waiting ..."
  local pool_fail=0
  for idx in "${!pids[@]}"; do
    local global_idx=$(( worker_offset + idx ))
    if wait "${pids[$idx]}"; then
      echo ">>> ${clt_tag:-ALL} worker $global_idx finished OK"
    else
      echo ">>> ${clt_tag:-ALL} worker $global_idx FAILED — see $OUTROOT/${shard_prefix}.$global_idx.out" >&2
      pool_fail=1
    fi
  done
  trap - INT TERM
  return "$pool_fail"
}
