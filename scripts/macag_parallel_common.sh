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
# Interrupt handling: workers are started in their own session (setsid) so
# Ctrl+C can kill the whole tree (shard -> pipeline -> python). Do not wrap
# workers in nohup; it leaves GPU children behind. To clean up orphans manually:
#   scripts/macag_kill_sweep.sh [OUTROOT]

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

  local shard_prefix="shard"
  local status_tag=""
  if [[ -n "$clt_tag" ]]; then
    shard_prefix="shard.${clt_tag}"
    status_tag="$clt_tag"
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

  for (( w = 0; w < nw; w++ )); do
    local out_log="$OUTROOT/${shard_prefix}.$w.out"
    echo ">>> launching ${clt_tag:-ALL} worker $w/$nw -> $out_log"
    local -a env_args=(
      NUM_WORKERS="$nw"
      WORKER_ID="$w"
      SHARD_TAG="$status_tag"
      JSON="$JSON"
      OUTROOT="$OUTROOT"
      FREEZE_MODE="$FREEZE_MODE"
      KL_RESCORE="${KL_RESCORE:-1}"
      RUN_ANALYSIS=0
    )
    [[ -n "$clt_tag" ]] && env_args+=(CLTS="$clt_tag")
    setsid env "${env_args[@]}" "$runner" > "$out_log" 2>&1 &
    pids+=("$!")
    if (( w < nw - 1 )); then sleep "${STAGGER:-45}"; fi
  done

  echo ">>> ${clt_tag:-ALL}: $nw shard(s) launched (pids: ${pids[*]}). Waiting ..."
  local pool_fail=0
  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      echo ">>> ${clt_tag:-ALL} worker $idx finished OK"
    else
      echo ">>> ${clt_tag:-ALL} worker $idx FAILED — see $OUTROOT/${shard_prefix}.$idx.out" >&2
      pool_fail=1
    fi
  done
  trap - INT TERM
  return "$pool_fail"
}
