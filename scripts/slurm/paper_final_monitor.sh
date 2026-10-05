#!/usr/bin/env bash
# Paper-campaign monitor. Slurm COMPLETED is not success: the last run exited 0
# after every prompt logged FAIL. This script counts artifacts and scans logs.
#
#   scripts/slurm/paper_final_monitor.sh
#   OUTROOT=/gscratch/$USER/paper_final/macag scripts/slurm/paper_final_monitor.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTROOT="${OUTROOT:-/gscratch/${USER}/paper_final/macag}"
LOG_DIR="$REPO_ROOT/logs/slurm"
REPORT_DIR="$REPO_ROOT/logs/paper_final_monitor"
mkdir -p "$REPORT_DIR"

ts="$(date '+%Y-%m-%d %H:%M:%S %Z')"
report="$REPORT_DIR/latest.txt"
alert=0

{
  echo "===== paper_final monitor @ $ts ====="
  echo "out=$OUTROOT"
  echo

  echo "===== Slurm ====="
  if ! squeue -u "$USER" -o "%.18i %.22j %.9T %.11M %R" | grep -E 'JOBID|paper_final' ; then
    echo "(no paper_final jobs in queue)"
  fi
  echo
  echo "===== recent paper_final accounting ====="
  sacct -S "$(date -d '2 days ago' '+%Y-%m-%d')" -u "$USER" \
    --format=JobID,JobName,State,Elapsed,ExitCode -P \
    | awk -F'|' 'NR==1 || ($2 ~ /^paper_final/ && $1 !~ /\.(batch|extern)$/)'
  echo

  echo "===== artifacts ====="
  if [[ ! -d "$OUTROOT" ]]; then
    echo "MISSING $OUTROOT"
    alert=1
  else
    for clt in gemma2-426k gemma2-2.5M llama32-524k; do
      root="$OUTROOT/$clt"
      if [[ ! -d "$root" ]]; then
        echo "  $clt: no prompt dirs yet"
        continue
      fi
      graphs=$(find "$root" -path '*/graphs/mib_*.json' 2>/dev/null | wc -l)
      g1=$(find "$root" -name 'macag_game1.json' 2>/dev/null | wc -l)
      g2=$(find "$root" -name 'macag_game2_*.json' 2>/dev/null | wc -l)
      bl=$(find "$root" -name 'macag_baselines.json' 2>/dev/null | wc -l)
      echo "  $clt: graphs=$graphs/75 game1=$g1/75 game2_files=$g2 baselines=$bl/75"
    done
  fi
  echo

  echo "===== worker status lines ====="
  if compgen -G "$OUTROOT/status.*.txt" >/dev/null; then
    awk '
      /^FAIL / { fail[$0]++ }
      /^OK / { ok++ }
      /^SKIP-done / { skip++ }
      END {
        printf "  FAIL=%d OK=%d SKIP-done=%d\n", length(fail), ok+0, skip+0
      }
    ' "$OUTROOT"/status.*.txt
    echo "  FAIL lines:"
    grep -h '^FAIL ' "$OUTROOT"/status.*.txt | sort | uniq -c | sort -nr | head -20
    if grep -q '^FAIL ' "$OUTROOT"/status.*.txt; then
      alert=1
    fi
  else
    echo "  (no status files yet)"
  fi
  echo

  echo "===== script / runtime errors in prompt logs ====="
  if compgen -G "$OUTROOT/"'*.log' >/dev/null; then
    hits=$(grep -E -n -H 'unbound variable|Traceback|CUDA out of memory|OutOfMemoryError|Killed' "$OUTROOT"/*.log 2>/dev/null | grep -v '/shard\.' || true)
    if [[ -n "$hits" ]]; then
      echo "$hits" | head -40
      alert=1
    else
      echo "  none"
    fi
  else
    echo "  (no prompt logs yet)"
  fi
  echo

  echo "===== live prompt stage (newest logs) ====="
  if compgen -G "$OUTROOT/"'*.log' >/dev/null; then
    # shellcheck disable=SC2012
    for f in $(ls -t "$OUTROOT"/*.log 2>/dev/null | grep -v '/shard\.' | head -6); do
      stage=$(grep -E 'Phase |>>> \[[0-9]/6\]|Feature influence computation:|unbound variable|FAIL ' "$f" | tail -1)
      echo "  $(basename "$f"): ${stage:-<no stage line>}"
    done
  fi
  echo

  if [[ "$alert" -eq 1 ]]; then
    echo "ALERT: a prompt failed or a script error is in the logs. Slurm state can still be RUNNING or COMPLETED."
  else
    echo "OK: no FAIL status lines and no script/OOM errors in prompt logs."
  fi
} | tee "$report"

exit "$alert"
