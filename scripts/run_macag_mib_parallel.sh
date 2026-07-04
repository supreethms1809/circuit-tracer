#!/usr/bin/env bash
# Parallel launcher for MIB-bench MACAG sweeps.
#
# Same per-CLT worker defaults as run_macag_acdc_parallel.sh:
#   gemma2-426k   8 workers
#   gemma2-2.5M   2 workers
#   llama32-524k  2 workers  (no MIB prompts today unless JSON extended)
#
# Usage:
#   scripts/run_macag_mib_parallel.sh
#   CLTS=gemma2-426k WORKERS_GEMMA2_426K=6 scripts/run_macag_mib_parallel.sh
#   NUM_WORKERS=3 scripts/run_macag_mib_parallel.sh   # legacy global pool
#   KL_RESCORE=0 scripts/run_macag_mib_parallel.sh    # skip KL faithfulness pass
#   scripts/macag_kill_sweep.sh results/macag_mib       # kill orphaned GPU workers after Ctrl+C
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STAGGER="${STAGGER:-45}"
JSON="${JSON:-macag/data/mib_benchmark_prompts.json}"
OUTROOT="${OUTROOT:-results/macag_mib}"
FREEZE_MODE="${FREEZE_MODE:-both}"

WORKERS_GEMMA2_426K="${WORKERS_GEMMA2_426K:-8}"
WORKERS_GEMMA2_2_5M="${WORKERS_GEMMA2_2_5M:-3}"
WORKERS_LLAMA32_524K="${WORKERS_LLAMA32_524K:-4}"

RUNNER_SCRIPT="scripts/run_macag_mib.sh"
# shellcheck source=scripts/macag_parallel_common.sh
source scripts/macag_parallel_common.sh

macag_run_parallel_sweep "$RUNNER_SCRIPT"
exit $?
