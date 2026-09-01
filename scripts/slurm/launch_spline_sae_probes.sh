#!/usr/bin/env bash
# Launch Phase-2 Spline-SAE analysis probes + gap eval on Phase-1 checkpoints.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs/slurm /gscratch/ssuresh/results/spline_sae

JOBS=()

jid=$(sbatch --parsable scripts/slurm/spline_sae_gap_eval.sbatch)
echo "submitted ${jid}  gap-eval"
JOBS+=("$jid")

for cfg in \
  spline_sae/configs/probe_gemma_kan_l0match.yaml \
  spline_sae/configs/probe_gemma_kan_b3_l0match.yaml \
  spline_sae/configs/probe_llama_kan_b3.yaml \
  spline_sae/configs/probe_llama_kan_freeze_base.yaml
do
  jid=$(sbatch --parsable scripts/slurm/spline_sae_run.sbatch "$cfg")
  echo "submitted ${jid}  ${cfg}"
  JOBS+=("$jid")
done

printf '%s\n' "${JOBS[@]}" > spline_sae/baselines/probe_jobids.txt
echo "Submitted ${#JOBS[@]} probe jobs"
