#!/usr/bin/env bash
# Launch Phase-1 Spline-SAE baselines (linear + kan) for the three Neuronpedia models.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs/slurm /gscratch/ssuresh/results/spline_sae

LIST=(
  spline_sae/configs/gemma2_2b_res_l12_linear.yaml
  spline_sae/configs/gemma2_2b_res_l12_kan.yaml
  spline_sae/configs/llama31_8b_res_l16_linear.yaml
  spline_sae/configs/llama31_8b_res_l16_kan.yaml
  spline_sae/configs/qwen25_7b_it_res_l20_linear.yaml
  spline_sae/configs/qwen25_7b_it_res_l20_kan.yaml
)

JOBS=()
for cfg in "${LIST[@]}"; do
  jid=$(sbatch --parsable scripts/slurm/spline_sae_run.sbatch "$cfg")
  echo "submitted ${jid}  ${cfg}"
  JOBS+=("$jid")
done

printf '%s\n' "${JOBS[@]}" > spline_sae/baselines/jobids.txt
printf '%s\n' "${LIST[@]}" > spline_sae/baselines/launched_configs.txt
echo "Submitted ${#JOBS[@]} jobs → spline_sae/baselines/jobids.txt"
