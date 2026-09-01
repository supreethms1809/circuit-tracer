#!/usr/bin/env bash
# Launch candidate B3+L0 recipe on Gemma / Llama / Qwen (KAN + matched linear).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs/slurm /gscratch/ssuresh/results/spline_sae/candidate_b3_l0

# Reuse winning Gemma KAN probe as the candidate Gemma KAN run (identical recipe).
WIN=/gscratch/ssuresh/results/spline_sae/probe_gemma_kan_b3_l0match
DST=/gscratch/ssuresh/results/spline_sae/candidate_b3_l0/gemma2_2b_l12_kan
if [[ -f "${WIN}/best.pt" && ! -e "${DST}" ]]; then
  ln -s "${WIN}" "${DST}"
  echo "linked Gemma KAN candidate → ${DST} -> ${WIN}"
elif [[ -e "${DST}" ]]; then
  echo "Gemma KAN candidate already present: ${DST}"
else
  echo "WARNING: winning Gemma probe missing; will train from config" >&2
fi

LIST=(
  spline_sae/configs/candidate_b3_l0_gemma_linear.yaml
  spline_sae/configs/candidate_b3_l0_llama_kan.yaml
  spline_sae/configs/candidate_b3_l0_llama_linear.yaml
  spline_sae/configs/candidate_b3_l0_qwen_kan.yaml
  spline_sae/configs/candidate_b3_l0_qwen_linear.yaml
)

# Only train Gemma KAN if link failed
if [[ ! -e "${DST}/best.pt" ]]; then
  LIST=(spline_sae/configs/candidate_b3_l0_gemma_kan.yaml "${LIST[@]}")
fi

JOBS=()
for cfg in "${LIST[@]}"; do
  jid=$(sbatch --parsable scripts/slurm/spline_sae_run.sbatch "$cfg")
  echo "submitted ${jid}  ${cfg}"
  JOBS+=("$jid")
done

printf '%s\n' "${JOBS[@]}" > spline_sae/baselines/candidate_b3_l0_jobids.txt
printf '%s\n' "${LIST[@]}" > spline_sae/baselines/candidate_b3_l0_configs.txt
echo "Submitted ${#JOBS[@]} jobs (Gemma KAN reused from winning probe when available)"
