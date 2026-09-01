#!/usr/bin/env bash
# Submit R5b-B3 spline campaign (Tier-1 B3 routing on all r5 spline arms).
# Output: /gscratch/ssuresh/results/paper_r5b_b3
set -euo pipefail

ROOT="/home/ssuresh/circuit-tracer"
cd "$ROOT"
mkdir -p logs/slurm "$ROOT/../.." 2>/dev/null || true
mkdir -p /gscratch/ssuresh/results/paper_r5b_b3 || true

ONE="$ROOT/scripts/slurm/r5b_b3_1node.sbatch"
TWO="$ROOT/scripts/slurm/r5b_b3_2node.sbatch"

GPT2_1NODE=(
  r5b_b3_gpt2_small_spline_feature_match
  r5b_b3_gpt2_small_spline_param_match
  r5b_b3_gpt2_small_spline_dt768
  r5b_b3_gpt2_small_spline_dt1536
  r5b_b3_gpt2_small_spline_dt3072
  r5b_b3_gpt2_small_spline_dt6144
)

MULTI_2NODE=(
  r5b_b3_qwen3_06b_spline_pm
  r5b_b3_llama32_1b_spline_dt8192
)

echo "=== R5b-B3 spline submit ==="
echo "output root: /gscratch/ssuresh/results/paper_r5b_b3"
echo "B3 knobs: scale_base=0.2  lr_spline_mult=5  lambda_kan_reg=0"
echo

for suite in "${GPT2_1NODE[@]}"; do
  jid=$(sbatch --parsable --job-name="b3-${suite#r5b_b3_}" "$ONE" "$suite")
  echo "submitted $suite -> job $jid (r5b-b3-1node)"
done

for suite in "${MULTI_2NODE[@]}"; do
  jid=$(sbatch --parsable --job-name="b3-${suite#r5b_b3_}" "$TWO" "$suite")
  echo "submitted $suite -> job $jid (r5b-b3-2node)"
done

echo
echo "=== queue (b3) ==="
squeue -u "$USER" -o '%.10i %.28j %.8T %.10M %R' | rg -e 'JOBID|b3-' | head -40
