#!/usr/bin/env bash
# Submit the full R5 campaign (gpt2 main+dt_sweep, qwen linear+spline, llama spline).
set -euo pipefail

ROOT="/home/ssuresh/circuit-tracer"
cd "$ROOT"
mkdir -p logs/slurm

ONE="$ROOT/scripts/slurm/r5_1node.sbatch"
TWO="$ROOT/scripts/slurm/r5_2node.sbatch"

# GPT-2 small: 1 node each
GPT2_1NODE=(
  r5_gpt2_small_spline_feature_match
  r5_gpt2_small_spline_param_match
  r5_gpt2_small_spline_dt768
  r5_gpt2_small_spline_dt1536
  r5_gpt2_small_spline_dt3072
  r5_gpt2_small_spline_dt6144
  r5_gpt2_small_linear_feature_match
  r5_gpt2_small_linear_param_match
  r5_gpt2_small_linear_dt1664
  r5_gpt2_small_linear_dt3392
  r5_gpt2_small_linear_dt6784
  r5_gpt2_small_linear_dt13504
)

# Qwen / Llama: 2 nodes each
MULTI_2NODE=(
  r5_qwen3_06b_spline_pm
  r5_qwen3_06b_linear_d8192
  r5_llama32_1b_spline_dt8192
)

echo "=== R5b submit ==="
echo "output root: /gscratch/ssuresh/results/paper_r5b"
echo

for suite in "${GPT2_1NODE[@]}"; do
  jid=$(sbatch --parsable "$ONE" "$suite")
  echo "submitted $suite -> job $jid (r5-1node)"
done

for suite in "${MULTI_2NODE[@]}"; do
  jid=$(sbatch --parsable "$TWO" "$suite")
  echo "submitted $suite -> job $jid (r5-2node)"
done

echo
echo "=== queue ==="
squeue -u "$USER" -o '%.10i %.12j %.8T %.10M %R' | head -40
