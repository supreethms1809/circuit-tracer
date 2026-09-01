# Candidate recipe: B3 + L0-match (JumpReLU)

Winner from `probe_gemma_kan_b3_l0match` (Gemma-2-2B L12, 50k steps):
best NMSE **0.244** vs linear **0.239**, L0≈51, frac≈0.38, recon_gap≈0.06.

## Locked knobs

| Knob | Value |
|---|---|
| `activation` | `jumprelu` |
| `d_sae` | 16384 |
| `scale_base` | **0.2** |
| `scale_spline` | 1.0 |
| `lr_spline_mult` | **5.0** |
| `lambda_sparsity` (init) | 4e-4 |
| `target_l0` | **51** (soft adapt every 200 steps) |
| `lambda_frac_hinge` | 0.05 |
| `frac_target` | 0.3 |
| `learning_rate` | 3e-4 |
| `n_steps` | 50000 |
| `dataset` | `monology/pile-uncopyrighted` (streaming) |
| `seed` | 101 |

Linear controls use the same JumpReLU / width / L0-adapt / λ init (no B3 / hinge).

## Campaign layout

```text
/gscratch/ssuresh/results/spline_sae/candidate_b3_l0/
  gemma2_2b_l12_kan/      # symlink to winning probe
  gemma2_2b_l12_linear/
  llama31_8b_l16_kan/
  llama31_8b_l16_linear/
  qwen25_7b_it_l20_kan/
  qwen25_7b_it_l20_linear/
```

Launch: `bash scripts/slurm/launch_spline_sae_candidate.sh`
