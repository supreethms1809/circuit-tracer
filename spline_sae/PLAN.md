# Spline-SAE plan

## Phase 0 — Setup (this PR / directory)

- [x] Create `spline_sae/` isolated from paper CLT runner
- [x] Document Neuronpedia top-3 baselines
- [x] Add minimal `SplineSAE` + loss + 1-GPU train (`spline_sae/train.py`)
- [ ] Optional: install `sae-lens` and score against published Gemma Scope weights

## Phase 1 — Baseline jobs (submitted)

Launch: `bash scripts/slurm/launch_spline_sae_baselines.sh`  
Slurm: `scripts/slurm/spline_sae_run.sbatch` on **ai4wy-1 / 1 GPU**, 50k steps.

| Job file | Arm |
|---|---|
| `gemma2_2b_res_l12_linear.yaml` | Gemma-2-2B L12 linear JumpReLU w16k |
| `gemma2_2b_res_l12_kan.yaml` | Gemma-2-2B L12 KAN JumpReLU w16k |
| `llama31_8b_res_l16_linear.yaml` | Llama-3.1-8B L16 linear TopK w32k |
| `llama31_8b_res_l16_kan.yaml` | Llama-3.1-8B L16 KAN TopK w16k |
| `qwen25_7b_it_res_l20_linear.yaml` | Qwen2.5-7B-IT L20 linear JumpReLU 8× |
| `qwen25_7b_it_res_l20_kan.yaml` | Qwen2.5-7B-IT L20 KAN JumpReLU w16k |

Outputs: `/gscratch/ssuresh/results/spline_sae/<run>/` (`best.pt`, `train_metrics.jsonl`, `summary.json`).  
Job IDs: `spline_sae/baselines/jobids.txt`.

Exit: compare linear vs kan NMSE / L0 / `spline_contribution_frac` at matched recipes; later score vs published HF SAEs on a shared val set.

## Phase 2 — Analysis probes (submitted)

Launch: `bash scripts/slurm/launch_spline_sae_probes.sh`

| Job | Probe | Question |
|---|---|---|
| gap-eval | `eval_gap` on Phase-1 best.pt | Is `recon_gap = NMSE(base)−NMSE(full)` ≤0 today? |
| `probe_gemma_kan_l0match` | JumpReLU KAN + soft L0→51 | Idle spline or just undersparsity? |
| `probe_gemma_kan_b3_l0match` | L0 match + B3 + frac hinge | Can JumpReLU KAN be forced usefully? |
| `probe_llama_kan_b3` | TopK + B3 (`scale_base=0.2`, `lr_spline×5`) | Does more forcing help past frac=0.67? |
| `probe_llama_kan_freeze_base` | Freeze base @10k + frac hinge | After warmup, must splines carry recon? |

Outputs under `/gscratch/ssuresh/results/spline_sae/probe_*` and `gap_eval_phase1.json`.  
Job IDs: `spline_sae/baselines/probe_jobids.txt`.

## Phase 2 — Spline swap + diagnostics

For each model, same data / width / sparsifier hyperparameters:

1. Linear SAE (Phase 1)
2. Spline SAE (KAN encoder, `λ_kan_reg=0` initially)
3. Log `spline_contribution_frac`, `recon_gap`, L0, NMSE every N steps

Exit: either `gap > 0` at matched L0, or a clear failure mode (idle splines /
worse NMSE) documented with plots.

## Phase 3 — Loss / curriculum for useful nonlinearity

Only if Phase 2 shows idle or harmful splines (expected from CLT B3 lessons):

1. Staged `scale_base` anneal (vanilla → open splines)
2. Gap loss: `L -= λ_nl * ReLU(NMSE_base_only - NMSE_full)`
3. Retune JumpReLU θ / bandwidth on KAN preacts
4. Do **not** optimize contribution fraction alone

## Phase 4 — Scale + SAEBench / Neuronpedia

- Full(er) token budgets on the winning recipe
- Optional upload / dashboard for qualitative feature checks
- Port encoder + loss recipe into `spline_clt` for one CLT model

## Non-goals (for this directory)

- Cross-layer decode / MACAG / RAVEL graphs
- Matching full Gemma Scope multi-width suites
- 70B models
