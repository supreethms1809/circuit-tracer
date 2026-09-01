# R5 / R5b training recipes (2026-07-30)

Target operating point: **act_lp ≈ 32**. Oversparsity (act ≲ 5–6) hurts eval.
Compare spline vs linear at **matched λ** on the same base model (and report L0).

## R5b campaign (current)

Revised after r5 trajectories: GPT decay was too late; GPT linear peak `0.005` and
qwen/llama peak `1e-5` were too strong.

- Suites: `experiments/paper_configs_v3/launch_suites/r5_*.json` (15 runs, suite_name `paper_r5b_*`)
- Output: `/gscratch/ssuresh/results/paper_r5b`
- Submit: `bash scripts/slurm/launch_r5_all.sh` (job names `r5-1node` / `r5-2node`)
- Prior r5 outputs kept under `/gscratch/ssuresh/results/paper_r5` for reference

## GPT-2 spline

| Knob | Value |
|---|---|
| `decoder_init_strategy` | **`kaiming`** |
| `jumprelu_bandwidth` | **0.12** |
| `lambda_sparsity` | peak **`1e-5`** |
| λ schedule | warmup **2000**, decay start **4500**, floor **`3e-6`** |

## GPT-2 linear (matched λ to GPT spline)

| Knob | Value |
|---|---|
| `decoder_init_strategy` | **`data_scaled`** |
| `jumprelu_bandwidth` | **calibrated** (`≈ 0.1 × θ_init`) |
| λ schedule | **same as GPT spline**: peak **`1e-5`**, warmup 2000, decay **4500**, floor **`3e-6`** |

## Qwen (spline + linear)

| Knob | Spline | Linear |
|---|---|---|
| `decoder_init_strategy` | **`kaiming`** | **`data_scaled`** |
| `jumprelu_bandwidth` | **0.14** | **calibrated** |
| `lambda_sparsity` | peak **`1e-6`** | peak **`1e-6`** (matched) |
| λ schedule | warmup **3000**, decay **4000**, floor **`3e-7`** | same |

**Note (2026-07-30):** qwen spline + `data_scaled` collapses act→~3 by step ~600 (identical with bw 0.09/0.12/0.14). Kaiming (job 9608) dipped then recovered to act~40–70. Relaunched as r5b kaiming (cleared checkpoints). Linear stays data_scaled for now (init differs across arms).

## Llama spline

| Knob | Value |
|---|---|
| `decoder_init_strategy` | **`data_scaled`** |
| `jumprelu_bandwidth` | **0.18** |
| `lambda_sparsity` | peak **`1e-6`** |
| λ schedule | warmup **2000**, decay start **3000**, floor **`3e-7`** |

## Gemma-2-2B spline — **current**

Two nested-FSDP arms share the BaseJump + B3 + r5b λ/bw recipe:
`sbatch scripts/slurm/r5b_gemma2_2node.sbatch …` (dt6144) or
`sbatch scripts/slurm/r5b_gemma2_4node.sbatch …` (dt10112).

| Arm | Suite | `d_transcoder` | Launch | vs published linear `d_t=16384` |
|---|---|---:|---|---|
| Memory-reduced | `r5b_b3_gemma2_2b_spline_dt6144_basejump` | **6144** | `r5b_gemma2_2node.sbatch` (2×2, `batch_size=16`) | ~61% params |
| **Param-match** | `r5b_b3_gemma2_2b_spline_dt10112_basejump` | **10112** | `r5b_gemma2_4node.sbatch` (4×2, `batch_size=4`, `total_steps=48828`) | ~100% params |

`dt6144` keeps **global batch 64** / `total_steps=24414` (~200M). `dt10112` on **4×2 × batch_size=4** (global batch **32**, `total_steps=48828` ≈ **200M**); mid-run FSDP released before isolated base-LM collect.

| Knob | Value |
|---|---|
| nodes × GPUs | **dt6144: 2×2** (`batch_size=16`); **dt10112: 4×2** (`batch_size=4`); `fsdp_cpu_offload=false`, **`shard_kan_encoders=true`** |
| `activation_function` | **`base_jump`** (gate on KAN base; mag = relu(base+spline)) |
| B3 | **`scale_base=0.2`**, **`lr_spline_mult=5`**, **`lambda_kan_reg=0`** |
| `decoder_init_strategy` | **`data_scaled`** |
| `jumprelu_bandwidth` | **0.10** |
| `threshold_init_target_l0` | **24** (was 32; hub banked L0/layer/tok ≈ **12.6**) |
| `lambda_sparsity` | peak **`1e-3`** (hub_l0_v3 retune; `1e-4` attempt still climbed) |
| λ schedule | warmup **4000**, **hold peak** (`sparsity_decay_start=0`, `lambda_sparsity_final=1e-3`) until act near hub |
| tokens / steps | **dt6144:** 200M / 24414 (global 64); **dt10112:** 200M / 48828 (global 32 × 128 = 4096 tok/step) |
| `batch_size` | Model JSON default **4**; launch overrides **16** (dt6144 2×2) or **4** (dt10112 4×2) |
| `collection_chunk_n_tokens` | **3 276 800** (~731 GiB train + sticky `_val_node_local` ≈37 GiB → ~**126 GiB free** on 894 GiB `/lscratch`) |
| OOM mitigations | Per-layer **encode gradient checkpointing**; **subprocess-isolated** activation collect; **`release_session_gpu` before mid-run collect**; post-collect CUDA scrub |
| `shard_kan_encoders` | **`true` on these suites** (nested fp32 FULL_SHARD per `KANEncoder`; outer bf16 MP for decoder). Default elsewhere remains `false` (replicated KAN escape hatch). See failure history below. |

### Hub-L0 retune note (2026-08-14)

| Probe | λ peak | Outcome |
|---|---|---|
| basejump **13014** | `1e-6` → floor `3e-7` | settled act **~110–113** (hub **~12.6**) |
| hub_l0_v1 **13859** | `3e-6`, decay@8k | early dip ~6@200, then act **~77 @5k** still climbing; cancelled — still ~6× above hub |
| hub_l0_v2 **14058** | `1e-5`, warmup 4000, no decay | act **~65–69 @3.1k** still climbing (`sparse≈5e-4` ≪ recon); cancelled |
| hub_l0_v3 **14062** (`1e-4`) | `1e-4`, warmup 4000, no decay | walltime @**1968**, act **~43** climbing (λ_eff only ~5e-5); wiped |
| **hub_l0_v3** (retune, current) | **`1e-3`** peak+final, warmup 4000, **no decay** | cold restart of same suite after wipe |

Prior under-penalty (θ frozen / sparse≪recon). Keep θ_init=24 and long λ warmup. Re-read act once past peak (~4k): if still ≫30 → raise further; if collapsing below ~8, back off toward `3e-4`.

### Why KAN encoders were FSDP-ignored (do not reintroduce)

Historical FSDP+KAN failures that forced full replication — nested sharding must keep these mitigations:

1. bf16 B-spline `grid` / `update_grid` NaNs (knots must stay fp32; lstsq rejects bf16).
2. Mixed-dtype FlatParameter when flattening fp32 KAN under outer bf16 MP → replicated path still uses `ignored_modules=list(encoders)`. Nested path wraps each encoder as its own FSDP unit first and must **not** list those units in `ignored_modules` (PyTorch raises `ignored_modules should not include FSDP modules`); nested children stay out of the parent FlatParameter automatically.
3. JumpReLU threshold Adam group swallowed without `use_orig_params=True`.
4. `b_enc[layer_id]` view-inplace (`ViewBackward0`) → keep `.clone()` in `encode_layer`.
5. Ignored-encoder rank drift (manual grad all-reduce; `grid` buffer broadcast) → nested path uses inner FSDP reduce + summon/`update_grid` buffer sync assert instead.
6. Must enter via `model_for_train(x)` / encoder `forward` (not bare `forward_split` on an FSDP module).
7. Per-rank NaN must skip backward on **all** ranks.

`KANEncoder.update_grid` fits on a CPU side-channel clone and never calls `.cpu()` on the live FSDP module.

Do **not** reuse plain r5b (no BaseJump/B3) or the old rebuttal gemma2 config (`λ=7.5e-4`, `bandwidth=0.001`, `d_t=8192`).

## Do not do
- Paste JumpReLU paper `ε=0.001` without unit-MS preact scaling.
- Copy GPT peak `1e-5` onto qwen/llama (oversparsifies).
- Use linear peak `0.005` when matching GPT spline at `1e-5`.
- Evaluate spline vs linear at mismatched act_lp.

## Code already in tree
- λ schedule: `sparsity_warmup_steps`, `sparsity_decay_start`, `lambda_sparsity_final`
  in `TrainConfig` / paper config (`get_lambda_sparsity` / `sparsity_lambda_at_step`).
- Use absolute `jumprelu_bandwidth` in suite JSON.
