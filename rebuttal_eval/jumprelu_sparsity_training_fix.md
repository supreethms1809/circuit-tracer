# JumpReLU sparsity mismatch: diagnosis and rebuttal fix

## Problem

The linear CLT baseline remained far denser than published CLTs. Even large
`lambda_sparsity` values left hundreds to thousands of active features per
layer/token, while the learned JumpReLU thresholds barely moved. The same
training defect affected spline CLTs, although their naturally different
preactivation distribution made the symptom less obvious.

## Confirmed causes

1. **FSDP discarded the dedicated threshold optimizer behavior.** With FSDP's
   default flattened parameters, `log(theta)` was merged into a
   `FlatParameter` and received the main AdamW group's settings instead of the
   intended threshold settings. In particular, it received weight decay and a
   larger Adam epsilon.
2. **The constant `theta=0.01` initialization was badly calibrated.** Linear
   preactivations had shifted to a scale where this threshold admitted far too
   many features. The very narrow JumpReLU STE window then made recovery slow.
3. **Lambda-only sweeps could not repair initialization and optimizer
   problems.** Large lambda values reduced L0, but plateaued far above the
   desired regime and harmed reconstruction.
4. **BF16 was not the primary cause.** Fixed-checkpoint FP32/BF16 gate-margin
   diagnostics showed negligible differences in measured L0.

## Training fixes

The following minimal, symmetric fixes apply to both linear and spline CLTs:

- FSDP now uses `use_orig_params=True`, preserving a distinct threshold
  optimizer group.
- The threshold group uses `weight_decay=0.0` and `adam_eps=1e-15`.
- Cold starts use deterministic data-quantile threshold initialization:
  - target initial L0: `32` per layer/token
  - calibration sequences: `16`
  - sampled values per sequence: `32768`
- Threshold metrics are gathered correctly across FSDP shards.
- Validation reconstruction loss now receives the configured
  `recon_layer_energy_beta`.
- Optimizer save/restore diagnostics verify that the threshold group survives
  FSDP checkpointing.

These are corrections to the existing JumpReLU training path, not a new
sparsification method.

## Rebuttal hyperparameters

All final arms use seed `101`, one seed per suite, and W&B project
`spline-clt-paper-v3-rebuttal`.

| Model | Linear d_t | Spline d_t | lambda | Steps |
|---|---:|---:|---:|---:|
| GPT-2 small | feature match + parameter match | feature match + parameter match | 0.005 | 29,000 |
| Qwen3-0.6B | 8,192 | 5,184 (parameter matched) | 0.0056756 | 58,000 |
| Gemma3-1B | 8,192 | 5,056 (parameter matched) | 0.005 | 58,000 |
| GPT-2 large (deferred) | 8,192 | 5,632 (parameter matched) | 0.00756 | config only |

GPT-2 small uses one two-GPU node. Qwen3 and Gemma3 use two nodes with two GPUs
per node. Their step counts are doubled relative to the one-node schedule so
that each arm still processes approximately the configured one-billion-token
budget.

## Validation-set handling

No runner locking change was retained. Each concurrently launched variant has a
separate validation cache path. Because model, corpus, seed, validation
fraction, token budget, and dedup settings are identical within a model family,
the generated validation examples are deterministic and equivalent across its
linear and spline arms. Separate paths avoid concurrent mmap writes without
changing validation content.

Launch-only configs are under:

`experiments/paper_configs_v3/launch_suites/`

## Verification

Regression and diagnostic coverage includes:

- dedicated threshold AdamW group and FSDP flattening rejection
- data-quantile initialization reaching the requested layer L0
- threshold-gradient direction and optimizer update
- FSDP optimizer-state save/restore
- NMSE normalization and layer-energy weighting
- sparsity normalization
- warmup/cosine scheduler boundaries
- fixed-holdout FP32/BF16 gate-margin comparison

## Current launch

Launched on July 24, 2026:

- GPT-2 small: jobs `8411`, `8413`, `8415`, `8412`
- Qwen3-0.6B: jobs `8408`, `8410`
- Gemma3-1B: jobs `8416`, `8417`

The first Gemma submissions (`8409`, `8414`) failed before collection because
`--export=NONE` removed the Hugging Face token required by the gated model.
They were replaced by authenticated submissions `8416` and `8417`.
