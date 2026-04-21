# `d_transcoder` Reference for Paper Baselines

Notes compiled while sizing the 4×4 paper matrix (Spline-CLT / Linear-CLT ×
param-match / feature-match × {GPT-2 small, GPT-2 large, Qwen3-0.6B,
Llama-3.2-1B}). All values below are verified directly from safetensors shapes
on HuggingFace — not guessed.

## 1. Open-sourced Cross-Layer Transcoders

Community CLTs compatible with `circuit-tracer`. Shapes read from each repo's
`W_enc_0.safetensors` header.

| Base model | HF repo | `n_layers` | `d_model` | `d_transcoder` (per layer) | Total features | R = `d_transcoder / d_model` |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B | `mntss/clt-llama-3.2-1b-524k` | 16 | 2 048 | **32 768** | 524 288 | **16×** |
| Gemma-2-2B | `mntss/clt-gemma-2-2b-426k` | 26 | 2 304 | **16 384** | 425 984 | **7.1×** |
| Gemma-2-2B | `mntss/clt-gemma-2-2b-2.5M` | 26 | 2 304 | **98 304** | 2 555 904 | **42.7×** |
| GPT-OSS-20B | `mntss/clt-131k` | 24 | 2 880 | **131 072** | 3 145 728 | **45.5×** |

Naming note: the "131k" in `mntss/clt-131k` refers to `d_transcoder` per layer,
not total features. The total across 24 layers is ~3.15 M.

## 2. Anthropic's own CLTs

From `transformer-circuits.pub/2025/attribution-graphs/methods`. Anthropic did
**not** publish per-layer `d_transcoder` values; only total feature counts.

| Model | Total features (range across runs) |
|---|---:|
| 18-layer toy model (17 MLP layers) | 300 K – 10 M |
| Claude 3.5 Haiku | 300 K – 30 M |

The headline results use the 10 M-feature 18L CLT.

## 3. Single-layer transcoders (not CLTs)

Included here only to avoid confusing these with the cross-layer numbers above.

- `mwhanna/qwen3-0.6b-transcoders-lowl0` — single-layer, ~160× expansion
  (~163 K features per layer for Qwen3-0.6B with `d_model=1024`).
- `mwhanna/qwen3-1.7b-transcoders-lowl0`
- `mwhanna/qwen3-4b-transcoders`
- `mwhanna/qwen3-8b-transcoders`
- `mwhanna/gemma-scope-2-{270m,1b,4b,12b,27b}-{pt,it}`

Single-layer transcoders train one per MLP layer independently, so `d_transcoder`
has to be much wider than in the cross-layer setting because the dictionary
can't amortize features across layers. **Not a direct baseline for CLT
experiments.**

## 4. TransformerLens

Confirmed: TransformerLens ships zero CLT / transcoder / `d_transcoder`
references. It's the model-hooking library only. Nothing to pull from there for
dictionary sizing.

## 5. Sanity check — what this repo's configs use today

| Config | `d_model` | `d_transcoder` | R |
|---|---:|---:|---:|
| `experiments/configs/gpt2_small.yaml` | 768 | 4 096 | 5.3× |
| `experiments/paper_configs/models/gpt2_spline_wide.json` | 768 | 16 384 | 21× |
| `experiments/paper_configs/models/gpt2_linear_wide.json` | 768 | 32 768 | 43× |
| `experiments/paper_configs/models/gpt2_spline_wide_feature_match.json` | 768 | 1 862 | 2.4× |
| `experiments/configs/qwen25_05b.yaml` / spline json | 896 | 4 096 | 4.6× |
| `experiments/configs/qwen3_06b.yaml` | 1 024 | 13 312 | 13× |
| `experiments/configs/gemma3_1b.yaml` | 1 152 | 4 096 | 3.6× |

## 6. Recommended reference column for the 4×4 paper matrix

Adopt **R = 16** across the board for the Linear-CLT *param-match* reference.
Reasoning: it matches the only published per-layer CLT size (`mntss/clt-llama-3.2-1b-524k`),
and R=16 is in the middle of the published range so reviewers won't see it as
cherry-picked.

| Model | `d_model` | Linear param-match `d_transcoder` | Total features |
|---|---:|---:|---:|
| GPT-2 small | 768 | 12 288 | 147 K (12 layers) |
| GPT-2 large | 1 280 | 20 480 | 737 K (36 layers) |
| Qwen3-0.6B | 1 024 | 16 384 | 459 K (28 layers) |
| Llama-3.2-1B | 2 048 | **32 768** | 524 K (matches mntss exactly) |

From each row, derive the other three variants algebraically (rough ratios for
G=5, k=3 KAN; confirm with the param-count helper in `spline_clt/utils.py`):

- **Linear param-match** = row above (the reference).
- **Spline feature-match** = same `d_transcoder`, swap encoder to KAN (~10× more
  encoder params, same decoder).
- **Spline param-match** ≈ `d_transcoder / 2` (KAN's `(G+k)=10` basis multiplier,
  decoder unchanged). Gives the same total param budget as Linear param-match.
- **Linear feature-match** = Spline param-match `d_transcoder` with a linear
  encoder. This is the "spline wins on params per feature" control.

The 2× ratio shrinks toward 1:1 as `d_model` grows (decoder dominates); for
Llama-3.2-1B it's closer to 1.4× in practice, so recompute per model using
the param formula:

```
encoder_params_kan    = (G+k) × d_model × d_transcoder × n_layers     # G=5,k=3 → (G+k)=10
encoder_params_linear = d_model × d_transcoder × n_layers
decoder_params        = d_transcoder × d_model × (n_layers × (n_layers+1) / 2)
total                 = encoder + decoder
```

## 7. Chosen paper matrix (3 models × 4 variants)

Scoped to what we can realistically train: GPT-2 small, GPT-2 large, Qwen3-0.6B.
Each model has four variants — two linear, two spline — defined as:

- **feature-match**: same `d_transcoder` across linear and spline. Fair comparison
  at a fixed dictionary size.
- **param-match**: `d_transcoder` chosen so total params equal the opposite
  encoder's feature-match total. Fair comparison at a fixed parameter budget.

Feature-match `d_transcoder` is pinned to §6's reference column (R ≈ 16) and
rounded down to what the published CLTs use where available:

| Model | n_layers | d_model | feature-match d_t | linear param-match d_t | spline param-match d_t |
|---|---:|---:|---:|---:|---:|
| gpt2-small  | 12 |   768 | 12 288 | 27 008 |  5 568 |
| gpt2-large  | 36 | 1 280 | 20 480 | 29 952 | 14 016 |
| qwen3-0.6B  | 28 | 1 024 | 16 384 | 25 920 | 10 368 |

Param-match values are derived from the §6 formula with G=5, k=3 (so KAN basis
multiplier is (G+k)=10), then rounded to the nearest multiple of 64 for
alignment-friendly shapes. Exact-match unrounded values:

- gpt2-small:  linear pm ≈ 27 041; spline pm ≈ 5 585.
- gpt2-large:  linear pm ≈ 29 931; spline pm ≈ 14 011.
- qwen3-0.6B:  linear pm ≈ 25 897; spline pm ≈ 10 365.

Config files live under `experiments/paper_configs/models/` as
`{model}_{encoder}_{match}.json`, with three corresponding suites under
`experiments/paper_configs/suites/paper_{model}.json`. The pre-matrix configs
were moved to `experiments/paper_configs_backup/` rather than deleted.

## 8. Token budget and training schedule

- **Total tokens per run**: 40 M (set in `base/common*.json` via
  `dataset.n_tokens`). This is the size of the activation corpus collected once
  per `(model, n_tokens)` pair; all 12 variants for a model share the corpus.
- **Epochs**: 2. Total tokens seen per run = 80 M.
- **Sequence length**: 128 tokens. Tokens per step = `batch_size × 128`.
- **Warmup**: 5% of `total_steps`.
- **LR**: 1e-4 linear / 5e-5 spline with AdamW, β2=0.95 (standard for SAE/CLT).
- **Cosine decay** to 0 over the remaining 95% of steps.
- **Log every**: 25 steps.
- **Eval every**: ~total_steps / 40 (forty evals per run).
- **Save every**: total_steps / 4 (four intermediate checkpoints plus `_best`
  and `_final`). Note: §10 flags that this may need to drop to 1 checkpoint for
  the 27 B variants to stay under the 7 TB HF storage budget.
- **Grid update** (spline only): single refit at `total_steps / 2`.

Per-variant step counts (2 × 40 M / (bs × 128), rounded to nice numbers):

| Variant | bs | total_steps | warmup | eval_every | save_every | update_grid |
|---|---:|---:|---:|---:|---:|---:|
| gpt2-small linear fm  | 32 |  20 000 | 1 000 |   500 |  5 000 |     — |
| gpt2-small linear pm  | 16 |  40 000 | 2 000 | 1 000 | 10 000 |     — |
| gpt2-small spline fm  | 16 |  40 000 | 2 000 | 1 000 | 10 000 | 20 000 |
| gpt2-small spline pm  | 16 |  40 000 | 2 000 | 1 000 | 10 000 | 20 000 |
| gpt2-large linear fm  |  8 |  80 000 | 4 000 | 2 000 | 20 000 |     — |
| gpt2-large linear pm  |  4 | 160 000 | 8 000 | 4 000 | 40 000 |     — |
| gpt2-large spline fm  |  4 | 160 000 | 8 000 | 4 000 | 40 000 | 80 000 |
| gpt2-large spline pm  |  4 | 160 000 | 8 000 | 4 000 | 40 000 | 80 000 |
| qwen3-0.6B linear fm  |  8 |  80 000 | 4 000 | 2 000 | 20 000 |     — |
| qwen3-0.6B linear pm  |  4 | 160 000 | 8 000 | 4 000 | 40 000 |     — |
| qwen3-0.6B spline fm  |  4 | 160 000 | 8 000 | 4 000 | 40 000 | 80 000 |
| qwen3-0.6B spline pm  |  8 |  80 000 | 4 000 | 2 000 | 20 000 | 40 000 |

## 9. Activation dataset sizing and HF hosting

At 40 M tokens in bfloat16, activation-per-layer storage is:

| Model | Size | Note |
|---|---:|---|
| gpt2-small  | ~700 GB | fits one HF dataset repo comfortably |
| qwen3-0.6B  | ~2.2 TB | chunked into ~20 GB safetensors shards |
| gpt2-large  | ~3.5 TB | tight under HF 300 GB/repo soft cap; needs multi-file |
| **Total**   | **~6.4 TB** | fits within the 7 TB public HF budget |

Planned layout: one HF dataset repo per `(model, n_tokens)` pair, e.g.
`supreethms1809/clt-activations-gpt2-small-40M`, with ~10–20 GB shards and a
JSON sidecar carrying `{model, dataset, seq_len, seed, tokenizer_rev, dtype,
n_tokens, layer_indices, creation_date}`.

Pre-collecting (rather than streaming from tokens each training run) is worth
it because each model is reused across 4 variants × 3 seeds = 12 training
runs; streaming would re-run the base-model forward 12× per model.

## 10. VRAM budget on a single GH200 (96 GB)

Parameter totals (encoder + decoder), decoder dominates at the large variants:

| Variant | Total params | bf16 param + grad | AdamW state (fp32 m+v) | Adafactor state | Fits 96 GB (Adafactor)? |
|---|---:|---:|---:|---:|:---:|
| gpt2-small (all 4)      |   0.8–1.9 B |   3.0–7.5 GB |   6–15 GB |  <1 GB | ✓ easy |
| qwen3 spline pm / lin fm | 7.3 B       | 29 GB        |  58 GB    |   1 GB | ✓ |
| qwen3 spline fm / lin pm | 11.5 B      | 46 GB        |  92 GB    |   2 GB | ✓ tight |
| gpt2-large lin fm / sp pm | 18.4 B     | 73 GB        | 147 GB    |   3 GB | ✓ very tight |
| gpt2-large lin pm / sp fm | 27 B       | 108 GB       | 216 GB    |   4 GB | **✗ over** |

Two gpt2-large variants (linear param-match @ 29 952 and spline feature-match
@ 20 480) exceed 96 GB even with Adafactor, because bf16 params + bf16 grads
alone reach ~108 GB. Every other variant fits with Adafactor.

## 11. CPU offload: what works and what doesn't

Reviewed the upstream circuit-tracer offload implementation at
`circuit_tracer/utils/disk_offload.py`:

```python
def cpu_offload_module(module):
    module.to("cpu")
    return lambda: module.to(gpu)

def disk_offload_module(module):
    save_file(module.state_dict(), tmpfile)
    module.to("meta")
    return lambda: module.load_state_dict(load_file(tmpfile), assign=True)
```

This is used in `attribute_transformerlens.py` to park transcoders, MLPs, and
embed/unembed between attribution *phases* (activation collection → attribution
→ decode). Each phase uses a subset of modules, so the rest can live off-GPU.

**This pattern does not translate to CLT training.** Training is not phased —
every step touches every parameter: encoder forward → decoder forward → both
backward → optimizer step over all. Moving full params on/off GPU each step
doesn't reduce peak VRAM (the peak happens *during* the step, not between).

Options that would actually fit the 27 B variants:

1. **FSDP / DeepSpeed with CPU offload.** Canonical fix. Streams one layer of
   params+grads to GPU at a time, everything else stays on CPU. ~150–200 LOC
   change in `train.py` plus checkpointing care for `summon_full_params` around
   the spline grid-update step. Deferred.
2. **Custom layer-streaming training loop.** CLT has no cross-token / deep
   compute-graph dependency, so we could stream layers manually. ~300–400 LOC,
   high correctness burden, slower than FSDP because transfers aren't
   overlapped with compute. Deferred.
3. **Shrink the 27 B variants** so they fit 96 GB with Adafactor. Cap
   d_transcoder around 23 000 for gpt2-large, recompute the opposite variant to
   keep the param-match exact. Keeps the paper claim intact but changes the
   exact R ratio slightly.

## 12. Open decision (to revisit after GH200 dry-run)

Decision on the 27 B gpt2-large variants is deferred until a first smoke run
on GH200. Until then:

- All 12 configs are written at the §7 matrix values.
- `optimizer` is left at the default `adamw` for gpt2-small and at the chosen
  default for gpt2-large/qwen3 (to be switched to `adafactor` during the
  smoke-run pass, along with potential CPU-offload integration).
- If the 27 B variants OOM on GH200, we either (a) shrink them per §11 option
  3, or (b) integrate FSDP / DeepSpeed per §11 option 1.

## 13. Sources

- [transformer-circuits.pub — Circuit Tracing methods (Anthropic, 2025)](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
- [safety-research/circuit-tracer README](https://github.com/safety-research/circuit-tracer)
- [mntss Cross-Layer Transcoders collection (HF)](https://huggingface.co/collections/mntss/cross-layer-transcoders)
- [mntss/clt-llama-3.2-1b-524k](https://huggingface.co/mntss/clt-llama-3.2-1b-524k)
- [mntss/clt-gemma-2-2b-426k](https://huggingface.co/mntss/clt-gemma-2-2b-426k)
- [mntss/clt-gemma-2-2b-2.5M](https://huggingface.co/mntss/clt-gemma-2-2b-2.5M)
- [mntss/clt-131k (GPT-OSS-20B)](https://huggingface.co/mntss/clt-131k)
- [mwhanna HF profile — single-layer transcoders](https://huggingface.co/mwhanna)
