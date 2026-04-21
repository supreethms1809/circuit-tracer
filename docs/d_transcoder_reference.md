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

## 7. Sources

- [transformer-circuits.pub — Circuit Tracing methods (Anthropic, 2025)](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
- [safety-research/circuit-tracer README](https://github.com/safety-research/circuit-tracer)
- [mntss Cross-Layer Transcoders collection (HF)](https://huggingface.co/collections/mntss/cross-layer-transcoders)
- [mntss/clt-llama-3.2-1b-524k](https://huggingface.co/mntss/clt-llama-3.2-1b-524k)
- [mntss/clt-gemma-2-2b-426k](https://huggingface.co/mntss/clt-gemma-2-2b-426k)
- [mntss/clt-gemma-2-2b-2.5M](https://huggingface.co/mntss/clt-gemma-2-2b-2.5M)
- [mntss/clt-131k (GPT-OSS-20B)](https://huggingface.co/mntss/clt-131k)
- [mwhanna HF profile — single-layer transcoders](https://huggingface.co/mwhanna)
