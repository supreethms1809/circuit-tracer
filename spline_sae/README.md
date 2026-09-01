# Spline-SAE investigation

Isolated workstream to validate **KAN / spline encoders as SAE replacements**
before bringing recipes back into Spline-CLT.

Cross-layer transcoders (CLT) hide encoder bugs behind multi-layer decode,
λ coupling, and circuit-tracer eval cost. A per-layer SAE is the right place to
answer: *can a nonlinear encoder beat a linear SAE at matched L0 on residual
reconstruction?*

This package is intentionally separate from `spline_clt/` and the paper runner.

## Why not GPT-2

GPT-2 small was useful for CLT plumbing. For the SAE question we want:

- models that already have **Neuronpedia-hosted, published SAEs**
- **open training recipes** (paper + GitHub + HF weights / `cfg.json`)
- families beyond GPT-2 so results are not GPT-2-specific

## Top-3 baselines (Neuronpedia + replicable)

| # | Base model | Published SAE | Sparsifier | Training stack | Neuronpedia |
|---|------------|---------------|------------|----------------|-------------|
| 1 | `google/gemma-2-2b` | **Gemma Scope** (DeepMind) | JumpReLU | [gemma-scope HF](https://huggingface.co/google/gemma-scope) + [JumpReLU training Colab](https://huggingface.co/google/gemma-scope) + [arxiv:2408.05147](https://arxiv.org/abs/2408.05147) | [gemma-scope](https://www.neuronpedia.org/) |
| 2 | `meta-llama/Llama-3.1-8B` | **Llama Scope** (OpenMOSS) | TopK-ReLU | [Llamascopium / Language-Model-SAEs](https://github.com/OpenMOSS/Language-Model-SAEs) + [HF](https://huggingface.co/OpenMOSS-Team/Llama-Scope) + [arxiv:2410.20526](https://arxiv.org/abs/2410.20526) | Llama Scope on Neuronpedia |
| 3 | `Qwen/Qwen2.5-7B-Instruct` | **Chanin Matryoshka SAE** | Matryoshka / SAELens | [SAELens](https://github.com/decoderesearch/SAELens) + [HF](https://huggingface.co/chanind/qwen2.5-7B-it-layer-20-saes) | [qwen2.5-7b-it-sae](https://www.neuronpedia.org/qwen2.5-7b-it-sae) |

Details, hook points, widths, and replication checklists:
[`baselines/NEURONPEDIA_BASELINES.md`](baselines/NEURONPEDIA_BASELINES.md).

**Fast path within #1:** Gemma Scope 2 also ships `gemma-3-270m` / `gemma-3-1b`
SAEs ([google/gemma-scope-2](https://huggingface.co/google/gemma-scope-2)) for
cheap ablations before scaling to Gemma-2-2B.

## Spline analog (what we build)

Same-layer SAE with a KAN encoder and a **linear** decoder (keep circuit /
steering compatibility):

```text
x  →  KAN_enc (or linear enc)  →  JumpReLU / TopK  →  a
ŷ  =  W_dec a + b
L  = NMSE(ŷ, x) + λ_sparsity [ − λ_nl · gap ]
```

where `gap = NMSE(ŷ_base_only, x) − NMSE(ŷ_full, x)` rewards *useful*
nonlinearity, not spline magnitude.

For each baseline model we train:

1. **Linear SAE** — replicate published recipe as closely as practical (control)
2. **Spline SAE** — swap encoder for KAN; start from published λ / width / site
3. Optional: load the **published checkpoint** as an external reference (not
   retrain) for fidelity ceilings

## Directory layout

```text
spline_sae/
├── README.md                 # this file
├── PLAN.md                   # phased experimentation plan
├── baselines/                # Neuronpedia / HF / recipe notes
├── configs/                  # one YAML/JSON per model × arm
├── scripts/                  # collect / train / eval entrypoints
├── model.py                  # SplineSAE (KAN or linear encoder)
├── loss.py                   # NMSE + sparsity + optional gap term
├── data.py                   # activation collection for one hook
├── train.py                  # single-GPU training loop
└── tests/
```

Reuse from the parent repo where safe: `spline_clt.kan_encoder.KANEncoder`,
JumpReLU utilities, seeding helpers. Do **not** depend on the paper runner or
CLT cross-layer decode.

## Success criteria (before returning to CLT)

On ≥1 layer per model, at matched L0 vs linear SAE:

- held-out NMSE / explained variance competitive with linear
- `spline_contribution_frac` mid-range **and** `gap > 0`
- optional SAEBench / Neuronpedia-style dashboards later

Only then port the winning recipe into Spline-CLT.
