# Neuronpedia baselines for Spline-SAE

Selection criteria:

1. Hosted / featured on [Neuronpedia](https://www.neuronpedia.org/)
2. Public weights (Hugging Face)
3. Public training recipe (paper + GitHub or official Colab with hyperparameters)
4. Not GPT-2
5. Feasible on GH200 / ai4wy nodes for 1-layer ablations

---

## 1. Gemma-2-2B — Gemma Scope (JumpReLU)

| Field | Value |
|---|---|
| Base model | `google/gemma-2-2b` (also 9B / 27B in suite) |
| SAE release | [google/gemma-scope](https://huggingface.co/google/gemma-scope) |
| Paper | [Gemma Scope, arXiv:2408.05147](https://arxiv.org/abs/2408.05147) |
| Training recipe | Official JumpReLU SAE training Colab linked from HF card (PyTorch + JAX) |
| Activation | JumpReLU (same family as our CLT) |
| Typical site | residual stream (`res`), also `mlp` / `attn` |
| Example Neuronpedia SAE | e.g. layer 20, width 16k ([HF path](https://huggingface.co/google/gemma-scope-2b-pt-res/tree/main/layer_20/width_16k/average_l0_71)) |
| Tokens (paper) | ~4B–16B depending on width |
| Why this one | Gold-standard open JumpReLU SAE; closest architectural match to Spline-CLT sparsifier; Neuronpedia demo |

**Replication checklist**

- [ ] Load published SAE via SAELens / HF and measure EV / L0 on a fixed val set
- [ ] Reproduce JumpReLU linear SAE on **one** mid layer (start: L12 or L20), width 16k, short token budget
- [ ] Match published `average_l0` band before claiming parity
- [ ] Swap encoder → KAN (`SplineSAE`) at same width / λ / bw

**Fast sibling (Gemma Scope 2):** [google/gemma-scope-2](https://huggingface.co/google/gemma-scope-2)
covers Gemma 3 from 270M–27B with JumpReLU training Colab. Use
`gemma-3-1b-pt` for cheap iteration, then confirm on Gemma-2-2B.

---

## 2. Llama-3.1-8B — Llama Scope (TopK)

| Field | Value |
|---|---|
| Base model | `meta-llama/Llama-3.1-8B` |
| SAE release | [OpenMOSS-Team/Llama-Scope](https://huggingface.co/OpenMOSS-Team/Llama-Scope) |
| Paper | [Llama Scope, arXiv:2410.20526](https://arxiv.org/abs/2410.20526) |
| Training stack | [OpenMOSS/Language-Model-SAEs](https://github.com/OpenMOSS/Language-Model-SAEs) (aka Llamascopium) |
| Activation | TopK-ReLU |
| Data | SlimPajama (subset proportions preserved) |
| Sites | R / A / M / TC × all 32 layers; widths 32k (8×) and 128k (32×) |
| Why this one | Full open distributed trainer + every-layer suite; TopK is a useful contrast to JumpReLU |

**Replication checklist**

- [ ] Install / skim `examples/` in Language-Model-SAEs for verified hyperparameters
- [ ] Load one published LXR-8x residual SAE; record EV / L0
- [ ] Reproduce TopK linear SAE on **one** layer (e.g. mid residual), 32k width, reduced tokens
- [ ] Spline analog: KAN encoder + TopK (or JumpReLU control) + linear decoder

**Note:** TopK vs JumpReLU is a confounding axis. First Spline-SAE runs on Llama
should include a JumpReLU linear control *and* a TopK linear control so we know
whether gaps come from the encoder or the sparsifier.

---

## 3. Qwen2.5-7B-Instruct — Chanin Matryoshka SAE (SAELens)

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| SAE release | [chanind/qwen2.5-7B-it-layer-20-saes](https://huggingface.co/chanind/qwen2.5-7B-it-layer-20-saes) |
| Neuronpedia | [qwen2.5-7b-it-sae](https://www.neuronpedia.org/qwen2.5-7b-it-sae) (e.g. `20-matryoshka-65k`) |
| Training stack | [SAELens](https://github.com/decoderesearch/SAELens) (Bloom / Chanin / Tigges / Duong) |
| Config source | HF `cfg.json` next to weights — SAELens best practice is to copy that cfg |
| Why this one | Different model family; SAELens is the community standard trainer; Matryoshka is SOTA-ish |

**Replication checklist**

- [ ] `SAE.from_pretrained(...)` + read `cfg.json` for exact hyperparameters
- [ ] Reproduce linear SAE on layer 20 residual with that cfg (shortened tokens OK for smoke)
- [ ] Spline SAE: same hook / width / k or λ; KAN encoder only

**Alt Qwen with explicit GitHub training fork:**
[andyrdt/saes-qwen2.5-7b-instruct](https://huggingface.co/andyrdt/saes-qwen2.5-7b-instruct)
+ [dictionary_learning fork](https://github.com/andyrdt/dictionary_learning/tree/andyrdt/qwen)
(BatchTopK). Prefer Chanin as primary because it is featured on Neuronpedia under
his name; keep Arditi as a backup recipe.

---

## Intentionally deferred

| Candidate | Why defer |
|---|---|
| GPT-2 Bloom / OpenAI SAEs | User request: leave GPT-2 |
| Llama 3.3 70B Goodfire | Too large for ablation loops |
| Gemma-2-27B / Gemma-3-27B full suites | Scale up only after 1B–8B wins |
| Eleuther Multi-TopK Llama | Overlaps Llama Scope; keep as optional SAEBench comparator |
| Pythia-70M SAE | Too small / outdated for “top model” claim |

---

## Shared evaluation protocol (all three)

1. **Site:** one residual-stream post layer (mid-network default)
2. **Metric:** explained variance / NMSE, L0, dead features, `spline_contribution_frac`, `recon_gap`
3. **Match:** tune λ or k so spline and linear land in the same L0 band (±10%)
4. **Data:** start with open pretraining mix close to the paper (Pile / SlimPajama / OpenWebText); document any deviation
5. **Compute:** single GPU, 1 layer, reduced token budget for iteration; full-budget replicate only for the winning recipe

## Loading published SAEs (reference ceiling)

```python
# Gemma Scope / many Neuronpedia SAEs via SAELens
from sae_lens import SAE

sae = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_12/width_16k/canonical",
    device="cuda",
)
```

Exact `release` / `sae_id` strings should be verified against the current
SAELens pretrained catalog and the HF repo layout before locking configs.
