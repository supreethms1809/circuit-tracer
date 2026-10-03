# TMLR-250 v3 results (MIB, seed 0)

> **Part of the MACAG docs pack.** Hub: [`macag.md`](macag.md).
> Protocol: [`run_todo.md`](run_todo.md). Fairness rules: [`TMLR_EVAL_RECIPE.md`](TMLR_EVAL_RECIPE.md).
> Method IDs: [`baseline_method_map.md`](baseline_method_map.md).
>
> **These are the current paper-scale numbers.** They replace the pre-v3 two-hop
> / \(n=500\) IOI tables in [`macag_appendix_legacy.md`](macag_appendix_legacy.md).
> Do not mix with v4 outdirs (`macag_mib_tmlr250v4_*`).

Aggregates below were recomputed from per-prompt JSON under
`/gscratch/ssuresh/macag_mib_tmlr250v3_{h200,gemma25m,llama}/macag_mib_seed0/`
(2026-08-15). Means are \(\pm\) SEM over prompts. **Frozen** Game 1 / baseline
rows use the frozen-attention oracle unless named otherwise.

A working notes dump also lives at each v3 root as `Notes.md`. That file
incorrectly aliases `acdc_native` with `acdc_edge` and `eap` with `eap_edge`,
and reports Game 1 \(|E|=8.0\) always. **Do not copy those aliases.** Native-edge
methods are Track B and exist for **one IOI prompt per CLT**. Graph EAP is a
partial Track A row (foil-seed failures).

---

## 1. What was run

| Knob | v3 value |
| --- | --- |
| Prompts | MIB IOI 100 + MCQA 50 + ARC-Easy 50 **per CLT** (200 × 3 = 600) |
| CLTs | `gemma2-426k` (h200), `gemma2-2.5M`, `llama32-524k` |
| Seed | **0** only |
| Graph export | circuit-tracer prune **`node_threshold=0.8`** (not the v4 unpruned `1.0`) |
| Candidates | feature nodes in that graph (mean 748 / 796 / 505) |
| Prefilter / connected | **off** |
| Selection | **`logit_gap`** |
| KL | post-hoc sidecar `macag_kl_faithfulness.json` + per-leg `kl_faithfulness` on Game 1 |
| Game 1 | `--freeze-mode both`, \(\alpha=0.5\), \(\lambda=0.02\), \(\varepsilon=0.1\), `stop_metric=raw_relative`, **`budget=8`** |
| Game 2 | **ABR and FP**, \(\beta=0.2\), `abr_iters=4`, budget 8, **`freeze_attention=true`** (single convention; not dual-freeze) |
| Track A baselines | `influence`, `eap` (graph), `eap_syed`, `acdc` (ported), `acdc_native`, `shapley` (partial) |
| Track B | `macag_original_baselines.json` on **one** prompt per CLT (`*_ioi_0000`) |

Coverage:

| Artifact | gemma2-426k | gemma2-2.5M | llama32-524k |
| --- | ---: | ---: | ---: |
| Game 1 dual-freeze | 200/200 | 200/200 | 200/200 |
| Game 2 ABR | 200/200 | 200/200 | 200/200 |
| Game 2 FP | 200/200 | 200/200 | 200/200 |
| `macag_baselines.json` | 200/200 | 200/200 | 200/200 |
| KL sidecar | 200/200 | 200/200 | 200/200 |
| Shapley `results@8` | **7/200** (ARC-Easy) | **0/200** | **196/200** |
| Graph EAP (`eap`) usable | 34/200 | 34/200 | 13/200 |
| Native-edge JSON | 1 (IOI_0000) | 1 | 1 |

InterpBench is a **separate** tree (`macag_interpbench_tmlr250v3`); it is not
folded into the 200-prompt MIB tables.

### Protocol caveats (read before quoting)

1. **Budget 8 is a cap, not “natural size.”** Frozen \(|E^\star|\) means 7.56 /
   7.11 / 7.86 (Gemma-426k / 2.5M / Llama); 161 / 128 / 189 prompts hit 8.
   Unfrozen is smaller (5.89 / 6.40 / 7.46). v4 and Dallas unbudgeted Game 1
   are a different experiment.
2. **Graphs are pruned.** Dallas showed 0.8/0.98 prune vs unpruned is a
   different candidate universe (279 vs 2028 features on that prompt). v3
   selector comparisons are on the *pruned* graph.
3. **`eap` ≠ `eap_syed` ≠ `eap_edge`.** Graph EAP failed foil-logit seeding on
   166/200 Gemma prompts (`unavailable`). Feature AtP (`eap_syed`) ran on all
   200 (3 forwards + 1 backward). Native-edge EAP is Track B, one prompt.
4. **`acdc` matched-\(k\) is not “ACDC at 8.”** Scores live in `methods.acdc.matched_k`
   (not `results["8"]`). Every prompt has `budget_capped=true`; achieved \(k\)
   averages 1.55–1.84. Report the \(\tau\) sweep (sparse vs dense) separately.
5. **`acdc_native` scores are not on the CLT feature oracle.** Do not Jaccard
   or stack F against Game 1. Many native matched-k score blocks are empty in
   v3 JSON — do not invent F from Notes.md.
6. **Shapley is Monte Carlo (64 antithetic perms), not exact**, and is missing
   on Gemma-2.5M. Game 1 F \(>\) Shapley top-8 F on Llama is **not** “greedy
   beats gold”; Game 1 optimizes joint \(F(E)\), Shapley ranks \(\phi_i\) then
   takes a prefix.

---

## 2. Game 1 (frozen) — primary table

\(F = 0.5\,S + 0.5\,N\) on logit-gap. \(R = S_{\mathrm{all}} - S_{\mathrm{empty}}\).
KL-F is the post-hoc \(-\mathrm{KL}\) faithfulness of the same \(E^\star\).

### 2.1 Overall (\(n=200\))

| CLT | \(F\) | \(S\) | \(N\) | mean \(\lvert E^\star\rvert\) | \(R\) | oracle calls | KL-F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama32-524k | **13.03 ± 0.57** | 16.55 ± 0.92 | **9.52 ± 0.49** | 7.86 | **+17.72 ± 0.62** | 7,958 | **5.14 ± 0.33** |
| gemma2-426k | **8.50 ± 0.18** | 14.59 ± 0.39 | 2.42 ± 0.16 | 7.56 | **−2.68 ± 0.73** | 11,587 | 1.15 ± 0.14 |
| gemma2-2.5M | **8.42 ± 0.22** | 14.99 ± 0.48 | 1.85 ± 0.11 | 7.11 | **−4.50 ± 0.87** | 12,077 | 1.72 ± 0.14 |

Llama is both more sufficient *and* more necessary, with a large **positive**
frozen recoverable range. Gemma’s overall \(R\) is pulled negative by IOI
(next table): frozen features cannot move the IOI gap, so **raw** \(F\) is
still the right headline (driven by keep-only overshoot, not by \(R\)).

### 2.2 By task (frozen \(F\))

| CLT | IOI \(n=100\) | MCQA \(n=50\) | ARC-Easy \(n=50\) |
| --- | ---: | ---: | ---: |
| llama32-524k | 8.95 ± 0.34 | **17.66 ± 1.23** | **16.58 ± 1.36** |
| gemma2-426k | 9.35 ± 0.24 | 7.50 ± 0.35 | 7.82 ± 0.34 |
| gemma2-2.5M | 9.92 ± 0.32 | 6.58 ± 0.22 | 7.28 ± 0.34 |

Gemma Game 1 is **strongest on IOI by \(F\)** even though that is the
attention-mediated task: keep-only of 7–8 features **overshoots** the clean
gap while remove barely moves it (IOI \(N \approx 1.02\) / \(0.68\)). That is
the reconstruction / copy quadrant ([§7](#7-sn-quadrant-how-to-read-f)), not
a knockout circuit. Llama’s high MCQA/ARC \(F\) comes with high \(N\) as well
(\(N \approx 13.4\) / \(13.7\)).

### 2.3 Unfrozen Game 1 (\(n=200\))

| CLT | \(F\) | \(S\) | \(N\) | mean \(\lvert E^\star\rvert\) | \(R\) |
| --- | ---: | ---: | ---: | ---: | ---: |
| gemma2-426k | 11.38 ± 0.22 | 21.32 ± 0.42 | 1.44 ± 0.14 | 5.89 | **+11.00 ± 0.53** |
| gemma2-2.5M | 10.84 ± 0.31 | 20.84 ± 0.61 | 0.84 ± 0.08 | 6.40 | **+10.73 ± 0.44** |
| llama32-524k | 10.76 ± 0.28 | 10.09 ± 0.34 | 11.43 ± 0.48 | 7.46 | +15.55 ± 0.72 |

Unfreezing **flips Gemma \(R\) from negative to \(\approx +11\)** and inflates
\(S\). Llama \(F\) is slightly *lower* unfrozen than frozen — the fact already
lived in features. Frozen \(\leftrightarrow\) unfrozen evidence Jaccard is
low (0.04 Gemma, 0.10 Llama): the two legs are not the same set.

---

## 3. Attention-mediation diagnostic

Verdict from paired Game 1 ranges (`attention_mediation.verdict` in
`macag_game1.json`). `range_flip` = frozen \(R<0\) and unfrozen \(R\ge 0\).

| CLT | feature_mediated | attention_mediated | indeterminate | IOI attention_mediated |
| --- | ---: | ---: | ---: | ---: |
| gemma2-426k | 83 | **103** | 14 | **83/100** |
| gemma2-2.5M | 89 | **104** | 7 | **89/100** |
| llama32-524k | **197** | 0 | 3 | **0/100** (97 feature, 3 indeterminate) |

Task split (Gemma-426k): IOI 83 attention / 4 feature / 13 indeterminate;
MCQA 4 attention / 46 feature; ARC-Easy 16 attention / 33 feature / 1
indeterminate. Gemma-2.5M is the same pattern (IOI 89/100 attention; MCQA
49/50 feature). Llama MCQA and ARC-Easy are 50/50 feature-mediated.

**Cite this, not the legacy 402/500 vs 476/500.** The direction is the same
(Gemma IOI attention-mediated, Llama feature-mediated); the v3 \(n\) is 100
IOI prompts per CLT on pruned graphs with budget 8.

Frozen IOI \(R\): Gemma-426k **−10.51 ± 0.68**, Gemma-2.5M **−14.69 ± 0.79**,
Llama **+10.35 ± 0.42**. A \(6\times\) Gemma dictionary does **not** flip IOI
to feature-mediated. Model family does.

---

## 4. Game 2 (contrastive)

Same graph + logit-gap oracle as Game 1. Budget 8, \(\beta=0.2\).

### 4.1 ABR (\(n=200\))

| CLT | mean overlap | exact-zero | ABR “converged” | mean iters | \(F_y\) | \(F_{\mathrm{foil}}\) | oracle calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemma2-426k | **0.44% ± 0.13%** | **188/200** | 193/200 | 2.14 | 8.71 ± 0.19 | 9.12 ± 0.21 | 24,567 |
| gemma2-2.5M | **0.10% ± 0.06%** | **197/200** | 194/200 | 2.11 | 8.71 ± 0.23 | 9.73 ± 0.22 | 25,908 |
| llama32-524k | **1.01% ± 0.18%** | **172/200** | 195/200 | 2.16 | 13.06 ± 0.57 | 10.70 ± 0.20 | 16,480 |

### 4.2 Fictitious play

FP overlap is essentially the same (0.50% / 0.07% / 1.04%) with exact-zero
186 / 198 / 171. FP reports `converged=true` on **200/200**. That is the
solver flag, not a theorem: existence of a PSNE is guaranteed; these dynamics
are the implemented search ([§3.4](macag_foundations.md#34-equilibrium-analysis)).

**Do not write “overlap_rate 0.0 in 600/600.”** Near-disjoint is the result;
the residual 0.1–1% is real. Wider Gemma dictionary \(\to\) slightly cleaner
separation (0.44% \(\to\) 0.10%).

Dallas is the warning label for **objective choice**: KL Game 2 on that prompt
shared 18 nodes (Jaccard 0.36). v3 Game 2 is logit-gap, which is why overlap
stays near zero here.

---

## 5. Track A baselines (frozen, \(k=8\) prefixes)

All methods evaluated under the **same** MACAG feature oracle \(v\) except
`acdc_native` (heads/MLPs — size only). Ranking methods report F of their
top-8 prefix (`methods.*.results["8"]`). Game 1 row is frozen \(E^\star\)
(size \(\le 8\), not always 8).

### 5.1 Macro (\(n=200\) unless noted)

**Llama-3.2-1B (524k)**

| Method | \(n\) | \(F@8\) | \(S\) | \(N\) | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| **Game 1** | 200 | **13.03 ± 0.57** | 16.55 | 9.52 | greedy set, \(\lvert E^\star\rvert\approx 7.86\) |
| Shapley top-8 | **196** | 11.12 ± 0.53 | 9.89 | **12.35** | 64 MC perms; ranking not a circuit search |
| `eap_syed` | 200 | 4.25 ± 0.32 | 3.10 | 5.40 | 3 fwd + 1 bwd |
| `acdc` matched \(k\le 8\) | 200 | 3.09 ± 0.42 | — | — | achieved \(k\) **1.84**; all budget-capped |
| influence | 200 | 2.17 ± 0.35 | 0.22 | 4.12 | 0 selection forwards |
| graph `eap` | **13** | −0.51 ± 0.44 | −1.59 | 0.56 | foil-seed unavailable on 187/200 |

**Gemma-2-2B (426k)**

| Method | \(n\) | \(F@8\) | \(S\) | \(N\) |
| --- | ---: | ---: | ---: | ---: |
| **Game 1** | 200 | **8.50 ± 0.18** | 14.59 | 2.42 |
| Shapley top-8 | **7** | 5.04 ± 0.77 | 6.06 | 4.03 |
| `eap_syed` | 200 | 1.29 ± 0.11 | 1.80 | 0.79 |
| `acdc` matched \(k\le 8\) | 200 | 1.08 ± 0.16 | — | — |
| graph `eap` | 34 | −0.91 ± 0.30 | −1.69 | −0.13 |
| influence | 200 | **−1.57 ± 0.29** | −2.89 | −0.25 |

**Gemma-2-2B (2.5M)** — no Shapley.

| Method | \(n\) | \(F@8\) |
| --- | ---: | ---: |
| **Game 1** | 200 | **8.42 ± 0.22** |
| graph `eap` | 34 | 1.40 ± 0.49 |
| `acdc` matched | 200 | 0.94 ± 0.17 |
| `eap_syed` | 200 | 0.94 ± 0.06 |
| influence | 200 | **−2.72 ± 0.27** |

Influence on Gemma IOI is actively harmful (keep-only of the top-8 influence
nodes *lowers* the gap). Feature AtP is better than influence but far from
Game 1. Graph EAP must be reported as a **partial** row, not a 200-prompt
mean.

### 5.2 Ported ACDC \(\tau\)-sweep (same \(v\), not matched-\(k\))

Mean over 200 prompts of the **sparsest** and **densest** \(\tau\) circuits
in `methods.acdc.sweep`:

| CLT | sparse \(\lvert E\rvert\) / \(F\) | dense \(\lvert E\rvert\) / \(F\) | selection oracle calls |
| --- | ---: | ---: | ---: |
| llama32-524k | 58 / **22.87** | 398 / 34.51 | 22,002 |
| gemma2-426k | 25 / 5.09 | 606 / 13.53 | 33,093 |
| gemma2-2.5M | 23 / 3.82 | 661 / 12.10 | 35,313 |

On Llama, unconstrained \(\tau\)-prune can beat Game 1’s **budgeted** \(F\)
by keeping tens to hundreds of nodes. That is a density–faithfulness
tradeoff, not a sparse win. On Gemma, even the sparse end (\(k\approx 24\))
is below Game 1’s 7-feature \(F\), and the dense end needs \(\sim 600\)
nodes to pass Game 1. Dallas (unpruned, KL \(v\)) showed the same geometry:
\(\tau=0.5 \to 14\) nodes below Game 1@5; \(\tau=0.1 \to 167\) nodes at high F.

### 5.3 Cost

| Method | Typical selection cost (v3) |
| --- | --- |
| influence / graph EAP | 0 intervention calls |
| `eap_syed` | 3 forwards + 1 backward (all 200 Gemma prompts) |
| Game 1 (one freeze leg) | ~8k–12k oracle calls (frozen means above) |
| Game 2 ABR | ~16k–26k |
| ported ACDC \(\tau\) grid | ~22k–35k |
| Shapley 64-perm (Llama) | \(\sim 8\times 10^4\) oracle calls/prompt (sample of stored `selection_stats`; 64 antithetic perms) |

Game 1 is expensive relative to influence/AtP and cheap relative to Shapley
*when Shapley actually ran*. Do not cite the missing nonlinear-benchmark
“44.7×” figure.

---

## 6. One-prompt illustration (Gemma IOI_0000)

Full writeup: `/gscratch/ssuresh/macag_mib_tmlr250v3_h200/mib_gemma2_ioi_0000_comparison.md`.

Prompt: *“After the lunch, Greg and Jeff went to the clinic. Greg gave a
necklace to”* — target ` Jeff`, foil ` Greg`, clean gap \(+4.75\).

| Setting | size | keep-only gap | \(S\) | \(N\) | \(F\) | keep KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Game 1 frozen | 8 feat. | +22.88 | +18.00 | +1.63 | **+9.81** | 5.66 |
| Game 1 unfrozen | 7 feat. | +16.25 | +15.48 | +0.88 | +8.18 | 1.73 |
| Game 2 \(E_y\) | 8 | +22.75 | +18.13 | +1.88 | +10.00 | — |
| Game 2 \(E_{\mathrm{foil}}\) | 8 | +2.88 | +7.38 | +2.50 | +4.94 | — (overlap **0**) |
| influence @8 | 8 | +5.50 | +0.75 | −1.63 | −0.44 | 0.82 |
| `eap_syed` @8 | 8 | +4.88 | +0.13 | −0.50 | −0.19 | 6.16 |
| `eap_edge` @8 | 8 **edges** | −3.62 | — | — | negative gap | 1.25 |
| `eap_edge` @4096 | 4096 edges | +4.62 | — | — | ~clean gap | **0.009** |
| `acdc_edge` \(\tau=0.1\) | 2 edges | −3.98 | — | — | inverted | 1.36 |

Equal-KL matching of influence prefixes to `eap_edge@4096` needed **~384
frozen / ~450 unfrozen features**. Game 1@8 is a logit-gap circuit, not a
full-distribution copy. That is the same lesson as Dallas KL Game 1
(keep-only still 5–7 nats).

Track B on this prompt: tiny edge circuits sit at the corrupt/empty floor;
dense EAP recovers the gap. **Do not Jaccard edges to Game 1 features.**

---

## 7. S/N quadrant (how to read \(F\))

Default \(F\) hides the application. Threshold used in the gscratch notes:
high \(S\) or \(N\) if \(\ge 0.5\cdot S_{\mathrm{all}}\). Qualitative pattern
from those counts (treat as approximate; recompute for camera-ready):

- **Llama Game 1:** many prompts in high-\(S\) high-\(N\) (full circuit) *and*
  a large high-\(S\) low-\(N\) reconstruction slice. Shapley top-8 is more
  knockout-heavy (high \(N\), weaker \(S\)) — especially MCQA/ARC.
- **Gemma Game 1:** dominated by **high \(S\), low \(N\)** (copy / redundant
  pathways). That matches IOI frozen \(R<0\) and IOI \(N\approx 1\): keep-only
  looks great, deletion does not unlearn.
- **Influence / `eap_syed` / matched ACDC:** mass in low-\(S\) low-\(N\)
  (wrong set) on Gemma.

This is why [`APPLICATIONS_METRICS.md`](APPLICATIONS_METRICS.md) splits
\(\alpha\): unlearning wants \(\alpha=0\) (Dallas PF20: remove flips to Texas);
steering wants \(\alpha=1\) (disjoint copy set). v3 paper tables use
\(\alpha=0.5\) and should say so.

---

## 8. What v3 supports vs what it does not

**Supports**

- RQ1 on **pruned** MIB graphs, budget \(\le 8\): Game 1 \(\gg\) influence and
  feature AtP; ported ACDC is not a sparse competitor.
- RQ2: logit-gap Game 2 is nearly disjoint on 600 prompts (ABR and FP).
- RQ3: Gemma IOI attention-mediated vs Llama feature-mediated, at \(n=100\).
- RQ4: \(6\times\) Gemma width does not fix IOI freeze; family does.
- KL rescoring is **wired and populated** (600 sidecars). Frozen Gemma IOI
  KL-F is near 0 (`0.11 ± 0.14`) — logit-gap \(F\) is not a distribution match.

**Does not support (yet)**

- Unpruned / unbudgeted MIB tables (that is v4).
- A complete Shapley-gold comparison on Gemma.
- Graph EAP as a 200-prompt baseline.
- Track B at scale (one IOI prompt per CLT).
- InterpBench gold-circuit recovery in these roots.
- Spline-CLT vs linear CLT (no Spline graphs in v3).
- Seeds 1–2, or bootstrap CIs beyond prompt SEM.
- “Game 1 is cheaper than Shapley on Gemma” (Shapley did not finish).

**v4 should change:** `node_threshold=1.0`, omit `--budget`, Pass A =
influence only, Pass B = `eap_syed` + ported ACDC, Pass C = edge originals,
Shapley later. Dallas already ran that *method* suite on one unpruned prompt.
