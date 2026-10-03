# MACAG: Minimal And Contrastive Attribution Games

> **On the name.** *Minimal* names the aspiration — the objective Game 1
> optimizes — while *small* names the guarantee it delivers. Game 1 greedily
> maximizes a sparsity-penalized utility over a value function that is not in
> general submodular ([§3.2](macag_foundations.md#32-the-value-function-and-submodularity)),
> so the selected set is small by construction and minimal only by intent, never
> by certificate.

This file is the **hub** for the MACAG writing pack. The former single
`macag.md` (~3.7k lines) is split across the files below. **Cite current
numbers from the v3 campaign and the Dallas–Austin case study**, not from the
legacy two-hop / 2026-07-18 audit tables.

## What to read

| File | Contents |
| --- | --- |
| [`macag_motivation.md`](macag_motivation.md) | Gap, prior work, RQs, contributions, wording guardrails |
| [`macag_framework.md`](macag_framework.md) | Notation, oracle, F/S/N, freeze + error floor, KL rescoring |
| [`macag_foundations.md`](macag_foundations.md) | Coalitional \(v\), submodularity, Game 2 potential, Shapley/Banzhaf |
| [`macag_game1.md`](macag_game1.md) | Game 1 objective, greedy solver, connectivity |
| [`macag_game2.md`](macag_game2.md) | Game 2 ABR / fictitious play, overlap, foil resolution |
| [`macag_implementation.md`](macag_implementation.md) | Graph wrapper, candidates, ReplacementModel, pipeline |
| **[`macag_experiments_v3.md`](macag_experiments_v3.md)** | **TMLR-250 v3 MIB results (3 CLTs × 200 prompts)** — current paper tables |
| **[`macag_dallas_austin.md`](macag_dallas_austin.md)** | **Dallas–Austin Llama case study** — unpruned graph, KL vs logit-gap, \(\alpha\), Track A/B |
| [`macag_discussion.md`](macag_discussion.md) | Interpretation, threats, limitations, conference-readiness |
| [`macag_references.md`](macag_references.md) | Bibliography |
| [`macag_appendix_baselines.md`](macag_appendix_baselines.md) | Baseline comparison notes + MACAG vs ACDC algorithms |
| [`macag_appendix_roadmap.md`](macag_appendix_roadmap.md) | Older submission roadmap (launch now: recipe + v4 runbook) |
| [`macag_appendix_background.md`](macag_appendix_background.md) | CLT / graph / intervention primer + glossary |
| [`macag_appendix_related.md`](macag_appendix_related.md) | Contemporary methods positioning |
| [`macag_appendix_legacy.md`](macag_appendix_legacy.md) | **Do not cite.** Pre-v3 two-hop tables, Appendix C, 2026-07-18 audit |

Operational companions (not split from this pack):

- [`TMLR_EVAL_RECIPE.md`](TMLR_EVAL_RECIPE.md) — fairness contract for paper tables
- [`run_todo.md`](run_todo.md) — **v3** launch (the campaign whose numbers are in `macag_experiments_v3.md`)
- [`run_todo_v4.md`](run_todo_v4.md) — **v4** launch (unpruned, unbudgeted; do not mix with v3 outdirs)
- [`baseline_method_map.md`](baseline_method_map.md) / [`baseline_originals_and_ports.md`](baseline_originals_and_ports.md)
- [`APPLICATIONS_METRICS.md`](APPLICATIONS_METRICS.md) — unlearning / steering / auditor
- [`INTERPRETATION_GUIDE.md`](INTERPRETATION_GUIDE.md) — how to read JSON

---

## Abstract

Mechanistic-interpretability pipelines now summarize a model's computation on a
prompt as an **attribution graph**: nodes are transcoder/SAE features and edges
are *observational* attribution scores (first-order / local-linear). Such a
graph reveals *structure* but leaves open the two questions a circuit claim
rests on: (i) **which small set of features is causally responsible** for the
prediction, and (ii) **whether a competing answer is carried by different
features**. We introduce **MACAG** (Minimal And Contrastive Attribution
Games), which **selects and tests causal evidence** by intervening on feature
nodes rather than reading off edge weights. **Game 1** greedily maximizes an
\(\alpha\)-mix of sufficiency and necessity minus a sparsity penalty.
**Game 2** allocates features between a target agent and a foil agent under an
overlap penalty. The scoring layer is a coalitional game: gold per-feature
credit is the Shapley value of the same \(v\) (Monte Carlo over the MACAG
oracle), and Game 2 is an exact potential game.

**Current empirical base (TMLR-250 v3, seed 0).** Three public CLTs —
gemma2-426k, gemma2-2.5M, llama32-524k — on MIB **IOI 100 + MCQA 50 +
ARC-Easy 50** (200 prompts each, 600 total). Protocol: `score_kind=logit_gap`,
Game 1 `--freeze-mode both`, **budget 8**, `stop_metric=raw_relative`,
prefilter/connected off, **pruned graphs** (`node_threshold=0.8`). Game 2 ran
ABR **and** fictitious play on every prompt. Track A baselines on the same
graphs: influence, feature AtP (`eap_syed`), ported ACDC, and (Llama)
Shapley-gold. Details and tables: [`macag_experiments_v3.md`](macag_experiments_v3.md).

Headline v3 findings (means \(\pm\) SEM, frozen Game 1 unless noted):

- **Game 1 vs cheap selectors at budget 8** (Game 1 mean \(|E^\star|\approx 7.1\)–\(7.9\), not always 8): Llama F \(13.03 \pm 0.57\) vs
  influence \(2.17 \pm 0.35\) vs `eap_syed` \(4.25 \pm 0.32\). Gemma-426k F
  \(8.50 \pm 0.18\) vs influence **negative** (\(-1.57 \pm 0.29\)) vs
  `eap_syed` \(1.29 \pm 0.11\). Ported ACDC **matched to \(k\le 8\)** collapses
  (mean \(|E|\approx 1.6\)–\(1.8\), F \(1.08\) / \(0.94\) / \(3.09\)); its
  high-F \(\tau\) settings are **dense** (hundreds of nodes).
- **Attention mediation (IOI):** gemma2-426k **83/100 `attention_mediated`**,
  gemma2-2.5M **89/100**, llama32-524k **97/100 `feature_mediated`** (0 IOI
  attention flips). Frozen recoverable range on Gemma IOI is negative
  (\(-10.5\) / \(-14.7\)); do not quote normalized F there.
- **Game 2 overlap:** mean overlap rate \(0.44\%\) / \(0.10\%\) / \(1.01\%\)
  (Gemma-426k / 2.5M / Llama); exact-zero overlap on **188 / 197 / 172** of
  200 ABR runs. ABR “converged” on 193–195/200 in \(\approx 2.1\) iterations;
  FP reports converged on 200/200. Pair *identity* is still solver-dependent.
- **Shapley-gold** is **not** complete: Llama 196/200, Gemma-426k **7/200**
  (ARC-Easy only), Gemma-2.5M **0/200**. On Llama, top-8 Shapley F
  \(11.12 \pm 0.53\) vs Game 1 \(13.03\) — expected when \(v\) has synergies
  (Game 1 optimizes the *set*, Shapley ranks average marginals). Do not call
  Shapley “exact ground truth” at 64 permutations.

**Dallas–Austin** (one unpruned Llama prompt, 2028 feature candidates) is the
protocol laboratory: KL-as-selection does not copy the next-token distribution;
\(\alpha=0\) vs \(\alpha=1\) yield **disjoint** knockout vs copy circuits;
feature AtP is a weak KL selector; native-edge ACDC/EAP are a **different
object** (Track B). That case study is why v4 moved to unpruned graphs,
unbudgeted Game 1, logit-gap selection, and KL-as-rescore.
See [`macag_dallas_austin.md`](macag_dallas_austin.md).

> **Phrases to avoid:** “smallest set”; “converges to Nash” (existence only);
> “beats Conmy ACDC / Syed EAP” when only the **port** was run; citing
> graph-EAP / `eap_edge` as if they were complete v3 Track A rows (graph EAP
> is unavailable on most prompts; edge methods exist for **one** IOI prompt
> per CLT); quoting pre-v3 Appendix C / J numbers as current.

---

## Contributions (status)

| ID | Claim | Where specified | Current evidence |
| --- | --- | --- | --- |
| C1 | Encoder-agnostic graph+oracle selector/tester | [motivation](macag_motivation.md), [implementation](macag_implementation.md) | Implemented; v3 on three hub CLTs |
| C2 | Coalitional \(v\), submodularity honesty, Shapley on same \(v\) | [foundations](macag_foundations.md) | Formal; Shapley **partial** on v3 (Llama almost complete) |
| C3 | Game 1 (suff+nec, freeze protocol, raw_relative stop) | [game1](macag_game1.md), [framework](macag_framework.md) | v3 dual-freeze, budget 8, 600/600 Game 1 JSONs |
| C4 | Game 2 potential game; ABR + FP | [game2](macag_game2.md) | v3 ABR+FP on 600/600; overlap near 0, not identically 0 |
| C5 | Attention-mediation diagnostic | [framework §2.3](macag_framework.md#23-attention-freezing-and-the-error-floor) | v3 IOI: Gemma attention-mediated, Llama feature-mediated |

---

## Campaigns (do not mix outdirs)

| Campaign | Outdirs | Graphs | Game 1 | Role |
| --- | --- | --- | --- | --- |
| **v3 (current tables)** | `macag_mib_tmlr250v3_{h200,gemma25m,llama}` | `node_threshold=0.8` | dual-freeze, **budget 8** | Completed seed-0 MIB 200×3 |
| **Dallas–Austin** | `macag_dallas_austin_llama/.../dallas-austin/` | **unpruned** (2028 features) | unbudgeted / prefilter grid | Protocol + \(\alpha\) + Track B case study |
| **v4 (in flight / next)** | `macag_mib_tmlr250v4_*` | unpruned (`1.0`) | unbudgeted | Paper recipe; **do not reuse v3 roots** |
| Pre-v3 / two-hop | various `macag_mib_*` without `tmlr250v3` | mixed | mixed | [legacy appendix](macag_appendix_legacy.md) only |
