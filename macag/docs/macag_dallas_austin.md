# Dallas–Austin case study (Llama-3.2-1B)

> **Part of the MACAG docs pack.** Hub: [`macag.md`](macag.md).
> Method-port definitions and the long tables also live in
> [`baseline_originals_and_ports.md`](baseline_originals_and_ports.md)
> (section “Dallas–Austin case study”). This file is the MACAG-facing
> writeup: what the prompt taught the framework, the games, and v3 \(\to\) v4.
>
> **Not a MIB table.** One prompt, one unpruned CLT graph, every selector we
> actually ran. Paper-scale numbers: [`macag_experiments_v3.md`](macag_experiments_v3.md).

Artifacts:

```
/gscratch/$USER/macag_dallas_austin_llama/llama32-524k/dallas-austin/
  graphs/dallas-austin.json
  kl_divergence/     # job 13913 Track A + Track B + KL Game 1 ablations
  logit_gap/         # jobs 14031–14035 Game 1 α / prefilter grid
```

Launch pattern (method-exclusive GPU shards after shared setup):
`scripts/run_macag_dallas_llama_setup.sh`,
`scripts/run_macag_dallas_llama_method.sh`,
`scripts/slurm/launch_macag_dallas_llama.sh`.

---

## 1. Why this prompt

Two-hop factual: *Dallas \(\to\) Texas \(\to\) Austin*, with a length-matched
corrupt *Houston* prompt for AtP / native edges.

| Knob | Value |
| --- | --- |
| Prompt | `Fact: The capital of the state containing Dallas is` |
| Target / foil | ` Austin` / ` Texas` |
| Corrupt (Syed + Track B) | `Fact: The capital of the state containing Houston is` |
| Model / CLT | `meta-llama/Llama-3.2-1B` / `mntss/clt-llama-3.2-1b-524k` |
| Graph | **`node_threshold=1.0` (unpruned)**, BOS-prepended, 11 tokens |
| Graph size | 2200 nodes, 1,140,436 links |
| Candidates | **2028** `cross layer transcoder` (+ 151 error, 11 embed, 10 logit) |
| Clean logit-gap | \(s_{\mathrm{all}} = 1.625\) (CLT) / \(1.662\) (native-edge scorer) |

Neuronpedia-style prune **0.8 / 0.98 left 279 features** on earlier exports of
this prompt. Selector comparisons on that graph are a different experiment.
v3 MIB used `node_threshold=0.8`; Dallas is why v4 requires `1.0`.

Feature IDs are `{layer}_{feat}_{ctx_idx}`; last content token is `ctx_idx=10`.
Hub feature **`0_25454_10`** dominates almost every KL run.

---

## 2. Track A — KL Game 1 (job 13913)

Selection on \(\mathrm{score}=-\mathrm{KL}(P_{\mathrm{full}}\|P_{\mathrm{int}})\),
\(\alpha=0.5\), \(\varepsilon=0.1\), **no prefilter**, 2028 candidates,
dual-freeze, **no budget**. \(s_{\mathrm{all}}\equiv 0\). Keep-only of a sparse
set is **not** “the circuit copies the model” unless keep KL \(\approx 0\).

| Leg | \(\lvert E\rvert\) | \(F\) | \(S\) | \(N\) | keep_only \(=-\mathrm{KL}\) | remove | empty | \(R\) | calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen | 5 | 6.75 | 2.44 | 11.06 | **−6.88** | −11.06 | −9.31 | 9.31 | 24,310 |
| Unfrozen | 6 | 10.08 | 9.72 | 10.44 | **−5.41** | −10.44 | −15.13 | 15.13 | 28,354 |

Frozen \(E^\star\): `0_12390_10`, `0_25454_10`, `15_16228_10`, `3_18384_10`,
`9_25557_10`. Unfrozen shares only `{0_25454_10, 15_16228_10}` (Jaccard 0.22).

**Read:** Game 1 found a 5–6 node KL *gain vs empty*, not a reconstruction.
Keep-only is still 5–7 nats from the full model. Necessity is large because
`remove` is *worse than empty*.

### Prefilter × \(\varepsilon\) (same graph, new JSON stems)

| Config | frozen \(\lvert E\rvert\) / \(F\) / calls | unfrozen \(\lvert E\rvert\) / \(F\) / calls | frozen Jaccard vs full |
| --- | --- | --- | ---: |
| Full, \(\varepsilon=0.1\) | 5 / 6.75 / 24,310 | 6 / 10.08 / 28,354 | 1 |
| PF20, \(\varepsilon=0.1\) | **1** / 4.69 / 4,098 | 5 / 8.44 / 4,230 | **0.20** |
| PF50, \(\varepsilon=0.01\) | 13 / 7.66 / 5,178 | 8 / 9.52 / 4,788 | 0.13 |
| PF500, \(\varepsilon=0.01\) | 24 / 8.61 / 27,460 | 19 / 12.75 / 22,680 | 0.16 |

PF20 frozen collapsed to `0_25454_10`: \(\varepsilon=0.1\) vs a huge first
step stops immediately. Two of five full-run frozen nodes sit at singleton
ranks **174 and 531** — prefilter by singleton utility **throws away
complements**. Treat PF as a speed/identity tradeoff, not a free
approximation.

---

## 3. Track A — other selectors on the frozen KL oracle

Same 2028 candidates, `freeze_attention=true`, budget 128 for prefix curves.

**Influence** (graph `influence_raw`, 0 forwards): F@1 = 4.61; **F@5 = 5.52**
vs Game 1’s 5-node F **6.75**; F@128 = 9.12; mean \(F\) over \(k=0..128\)
(trapezoid AUC\(/128\)) = **8.26**. Game 1 wins at natural size. Influence only
catches up at tens of nodes.

**Feature AtP / `eap_syed`** (Houston corrupt; decoder·residual-grad): ranking
does **not** start at the Game 1 hub. F@5 = **0.11**; F@128 = 0.30; mean \(F\)
over \(k=0..128\) = **0.47**. First-order AtP is a **weak selector** under KL
\(v\) on this prompt. Report it; do not drop the negative.

**Ported ACDC** (top-down \(\tau\) on feature nodes, 24,232 calls):

| \(\tau\) | kept | \(F\) (KL \(v\)) |
| ---: | ---: | ---: |
| 0.001 | 1919 | 12.32 |
| 0.1 | 167 | **12.45** |
| 0.5 | **14** | **5.56** |

\(\tau=0.5\) (closest to Game 1 size): 14 nodes, F **below** Game 1@5.
High-F ACDC is the dense end. Same story as v3 MIB §5.2, here on an unpruned
graph.

**Shapley** (64 antithetic perms): **ran** on 13913 wid5 to **56/64** in 24h;
JSON was not flushed. A follow-up job was left running. Influence/AtP/ACDC
do **not** substitute for the missing \(\phi\) table.

---

## 4. Track A — logit-gap Game 1 (jobs 14031–14035)

Same graph, **new** `logit_gap/` kwargs (KL artifacts untouched). No
full-universe logit-gap Game 1 (24h cap); all runs used a prefilter.

Auditor signature:

- Frozen empty \(\approx 0\) to \(+0.14\) \(\Rightarrow R \approx +1.5\) to
  \(+1.7 > 0\). **Feature-mediated** (agrees with v3 Llama).
- Unfrozen empty \(\approx +4.0\) to \(+4.3 > s_{\mathrm{all}}\). \(R<0\).
  Unfrozen \(F\) can look fine while relative faithfulness is meaningless.

### \(\alpha\) split at PF20, \(\varepsilon=0.1\) (the application result)

Frozen, same 20-node pool, only \(\alpha\) changes:

| \(\alpha\) | Role | \(\lvert E\rvert\) | \(F\) | \(S\) | \(N\) | keep_only | remove |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **0** | unlearning (pure N) | 4 | 7.53 | **−0.39** | **7.53** | **−0.25** | **−5.91** |
| **0.5** | paper F | 6 | 5.40 | 4.02 | 6.78 | **4.06** | **−5.16** |
| **1** | copy / steer (pure S) | 4 | 4.66 | **4.66** | **−0.25** | **4.69** | **+1.88** |

- \(\alpha=0\): remove flips the gap to **−5.91 (Texas wins)**. keep-only
  \(\approx 0\) — knockout does **not** reconstruct Austin.
  \(E^\star=\) `0_25454_10`, `0_28426_5`, `1_26758_5`, `9_25557_10`.
- \(\alpha=1\): keep-only **4.69 overshoots** clean 1.625. remove \(=+1.88\) —
  ablating \(E^\star\) does **not** kill the fact. Completely **disjoint**
  from the \(\alpha=0\) set (frozen Jaccard **0**).
- \(\alpha=0.5\) does both jobs at once (keep overshoots, remove flips) with a
  mixture set. Default \(F\) is not “the” circuit.

This is the empirical backbone of [`APPLICATIONS_METRICS.md`](APPLICATIONS_METRICS.md).

Larger pools at \(\alpha=0.5\) grow \(\lvert E^\star\rvert\) and \(F\) (PF500
frozen: 32 nodes, keep 10.56, remove −7.28) without recovering PF20’s identity
(Jaccard 0.06). Same warning as KL prefilter.

---

## 5. Track B — native-edge ACDC / EAP

Factorized Q/K/V residual graph, **195,865** edges. Corrupt = Houston.
**Do not Jaccard against Game 1.**

**`eap_edge`** (true Syed AtP): 14 fwd + 1 bwd, 31.5 s. Prefix sweep
(corrupt-patch the complement):

| \(k\) | circuit logit-gap | KL(full ‖ circuit) |
| ---: | ---: | ---: |
| 1–32 | \(\approx\) **2.97** | \(\approx\) **0.48** |
| 4096 | 1.48 | 0.24 |
| all 195,865 | 1.66 | 0 |

Small AtP sets sit at gap **2.97**, which **is** this scorer’s empty/corrupt
gap (`all=1.66`, `empty=2.97`). Keep-only of \(k=8\): \(S\approx 0\),
\(N=-0.19\), \(F=-0.09\). A tiny native EAP circuit here is “leave the model
in the Houston-patched world,” not restore Dallas\(\to\)Austin.

**`acdc_edge`** (true Conmy \(\tau=0.1\), KL prune, resample-corrupt): ~1.6 h.
Final circuit **3 edges**: `A15.7->Resid End`, `MLP 12->MLP 13`,
`MLP 13->Resid End`. Native scores: gap 2.97, KL 0.48 — same empty-like
point. Logit-gap S/N/F: \(S\approx 0\), \(N=-0.54\), \(F=-0.27\).

v3 Gemma IOI_0000 shows the same Track B geometry at a different task: edge
@8 inverts the gap; EAP@4096 recovers it; native ACDC stays tiny and inverted.

---

## 6. Game 2 on this prompt (incomplete, but informative)

KL ABR (13913 wid1, checkpoint `macag_game2_abr.ckpt.json`): **no \(\varepsilon\) stop**. Austin
side grew to **39** nodes (first five = frozen KL Game 1 \(E^\star\) in greedy
order: hub `0_25454_10` first). Texas side reached **29** nodes before the 24h
kill. **18 shared**, Jaccard **0.36**. First eight Austin nodes are all on the
Texas side. \(\beta=0.2\) did not disjoint them.

Cause: **KL is target-free.** Both agents climb the same KL-gain ridge
(`0_25454_10` first, gain \(\approx 4.6\)–\(4.7\)). A working foil circuit
should move *away* from the clean distribution. **Do not select Game 2 on KL.**
v3 MIB Game 2 used logit-gap; that is why overlap stayed \(\le 1\%\).

FP never started (ABR JSON missing). Logit-gap Game 2 was not in 13913
(`SCORE_KIND=kl_divergence`). Full-universe Game 2 / Shapley on ~2k nodes
do not finish in 24h without checkpointing or a prefilter.

---

## 7. What this case study changed in the project

1. **Two tracks, two tables.** Feature ports vs native edges are not
   interchangeable rows. Native 3-edge ACDC \(F\) is not a Game 1 competitor.
2. **Unpruned graphs for paper selectors.** 279 vs 2028 is not a footnote.
   v3 MIB is still pruned; v4 recipe forbids 0.8/0.98 while claiming full CLT.
3. **Unbudgeted Game 1 for “how many features do I clamp?”** Rankings need
   you to pick \(k\); ACDC needs \(\tau\); Game 1’s \(\varepsilon\) stop is
   the product claim. v3 budget 8 was a fleet compromise (IOI timings).
4. **Logit-gap selection, KL rescore.** KL Game 1 keep-only is 5–7 nats.
   Equal-KL matching forces hundreds of features (v3 IOI_0000: ~384 to match
   EAP@4096). TMLR fairness contract item 5.
5. **\(\alpha\) is the application.** Frozen PF20 Jaccard(\(\alpha=0\),
   \(\alpha=1\)) \(=0\). Paper tables may still use \(0.5\); product claims
   must not.
6. **Prefilter is not identity-preserving.** Singleton top-20 drops
   complementary nodes at ranks 174 and 531.
7. **Feature AtP can fail on the same nodes** (AUC 0.47 vs influence 8.26 vs
   Game 1@5 = 6.75 under KL).
8. **Auditor is frozen \(R\), not unfrozen \(E^\star\).**
9. **Plan wall-clock for unbudgeted Game 2 / Shapley on ~2k nodes.**

Paper MIB v4 should **not** copy Dallas KL selection. Copy Dallas
**logit-gap + \(\alpha=0.5\) + no prefilter + unpruned**, and keep the KL /
PF / \(\alpha\) grid as the methods appendix that *justifies* that protocol.
