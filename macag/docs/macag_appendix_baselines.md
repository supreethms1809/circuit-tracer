> **Part of the MACAG docs pack.** Hub: [`macag.md`](macag.md). Baseline comparison details and MACAG vs ACDC algorithms (former Appendices A and G).
>
> Method *definitions* live here. **v3 numbers** (Game 1 vs influence / `eap_syed` /
> ported ACDC / Shapley) are in [`macag_experiments_v3.md`](macag_experiments_v3.md).
> Track A vs Track B ports: [`baseline_originals_and_ports.md`](baseline_originals_and_ports.md).

## Appendix A: Baseline Comparison Details

This appendix expands the baseline table in [§9.3](macag_appendix_legacy.md#93-baselines-and-evaluation-protocol)
into a per-method assessment: is the method a suitable baseline, how should it be
run against MACAG, what does it share with MACAG, where it differs, and which
metrics make the comparison fair. The unifying view is the coalitional game
$(N, v)$ of [§3.0](macag_foundations.md#30-the-underlying-coalitional-game): every baseline's *selected
set* is scored under the same $v$ at the same $\alpha$ ([J.A1](macag_appendix_legacy.md#ja-baselines-and-the-shared-characteristic-function)).
Influence and EAP *rank* from graph-derived scores alone (zero oracle calls); ACDC,
Game 1, and Shapley/Banzhaf select using interventional $v$. Do **not** claim that
every method ranks under one $v$.

A recurring caveat: **ACDC and EAP operate natively on the model's own components
(attention heads, MLPs) and edges, whereas MACAG operates on CLT/SAE feature
nodes.** A clean head-to-head therefore needs either (a) the baseline's *selection
rule ported onto the CLT node set* (the apples-to-apples version, recommended for
the headline table), or (b) a comparison only at the behavioral-circuit level
(faithfulness-vs-size), acknowledging the granularity mismatch.

### A.1 ACDC (Conmy et al., 2023) — suitable: primary circuit-discovery baseline

- **Suitable?** Yes. It is the canonical automatic circuit-discovery method and the
  closest prior work to Game 1.
- **Similar:** Same goal — a minimal subgraph faithful to a behavior, found greedily
  by ablation.
- **Different:** Top-down *edge* pruning on the *native* graph; corrupted-activation
  patching with a KL threshold $\tau$; measures necessity only (effect of removal).
  MACAG is bottom-up *node* selection over CLT features, zero-ablation, an explicit
  sparsity-penalized utility, and scores sufficiency **and** necessity.
- **How to run it:** (a) *Ported* — apply ACDC's "remove if $\Delta$metric $< \tau$"
  rule to the CLT node set, sweeping $\tau$ to trace a size/faithfulness curve; this
  isolates **search direction** (prune vs. grow) on identical candidates. (b)
  *Native* — run ACDC on heads/MLPs and compare circuits as behavioral predictors.
- **Metrics:** faithfulness (Δ logit-gap or KL) vs. circuit size curve; oracle /
  forward-pass count; node-set agreement (Jaccard) where granularity permits.

### A.2 EAP / attribution patching (Syed et al., 2023; Nanda, 2023) — suitable: the most important baseline

- **Suitable?** Yes — and it is the baseline that most directly tests MACAG's
  thesis.
- **Similar:** Produces an importance score per node/edge from which a top-$k$
  circuit is read off.
- **Different — the crux:** EAP is a **first-order Taylor approximation** of
  activation patching (one backward pass). The circuit-tracer attribution graph's
  edge weights are *built from exactly this kind of gradient×activation score.* So
  "MACAG vs. EAP-top-$k$" asks the framework's reason for existing: do *real
  forward-pass interventions* recover faithful sets that the local-linear scores
  miss? If MACAG $\approx$ EAP, the method adds little; if MACAG wins on
  interacting / non-submodular features, that is the justification for the whole
  approach.
- **How to run it:** score every candidate node by attribution patching, take
  top-$k$ at the same sizes MACAG produces, and also correlate EAP scores with
  MACAG's greedy marginal gains.
- **Metrics:** faithfulness at matched $k$; **Spearman rank correlation** between
  EAP score and MACAG marginal gain (divergence localizes where linearity breaks);
  cost (1 backward pass vs. $O(|E^*|\cdot|C|)$ forwards).

### A.3 Top-k influence — suitable: the cheap "is search needed?" floor

- **Suitable?** Yes, as the trivial lower-bound selector.
- **Similar:** Uses the graph's own `influence` metric to choose $k$ nodes.
- **Different:** No interventions, no interaction modeling, no contrastive notion —
  pure magnitude ranking. Already present as a candidate policy
  ([§6.3](macag_implementation.md#63-candidate-policies)); here it is frozen as a *selector*.
- **How to run it:** take the $k$ highest-influence nodes and score the set's
  faithfulness directly. This is the floor MACAG must clear; if MACAG cannot beat
  it, the search is not earning its cost.
- **Metrics:** faithfulness at matched $|E|$; overlap (Jaccard) with MACAG's set.

### A.4 Shapley / Banzhaf (gold) — suitable: upper bound, not a competitor

- **Suitable?** Yes, but as a **gold-standard reference**, not a rival selector.
- **Similar:** Defined over the *same* characteristic function $v$
  ([§3.6](macag_foundations.md#36-relation-to-shapley-and-banzhaf-credit)).
- **Different:** Solves credit **assignment** (per-feature value), not set
  **selection**; far more expensive (averages over $2^{|C|}$ coalitions,
  MC-estimated). Banzhaf is the natural second reference (all-coalitions average,
  less order-sensitive for strongly interacting features) and is implemented
  alongside Shapley (`estimate_banzhaf`).
- **Now built and run for MACAG** (`macag/baselines/shapley_select.py`): MC
  permutation Shapley (antithetic) and MC Banzhaf, both over the MACAG oracle's $v$;
  `attribution/shapley.py` was not reused — canonical statement and reasons in
  [§3.6](macag_foundations.md#36-relation-to-shapley-and-banzhaf-credit). Validated on toy oracles
  and run on the 60-prompt nonlinear benchmark (§10.7/C.7: Shapley-gold faith\@8
  4.72 at 32 495 oracle calls); IOI/multi-hop real-graph numbers still pending.
- **How to run it:** `run_baselines` includes `shapley` in its default method list
  (`banzhaf` is opt-in via `--methods`; knobs: `--shapley-permutations`,
  `--banzhaf-samples`, `--shapley-seed`, `--no-antithetic`); it ranks features by
  estimated credit, measures how well every other method's evidence recovers the
  top-Shapley features (`agreement_vs_shapley`, falling back to Banzhaf as gold if
  only it ran), and reports each method's oracle calls against Shapley's.
- **Metrics:** set agreement (precision@$|E|$, Jaccard) and rank correlation vs.
  the gold ranking; **cost ratio** — the "matches gold at an order of magnitude
  less compute" claim.

### A.5 Metrics required across all baselines

For a fair, single comparison table every method should be reported on:

| Axis | Metric |
|------|--------|
| Faithfulness | raw Δ logit-gap (per [§2.3](macag_framework.md#23-attention-freezing-and-the-error-floor)) and/or KL, **at matched evidence size** |
| Parsimony | evidence size at matched faithfulness |
| Cost | oracle forward passes (plus backward passes for EAP) |
| Agreement | Jaccard / precision@$k$ vs. Shapley-gold and pairwise |
| Linearity diagnostic | Spearman(EAP score, MACAG marginal gain) |
| Stability | cross-seed Jaccard of selected sets |

**Two non-negotiables.** (1) **Always compare at matched set size** — otherwise the
faithfulness-vs-size confound reappears (the same confound flagged for the §10 CLT
tables). (2) **Report raw, not normalized, faithfulness** for the attention-mediated
baselines, because `recoverable_range` is unreliable there
([§2.3](macag_framework.md#23-attention-freezing-and-the-error-floor)).

**Implementation status.** All selectors and the head-to-head harness are now
implemented: `macag/baselines/` (influence B2.1, Shapley/Banzhaf-gold B2.2, EAP
B2.3, ported ACDC B2.4, brute-force B3.2) driven by
`python -m macag.cli.run_baselines` (B2.0), which already emits every metric in
the table above except cross-seed stability (one run = one seed; loop seeds for
that row). ACDC-*native* (heads/MLPs granularity) is still open. Beyond the
toy-oracle unit tests ([§3.6](macag_foundations.md#36-relation-to-shapley-and-banzhaf-credit)), the
real-graph sweep **has now run on the 60-prompt nonlinear benchmark** (§10.7/C.7,
`results/macag_nonlinear_connected/`); the IOI/multi-hop case-study graphs and the
cross-seed stability row are still open.

---

## Appendix G: MACAG vs ACDC Algorithmic Differences

A side-by-side of the two algorithms, for the related-work / method-positioning
section. ACDC pseudocode reproduced from Conmy et al. (2023, Algorithm 1).

### G.1 The two algorithms side by side

**ACDC (Conmy et al. 2023, Algorithm 1).** Top-down **edge** pruning.

```
Data:   computational graph G, clean dataset (x_i), corrupted datapoints (x'_i),
        threshold τ > 0
Result: subgraph H ⊆ G
1  H ← G                              # start from the FULL graph
2  H ← H.reverse_topological_sort()  # output node first
3  for v in H:
4      for w in parents(v):
5          H_new ← H \ {w → v}        # tentatively REMOVE candidate edge
6          if  D_KL(G ‖ H_new) − D_KL(G ‖ H) < τ:   # removal barely changes output
7              H ← H_new              # drop the edge permanently
8  return H
```

**MACAG Game 1.** Bottom-up **node** selection (the greedy of [§4.2](macag_game1.md#42-algorithm-greedy-hill-climbing), condensed).

```
Data:   attribution graph G=(C,A), oracle O, target y (+ foil), α, λ, ε, budget B
Result: evidence set E ⊆ C
1  E ← ∅                                   # start from the EMPTY set
2  (optional) C ← prefilter top-k by singleton utility
3  repeat:
4      for n in C \ E:                      # consider ADDING each node
5          gain(n) ← U(E ∪ {n}) − U(E)      # U = α·suff + (1−α)·nec − λ|E|
6      n* ← argmax_n gain(n)
7      if gain(n*) ≤ min_gain or |E| ≥ B or stop(ε): break
8      E ← E ∪ {n*}                          # ADD the most useful node
9  return E
```

### G.2 Where they differ (point by point)

| Axis | ACDC | MACAG Game 1 |
|------|------|--------------|
| **Search direction** | top-down: start at full $G$, **remove** | bottom-up: start at $\emptyset$, **add** |
| **Unit operated on** | **edges** $w\to v$ of the model's native graph | **nodes** (features) of the attribution graph |
| **Granularity** | model components (attention heads, MLPs) | transcoder/CLT features |
| **Stop rule** | per-edge threshold: drop if KL-increase $<\tau$ | global utility: stop at $\le$ `min_gain`, budget $B$, or $\varepsilon$-faithfulness |
| **Objective shape** | implicit; one threshold $\tau$, no size term | explicit $U=\alpha\,\text{suff}+(1-\alpha)\,\text{nec}-\lambda|E|$ (sparsity priced in) |
| **Causal quantity** | **necessity** only (effect of *removing* an edge) | **sufficiency *and* necessity** (keep-only *and* remove modes) |
| **Intervention** | **resample** patching from corrupted prompts $(x'_i)$ | **zero**-ablation (configurable), frozen/unfrozen attention |
| **Behavioral metric** | $D_{KL}(G\,\|\,H)$ vs. the full model, over a dataset | target−foil **logit gap** (configurable) on one prompt |
| **Error/floor handling** | none (native graph has no transcoder error term) | explicit **error floor** + recoverable-range normalization ([§2.3](macag_framework.md#23-attention-freezing-and-the-error-floor)) |
| **Contrastive variant** | — | **Game 2** (no ACDC analog) |
| **Theory** | greedy threshold heuristic | coalitional game; submodular $(1{-}1/e)$ where it holds; Game 2 = exact potential game ([§3](macag_foundations.md#3-game-theoretic-foundations)) |
| **Per-feature credit** | — | explicit Shapley/Banzhaf link ([§3.6](macag_foundations.md#36-relation-to-shapley-and-banzhaf-credit)) |
| **Output** | a faithful **sub-circuit of edges** | a **minimal evidence set of feature nodes** (+ contrastive split) |

### G.3 Why the differences matter (not just cosmetic)

1. **Add-vs-remove changes what you can detect.** ACDC's removal test is a
   *necessity* test: an edge is kept only if deleting it hurts. It is therefore
   blind to **jointly-necessary-but-individually-redundant** structure in the wrong
   direction, and it cannot directly certify *sufficiency*. MACAG's keep-only mode
   measures sufficiency outright, so a MACAG evidence set is scored on both axes
   (measured, not certified — the greedy carries no optimality certificate);
   the trade-off is that bottom-up greedy can stall on pure synergy (two features
   each useless alone) — the failure mode analyzed in
   [§3.2](macag_foundations.md#32-the-value-function-and-submodularity), which ACDC's top-down deletion
   does not share. The two methods thus have **complementary blind spots**, a point
   worth making explicitly in the paper.
2. **Edges vs feature nodes changes the object of study.** ACDC yields a wiring
   diagram among heads/MLPs; MACAG yields the **minimal set of interpretable
   features** that carries a prediction — directly usable for steering/annotation and
   comparable *across transcoder variants on the same footing*. This is what makes
   MACAG encoder-agnostic in a way edge-pruning the native graph is not.
3. **Zero-ablation + logit-gap vs resample + KL is a different causal question.**
   ACDC asks "which edges keep the *whole output distribution* close to the model
   under resampling?"; MACAG asks "which features are sufficient/necessary for *this
   target-vs-foil decision* under ablation?". The MACAG question is sharper for
   contrastive behaviors (hence Game 2) but depends on a good foil; the ACDC question
   is distribution-level and foil-free. (Both choices are configurable in MACAG —
   `score_kind` can be KL, and per-node ablation values support resample-style
   baselines — so the gap is narrowable for a controlled comparison; see
   Appendix A.1.)
4. **Explicit sparsity-penalized utility vs a single threshold.** MACAG prices
   evidence size into the objective ($-\lambda|E|$) and exposes the
   faithfulness-vs-size trade-off as a tunable curve; ACDC exposes it only
   indirectly through $\tau$. This makes MACAG's parsimony directly optimizable and
   directly plottable (roadmap B3.1).
5. **The error floor only exists for transcoder circuits.** ACDC on the native graph
   has no reconstruction-error term, so it never confronts the negative-`recoverable_range`
   regime. MACAG must (and does) handle it — which is also what turns into the
   **attention-mediation diagnostic** (§10.4) that ACDC, as specified, cannot
   produce.

### G.4 One-line summary

> ACDC **deletes edges** of the model's native graph until the output distribution
> would move too much (a top-down, necessity-only, KL-thresholded pruner). MACAG
> Game 1 **adds feature nodes** of a transcoder attribution graph while a
> sparsity-penalized sufficiency+necessity utility keeps rising (a bottom-up,
> game-theoretic selector), and MACAG adds a contrastive second game and a
> Shapley-grounded notion of per-feature credit that ACDC has no analog for.

---
