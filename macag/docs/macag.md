# MACAG: Minimal And Contrastive Attribution Games for Circuit Evaluation

## Abstract

Mechanistic-interpretability pipelines now summarize a model's computation on a prompt as an **attribution graph**: nodes are transcoder/SAE features and edges are *observational* attribution scores (first-order / local-linear). Such a graph reveals *structure* — which features are active and how they connect — but leaves open the two questions a circuit claim ultimately rests on: (i) **which minimal set of features is causally responsible** for the prediction, and (ii) **whether a competing answer is carried by different features**. Edge weights are a first-order approximation at a single input; they rank features by local influence, not by causal effect under intervention. We introduce **MACAG** (Minimal And Contrastive Attribution Games), a framework that addresses both questions by **intervening** on the graph's feature nodes — ablating them through the model and measuring the change in behavior (by default a target–foil logit gap) — rather than reading off attribution magnitudes. MACAG poses the search as two optimization games over the feature nodes: **Game 1 (minimal faithful evidence)** selects a small feature set scored under intervention for *both* sufficiency and necessity (an α-weighted mix), with smallness priced into the objective by an explicit sparsity penalty (optimized, not certified); **Game 2 (contrastive evidence)** allocates features between a target agent and a foil agent under an overlap penalty, revealing whether competing answers share their evidence. The framework needs only an attribution graph and an intervention oracle, so it applies on equal footing to any transcoder/SAE variant that exposes feature-level interventions (demonstrated on three public CLTs and local Spline-CLT checkpoints; per-layer SAE variants need an oracle adapter), and it casts circuit evaluation as a coalitional game: the gold-standard per-feature credit is, by construction, the Shapley value of the same characteristic function (the framework supplies this *definition* and a MACAG-oracle Monte-Carlo estimator, now run on the nonlinear benchmark — §3.6, §10.7), and the contrastive game is an exact potential game, so a pure-strategy equilibrium is guaranteed to exist. As an illustrative case study (single seed; three public cross-layer transcoders on Gemma-2 and Llama-3.2), MACAG finds target and foil features to be cleanly disjoint and yields an **attention-mediation diagnostic**: behaviors whose feature-level faithfulness becomes recoverable only once attention is unfrozen (e.g. indirect-object identification on Gemma) are distinguished from feature-mediated ones. A first quantitative baseline head-to-head (vs top-k influence, EAP, ACDC, and Shapley-gold) is now run on a 60-prompt nonlinear benchmark with bootstrap CIs and paired tests (§10.7); extending it to the multi-hop/IOI case-study graphs and to multi-seed scale is the main remaining step.

> **Abstract notes (for the paper writeup — not part of the abstract).**
> - **Provenance warning (pass 1, 2026-06-10).** Every number in this block, in
>   §10.1–10.6, and in Appendix C.1–C.6 comes from the 2026-06-03/05 runs, which
>   predate the 2026-06-09 fixes in `71a2ef6` (raw_relative-stop bug, Game 2
>   best-iterate tracking, per-solve oracle-stat reset). **Exception:** §10.7 and
>   Appendix C.7 (the nonlinear-benchmark baseline head-to-head) are a *post*-`71a2ef6`
>   re-run (`results/macag_nonlinear_connected/`) and are safe to quote. See the
>   provenance box at the top of §10 for what is and is not at risk. Do not quote the
>   pre-fix §10.1–10.6 / C.1–C.6 numbers in the paper until the post-fix re-run
>   regenerates them.
>   <!-- TODO(pass-2): refresh the §10.1–10.6 + C.1–C.6 numbers from the post-71a2ef6 re-run (§10.7/C.7 already done) -->
> - *Verified headline numbers available to cite* (sources in §10 / Appendix C, all
>   single-seed): overlap_rate **0.0 in 48/48** Game 2 runs (24 frozen + 24
>   unfrozen, 3 CLTs); gemma ACDC-benchmark `recoverable_range` **negative in 25/26
>   frozen, non-negative in 24/26 unfrozen, 23/26 strictly flipping sign**; llama
>   **0/13** negative (feature-mediated throughout); capacity control: a **~6×**
>   wider gemma CLT leaves the same reconstruction failure; cost (post-`71a2ef6`,
>   §10.7/C.7): Game 1 mean **718** oracle calls vs Shapley-gold **32 495**
>   (**44.7×** cheaper, 60/60). The older two-hop cost figures (Game 1 ~808,
>   Game 2 ~1761, >50% cache hits) are **pre-fix and provisional** (C.6 caveat —
>   counters were not reset per solve); do not cite them until re-derived.
> - *Phrases to avoid in the abstract:* "smallest set" (greedy + λ-penalty
>   optimizes toward small; minimality is not certified — see guardrails in §1.4);
>   "converges to Nash equilibrium" (only *existence* is guaranteed; the
>   implemented simultaneous-update dynamics can cycle and are protected by
>   best-iterate tracking + a fictitious-play solver, §3.4); comparative claims vs
>   EAP/ACDC/Shapley *generalized beyond the nonlinear benchmark* — the head-to-head
>   is run there (n=60, §10.7) but not yet on the IOI/multi-hop graphs (§9.3);
>   any phrasing implying the *gold per-feature Shapley credit assignment* is the
>   selector — the MACAG-oracle Shapley estimator exists and is run (§3.6, §10.7),
>   but it is a baseline reference, not Game 1's selection rule.
> - **New instrumentation since pass 2 (2026-07-01), no quotable numbers yet:**
>   (i) a selection-independent **KL faithfulness layer**
>   ([§2.5](#25-kl-rescoring-a-selection-independent-faithfulness-metric)) — built, unit-tested, wired through the drivers and
>   analyzers as `kl_faith` columns; the designated answer to the §11.3
>   circularity/single-foil objections once the stored roots are rescored; and
>   (ii) **MIB-bench integration** ([§9.5](#95-mib-bench-full-campaign-setup)) — standardized IOI/MCQA/ARC
>   prompts at full benchmark scale (500/50/570) routed to the gemma CLTs, now a
>   3-seed campaign (`results/macag_mib_seed{0,1,2}/`, in progress) plus a
>   native-component InterpBench gold-circuit validation ([§9.6](#96-interpbench-native-component-gold-circuit-validation-setup),
>   also 3-seed, in progress). Do not cite MIB, InterpBench, or KL numbers until
>   those runs finish.
> - *The third selling point* (beyond the two questions): the framework makes the
>   *evaluation conventions themselves* — attention freezing and the error floor —
>   explicit and measurable, which is what produces the attention-mediation
>   diagnostic (§1.1 gap 3, §2.3). Consider promoting this from "case-study
>   outcome" to "framework property" in the abstract's second paragraph.

---

## 1. Motivation

### 1.1 The gap: an attribution graph is structure, not a causal verdict

A circuit-tracing pipeline (e.g. Anthropic's circuit-tracer) turns a model and a prompt into an **attribution graph**: nodes are features (read from a transcoder / CLT or SAE), and edges are *observational* attribution scores — first-order, gradient-based estimates (gradient × activation; the local-linear / EAP form) of how much one node drives another. This graph is an excellent map of *what could matter*, but its edge weights are a first-order approximation around one input point, and node "influence" is derived from them: they rank features by *correlation with influence, not causal effect under intervention*. The graph by itself does not answer the questions a causal circuit claim ultimately rests on — and it fails all three below for the **same root reason: its scores are observational, not interventional** (the local-linearity is the form today's scores happen to take, not the defect itself):

1. **Minimality / causal faithfulness.** Which *small* set of nodes is, under a real intervention, both *sufficient* (the behavior survives when only they are kept) and *necessary* (the behavior breaks when they are removed)? A high influence score is not a causal guarantee — features can be redundant, or jointly necessary in ways linear attribution misses.
2. **Contrastive separation.** When the model chooses a target answer over a competing foil, does the circuit use *distinct* features for the two, or does it route both through the same nodes? Attribution scores computed for a single target say nothing about this.
3. **Sensitivity to the evaluation convention.** Even granting an intervention protocol, a faithfulness verdict silently depends on two conventions the graph does not expose: transcoder **error nodes** are never ablated, so the "all-features-off" baseline retains an *error floor*; and **attention freezing** decides whether ablating a feature also removes its attention-mediated downstream effects. Both choices can flip a verdict — on attention-mediated tasks the recoverable range can be *negative* under frozen attention and positive once unfrozen ([§2.3](#23-attention-freezing-and-the-error-floor), [§10.4](#104-acdc-benchmark-the-attention-mediation-result)). Prior work fixes these as a single convention (circuit-tracer acknowledges frozen attention as a limitation but evaluates under it); a circuit evaluator should instead *measure under both conventions and report the difference* — turning a nuisance parameter into a diagnostic in its own right (where does the behavior live — features or attention?).

Gaps 1 and 2 motivate the two games; gap 3 motivates the error-floor-aware metric layer and the frozen/unfrozen protocol, and is what the case study's attention-mediation finding cashes in.

### 1.2 What MACAG does

MACAG evaluates the attribution graph with **real interventions** instead of attribution scores: it ablates feature nodes through the circuit-tracer `ReplacementModel` and measures the change in an output score (by default the target–foil logit gap). It frames the search for an answer to questions 1 and 2 above as two optimization **games** over the graph's feature nodes — minimal faithful evidence (Game 1) and contrastive evidence (Game 2) — and answers question 3 with an error-floor-aware metric layer plus a frozen/unfrozen scoring protocol ([§2.3](#23-attention-freezing-and-the-error-floor)). Because the only inputs are *a graph* and *a scoring oracle*, MACAG is **encoder-agnostic**: it treats the CLT/SAE that produced the nodes as a black box and can compare different transcoder variants on exactly the same footing.

**The contract, concretely** (what "a graph and an oracle" means; all implemented):
- *Graph side:* any circuit-tracer-format JSON. Candidates are the graph's feature nodes (default `feature_type = "cross layer transcoder"`); each maps to a `(layer, position, feature_idx)` ablation site, and the oracle's intervention universe is restricted to exactly those nodes so the games and the traced circuit stay aligned ([§6](#6-graph-and-candidate-selection)).
- *Oracle side:* four intervention modes (**all / empty / keep-only / remove**, [§2.1](#21-oracle-scoring)) over a configurable scalar score (`logit_gap` default; raw logit, probability, negative loss, and a full-distribution `kl_divergence` mode used for selection-independent rescoring — [§2.5](#25-kl-rescoring-a-selection-independent-faithfulness-metric)), with configurable ablation values (zero default; per-node values supported, enabling mean/resample-style baselines) and the `freeze_attention` flag. Any backend implementing the four-mode protocol plugs in — the reference backend is `ReplacementModel.feature_intervention`; a dependency-free additive toy backend exists for tests ([§7](#7-oracle-backend)).
- *Encoder-agnosticism is operational, not aspirational:* the same factory loads hub-hosted transcoder sets or local checkpoints and auto-detects standard linear CLTs vs the project's Spline-CLT from the checkpoint contents ([§7.2](#72-auto-detection-of-clt-variant)) — the evaluation code is identical across encoders, so metric differences are attributable to the circuits.
- *Determinism and cost discipline:* greedy sweeps are deterministic (alphabetical tie-breaks), oracle calls are memoized with universe-aware cache invalidation, and per-solve call/hit counts are reported — the denominators for any cost comparison ([§2.1](#21-oracle-scoring), [§3.5](#35-complexity)).

### 1.3 Relation to prior work

**Automated circuit discovery (ACDC; Conmy et al., 2023).** ACDC also searches for a minimal faithful circuit, and Game 1 shares that goal, but the two differ in what they operate on and how they search:

| | ACDC | MACAG Game 1 |
|---|------|--------------|
| Operates on | the model's **native** computational graph (attention heads, MLPs) and its **edges** | the **feature nodes** of an already-built CLT/SAE attribution graph |
| Search direction | **top-down**: start from the full graph, prune edges in reverse-topological order | **bottom-up**: start from ∅, greedily add the most useful node |
| Intervention | resample (corrupted-prompt) activation patching along edges | zero-ablation of feature nodes, in both *keep-only* and *remove* modes |
| Objective | prune an edge if it changes the metric (KL) by less than a threshold $\tau$ | maximize an explicit utility $\alpha\cdot\text{suff}+(1-\alpha)\cdot\text{nec}-\lambda|E|$ |
| Measures | necessity (effect of removal) | both **sufficiency** (keep-only) and **necessity** (remove) |

So Game 1 is "ACDC-style minimality, posed over transcoder features as a sparsity-penalized selection problem, scoring sufficiency as well as necessity." **Game 2 has no ACDC analog** — contrastive target/foil separation is a distinct contribution. MACAG is also downstream of, and complementary to, the graph-pruning that circuit-tracer already performs: pruning trims the graph by attribution magnitude; MACAG re-evaluates the survivors causally.

**The methods MACAG is measured against.** Four families of node-importance methods double as the baselines for the evaluation (§9.3). They span the cheap-but-approximate to the expensive-but-exact:

- **Edge attribution patching (EAP / attribution patching; Syed et al., 2023; Nanda, 2023).** A first-order Taylor approximation of activation patching: one backward pass scores every edge at once. This is essentially *how the attribution graph's edge weights are computed in the first place* — so EAP is the local-linear incumbent that MACAG's real (forward-pass) interventions are meant to correct. Comparing Game 1 to a top-k-by-EAP set tests whether real interventions actually buy anything over the scores already in the graph.
- **Top-k influence.** Trivially take the $k$ highest-`influence` nodes the graph already exposes (the candidate policy in [§6.3](#63-candidate-policies)). Tests whether MACAG's *search* beats simply trusting attribution magnitude at matched set size.
- **ACDC** (above). The closest minimal-circuit method; tests search direction (top-down edge pruning vs bottom-up node selection) and granularity (native components vs CLT features).
- **Shapley / Banzhaf (gold).** Exact game-theoretic per-feature credit under the *same* characteristic function $v$ MACAG optimizes ([§3.6](#36-relation-to-shapley-and-banzhaf-values)), estimated by Monte Carlo over the MACAG intervention oracle. This is the *upper-bound* attribution: it tells us how close the cheap greedy selection gets to gold-standard credit assignment, and at what fraction of the oracle cost. **Implemented (`macag/baselines/shapley_select.py`) and run on the nonlinear benchmark** (§10.7/C.7: Game 1 reaches gold-level *faithfulness* at ~45× lower oracle cost; ranking agreement with gold is moderate — prec@k 0.46 / Jaccard 0.33); the IOI/multi-hop gold comparison is still open. *Not* the same thing as the repository's `attribution/shapley.py`, which is a separate spline-CLT attribution-graph tool over a different value function (see [§3.6](#36-relation-to-shapley-and-banzhaf-values)).

**Contemporary methods (2024–2026).** Beyond ACDC, recent circuit-discovery and attribution methods — CD-T (analytical decomposition), K-MSHC (minimal sufficient head circuits), SPEX (scalable Banzhaf interactions), Hedonic Neurons (coalitional synergy), Formal MI (verifier-certified circuits), MechRL (RL discovery), and REdit (circuit editing) — are analyzed and compared to MACAG in [Appendix H](#appendix-h-contemporary-methods-and-positioning). The short version: each operates on *native components or input tokens*, none on transcoder features with a contrastive game, and the closest (Hedonic Neurons) validates rather than supersedes the coalitional framing. Two likely "isn't this just X?" attacks worth pre-empting inline: **K-MSHC** is the nearest minimality method but is *sufficiency-only* over attention heads — no necessity term, no sparsity-penalized utility, no contrastive game; **MechRL** uses a "contrastive" reward, but its contrast is *task-vs-general-ability* damage, not MACAG's *target-vs-foil* evidence separation — a different axis entirely.

**Not to be confused with transcoder-quality metrics.** Reconstruction (MSE, cosine), sparsity ($L_0$), and monosemanticity (Gini) evaluate *the CLT* — how well it approximates and disentangles MLP outputs. MACAG evaluates *the circuit* the CLT produces. A transcoder can reconstruct well yet yield circuits whose features do not line up with the model's causal structure; MACAG is what surfaces that.

### 1.4 Problem statement, research questions, and contributions

**Problem statement (formalizing [§1.1](#11-the-gap-an-attribution-graph-is-structure-not-a-causal-verdict)).** Given an attribution circuit graph $G=(C,A)$ for a prompt — feature nodes $C$, observational (local-linear) attribution edges $A$, produced by a circuit-tracer over *any* transcoder/SAE — and intervention access to the underlying model, **select feature nodes that causally explain the prediction**: (i) a *small* subset $E\subseteq C$ that is causally *sufficient* and *necessary* for the target (minimality is optimized via a sparsity penalty, not certified — see guardrails below), and (ii) for a target/foil pair, subsets that reveal whether the two answers are carried by *different* nodes. Every selection is scored by intervention rather than by the edge weights $A$, under **explicit, reported conventions** — the error floor retained by unablatable error nodes and the attention-freezing mode — whose *effect on the verdict* is itself a measured output (gap 3).

**Research questions.** The paper is organized around four. Each has a designated instrument already implemented and a designated metric (consolidated in [§2.4](#24-reported-metrics-consolidated-definitions)):
<!-- TODO(pass-2): the "current evidence" numbers in the RQ list below are from the pre-71a2ef6 runs (§10 provenance box); refresh after the re-run. -->

- **RQ1 (faithfulness).** Can a small, intervention-verified node set reproduce (sufficiency) and be required for (necessity) the behavior, and how does selecting it by *real interventions* compare to selecting by attribution magnitude (top-k influence), first-order patching (EAP), edge-pruning (ACDC), and gold credit (Shapley)? *Instrument:* Game 1 + the baseline harness (implemented; run on the 60-prompt nonlinear benchmark — §10.7/C.7; the IOI/multi-hop graphs are not yet covered — §9.3). *Metrics:* raw sufficiency/necessity/faithfulness at matched $|E|$; faithfulness-vs-size curves; oracle-call counts. *Current evidence (nonlinear benchmark, n=60):* Game 1 faith\@8 5.50 [4.88, 6.15], faith/feature 0.859 [0.747, 0.977] — non-overlapping CI vs Shapley-gold's 0.590 [0.500, 0.683] — at 44.7× fewer oracle calls than Shapley. (On the 50 target-preferred prompts: fpf 0.938 [0.820, 1.068] vs 0.636.)
- **RQ2 (contrastive structure).** Do target and foil predictions route through *distinct* features, and is that separation a property of the circuit or an artifact of the scoring convention? *Instrument:* Game 2 under both attention modes (the convention-invariance check). *Metrics:* overlap_rate, shared/unique decomposition; current evidence: 0.0 overlap in 48/48 runs, frozen *and* unfrozen.
- **RQ3 (attention mediation).** Can MACAG *diagnose where a behavior lives* — features vs. attention — and does that diagnosis vary by task and by model? *Instrument:* paired frozen/unfrozen Game 1 runs on the same graph. *Metrics:* sign of `recoverable_range`, the range-flip rate, upstream-feature counts; current evidence: gemma 23/26 flips on IOI+docstring vs llama 0/13.
- **RQ4 (capacity & generality).** Does transcoder capacity, or the choice of model family, change circuit faithfulness as MACAG measures it? *Instrument:* identical games over CLTs differing only in width (gemma 426k vs 2.5M) or model family (gemma vs llama). *Metrics:* reconstruction-failure counts, faithfulness, evidence size; current evidence: the ~6× capacity increase does not remove the failure (§10.2).

**Contributions.** This document supports the following claimed contributions (evidence mapping in [§10.6](#106-claims-to-evidence-to-research-question-mapping)):
- **C1 — Framework.** MACAG: an encoder-agnostic, intervention-based, game-theoretic evaluator for attribution circuit graphs, taking only a graph and a scoring oracle. The contract is operational ([§1.2](#12-what-macag-does)): four intervention modes, configurable score/ablation/attention conventions, universe-restricted candidates, deterministic solvers, and a memoized oracle with universe-aware cache invalidation ([§2](#2-framework), [§6](#6-graph-and-candidate-selection), [§7](#7-oracle-backend)).
- **C2 — Formalization.** A coalitional-game formulation (players = features, $v$ = faithfulness) with grounded/bounded properties of $v$ ($v(\emptyset)=0$, $v(C)=$ recoverable range), a submodularity analysis with concrete synergy/redundancy counterexamples, an honest characterization of where the greedy $(1-1/e)$ guarantee holds and breaks, plus the explicit Shapley/Banzhaf connection — the gold credit is defined over the *same* $v$ the games optimize ([§3](#3-game-theoretic-foundations)).
- **C3 — Game 1.** A minimal-faithful-evidence game: ACDC-style minimality recast as a bottom-up, sparsity-penalized selection over transcoder features that scores *both* sufficiency and necessity, with an error-floor-aware normalization, a denominator-free (λ-free, eps-relative) stop for the degenerate regime, and an optional connectivity constraint that routes through feature/error intermediates while excluding logit/embedding hubs (without which the constraint is vacuous — [§4.4](#44-connectivity-constraint)) ([§2.3](#23-attention-freezing-and-the-error-floor), [§4](#4-game-1-minimal-faithful-evidence)).
- **C4 — Game 2.** A contrastive evidence game (target vs. foil with an overlap penalty), formalized as an exact potential game with provable pure-strategy equilibrium *existence* — no analog in the prior/contemporary circuit-discovery work surveyed in [Appendix H](#appendix-h-contemporary-methods-and-positioning). The solver story is stated honestly: simultaneous (Jacobi) greedy best response with best-iterate tracking, plus a fictitious-play variant whose expected-overlap utility is computed exactly at no extra oracle cost ([§3.4](#34-equilibrium-analysis), [§5](#5-game-2-contrastive-evidence)).
- **C5 — Diagnostic + empirical finding.** The frozen/unfrozen `recoverable_range` signature as an *attention-mediation diagnosis* (gap 3 made measurable), demonstrated on three public CLTs across two model families ([§10](#10-results), [§11](#11-discussion-of-the-case-study)).

*Status caveat:* C1–C4 are fully supported by the current document; C5's empirical half is single-seed. The C1/C3 baseline comparisons (vs influence/EAP/ACDC/Shapley) are implemented and now run on the 60-prompt nonlinear benchmark with bootstrap CIs + paired Wilcoxon tests (§10.7/C.7); what remains is running them on the IOI/multi-hop case-study graphs and adding multi-seed variance — see [§12.3](#123-conference-readiness-what-is-present-vs-missing) and the roadmap in [Appendix B](#appendix-b-roadmap-to-a-submission-ready-evaluation).

**Wording guardrails (overclaim risks for the writeup).** Each row pairs atempting phrase with the claim the code/results actually support:

| Tempting claim | Supported claim |
|----------------|-----------------|
| "finds the *smallest* faithful set" / "minimal circuit" | finds a *small* set by greedy maximization of a sparsity-penalized utility; minimality is the objective, not a certificate (greedy + non-submodular $v$, [§3.2](#32-the-value-function-and-submodularity)) |
| "jointly sufficient and necessary" | $\alpha$-weighted mix of sufficiency and necessity (both at the default $\alpha=0.5$; either alone at $\alpha\in\{0,1\}$) |
| "Game 2 converges to a Nash equilibrium" | a PSNE *exists* (exact potential game); the implemented greedy-Jacobi dynamics stabilized empirically in 2–4 rounds but carry no convergence guarantee — best-iterate tracking + FP are the mitigations ([§3.4](#34-equilibrium-analysis)) |
| "the $(1-1/e)$ guarantee" | holds only if $v$ is monotone submodular, which ablation logit-gaps are not in general; cite as best-case, report the empirical optimality gap (roadmap B3.2) |
| "MACAG beats EAP/ACDC/top-k/Shapley" | run on the **nonlinear benchmark only** (n=60, §10.7/C.7): Game 1 wins on *cost* (44.7× fewer calls than Shapley, 60/60) and *faith-per-feature* (non-overlapping CIs vs all baselines); raw faith\@8 beats Shapley on 44/60, not uniformly; **not yet run** on the IOI/multi-hop graphs, and faith\@k is on Game 1's own objective ([§11.3](#113-threats-to-validity--reviewer-rebuttals-to-pre-empt)) |
| "causally responsible features" | sufficient/necessary *under zero-ablation with the stated attention convention*; zero-ablation is off-manifold (rebuttal list, [§11.3](#113-threats-to-validity--reviewer-rebuttals-to-pre-empt)) |
| "robust finding" | single seed, 8–13 prompts per task; only the overlap-0.0 result is convention-invariant; CIs are roadmap Phase 1 |
| "encoder-agnostic (any SAE)" | demonstrated for hub CLTs, local CLTs, and Spline-CLT checkpoints through one factory; per-layer SAE variants would need an oracle adapter, not a framework change |

---

## 2. Framework

### 2.0 Notation

| Symbol | Meaning |
|--------|---------|
| $G=(C,A)$ | attribution graph: candidate feature nodes $C$, attribution edges $A$ |
| $C$ | candidate node set (the universe MACAG selects from); $|C|$ its size |
| $E, E_y, E_{\text{foil}}$ | evidence set (Game 1), target / foil evidence sets (Game 2) |
| $E^*$ | the returned evidence set; $|E^\*|$ its size |
| $y, y_{\text{foil}}$ | target and foil labels/tokens |
| $S_{\bullet}(\cdot)$ | oracle score in mode $\bullet \in \{$all, empty, keep, remove$\}$ |
| $v(S)\equiv f(S)$ | characteristic / faithfulness function of coalition $S$ |
| $\Delta_i(S)$ | marginal value of node $i$ to $S$: $v(S\cup\{i\})-v(S)$ |
| $\alpha$ | sufficiency/necessity mix (default 0.5) |
| $\lambda$ | sparsity penalty (default 0.01; case study 0.02) |
| $\beta$ | Game 2 overlap penalty (default 0.1; case study 0.2) |
| $\varepsilon$ | Game 1 early-stop threshold (`faithfulness_eps`) |
| $K$ | max solver rounds, ABR or fictitious play (Game 2; `abr_iters`) |
| $B$ | evidence-size budget |
| $\Phi$ | Game 2 exact potential function ([§3.4](#34-equilibrium-analysis)) |
| $\phi_i^{\text{Shapley}}, \phi_i^{\text{Banzhaf}}$ | per-feature gold credit ([§3.6](#36-relation-to-shapley-and-banzhaf-values)) |

### 2.1 Oracle Scoring

MACAG operates through a **scoring oracle** that measures model behavior under feature interventions. For a set of feature nodes $E$ and a target class $y$, four canonical intervention modes define the oracle:

| Mode | Description | Notation |
|------|-------------|----------|
| **All** | All feature nodes active (unmodified model) | $S_{\text{all}}(y)$ |
| **Empty** | All feature nodes ablated (null model) | $S_{\text{empty}}(y)$ |
| **Keep-only** | Only nodes in $E$ active, rest ablated | $S_{\text{keep}}(E, y)$ |
| **Remove** | Nodes in $E$ ablated, rest active | $S_{\text{remove}}(E, y)$ |

The score function is typically the **logit gap** between target and foil tokens:

$$S(y) = \text{logit}(y_{\text{target}}) - \text{logit}(y_{\text{foil}})$$

Other scoring modes are supported: raw logit, softmax probability, negative cross-entropy loss, and a full-distribution `kl_divergence` mode (the score is $-\mathrm{KL}(P_{\text{ref}}\,\|\,P_{\text{int}})$ at the scored position, where $P_{\text{ref}}$ is the clean-model distribution — see [§2.5](#25-kl-rescoring-a-selection-independent-faithfulness-metric)). KL is target-free: $S_{\text{all}} \equiv 0$ by construction, and the reference logits are computed once on the clean pass and cached.

**Implementation**: Interventions are implemented via the circuit-tracer `ReplacementModel`, which performs feature-level ablation (setting $a_f = 0$). Each intervention maps to a `(layer, position, feature_idx, ablation_value)` specification.

Attention patterns (and LayerNorm denominators) may be **frozen at their clean values or left free to recompute**, controlled by the scoring-time flag `freeze_attention` (default `True`). This choice is not cosmetic: it changes the **Empty** baseline and therefore every derived metric. See [§2.3](#23-attention-freezing-and-the-error-floor).

#### Oracle Caching

Oracle calls are expensive (each requires a forward pass through the transformer with modified feature activations). MACAG implements a **memoization cache** keyed by `(mode, type(target), str(target), frozenset(nodes), universe_fingerprint)`. The type is included in the key to differentiate integer vs. string targets with the same string representation. The universe fingerprint covers the universe-dependent modes (`empty`, `keep_only`, `remove` — they ablate the candidate universe, its complement, or its intersection with the request), so cached scores are invalidated automatically when `restrict_universe` changes the ablation universe.

Cache statistics (oracle_calls, cache_hits, cache_size) are tracked per-game and reported in the output, enabling analysis of computational efficiency across CLT variants.

### 2.2 Derived Metrics

From the four oracle scores, MACAG derives:

**Sufficiency** — does $E$ alone reproduce the model's behavior?

$$\text{sufficiency}(E) = S_{\text{keep}}(E) - S_{\text{empty}}$$

**Necessity** — is $E$ required for the model's behavior?

$$\text{necessity}(E) = S_{\text{all}} - S_{\text{remove}}(E)$$

**Faithfulness** — weighted combination of sufficiency and necessity:

$$\text{faithfulness}(E) = \alpha \cdot \text{sufficiency}(E) + (1 - \alpha) \cdot \text{necessity}(E)$$

where $\alpha \in [0, 1]$ balances the two (default $\alpha = 0.5$). When $\alpha = 1$, faithfulness equals sufficiency (can the evidence reproduce the behavior alone?). When $\alpha = 0$, faithfulness equals necessity (does removing the evidence destroy the behavior?).

**Utility** — faithfulness penalized by evidence size:

$$U(E) = \text{faithfulness}(E) - \lambda |E|$$

where $\lambda \geq 0$ is the sparsity penalty (default $\lambda = 0.01$). This encodes a preference for smaller evidence sets.

**Sparsity** — fraction of candidates not selected:

$$\text{sparsity}(E) = 1 - \frac{|E|}{|C|}$$

where $C$ is the full candidate set.

**Error-floor-aware (normalized) metrics.** Reconstruction-error nodes are, by default, never ablated, so $S_{\text{empty}}$ is not zero — it is the residual score the model retains from error nodes (and, when attention is unfrozen, from attention) when *all* features are off. To separate "how much the features can explain" from this floor, MACAG also reports a normalized view:

$$\text{error\_floor} = S_{\text{empty}}, \qquad
\text{recoverable\_range} = S_{\text{all}} - S_{\text{empty}}$$

$$\text{sufficiency}_{\text{norm}} = \frac{\text{sufficiency}(E)}{\text{recoverable\_range}}, \quad
\text{necessity}_{\text{norm}} = \frac{\text{necessity}(E)}{\text{recoverable\_range}}, \quad
\text{faithfulness}_{\text{norm}} = \alpha\,\text{sufficiency}_{\text{norm}} + (1-\alpha)\,\text{necessity}_{\text{norm}}$$

When $|\text{recoverable\_range}| < \varepsilon_{\text{range}}$ the normalized metrics are set to $0$ to avoid division by a vanishing denominator. This matters because `recoverable_range` can be **zero or negative**: ablating all features (especially with unfrozen attention) may fail to lower — or may even raise — the $S_{\text{empty}}$ baseline, which means the behavior lives in attention / error nodes rather than in features. In that regime the normalized metrics are degenerate and the **raw** sufficiency/necessity/faithfulness should be read instead. Attention-mediated tasks (e.g. IOI) routinely show negative `recoverable_range` under frozen attention.

### 2.3 Attention Freezing and the Error Floor

The single most consequential scoring choice in MACAG is whether attention is frozen. It interacts with the error floor to produce two failure modes we explicitly correct for; understanding it is required to read any Game 1 result.

**What the flag does.** `freeze_attention=True` holds every attention pattern at the value it took on the clean (unablated) forward pass, so feature ablations only remove the *direct* feature contribution to the residual stream. With `freeze_attention=False`, attention is recomputed from the ablated activations, so removing a feature also removes everything that feature would have caused *through* attention on downstream positions.

**Failure mode 1 — frozen attention hides features from the minimal set.** Under frozen attention, any upstream feature whose downstream effect is mediated by attention is already "paid for" by the frozen pattern. The Empty baseline $S_{\text{empty}}$ keeps that contribution even with *all* features ablated, so the greedy never needs to add the upstream feature to recover the behavior — it looks redundant. The minimal faithful set therefore collapses to a few late-layer / final-token features and silently drops the city/structure features that actually drive the circuit. Re-running with `freeze_attention=False` forces attention to be reconstructed from features, which **recruits those upstream and early-layer features back into the minimal set**. The `analyze_frozen_vs_unfrozen.py` analyzer quantifies exactly this: it counts upstream features (reverse-position $> 0$, i.e. not at the prediction token) and early-layer features recovered when attention is unfrozen. *Empirical caveat:* the **direction** of the evidence-set change is task-dependent — the recruitment described here is what the two-hop relational sweep shows (§10.3), but on IOI (§10.4) and the matched-budget nonlinear benchmark (§10.7(3)) unfreezing instead *shrinks* the set. The convention-invariant statement is that frozen attention *distorts* what the minimal set must contain (it can hide attention-mediated upstream features, or add spurious necessity that unfreezing removes); which way it errs is itself a per-task diagnostic.

**Failure mode 2 — the normalized denominator goes degenerate.** The normalized metrics divide by `recoverable_range` $= S_{\text{all}} - S_{\text{empty}}$. Two things push this denominator toward zero or negative:
- *Frozen attention on an attention-mediated task* (e.g. IOI): the answer lives in the frozen attention pattern, so ablating all features barely moves the score — $S_{\text{empty}} \approx S_{\text{all}}$ and the range collapses, sometimes going negative (the 426k gemma IOI runs are 10/10 such reconstruction "failures").
- *Unfrozen attention*: ablating all features and letting attention recompute can collapse or even invert the baseline, so $S_{\text{empty}}$ can exceed $S_{\text{all}}$.

Either way the normalized metric divides by a near-zero/negative number and produces wild values (the "weird denominator" results). The fixes are layered:
1. `metrics.py` guards the division (`_RANGE_EPS = 1e-9`) and zeroes the normalized metrics when $|\text{recoverable\_range}| < \varepsilon_{\text{range}}$, so the output is never a spurious huge ratio.
2. Game 1's normalized early-stop (`stop_metric=normalized`) inherits the same weakness, so a `raw_relative` stop was added that never touches the denominator (see [§4](#4-game-1-minimal-faithful-evidence)).
3. For analysis, the **raw** sufficiency / necessity / faithfulness_delta are denominator-free and are the metrics to report whenever the range is unreliable. `analyze_robust_frozen_vs_unfrozen.py` re-reads the stored raw scores (no new model runs) precisely to sidestep the broken denominator.

**Recommended practice.**

| Situation | Attention | Stop metric | Read |
|-----------|-----------|-------------|------|
| Clean, feature-mediated task | frozen | `normalized` | normalized + raw |
| Attention-mediated task (IOI), or `recoverable_range` ≤ 0 | unfrozen | `raw_relative` | raw |
| Diagnosing where the behavior lives (the matched protocol) | `--freeze-mode both` | `raw_relative` (forced on both legs) | `attention_mediation` block (verdict, range flip, upstream/early recruitment) |

Because `freeze_attention` is a scoring-time argument, the attribution graph is unchanged between frozen and unfrozen runs. The **matched protocol is now built into the CLI**: `run_macag game1 --freeze-mode both` builds the ReplacementModel oracle once, derives a freeze-flipped twin sharing the same model (`macag.scoring.derive_oracle_with_freeze` — fresh cache per leg, since every intervention score depends on the freeze convention), and runs both legs under identical budget / prefilter / eps / α / λ with `stop_metric=raw_relative` forced on both (the `normalized` stop is degenerate on the unfrozen leg, so mixing stop rules would make the legs incomparable; passing `--stop-metric normalized` with `--freeze-mode both` is an error). The output carries a per-prompt `attention_mediation` block (§4.5). **Historical caveat:** the pre-existing two-invocation sweeps (`scripts/run_macag_unfrozen*.sh`, analyzed by `experiments/analyze_frozen_vs_unfrozen.py`) were *not* parameter-matched — the stored unfrozen runs raised the Game 1 budget 8 → 20 and prefilter 20 → 30. `all`/`empty` — and therefore `recoverable_range` — are budget-independent, so the §10.4 sign-flip diagnostic is unaffected; evidence sizes and upstream-feature counts in those stored runs are partially confounded by the lifted cap ([§11.3](#113-threats-to-validity--reviewer-rebuttals-to-pre-empt)). The pass-2 re-run should use `--freeze-mode both`, which removes that confound by construction. <!-- pass-2 note: matched-protocol decision resolved 2026-06-11; implemented as --freeze-mode both -->

### 2.4 Reported Metrics: Consolidated Definitions

Every quantity that appears in the results (§10, Appendix C) and what it means. Raw quantities are in logit-gap units; normalized quantities are unitless ratios.

| Metric | Definition | Reads as | Caveat |
|--------|------------|----------|--------|
| `all` $S_{\text{all}}$ | score, no ablation | baseline target–foil gap | — |
| `empty` $S_{\text{empty}}$ | score, all features ablated | the **error floor** (residual from error nodes / frozen attention) | sign matters (see below) |
| `keep_only` $S_{\text{keep}}(E)$ | score, only $E$ active | how much $E$ alone reconstructs | — |
| `remove` $S_{\text{remove}}(E)$ | score, $E$ ablated from full | residual without $E$ | — |
| **sufficiency** | $S_{\text{keep}}(E)-S_{\text{empty}}$ | can $E$ *alone* drive the behavior? | raw; denominator-free |
| **necessity** | $S_{\text{all}}-S_{\text{remove}}(E)$ | does removing $E$ break it? | raw; denominator-free |
| **faithfulness** $f/v$ | $\alpha \cdot $ suff $+(1-\alpha)\cdot$ nec | overall causal explanation of $E$ | raw; the games' objective |
| **utility** $U$ | $f(E) - \lambda E $ (G1); $ -\beta E \cap E_{\text{other}} $ added (G2) | sparsity-penalized objective | — |
| **recoverable_range** | $S_{\text{all}}-S_{\text{empty}}$ | how much score features *can* recover above the floor | **≤0 ⇒ behavior not in features; normalized metrics degenerate** |
| **$E^*_{normalized}$** | raw $\div$ recoverable_range | fraction of recoverable range explained | unreliable when range ≤ 0 |
| **evidence size** $E^*$ | # nodes selected | parsimony | capped by budget $B$ |
| **sparsity** | $1-E^*/C$ | fraction of candidates *not* used | depends on $C$ |
| **upstream-feature count** | # nodes in $E^*$ with reverse-position $>0$ (not at the prediction token) | how much of the circuit is *upstream* structure vs. final-token readout | key for the frozen/unfrozen contrast |
| **early-layer count** | # nodes in $E^*$ in early layers | depth profile of evidence | — |
| **target_preferred** | $S_{\text{all}}>0$ (model predicts target over foil at baseline) | is the oracle measuring the right thing? | **exclude `False` rows from faithfulness aggregates** |
| **overlap_rate** (G2) | $\|E_y\cap E_{\text{foil}}\|/\|E_y\cup E_{\text{foil}}\|$ | target/foil feature sharing (0 = disjoint) | denominator-free; attention-invariant |
| **shared / unique_y / unique_foil** (G2) | $E_y\cap E_{\text{foil}}$, $E_y\setminus E_{\text{foil}}$, $E_{\text{foil}}\setminus E_y$ | contrastive decomposition | — |
| **range-flip rate** | fraction of prompts with recoverable_range $<0$ frozen but $\ge0$ unfrozen | **the attention-mediation diagnostic** (§10.4); emitted per-prompt as `attention_mediation.range_flip` / `verdict` by `--freeze-mode both` | report with CI once multi-seed |
| **oracle_calls / cache_hits** | # model forward passes / memoized hits | compute cost; cache efficiency | denominator for cost-ratio vs Shapley |
| **converged** (G2) | solver dynamics stabilized within $K$ (iterates repeated; FP also: frequencies within `fp_tol`) | solution-quality flag | the returned allocation is the best round (`best_iteration`), not necessarily the final/converged iterate |
| **cross-seed / cross-prompt Jaccard** | set agreement of $E^*$ across repeats | stability of the selected circuit | needs multi-seed (Phase 1) |
| **kl_faith** (`kl_faithfulness`) | faithfulness recomputed on the *already-selected* evidence with `score_kind="kl_divergence"` ([§2.5](#25-kl-rescoring-a-selection-independent-faithfulness-metric)) | selection-independent faithfulness check (evaluation metric ≠ greedy objective) | evaluation-only; in nats, not logit-gap units — compare across methods, not against raw faith |

**Sign conventions to keep straight.** A *negative* `empty` means the model prefers the foil once features are ablated (features carry the whole behavior — healthy). A *positive* `empty` larger than `all` means ablation *helped* the target → negative `recoverable_range` → a reconstruction failure where the normalized view is meaningless and only raw scores are valid.

### 2.5 KL Rescoring: a Selection-Independent Faithfulness Metric

Game 1 greedily maximizes logit-gap faithfulness and is then reported on that same
quantity — the circularity threat in [§11.3](#113-threats-to-validity--reviewer-rebuttals-to-pre-empt).
The implemented mitigation is a second, **selection-independent** faithfulness
metric: every stored evidence set is *re-scored* (never re-selected) under
`score_kind="kl_divergence"`, where the oracle score is
$-\mathrm{KL}(P_{\text{ref}}\,\|\,P_{\text{int}})$ over the full next-token
distribution at the scored position ($P_{\text{ref}}$ = clean model,
$P_{\text{int}}$ = ablated model). This matches the MIB/ACDC convention of
comparing the intervened circuit to the full model, and it is **foil-free** — it
also answers the "logit-gap / single-foil is the wrong metric" objection.

Properties worth stating:
- $S_{\text{all}} \equiv 0$ (the reference compared to itself), and every other
  mode is $\le 0$, so under KL: `error_floor` $= -\mathrm{KL}(P_{\text{ref}}\|P_{\text{empty}}) \le 0$
  and `recoverable_range` $= \mathrm{KL}(P_{\text{ref}}\|P_{\text{empty}}) \ge 0$ **always** —
  the degenerate negative-denominator regime of [§2.3](#23-attention-freezing-and-the-error-floor)
  cannot occur under KL (though the range can still be near zero, with the same
  $\varepsilon_{\text{range}}$ guard).
- The same four-mode oracle and `FaithfulnessMetrics` machinery is reused, so
  KL sufficiency/necessity/faithfulness are defined exactly as in
  [§2.2](#22-derived-metrics), just in nats.
- Cost is 4 oracle calls per evidence set (all/empty/keep/remove), per freeze leg.

**Implementation.** `macag/scoring.py` adds the `kl_divergence` score kind
(`compute_kl_score`; reference logits cached from the clean pass);
`macag/kl_rescore.py` + `python -m macag.cli.rescore_kl --run-dir <dir> | --root <sweep>`
walk saved run directories, rebuild the oracle from the stored
`oracle_kwargs.json` (per freeze leg for dual-freeze Game 1 outputs), and re-score
Game 1 evidence, Game 2 target/foil allocations, and every baseline selector's
selected set. Output: a per-run sidecar `macag_kl_faithfulness.json`, plus
`kl_faithfulness` blocks embedded into `macag_game1.json` / `macag_game2.json` /
`macag_baselines.json`. The per-prompt pipeline runs it as its final step and the
sweep drivers run it as the first analysis step, before the aggregators
(`KL_RESCORE=0` to skip; pipeline flag `--skip-kl-rescore`), so the aggregation
analyzers can emit `kl_faith` columns. The machinery is parametrized by a
`RescoreSpec` (2026-07-02), and a second flavor ships with it:
`python -m macag.cli.rescore_altfoil` re-scores the same stored evidence under
the run's own score kind but an **alternate foil token** (`--foil-token`, or
per-slug `metadata.alt_incorrect_token` from a manifest via `--bench`), writing
`macag_altfoil_faithfulness.json` / `altfoil_faithfulness` blocks — the
foil-choice robustness check of §11.3. The ACDC manifest now carries
`alt_incorrect_token` for every prompt (IOI: the corrupted-prompt third name;
greater_than: another below-threshold year; docstring: a generic wrong
parameter). Smoke-verified on a stored MIB IOI run (2026-07-02): both dual-freeze
legs rescored under the absent-name foil, sidecar written and blocks embedded
alongside `kl_faithfulness` (`summary.csv`, `baselines.csv`,
`frozen_vs_unfrozen.csv`). Unit coverage: `tests/test_macag_kl_scoring.py`.

---

## 3. Game-Theoretic Foundations

### 3.0 The Underlying Coalitional Game

MACAG's two games are both built on a single **cooperative (coalitional) game** whose **players are the feature nodes** $N = C$ and whose **characteristic function is the faithfulness contribution** of a coalition $S \subseteq N$:

$$v(S) = \alpha\bigl(S_{\text{keep}}(S) - S_{\text{empty}}\bigr) + (1-\alpha)\bigl(S_{\text{all}} - S_{\text{remove}}(S)\bigr), \qquad v(\emptyset)=0.$$

This is the same $v$ throughout the document (the faithfulness function $f$). Once the game is posed this way, the standard solution concepts apply, and the different MACAG objects are different questions *about the same $v$*:

- **Per-feature credit** — how much is each player worth? The classical answers are the **Shapley value** and the **Banzhaf value** ([§3.6](#36-relation-to-shapley-and-banzhaf-values)).
- **Best small coalition** — what is the most valuable cheap coalition? This is **Game 1** (a sparsity-penalized coalition-selection problem solved greedily).
- **Two-player extension** — split the players between a target and a foil agent with an overlap cost: **Game 2**.

The "single-agent optimizer" description of Game 1 below is the *solver's* view of this coalition-selection problem; the *game-theoretic* object is the coalitional game $(N, v)$.

### 3.1 Game Classification

As optimization problems, both MACAG games belong to the class of **cooperative combinatorial optimization games** — specifically, they are instances of **weighted maximum coverage / set function optimization** over the ground set of feature nodes. The key properties:

| Property | Game 1 | Game 2 |
|----------|--------|--------|
| Players | 1 (evidence selector) | 2 (target agent, foil agent) |
| Strategy space | $2^C$ (subsets of candidates) | $2^C \times 2^C$ (one subset per agent) |
| Payoff | $U_1(E) = f(E) - \lambda \| E \| $ | $U_2(E_i, E_j) = f(E_i) - \lambda\|E_i\| - \beta \|E_i \cap E_j \|$ |
| Game type | Single-agent optimization | Two-player symmetric game with externalities |
| Solution concept | Greedy maximum ($(1{-}1/e)$ only if submodular — [§3.2](#32-the-value-function-and-submodularity)) | Pure-strategy Nash equilibrium (exact potential game — [§3.4](#34-equilibrium-analysis)) |
| Interaction | None | Negative externality via overlap penalty |

### 3.2 The Value Function and Submodularity

The core value function underlying both games is the **faithfulness function**:

$$f(E) = \alpha \cdot \underbrace{\left(S_{\text{keep}}(E) - S_{\text{empty}}\right)}_{\text{sufficiency}} + (1-\alpha) \cdot \underbrace{\left(S_{\text{all}} - S_{\text{remove}}(E)\right)}_{\text{necessity}}$$

**Properties of the characteristic function $v \equiv f$.** Useful facts a paper can
state up front:
- *Grounded:* $v(\emptyset) = \alpha(S_{\text{empty}} - S_{\text{empty}}) + (1-\alpha)(S_{\text{all}} - S_{\text{all}}) = 0$. The error floor is subtracted on the sufficiency side and the full circuit on the necessity side, so the empty coalition is worth zero by construction.
- *Bounded above by the recoverable range:* the grand coalition gives $v(C) = \alpha(S_{\text{all}} - S_{\text{empty}}) + (1-\alpha)(S_{\text{all}} - S_{\text{empty}}) = S_{\text{all}} - S_{\text{empty}} = \text{recoverable\_range}$. Hence the normalized objective $v(E)/v(C)$ targets "fraction of the recoverable range explained," and is exactly why a non-positive range makes the normalized view degenerate ([§2.3](#23-attention-freezing-and-the-error-floor)).
- *Not guaranteed monotone:* adding a feature can lower $v$ (a feature whose ablation *helps* the target, i.e. a suppressor/foil-aligned feature, has negative marginal value). The $-\lambda|E|$ term further makes the *utility* non-monotone by design. Greedy therefore needs the `min_gain`/stop machinery rather than running to $E = C$.
- *Not guaranteed superadditive:* redundancy ($v(A\cup B) < v(A)+v(B)$) and synergy ($>$) both occur — see the violation example below.

**Submodularity analysis.** $v$ is *submodular* iff marginal gains diminish: $\Delta_i(S) \ge \Delta_i(T)$ for all $S \subseteq T$ and $i \notin T$. When $v$ is additionally monotone, the greedy algorithm achieves the classic $(1-1/e)\approx 0.632$ guarantee under a cardinality constraint (Nemhauser, Wolsey & Fisher, 1978); the modular $-\lambda|E|$ penalty shifts every value by a constant per element and so preserves the relative ordering of marginal gains (greedy with a modular penalty is equivalent to greedy on $v$ with an adjusted `min_gain`).

**Where it breaks — a concrete construction.** Submodularity fails exactly when features interact. Two canonical cases over the logit-gap oracle:
- *Synergy / joint necessity (super-modular spike).* Suppose a behavior needs both a "subject" feature $a$ and a "relation" feature $b$ (e.g. the two hops of city→state and state→capital). Individually each is nearly useless: $v(\{a\}) \approx v(\{b\}) \approx 0$, but together $v(\{a,b\}) \gg 0$. Then $\Delta_b(\emptyset) \approx 0 < \Delta_b(\{a\}) \gg 0$ — an *increasing* marginal return, the opposite of submodular. A pure greedy that adds the single best feature first can stall (no feature has positive singleton gain) and miss the pair; this is the failure the `prefilter`+budget and the two-hop prompts are most likely to expose.
- *Redundancy (the benign direction).* Two interchangeable copies $a,a'$ of the same computation: $v(\{a\}) = v(\{a'\}) = v(\{a,a'\})$. Here $\Delta_{a'}(\{a\}) = 0 < \Delta_{a'}(\emptyset)$ — submodular, and harmless: greedy takes one and the sparsity penalty rejects the other. Large/overcomplete dictionaries produce many such pairs (this is the mechanism behind the near-zero cross-seed Jaccard observed for wide CLTs in older notes).

**Consequences for MACAG (honest characterization).** The logit gap under feature ablation is **not** guaranteed submodular, so the $(1-1/e)$ bound is a *best case*, not a theorem about MACAG. Greedy is nonetheless a reasonable solver because:
1. empirically most features contribute near-independently to the gap (the synergy spikes above are the exception, concentrated on genuinely multi-hop prompts);
2. the sparsity penalty $\lambda$ keeps the search out of the deep diminishing-returns tail where violations accumulate;
3. the singleton prefilter biases toward high-marginal candidates, where diminishing-returns violations are rarer.
The right way to *report* this (not just argue it) is the empirical optimality gap in roadmap **B3.2**: brute-force the best size-$k$ subset on small pools and compare to greedy, and separately compare greedy's selection to the Shapley ranking ([§3.6](#36-relation-to-shapley-and-banzhaf-values)) — large greedy↔Shapley disagreement localizes the non-submodular prompts.

### 3.3 Game 2 as a Congestion Game

Game 2 can be viewed as a **two-player congestion game** where the shared resource is the set of feature nodes. Each agent (target and foil) wants to claim features for its own evidence set, but overlapping features incur a cost $\beta$ per shared node:

$$U_2(E_i \mid E_j) = f(E_i) - \lambda|E_i| - \beta|E_i \cap E_j|$$

This creates a **negative externality**: if Player $y$ includes a node already in $E_{\text{foil}}$, it pays the overlap penalty $\beta$. The penalty pushes the agents toward **complementary** evidence sets — different features for the target vs. foil prediction.

**Key insight**: The overlap penalty $\beta$ controls the degree of separation:
- $\beta = 0$: Both agents optimize independently (may converge to identical sets)
- $\beta \to \infty$: Agents are forced to select disjoint evidence sets
- $\beta = 0.1$ (default): Mild pressure toward separation, allowing shared features when they are highly faithful

### 3.4 Equilibrium Analysis

**Game 2 equilibrium concept**: MACAG seeks a **pure-strategy Nash equilibrium (PSNE)** — a pair $(E_y^*, E_{\text{foil}}^*)$ where neither agent can improve its utility by unilaterally changing its evidence set:

$$U_2(E_y^* \mid E_{\text{foil}}^*) \geq U_2(E' \mid E_{\text{foil}}^*) \quad \forall E' \subseteq C$$
$$U_2(E_{\text{foil}}^* \mid E_y^*) \geq U_2(E' \mid E_y^*) \quad \forall E' \subseteq C$$

**Existence via an exact potential function.** Game 2 is an *exact potential game*, which is the clean way to argue both existence and convergence. Define

$$\Phi(E_y, E_{\text{foil}}) = \big[f(E_y) - \lambda|E_y|\big] + \big[f(E_{\text{foil}}) - \lambda|E_{\text{foil}}|\big] - \beta\,|E_y \cap E_{\text{foil}}|.$$

The overlap term is the *only* coupling between the two players and it enters each player's utility identically: when player $y$ changes $E_y$ with $E_{\text{foil}}$ fixed,

$$U_2(E_y' \mid E_{\text{foil}}) - U_2(E_y \mid E_{\text{foil}}) = \Phi(E_y', E_{\text{foil}}) - \Phi(E_y, E_{\text{foil}}),$$

because the $f(E_{\text{foil}})-\lambda|E_{\text{foil}}|$ term is constant under $y$'s move and the shared $-\beta|E_y\cap E_{\text{foil}}|$ term is common to both $U_2$ and $\Phi$ (symmetrically for the foil player). So every unilateral *improving* move strictly increases the single scalar $\Phi$. Two consequences:
- **A pure-strategy Nash equilibrium exists.** The joint strategy space $2^C \times 2^C$ is finite, so $\Phi$ attains a maximum; any maximizer is a PSNE (no player can improve, since improving would raise $\Phi$ past its max). This is the finite-improvement property (FIP) of finite potential games — it does not need $C$ to be small, only finite.
- **Exact *sequential* best response cannot cycle.** If players move one at a time, each move is a unilateral improvement, hence (weakly) increases $\Phi$; a strict increase cannot repeat a state, so exact sequential best-response dynamics reach a fixed point in finitely many steps.

**Two caveats that matter in practice.**
1. *The implemented update is simultaneous (Jacobi), not sequential.* Both agents best-respond to the **same frozen opponent from the previous round** (this keeps the two players symmetric — neither sees a fresher opponent than the other). The FIP argument above is for unilateral moves; under simultaneous updates even *exact* best responses can 2-cycle (the regression tests construct exactly such an oscillation: $(\{A\},\{A\}) \leftrightarrow (\{B\},\{B\})$ under symmetric weights and high $\beta$). Existence of a PSNE is unaffected — only the convergence-of-dynamics argument changes.
2. *The inner solver is greedy*, an approximate best response, so even sequential dynamics would inherit the guarantee only up to greedy's sub-optimality in the non-submodular regime ([§3.2](#32-the-value-function-and-submodularity)).

The honest version of the convergence claim is therefore: **a PSNE exists (exact potential game); the implemented greedy-Jacobi dynamics are not guaranteed to reach it, are capped at $K$ rounds, and are protected by best-iterate tracking and an optional fictitious-play solver** (below).

**Best-iterate tracking (always on).** The solver evaluates every round's joint allocation $(E_y, E_{\text{foil}})$ by its **combined hard-overlap utility** $U_2^y + U_2^{\text{foil}} = \Phi - \beta\,|E_y\cap E_{\text{foil}}|$ (the overlap penalty is paid by *both* players, so the sum counts it twice where $\Phi$ counts it once; the two objectives coincide exactly on disjoint allocations, which is the empirically common case — overlap 0.0 in all 48 case-study runs). It **returns the best round seen, not the final iterate**, and reports `best_iteration` (0 means the initial empty allocation beat every round — a sign the $\lambda/\beta$ penalties outweigh realized faithfulness). `converged` refers to the *dynamics* (iterates repeated / frequencies stabilized), independently of which round is returned.

**Fictitious play (`solver="fp"`).** As a damping alternative to ABR, each agent can best-respond to the opponent's **empirical mixture** of past evidence sets instead of its last iterate. Because the opponent enters $U_2$ only through the overlap penalty, which is linear in membership, the expected utility against the mixture is exact: the penalty term becomes $\beta\sum_{n\in E} p_t(n)$, where $p_t(n)$ is the fraction of past rounds the opponent included $n$ — no extra oracle calls. FP stops early when best responses repeat or when both agents' empirical frequencies change by less than `fp_tol`; round 1 is identical to ABR round 1. Reported metrics always use the hard overlap of the returned joint allocation, so ABR and FP results are directly comparable; FP additionally reports the per-node inclusion frequencies (soft evidence membership).

**Empirics.** Convergence is observed within 2–4 rounds (the case-study runs used `abr_iters=4` and report `converged=True`); the first round fixes each player's high-value core and later rounds only adjust the few features near the $\beta$ margin. Near-interchangeable features for both players are the usual cause of non-convergence.

### 3.5 Complexity

| Component | Per-prompt complexity |
|-----------|----------------------|
| Game 1 greedy sweep | $O(\|E^*\| \cdot \|C\|)$ oracle calls |
| Game 1 with prefilter | $O(k)$ prefilter + $O(\|E^*\| \cdot k)$ greedy |
| Game 2 single ABR iteration | $O(\|E_y\| \cdot \|C\| + \|E_{\text{foil}}\| \cdot \|C\|)$ oracle calls |
| Game 2 full ABR | $O(K \cdot (\|E_y\| + \|E_{\text{foil}}\|) \cdot \|C\|)$ oracle calls |
| Oracle call | 1 forward pass through transformer with modified activations |

where $|E^*|$ is the final evidence set size, $|C|$ is the candidate pool size, $k$ is the prefilter budget, and $K$ is the ABR iteration count.

The **oracle memoization cache** reduces actual forward passes significantly. The cache key is `(mode, type(target), str(target), frozenset(nodes), universe_fingerprint)` ([§2.1](#21-oracle-scoring)), so identical intervention sets across different game iterations or agents are computed only once. Empirically, cache hit rates range from 30–60% for Game 1 and 50–80% for Game 2 (where the two agents probe overlapping subsets). Oracle-call/cache-hit counters are reset at solver entry, so reported stats are per-solve even when one oracle is reused across games.

### 3.6 Relation to Shapley and Banzhaf Values

Because the games are built on the coalitional game $(N, v)$ ([§3.0](#30-the-underlying-coalitional-game)), the classical per-player credit measures are directly available, and they clarify exactly what Game 1's greedy *is* and *is not*.

For a player (feature) $i$, the marginal contribution to a coalition $S$ is $\Delta_i(S) = v(S \cup \{i\}) - v(S)$. The two gold-standard attributions average this differently:

$$\phi_i^{\text{Shapley}} = \frac{1}{|N|!}\sum_{\pi}\Delta_i(S_\pi^{<i}) \quad\text{(average over all orderings)}, \qquad \phi_i^{\text{Banzhaf}} = \frac{1}{2^{|N|-1}}\sum_{S \subseteq N\setminus i}\Delta_i(S) \quad\text{(average over all coalitions)},$$

where $S_\pi^{<i}$ is the set of players preceding $i$ in permutation $\pi$. Equivalently, Shapley weights coalitions by $\tfrac{|S|!\,(|N|-|S|-1)!}{|N|!}$ (uniform over *ranks*), while Banzhaf weights all coalitions equally (uniform over *subsets*) — the two differ only in that weighting, which is why Banzhaf is less sensitive to where in an ordering a strongly-interacting feature happens to fall.

**Why Shapley is the principled per-feature credit (axioms).** $\phi^{\text{Shapley}}$ is the unique attribution satisfying: **efficiency** ($\sum_i \phi_i = v(N) - v(\emptyset) = \text{recoverable\_range}$ — the credits exactly partition the recoverable range); **symmetry** (features with identical marginals everywhere get equal credit — the two redundant copies $a,a'$ of [§3.2](#32-the-value-function-and-submodularity) split their shared value); **null player** (a feature with $\Delta_i(S)=0$ for all $S$ gets zero — a feature that never moves the gap; note a *suppressor* whose ablation helps the target has negative, not zero, marginal value and so receives negative credit); and **linearity** (credit is additive across score functions, e.g. target-logit and foil-logit pieces of the logit gap). These axioms are exactly the properties one wants from a circuit attribution, which is what makes Shapley the right *gold* reference even though it is not a *selector*.

**What Game 1's greedy computes instead.** At each step the greedy adds $\arg\max_i \Delta_i(S)$ for the *one* coalition $S$ it has built so far — a single marginal contribution along a single, greedily chosen permutation, not an average over all of them. So:

- Game 1 answers **"which small coalition is most valuable?"** (set selection), while Shapley/Banzhaf answer **"how much is each player worth?"** (credit assignment). They are different questions over the same $v$.
- The greedy is exponentially cheaper — $O(|E^*|\cdot|C|)$ oracle calls vs. the $2^{|C|}$ coalitions Shapley/Banzhaf average over (Monte-Carlo–estimated in practice, but still far more samples than the greedy).
- When $v$ is submodular, the greedy coalition carries the $(1-1/e)$ guarantee ([§3.2](#32-the-value-function-and-submodularity)); the Shapley/Banzhaf values carry no such selection guarantee because they are not a selection rule.

**How they are used here.** The Shapley value of the coalitional game $(N, v)$ plays two roles for MACAG: (1) as the **gold-standard baseline** (§9.3) — does Game 1's greedy evidence set coincide with the top-Shapley features, and at what fraction of the oracle cost?; and (2) as a **diagnostic** — large disagreement between a feature's Shapley value and its greedy marginal flags strong feature interactions (non-submodularity), i.e. the regime where the greedy guarantee is only heuristic.

**The implementation (`macag/baselines/shapley_select.py`).** `estimate_shapley` is a Monte-Carlo *permutation-sampling* estimator over the MACAG intervention oracle. Each sampled permutation walks the candidate pool front-to-back and charges every feature its marginal $\Delta_i(S)$ at the coalition built so far. Every coalition is priced by the shared `coalition_value` helper (`macag/baselines/common.py`), which returns `compute_faithfulness_metrics(...).faithfulness_delta` — *exactly* the $v(S)$ of [§3.0](#30-the-underlying-coalitional-game), evaluated through the oracle's `keep_only`/`remove` ablations (`ReplacementModel` forward passes in the real backend) at the same $\alpha$ Game 1 optimizes; every baseline in the harness ranks under this one $v$, so only the selection rule differs across methods. With `antithetic=True` (the default) each sampled permutation is paired with its exact reversal, which cancels order noise for near-additive $v$. Because each permutation's marginals telescope to $v(N)-v(\emptyset)$, the estimate is *exactly* efficient for any number of permutations (and since both CLI drivers `restrict_universe` the oracle to the candidate pool, $v(N)$ *is* the recoverable range, so the credits partition exactly the quantity the tables report); the returned `ShapleyEstimate` carries per-node standard errors plus an `efficiency_gap` field, which for the permutation estimator should be zero up to float error (a wiring diagnostic, not a statistical one). Cost: $|C|$ coalition evaluations per permutation, each at most two fresh oracle calls (`keep_only` + `remove`; the empty/full coalitions are memoized after the first hit) — with the default 64 permutations this is what dominates the gold baseline's cost (the ~32.5k mean oracle calls of §10.7). `estimate_banzhaf` is implemented alongside it: per sample it draws a coalition $S$ by independent fair coin flips and charges every node $v(S\cup\{i\}) - v(S\setminus\{i\})$; its all-coalitions average is less order-sensitive than Shapley for highly interacting features, making it the natural second gold reference (for Banzhaf, which does not satisfy efficiency even exactly, `efficiency_gap` is a genuinely informative quantity rather than a float-error check). Both estimators are exposed through `select_top_shapley(estimator="shapley"|"banzhaf")`, which returns the same best-first `SelectionResult` as every other baseline — so "Shapley evidence at budget $k$" means the top-$k$ prefix of the estimated-credit ranking.

**Harness wiring (`macag/cli/run_baselines.py`).** `shapley` is in the default method list (`influence,eap,shapley,game1,acdc`); `banzhaf` is opt-in via `--methods`. Knobs: `--shapley-permutations` (default 64), `--banzhaf-samples` (default 64), `--shapley-seed` (default 0), `--no-antithetic`; $\alpha$ comes from the same `--alpha` all methods share. The harness's per-$k$ agreement block (precision@$k$ / Jaccard, emitted as `agreement_vs_shapley`) uses the Shapley ranking as gold when it ran, falling back to Banzhaf otherwise, and the §A.5 Spearman linearity diagnostic correlates the Shapley/EAP/influence scores against Game 1's per-step marginal gains.

**Not the spline-CLT Shapley (`attribution/shapley.py`).** The repository contains a second, unrelated Shapley module, and the two must not be conflated (nor one wrapped around the other — `shapley_select.py` deliberately does not reuse it). They share only the generic estimator technique — Monte-Carlo permutation sampling with antithetic pairing — which is precisely what invites the confusion; everything that matters is different:

| | `macag/baselines/shapley_select.py` (MACAG gold) | `attribution/shapley.py` (spline-CLT tool) |
|---|---|---|
| Players | feature *nodes of a traced circuit graph* | active features of a `KANCrossLayerTranscoder` |
| Value function $v$ | §3.0 target–foil faithfulness ($\alpha$-mixed `keep_only`/`remove`) | reconstruction-MSE reduction, or logit-direction projection |
| Evaluation backend | MACAG `ScoringOracle` → full-model interventions (`ReplacementModel` forward passes) | the transcoder alone (`encode`/`decode_dense`); no transformer in the loop |
| Purpose | gold per-feature credit to benchmark Game 1's selection | edge weights for the spline-CLT attribution graph |
| Entry points | `estimate_shapley`, `estimate_banzhaf`, `select_top_shapley` | `shapley_attribution`, `shapley_logit_attribution` |

`attribution/shapley.py` must never be cited as MACAG's Shapley-gold: it is a different $v$ on a different object and is not wired to the MACAG oracle.

**Status.** Gold-baseline *numbers* exist for the **nonlinear benchmark** (§10.7/C.7: MC Shapley faith\@8 4.72 [4.02, 5.45] at 32 495 oracle calls, vs Game 1's 5.50 at 718 — a 44.7× cost gap), in addition to the toy-oracle unit coverage (`tests/test_macag_baselines.py`); the estimator has not yet been run on the IOI/multi-hop graphs, and multi-seed reruns of the MC estimator remain open (§12.3).

---

## 4. Game 1: Minimal Faithful Evidence

### 4.1 Objective

Find a small subset of feature nodes that explains the model's prediction (smallness is priced into the objective via $\lambda$, not certified — §1.4 guardrails):

$$E^* = \arg\max_{E \subseteq C} \left[\alpha \cdot \left(S_{\text{keep}}(E) - S_{\text{empty}}\right) + (1 - \alpha) \cdot \left(S_{\text{all}} - S_{\text{remove}}(E)\right) - \lambda |E|\right]$$

### 4.2 Algorithm: Greedy Hill-Climbing

**Pseudocode**:
```
Algorithm: MACAG Game 1 — Greedy Hill-Climbing
──────────────────────────────────────────────
Input:  Graph G, Oracle O, target y, candidates C,
        α, λ, ε (optional), stop_metric ∈ {normalized, raw_relative},
        budget B (optional)
Output: Evidence set E*, utility U*

1.  E ← ∅,  first_faith_gain ← nil
2.  if prefilter_top_k:
3.      C ← PREFILTER(C, O, y, α, λ, top_k)    // rank by singleton U({n})
4.  repeat
5.      if B ≠ nil and |E| ≥ B: break
6.      U_curr ← α·(O.keep(E,y) - O.empty(y)) + (1-α)·(O.all(y) - O.remove(E,y)) - λ|E|
7.      n* ← nil,  best_gain ← min_gain
8.      for each n ∈ C \ E:
9.          if connected and ¬CONNECTED_THROUGH(E ∪ {n}): skip   // hub-excluding, §4.4
10.         U_trial ← UTILITY(E ∪ {n})
11.         gain ← U_trial - U_curr
12.         if gain > best_gain:
13.             best_gain ← gain,  n* ← n
14.             faith_gain* ← Δfaith(E ∪ {n}) − Δfaith(E)   // λ-free faithfulness gain
15.         else if gain = best_gain and str(n) < str(n*):
16.             n* ← n,  faith_gain* ← Δfaith(E ∪ {n}) − Δfaith(E)
17.     if n* = nil: break                          // no improving move
18.     // raw_relative stop (BEFORE adding): diminishing returns vs first feature,
19.     // measured on RAW faithfulness gains (no λ penalty) so λ cannot distort it
20.     if ε ≠ nil and stop_metric = raw_relative and first_faith_gain > 0
21.            and faith_gain* < ε · first_faith_gain: break
22.     E ← E ∪ {n*}
23.     if first_faith_gain = nil: first_faith_gain ← faith_gain*
24.     // normalized stop (AFTER adding): error-floor-aware faithfulness target
25.     if ε ≠ nil and stop_metric = normalized
26.            and faithfulness_norm(E) ≥ 1 − ε: break
27. return E, UTILITY(E)
```

**Step-by-step**:

1. Initialize $E = \emptyset$
2. **(Optional) Prefilter**: Rank all candidates by singleton utility $U(\{n\})$, keep top-$k$ (default: no prefilter). This uses the same `game1_utility()` function as the main solver, biasing toward sparsity-conscious candidate selection.
3. At each step, evaluate all remaining candidates and add the one with the highest marginal utility gain: $$n^* = \arg\max_{n \in C \setminus E} \left[U(E \cup \{n\}) - U(E)\right]$$ Ties are broken alphabetically (by `_sort_key()`) for deterministic reproducibility.
4. Stop when:
   - No candidate provides positive marginal gain exceeding `min_gain`, or
   - Budget $|E| \geq B$ is reached, or
   - The faithfulness early-stop condition (set by `faithfulness_eps` $=\varepsilon$ and `stop_metric`) is met — see below.

**Stop metric** (`stop_metric`, default `normalized`) controls how the optional $\varepsilon$ early-stop is interpreted:

- **`normalized`**: after each addition, stop when the error-floor-aware faithfulness reaches its target, $\text{faithfulness}_{\text{norm}}(E) \geq 1 - \varepsilon$. This is the correct target with **frozen attention**, but goes degenerate when `recoverable_range` collapses toward zero/negative (e.g. unfrozen attention), producing spurious early/late stops.
- **`raw_relative`**: a denominator-free diminishing-returns rule checked *before* adding a node — stop when the best available marginal raw faithfulness gain falls below $\varepsilon \cdot (\text{the first added feature's gain})$. The gains in this test are the $\lambda$-free `faithfulness_delta` increases, NOT the $\lambda$-penalized utility gains the greedy maximizes — otherwise the sparsity penalty would shift both sides of the ratio test and distort the stop whenever $\lambda > 0$. Because it never divides by `recoverable_range`, it is stable when that range is unreliable, and is the recommended choice for **unfrozen-attention** and attention-mediated (IOI-style) runs.

**Approximation guarantee**: If the faithfulness function $f(E)$ is submodular and monotone, the greedy algorithm achieves a $(1 - 1/e) \approx 0.632$ approximation ratio to the optimal solution under a cardinality constraint (Nemhauser et al., 1978). The sparsity penalty $-\lambda|E|$ is modular and does not affect the approximation ratio. In practice, neural network logit gaps are not guaranteed submodular, so this serves as a best-case bound rather than a formal guarantee.

### 4.3 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| $\alpha$ | 0.5 | Sufficiency vs. necessity balance |
| $\lambda$ | 0.01 | Sparsity penalty coefficient |
| `faithfulness_eps` ($\varepsilon$) | None | Early-stop threshold in $[0,1]$; interpreted via `stop_metric` |
| `stop_metric` | resolved from `freeze_mode` | `normalized` (error-floor-aware) or `raw_relative` (denominator-free). CLI default: `normalized` for `--freeze-mode frozen`, `raw_relative` for `unfrozen`/`both`; `normalized` + `both` is rejected (non-comparable legs) |
| `freeze_mode` (CLI) | `frozen` | `frozen`: factory-built oracle as-is; `unfrozen`: derive a freeze-flipped oracle; `both`: matched dual run + `attention_mediation` diagnostic. `unfrozen`/`both` require a ReplacementModel-backed oracle (not `--toy-oracle-json`) |
| budget | None | Hard maximum on evidence set size |
| prefilter_top_k | None | Pre-filter candidates by singleton gain |
| connected | false | Require evidence to form a connected subgraph |
| min_gain | 0.0 | Minimum marginal gain required to add a node |

### 4.4 Connectivity Constraint

When `connected=true`, the algorithm only considers candidates that would maintain weak connectivity of the evidence set (checked via BFS treating the graph as undirected, routing through intermediate nodes that are not themselves candidates). This produces more interpretable circuits at the cost of potentially lower faithfulness.

**Hub exclusion**: logit and embedding nodes are NOT allowed as connectivity intermediates. In pruned attribution graphs they are extreme hubs — in a representative GPT-2 graph, ~47% of all edges terminate in 10 logit nodes, making the entire graph a single weak component. Routing connectivity through them would render the constraint vacuous (every feature pair "connected" because both influence the output). With the exclusion, connectivity means membership in the same feature/error-node sub-circuit: on the same graph, the constraint distinguishes one 291-node computational component from 46 isolated features instead of accepting all pairs. Error nodes remain valid intermediates (they are part of the per-layer computation path). The exclusion set is configurable via `CircuitGraph.connected_through(..., exclude_intermediate_types=...)`; passing `None` restores the permissive behavior.

### 4.5 Output

`EvidenceSetResult` containing:
- **evidence**: the selected node set $E^*$
- **selected_order**: insertion order of nodes (for reproducibility)
- **induced_subgraph**: the subgraph induced by $E^*$
- **metrics**: `FaithfulnessMetrics` — raw scores (`all`, `empty`, `keep_only`, `remove`, `sufficiency`, `necessity`, `faithfulness`) plus the error-floor-aware view (`error_floor`, `recoverable_range`, `sufficiency_normalized`, `necessity_normalized`, `faithfulness_normalized`)
- **utility**: final utility score
- **sparsity**: fraction of candidates not selected
- **iterations**: number of greedy steps taken
- **candidate_count** / **total_candidates**: candidates after prefilter / before prefilter
- **oracle_calls**, **cache_hits**, **cache_size**: oracle/memoization profile
- **params**: the resolved knobs (`alpha`, `lambda`, `budget`, `faithfulness_eps`, `stop_metric`, `prefilter_top_k`, `connected`, `min_gain`)

The CLI (`run_macag game1`) serializes this as `{params, evidence, scores, stats}`, mirroring the Game 1 evidence into the Game 2 schema keys (`E_star`/`E_y`/ `unique_y`) so a single annotator handles both games.

**Dual-freeze output** (`--freeze-mode both`). The single-mode schema is unchanged (byte-compatible: same top-level keys, no `freeze_mode` key). The dual run instead emits:

```json
{
  "input_id": "...", "target": "...", "foil": null, "game": "game1",
  "freeze_mode": "both",
  "params": { "...shared matched solver params...", "stop_metric": "raw_relative", "matched": true },
  "frozen":   { "params": {"...", "freeze_attention": true},  "evidence": {"E_star": ["..."]}, "scores": {"..."}, "stats": {"..."} },
  "unfrozen": { "params": {"...", "freeze_attention": false}, "evidence": {"E_star": ["..."]}, "scores": {"..."}, "stats": {"..."} },
  "attention_mediation": {
    "range_frozen": -3.1, "range_unfrozen": 4.2,
    "range_flip": true, "reverse_flip": false, "verdict": "attention_mediated",
    "evidence_size_frozen": 4, "evidence_size_unfrozen": 7,
    "evidence_jaccard": 0.3,
    "evidence_shared": ["..."], "evidence_only_frozen": ["..."], "evidence_only_unfrozen": ["..."],
    "upstream_count_frozen": 0, "upstream_count_unfrozen": 3,
    "early_count_frozen": 0, "early_count_unfrozen": 2,
    "n_layers": 26, "final_ctx_idx": 8
  }
}
```

Each leg sub-dict is exactly the single-mode payload minus the envelope, so the annotator (`annotate_graph --freeze-select {frozen,unfrozen,both}`) reuses the same group builder per leg (`MACAG:frozen:E_star` / `MACAG:unfrozen:E_star`). The `attention_mediation` verdict rule (strict zero threshold; raw ranges are reported so confidence bands can be applied downstream):

| `range_frozen` | `range_unfrozen` | `verdict` | flags |
|---|---|---|---|
| $<0$ | $\ge0$ | `attention_mediated` | `range_flip` (the §10.4 flip) |
| $\ge0$ | $\ge0$ | `feature_mediated` | — |
| $<0$ | $<0$ | `indeterminate` (behavior not recoverable from features under either convention) | — |
| $\ge0$ | $<0$ | `indeterminate` (unexpected inversion) | `reverse_flip` |

`upstream_count_*` counts evidence nodes at `ctx_idx < final_ctx_idx` (equivalent to the legacy reverse-position $>0$ convention — logit nodes carry the final prompt position, so the max position over the graph is the prediction token); `early_count_*` counts evidence nodes with `layer < n_layers/3`, with `n_layers` inferred from feature/error-node layers only. Layer/position resolve from node metadata first, then the `{layer}_{feature}_{pos}` node-ID convention; if the graph carries neither, the count fields are `null` (keys always present — stable schema for downstream aggregation). Note "matched" means matched *parameters*: each leg's prefilter ranks singletons under its own oracle, so the retained pools may differ — same $k$, mode-specific gains, by design.

---

## 5. Game 2: Contrastive Evidence

### 5.1 Objective

Find two evidence sets — one for the target class, one for the foil — that are maximally faithful to their respective classes while being minimally overlapping:

$$E_y^* = \arg\max_{E_y} \left[\text{faithfulness}(E_y) - \lambda |E_y| - \beta |E_y \cap E_{\text{foil}}^*|\right]$$
$$E_{\text{foil}}^* = \arg\max_{E_{\text{foil}}} \left[\text{faithfulness}(E_{\text{foil}}) - \lambda |E_{\text{foil}}| - \beta |E_{\text{foil}} \cap E_y^*|\right]$$

where $\beta \geq 0$ is the **overlap penalty** (default $\beta = 0.1$), encouraging the two evidence sets to identify *different* features for target vs. foil.

### 5.2 Algorithm: Simultaneous Best Response (the "ABR" solver)

> *Naming note:* the solver is called `abr` in code and CLI for historical
> reasons, but the implemented update is **simultaneous (Jacobi)**, not
> alternating (Gauss-Seidel): in every round, *both* agents best-respond to the
> opponent's evidence set **from the previous round**.

**Pseudocode**:
```
Algorithm: MACAG Game 2 — Simultaneous Best Response ("abr")
─────────────────────────────────────────────────────────────
Input:  Graph G, Oracle O, targets (y, y_foil), candidates C,
        α, λ, β, K (max rounds), budget B (optional)
Output: Best joint allocation (E_y*, E_foil*), convergence flag, best_iteration

1.  E_y ← ∅,  E_foil ← ∅
2.  best ← evaluate(∅, ∅),  best_iteration ← 0       // combined hard-overlap utility
3.  for k = 1, ..., K:
4.      // BOTH players respond to the SAME frozen opponent from round k-1
5.      E_y'    ← GREEDY_BEST_RESPONSE(G, O, y,      E_foil, C, α, λ, β, B)
6.      E_foil' ← GREEDY_BEST_RESPONSE(G, O, y_foil, E_y,    C, α, λ, β, B)
7.      if evaluate(E_y', E_foil') > best:
8.          best ← evaluate(E_y', E_foil'),  best_iteration ← k
9.      if E_y' = E_y and E_foil' = E_foil:
10.         converged ← true; break
11.     E_y ← E_y',  E_foil ← E_foil'
12. return best allocation, converged, best_iteration   // best round, NOT last iterate

──────────────────────────────────────────────────────────────────────────────
Subroutine: GREEDY_BEST_RESPONSE(G, O, target, w_other, C, α, λ, β, B,
                                  connected, min_gain, prefilter_top_k)
──────────────────────────────────────────────────────────────────────────────
Input:  Graph G, Oracle O, target,
        w_other : NodeId → [0,1]   // opponent inclusion weight. ABR passes hard
                                    // 0/1 weights (1.0 for every node in the
                                    // opponent's LAST evidence set); FP passes the
                                    // opponent's empirical per-node inclusion
                                    // frequency p_t(n) (§5.2.1). Both are handled
                                    // by the same linear overlap term below.
        candidates C, α, λ, β, budget B (optional),
        connected, min_gain, prefilter_top_k (optional)
Output: Best-response evidence set E

// U₂(E) = faithfulness(E) − λ|E| − β·Σ_{n∈E} w_other(n)
// This is exact for both ABR (w_other ∈ {0,1}, so the sum is the hard overlap
// |E ∩ E_other|) and FP (w_other ∈ [0,1], so the sum is the EXPECTED overlap
// against the opponent's empirical mixture — no extra oracle calls either way).

1.  E ← ∅
2.  if prefilter_top_k:
3.      C ← rank C by U₂({n}) for each singleton n ∈ C, keep top-k    // PREFILTER_WITH_OVERLAP
4.  repeat
5.      if B ≠ nil and |E| ≥ B: break
6.      U_curr ← U₂(E)
7.      n* ← nil,  best_gain ← min_gain
8.      for each n ∈ C \ E:
9.          if connected and |E ∪ {n}| > 1 and ¬CONNECTED_THROUGH(E ∪ {n}): skip
10.         gain ← U₂(E ∪ {n}) − U_curr
11.         if gain > best_gain:
12.             best_gain ← gain,  n* ← n
13.         else if gain = best_gain and str(n) < str(n*):
14.             n* ← n
15.     if n* = nil: break                          // no improving response
16.     E ← E ∪ {n*}
17. return E
```

This is the single greedy routine both solvers call — it differs from the Game 1 greedy of [§4.2](#42-algorithm-greedy-hill-climbing) only in the utility ($U_2$ instead of $U_1$) and in taking `w_other` as an extra input; the ABR/FP top-level loops differ only in what they pass as `w_other` (hard opponent membership vs. empirical frequency).

**Step-by-step**:

1. Initialize $E_y = \emptyset$, $E_{\text{foil}} = \emptyset$
2. Repeat for up to $K$ rounds (default $K = 10$):
   - **Player $y$**: Greedy hill-climb to optimize $U_2(E_y \mid E_{\text{foil}}^{(k-1)})$
   - **Player foil**: Greedy hill-climb to optimize $U_2(E_{\text{foil}} \mid E_y^{(k-1)})$
   - Both see the same round-$(k{-}1)$ opponent (Jacobi symmetry), so neither player has a first-mover information advantage and the players are exactly exchangeable — symmetric oracles provably yield symmetric per-agent utilities (pinned by a regression test).
3. Track the best joint allocation seen (by combined hard-overlap utility); stop early when both responses repeat, else stop at the round cap.
4. **Return the best round's allocation** and `best_iteration`; `converged` describes the dynamics, not the returned round ([§3.4](#34-equilibrium-analysis)).

Each player's greedy step uses a modified utility that includes the overlap penalty:

$$U_2(E, E_{\text{other}}) = \text{faithfulness}(E) - \lambda |E| - \beta \cdot \text{overlap}(E, E_{\text{other}})$$

**Candidate prefiltering** in Game 2 accounts for the existing evidence from the other player: `_prefilter_with_overlap_penalty()` ranks candidates using the full $U_2$ function, not just singleton gain.

**Convergence**: the solver reports `converged=true` when $(E_y^{(k)}, E_{\text{foil}}^{(k)}) = (E_y^{(k-1)}, E_{\text{foil}}^{(k-1)})$ — a fixed point at which neither agent's greedy response changes, i.e. a greedy approximation of a pure-strategy Nash equilibrium. A PSNE exists because Game 2 is an exact potential game, but the simultaneous update means even exact best responses can 2-cycle, and the greedy inner solver is approximate — see [§3.4](#34-equilibrium-analysis) for the precise statement. In practice the solver stabilizes in 2–4 rounds because:
- The greedy inner solver is deterministic given fixed opponents
- The overlap penalty (case study $\beta = 0.2$) is mild enough that agents settle quickly once they establish non-overlapping "cores"
- The finite candidate pool (hundreds of nodes in the case study) limits the space of possible oscillations

When the dynamics do oscillate, best-iterate tracking returns the highest-combined-utility round, so a cycling pair $(\{A\},\{A\}) \leftrightarrow (\{B\},\{B\})$ does not degrade the reported result.

### 5.2.1 Fictitious-Play Solver (`solver="fp"`)

The second solver damps oscillation by replacing the opponent's last iterate with its **empirical history**: at round $t$, each agent best-responds to the mixture that includes node $n$ with probability $p_t(n)$ = the fraction of past rounds the opponent's evidence contained $n$. Because the opponent enters $U_2$ only through the overlap penalty, which is linear in node membership, the expected utility against this mixture is computed *exactly* as $\beta\sum_{n\in E} p_t(n)$ — no additional oracle calls over ABR.

**Pseudocode**:
```
Algorithm: MACAG Game 2 — Fictitious Play ("fp")
─────────────────────────────────────────────────────────────
Input:  Graph G, Oracle O, targets (y, y_foil), candidates C,
        α, λ, β, K (max rounds), budget B (optional), fp_tol
Output: Best joint allocation (E_y*, E_foil*), convergence flag, best_iteration,
        node_frequencies_y, node_frequencies_foil

1.  E_y ← ∅,  E_foil ← ∅
2.  counts_y ← {},  counts_foil ← {}         // per-node inclusion counts across rounds
3.  freq_y ← {},  freq_foil ← {}             // empirical inclusion frequencies p_t(n)
4.  best ← evaluate(∅, ∅),  best_iteration ← 0    // combined HARD-overlap utility, §3.4
5.  for k = 1, ..., K:
6.      // round 1: freq_y = freq_foil = {} ⇒ identical to ABR round 1
7.      E_y'    ← GREEDY_BEST_RESPONSE(G, O, y,      w_other=freq_foil, C, α, λ, β, B)
8.      E_foil' ← GREEDY_BEST_RESPONSE(G, O, y_foil, w_other=freq_y,    C, α, λ, β, B)
9.      if evaluate(E_y', E_foil') > best:              // evaluate() uses HARD overlap
10.         best ← evaluate(E_y', E_foil'),  best_iteration ← k
11.     for n ∈ E_y':    counts_y[n]    ← counts_y.get(n, 0) + 1
12.     for n ∈ E_foil': counts_foil[n] ← counts_foil.get(n, 0) + 1
13.     freq_y'    ← { n : counts_y[n] / k    for n in counts_y }
14.     freq_foil' ← { n : counts_foil[n] / k for n in counts_foil }
15.     Δfreq ← max( max_n |freq_y'(n) − freq_y(n)|, max_n |freq_foil'(n) − freq_foil(n)| )
16.     if E_y' = E_y and E_foil' = E_foil:
17.         converged ← true
18.     else if k > 1 and Δfreq < fp_tol:
19.         converged ← true
20.     freq_y ← freq_y',  freq_foil ← freq_foil'
21.     E_y ← E_y',  E_foil ← E_foil'
22.     if converged: break
23. return best allocation, converged, best_iteration, freq_y, freq_foil    // best round, NOT last iterate
```

Round 1 matches ABR round 1 exactly because `freq_y = freq_foil = {}` (an empty mapping means `w_other(n) = 0` for every candidate, the same as ABR's round-1 empty opponent set). From round 2 onward the two solvers diverge: ABR's `w_other` is the opponent's single last set (hard 0/1), FP's is the running empirical frequency (soft, in $[0,1]$) — both are consumed by the identical `GREEDY_BEST_RESPONSE` subroutine of [§5.2](#52-algorithm-simultaneous-best-response-the-abr-solver), so the two top-level loops share every line of the inner greedy and differ only in what they feed it as the opponent weights and in the extra frequency-convergence check (line 18).

Properties (all pinned by regression tests):
- **Round 1 is identical to ABR round 1** (both respond to an empty history).
- **Early stop** when the best responses repeat, or when the empirical frequencies of both agents change by less than `fp_tol` (default $10^{-3}$).
- **Reported metrics always use the hard overlap** of the returned joint allocation, so ABR and FP results are directly comparable.
- FP additionally returns `node_frequencies_y` / `node_frequencies_foil` — the per-node empirical inclusion frequencies, a *soft evidence membership* useful when near-interchangeable features make any single hard set arbitrary.

### 5.3 Evidence Decomposition

After convergence:

$$\text{shared} = E_y \cap E_{\text{foil}}, \quad \text{unique}_y = E_y \setminus E_{\text{foil}}, \quad \text{unique}_{\text{foil}} = E_{\text{foil}} \setminus E_y$$

**Overlap rate** (Jaccard similarity):

$$\text{overlap\_rate} = \frac{|E_y \cap E_{\text{foil}}|}{|E_y \cup E_{\text{foil}}|}$$

Lower overlap indicates that the model uses distinct features for target vs. foil predictions — a sign of well-separated, interpretable circuits.

### 5.4 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| $\beta$ | 0.1 | Overlap penalty between target and foil evidence |
| $K$ (`abr_iters`) | 10 | Maximum solver rounds (ABR or FP) |
| `solver` | `"abr"` | `"abr"` (respond to opponent's last set) or `"fp"` (fictitious play: respond to opponent's empirical history, [§5.2.1](#521-fictitious-play-solver-solverfp)) |
| `fp_tol` | $10^{-3}$ | FP only: stop when both agents' empirical node frequencies change by less than this |
| All Game 1 parameters | (same defaults) | $\alpha$, $\lambda$, budget, prefilter, connected (Game 2 has no `faithfulness_eps`/`stop_metric`) |

### 5.5 Output

`ContrastiveEvidenceResult` containing:
- **evidence_y**, **evidence_foil**: the returned joint allocation — the **best-combined-utility round**, not necessarily the final iterate ([§3.4](#34-equilibrium-analysis))
- **shared**, **unique_y**, **unique_foil**: decomposition
- **metrics_y**, **metrics_foil**: separate `FaithfulnessMetrics` (same raw + normalized fields as Game 1), evaluated on the returned allocation
- **utility_y**, **utility_foil**: each side's $U_2$ under the hard overlap of the returned allocation
- **overlap_rate**: Jaccard similarity
- **converged**: whether the *dynamics* stabilized (iterates repeated, or FP frequencies within `fp_tol`) — a property of the run, not of the returned round
- **iterations**: number of solver rounds taken
- **best_iteration**: the round whose allocation is returned (0 = the initial empty allocation beat every solver round; the solver logs a warning when this happens despite non-empty rounds, since it means $\lambda/\beta$ outweighed realized faithfulness)
- **node_frequencies_y**, **node_frequencies_foil**: FP only — empirical per-node inclusion frequencies across rounds (empty dicts under ABR)
- **total_candidates**, **sparsity_y**, **sparsity_foil**: candidate pool and per-side sparsity
- **oracle_calls**, **cache_hits**, **cache_size**: oracle/memoization profile (per-solve)
- **params**: resolved knobs (adds `beta`, `abr_iters`, `solver`, `fp_tol` to the Game 1 set)

**Note — Game 2 is unaffected by the `stop_metric` issue.** Game 2 selects on the raw $U_2$ utility via ABR and has no `faithfulness_eps` early-stop, and `overlap_rate` is denominator-free, so neither depends on `recoverable_range`. The frozen→unfrozen attention switch therefore changes only the underlying faithfulness scores, not how Game 2 selects.

### 5.6 Foil Resolution

Game 2 requires a target-foil pair. The foil mapping is configured as:
- If exactly 2 targets: auto-creates bidirectional map (e.g., " Paris" ↔ " Lyon")
- Otherwise: requires explicit `foil_by_target` mapping

Target tokens are resolved to vocabulary indices via the model's tokenizer, supporting:
- Direct integer indices
- String tokens (tokenized, enforcing single-token constraint)
- Explicit `"id:<int>"` format

---

## 6. Graph and Candidate Selection

### 6.1 Circuit Graph Wrapper

MACAG operates on a lightweight directed graph (`CircuitGraph`) with per-node metadata:

```python
class CircuitGraph:
    _node_metadata: dict[NodeId, dict[str, Any]]
    _succ, _pred: dict[NodeId, set[NodeId]]  # adjacency lists
```

Key operations:
- `subgraph(nodes)`: extracts induced subgraph for evidence visualization (nodes/edges sorted for cross-process determinism)
- `is_weakly_connected(nodes)`: subset-internal BFS connectivity check (treats graph as undirected)
- `connected_through(nodes)`: the connectivity check used by the solvers — routes through intermediate feature/error nodes but excludes logit/embedding hubs (see §4.4)
- `from_dict()` / `to_dict()`: JSON serialization compatible with circuit-tracer graph format (`node_id` takes precedence over `id`, matching the intervention loader)

### 6.2 Candidate Extraction

Candidates are extracted from circuit-tracer graph JSON:
1. Scan all nodes for matching `feature_type` (default: `"cross layer transcoder"`)
2. Parse node ID format: `"{layer}_{feature}_{position}"` → `(layer, position, feature_idx)` tuple
3. Build intervention map: each candidate maps to a `(layer, ctx_idx, feature_idx, 0.0)` ablation spec

**Filtering**: Candidates are restricted to feature nodes present in the graph JSON, ensuring MACAG and the scorer stay aligned to the traced circuit. The `node_universe` parameter on `ReplacementModelInterventionScorer` enforces this restriction.

### 6.3 Candidate Policies

The paper runner supports configurable candidate selection (`candidate_policy.strategy`):
- **graph_features** (default): feature nodes from the circuit graph ranked by `influence` (tie-broken by activation, then node ID), truncated to the policy's `top_k` (default 40)
- **auto_supernodes**: runs the supernode proposer ([§8.4](#84-supernode-suggestion)) over the graph and takes the union of all proposed supernode members as the candidate pool (the proposed groups are also written to `auto_supernodes.json`)

Both strategies share the `top_k` / `feature_types` knobs; there is no separate "top_k" strategy — truncation is a parameter of both.

---

## 7. Oracle Backend

### 7.1 ReplacementModel Scorer

The primary oracle backend uses circuit-tracer's `ReplacementModel` for real interventions:

```python
class ReplacementModelInterventionScorer:
    def score_keep_only(self, nodes, target) -> float
    def score_remove(self, nodes, target) -> float
    def score_all(self, target) -> float
    def score_empty(self, target) -> float
```

Each intervention:
1. Maps node IDs to `(layer, position, feature_idx, ablation_value)` specs
2. Calls `model.feature_intervention()` with the spec list
3. Extracts last-token logits from the output
4. Computes scalar score via configurable mode: `logit_gap` (default), `logit`, `prob`, `negative_loss`, or `kl_divergence` (full-distribution, target-free; reference logits cached from the clean pass — [§2.5](#25-kl-rescoring-a-selection-independent-faithfulness-metric))

By default (`freeze_attention=True`), interventions hold attention patterns and LayerNorm denominators at their clean values, isolating the direct effect of feature ablation from second-order attention changes. Setting `freeze_attention=False` lets attention recompute from the ablated activations, capturing attention-mediated effects; the four scoring methods above (`score_empty` in particular) are evaluated under whichever mode is set, which is why the flag changes every derived metric. See [§2.3](#23-attention-freezing-and-the-error-floor).

### 7.2 Auto-Detection of CLT Variant

The factory `create_replacement_model_scorer()` auto-detects the CLT type:
- If checkpoint directory contains `metadata.safetensors` → loads via `load_spline_clt()` (Spline-CLT)
- Otherwise → loads via `load_clt()` from circuit-tracer (standard linear CLT)

This makes MACAG transparent to the encoder architecture — the same evaluation code runs on both.

### 7.3 Toy Oracle

For testing and development, `ToyAdditiveInterventionScorer` provides a simple additive model:
- Each node has a fixed weight
- `keep_only(E)` = base_score + $\sum_{n \in E}$ weight($n$)
- `remove(E)` = all_score - $\sum_{n \in E}$ weight($n$)

This allows unit testing of Game 1/Game 2 solvers without a transformer model.

---

## 8. Integration Pipeline

> **Two consumption paths (read this first).** MACAG is consumed two ways, and
> mixing them is a recurring source of confusion: (1) the **paper-runner path**
> ([§8.2](#82-paper-runner-integration), [§6.3](#63-candidate-policies)) — the
> Spline-CLT/GPT-2 paper suites, where candidate selection goes through
> `candidate_policy` (top-k-by-influence, default 40) and results land in
> `macag_records.jsonl`; and (2) the **CLI path** (`macag/cli/run_macag.py` +
> `scripts/run_macag_*.sh`), used by the §9 case study, where the candidate set
> is the graph's full feature-node set (~260–338 per prompt in the case study)
> and any narrowing happens via the solver's `prefilter_top_k`. The case study
> did **not** use §6.3's candidate policies.

### 8.1 End-to-End Flow

```
1. Obtain a CLT (train one, or load a public checkpoint as in the case study)
2. Generate circuit graph via causal attribution for each prompt
3. Extract candidate feature nodes from graph (feature_type = "cross layer transcoder")
4. Build ReplacementModel scorer (auto-detects CLT variant)
5. Run Game 1 and/or Game 2 on each prompt (optionally over multiple ε / seeds)
6. Record per-prompt metrics to macag_records.jsonl
7. Aggregate metrics across prompts (and seeds / variants if present)
8. (Optional) Annotate graph with evidence sets for visualization
```

### 8.2 Paper Runner Integration

The paper suite runner (`PaperSuiteRunner`) orchestrates MACAG as a stage:
- Reads `prompt_metrics.jsonl` from the evaluation stage (filtering by `include_macag=true`)
- For each prompt: loads graph JSON, selects candidates, runs configured games
- Supports multiple Game 1 configurations (e.g., $\varepsilon = 0.10$ and $\varepsilon = 0.05$) and Game 2 configurations in a single pipeline invocation
- Records to `macag_records.jsonl` with full provenance (checkpoint path, git commit, seed, suite name)

### 8.3 Graph Annotation

After MACAG completes, evidence sets can be merged back into the circuit-tracer graph JSON for visualization:

- **Game 1**: Creates a single supernode `"[prefix]:E_star"` containing all evidence nodes
- **Game 2**: Creates supernodes for `shared`, `unique_y`, and `unique_foil` with descriptive labels
- Merges into `qParams.pinnedIds` (individual nodes) and `qParams.supernodes` (grouped nodes)
- Updates `graph-metadata.json` index (with graceful failure if circuit-tracer metadata indexing is unavailable)

### 8.4 Supernode Suggestion

For large circuit graphs, `suggest_supernodes.py` automatically discovers candidate supernodes:
1. **Salience ranking**: scores nodes by a weighted combination of influence, activation magnitude, and token probability
2. **Grouping**: finds connected components via BFS, optionally splits by context position
3. **Chunking**: breaks large groups into `max_group_size` (default 12) chunks
4. **Labeling**: auto-generates labels from the most common `clerp` (human-readable feature description) metadata

---

## 9. Case Study: Attention Mediation and CLT Capacity (Gemma-2, Llama-3.2)

> The remainder of this document (§9–§11) is **one application** of MACAG, not part
> of the framework. It feeds MACAG the attribution graphs of three publicly
> available cross-layer transcoders and runs the identical game machinery on each.
> Nothing in §2–§8 depends on these results; a different study would swap in
> different graphs and read the same metrics. *(These runs use public linear CLTs
> on Gemma-2 / Llama-3.2; they do not include a Spline-CLT or GPT-2 — see
> [§11.1](#111-what-this-case-study-does-not-yet-cover).)*

### 9.1 Setup

**Models and transcoders.** Three CLTs span a clean capacity control and a cross-model check:

| Tag | Model | Transcoder set | Candidate pool $|C|$ (per prompt) |
|-----|-------|----------------|-----------------------------------|
| `gemma2-426k` | google/gemma-2-2b | `mntss/clt-gemma-2-2b-426k` | ~311–338 |
| `gemma2-2.5M` | google/gemma-2-2b | `mntss/clt-gemma-2-2b-2.5M` | ~297–321 |
| `llama32-524k` | meta-llama/Llama-3.2-1B | `mntss/clt-llama-3.2-1b-524k` | ~260–291 |

`gemma2-426k` → `gemma2-2.5M` is the **capacity** axis (model and architecture fixed, ~6× transcoder width); `gemma2-*` → `llama32-524k` is the **cross-model** axis. A fourth CLT (`gpt-oss-20b / mntss/clt-131k`) was configured but **did not run**: `transformer_lens` does not recognize `openai/gpt-oss-20b`, so it is absent from all results.

**Prompts.** Two sets:
- *Two-hop factual* — 8 prompts of the form "Fact: The capital of the state containing {CITY} is", target = capital, foil = state (the city→state→capital circuit), shared across all three CLTs.
- *ACDC benchmark* — `indirect_object_identification` (10) and `docstring_completion` (3); `greater_than` is excluded because its target/foil share a first token, giving a degenerate first-token logit gap.

**Parameters** (verified against the stored run JSONs). Game 1 **frozen**: $\alpha{=}0.5$, $\lambda{=}0.02$, budget 8, prefilter top-20, $\varepsilon{=}0.1$, `normalized` stop. Game 1 **unfrozen** (two-hop and ACDC): same $\alpha$, $\lambda$, $\varepsilon$ but budget **20** and prefilter top-**30** — *not* parameter-matched to the frozen legs ([§2.3](#23-attention-freezing-and-the-error-floor)). Game 2 (frozen *and* unfrozen): $\beta{=}0.2$, ABR iters 4, budget 8, prefilter top-20. Each prompt is run **frozen** and, reusing the same graph, **unfrozen**. Unfrozen stop metrics: the two-hop unfrozen rerun in `macag_unfrozen/` used the `normalized` stop (against §2.3's own recommendation — see the C.3 source note); a second `raw_relative` rerun lives in `macag_unfrozen_raw/`; the ACDC-benchmark unfrozen runs used `raw_relative` because their frozen `recoverable_range` is negative. **All stored `raw_relative` runs predate the 2026-06-09 stop fix** and used the buggy λ-penalized variant (§10 provenance box). Regardless of which stop selected the evidence, unfrozen results must be *read* on raw metrics only — normalized values are meaningless there (§2.3).

### 9.2 What Was Run

| Experiment | CLTs | Prompts | Attention | Status |
|------------|------|---------|-----------|--------|
| Two-hop generalization sweep | gemma2-426k | 8 two-hop | frozen | ✅ 8/8 |
| Capacity / cross-model compare | all 3 | 8 two-hop | frozen | ✅ (gpt-oss failed) |
| Game 1 frozen vs unfrozen | all 3 | 8 two-hop | both | ✅ 24/24 |
| Game 2 frozen vs unfrozen | all 3 | 8 two-hop | both | ✅ 24/24 |
| ACDC-benchmark Game 1 | all 3 | 13 IOI+docstring | both | ✅ 39/39 |

This is a single-seed study; multi-seed repeats are not yet present ([§11.1](#111-what-this-case-study-does-not-yet-cover)).

### 9.3 Baselines and Evaluation Protocol

The CLT-variant comparison above varies the *input graph*. The complementary — and, for positioning MACAG, more important — axis varies the *selection method* on a fixed graph: it asks whether Game 1's intervention-based greedy actually beats the cheaper incumbents and how close it gets to the expensive gold. Each baseline ([§1.3](#13-relation-to-prior-work)) supports a specific claim:

| Baseline | What it produces | Claim the comparison supports | Status |
|----------|------------------|-------------------------------|--------|
| **Top-k influence** | the $k$ highest-`influence` graph nodes | MACAG's *search* beats trusting attribution magnitude (higher faithfulness at matched $|E|$) | **implemented + run** (`macag/baselines/influence.py`); nonlinear benchmark (§10.7): faith\@8 0.08, loses 60/60 — IOI/multi-hop pending |
| **EAP / attribution patching** | top-$k$ nodes by first-order patching score | *real* interventions beat the local-linear scores the graph is already built from | **implemented + run** (`macag/baselines/eap.py`, graph-derived signed-path-effect variant seeded at the target/foil logits); nonlinear benchmark (§10.7): faith\@8 1.15, loses 58/60 — IOI/multi-hop pending |
| **ACDC** | a minimal circuit via top-down edge pruning | bottom-up node selection is competitive at far lower granularity/cost | **implemented + run** (ported node version, `macag/baselines/acdc_prune.py`, τ-sweep); ACDC benchmark prompts wired in (`macag/data/acdc_benchmark_prompts.json`); nonlinear benchmark (§10.7) at uncapped k≈193 (not budget-matched); native-granularity comparison still open |
| **Shapley / Banzhaf (gold)** | exact-ish per-feature credit ([§3.6](#36-relation-to-shapley-and-banzhaf-values)) | greedy evidence ≈ top-Shapley features at a fraction of the oracle cost | **implemented + run** (`macag/baselines/shapley_select.py`: MC permutation Shapley with antithetic pairing + MC Banzhaf, both over the MACAG oracle); nonlinear benchmark (§10.7): Game 1 matches gold-level faithfulness at 44.7× lower cost (ranking agreement moderate: prec@k 0.46); the existing `attribution/shapley.py` remains a separate spline-CLT tool on a different $v$, not this baseline; IOI/multi-hop pending |

**Common protocol.** Every method scores the *same* candidate node set under the *same* oracle (`logit_gap`, `freeze_attention` per [§2.3](#23-attention-freezing-and-the-error-floor)), and is compared on three axes: faithfulness at matched evidence size, evidence size at matched faithfulness, and oracle calls. The intended headline ("one killer result") is a single plot: **MACAG reaches Shapley-gold-level faithfulness at ~45× fewer oracle calls, and dominates top-k-influence / EAP / ACDC on faithfulness-per-feature** (ranking agreement with gold is moderate — prec@k 0.46 / Jaccard 0.33 — and is reported as a secondary, metric-independent signal, not the headline).

> **Status note.** All four baselines and the head-to-head harness are now
> implemented: selectors live in `macag/baselines/` (influence, EAP, MC
> Shapley/Banzhaf over the MACAG oracle, ported ACDC, plus the B3.2 brute-force
> optimality-gap tool) and `python -m macag.cli.run_baselines` runs every
> selector on the same candidate set and oracle, emitting per-k evidence/scores,
> per-method oracle-call counts, precision@k/Jaccard vs Shapley-gold, the
> faithfulness-vs-size AUC, and the Spearman(EAP, greedy-marginal) linearity
> diagnostic. The repo's `attribution/shapley.py` is still a spline-CLT
> attribution tool, **not** the MACAG Shapley-gold
> ([§3.6](#36-relation-to-shapley-and-banzhaf-values)). The harness has unit
> coverage on toy oracles (`tests/test_macag_baselines.py`) **and has now been
> run on real CLT graphs** — the 60-prompt nonlinear benchmark
> (`results/macag_nonlinear_connected/`, reported in §10.7/C.7 with bootstrap CIs
> and paired Wilcoxon tests). The harness is now also **wired into the sweep
> drivers**: `run_macag_acdc.sh` / `run_macag_mib.sh` run every selector
> per-prompt (writing `macag_baselines.json` per run directory) and
> `experiments/analyze_macag_baselines.py` aggregates a sweep root into
> `baselines.csv` (faith@own-k, faith-per-feature, AUC, precision@k/Jaccard vs
> gold, oracle calls, plus `kl_faith_*` columns from [§2.5](#25-kl-rescoring-a-selection-independent-faithfulness-metric)) —
> so the IOI/multi-hop head-to-head is now an *execution* task, not a build task.
> The full 3-seed MIB-bench campaign ([§9.5](#95-mib-bench-full-campaign-setup), `results/macag_mib_seed{0,1,2}/`)
> is in progress and supplies the multi-seed Shapley variance that was previously
> open; the ACDC-benchmark re-run through the consolidated driver remains open.

### 9.4 Analysis Protocol (raw outputs → claims)

The reproducible path from a run to a reported number, so results can be regenerated and audited.

1. **Run** the pipeline per prompt (`scripts/run_macag_pipeline.sh`) or batched via the consolidated drivers: `scripts/run_macag_acdc.sh` (ACDC benchmark; the old `run_macag_sweep.sh` / `run_macag_unfrozen*.sh` two-pass scripts are consolidated into it — `FREEZE_MODE=both` is the default) and `scripts/run_macag_mib.sh` (same pipeline over MIB-bench prompts), with per-CLT parallel launchers `run_macag_{acdc,mib}_parallel.sh` (shared logic in `macag_parallel_common.sh`; resume via `--skip-attribute/--skip-game1/--skip-game2`). Each run directory gets `macag_game1.json` (dual-freeze schema when `FREEZE_MODE=both`), `macag_game2_{abr,fp}.json` (+ canonical `macag_game2.json`), `macag_baselines.json` (per-prompt head-to-head, unless `SKIP_BASELINES=1`), `macag_kl_faithfulness.json` ([§2.5](#25-kl-rescoring-a-selection-independent-faithfulness-metric)), and `oracle_kwargs.json`, each with `params`, `evidence`, `scores`, `stats`.
2. **Validity filter.** Drop prompts with `target_preferred = False` (oracle is measuring the wrong direction) from any faithfulness aggregate; keep them only for coverage/robustness counts. (In C.2/C.3 this removes 3 llama rows.)
3. **Pick the right metric per regime** ([§2.3](#23-attention-freezing-and-the-error-floor)): if `recoverable_range > 0` you may report normalized; if `≤ 0` report **raw** sufficiency/necessity/faithfulness only. Report the `kl_faith` column alongside raw faithfulness as the selection-independent cross-check ([§2.5](#25-kl-rescoring-a-selection-independent-faithfulness-metric)); its range is non-negative by construction, so it is also usable in the degenerate-denominator regime.
4. **Aggregate** with the `experiments/analyze_*.py` scripts. For the consolidated sweep roots (`results/macag_acdc`, `results/macag_mib`, `results/macag_nonlinear*`) the drivers auto-run four aggregators against `--root`/`--bench`: `analyze_macag_acdc.py` → `summary.csv` (raw + `kl_faith` faithfulness, per-prompt `verdict`/`range_flip`, per-CLT×task flip rate), `analyze_game2_abr_vs_fp.py` → `abr_vs_fp.csv` (ABR↔FP evidence Jaccard, overlap, utilities), `analyze_macag_baselines.py` → `baselines.csv` (per-method faith@own-k, faith-per-feature, AUC, precision@k/Jaccard vs Shapley-gold, oracle calls), and `analyze_acdc_frozen_vs_unfrozen.py --root` → `frozen_vs_unfrozen.csv` (dual-freeze legs from one JSON; legacy `--frozen-root`/`--unfrozen-root` fallback for the old two-pass results). The older CSVs in `macag/macagresults/` remain the source for §10.1–10.6. Per CLT/task report: mean raw faithfulness, mean $|E^\*|$, mean upstream count, fraction with `range < 0`, mean overlap_rate.
5. **Attention-mediation diagnostic.** Run each prompt with `--freeze-mode both` and aggregate the **range-flip rate** = fraction with `attention_mediation.range_flip == true`, read directly from the dual-run JSONs (no pairing step). This is the headline of §10.4. `experiments/analyze_frozen_vs_unfrozen.py` remains as the legacy path for the pre-existing two-invocation (non-parameter-matched) sweep results.
6. **Uncertainty** (Phase 1): bootstrap over prompts (resample with replacement, 10k draws) → 95% CI for every aggregate; for the flip rate, CI on the proportion.
7. **Baselines** (Phase 2): run each selector through the *same* oracle on the *same* $C$; compare at **matched $|E|$** and via faithfulness-vs-size AUC.
8. **Report** each number against [§10.6](#106-claims-to-evidence-to-research-question-mapping) so every claim has a traceable source.

### 9.5 MIB-Bench Full Campaign: Setup

This is the paper's primary quantitative evaluation — the successor to the 8-prompt
two-hop sweep and the 13-prompt ACDC benchmark of [§9.1](#91-setup), scaled to
community-standard prompts and a 3-seed design. It grounds the prompt sets and the
baseline family in the **MIB circuit-localization benchmark**
(`mib-bench/*` on HuggingFace, MIB circuit-track code vendored at
`external/MIB-circuit-track`, one-time setup `external/setup_mib.sh`). What follows is
the full protocol; results are reported in §10 once the campaign completes.

**9.5.1 Prompts.** `experiments/build_mib_benchmark_prompts.py` exports MIB-bench
HuggingFace prompts into `macag/data/mib_benchmark_prompts.json` — the same per-prompt
schema as the hand-written ACDC manifest (clean/corrupted prompt, single-token
target/foil) plus `mib_model` / `mib_task` / `mib_split` routing metadata, with prompts
tokenizer-aligned to their MIB model. gemma-2-2b is the only MIB model with public CLTs
available, so every prompt routes to the two gemma CLTs (`gemma2-426k`, `gemma2-2.5M`);
`llama32-524k` carries no MIB prompts and is absent from this campaign. Benchmark-size
export command:

```bash
python experiments/build_mib_benchmark_prompts.py \
  --models gemma2 --tasks ioi mcqa arc_easy --split validation --task-limit ioi=500
```

Task sizes (validation split): `mcqa` (copycolors) and `arc_easy` are exported **in
full** — 50 and 570 prompts respectively, their entire validation splits. `ioi` is
**capped at 500** of its ~10,000 validation prompts (`--task-limit ioi=500`) — running
the full IOI split at the campaign's per-prompt cost was estimated at roughly 2
GPU-years and judged not worth it relative to the marginal statistical value past 500
prompts; `run_mib_benchmark.sh` refuses to launch against a prompt file with
$\le 10$ IOI prompts (`ALLOW_SMALL_JSON=1` overrides this refusal for a smoke run) so a
pilot-sized manifest cannot be mistaken for the campaign manifest. Total: 1120
(CLT, prompt) cells per seed (2 CLTs $\times$ 560 prompts).

**9.5.2 Per-prompt game parameters.** Every cell runs the same consolidated pipeline as
the case study ([§9.4](#94-analysis-protocol-raw-outputs--claims), `scripts/run_macag_pipeline.sh`), with the
pipeline's defaults used unchanged — these are the same values as [§9.1](#91-setup)
except the freeze protocol, which is *always* dual here:

| Parameter | Value |
|---|---|
| $\alpha$ | 0.5 |
| $\lambda$ | 0.02 |
| $\beta$ | 0.2 |
| Game 1/2 budget | 8 |
| prefilter top-$k$ | 20 |
| Game 2 solvers | `abr` **and** `fp` (both run; [§5.2](#52-algorithm-simultaneous-best-response-the-abr-solver)/[§5.2.1](#521-fictitious-play-solver-solverfp)), $K{=}4$ rounds, `fp_tol`$=10^{-3}$ |
| Game 1 `freeze_mode` | `both` (matched dual-freeze legs, [§2.3](#23-attention-freezing-and-the-error-floor)) |
| Game 1 stop | `raw_relative`, $\varepsilon{=}0.1$ (forced by `freeze_mode=both`) |
| ablation | zero (`ABLATION_MODE=zero`) |
| score kind | `logit_gap` (no MIB task here has a first-token target/foil collision, so `answer_span` is not needed) |
| candidate universe | full graph feature-node set, `--connected` (the CLI default) |

Each cell also gets a post-hoc KL rescore ([§2.5](#25-kl-rescoring-a-selection-independent-faithfulness-metric)) of every stored evidence set,
run automatically at the end of `run_macag_pipeline.sh`.

**9.5.3 Baseline harness: fast pass + deferred gold pass.** Every cell also runs the
[§9.3](#93-baselines-and-evaluation-protocol) head-to-head harness on the identical graph and oracle. Because MC
Shapley/Banzhaf is $\approx$90% of a cell's baseline cost ($\approx$34k oracle calls vs.
Game 1's $\approx$760), the campaign splits it into two passes:

- **Fast pass** (`BASELINE_METHODS=influence,eap,game1,acdc`, run inline by
  `run_macag_mib.sh`): top-$k$ influence, graph-derived EAP, a second budget-matched
  Game 1 run (`game1_connected=True` matching the standalone CLI), and **budget-matched
  ACDC** — `acdc_target_size` bisects the prune threshold $\tau$ to the target evidence
  size (`ACDC_TARGET_K=-1` $\Rightarrow$ match `--budget`$=8$; up to 24 bisection
  iterations against the oracle's memoization cache; `exact=false` is reported honestly
  when 1/16-quantized bf16 scores make the exact budget unreachable, with
  `achieved_k` recorded instead of silently substituting a nearby size).
- **Gold pass** (`scripts/run_macag_shapley_pass.sh`, run after the fast pass
  completes): MC-Shapley and MC-Banzhaf ([§3.6](#36-relation-to-shapley-and-banzhaf-values)) over the same oracle, 64
  permutations, antithetic pairing, seeded by the campaign seed
  (`SHAPLEY_SEED=$SEED`) — restricted to the first `GOLD_PER_TASK=50` prompts per
  task (matched by slug index, `SLUG_REGEX='_(ioi|mcqa|arc_easy)_00[0-4][0-9]$'`) to
  bound the gold-baseline cost, then merged back into `macag_baselines.json` via
  `python -m macag.cli.merge_baselines` (pure JSON recombination — no extra oracle
  calls) and re-KL-rescored.

**9.5.4 Three-seed design and orchestration.** The whole protocol (fast pass, gold
pass, post-processing) is one command, run once per seed:

```bash
nohup scripts/run_mib_benchmark.sh > results/mib_campaign.log 2>&1 &
```

`SEEDS="0 1 2"` (default) controls the only stochastic stage of an otherwise
deterministic pipeline — the MC-Shapley/Banzhaf sampling seed — so the three
replicates give the cross-seed ranking-stability check called for in roadmap **B1.3**
([§3.6](#36-relation-to-shapley-and-banzhaf-values)); every other stage (graph construction, greedy Game 1/2, ACDC
bisection) is seed-independent, hence identical bar bf16 run-to-run jitter across
processes. Each seed writes to its own root `results/macag_mib_seed<SEED>/`; after all
seeds finish, `scripts/macag_combine_seeds.py` concatenates the per-seed analyzer CSVs
with a `seed` column and additionally emits per-cell cross-seed mean/std/$n$ into
`results/macag_mib_seeds/`. Every stage is resumable: `run_mib_benchmark.sh` and the
per-prompt pipeline both skip already-completed work, so a killed or crashed run
restarts from where it left off with `SEEDS=<remaining>`.

**9.5.5 Compute.** All work runs on a single NVIDIA GH200 (96 GB HBM3), shared by a
pool of parallel workers *within* one CLT group at a time — `gemma2-426k` and
`gemma2-2.5M` never run concurrently (`macag_parallel_common.sh`), only workers of the
*same* CLT do, at `WORKERS_GEMMA2_426K=8` / `WORKERS_GEMMA2_2_5M=3` (`llama32-524k`'s 4
workers are unused here — no MIB prompts route to it), staggered 45s apart at launch to
avoid a simultaneous cold-start spike, with `PYTORCH_ALLOC_CONF=expandable_segments:True`
to reduce allocator fragmentation. Measured throughput on the 426k pool at 8 workers is
$\approx$7.5 cells/hour; at that rate the full 1120-cell fast pass for one seed is on
the order of a week of wall-clock time (426k finishes before 2.5M starts, since CLT
groups are sequential). The dominant failure mode is **collateral CUDA OOM**: one
worker's attribution backward pass can spike to 16–26 GiB against the other workers'
steady $\approx$10 GiB each, exhausting the 96 GB budget and killing neighbors on small
allocations — an $\approx$10% per-pass cell-failure rate at 8 concurrent workers despite
`expandable_segments`, fully mitigated by the resumable re-run rather than by reducing
worker count (which would cut throughput roughly proportionally).

**9.5.6 Post-processing (per seed, after the gold pass).** `run_mib_benchmark.sh` runs,
in order: `experiments/analyze_macag_baselines.py` (refresh `baselines.csv` with the
merged gold columns), `scripts/macag_bootstrap_wilcoxon.py` (bootstrap CIs over prompts
+ paired Wilcoxon signed-rank tests with Holm correction, Game 1 vs. each baseline,
[§B1.2](#appendix-b-roadmap-to-a-submission-ready-evaluation)), `experiments/plot_faithfulness_curves.py`
(per-method faithfulness-vs-size curves with CI bands, [§B3.1](#appendix-b-roadmap-to-a-submission-ready-evaluation)), and
`experiments/analyze_gold_circuits.py --task ioi --include-baselines` (the
(layer, token-role) IOI gold-circuit scorer of [§9.6](#96-interpbench-native-component-gold-circuit-validation-setup)'s sibling diagnostic, **B4.1** —
precision/component-recall of every selector's evidence against the published IOI
circuit's layer bands and token roles). The same four aggregate CSVs as the ACDC sweep
([§9.4](#94-analysis-protocol-raw-outputs--claims) step 4: `summary.csv`, `abr_vs_fp.csv`, `baselines.csv`,
`frozen_vs_unfrozen.csv`) are produced automatically as part of the fast-pass driver
(`run_macag_mib.sh`'s built-in analysis stage) before the gold pass runs.

**9.5.7 External reference frame (historical, not part of the running campaign).**
Separately from the campaign above, `docs/mib-reproduction.md` documents a one-off
calibration exercise that reproduced the MIB circuit-track's own methods (EAP,
EAP-IG variants, clean-corrupted, exact patching, information-flow-routes) against
the MIB paper's published CPR/CMD numbers, to sanity-check that the [§9.3](#93-baselines-and-evaluation-protocol) EAP
baseline (a cheaper, graph-derived variant) is calibrated against the real
implementation. A gpt2/IOI smoke run is preserved at `results/mib_reproduction/`; the
driver script itself has since been pruned from `scripts/` (the repository's script
tree was narrowed to the two active campaign chains — this one and InterpBench,
[§9.6](#96-interpbench-native-component-gold-circuit-validation-setup)), so this is context for the EAP baseline's provenance, not a
reproducible step of the current protocol. *Caveat if resurrected:* MIB leaderboard
methods select **edges over native components** and report CPR/CMD, while MACAG
selects **CLT feature nodes** (or, in [§9.6](#96-interpbench-native-component-gold-circuit-validation-setup), native components) and reports
faith@k / precision-recall — the shared object is the task/prompt distribution, not
the metric, so cross-benchmark numbers are context, not a head-to-head.

### 9.6 InterpBench Native-Component Gold-Circuit Validation: Setup

Every diagnostic in [§9.5](#95-mib-bench-full-campaign-setup) validates MACAG's evidence against a
*heuristic* layer/token-role encoding of the published IOI circuit (roadmap **B4.1**,
[Appendix B](#appendix-b-roadmap-to-a-submission-ready-evaluation)), because CLT features have no exact
ground-truth membership. InterpBench closes that gap by giving MACAG a model whose
circuit is known **exactly**, at native-component granularity, so both precision *and*
recall are well-defined (roadmap **B4.2**).

**Model and candidate universe.** `mib-bench/interpbench` (hub-hosted) is a 6-layer,
4-head transformer trained to implement IOI, whose ground-truth circuit — `m0`, `a1.h1`,
`a2.h1`, `a4.h1` (MLP-0 and three attention heads, MIB naming) — is published exactly
via `interpbench_graph.json`'s `in_graph` flags. It is loaded as a `HookedTransformer`
with `use_hook_mlp_in=True`, `use_attn_result=True`, `use_split_qkv_input=True` (required
for the per-head ablation hooks; the evaluation config differs from the training config
that produced the checkpoint). Because there are no CLT features to select over, the
oracle backend swaps the `ReplacementModel` feature scorer for
`macag.scoring_components.HookedComponentInterventionScorer`, which ablates **native
components** — attention heads `a{layer}.h{head}` and MLPs `m{layer}` — via forward
hooks, implementing the identical four-mode oracle contract ([§2.1](#21-oracle-scoring)) the CLT games
use. The candidate universe is every component in the model
(`component_universe`: $6 \times 4 = 24$ heads $+ 6$ MLPs $= 30$ nodes), so Game 1, Game
2, and the MC-Shapley/Banzhaf estimator all run **completely unmodified** against this
backend — the only thing that changed is what an "intervention" ablates.

**Prompts.** MIB's IOI validation split, capped at `--limit 500` (`LIMIT=500`,
matching the [§9.5](#95-mib-bench-full-campaign-setup) IOI cap so both benchmarks probe the same prompt budget).

**Protocol** (`experiments/run_interpbench_macag.py`, one call per seed):
1. Build the component-level graph/candidate set for the prompt.
2. Run `estimate_shapley` (64 permutations, antithetic) over the component oracle,
   producing a per-component credit ranking; score it against the gold `in_graph`
   flags with node-level **AUROC** and **average precision** (`macag.eval.gold_circuits`)
   — the InterpBench-native counterpart of the MIB circuit track's own (edge-level)
   AUROC metric, computed here at node level.
3. Run Game 1 with `--budget 4` (matching $|{\text{gold circuit}}|=4$, so the
   comparison is against a same-size draw rather than an arbitrary cutoff); score the
   returned evidence against the gold node set with set-level **precision / recall /
   F1** — recall is well-defined here (unlike the CLT-feature case) because gold and
   evidence share the same component ontology.

**Three-seed design and orchestration**, mirroring [§9.5.4](#95-mib-bench-full-campaign-setup):

```bash
nohup scripts/run_interpbench_benchmark.sh > results/interpbench_campaign.log 2>&1 &
```

`SEEDS="0 1 2"` again seeds only the MC-Shapley/Banzhaf sampling (and the bootstrap
resampling in the aggregate); each seed writes `results/interpbench_macag_seed<SEED>/interpbench_macag.csv`
plus a bootstrap-CI aggregate, and a seed whose output CSV already exists is skipped on
re-run (crash-safe, same resumability contract as [§9.5](#95-mib-bench-full-campaign-setup)). Knobs: `SHAPLEY_PERMUTATIONS=64`,
`BUDGET=4`, `DEVICE=cuda` — all defaulted to the campaign configuration in the script
header, overridable via environment variables for debugging runs.

**Why this is a distinct diagnostic from §9.5's gold-circuit check.** The MIB campaign's
`analyze_gold_circuits.py` pass asks whether MACAG's CLT-feature evidence reads from the
*known layer bands and token roles* of the IOI circuit on the *real* gemma CLTs — a
necessarily heuristic comparison, because "feature 41823 of a 426k-feature CLT" has no
canonical mapping to "S-inhibition head." InterpBench asks the sharper question — exact
node-for-node recovery — but only on a small synthetic model built to *have* an exact
answer, not on the CLTs the rest of the paper studies. The two results are complementary,
not substitutable: agreement between them is the strongest gold-circuit evidence
available; disagreement localizes whether a shortfall is a CLT-representation problem
(features don't align with the named components) or a Game-1-selection problem (the
right evidence exists but greedy doesn't find it), since InterpBench removes the CLT
from the loop entirely.

---

## 10. Results

> **How to read these.** All numbers are **raw** scores (logit-gap units),
> single seed, computed from the analysis CSVs in `macag/macagresults/`. Per
> [§2.3](#23-attention-freezing-and-the-error-floor), normalized faithfulness is
> unreliable wherever `recoverable_range` is small or negative — which is most of
> the ACDC benchmark — so the headline metrics here are raw sufficiency,
> raw faithfulness, evidence size, upstream-feature count, and the sign of
> `recoverable_range`. **Never read normalized metrics on unfrozen runs** — the
> denominator is unreliable there and the values are meaningless (§2.3).

> **⚠ Result provenance (pass 1, 2026-06-10).** Every number in §10 and Appendix C
> comes from runs executed **2026-06-03/05** (`macag/macagresults/`, code state
> ≤ `d5000d7`), predating the **2026-06-09** fixes in `71a2ef6`:
> - the `raw_relative` stop then compared **λ-penalized utility gains**, not the
>   λ-free faithfulness gains described in §4.2 — all stored `raw_relative` runs
>   (`macag_unfrozen_raw/`, `macag_acdc_unfrozen/`) used the buggy stop, so their
>   evidence sizes and faithfulness-at-stop will change on re-run;
> - Game 2 **best-iterate tracking / `best_iteration` did not exist** — the 48
>   stored Game 2 runs return the final iterate (low risk: all converged in 2
>   rounds);
> - oracle-call counters were **not reset per solve** — treat the C.6 cost numbers
>   (~808 / ~1761) as provisional.
>
> **Robust to all of the above:** `recoverable_range` signs (the §10.4 flip
> diagnostic) and Game 2 `overlap_rate`, which depend on neither the stop rule nor
> the returned iterate. **Also remember** the frozen/unfrozen parameter mismatch:
> frozen legs ran budget 8 / prefilter 20, unfrozen legs 20 / 30 (§2.3, §9.1,
> §11.3). Regenerate every table below from a post-`71a2ef6` re-run before quoting
> numbers in the paper. **§10.7 / Appendix C.7 are the exception — that
nonlinear-benchmark run is already post-`71a2ef6` and may be quoted as-is.**
<!-- TODO(pass-2): refresh §10.1–10.6 + Appendix C.1–C.6 numbers (§10.7/C.7 already post-fix) -->

### 10.1 The two-hop factual circuit (gemma2-426k, frozen)

All 8 city→capital prompts are target-preferred. MACAG recovers a small, high-faithfulness evidence set with the normalized stop, and the *contrastive* structure is perfectly clean — every prompt has **overlap_rate = 0.0** (target and foil evidence are disjoint). One prompt (philadelphia-harrisburg) has a **negative `recoverable_range`** (−2.6): a reconstruction failure where ablating all features does not collapse the behavior, so its normalized faithfulness is meaningless (−2.5) and must be read raw.

| Quantity | Value (mean over 8 prompts) |
|----------|------------------------------|
| target-preferred | 8 / 8 |
| `recoverable_range` < 0 (reconstruction failures) | 1 / 8 (philadelphia) |
| evidence size $|E^*|$ | 5.25 (range 1–8) |
| Game 2 overlap_rate | 0.0 (all 8) |
| oracle calls — Game 1 / Game 2 | ~808 / ~1761 |

### 10.2 Capacity and cross-model comparison (frozen)

Same 8 prompts, three CLTs (gpt-oss-20b failed to load):

| CLT | target-preferred | recon. failures (range < 0) | mean faith_norm (valid) | mean sparsity | mean overlap |
|-----|:---:|:---:|:---:|:---:|:---:|
| gemma2-426k | 8/8 | 1/8 | 0.99 | 0.984 | 0.0 |
| gemma2-2.5M | 8/8 | 1/8 | 1.10 | 0.983 | 0.0 |
| llama32-524k | 5/8 | 2/8 | 0.97 | 0.978 | 0.0 |

*Counting conventions (from `analyze_clt_comparison.py`; stated here because the
raw C.2 table otherwise looks inconsistent with this one):* "recon. failures"
counts negative `recoverable_range` **only among target-preferred prompts** —
llama has 5/8 negative ranges in C.2, but three of those are on prompts the model
does not even perform, which makes them invalid rows, not transcoder failures.
"mean faith_norm (valid)" averages over prompts that are target-preferred **and**
have positive range.

**Capacity did not fix reconstruction failures.** Increasing the gemma CLT ~6× (426k → 2.5M) left exactly one prompt with negative `recoverable_range` in both — the behavior that lives outside the features is a property of the *task/attention*, not of transcoder width. The cross-model llama CLT is weaker on this template (only 5/8 prompts target-preferred, 2 reconstruction failures). Contrastive separation (overlap 0.0) holds across all three.

### 10.3 Frozen vs unfrozen Game 1: attention hides upstream features

Reusing each frozen graph with `freeze_attention=False` recruits **more upstream features** (reverse-position > 0, i.e. not at the prediction token) into the minimal set — the mechanism predicted in [§2.3](#23-attention-freezing-and-the-error-floor):

| CLT | upstream (frozen) | upstream (unfrozen) | $|E^*|$ frozen | $|E^*|$ unfrozen |
|-----|:---:|:---:|:---:|:---:|
| gemma2-426k | 1.25 | **2.50** | 5.5 | 5.1 |
| gemma2-2.5M | 1.25 | **3.38** | 5.25 | 9.75 |
| llama32-524k | 3.62 | 3.62 | 6.25 | 6.25 |

The two gemma CLTs roughly double their upstream-feature count when attention is unfrozen; llama's **means** are unchanged on the two-hop set (upstream 3.62 → 3.62, $|E^*|$ 6.25 → 6.25) — an averaging coincidence, not per-prompt stability (dallas upstream 3 → 0, portland 3 → 8; see C.3). The effect is real but **noisy at single seed** — individual prompts can destabilize under unfrozen attention (e.g. gemma2-2.5M portland-salem, whose *normalized* faithfulness blows up to −15.8 because `range_u` collapses to −0.375; the raw faithfulness is a sane 5.94 — exactly the §2.3 rule that normalized values must never be read on unfrozen runs), which is why these need multiple seeds and why the `raw_relative` stop exists.

*Caveats for this table:* the unfrozen legs ran at budget 20 / prefilter 30 vs the frozen legs' 8 / 20, so the evidence-size growth (2.5M's 9.75 exceeds the frozen cap of 8) and part of the upstream recruitment is mechanically enabled by the lifted cap (§2.3); and these unfrozen runs selected evidence with the `normalized` stop, which §2.3 recommends against unfrozen — the `raw_relative` rerun in `macag_unfrozen_raw/` gives different unfrozen numbers (e.g. miami $E_u$ 15 vs 13). <!-- TODO(pass-2): re-run with matched budget + fixed raw_relative stop and regenerate this table -->

> **Task-dependence (matched-protocol correction, [§10.7](#107-nonlinear-benchmark-baseline-head-to-head-and-matched-frozenunfrozen-pass-2)/C.7).** "Unfreezing recruits upstream features" is **specific to the two-hop relational circuit**. On the matched-budget nonlinear benchmark (no lifted-cap confound), unfreezing instead *shrinks* the set — upstream 3.45 → 2.87, $|E|$ 5.37 → 4.38 — the same direction as the ACDC/IOI sweep (§10.4). So the headline mechanism here is **"frozen attention *adds spurious necessity* that unfreezing removes,"** and recruitment vs shrinkage is itself a per-task diagnostic, not a universal law. Treat the gemma "doubling" above as a property of multi-hop tasks until replicated under matched budgets.

### 10.4 ACDC benchmark: the attention-mediation result

This is the strongest finding. On IOI + docstring (13 prompts/CLT), frozen attention makes the behavior look *unrecoverable from features* — `recoverable_range` is negative almost everywhere — and unfreezing flips it positive:
<!-- TODO(pass-2): the unfrozen faith/E/upstream columns below came from the buggy (λ-penalized) raw_relative stop; the range columns and flip counts are stop-independent and robust. -->

| CLT | range < 0, **frozen** | range < 0, **unfrozen** | upstream frozen → unfrozen | $|E^*|$ frozen → unfrozen |
|-----|:---:|:---:|:---:|:---:|
| gemma2-426k | **12 / 13** | 1 / 13 | 5.5 → 2.3 | 6.6 → 3.3 |
| gemma2-2.5M | **13 / 13** | 1 / 13 | 6.0 → 3.7 | 7.2 → 5.1 |
| llama32-524k | 0 / 13 | 0 / 13 | 6.4 → 3.4 | 7.6 → 5.0 |

For gemma, the ACDC-benchmark tasks (especially IOI) are **attention-mediated**: with attention frozen, ablating all features barely moves the logit gap (range strongly negative, e.g. IOI ranges of −8 to −24), so feature-level faithfulness is ill-defined. Across gemma's 26 ACDC-benchmark cases (IOI + docstring × two CLTs), `recoverable_range` is **negative in 25/26 when frozen and non-negative in 24/26 when unfrozen**, with **23/26 strictly flipping sign** (negative→non-negative); unfreezing attention recruits the features that drive the behavior. Llama's CLT already places the behavior in features (range positive throughout, 0/13 negative), a genuine cross-model difference MACAG surfaces automatically. This is the empirical payoff of the frozen/unfrozen + raw-metric machinery in [§2.3](#23-attention-freezing-and-the-error-floor).

### 10.5 Contrastive separation is robust (Game 2)

<!-- TODO(pass-2): re-verify 48/48 after the re-run; the result is stop-rule-independent and expected to hold. -->
Across **all 24** frozen Game 2 runs *and* all 24 unfrozen runs, **overlap_rate = 0.0**: target and foil evidence sets are completely disjoint, and this survives unfreezing attention. The two-hop target (capital) and foil (state) are carried by strictly different features — a clean, reproducible structural result, and the one headline number that is stable at single seed because it is denominator-free and does not depend on the attention mode.

### 10.6 Claims to Evidence to Research-Question Mapping

How each result supports the contributions and RQs in [§1.4](#14-problem-statement-research-questions-and-contributions). "Status" marks whether the current data settles the claim or whether a roadmap phase is still needed.

| Claim | RQ / Contribution | Evidence (this doc) | Status |
|-------|-------------------|---------------------|--------|
| Minimal node sets are causally sufficient + necessary | RQ1 / C3 | §10.1, C.1 (faith_norm ≈ 0.83–1.32 on feature-mediated prompts) | partial on these graphs — search > influence is shown on the nonlinear benchmark (§10.7); the Phase-2 sweep on the two-hop/IOI graphs is pending |
| Real interventions beat attribution magnitude / EAP | RQ1 / C1,C3 | §10.7, C.7 (Game 1 faith 5.50 vs EAP 1.15, influence 0.08, at matched budget 8) | **shown** (nonlinear benchmark, 60 prompts, single-seed) |
| Target/foil carried by distinct features | RQ2 / C4 | §10.5, C.5 (overlap 0.0 in 48/48) | strong (single-seed but denominator-free) |
| Contrastive separation is not a scoring artifact | RQ2 / C4 | §10.5 (holds frozen *and* unfrozen) | supported; widen task families |
| MACAG diagnoses attention- vs feature-mediation | RQ3 / C5 | §10.4, C.4 (gemma 23/26 sign-flip, 25/26 neg-frozen→24/26 non-neg-unfrozen; llama 0/13 neg) | strong direction; needs CIs (Phase 5) + positive control |
| Diagnosis is model/task dependent | RQ3 / C5 | §10.4 (gemma vs llama; IOI vs docstring) | supported, small n |
| Capacity ≠ more faithful circuit | RQ4 / C5 | §10.2, C.2 (failure moves 426k→2.5M) | suggestive; needs more CLTs |
| Game 2 equilibrium exists & solver stabilizes | C4 | §3.4 (exact-potential existence proof; greedy-Jacobi dynamics not guaranteed, mitigated by best-iterate + FP; empirically `converged=True`) | existence theory done; convergence empirical, single-config |
| Greedy ≈ gold-level faithfulness at far lower cost | C2 / RQ1 | §10.7, C.7 (Game 1 faith 5.50 ≥ Shapley-gold 4.72 at **45× fewer** oracle calls: 718 vs 32 495; ranking agreement moderate — prec@k 0.46 / Jaccard 0.33 vs gold) | **shown** (nonlinear benchmark, single-seed) |

The two formerly **missing** rows are now populated by the matched-protocol
nonlinear-benchmark run ([§10.7](#107-nonlinear-benchmark-baseline-head-to-head-and-matched-frozenunfrozen-pass-2), [Appendix C.7](#c7-nonlinear-benchmark-matched-protocol-baselines--frozenunfrozen)):
the Phase-2 baseline head-to-head exists at single seed, so RQ1 and the "search is
worth it" half of C1 are now *shown* (pending multi-seed CIs from Phase 1/5).

### 10.7 Nonlinear-benchmark baseline head-to-head and matched frozen/unfrozen (pass-2)

A second case study runs the **matched-protocol** `run_macag game1 --freeze-mode
both` (one model load, both legs at budget 8 / prefilter 20 / `raw_relative` /
`connected=True`) plus the full Phase-2 baseline harness over a new **nonlinear
benchmark**: 4 task families (`boolean_logic`, `negation_polarity`,
`context_polysemy`, `hard_semantic_foil`) × 5 prompts × 3 CLTs (gemma2-426k,
gemma2-2.5M, llama32-524k) = **60 prompts**. These are single-step completion prompts
chosen so the target/foil contrast is a nonlinear function of context (e.g. *"We sat
on the grassy bank of the"* → `river` vs `money`), complementing the multi-hop
two-hop/IOI prompts of §10.1–10.5. Full per-prompt rows: [Appendix C.7](#c7-nonlinear-benchmark-matched-protocol-baselines--frozenunfrozen);
source CSVs `results/macag_nonlinear_connected/{baselines,summary,frozen_vs_unfrozen,abr_vs_fp}.csv`.
This run is **post-`71a2ef6`** (fixed `raw_relative` stop, per-solve oracle-stat
reset, Game-2 best-iterate) and is the first to carry real baseline numbers — the
Phase-2 head-to-head was previously "implemented, not run."

**(1) Game 1 owns the efficiency frontier (the C2/RQ1 result).** At matched budget
$k\!=\!8$, means with **95% bootstrap CIs over prompts** (10 000 resamples) and
**paired Wilcoxon** tests of Game 1 vs each budget-matched baseline (Holm-corrected,
matched-pairs rank-biserial effect $r$). The resampling unit is the *prompt*, not a
random seed: Game 1 / influence / EAP / ACDC are deterministic given a fixed graph +
oracle, so prompt variation is the only meaningful source of uncertainty
(reproduce: `scripts/macag_bootstrap_wilcoxon.py` <!-- TODO(pass-2): this script is not currently checked into scripts/ — restore it from the run environment before submission -->).

| selector | faith@8 [95% CI] | faith / feature [95% CI] | oracle calls | vs G1: median Δ | win/60 | $p$ (Holm) | $r$ |
|---|--:|--:|--:|--:|--:|--:|--:|
| top-k influence | 0.08 [−0.02, 0.20] | 0.010 [−0.003, 0.025] | 0 | +4.95 | 60/60 | <1e‑9 | 1.00 |
| EAP (graph-derived) | 1.15 [0.55, 1.83] | 0.144 [0.065, 0.226] | 0 | +3.72 | 58/60 | <1e‑9 | 0.98 |
| MC Shapley (gold) | 4.72 [4.02, 5.45] | 0.590 [0.500, 0.683] | 32 495 | +1.00 | 44/60 | 8.2e‑5 | 0.60 |
| **MACAG Game 1** | **5.50 [4.88, 6.15]** | **0.859 [0.747, 0.977]** | **718** | — | — | — | — |
| ACDC (τ-prune, k≈193) | 10.50 [8.60, 12.46] | 0.067 [0.050, 0.088] | 2 612 | *(not matched)* | — | — | — |

(faith/feature is per-prompt faith ÷ selected-set size — the budget-8 selectors
divide by 8, Game 1 by its own $|E|$ (mean 6.8). All numbers in this table are the
**all-60** aggregates; the target-preferred-only (n=50) variants are quoted where
marked. AUC, prec\@k 0.46 and Jaccard 0.33 vs gold are in C.7 — the gold-agreement
columns are defined only on the 34/60 prompts where Game 1 filled the full k=8
budget, since a shorter evidence set has no k-matched gold prefix.) Reading the table
in the order the evidence is strongest:

- **Cost is the unimpeachable result.** Game 1 uses **44.7× fewer** oracle calls than
  Shapley-gold (95% CI [43.5, 45.8]), **cheaper on 60/60 prompts**, one-sided Wilcoxon
  $p<10^{-9}$. This is the cost-Pareto claim and it is not contestable.
- **Faith-per-feature separates cleanly.** Game 1 **0.859 [0.747, 0.977]** vs Shapley
  0.590 [0.500, 0.683] — the **CIs do not overlap** (target-preferred-only: 0.938
  [0.820, 1.068] vs 0.636). Unlike raw faith, fpf cannot be inflated by
  spending more features, which is exactly how ACDC posts a high raw number; its fpf
  collapses to 0.067 (worse than everything but influence) at k≈193.
- **Raw faith\@8: Game 1 wins on a clear majority, not uniformly.** Median paired
  advantage over Shapley is +1.00 ($p=8.2\times10^{-5}$, $r=0.60$), but the **win-rate
  is 44/60** — Shapley wins ~16 prompts. The honest sentence is "significantly higher
  faithfulness on most prompts," and this comparison is on *Game 1's own greedy
  objective*, so we treat it as supporting, not headline (see threats, §11.3).
- **Influence ≈ noise, EAP weak.** Both lose on 58–60/60 prompts at $r\ge0.98$ — the
  §A.3 "is search needed?" floor answers *yes, decisively*.

ACDC is **excluded from the significance tests**: its uncapped k≈193 is not
budget-matched to the k=8 selectors, so its higher raw faith (10.5, "wins" 46/60) is a
ceiling at ~28× the feature budget, not a competitor. Restricting to the 50
target-preferred prompts (§(2): the logit-gap oracle is ill-posed on the other 10)
*strengthens* every effect — Game 1 faith 5.96, fpf 0.938, vs-Shapley median Δ +1.08
($p=9.6\times10^{-5}$), cost 44.8× — so the benchmark-hygiene filter helps rather than
hides the result.

**(2) Attention-mediation is the exception, not the rule, on this benchmark.** Of 60
prompts the per-prompt verdict is **42 feature_mediated, 12 indeterminate, 6
attention_mediated**, and only **6/60 (10%) range-flip** (frozen `range` < 0 →
unfrozen ≥ 0). The flips are concentrated almost entirely in **gemma `hard_semantic_foil`**
(sem_03/04/05 on both gemma CLTs) plus one llama `boolean_logic` case; `context_polysemy`
is **15/15 feature_mediated**. This sharply contrasts the §10.4 ACDC/IOI result where
gemma flipped 23/26 — i.e. these single-step "nonlinear" completions are
**predominantly feature-carried**, and attention-mediation is a property of
*multi-hop relational* tasks (IOI), not of nonlinearity per se. `boolean_logic` is the
weak family (7/15 indeterminate, and 9/60 of the not-target-preferred prompts live
here — the logit-gap oracle is measuring the wrong thing when the model does not even
prefer the target).

**(3) Unfreezing *shrinks* evidence here — opposite of the two-hop sweep.** Mean
upstream-feature count goes **3.45 → 2.87** and evidence size **5.37 → 4.38** when
attention is unfrozen (consistent across all three CLTs). This matches the §10.4
ACDC/IOI direction (unfreeze → fewer features once attention recomputes) and is the
**opposite** of the §10.3/C.3 two-hop sweep (unfreeze → *recruits* upstream features).
So the "frozen attention hides upstream features" claim is **task-dependent**: it is a
property of the two-hop relational circuit, not a universal — a threat-to-validity
correction to the §10.3 framing ([§11.3](#113-threats-to-validity--reviewer-rebuttals-to-pre-empt)).

**(4) Game 2 ABR and fictitious-play agree; contrastive disjointness replicates at
scale.** All 60 contrastive runs converge in 2 iterations under **both** solvers; the
ABR and FP solutions are close (target-set Jaccard 0.63, foil-set 0.55) with
near-identical target utility (5.50 vs 5.45), validating FP as a drop-in alternative
to ABR. Target/foil evidence overlap is **0.0 in every row** — across the frozen
Game-1 verdicts (60), the Game-2 ABR runs (60), and the FP runs (60), giving
**180/180 disjoint** target/foil sets and extending the §10.5 "overlap 0.0"
separation result (previously 48/48) to a second, larger benchmark.

---

## 11. Discussion of the Case Study

The case study delivers two clear scientific results and one honest caution.

**Result — MACAG localizes *where* a behavior lives.** The frozen/unfrozen contrast turns "is this circuit faithful?" into a diagnosis: a negative `recoverable_range` under frozen attention that becomes non-negative when unfrozen is a fingerprint of an **attention-mediated** behavior (gemma ACDC-benchmark, 23/26 cases flip sign; 25/26 negative frozen → 24/26 non-negative unfrozen), as opposed to a **feature-mediated** one (llama, range positive throughout; the two-hop factual circuit). This is a property of the model/CLT pair that aggregate reconstruction metrics cannot see, and it falls straight out of MACAG's intervention oracle.

**Result — contrastive structure is clean and robust.** Overlap_rate is 0.0 in all 48 Game 2 runs (frozen and unfrozen): competing answers are carried by disjoint feature sets. Because overlap is denominator-free, this is the most trustworthy single-seed claim in the study — and it replicates on the nonlinear benchmark (§10.7(4)), extending the count to **180/180 disjoint** target/foil sets.

**Caution — capacity is not the lever.** A 6× larger gemma CLT did not remove the reconstruction failures, so "bigger transcoder ⇒ more faithful circuit" is not supported by these data.

> Note: an earlier draft reported a Spline-CLT-vs-linear *faithfulness–parsimony
> trade-off* on GPT-2. Those runs are **not** part of this result set (no Spline-CLT
> or GPT-2 graphs are present in `macag/macagresults/`), so that claim has been
> removed pending data; see [§11.1](#111-what-this-case-study-does-not-yet-cover).

### 11.1 What this case study does *not* yet cover

Gaps that matter for a top-tier submission are catalogued in [§12.3](#123-conference-readiness-what-is-present-vs-missing). For the §10.1–10.5 two-hop/IOI case study the local gaps are: single seed (no variance/CIs), no baseline comparison run *on these graphs* (the §9.3 harness is built and run, but only on the nonlinear benchmark), no Spline-CLT or GPT-2, the gpt-oss CLT failed to load, and `greater_than` was excluded. The §10.7/C.7 nonlinear benchmark closes two of these — it ships the **baseline head-to-head with bootstrap CIs + paired tests (n=60)** — leaving, for that benchmark: still standard-CLT-only (not Spline-CLT), faith\@k circular on Game 1's objective ([§11.3](#113-threats-to-validity--reviewer-rebuttals-to-pre-empt); the KL rescoring layer of [§2.5](#25-kl-rescoring-a-selection-independent-faithfulness-metric) is built as the fix but has not yet been run on this root), ACDC not budget-matched, and no gold-circuit recovery.

### 11.2 Framing options for the paper (pick one)

The same results support several different paper framings; they are not mutually exclusive but they imply different titles, baselines, and emphasis. Captured here so the writeup can choose deliberately.

- **A — "MACAG: a game-theoretic evaluator for attribution circuits."** *Thesis:* the framework is the contribution; attention-mediation is the demonstrating result. *Needs:* baselines (Phase 2) to justify the games over influence/EAP; gold-circuit recovery (Phase 4). *Strength:* broad, method-paper framing; *risk:* reviewers ask "why games, not just ablation ranking?" — answer with the Shapley connection + contrastive Game 2 (no prior analog).
- **B — "Attention mediation is a measurable, model-dependent property of CLT circuits."** *Thesis:* the empirical phenomenon leads; MACAG is the instrument. *Needs:* scale IOI/greater-than + CIs (Phases 1, 5), the feature-mediated positive control. *Strength:* a crisp scientific claim with a clean cross-model contrast (gemma vs llama); *risk:* depends on the frozen/unfrozen distinction being accepted as meaningful — pre-empt with §2.3.
- **C — "Contrastive circuit separation."** *Thesis:* Game 2's overlap_rate = 0 across 48 runs shows competing answers are carried by disjoint features; lead with the contrastive game. *Strength:* the single most robust result (denominator-free, attention-invariant, single-seed-stable); *risk:* one number — needs more task families (beyond two-hop) and a baseline notion of "expected overlap."
- **D — "Evaluating transcoder capacity with causal games."** *Thesis:* the capacity/cross-model comparison; lead with "bigger CLT ≠ more faithful circuit." *Strength:* practitioner-relevant; *risk:* the current 3-CLT, 8-prompt evidence is thin — would need many more public CLTs.

*Recommendation in these notes:* **A as the frame, B as the headline result**, C as the robustness highlight, with the capacity finding (D) as a secondary section. This ordering matches where the evidence is strongest (B, C) and where the conceptual novelty is (A: games + Shapley + contrastive).

### 11.3 Threats to validity / reviewer rebuttals to pre-empt

- **"Zero-ablation is not a clean intervention."** Zeroing a feature is off-manifold;
  resample/mean ablation (as in ACDC) may behave differently. *Mitigation (build
  done 2026-07-02):* the factory now computes per-node alternative ablation values —
  `ablation_mode="mean"` (per-feature mean over clean-prompt positions) or
  `"corrupted"` (patch-style value from the manifest's `corrupted_prompt` at the
  same position, the ACDC convention; length-mismatch falls back to mean) — via
  `compute_mean_ablation_values`, rewriting the intervention specs to 4-tuples
  ([§7.1](#71-replacementmodel-scorer)). Drivers opt in with
  `ABLATION_MODE=corrupted` (`--ablation-mode/--corrupted-prompt` on the
  pipeline); zero stays the default. *Smoke-verified (2026-07-02):* a stored
  MIB IOI prompt re-run under corrupted ablation reproduces the
  ablation-independent `all` score exactly (4.75) while the ablation-dependent
  scores move as expected (faith 10.06 → 0.25, |E*| 8 → 3 — patch-style
  ablation is far gentler than zeroing). Remaining: *run* the headline
  (attention-mediation flip) under a non-zero mode at sweep scale to show it is
  robust to the choice (`run_todo.md` step 8).
- **"Logit-gap is the wrong metric / single foil."** Results may hinge on the chosen
  foil. *Mitigation (now implemented):* the foil-free, full-distribution KL rescoring
  layer ([§2.5](#25-kl-rescoring-a-selection-independent-faithfulness-metric)) re-scores every stored evidence set and ships as `kl_faith`
  columns in the sweep CSVs (already computed for the in-progress MIB runs; the
  nonlinear-benchmark and case-study roots still need `python -m macag.cli.rescore_kl --root ...`).
  One nuance: the `recoverable_range` **sign** diagnostic is logit-gap-specific — under
  KL the range is non-negative by construction, so the attention-mediation check under
  KL reads the frozen-vs-unfrozen *magnitudes*, not a sign flip. Exclude
  non-target-preferred prompts from faithfulness aggregates (llama rows in C.2/C.3).
- **"Greedy is suboptimal so your evidence sets are arbitrary."** *Mitigation:* the
  optimality-gap experiment (B3.2) and greedy↔Shapley agreement (B2.2).
- **"Game 1's faithfulness win over Shapley is circular — it optimizes that metric."**
  Real and important: Game 1 greedily hill-climbs the oracle's logit-gap faithfulness,
  then we report faith\@k, so its raw-faith edge (median +1.00 vs Shapley, §10.7/C.7)
  is graded on its own objective and a reviewer will discount it. *Mitigation:* lead
  with the two claims that are **not** on the greedy objective — (i) **oracle cost**
  (44.7× cheaper, 60/60 prompts, $p<10^{-9}$) and (ii) **faith-per-feature**
  (non-overlapping CIs vs Shapley; not inflatable by spending features) — and treat
  raw-faith superiority as supporting only. The independent check is now
  implemented: the KL rescoring layer ([§2.5](#25-kl-rescoring-a-selection-independent-faithfulness-metric)) re-scores every
  method's selected set under a metric that is *not* the selection objective, and
  the baseline aggregation reports `kl_faith` per method next to logit-gap faith
  (run it on the nonlinear benchmark via `rescore_kl --root results/macag_nonlinear_connected`
  — still pending there); report agreement-with-gold (prec\@k 0.46 / Jaccard 0.33) as the
  second metric-independent quality signal.
- **"Frozen vs unfrozen is a knob you tuned to get the story."** *Mitigation:* it is a
  *fixed* scoring convention reported both ways for every prompt; the diagnosis is the
  *difference*, and the positive control (B5.2) shows feature-mediated tasks do not
  flip.
- **"Your frozen and unfrozen runs aren't comparable — you changed the budget."**
  True of the stored runs: frozen used budget 8 / prefilter 20, unfrozen 20 / 30
  ([§2.3](#23-attention-freezing-and-the-error-floor)). The range-sign diagnostic is
  budget-independent, but evidence-size and upstream-count contrasts are partially
  confounded. *Mitigation:* pass-2 re-run with matched budgets, or report matched-
  and high-budget variants side by side.
- **"Normalized faithfulness is broken."** Anticipated — own it: §2.3 documents the
  failure mode, the code guards it, and all headline numbers are raw.
- **"n is tiny."** True of the §10.1–10.5 two-hop/IOI case study (8–13 prompts, single
  seed); Phase 1 is the fix. The §10.7/C.7 nonlinear benchmark is the first to ship
  **n=60 with bootstrap CIs and paired Wilcoxon tests** (`scripts/macag_bootstrap_wilcoxon.py`),
  so the cost and faith-per-feature claims now have uncertainty + significance — but
  it is still single-seed *on the CLT/graph* (the selectors are deterministic, so the
  60 prompts are the statistical sample, not seeds). Do not over-claim beyond what the
  prompt-level CIs support.
- **"CLT features ≠ the circuit's true units."** MACAG evaluates the transcoder's
  circuit, not ground truth; gold-circuit recovery (Phase 4) is the bridge, with the
  feature-vs-head mapping stated as a limitation.

---

## 12. Discussion

### 12.1 MACAG as a General-Purpose Evaluator

MACAG is designed to be independent of the method that produced the circuit. It takes a circuit graph and a scoring oracle as inputs and produces faithfulness, utility, and evidence metrics as outputs. This generality means:

1. **Fair comparison across methods**: any two circuit-construction methods — linear CLT, Spline-CLT, a future SAE-based transcoder — are evaluated through the exact same game-theoretic lens. The only thing that differs is the candidate node set, so differences in the metrics are attributable to the circuits, not the evaluator.

2. **Diagnosing coverage gaps**: MACAG's per-prompt and per-task-family breakdown identifies *where* a method fails — which prompts and which task families (and, with repeated runs, which seeds). This is more informative than aggregate reconstruction metrics.

3. **Evidence-based interpretability**: Rather than asking "how good is the transcoder at reconstructing MLP outputs?" (a model-level question), MACAG asks "which specific features causally explain this prediction?" (a circuit-level question).

### 12.2 Future Directions

1. **MACAG-aware training**: Add a regularization term during CLT training that encourages features to have high marginal MACAG utility, not just low reconstruction error. This would directly optimize for circuit faithfulness.

2. **Adaptive $\lambda$ and $\beta$**: Currently fixed across all prompts and variants. Prompt-specific or variant-specific penalty schedules could improve evidence quality.

3. **Beyond greedy**: The greedy hill-climbing algorithm has no optimality guarantees for non-submodular utility functions. Beam search, simulated annealing, or exact solvers for small candidate pools could improve evidence quality.

4. **Multi-model evaluation**: Applying MACAG to circuits from more model families (Pythia, Qwen, larger Gemma) to test whether the attention-mediation diagnosis and contrastive-separation results generalize.

### 12.3 Conference-Readiness: What Is Present vs Missing

An honest accounting of the current `macag/macagresults/` evidence against what a
top-tier venue (NeurIPS/ICLR/ICML interpretability track) would expect.
<!-- TODO(pass-2): the §10.1–10.6 evidence numbers cited in this section are pre-71a2ef6 (§10 provenance box); the §10.7/C.7 baseline numbers are post-fix and current. -->

**Present (framework + mechanism).**
- A complete, encoder-agnostic framework with two well-posed games, a coalitional
  formalization, and a Shapley/Banzhaf connection ([§3](#3-game-theoretic-foundations)).
- A real, non-obvious empirical phenomenon — the **attention-mediation diagnosis**
  (gemma ACDC-benchmark: `recoverable_range` 25/26 negative frozen → 24/26
  non-negative unfrozen, 23/26 strictly flipping), with a clean cross-model contrast
  (llama is feature-mediated) — and a robust structural result (overlap_rate 0.0
  across 48 case-study Game 2 runs, extended to 180/180 including the nonlinear
  benchmark, §10.7(4)).
- Correct treatment of the normalization/denominator failure mode and a
  denominator-free stop ([§2.3](#23-attention-freezing-and-the-error-floor)).
- Three CLTs, two model families, two prompt families (factual + ACDC tasks).
- **A first baseline head-to-head** (Game 1 vs top-k influence, EAP, ACDC,
  Shapley-gold) on a 60-prompt nonlinear benchmark, with bootstrap CIs and paired
  Wilcoxon tests ([§10.7](#107-nonlinear-benchmark-baseline-head-to-head-and-matched-frozenunfrozen-pass-2)/C.7):
  Game 1 owns the cost/faith-per-feature frontier (44.7× fewer oracle calls than
  Shapley-gold, non-overlapping fpf CIs vs every baseline).
- **A selection-independent faithfulness metric** — the KL rescoring layer
  ([§2.5](#25-kl-rescoring-a-selection-independent-faithfulness-metric)): every stored evidence set (Game 1 legs, Game 2 allocations,
  every baseline's set) is re-scored under $-\mathrm{KL}(P_{\text{ref}}\|P_{\text{int}})$,
  wired through the pipeline/drivers/analyzers as `kl_faith` columns and unit-tested.
  This is the built answer to the §11.3 circularity and single-foil threats.
- **MIB-bench integration** ([§9.5](#95-mib-bench-full-campaign-setup)): standardized MIB prompts exported into the MACAG
  benchmark format with per-model CLT routing, now a full-scale 3-seed campaign
  (500 ioi / 50 mcqa / 570 arc_easy per CLT, `results/macag_mib_seed{0,1,2}/`, in
  progress) with a deferred Shapley-gold pass and a matching native-component
  InterpBench validation ([§9.6](#96-interpbench-native-component-gold-circuit-validation-setup), also in progress).

**Missing / blocking for a strong submission.**

1. **Baselines run on one benchmark only.** The selectors and harness are
   implemented (`macag/baselines/`, `python -m macag.cli.run_baselines` — §9.3)
   and now run on the nonlinear benchmark (§10.7/C.7); the harness is also wired
   per-prompt into the sweep drivers with an aggregation analyzer
   (`analyze_macag_baselines.py`), so what is still missing is **execution**, not
   code: the *same* sweep on the **IOI/multi-hop case-study graphs** (§10.1–10.6
   is protocol + diagnostic only, no baseline run; the MIB IOI sweep now underway
   covers part of this — §9.5), and a budget-matched ACDC (its k≈193 run is
   a ceiling, not a matched competitor). **Extending the head-to-head to the
   gold-circuit tasks is the single biggest remaining gap.**
2. **Variance estimates exist only for the nonlinear benchmark.** §10.7/C.7 reports
   95% bootstrap CIs (10 000 resamples over prompts) + paired Wilcoxon; the
   §10.1–10.6 tables are still point estimates, several visibly noisy
   (frozen↔unfrozen swings; e.g. one prompt's *normalized* faithfulness blows up to
   −15.8 on a collapsed denominator — a discarded artifact, not a result, §10.3).
   Note the CLTs are *pretrained and public*, so there is **no training seed to
   vary** — variance must come from (a) **bootstrap over a larger prompt set** (done
   for the nonlinear benchmark) and (b) **repeat runs of the stochastic estimators**
   (Shapley Monte-Carlo seeds — still TODO). Bring the §10 case-study tables up to
   the same CI standard.
3. **Tiny prompt counts (for the §10.1–10.6 case study).** 8 two-hop + 13 ACDC. This
   item is **closed for the MIB tasks**: the MIB exporter ([§9.5](#95-mib-bench-full-campaign-setup)) now runs at
   full benchmark scale (500 ioi / 50 mcqa / 570 arc_easy per CLT, 3 seeds,
   [§9.5.1](#95-mib-bench-full-campaign-setup)) rather than the original 10-per-task pilot; what remains is
   scaling the *hand-written* two-hop/ACDC manifests the same way, or retiring them
   in favor of the MIB prompts entirely.
4. **`greater_than` excluded and gpt-oss failed.** Fix the first-token-foil
   scoring (use full-sequence or answer-span logit gap) so greater-than is usable;
   either drop gpt-oss or load it through a supported path. Right now the coverage
   table has visible holes.
5. **No human / ground-truth circuit validation.** For IOI and greater-than the
   *true* circuits are published; show MACAG's evidence recovers the known
   name-mover / S-inhibition (IOI) and the relevant components, with precision/
   recall against the gold circuit. This is what made ACDC credible.
   *Build done (2026-07-02):* B4.1's (layer, token-role) scorer
   (`macag/eval/gold_circuits.py` + `analyze_gold_circuits.py`) and B4.2's
   InterpBench native-component runner (`run_interpbench_macag.py`, node-level
   AUROC vs the known circuit) are implemented with first smoke numbers (MIB
   IOI, gemma-426k frozen: Game 1 precision 0.38 [0.29, 0.48] vs influence
   0.14, EAP 0.00); the full-scale runs remain.
6. **No statistical/faithfulness curve.** Report faithfulness-vs-size curves
   (sufficiency as $|E|$ grows) and AUC, not just one stop point — this is the
   standard ACDC/EAP comparison axis and removes the matched-size confound.
7. **Greedy optimality is asserted, not bounded.** Either show empirical optimality
   gap vs brute force on small pools, or vs Shapley ranking, to back the
   $(1-1/e)$-or-not discussion ([§3.2](#32-the-value-function-and-submodularity)).
8. **Spline-CLT claim unsupported by these data.** The original motivation (does a
   nonlinear encoder yield more faithful circuits?) has *no* runs here. If that is
   the thesis, the Spline-vs-linear MACAG comparison must be produced; if the
   thesis is the framework + attention diagnosis, state that explicitly.

**Minimum bar to be competitive:** baselines (≥ top-k influence + Shapley-gold,
ideally + ACDC/EAP) on shared graphs *[met on the nonlinear benchmark; pending on
the IOI/multi-hop graphs]*, multi-seed CIs *[prompt-bootstrap CIs done for the
nonlinear benchmark; estimator-seed repeats and §10 case-study CIs pending]*, ≥1
task with gold-circuit recovery *[still TODO]*, and faithfulness-vs-size curves
*[AUC reported in C.7; full curves pending]*. The framework and the
attention-mediation finding are a genuine contribution; the remaining gap is
*comparative and statistical rigor on the gold-circuit tasks*, not ideas.

### 12.4 Limitations

1. **Greedy approximation**: Game 1 and Game 2 both use greedy hill-climbing, which may miss globally optimal evidence sets. The quality of the greedy solution depends on approximate submodularity of the utility function.

2. **Oracle cost**: Each oracle call requires a forward pass through the transformer with modified activations. For Game 2 with large candidate pools, this can require 300+ calls per prompt.

3. **Fixed scoring function**: The logit gap score assumes the target-foil distinction is the relevant behavioral signal. For tasks without a clear foil (e.g., open-ended generation), alternative scoring functions would be needed.

4. **Single-position evaluation**: MACAG evaluates circuits at the final token position. Circuits that operate across multiple positions (e.g., induction heads) may require position-aware scoring.

5. **Attention freezing biases the minimal set**: with `freeze_attention=True`, attention-mediated upstream features are already accounted for by the frozen pattern, so Game 1 can drop them from the minimal faithful set even though they are part of the circuit ([§2.3](#23-attention-freezing-and-the-error-floor)). The direction of the bias is **task-dependent**: on the two-hop relational sweep unfreezing *recruits* hidden upstream/early features, while on IOI and the matched-budget nonlinear benchmark it *shrinks* the set — frozen attention there adds spurious necessity that unfreezing removes (§10.3 correction, §10.7(3)). Either way, single-convention evidence sets are not trustworthy on their own; run `--freeze-mode both` and report the contrast.

6. **Normalized metrics depend on a fragile denominator**: `recoverable_range = all − empty` collapses toward zero or goes negative for attention-mediated tasks (frozen) or when ablate-all collapses the baseline (unfrozen), making the normalized scores and the `normalized` Game 1 stop unreliable. Report the raw (denominator-free) sufficiency/necessity/faithfulness and use the `raw_relative` stop in those regimes ([§2.3](#23-attention-freezing-and-the-error-floor), [§4](#4-game-1-minimal-faithful-evidence)).

---

## 13. Conclusion

MACAG provides a general-purpose, game-theoretic framework for evaluating
attribution circuit graphs via causal interventions. Where a circuit-tracing
pipeline produces a graph whose edges are local linear attribution scores, MACAG
asks the causal follow-up questions those scores cannot answer: its two
complementary games measure whether a *minimal* set of feature nodes is
sufficient and necessary for a prediction (Game 1), and whether competing
predictions are carried by *distinct* features (Game 2). Game 1 is in the spirit
of automated circuit discovery (ACDC) but recast as a bottom-up, sparsity-penalized
selection over transcoder features that scores sufficiency as well as necessity;
Game 2 is a new contrastive evaluation with no analog in the circuit-discovery
work surveyed here (Appendix H).

Because MACAG consumes only a graph and a scoring oracle, its encoder-agnostic
design makes it applicable to any circuit-construction method — linear CLT,
Spline-CLT, or a future transcoder/SAE variant — providing a standardized,
intervention-based evaluation layer for mechanistic interpretability.

<!-- TODO(pass-2): refresh the numbers in this paragraph after the post-71a2ef6 re-run. -->
As an illustration ([§9–§11](#9-case-study-attention-mediation-and-clt-capacity-gemma-2-llama-32)),
applying MACAG to three public CLTs on Gemma-2 and Llama-3.2 surfaces an
**attention-mediation diagnosis** — gemma's ACDC-benchmark behavior is recoverable
from features only once attention is unfrozen (`recoverable_range` negative in
25/26 frozen cases, non-negative in 24/26 unfrozen, 23/26 strictly flipping),
while llama's is feature-mediated throughout — together with perfectly disjoint
target/foil evidence (overlap_rate 0.0 across all 48 Game 2 runs, extended to
180/180 on the nonlinear benchmark). These are
findings about *those circuits*, demonstrating the kind of diagnosis the framework
enables rather than a property of MACAG itself.

On the quantitative side, the post-fix 60-prompt nonlinear benchmark
([§10.7](#107-nonlinear-benchmark-baseline-head-to-head-and-matched-frozenunfrozen-pass-2)/C.7)
supplies the head-to-head that positions the games against the incumbents: Game 1
reaches Shapley-gold-level faithfulness at **44.7× fewer oracle calls** (60/60
prompts, $p<10^{-9}$) and dominates top-k-influence, EAP, and ACDC on
faithfulness-per-feature with non-overlapping 95% CIs — the two claims, per §11.3,
that are *not* graded on Game 1's own greedy objective.

---

## References

1. Ameisen, E., et al. (2025). "Circuit Tracing: Revealing Computational Graphs in Language Models." Anthropic.
2. Lindsey, J., et al. (2025). "On the Biology of a Large Language Model." Anthropic.
3. Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2023). "Towards Automated Circuit Discovery for Mechanistic Interpretability." NeurIPS 2023. arXiv:2304.14997.
4. Syed, A., Rager, C., & Conmy, A. (2023). "Attribution Patching Outperforms Automated Circuit Discovery." arXiv:2310.10348.
5. Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). "An Analysis of Approximations for Maximizing Submodular Set Functions." Mathematical Programming.
6. Shapley, L. S. (1953). "A Value for n-Person Games." Contributions to the Theory of Games.
7. Nanda, N. (2023). "Attribution Patching: Activation Patching at Industrial Scale." (blog/technical note).
8. Elhage, N., et al. (2022). "Toy Models of Superposition." Anthropic / Transformer Circuits.
9. Bricken, T., et al. (2023). "Towards Monosemanticity: Decomposing Language Models with Dictionary Learning." Anthropic / Transformer Circuits.
10. Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). "Sparse Autoencoders Find Highly Interpretable Features in Language Models." arXiv:2309.08600.
11. Templeton, A., et al. (2024). "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." Anthropic / Transformer Circuits.
12. Dunefsky, J., Chlenski, P., & Nanda, N. (2024). "Transcoders Find Interpretable LLM Feature Circuits." arXiv:2406.11944.
13. Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2022). "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small." arXiv:2211.00593.
14. Chan, L., Garriga-Alonso, A., et al. (2022). "Causal Scrubbing: a Method for Rigorously Testing Interpretability Hypotheses." Alignment Forum / Redwood Research.
15. Lundberg, S. M., & Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions (SHAP)." NeurIPS 2017.
16. Ghorbani, A., & Zou, J. (2019/2020). "Data Shapley" (ICML 2019) and "Neuron Shapley" (NeurIPS 2020).
17. Hsu, A. R., Zhou, G., Cherapanamjeri, Y., Huang, Y., Odisho, A. Y., Carroll, P. R., & Yu, B. (2024). "Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition (CD-T)." arXiv:2407.00886.
18. Chowdhary, P., Chin, P., & Chakrabarty, D. (2025). "K-MSHC: Unmasking Minimally Sufficient Head Circuits in Large Language Models." arXiv:2505.12268.
19. Chowdhury, T., Nijasure, A., Zick, Y., & Allan, J. (2025). "Hedonic Neurons: A Mechanistic Mapping of Latent Coalitions in Transformer MLPs." arXiv:2509.23684.
20. Kang, J. S., Butler, L., Agarwal, A., Erginbas, Y. E., Pedarsani, R., Ramchandran, K., & Yu, B. (2025). "SPEX: Scaling Feature Interaction Explanations for LLMs." ICML 2025. arXiv:2502.13870.
21. Hadad, I., Katz, G., & Bassan, S. (2026). "Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees." ICLR 2026. arXiv:2602.16823.
22. Khadka, B. (2026). "MechRL: Reinforcement Learning Agents Perform Circuit Discovery for Mechanistic Interpretability." arXiv:2605.26343.
23. Lei, Z., Wu, Q., Dong, J., He, Y., Dodwell, E., Dong, Y., & Li, J. (2026). "Reforming the Mechanism: Editing Reasoning Patterns in LLMs with Circuit Reshaping (REdit)." arXiv:2603.06923.
24. Hudson et al. (2025). "Transcoders Beat Sparse Autoencoders for Interpretability." arXiv:2501.18823.

---

## Appendix A: Baseline Comparison Details

This appendix expands the baseline table in [§9.3](#93-baselines-and-evaluation-protocol)
into a per-method assessment: is the method a suitable baseline, how should it be
run against MACAG, what does it share with MACAG, where it differs, and which
metrics make the comparison fair. The unifying view is the coalitional game
$(N, v)$ of [§3.0](#30-the-underlying-coalitional-game): every baseline is some way
of ranking or selecting feature nodes under (an approximation of) the same value
function $v$.

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
  ([§6.3](#63-candidate-policies)); here it is frozen as a *selector*.
- **How to run it:** take the $k$ highest-influence nodes and score the set's
  faithfulness directly. This is the floor MACAG must clear; if MACAG cannot beat
  it, the search is not earning its cost.
- **Metrics:** faithfulness at matched $|E|$; overlap (Jaccard) with MACAG's set.

### A.4 Shapley / Banzhaf (gold) — suitable: upper bound, not a competitor

- **Suitable?** Yes, but as a **gold-standard reference**, not a rival selector.
- **Similar:** Defined over the *same* characteristic function $v$
  ([§3.6](#36-relation-to-shapley-and-banzhaf-values)).
- **Different:** Solves credit **assignment** (per-feature value), not set
  **selection**; far more expensive (averages over $2^{|C|}$ coalitions,
  MC-estimated). Banzhaf is the natural second reference (all-coalitions average,
  less order-sensitive for strongly interacting features) and is implemented
  alongside Shapley (`estimate_banzhaf`).
- **Now built and run for MACAG** (`macag/baselines/shapley_select.py`): MC
  permutation Shapley (antithetic) and MC Banzhaf, both over the MACAG oracle's $v$;
  `attribution/shapley.py` was not reused — canonical statement and reasons in
  [§3.6](#36-relation-to-shapley-and-banzhaf-values). Validated on toy oracles
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
| Faithfulness | raw Δ logit-gap (per [§2.3](#23-attention-freezing-and-the-error-floor)) and/or KL, **at matched evidence size** |
| Parsimony | evidence size at matched faithfulness |
| Cost | oracle forward passes (plus backward passes for EAP) |
| Agreement | Jaccard / precision@$k$ vs. Shapley-gold and pairwise |
| Linearity diagnostic | Spearman(EAP score, MACAG marginal gain) |
| Stability | cross-seed Jaccard of selected sets |

**Two non-negotiables.** (1) **Always compare at matched set size** — otherwise the
faithfulness-vs-size confound reappears (the same confound flagged for the §10 CLT
tables). (2) **Report raw, not normalized, faithfulness** for the attention-mediated
baselines, because `recoverable_range` is unreliable there
([§2.3](#23-attention-freezing-and-the-error-floor)).

**Implementation status.** All selectors and the head-to-head harness are now
implemented: `macag/baselines/` (influence B2.1, Shapley/Banzhaf-gold B2.2, EAP
B2.3, ported ACDC B2.4, brute-force B3.2) driven by
`python -m macag.cli.run_baselines` (B2.0), which already emits every metric in
the table above except cross-seed stability (one run = one seed; loop seeds for
that row). ACDC-*native* (heads/MLPs granularity) is still open. Beyond the
toy-oracle unit tests ([§3.6](#36-relation-to-shapley-and-banzhaf-values)), the
real-graph sweep **has now run on the 60-prompt nonlinear benchmark** (§10.7/C.7,
`results/macag_nonlinear_connected/`); the IOI/multi-hop case-study graphs and the
cross-seed stability row are still open.

---

## Appendix B: Roadmap to a Submission-Ready Evaluation

This turns the gaps in [§12.3](#123-conference-readiness-what-is-present-vs-missing)
into an ordered, concrete plan. Each task lists **build** (code to add/change),
**run** (command), and **done-when** (acceptance criterion). **As of 2026-07-02
every build item below is implemented; the exact commands for the remaining
*runs* are consolidated in [`macag/docs/run_todo.md`](run_todo.md).** Reuse the existing
drivers wherever possible (`scripts/run_macag_*.sh`, `macag/cli/run_macag.py`).
The MACAG Shapley-gold baseline was written fresh over the MACAG oracle —
deliberately *not* a reuse of `attribution/shapley.py`
([§3.6](#36-relation-to-shapley-and-banzhaf-values)).

**Scope (fixed).** *Public pretrained CLTs only* — `mntss/clt-gemma-2-2b-426k`,
`mntss/clt-gemma-2-2b-2.5M`, `mntss/clt-llama-3.2-1b-524k`, plus any further public
CLTs as they appear. **No Spline-CLT, no GPT-2, no gpt-oss.** Because the CLTs are
pretrained, there is no training seed; statistical variance comes from the prompt
sample (bootstrap) and from stochastic estimators (Shapley MC), not from retraining.

### Phase 0 — Close coverage holes *(small; unblocks the rest)*

- **B0.1 Fix `greater_than` scoring.** Its target/foil (e.g. "42"/"40") collide on
  the first token, so the first-token logit gap is degenerate.
  - *Build:* ✅ **DONE (2026-07-02)** — `score_kind="answer_span"` in
    `macag/scoring.py` scores the teacher-forced summed log-prob gap over the full
    target/foil answer spans (two forwards per oracle score; single-token spans
    reduce exactly to `logit_gap`). Spans resolve from the existing
    `target_token_by_label` via `resolve_target_to_token_span`, so
    `oracle_kwargs.json` and the KL rescorer round-trip unchanged. The pipeline
    takes `--score-kind` / `SCORE_KIND`, and `run_macag_acdc.sh` now includes
    `greater_than` by default, routed to `answer_span`. Tests:
    `tests/test_macag_answer_span.py`.
  - *Run:* ✅ smoke passed (2026-07-02): gt_01 through the full dual-freeze
    pipeline with `SCORE_KIND=answer_span` gives `all = 4.5` (target-preferred;
    the first-token gap was identically 0) with nonzero range on both legs
    (0.39 frozen / 2.79 unfrozen, `feature_mediated`), and the KL rescore
    round-trips the answer_span kwargs.
  - *Done when:* `greater_than` prompts are target-preferred with a nonzero baseline
    gap, and can be re-included in the ACDC benchmark — met; the driver now
    includes greater_than by default.
- **B0.2 Remove gpt-oss from the comparison config** (it is unsupported by
  `transformer_lens`). ✅ **DONE (2026-07-02)** — `gptoss20b-131k` deleted from
  `experiments/macag_clt_compare.json`. Also fixed alongside: the blanket `data/`
  gitignore rule now carries `!macag/data/` exceptions, so the benchmark prompt
  manifests are committable (Appendix I item 6), and `scipy` is declared in
  `pyproject.toml`.
- **B0.3 Selection-independent faithfulness metric (KL).** ✅ **DONE** — built,
  wired, and unit-tested ([§2.5](#25-kl-rescoring-a-selection-independent-faithfulness-metric)): `score_kind="kl_divergence"` in
  `macag/scoring.py`, post-hoc rescoring via `macag/kl_rescore.py` /
  `python -m macag.cli.rescore_kl` (per run dir or whole sweep root), automatic in
  `run_macag_pipeline.sh` and the sweep drivers, `kl_faith` columns in every
  aggregate CSV, tests in `tests/test_macag_kl_scoring.py`. *Remaining execution:*
  rescore the already-stored roots
  (`rescore_kl --root results/macag_nonlinear_connected`, and `macag/macagresults/`
  legacy runs if they are re-run) — the in-progress MIB sweep already carries it.

### Phase 1 — Scale prompts + add confidence intervals *(statistical rigor)*

- **B1.1 Expand prompt sets.** Two-hop factual → ~30–50 city/state pairs; IOI → 
  ~50–100 (standard ABBA/BABA templates); `greater_than` → ~50; docstring → as many
  as available.
  - *Build:* ✅ **done for MIB tasks** — the MIB exporter
    (`experiments/build_mib_benchmark_prompts.py`, [§9.5.1](#95-mib-bench-full-campaign-setup)) now supplies
    standardized IOI/MCQA/ARC prompts in the MACAG manifest format at full
    benchmark scale (500/50/570, `--task-limit ioi=500`, validation split); the
    hand-written manifests (`macag/data/acdc_benchmark_prompts.json`,
    `experiments/macag_generalization_prompts.json`) are unchanged and still small.
  - *Run:* `scripts/run_macag_acdc.sh` (the consolidated driver — the old
    `run_macag_sweep.sh`/`run_macag_unfrozen*.sh` are folded into it) for the
    hand-written manifests, and `scripts/run_mib_benchmark.sh` ([§9.5.4](#95-mib-bench-full-campaign-setup)) for the
    full-scale, 3-seed MIB campaign — **in progress**
    (`results/macag_mib_seed{0,1,2}/`).
- **B1.2 Bootstrap CIs over prompts.** ✅ **Build DONE (2026-07-02)** —
  `scripts/macag_bootstrap_wilcoxon.py` is (re)implemented and checked in: bootstrap
  CIs (reusing `spline_clt.paper.reporting.bootstrap_mean_ci`), paired Wilcoxon with
  Holm correction and hand-rolled rank-biserial, win/loss counts, pref-only blocks,
  and the Shapley/Game-1 cost ratio; verified to reproduce the §10.7 numbers from
  `results/macag_nonlinear_connected/baselines.csv` (faith\@8 5.50 [4.88, 6.15],
  fpf 0.859, cost 44.7× [43.5, 45.9]). The flip-rate CI is also in:
  `analyze_acdc_frozen_vs_unfrozen.py` (`aggregate_flip_stats` →
  `frozen_vs_unfrozen_agg.csv`) and `analyze_macag_acdc.py` (`flip_lo`/`flip_hi` →
  `summary_agg.csv`) bootstrap the *fraction of prompts with
  `recoverable_range` < 0* / range-flip proportions per CLT×task. Tests:
  `tests/test_macag_stats.py`.
  - *Done when:* every number in §10 is reported as mean [lo, hi] — remaining is
    execution: re-run the §10.1–10.6 sweeps and pipe them through these scripts.
- **B1.3 Shapley estimator variance.** Run the Shapley baseline (Phase 2) with ≥3 MC
  seeds; report ranking std / rank-correlation stability.

### Phase 2 — Baselines on shared graphs *(was the #1 blocker; now run on the nonlinear benchmark)*

Run every selector on the **same** candidate node set and the **same** oracle as
MACAG, so only the selection rule differs. Build one harness, four selectors.

> **Status: B2.0–B2.4 are BUILT and RUN** (plus B3.2's brute-forcer), unit-tested
> on toy oracles in `tests/test_macag_baselines.py` and **run end-to-end with the
> real `ReplacementModel` oracle on the 60-prompt nonlinear benchmark**
> (§10.7/C.7, `results/macag_nonlinear_connected/baselines.csv`) — the master
> table with bootstrap CIs + paired Wilcoxon exists. The harness is now also
> **embedded in the sweep drivers**: `run_macag_acdc.sh` / `run_macag_mib.sh` emit
> a per-prompt `macag_baselines.json` (disable with `SKIP_BASELINES=1`;
> `BASELINE_METHODS`/`SHAPLEY_PERMUTATIONS` configurable) and
> `experiments/analyze_macag_baselines.py` aggregates a sweep root into
> `baselines.csv` — including `kl_faith` columns from the [§2.5](#25-kl-rescoring-a-selection-independent-faithfulness-metric)
> rescoring pass. **Cost note / two-pass split (2026-07-02):** MC Shapley is
> ~90% of a prompt's baseline cost (≈33.9k oracle calls vs Game 1's ~760, ACDC's
> ~2.6k), so the drivers support a fast pass without it
> (`BASELINE_METHODS="influence,eap,game1,acdc"`) followed by a deferred gold
> pass: `scripts/run_macag_shapley_pass.sh <sweep_root>` runs shapley-only per
> stored graph and `python -m macag.cli.merge_baselines` merges it back,
> rebuilding the full comparison block (gold agreement, prec@k/Jaccard,
> Spearman) from stored rankings — pure JSON, no extra oracle calls
> (round-trip pinned by `test_merge_baselines_deferred_shapley`). What remains
> for Phase 2 is **execution**: finish the MIB sweep, re-run the ACDC benchmark
> through the consolidated driver, and add the cross-seed stability row
> (Shapley MC seeds, B1.3). Exact commands: `macag/docs/run_todo.md`.

- **B2.0 Harness.** *(built)* `macag/cli/run_baselines.py`: load graph + oracle (reuse
  `macag.factories.replacement_model`), then for each method emit
  `{method: {k: {evidence, scores}}}` for k = 1..budget, using the **same**
  `FaithfulnessMetrics` scoring as the games.
- **B2.1 Top-k influence** *(built)* — `macag/baselines/influence.py`: read `influence` from
  the graph JSON, take top-k. (Cheapest; the floor MACAG must beat.)
- **B2.2 Shapley-gold** *(built; Banzhaf included)* — `macag/baselines/shapley_select.py`: implement a
  Monte-Carlo Shapley estimator **over the MACAG oracle** ($v$ = `FaithfulnessMetrics`
  on keep/remove ablations through `ReplacementModel`), rank by Shapley value, take
  top-k. (Upper bound; also the validation target.) **Do not** wrap
  `attribution/shapley.py` ([§3.6](#36-relation-to-shapley-and-banzhaf-values)) —
  it would silently measure the wrong game.
- **B2.3 EAP / attribution patching** *(built — the graph-derived cheap variant)* — `macag/baselines/eap.py`: first-order
  grad×activation node scores via the `ReplacementModel`; or, since the graph edges
  *are* attribution scores, derive a node score directly from the graph as the cheap
  variant. Take top-k.
- **B2.4 ACDC (ported)** *(built)* — `macag/baselines/acdc_prune.py`: top-down — start from the
  full candidate set, remove a node if ablating it changes the score by < τ; sweep τ.
  Reuses the same oracle; isolates prune-vs-grow. **Budget-matched ACDC added
  (2026-07-02):** `acdc_target_size` bisects τ to a target evidence size (nearest
  achievable size with a value tie-break and an `exact` flag when integer-size
  plateaus make k unreachable); `run_baselines --acdc-target-k` (−1 ⇒ `--budget`;
  driver env `ACDC_TARGET_K`) emits `methods.acdc.matched_k` and mirrors ACDC into
  `comparison.faithfulness_at_k` — removing the "ACDC k≈193 is not a competitor"
  caveat from future runs (§10.7, §11.3). The real-graph smoke hardened the
  search twice: a bisection midpoint can collide with the τ=0 seed (now reused
  to narrow the bracket instead of bailing), and prune-cascade plateaus can skip
  sizes entirely (the search is now seeded with the τ-sweep's results, so
  nearest-size ranking sees them; on the smoke prompt k=8 is genuinely
  unreachable — sizes jump 16→3→1 — and the returned `matched_k` honestly
  reports `achieved_k=3, exact=false`). Expect and report `exact=false` rows.
- *Run:* loop `run_baselines.py` over every `(CLT, prompt)` graph — now automatic
  inside `run_macag_acdc.sh` / `run_macag_mib.sh`. **Done for the
  60-prompt nonlinear benchmark** (`results/macag_nonlinear_connected/`); **in
  progress for the MIB IOI/MCQA/ARC prompts** (`results/macag_mib/`); still to
  do over the stored IOI/multi-hop graphs in `macag/macagresults/` (re-run them
  through the consolidated driver).
- *Done when:* a master table reports, per method, faithfulness@matched-k,
  |E|@matched-faithfulness, and oracle calls — plus precision@|E| and rank
  correlation of MACAG vs Shapley-gold. **This table is the paper's core result.**
  *(Done for the nonlinear benchmark — §10.7 table + C.7; precision@k 0.46,
  Jaccard 0.33 vs gold. Pending for the IOI/multi-hop tasks.)*

### Phase 3 — Faithfulness–size curves + optimality gap

- **B3.1 Curves.** ✅ **Build DONE (2026-07-02)** — `experiments/plot_faithfulness_curves.py`
  aggregates each run's stored `comparison.faithfulness_at_k` (plus ACDC
  `best_by_size`/`matched_k`) into per-(CLT, task, method) mean-faith(k) curves with
  bootstrap CI bands, `curves.csv` + `auc.csv`, and one PNG panel per (CLT, task)
  with Game 1's mean own-|E*| stop marker; smoke-verified on
  `results/macag_nonlinear_connected/` (12 panels). *Done when:* one curve per
  method per task — remaining is re-running sweeps with `--acdc-target-k` so ACDC
  contributes budget-range points.
- **B3.2 Greedy optimality gap.** *(brute-forcer built — `--bruteforce-k` in
  `run_baselines.py` reports the gap vs every method's k-prefix; the sweep itself
  is still to run.)* On small pools (prefilter to ~12–15 candidates),
  brute-force the best size-k subset (`macag/baselines/bruteforce.py`) and compare to
  greedy. *Done when:* you can state the empirical optimality gap, backing the
  $(1-1/e)$/non-submodular discussion in
  [§3.2](#32-the-value-function-and-submodularity).

### Phase 4 — Gold-circuit validation

- **B4.1 Known circuits.** Encode the published IOI circuit (name-mover,
  S-inhibition, induction, duplicate-token heads) and the `greater_than` components.
  Because CLT features are not attention heads, validate at the **(layer,
  token-role)** level — does MACAG's evidence read from the known positions/layers?
  - *Build:* ✅ **DONE (2026-07-02)** — `macag/eval/gold_circuits.py`: `IOI_GOLD`
    encodes the published components as (depth-fraction band, token-role) regions
    (bands derived from the GPT-2-small layers, widened; a documented judgment
    call), `assign_token_roles` maps S1/S2/IO/END from manifest metadata with a
    heuristic fallback for MIB prompts, and `score_evidence_against_gold` reports
    node-level precision + **component-level** recall (feature-level recall is
    undefined for CLT features — the flagged caveat). Analyzer:
    `experiments/analyze_gold_circuits.py` (`--include-baselines` scores every
    selector's set) → `gold_circuits.csv` + bootstrap-CI aggregate. Tests:
    `tests/test_macag_gold_circuits.py`.
  - *First real numbers* (gemma2-426k, 10 MIB IOI prompts, frozen leg): Game 1
    precision **0.38 [0.29, 0.48]** vs Shapley-gold 0.42, influence 0.14, EAP
    0.00; the unfrozen leg drops to ~0 (evidence leaves the late/END gold
    regions once attention recomputes) — single-CLT, n=10, read as a smoke
    result until the full sweep re-runs.
  - *Done when:* ≥1 task (IOI) shows MACAG recovers the known structure with reported
    precision/recall. This is what made ACDC credible; flag the feature-vs-head
    mapping as an explicit caveat.
  - *InterpBench:* the exact-ground-truth path (known circuit, node-level AUROC)
    is B4.2 below.
- **B4.2 InterpBench exact validation.** ✅ **Build DONE (2026-07-02)** — MACAG
  now runs on **native components**: `macag/scoring_components.py` implements the
  full four-mode oracle contract over attention heads `a{l}.h{h}` and MLPs
  `m{l}` via TransformerLens hooks (`HookedComponentInterventionScorer`; needs
  `use_attn_result`), demonstrating the framework's encoder-agnosticism beyond
  transcoder features. `experiments/run_interpbench_macag.py` loads the
  InterpBench IOI model (vendored MIB loader, device-parametrized), runs Game 1
  + MC Shapley over the 30-component universe, and scores against the *known*
  circuit (`interpbench_graph.json` node `in_graph` flags = {m0, a1.h1, a2.h1,
  a4.h1}): **node-level AUROC/AP** of Shapley credit (hand-rolled in
  `macag/eval/gold_circuits.py` — upstream MIB's AUROC is edge-level only) and
  set-level P/R/F1 of the Game 1 evidence (well-defined here — both sides are
  components). Tests: `tests/test_macag_component_scorer.py`.
  - *Run DONE (2026-07-02, n=50 validation IOI prompts, 64 Shapley
    permutations, budget 4; `results/interpbench_macag/interpbench_macag.csv`):*
    **Shapley AUROC 0.630 [0.570, 0.688]** (0.651 on the 40/50 target-preferred
    prompts) — the CI excludes the 0.5 chance level, so per-component gold
    credit does rank the known circuit above other components. **Game 1 set
    precision/recall are weak: 0.175 [0.125, 0.228] / 0.165 [0.120, 0.215]**,
    only marginally above the random-size-4-of-30 baseline (~0.13): the greedy
    evidence usually contains a1.h1 but fills the rest of the budget with
    non-gold heads (a1.h3, a5.h1/h2 recur). Honest reading for the paper: on
    this semi-synthetic model, *credit assignment* (Shapley over the MACAG
    oracle) recovers the known circuit signal, while *minimal-set selection*
    under zero-ablation + logit-gap does not — either redundancy in the trained
    InterpBench model or a real Game 1 limitation; report both numbers, don't
    cherry-pick the early 2-prompt smoke (P=R=0.75 on its target-preferred
    prompt — small-n optimism). Follow-ups: sweep the budget (curves), try
    `kl_divergence`/mean-ablation scoring, and compare against the MIB
    leaderboard methods' node sets on the same 50 prompts.

### Phase 5 — Harden the attention-mediation headline

- **B5.1 Scale + CI the flip.** Re-run the frozen/unfrozen ACDC pipeline on the
  larger IOI/greater_than sets; report the negative→positive `recoverable_range`
  flip rate with bootstrap CIs, per task and per CLT. The matched dual-freeze
  protocol is now the driver default (`scripts/run_macag_acdc.sh` with
  `FREEZE_MODE=both`; the old separate `run_macag_acdc_unfrozen.sh` two-pass path
  is consolidated away), and the analyzers already emit per-prompt
  `verdict`/`range_flip` and per-CLT×task `flip_rate` columns — what remains is
  the larger prompt sets (B1.1/MIB) and the CI aggregation (B1.2). The MIB IOI
  sweep in progress is the first installment.
- **B5.2 Positive control.** Confirm a known *feature-mediated* task (factual recall /
  the two-hop sweep) does **not** show the flip — this makes the diagnosis a
  discriminating test, not an artifact.

### Definition of done (minimum competitive bar)

1. Phase 2 table with **≥ top-k-influence + Shapley-gold** (ideally + ACDC/EAP) on
   shared graphs. *[done on the nonlinear benchmark (§10.7/C.7); harness now embedded
   in the sweep drivers — execution pending on the IOI/multi-hop graphs, with the
   MIB IOI sweep in progress (§9.5)]*
2. Phase 1 bootstrap CIs on every headline number. *[done for the nonlinear
   benchmark (script needs re-check-in — B1.2); §10 case-study tables +
   estimator-seed repeats pending]*
3. Phase 3 faithfulness-vs-size curves. *[AUC reported in C.7 and now emitted per
   sweep in `baselines.csv`; full curves pending]*
4. Phase 4 gold-circuit recovery on ≥1 task. *[build done (B4.1 + B4.2,
   2026-07-02): (layer, token-role) IOI scoring + native-component InterpBench
   validation with node-level AUROC; first smoke numbers on MIB IOI (Game 1
   precision 0.38 vs influence 0.14); full runs pending]*
5. Phase 5 attention-mediation flip with CIs + positive control. *[matched
   frozen/unfrozen + CIs done on the nonlinear benchmark (§10.7); dual-freeze is now
   the driver default with `verdict`/`flip_rate` in the CSVs; IOI/greater_than
   scale-up + positive control pending]*
6. Selection-independent metric (KL, B0.3) reported next to logit-gap faith. *[built +
   wired (§2.5); computed for the in-progress MIB runs; rescore of the nonlinear
   benchmark and any re-run case-study roots pending]*

Phases 0–2 are the critical path; 3–5 can proceed in parallel once the baseline
harness (B2.0) exists. The framework and the attention-mediation finding are already
a contribution — this roadmap supplies the comparative and statistical rigor a
top-tier venue expects.

---

## Appendix C: Full Per-Prompt Results

Raw material for tables/figures, taken verbatim from the analysis CSVs in
`macag/macagresults/`. Single seed; all scores in logit-gap units. Columns:
`E*`/`E_f`/`E_u` = evidence size (frozen/unfrozen); `up` = upstream feature count
(reverse-position > 0, i.e. not at the prediction token); `range` =
`recoverable_range` = all − empty; `faith`/`suff`/`nec` with `_norm` = divided by
`range` (unreliable when |range| small/negative — read raw); `pref` = model
predicts target at baseline.

### C.1 Two-hop factual, gemma2-426k, frozen (`macag_sweep/summary.csv`)

> *Source note (pass 1).* `summary.csv` is dated 2026-06-03 but the
> `macag_game1.json` files now in `macag_sweep/` are from a 2026-06-04 re-run, and
> the two disagree slightly on the same prompts (miami `empty` 6.75 here vs 6.5 in
> the JSON, range 1.5 vs 1.75; detroit $E^*$ 1 vs 2; houston 6 vs 7). C.3's frozen
> columns read the JSONs — which is why C.1 and C.3 differ on the same frozen runs.
> <!-- TODO(pass-2): regenerate summary.csv and this table from a single re-run. -->

| slug | all | empty | range | E* | faith_norm | suff_norm | nec_norm | sparsity | overlap |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| chicago-springfield | 2.75 | -1.81 | 4.56 | 5 | 0.99 | 1.11 | 0.86 | 0.99 | 0 |
| cleveland-columbus | 3.56 | -2.52 | 6.08 | 8 | 0.93 | 0.23 | 1.62 | 0.98 | 0 |
| dallas-austin | 5.0 | 1.31 | 3.69 | 4 | 1.03 | -0.05 | 2.12 | 0.99 | 0 |
| detroit-lansing | 3.5 | 1.41 | 2.09 | 1 | 0.91 | 1.94 | -0.12 | 1.00 | 0 |
| houston-austin | 4.62 | -0.75 | 5.38 | 6 | 0.91 | 0.05 | 1.77 | 0.98 | 0 |
| miami-tallahassee | 8.25 | 6.75 | 1.5 | 2 | 1.32 | 0.06 | 2.58 | 0.99 | 0 |
| philadelphia-harrisburg | 1.62 | 4.25 | **-2.62** | 8 | **-2.52** | -1.48 | -3.57 | 0.98 | 0 |
| portland-salem | 5.19 | -1.16 | 6.34 | 8 | 0.83 | 0.6 | 1.05 | 0.98 | 0 |

*Observations.* (i) `empty` is frequently **negative** (chicago −1.81, cleveland
−2.52, houston −0.75, portland −1.16): with all features ablated and attention
frozen, the model prefers the *foil*, so features carry the entire target
preference — a clean feature-mediated signature, and the regime where the normalized
metric behaves (faith_norm ≈ 0.83–1.32). (ii) philadelphia is the lone
reconstruction failure: `empty` = +4.25 > `all` = +1.62, so ablating features
*raises* the gap → negative range, nonsense normalized faith (−2.52); its raw
numbers are the ones to quote. (iii) miami has a high error floor (`empty` 6.75):
most of the behavior survives full ablation, small recoverable range (1.5),
faith_norm inflated to 1.32. (iv) Evidence size varies 1–8 with no obvious tie to
range; detroit needs a single feature. (v) overlap = 0 everywhere.

### C.2 Capacity / cross-model, two-hop, frozen (`macag_clt_compare/comparison_per_prompt.csv`)

| CLT | slug | pref | range | faith_norm | E* | overlap |
|---|---|:-:|--:|--:|--:|--:|
| gemma2-426k | chicago-springfield | True | 4.56 | 0.99 | 5 | 0 |
| gemma2-426k | cleveland-columbus | True | 6.08 | 0.93 | 8 | 0 |
| gemma2-426k | dallas-austin | True | 3.69 | 1.03 | 4 | 0 |
| gemma2-426k | detroit-lansing | True | 2.09 | 0.91 | 1 | 0 |
| gemma2-426k | houston-austin | True | 5.38 | 0.91 | 6 | 0 |
| gemma2-426k | miami-tallahassee | True | 1.5 | 1.32 | 2 | 0 |
| gemma2-426k | philadelphia-harrisburg | True | -2.62 | -2.52 | 8 | 0 |
| gemma2-426k | portland-salem | True | 6.34 | 0.83 | 8 | 0 |
| gemma2-2.5M | chicago-springfield | True | 3.14 | 1.05 | 3 | 0 |
| gemma2-2.5M | cleveland-columbus | True | 8.62 | 0.85 | 8 | 0 |
| gemma2-2.5M | dallas-austin | True | 8.75 | 0.54 | 8 | 0 |
| gemma2-2.5M | detroit-lansing | True | **-3.44** | -1.95 | 8 | 0 |
| gemma2-2.5M | houston-austin | True | 7.47 | 0.58 | 8 | 0 |
| gemma2-2.5M | miami-tallahassee | True | 1.44 | 1.17 | 2 | 0 |
| gemma2-2.5M | philadelphia-harrisburg | True | 1.19 | 2.45 | 1 | 0 |
| gemma2-2.5M | portland-salem | True | 3.22 | 1.04 | 4 | 0 |
| llama32-524k | chicago-springfield | True | 3.39 | 1.08 | 1 | 0 |
| llama32-524k | cleveland-columbus | True | -1.84 | -3.44 | 8 | 0 |
| llama32-524k | dallas-austin | True | 6.06 | 0.92 | 5 | 0 |
| llama32-524k | detroit-lansing | **False** | -4.75 | -1.74 | 8 | 0 |
| llama32-524k | houston-austin | True | 5.27 | 0.91 | 7 | 0 |
| llama32-524k | miami-tallahassee | True | -6.22 | -0.88 | 8 | 0 |
| llama32-524k | philadelphia-harrisburg | **False** | -8.75 | -0.73 | 8 | 0 |
| llama32-524k | portland-salem | **False** | -3.06 | -2.27 | 5 | 0 |

*Observations.* (i) The reconstruction failure **moves** with capacity: 426k fails
on philadelphia, 2.5M fails on detroit instead — capacity changes *which* prompt is
unrecoverable, not *whether* one is. (ii) 2.5M tends to larger evidence sets at high
range (dallas 8.75/8, houston 7.47/8) → lower faith_norm (0.54, 0.58): more capacity
spreads the behavior over more features, so a budget-8 set recovers a smaller
*fraction* of the larger range. (iii) llama is the weak case: 3/8 not
target-preferred and 4/8 negative range — when a prompt is not even target-preferred
the logit-gap oracle is measuring the wrong thing, so those rows should be excluded
from faithfulness aggregates (a protocol note for Phase 1).

### C.3 Game 1 frozen vs unfrozen, two-hop (`macag_unfrozen/robust_frozen_vs_unfrozen.csv`)

> *Source note (pass 1).* Two unfrozen two-hop reruns exist: `macag_unfrozen/`
> (`normalized` stop — **this table**) and `macag_unfrozen_raw/` (`raw_relative`
> stop; also the source of C.5's Game 2 CSV). The raw-stop rerun gives different
> unfrozen numbers (e.g. miami $E_u$ 15 vs 13, dallas 10 vs 8). Both unfrozen
> reruns used budget 20 / prefilter 30 vs the frozen legs' 8 / 20 (§9.1). All
> columns here are raw scores — fine to read under either attention mode.
> <!-- TODO(pass-2): one matched-protocol re-run; quote the fixed-raw_relative version. -->

| CLT | slug | pref | E_f | E_u | up_f | up_u | suff_f | suff_u | faith_f | faith_u | range_f | range_u |
|---|---|:-:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| gemma2-426k | dallas-austin | T | 4 | 8 | 0 | 2 | 0.0 | 3.88 | 3.91 | 6.59 | 3.81 | 6.94 |
| gemma2-426k | houston-austin | T | 7 | 5 | 0 | 3 | 0.0 | 2.56 | 5.34 | 5.41 | 5.31 | 5.69 |
| gemma2-426k | chicago-springfield | T | 5 | 3 | 2 | 2 | 4.88 | 2.5 | 4.44 | 4.38 | 4.56 | 4.12 |
| gemma2-426k | miami-tallahassee | T | 2 | 13 | 0 | 8 | 0.03 | 9.72 | 1.95 | 10.86 | 1.75 | 15.84 |
| gemma2-426k | detroit-lansing | T | 2 | 2 | 1 | 1 | 3.78 | 6.56 | 3.61 | 4.88 | 2.31 | 4.62 |
| gemma2-426k | portland-salem | T | 8 | 1 | 3 | 1 | 4.34 | 3.66 | 5.14 | 1.64 | 6.34 | 0.94 |
| gemma2-426k | cleveland-columbus | T | 8 | 7 | 1 | 2 | 1.28 | 2.69 | 5.58 | 6.53 | 6.12 | 7.06 |
| gemma2-426k | philadelphia-harrisburg | T | 8 | 2 | 3 | 1 | 4.88 | 3.44 | 6.5 | 2.78 | -2.31 | 1.62 |
| gemma2-2.5M | dallas-austin | T | 8 | 7 | 3 | 2 | 4.31 | 6.38 | 4.72 | 5.22 | 8.75 | 5.5 |
| gemma2-2.5M | houston-austin | T | 8 | 1 | 1 | 0 | 2.97 | 0.09 | 4.33 | 0.89 | 7.47 | 0.12 |
| gemma2-2.5M | chicago-springfield | T | 3 | 17 | 1 | 5 | 3.95 | 0.25 | 3.29 | 6.97 | 3.14 | -2.91 |
| gemma2-2.5M | miami-tallahassee | T | 2 | 14 | 0 | 5 | 0.12 | 5.84 | 1.69 | 7.61 | 1.44 | 9.5 |
| gemma2-2.5M | detroit-lansing | T | 8 | 17 | 1 | 4 | 1.31 | 3.34 | 6.72 | 9.86 | -3.44 | -2.22 |
| gemma2-2.5M | portland-salem | T | 4 | 13 | 1 | 8 | 3.12 | 3.69 | 3.34 | 5.94 | 3.22 | -0.38 |
| gemma2-2.5M | cleveland-columbus | T | 8 | 4 | 3 | 1 | 8.44 | 2.75 | 7.31 | 3.65 | 8.62 | 3.81 |
| gemma2-2.5M | philadelphia-harrisburg | T | 1 | 5 | 0 | 2 | 5.69 | 5.28 | 2.91 | 5.27 | 1.19 | 5.66 |
| llama32-524k | dallas-austin | T | 5 | 1 | 3 | 0 | 4.38 | 0.09 | 5.59 | 2.44 | 6.06 | 2.44 |
| llama32-524k | houston-austin | T | 7 | 2 | 5 | 1 | 6.3 | 1.27 | 4.77 | 2.76 | 5.27 | 2.96 |
| llama32-524k | chicago-springfield | T | 1 | 7 | 0 | 4 | 4.59 | 1.22 | 3.66 | 5.22 | 3.39 | -1.97 |
| llama32-524k | miami-tallahassee | T | 8 | 6 | 4 | 2 | 1.84 | 9.58 | 5.45 | 7.9 | -6.22 | -1.73 |
| llama32-524k | detroit-lansing | F | 8 | 8 | 5 | 4 | 11.62 | 7.66 | 8.28 | 7.52 | -4.75 | -6.34 |
| llama32-524k | portland-salem | F | 5 | 11 | 3 | 8 | 9.19 | 15.09 | 6.95 | 9.73 | -3.06 | -2.91 |
| llama32-524k | cleveland-columbus | T | 8 | 9 | 6 | 7 | 8.41 | 10.94 | 6.35 | 7.48 | -1.84 | -3.81 |
| llama32-524k | philadelphia-harrisburg | F | 8 | 6 | 3 | 3 | 3.5 | 8.02 | 6.41 | 7.05 | -8.75 | -0.05 |

*Observations.* (i) Upstream recruitment is real but **prompt-dependent**: gemma
dallas 0→2, houston 0→3, miami 0→8 gain upstream features when unfrozen, while
others are flat or drop (portland 3→1). Aggregate gemma upstream roughly doubles
(§10.3) but the per-prompt spread is large — argues for more prompts + CIs. (ii)
Unfrozen can **destabilize**: chicago-2.5M range goes 3.14→−2.91 and the set
balloons to 17, portland-2.5M faith 3.34→5.94 but range goes negative — unfreezing
sometimes *creates* a reconstruction failure by collapsing `empty`. (iii) gemma
frozen `suff` is often ~0 (dallas 0.0, houston 0.0, miami 0.03) while faith is
positive: frozen sufficiency (keep-only − empty) is near zero because frozen
attention already reconstructs the behavior from almost nothing — the necessity term
carries the frozen faithfulness. This is the cleanest single illustration of the
"frozen hides features" claim.

### C.4 ACDC Game 1 frozen vs unfrozen (`macag_acdc_unfrozen/acdc_frozen_vs_unfrozen.csv`)

> *Source note (pass 1).* The unfrozen columns were selected with the pre-fix
> (λ-penalized) `raw_relative` stop (§10 provenance box): the `range` columns and
> flip counts are stop-independent and robust; `faith_u` / `E_u` / `up_u` are
> provisional. Unfrozen legs ran budget 20 / prefilter 30 vs frozen 8 / 20.
> <!-- TODO(pass-2): regenerate with the fixed stop and matched budgets. -->

| CLT | task | slug | range_f | range_u | faith_f | faith_u | E_f | E_u | up_f | up_u |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| gemma2-426k | docstring | docstring_01 | 2.5 | 25.44 | 7.06 | 13.62 | 1 | 3 | 0 | 2 |
| gemma2-426k | docstring | docstring_02 | -1.56 | 22.75 | 8.03 | 13.12 | 8 | 2 | 3 | 0 |
| gemma2-426k | docstring | docstring_03 | -8.56 | 23.38 | 3.84 | 14.53 | 8 | 2 | 5 | 1 |
| gemma2-426k | IOI | ioi_01 | -8.12 | 7.25 | 5.69 | 7.31 | 8 | 3 | 7 | 2 |
| gemma2-426k | IOI | ioi_02 | -18.5 | 6.38 | 8.69 | 5.94 | 6 | 4 | 6 | 3 |
| gemma2-426k | IOI | ioi_03 | -12.88 | 17.31 | 6.19 | 12.16 | 8 | 3 | 7 | 2 |
| gemma2-426k | IOI | ioi_04 | -9.88 | 7.83 | 12.31 | 6.88 | 8 | 2 | 8 | 1 |
| gemma2-426k | IOI | ioi_05 | -15.5 | 7.47 | 4.19 | 13.05 | 6 | 7 | 4 | 6 |
| gemma2-426k | IOI | ioi_06 | -6.5 | 0.06 | 8.06 | 4.5 | 8 | 3 | 8 | 2 |
| gemma2-426k | IOI | ioi_07 | -12.25 | 16.12 | 8.38 | 7.84 | 8 | 2 | 8 | 1 |
| gemma2-426k | IOI | ioi_08 | -19.12 | 16.19 | 10.75 | 10.88 | 4 | 3 | 4 | 2 |
| gemma2-426k | IOI | ioi_09 | -24.12 | 3.62 | 4.94 | 7.22 | 8 | 2 | 7 | 1 |
| gemma2-426k | IOI | ioi_10 | -6.75 | -0.25 | 8.75 | 10.44 | 5 | 7 | 5 | 7 |
| gemma2-2.5M | docstring | docstring_01 | -8.0 | 25.19 | 6.34 | 15.03 | 5 | 3 | 2 | 2 |
| gemma2-2.5M | docstring | docstring_02 | -0.25 | 27.0 | 8.44 | 17.06 | 8 | 3 | 4 | 2 |
| gemma2-2.5M | docstring | docstring_03 | -4.19 | 18.41 | 5.16 | 10.61 | 8 | 3 | 2 | 1 |
| gemma2-2.5M | IOI | ioi_01 | -8.25 | 4.19 | 12.5 | 1.81 | 8 | 6 | 8 | 6 |
| gemma2-2.5M | IOI | ioi_02 | -24.0 | -3.5 | 7.12 | 2.94 | 4 | 4 | 3 | 2 |
| gemma2-2.5M | IOI | ioi_03 | -22.12 | 8.62 | 6.44 | 15.31 | 7 | 11 | 7 | 9 |
| gemma2-2.5M | IOI | ioi_04 | -10.75 | 8.25 | 11.5 | 7.03 | 7 | 3 | 7 | 2 |
| gemma2-2.5M | IOI | ioi_05 | -23.25 | 3.88 | 5.69 | 9.38 | 8 | 5 | 7 | 4 |
| gemma2-2.5M | IOI | ioi_06 | -10.44 | 2.38 | 8.78 | 7.69 | 8 | 7 | 7 | 6 |
| gemma2-2.5M | IOI | ioi_07 | -27.5 | 11.19 | 6.75 | 5.97 | 8 | 8 | 8 | 5 |
| gemma2-2.5M | IOI | ioi_08 | -13.75 | 13.5 | 13.81 | 9.06 | 8 | 2 | 8 | 1 |
| gemma2-2.5M | IOI | ioi_09 | -25.62 | 3.75 | 6.12 | 4.72 | 7 | 5 | 7 | 4 |
| gemma2-2.5M | IOI | ioi_10 | -10.5 | 0.75 | 10.94 | 5.72 | 8 | 6 | 8 | 4 |
| llama32-524k | docstring | docstring_01 | 27.28 | 22.44 | 12.81 | 15.16 | 8 | 4 | 5 | 1 |
| llama32-524k | docstring | docstring_02 | 17.66 | 15.5 | 16.2 | 16.22 | 8 | 9 | 4 | 5 |
| llama32-524k | docstring | docstring_03 | 17.05 | 21.19 | 6.88 | 17.41 | 8 | 7 | 7 | 4 |
| llama32-524k | IOI | ioi_01 | 12.5 | 4.31 | 4.8 | 5.34 | 8 | 5 | 6 | 3 |
| llama32-524k | IOI | ioi_02 | 14.88 | 6.64 | 8.91 | 9.23 | 8 | 7 | 8 | 5 |
| llama32-524k | IOI | ioi_03 | 10.58 | 8.12 | 6.52 | 5.03 | 8 | 3 | 7 | 3 |
| llama32-524k | IOI | ioi_04 | 11.5 | 11.38 | 4.66 | 7.87 | 4 | 6 | 4 | 5 |
| llama32-524k | IOI | ioi_05 | 9.78 | 12.12 | 6.62 | 8.33 | 8 | 4 | 7 | 3 |
| llama32-524k | IOI | ioi_06 | 8.58 | 6.78 | 3.9 | 4.14 | 8 | 3 | 8 | 2 |
| llama32-524k | IOI | ioi_07 | 12.81 | 6.17 | 5.59 | 5.99 | 8 | 5 | 8 | 4 |
| llama32-524k | IOI | ioi_08 | 12.44 | 8.52 | 6.54 | 8.01 | 8 | 6 | 4 | 5 |
| llama32-524k | IOI | ioi_09 | 10.25 | 8.47 | 7.59 | 6.92 | 8 | 3 | 8 | 2 |
| llama32-524k | IOI | ioi_10 | 7.22 | 5.48 | 4.64 | 5.59 | 7 | 3 | 7 | 2 |

*Observations.* (i) The gemma flip is dramatic and consistent: every gemma IOI row
has strongly negative `range_f` (−6 to −27.5) → positive `range_u` except
ioi_02-2.5M (−3.5) and the two ioi_10/ioi_06 near-zero cases. The magnitude of the
frozen negativity (e.g. −27.5) is itself a measure of *how* attention-mediated the
task is. (ii) docstring is **less** attention-mediated than IOI even on gemma
(docstring_01-426k already positive frozen at +2.5), consistent with docstring being
a more "local"/feature-carried completion. (iii) llama is positive throughout
(frozen ranges +7 to +27) — its CLT genuinely places IOI behavior in features; this
is the cross-model control that makes the gemma result a *diagnosis* rather than a
universal artifact. (iv) Unfreezing **shrinks** evidence and upstream count on gemma
(e.g. ioi_04 8→2 features, 8→1 upstream) — once attention recomputes, fewer features
are needed, the opposite of the two-hop sweep where unfreezing *recruited* features.
This sign difference between tasks is worth a dedicated paragraph in the paper.

### C.5 Game 2 contrastive, frozen vs unfrozen (`macag_unfrozen_raw/game2_frozen_vs_unfrozen.csv`)

<!-- TODO(pass-2): re-verify after re-run; predates best-iterate tracking (all runs converged in 2 rounds, so low risk). -->
All 24 prompts × {frozen, unfrozen}: **overlap_f = overlap_u = 0.0** and
**shared = 0** in every row. The unique-set sizes (|E_y|, |E_foil|) shift slightly
between frozen/unfrozen (e.g. gemma2-2.5M dallas |E_foil| 8→4, cleveland 8→4) but the
disjointness never breaks. Full row-level numbers are in the CSV; the single fact to
carry into the paper is **48/48 runs at overlap 0.0**.

### C.6 Oracle cost (from `*/macag_game{1,2}.json` stats)

Two-hop gemma-426k: Game 1 mean **807.5** oracle calls (range reflects budget 8 ×
prefilter 20 + full singleton scan over ~325 candidates), Game 2 mean **1761.25**
(the runs converged after 2 of the configured 4 ABR rounds, two greedy solves per
round). Cache hit counts exceed oracle calls (e.g. one prompt:
762 calls / 838 hits for G1; 1756 / 5596 for G2), i.e. **>50% of probes are served
from cache** — the memoization in [§2.1](#21-oracle-scoring) is load-bearing. These
are the denominators for the "MACAG vs Shapley cost ratio" claim once Shapley is run
(Phase 2). *Caveat:* these stats predate the per-solve counter reset (`71a2ef6`), so
Game 2 counters on a shared oracle may include residue from the preceding Game 1
solve — re-derive on re-run. <!-- TODO(pass-2): recompute cost stats from the re-run. -->

### C.7 Nonlinear benchmark: matched-protocol baselines + frozen/unfrozen (`results/macag_nonlinear_connected/`)

> *Source note.* First **post-`71a2ef6`** run (fixed `raw_relative` stop, per-solve
> oracle-stat reset, Game-2 best-iterate) and the first with **real baseline
> numbers**. Single load via `run_macag game1 --freeze-mode both`; both legs matched
> at budget 8 / prefilter 20 / `raw_relative` / `connected=True` (no 8/20-vs-20/30
> confound of C.3–C.5). 60 prompts = 4 task families × 5 prompts × 3 CLTs. `faith_f`
> / `range_f` / `n` are the **frozen** Game-1 leg; `verdict`/`flip` from the
> per-prompt `attention_mediation` block. Baseline columns are faith@budget-8 from
> `baselines.csv`. **Statistics:** means carry 95% bootstrap CIs over prompts
> (10 000 resamples) and Game 1 vs each matched baseline is tested with paired
> Wilcoxon signed-rank (Holm-corrected, rank-biserial $r$); the prompt is the
> resampling unit because the non-Shapley selectors are deterministic per graph.
> Single seed on the CLT/graph; uncertainty is over the 60-prompt sample. Reproduce
> with `scripts/macag_bootstrap_wilcoxon.py`.

**Aggregate baseline head-to-head (n=60, all prompts).** Influence and EAP run at the
graph layer (0 oracle calls); Shapley/Game 1/ACDC use the real ReplacementModel
oracle. ACDC is uncapped (mean $k\approx193$) and is **excluded from the paired tests**
(not budget-matched); all others budget-8.

| selector | faith@8 [95% CI] | $|E|$ (k) | faith/feat [95% CI] | AUC | oracle calls | prec@k | Jaccard | vs G1: Δ̃ | win/60 | $p$ (Holm) | $r$ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| top-k influence | 0.08 [−0.02, 0.20] | 8.0 | 0.010 [−0.003, 0.025] | 0.03 | 0 | 0.004 | 0.002 | +4.95 | 60/60 | <1e‑9 | 1.00 |
| EAP (graph-derived) | 1.15 [0.55, 1.83] | 8.0 | 0.144 [0.065, 0.226] | 0.81 | 0 | 0.206 | 0.128 | +3.72 | 58/60 | <1e‑9 | 0.98 |
| MC Shapley (gold) | 4.72 [4.02, 5.45] | 8.0 | 0.590 [0.500, 0.683] | 3.48 | 32 495 | — | — | +1.00 | 44/60 | 8.2e‑5 | 0.60 |
| **MACAG Game 1** | **5.50 [4.88, 6.15]** | **6.8** | **0.859 [0.747, 0.977]** | **4.28** | **718** | **0.46** | **0.33** | — | — | — | — |
| ACDC (τ-prune) | 10.50 [8.60, 12.46] | 193.4 | 0.067 [0.050, 0.088] | — | 2 612 | — | — | *(not matched)* | — | — | — |

prec@k / Jaccard are vs the gold Shapley top-k; Δ̃ = median paired difference (Game 1
− baseline). **Cost (the unimpeachable claim):** Game 1 uses **44.7× fewer** oracle
calls than Shapley, 95% CI [43.5, 45.8], cheaper on **60/60** prompts, one-sided
Wilcoxon $p<10^{-9}$. **Faith/feature:** Game 1 0.859 [0.747, 0.977] vs Shapley
0.590 [0.500, 0.683] with
**non-overlapping CIs** — and unlike raw faith it cannot be inflated by spending
features (ACDC's 0.067 at k≈193 is the cautionary case). **Raw faith\@8:** Game 1's
median paired advantage over Shapley is +1.00 ($p=8.2\times10^{-5}$, $r=0.60$) but the
win-rate is **44/60**, so the claim is "higher on most prompts," not "uniformly," and
it is on Game 1's own greedy objective (§11.3). ACDC's raw-faith "wins" (46/60) are at
~28× the feature budget. On the 50 target-preferred prompts every effect strengthens
(Game 1 faith 5.96, fpf 0.938, vs-Shapley Δ̃ +1.08 at $p=9.6\times10^{-5}$, cost 44.8×).

**Verdict mix by task** (per-prompt frozen-vs-unfrozen `attention_mediation.verdict`):

| task | feature_mediated | indeterminate | attention_mediated |
|---|:-:|:-:|:-:|
| boolean_logic | 7 | 7 | 1 |
| negation_polarity | 12 | 3 | 0 |
| context_polysemy | **15** | 0 | 0 |
| hard_semantic_foil | 8 | 2 | 5 |
| **total** | **42** | **12** | **6** |

Only **6/60 range-flip** (frozen `range`<0 → unfrozen ≥0); they are:

| CLT | task | slug | range_f → range_u | faith_f → faith_u | $|E|$_f → $|E|$_u |
|---|---|---|--:|--:|--:|
| gemma2-426k | hard_semantic_foil | sem_03 | −9.00 → 8.00 | 5.88 → 2.56 | 4 → 2 |
| gemma2-426k | hard_semantic_foil | sem_04 | −4.28 → 2.13 | 2.58 → 2.25 | 7 → 5 |
| gemma2-426k | hard_semantic_foil | sem_05 | −5.38 → 1.31 | 4.31 → 6.16 | 2 → 7 |
| gemma2-2.5M | hard_semantic_foil | sem_03 | −11.38 → 2.52 | 7.31 → 1.82 | 5 → 4 |
| gemma2-2.5M | hard_semantic_foil | sem_04 | −4.47 → 3.50 | 4.64 → 3.69 | 6 → 4 |
| llama32-524k | boolean_logic | bool_01 | −1.45 → 0.38 | 3.12 → 2.82 | 8 → 4 |

**Frozen → unfrozen aggregates (per CLT).** Unfreezing **shrinks** evidence and
upstream count on *every* CLT here — opposite of the C.3 two-hop sweep, same as the
C.4 IOI sweep:

| CLT | upstream f→u | $|E|$ f→u | range f→u | faith f→u |
|---|--:|--:|--:|--:|
| gemma2-426k | 3.45 → 2.95 | 5.40 → 4.40 | 2.87 → 5.22 | 4.48 → 3.83 |
| gemma2-2.5M | 3.30 → 2.80 | 5.45 → 4.35 | 3.94 → 5.06 | 5.28 → 3.86 |
| llama32-524k | 3.60 → 2.85 | 5.25 → 4.40 | 6.34 → 4.47 | 5.84 → 4.38 |
| **all** | **3.45 → 2.87** | **5.37 → 4.38** | **4.38 → 4.91** | **5.20 → 4.02** |

**Game 2 (contrastive), ABR vs FP** (`abr_vs_fp.csv`): all 60 converge in 2
iterations under both solvers; **overlap = 0.0 in every row** for both. ABR≈FP:
target-set Jaccard 0.63, foil-set 0.55; target utility 5.50 (ABR) vs 5.45 (FP); mean
$|E_y|$ 6.80/6.92, $|E_{\text{foil}}|$ 7.28/7.02. Combined with the 60 frozen Game-1
verdicts → **180/180 disjoint** target/foil sets (extends C.5's 48/48).

**Full per-prompt table** (frozen Game-1 + baseline faith@8; ✱ = range-flip;
faith/feat best in **bold** is Game 1; full 40-column data in `baselines.csv`):

| CLT | task | slug | pref | faith_f | range_f | n | verdict | flip | infl | eap | shap | **G1** | acdc (k) |
|---|---|---|:-:|--:|--:|--:|---|:-:|--:|--:|--:|--:|--:|
| gemma2-426k | bool | bool_01 | F | 2.22 | 2.48 | 5 | feat |  | 0.32 | 1.58 | 4.01 | **2.61** | 10.39 (221) |
| gemma2-426k | bool | bool_02 | F | 4.76 | 3.78 | 6 | feat |  | 0.03 | 1.86 | 4.48 | **5.12** | 4.23 (8) |
| gemma2-426k | bool | bool_03 | T | 5.62 | 0 | 7 | indet |  | 0.03 | 0.72 | 3.66 | **5.75** | 3.61 (213) |
| gemma2-426k | bool | bool_04 | T | 2.49 | −2.52 | 3 | indet |  | 0.30 | −1.17 | 2.92 | **2.61** | 1.44 (206) |
| gemma2-426k | bool | bool_05 | F | 4.33 | 3.16 | 4 | feat |  | 0.17 | 1.88 | 3.11 | **4.23** | 9 (217) |
| gemma2-426k | neg | neg_01 | T | 6.09 | 3.31 | 7 | feat |  | 0.25 | 0.56 | 3.16 | **6.09** | 7.20 (162) |
| gemma2-426k | neg | neg_02 | T | 4.44 | 2.88 | 2 | feat |  | −0.31 | 3.22 | 3.44 | **4.88** | 7.59 (140) |
| gemma2-426k | neg | neg_03 | T | 7.19 | 2.38 | 7 | feat |  | −0.23 | 0.22 | 6.03 | **7.25** | 9.09 (119) |
| gemma2-426k | neg | neg_04 | F | 2.89 | −8.47 | 8 | indet |  | −0.48 | −3.25 | 0.25 | **2.38** | −0.38 (185) |
| gemma2-426k | neg | neg_05 | T | 7.23 | 3.33 | 4 | feat |  | −1.51 | −1.35 | 5.65 | **7.87** | 9.18 (203) |
| gemma2-426k | poly | poly_01 | T | 2.81 | 7.38 | 2 | feat |  | 0.19 | 1.78 | 5.28 | **7.12** | 12.84 (187) |
| gemma2-426k | poly | poly_02 | T | 3.88 | 24.25 | 8 | feat |  | 0.25 | 6.50 | 11.19 | **3.70** | 28.41 (193) |
| gemma2-426k | poly | poly_03 | T | 5.22 | 5.25 | 2 | feat |  | −0.09 | −1.28 | 3.47 | **5.66** | 14.67 (260) |
| gemma2-426k | poly | poly_04 | T | 3.81 | 12.38 | 8 | feat |  | 0.19 | 0.91 | 1.69 | **3.81** | 17.50 (298) |
| gemma2-426k | poly | poly_05 | T | 4.66 | 8.69 | 6 | feat |  | 0 | 2.53 | 6.75 | **4.78** | 22.09 (244) |
| gemma2-426k | sem | sem_01 | T | 4.88 | 2.75 | 8 | feat |  | 0.17 | −1.48 | 5.11 | **5.02** | 9.33 (164) |
| gemma2-426k | sem | sem_02 | T | 4.23 | 5.12 | 8 | feat |  | −0.12 | 0.56 | 3.66 | **4.09** | 9.53 (248) |
| gemma2-426k | sem | sem_03 | T | 5.88 | −9 | 4 | attn | ✱ | 0.84 | 0.44 | 7.25 | **7.56** | 2.12 (187) |
| gemma2-426k | sem | sem_04 | T | 2.58 | −4.28 | 7 | attn | ✱ | −0.17 | −0.56 | 2.81 | **3.09** | 1.86 (263) |
| gemma2-426k | sem | sem_05 | T | 4.31 | −5.38 | 2 | attn | ✱ | 0.19 | −0.50 | 2.81 | **5.62** | −0.59 (280) |
| gemma2-2.5M | bool | bool_01 | F | 2.06 | 2.81 | 8 | feat |  | −0.16 | 1.81 | 3.25 | **2** | 7.97 (186) |
| gemma2-2.5M | bool | bool_02 | F | 3.26 | 5.50 | 8 | feat |  | −0.02 | 2.75 | 4.28 | **1.94** | 9.09 (193) |
| gemma2-2.5M | bool | bool_03 | T | 1.62 | −5.62 | 3 | indet |  | 0.20 | −0.72 | 0.44 | **1.88** | −1.66 (190) |
| gemma2-2.5M | bool | bool_04 | T | 4.09 | −3.62 | 2 | indet |  | 0.12 | −1.84 | 3.78 | **4.31** | 0.47 (182) |
| gemma2-2.5M | bool | bool_05 | F | 4.31 | 7.25 | 5 | feat |  | 0.05 | 2.97 | 4.33 | **4.97** | 9.61 (210) |
| gemma2-2.5M | neg | neg_01 | T | 6.25 | 8.25 | 2 | feat |  | 0.03 | 2.30 | 4.88 | **6.86** | 11.05 (154) |
| gemma2-2.5M | neg | neg_02 | T | 5.78 | 10.69 | 2 | indet |  | 0.03 | 5.22 | 5.66 | **6.47** | 12.88 (151) |
| gemma2-2.5M | neg | neg_03 | T | 8.81 | 0.88 | 8 | feat |  | −0.06 | 0.31 | 2.38 | **7.44** | 6.09 (151) |
| gemma2-2.5M | neg | neg_04 | F | 3.45 | −3.97 | 4 | indet |  | −0.16 | −4.20 | 1.55 | **3.18** | 1.02 (229) |
| gemma2-2.5M | neg | neg_05 | T | 9.70 | 4.22 | 8 | feat |  | 0.50 | −1.83 | 6.16 | **9.03** | 10.16 (217) |
| gemma2-2.5M | poly | poly_01 | T | 4.16 | 7.31 | 7 | feat |  | 0.25 | 1.09 | 4.75 | **3.94** | 11.88 (195) |
| gemma2-2.5M | poly | poly_02 | T | 2.58 | 16.50 | 6 | feat |  | 0.06 | −0.99 | −0.02 | **2.32** | 22.73 (255) |
| gemma2-2.5M | poly | poly_03 | T | 6.16 | 6.31 | 3 | feat |  | −0.34 | 0.19 | 3.09 | **6.62** | 12.19 (294) |
| gemma2-2.5M | poly | poly_04 | T | 3.41 | 12.38 | 8 | feat |  | 0.09 | −0.72 | 1.69 | **3.56** | 16.66 (314) |
| gemma2-2.5M | poly | poly_05 | T | 11.22 | 16.38 | 8 | feat |  | 0.03 | 3.22 | 11.06 | **11.06** | 24.16 (252) |
| gemma2-2.5M | sem | sem_01 | T | 7.96 | 8.98 | 6 | feat |  | −0.13 | −0.74 | 7.09 | **8.57** | 14.99 (253) |
| gemma2-2.5M | sem | sem_02 | T | 4.69 | 3.50 | 7 | feat |  | −0.19 | −1.33 | 4.80 | **4.80** | 7.32 (252) |
| gemma2-2.5M | sem | sem_03 | T | 7.31 | −11.38 | 5 | attn | ✱ | −0.19 | 1.06 | 5.19 | **7.12** | −1.91 (234) |
| gemma2-2.5M | sem | sem_04 | T | 4.64 | −4.47 | 6 | attn | ✱ | 0.08 | 0.36 | 1.55 | **3.48** | −0.95 (282) |
| gemma2-2.5M | sem | sem_05 | T | 4.19 | −3.19 | 3 | indet |  | −0.03 | 1.50 | 2.28 | **4.22** | 0.03 (252) |
| llama32-524k | bool | bool_01 | F | 3.12 | −1.45 | 8 | attn | ✱ | −0.11 | −0.58 | 2 | **3.05** | 3.36 (143) |
| llama32-524k | bool | bool_02 | T | 4.47 | 2.28 | 5 | feat |  | 0.64 | 0.56 | 3.59 | **4.48** | 4.66 (162) |
| llama32-524k | bool | bool_03 | T | 3.52 | 2.27 | 3 | indet |  | −0.23 | 1.86 | 3.84 | **4.39** | 6.47 (145) |
| llama32-524k | bool | bool_04 | T | 3.64 | 1.81 | 2 | indet |  | 0.65 | 0.51 | 2.81 | **3.76** | 6.98 (125) |
| llama32-524k | bool | bool_05 | F | 2.09 | −1.38 | 6 | indet |  | 0.40 | −0.36 | 1.31 | **2.58** | 4.70 (135) |
| llama32-524k | neg | neg_01 | T | 5.71 | 7.41 | 4 | feat |  | −0.21 | 4.61 | 6.49 | **6.80** | 14.41 (94) |
| llama32-524k | neg | neg_02 | T | 6.17 | 4.56 | 5 | feat |  | 2.14 | 4.55 | 6.49 | **6.64** | 9.30 (87) |
| llama32-524k | neg | neg_03 | T | 11.94 | 7.50 | 8 | feat |  | 0.76 | 9.03 | 10.12 | **11.59** | 14.56 (81) |
| llama32-524k | neg | neg_04 | T | 7.38 | 7.06 | 5 | feat |  | 0.23 | 4.66 | 7.28 | **7.48** | 11.81 (132) |
| llama32-524k | neg | neg_05 | T | 5.80 | 5.62 | 4 | feat |  | 0.36 | 3.23 | 6.97 | **6.12** | 10.22 (126) |
| llama32-524k | poly | poly_01 | T | 4.70 | 6.91 | 7 | feat |  | −0.05 | −0.22 | 3.33 | **5.38** | 18.56 (158) |
| llama32-524k | poly | poly_02 | T | 9.06 | 12.94 | 8 | feat |  | 0.02 | −0.18 | 8.87 | **10.20** | 21.73 (180) |
| llama32-524k | poly | poly_03 | T | 10.31 | 15.25 | 3 | feat |  | −0.03 | 8.06 | 12.87 | **12.55** | 28.80 (235) |
| llama32-524k | poly | poly_04 | T | 2.29 | 5.70 | 8 | feat |  | −0.14 | 1.25 | 2.98 | **2.11** | 21.25 (255) |
| llama32-524k | poly | poly_05 | T | 3.69 | 8.54 | 5 | feat |  | 0.22 | 0.12 | 1.27 | **4.37** | 20.59 (204) |
| llama32-524k | sem | sem_01 | T | 7.14 | 10.92 | 4 | feat |  | 0.88 | −0.15 | 7.35 | **9.01** | 19.35 (146) |
| llama32-524k | sem | sem_02 | T | 8.62 | 12.80 | 7 | feat |  | −0.55 | 5.95 | 8.20 | **8.80** | 21.25 (190) |
| llama32-524k | sem | sem_03 | T | 6.32 | 9.81 | 5 | indet |  | 0.22 | 2.05 | 11.23 | **6.94** | 16.17 (167) |
| llama32-524k | sem | sem_04 | T | 2.75 | 2.38 | 5 | feat |  | −0.25 | 0.27 | 2.62 | **2.62** | 8.75 (232) |
| llama32-524k | sem | sem_05 | T | 8.11 | 5.88 | 3 | feat |  | −0.25 | 0.41 | 8.50 | **8.16** | 13.94 (162) |

*Observations.* (i) **Game 1 is the efficient frontier.** Across the 60 rows its
faith\@8 tracks or beats Shapley-gold at a fraction of the cost, and it dominates
faith-per-feature; ACDC's larger raw faith always comes with a 100–300-feature set
(the `acdc (k)` column), so it is not a budget-matched competitor — it is the
"spend everything" ceiling. (ii) **Influence ≈ noise** (column `infl` hovers near 0,
often negative), the clearest possible answer to A.3's "is search needed?" — yes.
(iii) **`context_polysemy` is the clean feature-mediated family** (15/15, no flips, no
indeterminates) — the bank/bat/spring polysemy is carried by features, exactly the
nonlinear-context case the benchmark targets. (iv) **`hard_semantic_foil` on gemma is
where attention mediates** (5 of 6 flips); the magnitude of frozen negativity (sem_03:
−9 to −11) again measures *how* attention-mediated a prompt is, mirroring §10.4. (v)
**`boolean_logic` is the soft spot**: 7/15 indeterminate and the home of 5 of the 10
not-target-preferred (`pref=F`) prompts — when the model does not prefer the target
the logit-gap oracle is ill-posed, so those rows should be excluded from faithfulness
aggregates (the Phase-1 protocol note, now reproduced on a second benchmark).

---

## Appendix D: Background and Preliminaries

Everything a reader needs to follow the work, from the transformer up to the
attribution graph MACAG consumes. *(When composing the paper this becomes the
"Background" section; here it is an appendix to avoid renumbering.)*

### D.1 Transformer internals MACAG touches

A decoder-only transformer maintains a **residual stream** $x^\ell \in \mathbb{R}^d$
at each layer $\ell$ and token position, updated additively by attention and MLP
sublayers: $x^{\ell+1} = x^\ell + \text{Attn}^\ell(x^\ell) + \text{MLP}^\ell(x^\ell)$.
Two facts matter here: (i) the residual stream is a **linear sum** of component
outputs, so a "direction" in it is meaningful and ablations compose additively; (ii)
**attention** mixes information *across token positions*, which is the entire reason
the frozen/unfrozen distinction ([§2.3](#23-attention-freezing-and-the-error-floor))
exists — freezing attention pins that cross-position routing to its clean value.

### D.2 Sparse dictionary learning and "features"

The polysemantic-neuron problem (single neurons fire for many unrelated concepts)
motivates **sparse autoencoders (SAEs)**: learn an overcomplete dictionary so that
activations decompose into a sparse set of more interpretable **features**
(Bricken et al. 2023, *Towards Monosemanticity*; Cunningham et al. 2023; scaled in
Templeton et al. 2024). A "feature" in this document is one such learned dictionary
direction — the unit MACAG selects over.

### D.3 Transcoders and Cross-Layer Transcoders (CLTs)

A **transcoder** is an SAE-like module that does not reconstruct its own input but
**predicts the MLP output from the MLP input** through a sparse feature bottleneck
(Dunefsky et al. 2024). A **Cross-Layer Transcoder (CLT)** generalizes this: each
feature **reads from one layer but writes to all downstream layers** via separate
decoder matrices, which is what lets a single feature participate in a multi-layer
circuit (Anthropic's circuit-tracing line, Ameisen et al. / Lindsey et al. 2025).
The encoder is linear in the standard CLT (and nonlinear in the project's Spline-CLT
variant — not used in this case study). Sparsity comes from a **JumpReLU**
activation: a ReLU with a learned per-feature threshold that hard-gates small
pre-activations to zero, giving an $L_0$-like sparsity without shrinking the
surviving activations.

### D.4 The ReplacementModel and error nodes

Circuit-tracer's **ReplacementModel** swaps the model's MLPs for the (C)LT so that
computation flows through named, ablatable feature units while staying behaviorally
close to the original model. Because the transcoder does not reconstruct the MLP
output perfectly, the residual is captured as a per-layer **error node** — an
unmodelled term added back so the replacement model matches the original. Error nodes
are, by default, **never ablated**: that is precisely why $S_{\text{empty}}$ (all
features off) is not zero but an **error floor**, the root of the
`recoverable_range` story in [§2.3](#23-attention-freezing-and-the-error-floor).

### D.5 Attribution graphs

Running attribution over the ReplacementModel for a prompt yields an **attribution
graph**: nodes are active feature instances (per layer/position), plus error,
embedding/token, and logit nodes; **edges are local linear attribution scores** —
essentially gradient×activation between nodes, a first-order estimate of how much one
node's activation pushes another. A node's **influence** is an aggregate of its edge
weights. This graph is MACAG's input $G=(C,A)$: MACAG takes the feature nodes as the
candidate set $C$ and *re-scores* them causally.

### D.6 Interventions: ablation and patching

To measure causal effect one **intervenes** on a node and re-runs the model:
- **Zero ablation** (MACAG default): set the feature activation to 0.
- **Mean / resample ablation**: replace with a dataset mean or a value from a
  *corrupted* prompt (ACDC's choice). Resampling keeps activations on-distribution
  but defines effect relative to a baseline distribution rather than absolute zero.
- **Freeze attention**: hold attention patterns at clean values during the
  intervention so only the direct feature contribution is removed
  ([§2.3](#23-attention-freezing-and-the-error-floor)).
MACAG's four oracle modes (all/empty/keep-only/remove, [§2.1](#21-oracle-scoring))
are specific ablation patterns over the candidate set.

### D.7 The behavioral metric: logit gap

For a target token $y$ and a competing foil $y_{\text{foil}}$, the **logit gap**
$S = \text{logit}(y) - \text{logit}(y_{\text{foil}})$ is the scalar MACAG scores by
default; alternatives are raw logit, probability, and negative cross-entropy. The gap
is the standard contrastive behavioral signal in circuit work (it cancels prompt-wide
shifts and isolates the target-vs-foil decision). It requires a sensible foil and a
single-token target — hence the `greater_than` first-token-collision caveat (B0.1).

### D.8 Circuits and faithfulness

A **circuit** is a subgraph hypothesized to implement a behavior. A circuit is
**faithful** if intervening on it reproduces the model's behavior — operationalized
here as sufficiency (keep only the circuit) and necessity (remove the circuit).
MACAG's contribution is to *search for* and *score* such subgraphs over CLT features
by intervention, rather than reading them off attribution magnitude.

---

## Appendix E: Extended Related Work

Grouped by what each line contributes relative to MACAG. (Citations are by
author/year; full entries collected in §References. Where a precise detail is not
load-bearing it is stated generally.)

### E.1 Features and dictionary learning

Superposition and polysemanticity (Elhage et al. 2022, *Toy Models of
Superposition*) motivate sparse dictionaries; SAEs recover monosemantic features
(Bricken et al. 2023; Cunningham et al. 2023) and scale to frontier models
(Templeton et al. 2024). **Relation to MACAG:** these produce the *units*; MACAG is
agnostic to which dictionary method made them — it evaluates the circuit they form.

### E.2 Transcoders, CLTs, and circuit tracing

Transcoders predict MLP outputs through a sparse bottleneck (Dunefsky et al. 2024);
cross-layer transcoders and the attribution-graph pipeline come from the
circuit-tracing work (Ameisen et al. 2025, *Circuit Tracing*; Lindsey et al. 2025,
*On the Biology of a Large Language Model*). **Relation:** this is the upstream that
*builds* MACAG's input graph; MACAG is the downstream causal evaluator, complementary
to circuit-tracer's attribution-magnitude pruning.

### E.3 Automated circuit discovery

ACDC (Conmy et al. 2023) prunes the native computational graph top-down via patching
against a KL threshold. **Edge/attribution patching** (Nanda 2023; Syed et al. 2023,
*Attribution Patching Outperforms ACDC*) approximates patching effects with a single
backward pass — fast but first-order. Related threads: edge pruning, subnetwork
probing, and head-level manual circuits (Wang et al. 2022, *IOI*; the greater-than
circuit). **Relation:** Game 1 shares the minimal-faithful-circuit goal but works
bottom-up over *transcoder features* with an explicit sparsity-penalized utility and
scores sufficiency as well as necessity ([§1.3](#13-relation-to-prior-work)); these
methods are MACAG's primary baselines (Appendix A). Game 2 has no analog here.

### E.4 Causal faithfulness evaluation

Causal scrubbing (Chan et al. 2022) and activation-patching methodologies formalize
"does this hypothesis explain the behavior under intervention?" **Relation:** MACAG
adopts the intervention-as-ground-truth stance but turns it into an *optimization over
which nodes to keep*, with sufficiency/necessity as the objective, and adds the
error-floor-aware normalization for transcoder circuits (which carry error nodes).

### E.5 Game theory for attribution

Shapley values give axiomatic credit (Shapley 1953); SHAP applies them to ML
predictions (Lundberg & Lee 2017); the Banzhaf index is the equal-weight coalition
alternative. Data/neuron variants (Ghorbani & Zou, *Data Shapley*, *Neuron Shapley*)
attribute to training points / neurons. **Relation:** MACAG poses circuit evaluation
as a coalitional game over features ([§3.0](#30-the-underlying-coalitional-game)),
uses Shapley as the *gold* per-feature reference ([§3.6](#36-relation-to-shapley-and-banzhaf-values)),
and contributes the *contrastive two-player* extension (Game 2) absent from prior
attribution-game work.

### E.6 Feature interventions and steering

Activation steering / feature clamping use the same intervention surface MACAG scores
over, for control rather than evaluation. **Relation:** MACAG's evidence sets are
candidate steering targets; clean directions (the decoder stays linear) are what make
both steering and MACAG's keep-only intervention well-defined.

---

## Appendix F: Glossary

Quick definitions of recurring terms (cross-refs to fuller treatment).

- **Feature** — a learned sparse-dictionary direction; MACAG's atomic unit. (D.2)
- **Transcoder / CLT** — module predicting MLP output through a sparse feature
  bottleneck; CLT features read from one layer, write to all later layers. (D.3)
- **JumpReLU** — ReLU with a learned per-feature threshold; the sparsity gate. (D.3)
- **ReplacementModel** — model with MLPs replaced by the (C)LT, enabling feature
  ablation. (D.4)
- **Error node** — per-layer transcoder reconstruction residual; never ablated by
  default ⇒ the **error floor** in $S_{\text{empty}}$. (D.4)
- **Attribution graph** $G=(C,A)$ — feature/error/logit nodes with local-linear
  attribution edges; MACAG's input. (D.5)
- **Influence** — aggregate edge-weight importance of a node in the graph. (D.5)
- **Oracle / scoring oracle** — the function returning $S_\bullet$ under an
  intervention; memoized. (§2.1)
- **Ablation modes** — *all* (none ablated), *empty* (all features ablated),
  *keep-only* (only $E$ kept), *remove* ($E$ ablated). (§2.1)
- **Frozen attention** — attention pinned to clean values during intervention. (§2.3)
- **Logit gap** — target-minus-foil logit; default behavioral score. (D.7)
- **Evidence set $E^\*$** — the node subset MACAG returns. (§4)
- **Sufficiency / Necessity** — keep-only minus empty / all minus remove. (§2.2)
- **Recoverable range** — all minus empty; the normalization denominator; ≤0 ⇒
  behavior not in features. (§2.2–2.3)
- **Upstream feature** — evidence node not at the prediction token (reverse-pos > 0).
  (§2.4)
- **Target-preferred** — model predicts the target over the foil at baseline. (§2.4)
- **Overlap rate** — Jaccard of target/foil evidence (Game 2); 0 = disjoint. (§5.3)
- **Range-flip** — recoverable_range negative frozen → non-negative unfrozen; the
  attention-mediation diagnostic. (§10.4)
- **Attention-mediated vs feature-mediated** — whether a behavior is carried by
  attention routing (range flips) or by features (range positive frozen). (§11)
- **Supernode** — a labeled group of nodes for visualization/annotation. (§8.3)
- **Coalitional game $(N,v)$** — players = features, $v$ = faithfulness. (§3.0)
- **Potential game** — game admitting a scalar $\Phi$ aligning all players'
  incentives; guarantees Game 2 equilibrium existence. (§3.4)
- **Fictitious play (FP)** — Game 2 solver where each agent best-responds to the
  opponent's *empirical history* of evidence sets (expected overlap penalty)
  instead of its last iterate; damps ABR cycling. (§5.2.1)
- **best_iteration** — the solver round whose joint allocation Game 2 returns
  (best combined hard-overlap utility); 0 = the empty allocation won. (§3.4, §5.5)

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

**MACAG Game 1.** Bottom-up **node** selection (the greedy of [§4.2](#42-algorithm-greedy-hill-climbing), condensed).

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
| **Error/floor handling** | none (native graph has no transcoder error term) | explicit **error floor** + recoverable-range normalization ([§2.3](#23-attention-freezing-and-the-error-floor)) |
| **Contrastive variant** | — | **Game 2** (no ACDC analog) |
| **Theory** | greedy threshold heuristic | coalitional game; submodular $(1{-}1/e)$ where it holds; Game 2 = exact potential game ([§3](#3-game-theoretic-foundations)) |
| **Per-feature credit** | — | explicit Shapley/Banzhaf link ([§3.6](#36-relation-to-shapley-and-banzhaf-values)) |
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
   [§3.2](#32-the-value-function-and-submodularity), which ACDC's top-down deletion
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

## Appendix H: Contemporary Methods and Positioning

A "deep-research" report circulated alongside this project argued MACAG is
"obsolete" given newer methods. This appendix is the honest response: each cited
method was located on arXiv/OpenReview, verified to exist, and compared to MACAG in
the same style as ACDC ([§1.3](#13-relation-to-prior-work), [Appendix G](#appendix-g-macag-vs-acdc-algorithmic-differences)).
**Conclusion up front:** all the cited works are real, none renders MACAG obsolete,
and several are *complementary* or *concurrent* (2026) rather than prior art that
defeats novelty. The one paper that overlaps most (Hedonic Neurons) shares MACAG's
coalitional framing but operates on different units and validates — rather than
refutes — the game-theoretic approach.

### H.1 Verification status

Every citation below was checked; author lists corroborate domain experts (e.g.
Yair Zick → hedonic games; Deeparnab Chakrabarty → hitting sets; Guy Katz → neural
verification; Bin Yu → CD-T/SPEX), so these are **not** hallucinated.

| Method | arXiv | Date | Real? | Object of study | Relation to MACAG |
|--------|-------|------|:-----:|-----------------|-------------------|
| **ACDC** (Conmy et al.) | 2304.14997 | 2023 | ✓ | native heads/MLPs, edges | prior art; primary Game 1 baseline |
| **EAP / attribution patching** (Syed; Nanda) | 2310.10348 | 2023 | ✓ | native edges (1st-order) | prior art; the incumbent MACAG improves on |
| **CD-T** (Hsu et al.) | 2407.00886 | 2024 | ✓ | native heads/MLPs | prior art; discovery competitor (analytical) |
| **K-MSHC** (Chowdhary et al.) | 2505.12268 | 2025 | ✓ | native attention heads | prior art; closest to Game 1 (sufficiency) |
| **SPEX** (Kang et al.) | 2502.13870 | 2025 (ICML) | ✓ | **input tokens** | prior art; attribution scaling, not circuits |
| **Hedonic Neurons** (Chowdhury et al.) | 2509.23684 | 2025 | ✓ | MLP **neurons** | closest concurrent; coalitional + synergy |
| **Formal MI** (Hadad, Katz, Bassan) | 2602.16823 | 2026 (ICLR) | ✓ | native components, **vision only** | concurrent; different paradigm |
| **MechRL** (Khadka) | 2605.26343 | 2026 | ✓ | native heads (GPT-2) | concurrent; RL discovery |
| **REdit** (Lei et al.) | 2603.06923 | 2026 | ✓ | weights (editing) | complementary; *uses* circuit overlap |

A structural pattern: **every competitor that does circuit *discovery* operates on
the model's native components (attention heads / MLP neurons), not on
transcoder/SAE features.** MACAG's encoder-agnostic, feature-level evaluation — plus
the contrastive game and the attention-mediation diagnostic — remains an unoccupied
niche.

### H.2 Per-method comparison

**CD-T — Contextual Decomposition for Transformers (Hsu et al., 2024).**
- *What it is:* circuit **discovery** by analytical contextual decomposition — a set
  of closed-form equations recursively isolate each component's contribution; prune
  low-contribution nodes. No iterative ablation. 97% ROC-AUC vs manual IOI circuits,
  hours→seconds.
- *Similar:* finds minimal faithful circuits; fine-grained (head-at-position).
- *Different:* operates on **native heads/MLPs**, not transcoder features; analytical
  (no real interventions) — so it inherits a linearity-style approximation MACAG's
  forward-pass oracle avoids; necessity-style, no contrastive game.
- *Relation:* a fast **discovery** front-end, not a causal **evaluator**. CD-T could
  *produce* a circuit that MACAG then evaluates. Not obsoleting; orthogonal stage.

**K-MSHC — Minimally Sufficient Head Circuits (Chowdhary et al., 2025).**
- *What it is:* stochastic search (Search-K-MSHC) for the minimal set of attention
  heads that is **k-sufficient** (redundantly sufficient) for a task; Gemma-9B,
  syntactic tasks; finds task-specific "super-heads" with low cross-task overlap.
- *Similar:* minimality + **sufficiency** focus (like Game 1's keep-only term); the
  "super-heads, low overlap" finding rhymes with Game 2's disjoint target/foil sets.
- *Different:* **attention heads**, not transcoder features; sufficiency-centric (no
  explicit necessity term or sparsity-penalized utility); no contrastive two-player
  game; no attention-mediation diagnostic.
- *Relation:* the closest head-level analog of Game 1. A natural *baseline to cite*
  and, on shared tasks, to compare against — but at a coarser granularity.

**SPEX — Scaling Feature Interaction Explanations (Kang et al., ICML 2025).**
- *What it is:* scalable **input-feature** interaction attribution via sparse Fourier
  transform + channel decoding; recovers high-order (Banzhaf-type) interactions over
  ~1000 input tokens; +20% output reconstruction vs marginal methods.
- *Similar:* game-theoretic interactions (Banzhaf/Shapley); directly relevant to
  MACAG's Shapley/Banzhaf connection ([§3.6](#36-relation-to-shapley-and-banzhaf-values)).
- *Different:* attributes **input tokens**, not internal circuit nodes — a different
  object entirely. It explains *which inputs interact*, not *which features form the
  circuit*.
- *Relation:* **complementary, and an opportunity** — SPEX is exactly the tool to
  make MACAG's Shapley/Banzhaf baseline *scalable* (it could replace the Monte-Carlo
  estimator in Phase 2). Cite as the scalable-credit method; not a competitor.

**Hedonic Neurons (Chowdhury, Nijasure, Zick, Allan, 2025) — the closest work.**
- *What it is:* models MLP **neurons** as agents in a **hedonic coalitional game**;
  introduces **Pairwise Ablation Synergy (PAS)** to measure non-additive joint
  effects; extracts stable coalitions via **PAC-Top-Cover**; tracks coalitions across
  layers by bipartite matching.
- *Similar:* the **same coalitional-game-theory lens** MACAG uses
  ([§3.0](#30-the-underlying-coalitional-game)); explicitly about synergy — precisely
  the submodularity gap MACAG acknowledges ([§3.2](#32-the-value-function-and-submodularity)).
- *Different:* players are **raw MLP neurons**, not transcoder features; the value is
  **layer-local synergy (PAS)**, not the global target–foil logit gap; it does
  *clustering/coalition formation*, not minimal-faithful-evidence selection or
  contrastive target/foil separation; no attention-mediation diagnostic.
- *Relation:* the **must-cite, must-position** paper. It does **not** obsolete
  MACAG — it answers a different question (which neurons co-act synergistically vs.
  which features causally and contrastively explain a *specific prediction*). It does
  the opposite of refute: it independently validates that coalitional game theory is
  the right language for transformer internals, and its PAS metric is the natural
  remedy for MACAG's greedy synergy blind spot (a concrete Future-Work item: replace
  the singleton prefilter with a PAS-style pairwise pre-scan, §3.2).

**Formal MI — provable circuits (Hadad, Katz, Bassan, ICLR 2026).**
- *What it is:* uses neural-network **verifiers** to certify circuits with provable
  input-domain robustness, patching robustness, and **cardinal minimality** (via a
  blocking-set / minimum-hitting-set duality).
- *Similar:* shares the minimality goal; its patching-robustness directly targets the
  out-of-distribution fragility of zero-ablation that MACAG flags in §2.3.
- *Different (decisive):* **evaluated only on vision models**; verifier-based
  certification does not yet scale to LLM-sized transcoder circuits (hundreds–
  thousands of feature nodes). It certifies small components, not feature coalitions
  over Gemma/Llama.
- *Relation:* a different **paradigm** (certification vs. search), aspirational for
  LLMs. "MACAG is obsolete because Formal MI exists" does not hold: Formal MI does
  not currently operate on the objects or scale MACAG targets. Cite as the rigor
  frontier and as motivation for adding robustness checks; not a current competitor.

**MechRL — RL circuit discovery (Khadka, 2026).**
- *What it is:* a PPO agent over GPT-2's 144 attention heads chooses heads to
  zero-ablate, with a **contrastive reward** = task logit-gap damage minus
  general-LM (cross-entropy) damage; learns transferable structural priors.
- *Similar:* search over an ablation action space; a *contrastive* reward; zero-ablation.
- *Different:* "contrastive" here means **task-vs-general-ability**, not MACAG's
  **target-vs-foil** separation — a different contrast; operates on **heads**, not
  transcoder features; learns a policy (amortized) vs MACAG's per-prompt greedy; no
  per-feature credit / Shapley grounding.
- *Relation:* concurrent (2026), an alternative **discovery** search. Could be cited
  as a learned alternative to greedy selection; does not address feature-level
  contrastive evaluation or the attention-mediation diagnostic.

**REdit — Circuit Reshaping for reasoning editing (Lei et al., 2026).**
- *What it is:* a model-**editing** method that modifies weights; its "Circuit-
  Interference Law" states edit interference ∝ **overlap of reasoning circuits**, and
  it disentangles overlapping circuits to balance generality vs locality.
- *Similar:* centers on **circuit overlap** between competing behaviors — the exact
  quantity Game 2 measures (`overlap_rate`).
- *Different:* it *changes the model* (intervention/editing), whereas MACAG *measures*
  the model (evaluation). Different goal entirely.
- *Relation:* **complementary and corroborating.** REdit independently establishes
  that low circuit overlap is causally important (less interference ⇒ safer edits),
  which is precisely why Game 2's overlap metric is worth computing. A natural
  *downstream consumer* of MACAG's contrastive output, and external evidence for its
  relevance — the opposite of obsoletion.

### H.3 Synthesis: does any of this obsolete MACAG?

No. Three reasons, in order of importance:

1. **Different objects.** Every *discovery* competitor (ACDC, CD-T, K-MSHC, MechRL,
   Formal MI) works on **native heads/MLPs or raw neurons**; the *attribution*
   competitors (SPEX) work on **input tokens**. MACAG is the one operating on
   **transcoder/SAE feature nodes** with an **encoder-agnostic** contract, plus the
   only one with a **contrastive (target-vs-foil) game** and the
   **attention-mediation diagnostic**. That niche is unoccupied.
2. **Concurrency, not priority.** Formal MI, MechRL, and REdit are **2026** works
   (Feb/May/Mar), i.e. concurrent with this project; they do not establish prior art
   that defeats novelty, and at least two (Formal MI, REdit) are complementary.
3. **The strongest overlap validates, not refutes.** Hedonic Neurons shows the
   coalitional-game lens is being adopted independently; its synergy metric (PAS) is
   a *gift* — the concrete fix for MACAG's acknowledged greedy/submodularity gap, not
   a reason to abandon the framework.

**Honest concessions (already in the doc, reinforced by this landscape):** the field
*has* moved on greedy-vs-better-search (CD-T analytical, MechRL learned, Formal MI
certified) and on synergy (Hedonic/PAS). MACAG should therefore (i) stop leaning on
greedy as a selling point and frame it as a cheap default, (ii) add a PAS-style
pairwise pre-scan to address synergy, (iii) use SPEX to scale the Shapley/Banzhaf
baseline, and (iv) cite Formal MI / REdit as the rigor and application frontiers.
None of these is fatal; all are positioning and Future-Work, consistent with
[§12.2](#122-future-directions)–[§12.4](#124-limitations).

**Net for publishability:** the contribution to defend is the **encoder-agnostic,
intervention-based contrastive evaluation framework + attention-mediation
diagnostic**, positioned against this landscape — not "the best circuit-discovery
search," a claim the landscape would indeed contest.

---

## Appendix I: Path to Top-Conference *and* Top-Journal Readiness

A consolidated, prioritized checklist for getting this work over **both** bars at
once. The two venues reward different things, and the union — not either alone — is
the target:

- **TMLR (journal).** Two criteria only: *are the claims supported by accurate,
  convincing, clear evidence?* and *would some of the audience care?* **No novelty or
  significance gate.** Failure mode = claim–evidence mismatch and overclaiming.
- **Top conference (NeurIPS/ICML/ICLR).** TMLR's soundness bar **plus** novelty,
  significance, scale, and a crisp story. Failure mode = "sound but incremental /
  too-small / not a big enough advance."

Doing the **shared core** makes it TMLR-submittable; adding the **conference layer**
makes the same paper competitive at a top conference. This appendix supersedes the
scattered guidance and maps onto the Appendix B phases.

### I.1 Shared core — required for *both* (the soundness floor)

These are non-negotiable; TMLR rejects without them and a conference desk-rejects.

1. **Claim discipline — tighten every sentence to what the evidence shows.** This is
   the single highest-leverage edit. Concretely (already staged in §10.7/C.7/§11.3):
   - Lead with **oracle cost** (44.7× cheaper, 60/60, $p<10^{-9}$) and
     **faith-per-feature** (non-overlapping CIs) — the two claims *not* on Game 1's
     greedy objective.
   - Demote raw-faith superiority to "higher on most prompts (44/60)," never
     "uniformly more faithful."
   - Scope every claim to "these 3 CLTs / 60 prompts / standard (linear-encoder)
     CLTs," not "in general."
   - *Status (2026-07-01 pass): largely applied in these notes* — fixed the fpf
     population mixing (all-60 headline is 0.859 [0.747, 0.977], not the
     target-preferred-only 0.938), reworded "matches Shapley-gold's *ranking*" to
     gold-level *faithfulness* (prec@k 0.46 is only moderate, and defined on the
     34/60 full-budget prompts), brought the abstracts into compliance with the
     §1.4 guardrails ("jointly sufficient and necessary", "minimality enforced",
     "any SAE"), and made the freezing-bias claim task-dependent (§2.3, §12.4).
     Re-apply the same sweep to the actual manuscript draft.
2. **Resolve the circularity (independent faithfulness metric).** Re-score the
   *selected* sets under a metric Game 1 did **not** optimize — KL over the full
   next-token distribution, and/or a second foil. Without this, "MACAG selects
   faithful sets" is graded on its own training signal ([§11.3](#113-threats-to-validity--reviewer-rebuttals-to-pre-empt)).
   *Highest-value single experiment.* **Implementation done (2026-07-01):** the KL
   rescoring layer ([§2.5](#25-kl-rescoring-a-selection-independent-faithfulness-metric), B0.3) re-scores Game 1/Game 2/baseline
   sets and ships `kl_faith` columns through the drivers and analyzers; what
   remains is running it on the stored nonlinear-benchmark root
   (`python -m macag.cli.rescore_kl --root results/macag_nonlinear_connected`)
   and reporting the numbers. The second-foil variant is also built
   (`macag.cli.rescore_altfoil` + per-prompt `alt_incorrect_token` in the ACDC
   manifest — §2.5); running it on the stored roots remains.
3. **Fair baselines — budget-match ACDC or show faith-vs-k curves** (Appendix B
   Phase 3). The k≈193-vs-8 comparison is contestable as-is; either tune ACDC's τ to
   k≈8 or plot faithfulness(k) for every selector and compare at matched k.
   *Build done (2026-07-02):* `run_baselines --acdc-target-k` bisects τ to the
   budget (B2.4 note); re-run the benchmark with it to regenerate the table.
4. **Statistics over prompts — done, keep it.** Bootstrap CIs + paired Wilcoxon
   (`scripts/macag_bootstrap_wilcoxon.py` — currently missing from `scripts/`;
   restore it, see B1.2). Report CIs and Holm-corrected $p$ on every
   headline number; never a bare point estimate.
5. **Benchmark hygiene.** Exclude or fix the 10 not-target-preferred prompts (the
   logit-gap oracle is ill-posed there); report with/without. Already split in the
   stats script.
6. **Reproducibility package.** Commit the missing `macag/data/nonlinear_benchmark_prompts.json`
   — the file *exists in the working tree* (verified 2026-07-01) but is swallowed by
   the blanket `data/` rule at `.gitignore:194`, as is the new
   `mib_benchmark_prompts.json`; add explicit `!macag/data/*.json` exceptions and
   commit both. Release code, the exact `run_macag --freeze-mode both` +
   `run_baselines` commands, the four CSVs, oracle-kwargs, CLT checkpoint IDs, and
   seeds. TMLR weights this heavily; a top conference expects an artifact.
7. **Honest, specific limitations section.** Single-seed-on-graph, standard-CLT-only,
   logit-gap-as-primary-metric, greedy-suboptimality. TMLR *rewards* this; reviewers
   trust a paper that names its own holes.
8. **Post-fix, matched-protocol re-run of the §10.1–10.6 case study (the pass-2
   TODO markers).** The attention-mediation *demonstrating result* of the chosen
   frame currently cites pre-`71a2ef6`, budget-confounded runs (frozen 8/20 vs
   unfrozen 20/30, buggy `raw_relative` stop). Under TMLR's claim–evidence bar
   the paper must either (a) re-run the two-hop + IOI/docstring sweeps with
   `run_macag game1 --freeze-mode both` (matched budgets, fixed stop, one model
   load) and add B1.2 prompt-bootstrap CIs, or (b) scope the case-study claims to
   the quantities that are provably robust to the known defects — the
   `recoverable_range` sign flips and Game 2 `overlap_rate` — and drop the
   evidence-size / upstream-count contrasts. (a) is a few GPU-days and removes
   all reviewer friction; do (a).

### I.2 Conference layer — added on top for a *top-tier* venue

The shared core is sound but a conference reviewer asks "why is this a big enough
advance?" Answer with:

9. **A gold/known-circuit validation** (Appendix B Phase 4). One task with a
   semi-known circuit (IOI) where MACAG's selected set **recovers** the known
   structure. Converts "internally consistent" into "validated against ground truth" —
   the strongest single credibility addition.
10. **Positive *and* negative control for the attention-mediation diagnostic**
    (Phase 5). 6/60 flips is thin: show a task *known* to be attention-mediated reliably
    flips and a feature-mediated one reliably does not, with the cross-model (gemma vs
    llama) contrast as the discriminating axis.
11. **Scale + multi-seed where it is meaningful.** More prompts and more public CLTs
    for tighter CIs; multi-seed specifically for the *stochastic* components (MC
    Shapley, any sampled oracle) — not for the deterministic selectors, where prompts
    are the sample.
12. **The novelty hook, stated and defended.** Per Appendix H, the defensible novelty
    is the **encoder-agnostic, intervention-based *contrastive* evaluation framework +
    attention-mediation diagnostic**, *not* "best search." Lead with Game 2's
    contrastive disjointness (180/180 overlap-0, no analog in the surveyed
    literature) and the diagnostic;
    frame greedy as a cheap default, not a contribution.
13. **Position against the contemporary landscape** (Appendix H): CD-T, MechRL, Formal
    MI, PAS/Hedonic synergy, SPEX. Cite as related work and Future Work; a top venue
    checks you know the field has moved.
14. **Connect to the project thesis (optional but strategic).** These runs are on
    standard CLTs; the PhD's Spline/KAN-CLT claim is untested here. Either (a) keep
    MACAG as a self-contained evaluator paper (cleanest), or (b) add a Spline-CLT vs
    linear-CLT head-to-head *through MACAG* to make the encoder-agnostic claim concrete
    — a distinct, higher-effort contribution.

### I.3 Suggested sequence (lowest effort / highest marginal value first)

1. Claim-tightening rewrite + commit prompts file + repro appendix — *days, no GPU*
   → makes it **TMLR-submittable**. *[claim-tightening pass applied to these notes
   2026-07-01 (item 1 status); `macag/data/nonlinear_benchmark_prompts.json` exists
   in the working tree but is gitignored (`data/` rule) — needs an explicit
   exception + commit, same for `mib_benchmark_prompts.json`]*
2. Independent faithfulness metric (KL / second foil) on already-selected sets —
   *cheap, no re-selection* → closes the #1 reviewer objection. *[KL implementation
   done and wired ([§2.5](#25-kl-rescoring-a-selection-independent-faithfulness-metric), B0.3); computed for the in-progress MIB runs;
   the nonlinear-benchmark rescore + reporting and the second-foil variant remain]*
3. Post-fix matched re-run of the two-hop + IOI case study (`--freeze-mode both`)
   with prompt-bootstrap CIs (item 8) — *moderate GPU* → the demonstrating result
   becomes quotable without provenance caveats. *[not started, but now one command:
   the consolidated `scripts/run_macag_acdc.sh` defaults to `FREEZE_MODE=both` with
   per-prompt baselines + KL rescore, and the analyzers emit the flip/verdict
   columns; until then only the range-sign flips and overlap_rate are quotable from
   §10.1–10.6]*
4. Budget-matched ACDC / faith-vs-k curves — *moderate GPU* → baseline table becomes
   uncontestable. *[not started; AUC exists in C.7, full curves pending]*
5. Gold-circuit (IOI) recovery + diagnostic positive/negative control — *moderate* →
   crosses into **top-conference** territory. *[not started]*
6. Scale prompts/CLTs + multi-seed on stochastic parts; landscape positioning &
   narrative polish — *ongoing* → competitive submission.

**Definition of done.** *TMLR:* items 1–8 (every claim CI-backed, circularity
resolved, baselines fair, case-study evidence post-fix, fully reproducible). *Top
conference:* 1–8 **plus** 9–10 (gold-circuit validation + controlled diagnostic) and
12–13 (novelty hook + landscape positioning), with 11/14 strengthening the case. The
work is closer to the journal bar than the conference bar today; the gap to the
journal bar is items 2, 3, 6, and 8 (independent metric, fair ACDC, repro package,
post-fix re-run), and the further gap to the conference bar is the gold-circuit
validation and the controlled diagnostic, not a new headline result.
