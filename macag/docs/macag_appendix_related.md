> **Part of the MACAG docs pack.** Hub: [`macag.md`](macag.md). Contemporary methods positioning (former Appendix H).

## Appendix H: Contemporary Methods and Positioning

A "deep-research" report circulated alongside this project argued MACAG is
"obsolete" given newer methods. This appendix is the honest response: each cited
method was located on arXiv/OpenReview, verified to exist, and compared to MACAG in
the same style as ACDC ([§1.3](macag_motivation.md#13-relation-to-prior-work), [Appendix G](macag_appendix_baselines.md#appendix-g-macag-vs-acdc-algorithmic-differences)).
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
transcoder/SAE features.** MACAG's encoder-agnostic, feature-level selection and testing — plus
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
- *Relation:* a fast **discovery** front-end, not a causal **selector/tester**. CD-T could
  *produce* a circuit that MACAG then selects evidence from and tests. Not obsoleting; orthogonal stage.

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
  MACAG's Shapley/Banzhaf connection ([§3.6](macag_foundations.md#36-relation-to-shapley-and-banzhaf-credit)).
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
  ([§3.0](macag_foundations.md#30-the-underlying-coalitional-game)); explicitly about synergy — precisely
  the submodularity gap MACAG acknowledges ([§3.2](macag_foundations.md#32-the-value-function-and-submodularity)).
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
  contrastive evidence selection or the attention-mediation diagnostic.

**REdit — Circuit Reshaping for reasoning editing (Lei et al., 2026).**
- *What it is:* a model-**editing** method that modifies weights; its "Circuit-
  Interference Law" states edit interference ∝ **overlap of reasoning circuits**, and
  it disentangles overlapping circuits to balance generality vs locality.
- *Similar:* centers on **circuit overlap** between competing behaviors — the exact
  quantity Game 2 measures (`overlap_rate`).
- *Different:* it *changes the model* (intervention/editing), whereas MACAG *selects
  and tests evidence* about the model. Different goal entirely.
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
[§12.2](macag_discussion.md#122-future-directions)–[§12.4](macag_discussion.md#124-limitations).

**Net for publishability:** the contribution to defend is the **encoder-agnostic,
intervention-based framework that selects and tests contrastive causal evidence +
attention-mediation diagnostic**, positioned against this landscape — not "the best
circuit-discovery search," a claim the landscape would indeed contest.

---
