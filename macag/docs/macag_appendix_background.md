> **Part of the MACAG docs pack.** Hub: [`macag.md`](macag.md). Transformer/CLT/graph background, extended related work, glossary (former Appendices D–F).

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
the frozen/unfrozen distinction ([§2.3](macag_framework.md#23-attention-freezing-and-the-error-floor))
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
`recoverable_range` story in [§2.3](macag_framework.md#23-attention-freezing-and-the-error-floor).

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
  ([§2.3](macag_framework.md#23-attention-freezing-and-the-error-floor)).
MACAG's four oracle modes (all/empty/keep-only/remove, [§2.1](macag_framework.md#21-oracle-scoring))
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
agnostic to which dictionary method made them — it selects and tests causal evidence from the circuit they form.

### E.2 Transcoders, CLTs, and circuit tracing

Transcoders predict MLP outputs through a sparse bottleneck (Dunefsky et al. 2024);
cross-layer transcoders and the attribution-graph pipeline come from the
circuit-tracing work (Ameisen et al. 2025, *Circuit Tracing*; Lindsey et al. 2025,
*On the Biology of a Large Language Model*). **Relation:** this is the upstream that
*builds* MACAG's input graph; MACAG is the downstream causal selector and tester,
complementary to circuit-tracer's attribution-magnitude pruning.

### E.3 Automated circuit discovery

ACDC (Conmy et al. 2023) prunes the native computational graph top-down via patching
against a KL threshold. **Edge/attribution patching** (Nanda 2023; Syed et al. 2023,
*Attribution Patching Outperforms ACDC*) approximates patching effects with a single
backward pass — fast but first-order. Related threads: edge pruning, subnetwork
probing, and head-level manual circuits (Wang et al. 2022, *IOI*; the greater-than
circuit). **Relation:** Game 1 shares the minimal-faithful-circuit goal but works
bottom-up over *transcoder features* with an explicit sparsity-penalized utility and
scores sufficiency as well as necessity ([§1.3](macag_motivation.md#13-relation-to-prior-work)); these
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
attribute to training points / neurons. **Relation:** MACAG poses selecting and testing
causal evidence as a coalitional game over features ([§3.0](macag_foundations.md#30-the-underlying-coalitional-game)),
uses Shapley as the *gold* per-feature reference ([§3.6](macag_foundations.md#36-relation-to-shapley-and-banzhaf-credit)),
and contributes the *contrastive two-player* extension (Game 2) absent from prior
attribution-game work.

### E.6 Feature interventions and steering

Activation steering / feature clamping use the same intervention surface MACAG scores
over, for control rather than evidence selection. **Relation:** MACAG's evidence sets are
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
