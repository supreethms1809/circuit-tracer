> **Part of the MACAG docs pack.** Hub: [`macag.md`](macag.md).
> Empirical tables: [`macag_experiments_v3.md`](macag_experiments_v3.md),
> [`macag_dallas_austin.md`](macag_dallas_austin.md).
> §11.2–11.3 (framing + threats) are still the reviewer-facing checklist.

## 11. Discussion of the current evidence

v3 + Dallas deliver three scientific results and several protocol cautions.

**Result — MACAG localizes *where* a behavior lives.** Frozen/unfrozen Game 1
turns “is this circuit faithful?” into a diagnosis: negative `recoverable_range`
under freeze that becomes non-negative when unfrozen is **attention-mediated**.
**Primary evidence (v3 IOI, \(n=100\)):** gemma2-426k **83/100** and
gemma2-2.5M **89/100** `attention_mediated`; llama32-524k **97/100**
`feature_mediated` (0 IOI attention flips). Frozen IOI \(R\): Gemma
**−10.5 / −14.7**, Llama **+10.4**. Dallas frozen \(R>0\) on the same Llama
CLT (two-hop fact). Never cite *normalized* F on Gemma IOI frozen. Legacy
402/500 vs 476/500 figures are the same *direction* on a different campaign
([appendix](macag_appendix_legacy.md)); quote v3.

**Result — logit-gap Game 2 is nearly disjoint.** v3 ABR overlap means
**0.44% / 0.10% / 1.01%** (Gemma-426k / 2.5M / Llama) with exact-zero on
188 / 197 / 172 of 200. FP agrees. This **replaces** “0.0 in 1712/1712.”
Dallas KL Game 2 is the counterexample (18 shared nodes): **do not select
Game 2 on KL.**

**Result — intervention search beats cheap ranking at small \(k\), on the
shared feature oracle.** v3 frozen Game 1 F exceeds influence and `eap_syed`
on all three CLTs; influence is *negative* on Gemma. Ported ACDC at
\(k\le 8\) collapses; high-F ACDC is dense. Dallas KL: Game 1@5 F 6.75 vs
influence@5 5.52 vs AtP@5 0.11 vs ACDC \(\tau=0.5\) (14 nodes) 5.56.

**Caution — capacity is not the lever; family is.** Gemma \(6\times\) width
leaves Game 1 F at \(8.50 \to 8.42\) and IOI still attention-mediated. Llama
is a different routing story (positive frozen \(R\), Shapley actually
runnable).

**Caution — \(F\) at \(\alpha=0.5\) is a mixture.** Gemma Game 1 is
high-\(S\) / low-\(N\) (copy). Dallas PF20: \(\alpha=0\) vs \(\alpha=1\)
are **disjoint** knockout vs copy circuits. Paper tables may use \(0.5\);
unlearning/steering claims must not.

**Caution — v3 is not the camera-ready protocol.** Pruned graphs
(`node_threshold=0.8`) and Game 1 **budget 8**. Dallas +
[`TMLR_EVAL_RECIPE.md`](TMLR_EVAL_RECIPE.md) / [`run_todo_v4.md`](run_todo_v4.md)
are unpruned + unbudgeted.

> Spline-CLT vs linear on GPT-2 is **not** in this result set. If that is the
> thesis, those MACAG runs still have to be produced.

### 11.1 What is *not* yet covered

Catalogued in [§12.3](#123-conference-readiness-what-is-present-vs-missing).
Short list: Gemma Shapley-gold; Track B at scale; InterpBench recovery;
v4 unpruned/unbudgeted MIB; seeds 1–2; Spline-CLT; estimator-seed repeats
for Shapley; bootstrap CIs beyond prompt SEM. KL rescoring **has** been run
on v3 (600 sidecars) — the remaining circularity threat is that F@k is still
Game 1’s logit-gap \(v\), so quote KL-F as the independent column.

### 11.2 Framing options for the paper (pick one)

> **Decisions logged 2026-07-18** (from `macag/docs/code_questions.md` List 2;
> evidence in [Appendix J](macag_appendix_legacy.md#appendix-j-code-and-campaign-audit-2026-07-18)).
>
> **1. Headline verb.** MACAG **"selects and tests causal evidence from attribution
> graphs"** — not "evaluates attribution graphs." *Rationale:* aligns the intro,
> contributions, and methodology with the §1.4 problem statement and the abstract,
> and closes the evaluator-versus-competitor inconsistency created by RQ1's baseline
> comparison. "Evaluation" is retained in exactly three places: the scoring layer and
> its conventions, references to the faithfulness-evaluation literature, and the
> evaluation *of MACAG itself*.
>
> **2. Canonical graph-level outputs.** Fixed list, referenced identically everywhere:
> (i) best achievable faithfulness at budget, (ii) recoverable range, (iii) agreement
> between the graph's own rankings and the interventional verdicts, (iv) the
> frozen/unfrozen attention gap.
>
> **3. Llama-3.2 scope: in-paper, all three MIB tasks at v3.** Seed-0 v3 is
> 200/200 on llama32-524k (IOI 100 / MCQA 50 / ARC-Easy 50) — see
> [`macag_experiments_v3.md`](macag_experiments_v3.md). Pre-v3 Llama was
> IOI 500/500 + MCQA 50/50 with ARC-Easy incomplete
> ([J.C11](macag_appendix_legacy.md#jc-campaign-status-and-coverage)); do not
> mix those denominators with v3. **Do not** revive older wording that Llama
> "has no MIB prompts" or is "absent from this campaign."
>
> **4. Nonlinear benchmark: appendix-only, marked provisional.** Its result artifacts
> are not present on this filesystem ([J.C14](macag_appendix_legacy.md#jd-numbers-behind-claims-already-in-the-draft)),
> so §10.7/C.7 cannot currently be regenerated. It may not carry a headline claim
> until the artifacts are recovered or the benchmark is re-run.
>
> **5. Diagnostic reading.** Dual — model-side (which pathway carries the behavior)
> and graph-side (whether the graph's feature-level picture is complete for that
> prompt). The completeness clause **is** part of the diagnostic contribution (C5).
>
> **6. Acronym.** Expansion retained with the one-sentence acknowledgment at the top
> of this document.
>
> **7. Banzhaf.** Diagnostic/reported credit only, never selection. Not in the intro
> evaluation preview.
>
> **8. Results claim discipline (pre-registered order, confirmed).** (a) oracle cost
> vs gold; (b) faithfulness per selected feature; (c) raw faithfulness as supporting
> evidence only. *Promotion rule:* InterpBench ground-truth recovery is promoted to a
> headline claim only if it lands with CIs on at least two task families; otherwise it
> stays in the appendix. Result verbs remain placeholders until the rerun lands.

The same results support several different paper framings; they are not mutually exclusive but they imply different titles, baselines, and emphasis. Captured here so the writeup can choose deliberately.

- **A — "MACAG: selecting and testing causal evidence from attribution graphs."** *Thesis:* the framework is the contribution; attention-mediation is the demonstrating result. *Needs:* completed gold pass (or recovered nonlinear artifacts) for cost-vs-gold; gold-circuit recovery (Phase 4). *Strength:* broad, method-paper framing aligned with §11.2 item 1; *risk:* reviewers ask "why games, not just ablation ranking?" — answer with the Shapley connection + contrastive Game 2 (no prior analog).
- **B — "Attention mediation is a measurable, model-dependent property of CLT circuits."** *Thesis:* the empirical phenomenon leads; MACAG is the instrument. *Needs:* scale IOI/greater-than + CIs (Phases 1, 5), the feature-mediated positive control. *Strength:* a crisp scientific claim with a clean cross-model contrast (gemma vs llama); *risk:* depends on the frozen/unfrozen distinction being accepted as meaningful — pre-empt with §2.3.
- **C — "Contrastive circuit separation."** *Thesis:* v3 Game 2 overlap is
  \(\le 1\%\) mean (exact-zero on 172–197/200 per CLT) on logit-gap, showing
  competing answers are usually carried by disjoint features. *Strength:*
  denominator-free; ABR and FP agree on *disjointness*. *Risk:* pair identity
  is solver-dependent; Dallas KL Game 2 is *not* disjoint — the claim is
  about logit-gap Game 2, not any \(v\). Do not revive “0.0 in 1712/1712.”
- **D — "Testing transcoder capacity with causal games."** *Thesis:* the capacity/cross-model comparison; lead with "bigger CLT ≠ more faithful circuit." *Strength:* practitioner-relevant; *risk:* three public CLTs is still a small family sample (v3 does give $n=200$ prompts × 3, so the *prompt* sample is no longer the 8-prompt two-hop toy) — would need more public CLTs.

*Recommendation in these notes:* **A as the frame, B as the headline result**, C as the robustness highlight, with the capacity finding (D) as a secondary section. This ordering matches where the evidence is strongest (B, C) and where the conceptual novelty is (A: games + Shapley + contrastive).

### 11.3 Threats to validity / reviewer rebuttals to pre-empt

- **"Zero-ablation is not a clean intervention."** Zeroing a feature is off-manifold;
  resample/mean ablation (as in ACDC) may behave differently. *Mitigation (build
  done 2026-07-02):* the factory now computes per-node alternative ablation values —
  `ablation_mode="mean"` (per-feature mean over clean-prompt positions) or
  `"corrupted"` (patch-style value from the manifest's `corrupted_prompt` at the
  same position, the ACDC convention; length-mismatch falls back to mean) — via
  `compute_mean_ablation_values`, rewriting the intervention specs to 4-tuples
  ([§7.1](macag_implementation.md#71-replacementmodel-scorer)). Drivers opt in with
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
  layer ([§2.5](macag_framework.md#25-kl-rescoring-a-selection-independent-faithfulness-metric)) re-scores every stored evidence set and ships as `kl_faith`
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
  then we report faith\@k, so its raw-faith edge on Llama v3 (Game 1 $13.03$ vs
  Shapley top-8 $11.12$, $n=196$) is graded on its own objective and a reviewer
  will discount it. *Mitigation:* lead with claims that are **not** “greedy
  wins on $F$” — (i) **oracle cost** on Llama (Shapley $\sim 8\times 10^4$
  calls vs Game 1 $\sim 8$k; do **not** cite the missing nonlinear 44.7×
  figure) and (ii) **faith-per-feature** / KL rescoring of the *same* sets —
  and treat raw-faith superiority as supporting only. The independent check is now
  implemented: the KL rescoring layer ([§2.5](macag_framework.md#25-kl-rescoring-a-selection-independent-faithfulness-metric)) re-scores every
  method's selected set under a metric that is *not* the selection objective
  (v3: 600 `macag_kl_faithfulness.json` sidecars). Agreement-with-gold
  (precision\@k / Jaccard vs Shapley) is Llama-only until Gemma gold finishes.
- **"Frozen vs unfrozen is a knob you tuned to get the story."** *Mitigation:* it is a
  *fixed* scoring convention reported both ways for every prompt; the diagnosis is the
  *difference*, and the positive control (B5.2) shows feature-mediated tasks do not
  flip.
- **"Your frozen and unfrozen runs aren't comparable — you changed the budget."**
  True of some *legacy* two-hop runs. **v3** uses `--freeze-mode both` with the
  same budget 8 / `raw_relative` on both legs. Dallas unbudgeted Game 1 is also
  matched across freeze. Evidence Jaccard between legs is still low (v3: 0.04
  Gemma, 0.10 Llama) — that is a finding, not a budget confound.
- **"Normalized faithfulness is broken."** Anticipated — own it: §2.3 documents the
  failure mode, the code guards it, and all headline numbers are raw.
- **"n is tiny."** Closed for v3 MIB (\(n=200\) per CLT, prompt SEM reported).
  Still single seed; Game 1 is deterministic so prompts *are* the sample.
  Dallas is \(n=1\) by design. Do not pad v3 with legacy \(n=500\) IOI counts.
- **"CLT features ≠ the circuit's true units."** MACAG selects and tests evidence on
  the transcoder's circuit, not ground truth; gold-circuit recovery (Phase 4) is the
  bridge, with the feature-vs-head mapping stated as a limitation.

---

## 12. Discussion

### 12.1 MACAG as a General-Purpose Selector and Tester

MACAG is designed to be independent of the method that produced the circuit. It takes a circuit graph and a scoring oracle as inputs and **selects** evidence sets, then **tests** them under explicit scoring conventions (faithfulness, utility, recoverable range, frozen/unfrozen gap). "Evaluation" here names the scoring layer and the evaluation *of MACAG*, not the headline verb ([§11.2](#112-framing-options-for-the-paper-pick-one) item 1). This generality means:

1. **Fair comparison across methods**: any two circuit-construction methods — linear CLT, Spline-CLT, a future SAE-based transcoder — are scored through the exact same game-theoretic lens. The only thing that differs is the candidate node set, so differences in the metrics are attributable to the circuits, not the scoring conventions.

2. **Diagnosing coverage gaps**: MACAG's per-prompt and per-task-family breakdown identifies *where* a method fails — which prompts and which task families. (Game 1 is seed-invariant — [J.B9](macag_appendix_legacy.md#jb-protocol-integrity) — so cross-seed selected-set stability must not be reported as a result.)

3. **Evidence-based interpretability**: Rather than asking "how good is the transcoder at reconstructing MLP outputs?" (a model-level question), MACAG asks "which specific features causally explain this prediction?" (a circuit-level question).

### 12.2 Future Directions

1. **MACAG-aware training**: Add a regularization term during CLT training that encourages features to have high marginal MACAG utility, not just low reconstruction error. This would directly optimize for circuit faithfulness.

2. **Adaptive $\lambda$ and $\beta$**: Currently fixed across all prompts and variants. Prompt-specific or variant-specific penalty schedules could improve evidence quality.

3. **Beyond greedy**: The greedy hill-climbing algorithm has no optimality guarantees for non-submodular utility functions. Beam search, simulated annealing, or exact solvers for small candidate pools could improve evidence quality.

4. **Multi-model application**: Applying MACAG to circuits from more model families (Pythia, Qwen, larger Gemma) to test whether the attention-mediation diagnosis and contrastive-separation results generalize.

### 12.3 Conference-Readiness: What Is Present vs Missing

Honest accounting against a TMLR / top-venue interpretability bar, using
**v3 + Dallas** as the current evidence
([`macag_experiments_v3.md`](macag_experiments_v3.md),
[`macag_dallas_austin.md`](macag_dallas_austin.md)).

**Present**

- Framework, two games, coalitional \(v\), freeze protocol, KL rescoring
  ([framework](macag_framework.md), [foundations](macag_foundations.md)).
- **v3 seed-0 MIB:** 3 CLTs × 200 prompts (IOI 100 / MCQA 50 / ARC-Easy 50),
  Game 1 dual-freeze, Game 2 ABR+FP, KL sidecars, Track A influence /
  `eap_syed` / ported ACDC on every prompt.
- Attention-mediation at \(n=100\) IOI: Gemma attention-mediated, Llama
  feature-mediated; \(6\times\) Gemma width does not flip it.
- Near-disjoint Game 2 on logit-gap (mean overlap \(\le 1\%\)).
- Llama Shapley-gold on **196/200** prompts (prefix F vs Game 1).
- Dallas unpruned case study that *justifies* the v4 protocol (unpruned,
  unbudgeted, logit-gap select, KL rescore, \(\alpha\) split, Track B
  separation) and the application metric split.
- Fairness contract written down ([`TMLR_EVAL_RECIPE.md`](TMLR_EVAL_RECIPE.md))
  after a withdrawn-paper baseline failure mode.

**Missing / blocking**

1. **v4 protocol on MIB.** v3 graphs are pruned (`node_threshold=0.8`) and
   Game 1 is budget-capped at 8. Camera-ready Track A should be unpruned +
   unbudgeted ([`run_todo_v4.md`](run_todo_v4.md)). Do not pretend v3 *is* v4.
2. **Gemma Shapley-gold.** 7/200 on 426k, **0/200** on 2.5M. Cost-vs-gold
   claims are Llama-only until that pass finishes. Do not cite the missing
   nonlinear-benchmark 44.7× figure.
3. **Graph EAP as a full row.** Foil-seed failure leaves 13–34/200 usable
   prompts. Feature AtP (`eap_syed`) is the Syed-formula row.
4. **Track B at scale.** Native-edge JSON exists for one IOI prompt per CLT
   plus Dallas. Pipeline comparison, not selector isolation.
5. **InterpBench / published IOI gold-circuit recovery** at v3/v4 scale.
   The scorer exists; the campaign is not in these tables.
6. **Seeds 1–2, bootstrap CIs, Shapley estimator-seed repeats.** Prompt SEM
   is reported; that is not a CLT seed and Game 1 is deterministic.
7. **Spline-CLT vs linear.** No Spline graphs in v3/Dallas.
8. **Greedy optimality gap** vs brute force on small pools (roadmap B3.2).

**Minimum bar still open after v3:** unpruned unbudgeted tables (v4), Gemma
Shapley or an explicit “Llama-only gold” scope, Track B on a declared
subset, InterpBench or IOI component recovery, and KL-F next to logit-gap F
in the main table (the sidecars exist — they need to be the quoted column).

### 12.4 Limitations

1. **Greedy approximation**: Game 1 and Game 2 both use greedy hill-climbing, which may miss globally optimal evidence sets. The quality of the greedy solution depends on approximate submodularity of the utility function.

2. **Oracle cost**: Each oracle call is a forward pass with modified activations.
   v3 Game 1 frozen is ~8k–12k calls/prompt; Game 2 ABR ~16k–26k; ported ACDC
   \(\tau\) grid ~22k–35k. Unbudgeted Game 2 on Dallas’s 2028 nodes did not
   finish ABR in 24h. Influence is free; `eap_syed` is 3+1 model passes.

3. **Fixed scoring function**: The logit gap score assumes the target-foil distinction is the relevant behavioral signal. For tasks without a clear foil (e.g., open-ended generation), alternative scoring functions would be needed.

4. **Single-position scoring**: MACAG tests interventions at the final token position. Circuits that operate across multiple positions (e.g., induction heads) may require position-aware scoring.

5. **Attention freezing biases the minimal set**: with `freeze_attention=True`, attention-mediated upstream features are already accounted for by the frozen pattern, so Game 1 can drop them from the minimal faithful set even though they are part of the circuit ([§2.3](macag_framework.md#23-attention-freezing-and-the-error-floor)). The direction of the bias is **task-dependent**. On v3 (matched budget 8 both legs), unfreezing *shrinks* mean $|E^\star|$ (Gemma-426k $7.56\to 5.89$, Gemma-2.5M $7.11\to 6.40$, Llama $7.86\to 7.46$) while flipping Gemma $R$ from negative to $\approx +11$. Legacy two-hop recruitment vs IOI shrinkage ([appendix](macag_appendix_legacy.md)) is the same diagnostic, not a budget confound. Single-convention evidence sets are not trustworthy on their own; run `--freeze-mode both` and report the contrast.

6. **Normalized metrics depend on a fragile denominator**: `recoverable_range = all − empty` collapses toward zero or goes negative for attention-mediated tasks (frozen) or when ablate-all collapses the baseline (unfrozen), making the normalized scores and the `normalized` Game 1 stop unreliable. Report the raw (denominator-free) sufficiency/necessity/faithfulness and use the `raw_relative` stop in those regimes ([§2.3](macag_framework.md#23-attention-freezing-and-the-error-floor), [§4](macag_game1.md#4-game-1-minimal-faithful-evidence)).

---

## 13. Conclusion

MACAG provides a general-purpose, game-theoretic framework that **selects and
tests causal evidence** from attribution circuit graphs via causal interventions.
Where a circuit-tracing pipeline produces a graph whose edges are local linear
attribution scores, MACAG asks the causal follow-up questions those scores cannot
answer: its two complementary games measure whether a *small* set of feature nodes
is sufficient and necessary for a prediction (Game 1; minimality optimized, not
certified), and whether competing predictions are carried by *distinct* features
(Game 2). Game 1 is in the spirit of automated circuit discovery (ACDC) but
recast as a bottom-up, sparsity-penalized selection over transcoder features that
scores sufficiency as well as necessity; Game 2 is a new contrastive game with no
analog in the circuit-discovery work surveyed here (Appendix H).

Because MACAG consumes only a graph and a scoring oracle, its encoder-agnostic
design makes it applicable to any circuit-construction method — linear CLT,
Spline-CLT, or a future transcoder/SAE variant — providing a standardized
intervention-based selection and testing layer for mechanistic interpretability.

As an illustration on **TMLR-250 v3** (MIB IOI/MCQA/ARC-Easy, seed 0, 200
prompts × 3 CLTs — [`macag_experiments_v3.md`](macag_experiments_v3.md)),
MACAG surfaces an **attention-mediation diagnosis** on IOI (\(n=100\)):
gemma2-426k 83/100 and gemma2-2.5M 89/100 `attention_mediated` vs
llama32-524k 97/100 `feature_mediated`. Logit-gap Game 2 is *nearly*
disjoint (mean overlap \(0.10\%\)–\(1.01\%\); exact-zero on 172–197 of 200).
On the same pruned graphs, Game 1 exceeds top-k influence and feature AtP
at budget 8; ported ACDC is not a sparse competitor. The Dallas–Austin
unpruned Llama prompt
([`macag_dallas_austin.md`](macag_dallas_austin.md)) is why the paper
protocol moved to unpruned graphs, unbudgeted Game 1, logit-gap selection,
and KL-as-rescore, and why \(\alpha\in\{0,0.5,1\}\) are different circuits.

Shapley-gold is **Llama-complete** (196/200), not Gemma. Native-edge ACDC/EAP
are a separate track (Dallas + one IOI prompt per CLT). v4 still has to
rerun MIB without prune and without a Game 1 budget cap. Do not cite
pre-v3 1712/1712 overlap or 402/500 mediation counts as current.

---
