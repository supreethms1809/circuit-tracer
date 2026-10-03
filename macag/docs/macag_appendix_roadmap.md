> **Part of the MACAG docs pack.** Hub: [`macag.md`](macag.md). Submission roadmap (former Appendices B and I). Operational launch is `TMLR_EVAL_RECIPE.md` / `run_todo_v4.md`.
>
> **Current paper numbers are in [`macag_experiments_v3.md`](macag_experiments_v3.md)
> and [`macag_dallas_austin.md`](macag_dallas_austin.md), not here.** This file is
> the 2026-07 plan (build items done; remaining *runs* now mean v4). Do not cite
> “MIB 1712/1712 overlap-0” from this appendix as current — v3 Game 2 overlap is
> near-zero, not identically zero.

## Appendix B: Roadmap to a Submission-Ready Evaluation

This turns the gaps in [§12.3](macag_discussion.md#123-conference-readiness-what-is-present-vs-missing)
into an ordered, concrete plan. Each task lists **build** (code to add/change),
**run** (command), and **done-when** (acceptance criterion). **As of 2026-07-02
every build item below is implemented; the exact commands for the remaining
*runs* are consolidated in [`macag/docs/run_todo.md`](run_todo.md).** Reuse the existing
drivers wherever possible (`scripts/run_macag_*.sh`, `macag/cli/run_macag.py`).
The MACAG Shapley-gold baseline was written fresh over the MACAG oracle —
deliberately *not* a reuse of `attribution/shapley.py`
([§3.6](macag_foundations.md#36-relation-to-shapley-and-banzhaf-credit)).

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
  wired, and unit-tested ([§2.5](macag_framework.md#25-kl-rescoring-a-selection-independent-faithfulness-metric)): `score_kind="kl_divergence"` in
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
    (`experiments/build_mib_benchmark_prompts.py`, [§9.5.1](macag_appendix_legacy.md#95-mib-bench-full-campaign-setup)) now supplies
    standardized IOI/MCQA/ARC prompts in the MACAG manifest format at full
    benchmark scale (500/50/570, `--task-limit ioi=500`, validation split); the
    hand-written manifests (`macag/data/acdc_benchmark_prompts.json`,
    `experiments/macag_generalization_prompts.json`) are unchanged and still small.
  - *Run:* `scripts/run_macag_acdc.sh` (the consolidated driver — the old
    `run_macag_sweep.sh`/`run_macag_unfrozen*.sh` are folded into it) for the
    hand-written manifests, and `scripts/run_mib_benchmark.sh` ([§9.5.4](macag_appendix_legacy.md#95-mib-bench-full-campaign-setup)) for the
    full-scale, 3-seed MIB campaign — **in progress**
    (`results/macag_mib_seed{0,1,2}/`).
- **B1.2 Bootstrap CIs over prompts.** ✅ **Build DONE (2026-07-02)** —
  `scripts/macag_bootstrap_wilcoxon.py` is (re)implemented and checked in: bootstrap
  CIs (reusing `spline_clt.paper.reporting.bootstrap_mean_ci`), paired Wilcoxon with
  Holm correction and hand-rolled rank-biserial, win/loss counts, pref-only blocks,
  and the Shapley/Game-1 cost ratio; verified to reproduce the §10.7 numbers from
  `results/macag_nonlinear_connected/baselines.csv` (faith\@8 5.50 [4.88, 6.15],
  fpf 0.859, cost 44.7× [43.5, 45.9] — **provisional**, [J.C14](macag_appendix_legacy.md#jc-campaign-status-and-coverage)).
  The flip-rate CI is also in:
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

Run every selector on the **same** candidate node set; score every selected set
under the **same** oracle $v$ and $\alpha$. Influence/EAP still *rank* from the
graph (weak shared-$v$ claim — [J.A1](macag_appendix_legacy.md#ja-baselines-and-the-shared-characteristic-function)).
Build one harness, four selectors.

> **Status: B2.0–B2.4 are BUILT and RUN** (plus B3.2's brute-forcer), unit-tested
> on toy oracles in `tests/test_macag_baselines.py` and **run end-to-end with the
> real `ReplacementModel` oracle on the 60-prompt nonlinear benchmark**
> (§10.7/C.7, `results/macag_nonlinear_connected/baselines.csv`) — the master
> table with bootstrap CIs + paired Wilcoxon exists. The harness is now also
> **embedded in the sweep drivers**: `run_macag_acdc.sh` / `run_macag_mib.sh` emit
> a per-prompt `macag_baselines.json` (disable with `SKIP_BASELINES=1`;
> `BASELINE_METHODS`/`SHAPLEY_PERMUTATIONS` configurable) and
> `experiments/analyze_macag_baselines.py` aggregates a sweep root into
> `baselines.csv` — including `kl_faith` columns from the [§2.5](macag_framework.md#25-kl-rescoring-a-selection-independent-faithfulness-metric)
> rescoring pass. **Cost note / two-pass split (2026-07-02):** MC Shapley is
> ~90% of a prompt's baseline cost (nonlinear tables claimed ≈33.9k oracle calls
> vs Game 1's ~760 — **provisional**, [J.C14](macag_appendix_legacy.md#jc-campaign-status-and-coverage);
> MIB Game 1 measures ~1200 calls/prompt at ~488 candidates — [J.D20](macag_appendix_legacy.md#jd-numbers-behind-claims-already-in-the-draft)),
> so the drivers support a fast pass without it
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
  `attribution/shapley.py` ([§3.6](macag_foundations.md#36-relation-to-shapley-and-banzhaf-credit)) —
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
  [§3.2](macag_foundations.md#32-the-value-function-and-submodularity).

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
   - Lead with **oracle cost vs gold only after a completed gold pass** (nonlinear
     44.7× / 60/60 is provisional — [J.C14](macag_appendix_legacy.md#jc-campaign-status-and-coverage); MIB gold = 0)
     and
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
   rescoring layer ([§2.5](macag_framework.md#25-kl-rescoring-a-selection-independent-faithfulness-metric), B0.3) re-scores Game 1/Game 2/baseline
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
    is the **encoder-agnostic, intervention-based framework that *selects and tests*
    contrastive causal evidence + attention-mediation diagnostic**, *not* "best search."
    Lead with Game 2's
    contrastive disjointness (MIB **1712/1712** overlap-0 — [J.D17](macag_appendix_legacy.md#jd-numbers-behind-claims-already-in-the-draft);
    no analog in the surveyed
    literature) and the diagnostic;
    frame greedy as a cheap default, not a contribution.
13. **Position against the contemporary landscape** (Appendix H): CD-T, MechRL, Formal
    MI, PAS/Hedonic synergy, SPEX. Cite as related work and Future Work; a top venue
    checks you know the field has moved.
14. **Connect to the project thesis (optional but strategic).** These runs are on
    standard CLTs; the PhD's Spline/KAN-CLT claim is untested here. Either (a) keep
    MACAG as a self-contained selector/tester paper (cleanest), or (b) add a Spline-CLT vs
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
   done and wired ([§2.5](macag_framework.md#25-kl-rescoring-a-selection-independent-faithfulness-metric), B0.3); computed for the in-progress MIB runs;
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

---
