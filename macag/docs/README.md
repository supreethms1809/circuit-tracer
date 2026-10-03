# MACAG Thesis Writing Pack

This folder is a writing support bundle for the MACAG paper/thesis chapter
and the TMLR evaluation campaign.

## MACAG draft (split from the former monolith)

- **[`macag.md`](macag.md)** — hub: abstract, TOC, campaign map. **Start here.**
- [`macag_motivation.md`](macag_motivation.md), [`macag_framework.md`](macag_framework.md), [`macag_foundations.md`](macag_foundations.md), [`macag_game1.md`](macag_game1.md), [`macag_game2.md`](macag_game2.md), [`macag_implementation.md`](macag_implementation.md) — methods.
- **[`macag_experiments_v3.md`](macag_experiments_v3.md)** — **current MIB tables** (TMLR-250 v3, 3 CLTs × 200 prompts).
- **[`macag_dallas_austin.md`](macag_dallas_austin.md)** — unpruned Llama Dallas–Austin case study.
- [`macag_discussion.md`](macag_discussion.md), [`macag_references.md`](macag_references.md)
- `macag_appendix_*.md` — baselines, roadmap, background, related work. **`macag_appendix_legacy.md` is provenance-only** (do not cite).

## Other files

- `FRAMEWORK_EVOLUTION_PLAN.md`
  - Proposal-facing architecture decision: CDEA-aligned design principles with separate repos now.
- `IMPLEMENTATION_ROADMAP_POST_PRELIMS.md`
  - Phased plan, milestones, and risks for implementing the planned framework changes.
- `THESIS_CHAPTER_BLUEPRINT.md`
  - Chapter structure, section goals, and claim framing.
- `METHODS_DRAFT_MACAG.md`
  - Copy-ready methods text (notation, objectives, algorithms, implementation mapping).
- `EXPERIMENTS_RESULTS_TEMPLATE.md`
  - Experiment plan, command templates, and table/figure shells.
- `INTERPRETATION_GUIDE.md`
  - How to interpret MACAG JSON outputs and common failure modes.
- `PRELIMS_DEFENSE_QA.md`
  - Likely committee questions and concise technical answers.
- `baseline_method_map.md`
  - Stable IDs vs aliases for influence / graph-EAP / Syed EAP / ported ACDC / native ACDC.
- `baseline_originals_and_ports.md`
  - Original-paper definition, problem, method, and metrics for ACDC, Syed EAP, attribution-graph influence, and Shapley; how/why each is ported here and whether the evaluation metric matches; closing section on using the published methods as-is (miss / gain / why or why not).
- `baseline_original_track.md`
  - True AutoCircuit edge EAP / ACDC track (separate JSON; not Jaccardable to Game 1).
- **`TMLR_EVAL_RECIPE.md`**
  - **Canonical TMLR evaluation protocol:** fair Track A (shared feature oracle) vs Track B (native edges), unpruned graphs, logit-gap selection + KL rescore, method suite, tables, and a pre-submission baseline audit checklist. Prefer this over ad hoc CLI when launching paper runs.
- **`run_todo.md`**
  - Cluster launch runbook for the **v3** campaign whose numbers are in `macag_experiments_v3.md`.
- **`run_todo_v4.md`**
  - Cluster launch runbook for MIB TMLR-250 **v4** (Pass A/B/C; Shapley deferred).
- **`APPLICATIONS_METRICS.md`**
  - Three practical applications (surgical unlearning, contrastive steering, transcoder/model auditor): which of F/S/N/KL to headline, Game 1 \(\alpha\) / `score_kind` / freeze / stop-metric tuning, and budgeted vs goal-oriented runs. Does not replace the TMLR recipe for paper tables.

## Suggested Use

1. Read [`macag.md`](macag.md) for the claim map, then [`macag_experiments_v3.md`](macag_experiments_v3.md) and [`macag_dallas_austin.md`](macag_dallas_austin.md) for numbers.
2. Start with `FRAMEWORK_EVOLUTION_PLAN.md` to frame the architecture strategy in the proposal.
3. Use `THESIS_CHAPTER_BLUEPRINT.md` to set chapter scope and section flow.
4. Adapt `METHODS_DRAFT_MACAG.md` into your methods chapter section.
5. Use `IMPLEMENTATION_ROADMAP_POST_PRELIMS.md` for timeline and milestone writing.
6. **Before any camera-ready MACAG/baseline run, follow `TMLR_EVAL_RECIPE.md`** (fairness contract + audit checklist).
7. **v3 numbers** were launched with [`run_todo.md`](run_todo.md). **v4** (unpruned, unbudgeted) uses [`run_todo_v4.md`](run_todo_v4.md).
8. Run experiments using `EXPERIMENTS_RESULTS_TEMPLATE.md` aligned to that recipe.
9. Use `INTERPRETATION_GUIDE.md` while analyzing outputs.
10. Prepare the oral defense with `PRELIMS_DEFENSE_QA.md`.
