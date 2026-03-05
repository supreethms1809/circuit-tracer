# Paper Evaluation Runner

This document explains how to run the config-driven NeurIPS evaluation suites and how to interpret the artifacts they produce.

## What This Runner Is For

`paper-eval` is the conference-facing evaluation interface for Spline-CLT. It is designed to remove manual CLI tweaking during the final experimental campaign.

You give it exactly one suite JSON:

```bash
paper-eval --suite experiments/paper_configs/suites/neurips_core_gpt2.json
```

The runner then expands and executes the full paper workflow from configuration:

1. activation collection
2. per-seed training
3. per-seed evaluation
4. MACAG runs on eligible prompts
5. aggregate reporting

The only supported flags are:

- `--suite`
- `--dry-run`
- `--validate-only`

There are intentionally no CLI hyperparameter overrides.

## Prerequisites

Install the project and dependencies from the repo root:

```bash
pip install -e .
```

If you want to run tests for the paper runner:

```bash
pytest -q tests/test_paper_config.py tests/test_paper_reporting.py tests/test_paper_runner.py
```

## Available Suites

The suite files live in `experiments/paper_configs/suites/`.

- `neurips_core_gpt2.json`
  Core paper run on GPT-2 small.
  3 seeds.
  60 benchmark prompts.

- `neurips_high_gpt2.json`
  Higher-budget GPT-2 run.
  5 seeds.
  120 benchmark prompts.
  Includes the spline grid ablation.

- `neurips_high_gpt2_pythia160m.json`
  Stretch suite.
  Adds Pythia-160M variants alongside the GPT-2 variants.

- `macag_case_studies.json`
  Small curated suite for downstream MACAG analysis and figure-ready cases.

## Before You Launch a Long Run

Validate the suite:

```bash
paper-eval --suite experiments/paper_configs/suites/neurips_core_gpt2.json --validate-only
```

Inspect the expanded job graph:

```bash
paper-eval --suite experiments/paper_configs/suites/neurips_core_gpt2.json --dry-run
```

Use `--dry-run` to verify:

- the suite name
- the model variants
- the seed count
- the benchmark size
- the derived output path
- the dataset, train, eval, MACAG, and report jobs that will run

## Running a Suite

Launch the core GPT-2 paper suite:

```bash
paper-eval --suite experiments/paper_configs/suites/neurips_core_gpt2.json
```

Launch the higher-budget GPT-2 suite:

```bash
paper-eval --suite experiments/paper_configs/suites/neurips_high_gpt2.json
```

Launch the stretch GPT-2 + Pythia-160M suite:

```bash
paper-eval --suite experiments/paper_configs/suites/neurips_high_gpt2_pythia160m.json
```

Launch the MACAG case-study suite:

```bash
paper-eval --suite experiments/paper_configs/suites/macag_case_studies.json
```

## Resume Behavior

The runner is resumable.

If you rerun the same suite:

- completed dataset collection is reused
- completed training runs are reused
- completed evaluation runs are reused
- completed MACAG runs are reused
- aggregate reporting is regenerated from the artifacts currently on disk

You should rerun the same suite path instead of trying to manually patch intermediate files.

## Output Layout

For suite `neurips_core_gpt2`, outputs go under:

```text
results/paper/neurips_core_gpt2/
```

The main files are:

```text
results/paper/<suite_name>/
├── resolved_config.json
├── manifest.json
├── per_example_metrics.jsonl
├── aggregate_metrics.json
├── tables.csv
├── report.md
├── figures/
│   └── figure_manifest.json
├── shared/
│   └── activations/
└── runs/
    └── <variant_name>/
        └── seed_<seed>/
            ├── manifest.json
            ├── train_summary.json
            ├── evaluation/
            │   ├── evaluation_summary.json
            │   ├── reconstruction_records.jsonl
            │   ├── prompt_metrics.jsonl
            │   ├── monosemanticity_records.jsonl
            │   ├── graphs/
            │   └── splines/
            └── macag/
                ├── macag_summary.json
                └── macag_records.jsonl
```

## What Each Output Means

### `resolved_config.json`

The fully merged suite config after OmegaConf default resolution.

Use this to answer:

- exactly which variants were run
- which seeds were used
- which benchmark manifest was used
- which MACAG settings were used

This is the source of truth for experiment provenance.

### `manifest.json`

Top-level run metadata, including:

- suite name
- benchmark manifest version
- git commit
- hardware summary
- seeds
- variants

Use this to confirm that the run matches the code and environment you meant to evaluate.

### `per_example_metrics.jsonl`

The raw row-level outputs across the suite.

This is where you go when:

- a table looks suspicious
- you need per-prompt failures
- you want to audit a specific seed
- you want to re-aggregate with a different analysis notebook

Each line is one JSON record. Important `record_type` values include:

- `reconstruction_sample`
- `prompt_metric`
- `monosemanticity_feature`
- `macag_game1`
- `macag_game2`
- `macag_error`

### `aggregate_metrics.json`

The main machine-readable paper summary.

This file contains:

- `variants`
  Per-variant summaries for reconstruction, replacement fidelity, circuit metrics, monosemanticity, MACAG, and diagnostics.

- `comparisons`
  Paired comparisons between the configured primary variant and baseline variant.
  Includes mean advantage, bootstrap confidence interval, win/loss counts, and per-seed consistency.

- `inclusion_gate`
  The automatic decision summary for whether MACAG belongs in the main paper narrative or should stay in appendix/case-study framing.

### `tables.csv`

Flattened table-ready metrics for:

- Table 1: reconstruction and replacement fidelity
- Table 2: circuit metrics by family
- Table 3: MACAG metrics by family

This should be the first export you use for paper tables, not ad hoc spreadsheet copying from logs.

### `report.md`

A human-readable summary of the whole suite.

This is the best starting point after a run finishes.

### `figures/figure_manifest.json`

Pointers for downstream figure generation.

It does not draw the figures itself. It tells you which artifacts correspond to:

- reconstruction vs fidelity tradeoff
- monosemanticity distribution
- spline case-study curves
- annotated MACAG graph

## How To Read the Main Results

The paper is structured around the comparison between the primary spline variant and the baseline linear variant.

The configured comparison is stored in:

- `aggregate_metrics.json -> inclusion_gate.primary_variant`
- `aggregate_metrics.json -> inclusion_gate.baseline_variant`

### 1. Reconstruction and Replacement Fidelity

Start with Table 1 or:

- `aggregate_metrics.json -> variants -> <variant> -> reconstruction`
- `aggregate_metrics.json -> variants -> <variant> -> replacement`

Interpretation:

- `mse_total`
  Lower is better.
  This is the main reconstruction error metric.

- `cosine_similarity`
  Higher is better.
  Useful when scale differs but direction quality matters.

- `relative_error`
  Lower is better.
  Normalizes error magnitude by output magnitude.

- `top1_match_rate`
  Higher is better.
  Measures how often the replacement model predicts the same top token as the base LM.

- `kl_divergence`
  Lower is better.
  Measures how much the replacement model's token distribution deviates from the base LM.

Conference reading rule:

- If spline improves MSE but degrades `top1_match_rate` or KL badly, do not claim a clean win.
- The strongest result is lower reconstruction error together with equal-or-better replacement fidelity.

### 2. Circuit Utility Metrics

Use Table 2 or:

- `aggregate_metrics.json -> variants -> <variant> -> circuit`

Important metrics:

- `keep_only_gap_ratio`
  Higher is better.
  Keeps only the top selected features and measures how much of the target-vs-foil logit gap survives.
  High values mean the selected circuit is sufficient.

- `gap_drop_ratio`
  Higher is better.
  Removes the top selected features and measures how much of the target-vs-foil gap disappears.
  High values mean the selected circuit is necessary.

- `active_feature_count`
  Smaller is not automatically better.
  Use it as a complexity/context metric, not as a standalone quality score.

- `shapley_causal_jaccard`
  Higher is better.
  Measures agreement between causal ranking and Shapley ranking.
  High agreement means the circuit ranking is more stable across attribution views.

Conference reading rule:

- Prefer spline when `keep_only_gap_ratio` and `gap_drop_ratio` improve without exploding the active feature count.
- If a variant has slightly better faithfulness but far more active features, present that as a tradeoff, not an unconditional win.

### 3. Monosemanticity

Use:

- `aggregate_metrics.json -> variants -> <variant> -> monosemanticity`

Important metrics:

- `mean_gini`
  Higher usually means activations are concentrated on fewer examples.

- `median_gini`
  More robust than the mean if a few features dominate.

- `fraction_gini_gt_0_7`
- `fraction_gini_gt_0_8`
  Higher means more features look highly specific.

Interpretation:

- This is a supporting interpretability signal, not the primary decision metric.
- Use it to argue that spline features are at least as specific, or more specific, than linear baseline features.

### 4. MACAG Results

Use Table 3 or:

- `aggregate_metrics.json -> variants -> <variant> -> macag`

For Game 1:

- `evidence_size`
  Lower is better if faithfulness is comparable.

- `faithfulness`
  Higher is better.

- `sufficiency`
  Higher is better.
  Selected evidence alone preserves target support.

- `necessity`
  Higher is better.
  Removing the evidence substantially damages target support.

- `utility`
  Higher is better.
  This is the solver's objective after sparsity penalty.

For Game 2:

- `overlap_rate`
  Lower is usually better.
  Lower overlap suggests better contrastive separation between target and foil evidence.

Interpretation rule:

- Do not celebrate smaller evidence sets if faithfulness collapses.
- Do not celebrate low overlap if both target and foil evidence become weak.
- The preferred outcome is:
  matched-or-better faithfulness with smaller evidence sets, or
  matched-or-better evidence size with stronger faithfulness.

### 5. MACAG Stability

Use:

- `aggregate_metrics.json -> variants -> <variant> -> macag -> stability`

Important field:

- `mean_jaccard`
  Higher means evidence sets are more stable across seeds.

Interpretation:

- High MACAG performance with poor cross-seed stability should be presented cautiously.
- Stable evidence sets are much more convincing for the paper.

## How To Read the Comparison Section

Use:

- `aggregate_metrics.json -> comparisons`

These comparisons are paired and bootstrap-aggregated between the configured primary and baseline variants.

For each metric, look at:

- `mean_advantage`
  Positive means the primary spline variant is better after metric direction is normalized.

- `ci_low`, `ci_high`
  If the confidence interval crosses zero, treat the comparison as weak or inconclusive.

- `wins`, `losses`
  Prompt/example-level directional support.

- `seed_consistency`
  Whether the direction is stable across seeds.

Practical interpretation:

- Strong result:
  positive mean advantage, CI above zero, and most seeds positive.

- Suggestive but not decisive:
  positive mean advantage, CI overlaps zero, but seed direction is mostly consistent.

- Weak:
  mixed signs across seeds or many prompt-level losses.

## How To Read the Inclusion Gate

Use:

- `aggregate_metrics.json -> inclusion_gate`

This answers the paper-framing question:

- Should MACAG stay in the main paper narrative?
- Or should it move to appendix / case studies?

Important fields:

- `primary_metric_better_count`
  Number of primary metrics where spline beats the baseline.

- `direction_consistent_metric_count`
  Number of those metrics that also move in the same direction across seeds.

- `macag_family_win_count`
  Number of task families where MACAG looks better under the spline circuits.

- `include_macag_in_main_text`
  Final boolean decision from the pre-registered gate.

Interpretation:

- `true`
  The downstream MACAG result is strong enough to support the main paper story.

- `false`
  The paper should remain primarily a spline-vs-linear CLT paper, with MACAG treated as downstream analysis or appendix evidence.

## Recommended Read Order After a Run

1. Open `report.md`.
2. Check `aggregate_metrics.json -> inclusion_gate`.
3. Check Table 1 metrics for reconstruction and replacement fidelity.
4. Check Table 2 for circuit sufficiency/necessity behavior by family.
5. Check Table 3 only after the spline-vs-linear story is already quantitatively defensible.
6. If anything looks unstable, inspect `per_example_metrics.jsonl`.

## Common Failure Modes

### Nonzero error counts

If `diagnostics.error_count > 0`, inspect `per_example_metrics.jsonl` before drawing conclusions.

Likely causes:

- empty MACAG candidate sets
- prompt-level graph creation failures
- model-loading mismatches

### Good reconstruction, weak replacement fidelity

This usually means the transcoder reconstructs MLP outputs numerically but fails to preserve the base model's end-task behavior closely enough.

Do not claim interpretability wins from reconstruction alone.

### Good MACAG sparsity, weak faithfulness

This usually means the solver found small evidence sets by becoming too aggressive.

Treat this as a failure mode, not a positive result.

### Wide confidence intervals

This usually means:

- too few seeds
- too few prompts
- or highly variable prompt behavior

If this happens in the core suite, prefer adding seeds or using the high suite before strengthening claims.

## Minimal Workflow for a Real Campaign

```bash
paper-eval --suite experiments/paper_configs/suites/neurips_core_gpt2.json --validate-only
paper-eval --suite experiments/paper_configs/suites/neurips_core_gpt2.json --dry-run
paper-eval --suite experiments/paper_configs/suites/neurips_core_gpt2.json
```

After completion:

1. read `results/paper/neurips_core_gpt2/report.md`
2. inspect `results/paper/neurips_core_gpt2/aggregate_metrics.json`
3. use `results/paper/neurips_core_gpt2/tables.csv` for paper tables

