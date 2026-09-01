"""Aggregation and reporting helpers for config-driven paper runs."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from spline_clt.paper.config import SuiteConfig

SUMMARY_METRICS: dict[str, bool] = {
    "mse_total": False,
    "cosine_similarity": True,
    "relative_error": False,
    "top1_match_rate": True,
    "top5_match_rate": True,
    "top10_match_rate": True,
    "kl_divergence": False,
    "attribution_seconds": False,
    "attribution_peak_mem_gib": False,
    "mean_gini": True,
    "fraction_gini_gt_0_8": True,
    "graph_replacement_score": True,
    "graph_completeness_score": True,
    "keep_only_gap_ratio": True,
    "gap_drop_ratio": True,
    "retained_error_node_count": False,
    "retained_error_node_fraction": False,
    "shapley_causal_jaccard": True,
    "evidence_size": False,
    "faithfulness": True,
    "utility": True,
}

RECONSTRUCTION_METRICS = [
    "mse_total",
    "cosine_similarity",
    "relative_error",
    "active_per_pos",
    "activation_density",
]

REPLACEMENT_METRICS = [
    "top1_match_rate",
    "top5_match_rate",
    "top10_match_rate",
    "kl_divergence",
]

CIRCUIT_METRICS = [
    "active_feature_count",
    "retained_feature_node_count",
    "retained_error_node_count",
    "retained_error_node_fraction",
    "graph_replacement_score",
    "graph_completeness_score",
    "keep_only_gap_ratio",
    "gap_drop_ratio",
    "shapley_causal_jaccard",
    # Compute-cost metrics (REQ-7); absent in records from older runs, which
    # summarize_metric tolerates (count 0, NaN mean).
    "attribution_seconds",
    "attribution_peak_mem_gib",
]


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, payload: Any) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    if not resolved.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in resolved.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: Sequence[dict[str, Any]]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, sort_keys=True) for record in records]
    resolved.write_text("\n".join(lines) + ("\n" if lines else ""))


def load_suite_records(suite_root: str | Path) -> list[dict[str, Any]]:
    """Load every per-example record emitted by run stages under a suite directory."""
    root = Path(suite_root)
    patterns = [
        "runs/*/seed_*/evaluation/reconstruction_records.jsonl",
        "runs/*/seed_*/evaluation/prompt_metrics.jsonl",
        "runs/*/seed_*/evaluation/monosemanticity_records.jsonl",
        "runs/*/seed_*/evaluation/spline_records.jsonl",
        "runs/*/seed_*/macag/macag_records.jsonl",
    ]
    records: list[dict[str, Any]] = []
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            records.extend(read_jsonl(path))
    return records


def _numeric_values(records: Iterable[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            values.append(float(value))
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return float(math.sqrt(variance))


def bootstrap_mean_ci(
    values: Sequence[float],
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap a confidence interval around the mean."""
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])

    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(bootstrap_samples):
        sample = rng.choice(array, size=len(array), replace=True)
        means.append(float(np.mean(sample)))
    lower_q = (1.0 - confidence_level) / 2.0
    upper_q = 1.0 - lower_q
    return float(np.quantile(means, lower_q)), float(np.quantile(means, upper_q))


def paired_bootstrap_summary(
    left_records: Sequence[dict[str, Any]],
    right_records: Sequence[dict[str, Any]],
    *,
    metric: str,
    pair_keys: Sequence[str],
    higher_is_better: bool,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Compare two variants on aligned prompt/example keys."""
    left_map = {
        tuple(record.get(key) for key in pair_keys): record
        for record in left_records
        if record.get(metric) is not None
    }
    right_map = {
        tuple(record.get(key) for key in pair_keys): record
        for record in right_records
        if record.get(metric) is not None
    }
    shared_keys = sorted(set(left_map) & set(right_map))
    deltas: list[float] = []
    for shared_key in shared_keys:
        left_value = float(left_map[shared_key][metric])
        right_value = float(right_map[shared_key][metric])
        delta = left_value - right_value
        deltas.append(delta if higher_is_better else -delta)

    if not deltas:
        return {
            "metric": metric,
            "paired_count": 0,
            "mean_advantage": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "seed_consistency": {},
            "winner": "insufficient_data",
        }

    ci_low, ci_high = bootstrap_mean_ci(
        deltas,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )
    wins = sum(1 for value in deltas if value > 0)
    losses = sum(1 for value in deltas if value < 0)
    ties = len(deltas) - wins - losses

    by_seed: dict[Any, list[float]] = defaultdict(list)
    for shared_key, advantage in zip(shared_keys, deltas):
        if pair_keys and pair_keys[0] == "seed":
            by_seed[shared_key[0]].append(advantage)
    seed_consistency = {
        str(seed_value): _mean(values)
        for seed_value, values in sorted(by_seed.items(), key=lambda item: str(item[0]))
    }
    positive_seeds = sum(1 for value in seed_consistency.values() if value > 0)

    winner = "tie"
    if _mean(deltas) > 0:
        winner = "left"
    elif _mean(deltas) < 0:
        winner = "right"

    return {
        "metric": metric,
        "paired_count": len(deltas),
        "mean_advantage": _mean(deltas),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "seed_consistency": seed_consistency,
        "positive_seed_count": positive_seeds,
        "winner": winner,
    }


def summarize_metric_per_seed(
    records: Sequence[dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    """Mean-of-per-seed-means aggregation (paper headline convention).

    The pooled `summarize_metric` mixes seeds and prompts into one sample;
    the rebuttal tables report `mean ± std across seeds`, which requires
    grouping by the `seed` field first. Returned additively so pooled
    consumers are unaffected.
    """
    by_seed: dict[Any, list[float]] = {}
    for record in records:
        seed_value = record.get("seed")
        if seed_value is None:
            continue
        value = record.get(metric)
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            by_seed.setdefault(seed_value, []).append(float(value))
    seed_means = {str(seed): _mean(values) for seed, values in sorted(by_seed.items())}
    means = list(seed_means.values())
    return {
        "per_seed_means": seed_means,
        "n_seeds": len(means),
        "mean_of_seed_means": _mean(means) if means else float("nan"),
        "std_of_seed_means": _std(means) if len(means) > 1 else float("nan"),
    }


def summarize_metric(
    records: Sequence[dict[str, Any]],
    metric: str,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    values = _numeric_values(records, metric)
    ci_low, ci_high = bootstrap_mean_ci(
        values,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )
    return {
        "metric": metric,
        "count": len(values),
        "mean": _mean(values),
        "std": _std(values),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "per_seed": summarize_metric_per_seed(records, metric),
    }


def _family_summary(
    records: Sequence[dict[str, Any]],
    metrics: Sequence[str],
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    families = sorted(
        {
            str(record.get("family"))
            for record in records
            if record.get("family") is not None and record.get("status", "ok") == "ok"
        }
    )
    payload: dict[str, Any] = {}
    for family in families:
        family_records = [record for record in records if record.get("family") == family]
        payload[family] = {
            metric: summarize_metric(
                family_records,
                metric=metric,
                bootstrap_samples=bootstrap_samples,
                confidence_level=confidence_level,
                seed=seed,
            )
            for metric in metrics
        }
    return payload


def _aggregate_variant(
    records: Sequence[dict[str, Any]],
    variant_name: str,
    config: SuiteConfig,
) -> dict[str, Any]:
    variant_records = [record for record in records if record.get("variant_name") == variant_name]
    recon_records = [record for record in variant_records if record.get("record_type") == "reconstruction_sample"]
    prompt_records = [
        record
        for record in variant_records
        if record.get("record_type") == "prompt_metric" and record.get("status", "ok") == "ok"
    ]
    mono_records = [record for record in variant_records if record.get("record_type") == "monosemanticity_feature"]
    spline_records = [record for record in variant_records if record.get("record_type") == "spline_feature"]
    macag_game1_records = [
        record
        for record in variant_records
        if record.get("record_type") == "macag_game1" and record.get("status", "ok") == "ok"
    ]
    macag_game2_records = [
        record
        for record in variant_records
        if record.get("record_type") == "macag_game2" and record.get("status", "ok") == "ok"
    ]
    error_records = [record for record in variant_records if record.get("status") == "error"]

    bootstrap_samples = config.reporting.bootstrap_samples
    confidence_level = config.reporting.confidence_level
    bootstrap_seed = hash((config.suite_name, variant_name)) & 0xFFFF

    nonlinearity_scores = _numeric_values(spline_records, "nonlinearity_score")
    ginis = _numeric_values(mono_records, "gini_coefficient")
    mean_gini = _mean(ginis)
    fraction_gini_gt_0_7 = (
        float(sum(1 for value in ginis if value > 0.7) / len(ginis)) if ginis else float("nan")
    )
    fraction_gini_gt_0_8 = (
        float(sum(1 for value in ginis if value > 0.8) / len(ginis)) if ginis else float("nan")
    )

    variant_summary = {
        "label": config.model_variants[variant_name].label,
        "reconstruction": {
            metric: summarize_metric(
                recon_records,
                metric=metric,
                bootstrap_samples=bootstrap_samples,
                confidence_level=confidence_level,
                seed=bootstrap_seed,
            )
            for metric in RECONSTRUCTION_METRICS
        },
        "replacement": {
            metric: summarize_metric(
                prompt_records,
                metric=metric,
                bootstrap_samples=bootstrap_samples,
                confidence_level=confidence_level,
                seed=bootstrap_seed,
            )
            for metric in REPLACEMENT_METRICS
        },
        "circuit": {
            "overall": {
                metric: summarize_metric(
                    prompt_records,
                    metric=metric,
                    bootstrap_samples=bootstrap_samples,
                    confidence_level=confidence_level,
                    seed=bootstrap_seed,
                )
                for metric in CIRCUIT_METRICS
            },
            "by_family": _family_summary(
                prompt_records,
                metrics=CIRCUIT_METRICS,
                bootstrap_samples=bootstrap_samples,
                confidence_level=confidence_level,
                seed=bootstrap_seed,
            ),
        },
        "monosemanticity": {
            "feature_count": len(ginis),
            "mean_gini": mean_gini,
            "median_gini": float(np.median(ginis)) if ginis else float("nan"),
            "fraction_gini_gt_0_7": fraction_gini_gt_0_7,
            "fraction_gini_gt_0_8": fraction_gini_gt_0_8,
        },
        "splines": {
            "feature_count": len(nonlinearity_scores),
            "mean_nonlinearity_score": _mean(nonlinearity_scores),
            "median_nonlinearity_score": (
                float(np.median(nonlinearity_scores)) if nonlinearity_scores else float("nan")
            ),
            "fraction_nonlinear_gt_0_05": (
                float(sum(1 for s in nonlinearity_scores if s > 0.05) / len(nonlinearity_scores))
                if nonlinearity_scores else float("nan")
            ),
            "fraction_nonlinear_gt_0_10": (
                float(sum(1 for s in nonlinearity_scores if s > 0.10) / len(nonlinearity_scores))
                if nonlinearity_scores else float("nan")
            ),
        },
        "macag": {
            "game1": {
                "overall": {
                    metric: summarize_metric(
                        macag_game1_records,
                        metric=metric,
                        bootstrap_samples=bootstrap_samples,
                        confidence_level=confidence_level,
                        seed=bootstrap_seed,
                    )
                    for metric in [
                        "evidence_size",
                        "faithfulness",
                        "sufficiency",
                        "necessity",
                        "utility",
                        "oracle_calls",
                    ]
                },
                "by_family": _family_summary(
                    macag_game1_records,
                    metrics=[
                        "evidence_size",
                        "faithfulness",
                        "sufficiency",
                        "necessity",
                        "utility",
                    ],
                    bootstrap_samples=bootstrap_samples,
                    confidence_level=confidence_level,
                    seed=bootstrap_seed,
                ),
            },
            "game2": {
                "overall": {
                    metric: summarize_metric(
                        macag_game2_records,
                        metric=metric,
                        bootstrap_samples=bootstrap_samples,
                        confidence_level=confidence_level,
                        seed=bootstrap_seed,
                    )
                    for metric in [
                        "evidence_size",
                        "faithfulness",
                        "sufficiency",
                        "necessity",
                        "utility",
                        "overlap_rate",
                        "oracle_calls",
                    ]
                },
                "by_family": _family_summary(
                    macag_game2_records,
                    metrics=[
                        "evidence_size",
                        "faithfulness",
                        "sufficiency",
                        "necessity",
                        "utility",
                        "overlap_rate",
                    ],
                    bootstrap_samples=bootstrap_samples,
                    confidence_level=confidence_level,
                    seed=bootstrap_seed,
                ),
            },
        },
        "diagnostics": {
            "error_count": len(error_records),
            "errors_by_type": {
                record_type: sum(
                    1 for record in error_records if record.get("record_type") == record_type
                )
                for record_type in sorted(
                    {str(record.get("record_type")) for record in error_records}
                )
            },
        },
    }
    return variant_summary


def _macag_stability(
    records: Sequence[dict[str, Any]],
    variant_name: str,
) -> dict[str, Any]:
    relevant = [
        record
        for record in records
        if record.get("variant_name") == variant_name
        and record.get("record_type") in {"macag_game1", "macag_game2"}
        and record.get("status", "ok") == "ok"
    ]
    grouped: dict[tuple[str, str, str], list[set[str]]] = defaultdict(list)
    for record in relevant:
        evidence = set(str(node) for node in record.get("evidence_nodes", []))
        grouped[
            (
                str(record.get("record_type")),
                str(record.get("run_name")),
                str(record.get("prompt_id")),
            )
        ].append(evidence)

    def overlap(set_a: set[str], set_b: set[str]) -> float:
        union = set_a | set_b
        if not union:
            return 1.0
        return len(set_a & set_b) / len(union)

    per_prompt = []
    for group_key, evidence_sets in grouped.items():
        if len(evidence_sets) <= 1:
            continue
        overlaps = []
        for left_index in range(len(evidence_sets)):
            for right_index in range(left_index + 1, len(evidence_sets)):
                overlaps.append(overlap(evidence_sets[left_index], evidence_sets[right_index]))
        per_prompt.append(
            {
                "record_type": group_key[0],
                "run_name": group_key[1],
                "prompt_id": group_key[2],
                "mean_jaccard": _mean(overlaps),
                "comparisons": len(overlaps),
            }
        )
    return {
        "per_prompt": per_prompt,
        "mean_jaccard": _mean([entry["mean_jaccard"] for entry in per_prompt]),
    }


def aggregate_suite_records(records: Sequence[dict[str, Any]], config: SuiteConfig) -> dict[str, Any]:
    """Aggregate every per-example record into suite-level summaries."""
    variants = {
        variant_name: _aggregate_variant(records, variant_name=variant_name, config=config)
        for variant_name in config.model_variants
    }
    for variant_name in variants:
        variants[variant_name]["macag"]["stability"] = _macag_stability(records, variant_name)

    primary = config.reporting.primary_variant
    baseline = config.reporting.baseline_variant
    primary_records = [record for record in records if record.get("variant_name") == primary]
    baseline_records = [record for record in records if record.get("variant_name") == baseline]

    comparisons = {
        "reconstruction": {
            metric: paired_bootstrap_summary(
                [record for record in primary_records if record.get("record_type") == "reconstruction_sample"],
                [record for record in baseline_records if record.get("record_type") == "reconstruction_sample"],
                metric=metric,
                pair_keys=["seed", "sample_index"],
                higher_is_better=SUMMARY_METRICS[metric],
                bootstrap_samples=config.reporting.bootstrap_samples,
                confidence_level=config.reporting.confidence_level,
                seed=hash(("reconstruction", metric)) & 0xFFFF,
            )
            for metric in ["mse_total", "cosine_similarity", "relative_error"]
        },
        "prompt": {
            metric: paired_bootstrap_summary(
                [
                    record
                    for record in primary_records
                    if record.get("record_type") == "prompt_metric" and record.get("status", "ok") == "ok"
                ],
                [
                    record
                    for record in baseline_records
                    if record.get("record_type") == "prompt_metric" and record.get("status", "ok") == "ok"
                ],
                metric=metric,
                pair_keys=["seed", "prompt_id"],
                higher_is_better=SUMMARY_METRICS[metric],
                bootstrap_samples=config.reporting.bootstrap_samples,
                confidence_level=config.reporting.confidence_level,
                seed=hash(("prompt", metric)) & 0xFFFF,
            )
            for metric in REPLACEMENT_METRICS
        },
        "circuit": {
            metric: paired_bootstrap_summary(
                [
                    record
                    for record in primary_records
                    if record.get("record_type") == "prompt_metric" and record.get("status", "ok") == "ok"
                ],
                [
                    record
                    for record in baseline_records
                    if record.get("record_type") == "prompt_metric" and record.get("status", "ok") == "ok"
                ],
                metric=metric,
                pair_keys=["seed", "prompt_id"],
                higher_is_better=SUMMARY_METRICS[metric],
                bootstrap_samples=config.reporting.bootstrap_samples,
                confidence_level=config.reporting.confidence_level,
                seed=hash(("circuit", metric)) & 0xFFFF,
            )
            for metric in [
                "graph_replacement_score",
                "graph_completeness_score",
                "keep_only_gap_ratio",
                "gap_drop_ratio",
                "retained_error_node_count",
                "retained_error_node_fraction",
                "shapley_causal_jaccard",
            ]
        },
        "macag_game1": {
            metric: paired_bootstrap_summary(
                [
                    record
                    for record in primary_records
                    if record.get("record_type") == "macag_game1" and record.get("status", "ok") == "ok"
                ],
                [
                    record
                    for record in baseline_records
                    if record.get("record_type") == "macag_game1" and record.get("status", "ok") == "ok"
                ],
                metric=metric,
                pair_keys=["seed", "prompt_id", "run_name"],
                higher_is_better=SUMMARY_METRICS[metric],
                bootstrap_samples=config.reporting.bootstrap_samples,
                confidence_level=config.reporting.confidence_level,
                seed=hash(("macag_game1", metric)) & 0xFFFF,
            )
            for metric in ["evidence_size", "faithfulness", "utility"]
        },
    }

    primary_metric_names = [
        ("reconstruction", "mse_total"),
        ("reconstruction", "cosine_similarity"),
        ("reconstruction", "relative_error"),
        ("prompt", "top1_match_rate"),
        ("prompt", "kl_divergence"),
        ("circuit", "graph_replacement_score"),
        ("circuit", "graph_completeness_score"),
        ("circuit", "keep_only_gap_ratio"),
        ("circuit", "gap_drop_ratio"),
    ]
    primary_better_count = 0
    seed_consistent_count = 0
    for section_name, metric_name in primary_metric_names:
        comparison = comparisons[section_name][metric_name]
        if comparison["winner"] == "left":
            primary_better_count += 1
        seed_consistency = comparison.get("seed_consistency", {})
        if seed_consistency and all(value > 0 for value in seed_consistency.values()):
            seed_consistent_count += 1

    macag_family_wins = 0
    all_families = sorted(
        set(variants[primary]["macag"]["game1"]["by_family"]) & set(variants[baseline]["macag"]["game1"]["by_family"])
    )
    for family in all_families:
        primary_family = variants[primary]["macag"]["game1"]["by_family"][family]
        baseline_family = variants[baseline]["macag"]["game1"]["by_family"][family]
        primary_size = primary_family["evidence_size"]["mean"]
        baseline_size = baseline_family["evidence_size"]["mean"]
        primary_faithfulness = primary_family["faithfulness"]["mean"]
        baseline_faithfulness = baseline_family["faithfulness"]["mean"]
        if (
            math.isfinite(primary_size)
            and math.isfinite(baseline_size)
            and primary_size < baseline_size
        ) or (
            math.isfinite(primary_faithfulness)
            and math.isfinite(baseline_faithfulness)
            and primary_faithfulness > baseline_faithfulness
        ):
            macag_family_wins += 1

    inclusion_gate = {
        "primary_variant": primary,
        "baseline_variant": baseline,
        "primary_metric_better_count": primary_better_count,
        "primary_metric_total": len(primary_metric_names),
        "direction_consistent_metric_count": seed_consistent_count,
        "macag_family_win_count": macag_family_wins,
        "macag_family_total": len(all_families),
        "include_macag_in_main_text": (
            primary_better_count >= math.ceil(len(primary_metric_names) / 2)
            and seed_consistent_count >= math.ceil(len(primary_metric_names) / 2)
            and macag_family_wins >= 2
        ),
    }

    return {
        "suite_name": config.suite_name,
        "description": config.description,
        "benchmark_manifest_version": config.benchmark_manifest_version,
        "benchmark_size": len(config.benchmark_entries),
        "variants": variants,
        "comparisons": comparisons,
        "inclusion_gate": inclusion_gate,
    }


def build_tables_csv_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the aggregate metrics into fixed conference-style table rows."""
    rows: list[dict[str, Any]] = []
    for variant_name, variant_summary in aggregate["variants"].items():
        label = variant_summary["label"]
        for metric_name, metric_summary in variant_summary["reconstruction"].items():
            rows.append(
                {
                    "table": "Table 1",
                    "variant": variant_name,
                    "label": label,
                    "family": "",
                    "metric": metric_name,
                    "mean": metric_summary["mean"],
                    "std": metric_summary["std"],
                    "ci_low": metric_summary["ci_low"],
                    "ci_high": metric_summary["ci_high"],
                }
            )
        for metric_name, metric_summary in variant_summary["replacement"].items():
            rows.append(
                {
                    "table": "Table 1",
                    "variant": variant_name,
                    "label": label,
                    "family": "",
                    "metric": metric_name,
                    "mean": metric_summary["mean"],
                    "std": metric_summary["std"],
                    "ci_low": metric_summary["ci_low"],
                    "ci_high": metric_summary["ci_high"],
                }
            )
        for family, family_summary in variant_summary["circuit"]["by_family"].items():
            for metric_name, metric_summary in family_summary.items():
                rows.append(
                    {
                        "table": "Table 2",
                        "variant": variant_name,
                        "label": label,
                        "family": family,
                        "metric": metric_name,
                        "mean": metric_summary["mean"],
                        "std": metric_summary["std"],
                        "ci_low": metric_summary["ci_low"],
                        "ci_high": metric_summary["ci_high"],
                    }
                )
        for game_name in ["game1", "game2"]:
            for family, family_summary in variant_summary["macag"][game_name]["by_family"].items():
                for metric_name, metric_summary in family_summary.items():
                    rows.append(
                        {
                            "table": "Table 3",
                            "variant": variant_name,
                            "label": label,
                            "family": family,
                            "metric": f"{game_name}:{metric_name}",
                            "mean": metric_summary["mean"],
                            "std": metric_summary["std"],
                            "ci_low": metric_summary["ci_low"],
                            "ci_high": metric_summary["ci_high"],
                        }
                    )
    return rows


def write_tables_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["table", "variant", "label", "family", "metric", "mean", "std", "ci_low", "ci_high"]
    with resolved.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_report_markdown(aggregate: dict[str, Any], config: SuiteConfig) -> str:
    """Create a concise suite-level markdown report."""
    inclusion_gate = aggregate["inclusion_gate"]
    lines = [
        f"# {config.suite_name}",
        "",
        config.description or "Config-driven paper evaluation suite.",
        "",
        "## Summary",
        f"- Benchmark manifest: `{aggregate['benchmark_manifest_version']}`",
        f"- Benchmark size: {aggregate['benchmark_size']} prompts",
        f"- Variants: {', '.join(aggregate['variants'].keys())}",
        f"- Seeds: {', '.join(str(seed) for seed in config.seeds)}",
        f"- MACAG main-text recommendation: {'include' if inclusion_gate['include_macag_in_main_text'] else 'appendix'}",
        "",
        "## Main Comparisons",
    ]

    for section_name, section in aggregate["comparisons"].items():
        lines.append(f"### {section_name.title()}")
        for metric_name, comparison in section.items():
            lines.append(
                "- "
                f"{metric_name}: advantage={comparison['mean_advantage']:.4f}, "
                f"95% CI [{comparison['ci_low']:.4f}, {comparison['ci_high']:.4f}], "
                f"wins={comparison['wins']}, losses={comparison['losses']}, "
                f"paired={comparison['paired_count']}"
            )
        lines.append("")

    lines.extend(
        [
            "## Variant Highlights",
        ]
    )
    for variant_name, variant_summary in aggregate["variants"].items():
        lines.extend(
            [
                f"### {variant_summary['label']} (`{variant_name}`)",
                (
                    "- Reconstruction: "
                    f"MSE={variant_summary['reconstruction']['mse_total']['mean']:.6f}, "
                    f"cos={variant_summary['reconstruction']['cosine_similarity']['mean']:.4f}, "
                    f"rel_err={variant_summary['reconstruction']['relative_error']['mean']:.4f}"
                ),
                (
                    "- Replacement: "
                    f"top1={variant_summary['replacement']['top1_match_rate']['mean']:.4f}, "
                    f"top5={variant_summary['replacement']['top5_match_rate']['mean']:.4f}, "
                    f"top10={variant_summary['replacement']['top10_match_rate']['mean']:.4f}, "
                    f"KL={variant_summary['replacement']['kl_divergence']['mean']:.4f}"
                ),
                (
                    "- Circuit: "
                    f"repl_score={variant_summary['circuit']['overall']['graph_replacement_score']['mean']:.4f}, "
                    f"complete={variant_summary['circuit']['overall']['graph_completeness_score']['mean']:.4f}, "
                    f"err_frac={variant_summary['circuit']['overall']['retained_error_node_fraction']['mean']:.4f}, "
                    f"keep_only={variant_summary['circuit']['overall']['keep_only_gap_ratio']['mean']:.4f}"
                ),
                (
                    "- Monosemanticity: "
                    f"mean_gini={variant_summary['monosemanticity']['mean_gini']:.4f}, "
                    f"frac_gini>0.8={variant_summary['monosemanticity']['fraction_gini_gt_0_8']:.4f}"
                ),
                (
                    "- Diagnostics: "
                    f"errors={variant_summary['diagnostics']['error_count']}, "
                    f"MACAG stability={variant_summary['macag']['stability']['mean_jaccard']:.4f}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Inclusion Gate",
            (
                "- Primary metrics better: "
                f"{inclusion_gate['primary_metric_better_count']}/{inclusion_gate['primary_metric_total']}"
            ),
            (
                "- Direction-consistent metrics across seeds: "
                f"{inclusion_gate['direction_consistent_metric_count']}/{inclusion_gate['primary_metric_total']}"
            ),
            (
                "- MACAG family wins: "
                f"{inclusion_gate['macag_family_win_count']}/{inclusion_gate['macag_family_total']}"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def build_figure_manifest(aggregate: dict[str, Any], config: SuiteConfig) -> dict[str, Any]:
    """Emit the figure spec and source pointers for downstream plotting."""
    return {
        "figure_1": {
            "title": "Reconstruction vs replacement fidelity",
            "source_table": "Table 1",
            "x_metric": "mse_total",
            "y_metric": "top1_match_rate",
        },
        "figure_2": {
            "title": "Monosemanticity distribution",
            "source": "runs/*/seed_*/evaluation/monosemanticity_records.jsonl",
            "metric": "gini_coefficient",
        },
        "figure_3": {
            "title": "Representative spline curves",
            "source": "runs/*/seed_*/evaluation/splines/*.csv",
        },
        "figure_4": {
            "title": "Annotated MACAG case study graph",
            "prompt_id": config.reporting.figure_case_study_prompt_id,
            "source": "runs/*/seed_*/macag/*/*annotated.json",
        },
        "recommendation": aggregate["inclusion_gate"],
    }
