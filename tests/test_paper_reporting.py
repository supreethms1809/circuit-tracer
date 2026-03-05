"""Tests for paper reporting and aggregation helpers."""

from __future__ import annotations

from spline_clt.paper.config import SuiteConfig
from spline_clt.paper.reporting import aggregate_suite_records, build_tables_csv_rows


def make_test_suite() -> SuiteConfig:
    return SuiteConfig.model_validate(
        {
            "suite_name": "report_suite",
            "description": "report test",
            "output_root": "results/paper",
            "seeds": [1, 2],
            "dataset": {
                "model_name": "gpt2",
                "dataset_name": "Salesforce/wikitext",
                "dataset_config": "wikitext-103-raw-v1",
                "n_tokens": 100,
                "seq_len": 16,
                "batch_size": 1,
                "cache_dir": "shared/activations",
                "max_train_samples": 4,
                "eval_samples": 2,
                "dtype": "float32",
                "device": None,
                "seed": 0,
            },
            "model_variants": {
                "spline_default_gpt2": {
                    "label": "Spline-CLT",
                    "training": {
                        "d_transcoder": 8,
                        "encoder_type": "kan",
                    },
                },
                "linear_param_match_gpt2": {
                    "label": "Linear CLT",
                    "training": {
                        "d_transcoder": 8,
                        "encoder_type": "linear",
                    },
                },
            },
            "benchmark_manifest_version": "report-v1",
            "benchmark_entries": [
                {
                    "prompt_id": "p1",
                    "family": "ioi",
                    "prompt": "John gave Mary the book. She handed it to",
                    "target_token": " Mary",
                    "foil_token": " John",
                    "split": "main",
                    "include_macag": True,
                },
                {
                    "prompt_id": "p2",
                    "family": "factual",
                    "prompt": "The capital of France is",
                    "target_token": " Paris",
                    "foil_token": " Lyon",
                    "split": "main",
                    "include_macag": True,
                },
            ],
            "reporting": {
                "primary_variant": "spline_default_gpt2",
                "baseline_variant": "linear_param_match_gpt2",
                "bootstrap_samples": 100,
                "confidence_level": 0.95,
            },
        }
    )


def test_aggregate_suite_records_handles_partial_macag_failures() -> None:
    suite = make_test_suite()
    common = {
        "suite_name": suite.suite_name,
        "benchmark_manifest_version": suite.benchmark_manifest_version,
        "checkpoint_path": "/tmp/checkpoint",
        "dtype": "float32",
        "git_commit": "deadbeef",
        "model_name": "gpt2",
    }
    records = [
        common
        | {
            "record_type": "reconstruction_sample",
            "variant_name": "spline_default_gpt2",
            "variant_label": "Spline-CLT",
            "seed": 1,
            "sample_index": 0,
            "mse_total": 0.8,
            "cosine_similarity": 0.7,
            "relative_error": 0.3,
            "active_per_pos": 4.0,
            "activation_density": 0.1,
        },
        common
        | {
            "record_type": "reconstruction_sample",
            "variant_name": "linear_param_match_gpt2",
            "variant_label": "Linear CLT",
            "seed": 1,
            "sample_index": 0,
            "mse_total": 1.1,
            "cosine_similarity": 0.6,
            "relative_error": 0.4,
            "active_per_pos": 5.0,
            "activation_density": 0.2,
        },
        common
        | {
            "record_type": "prompt_metric",
            "variant_name": "spline_default_gpt2",
            "variant_label": "Spline-CLT",
            "seed": 1,
            "status": "ok",
            "prompt_id": "p1",
            "family": "ioi",
            "top1_match_rate": 0.8,
            "kl_divergence": 0.2,
            "active_feature_count": 12,
            "retained_feature_node_count": 9,
            "retained_error_node_count": 6,
            "retained_error_node_fraction": 0.4,
            "graph_replacement_score": 0.72,
            "graph_completeness_score": 0.81,
            "keep_only_gap_ratio": 0.7,
            "gap_drop_ratio": 0.4,
            "shapley_causal_jaccard": 0.5,
        },
        common
        | {
            "record_type": "prompt_metric",
            "variant_name": "linear_param_match_gpt2",
            "variant_label": "Linear CLT",
            "seed": 1,
            "status": "ok",
            "prompt_id": "p1",
            "family": "ioi",
            "top1_match_rate": 0.6,
            "kl_divergence": 0.3,
            "active_feature_count": 15,
            "retained_feature_node_count": 8,
            "retained_error_node_count": 10,
            "retained_error_node_fraction": 10 / 18,
            "graph_replacement_score": 0.51,
            "graph_completeness_score": 0.62,
            "keep_only_gap_ratio": 0.5,
            "gap_drop_ratio": 0.3,
            "shapley_causal_jaccard": 0.2,
        },
        common
        | {
            "record_type": "monosemanticity_feature",
            "variant_name": "spline_default_gpt2",
            "variant_label": "Spline-CLT",
            "seed": 1,
            "layer": 0,
            "feature_id": 0,
            "activation_frequency": 0.1,
            "mean_activation": 0.5,
            "max_activation": 1.0,
            "gini_coefficient": 0.9,
        },
        common
        | {
            "record_type": "monosemanticity_feature",
            "variant_name": "linear_param_match_gpt2",
            "variant_label": "Linear CLT",
            "seed": 1,
            "layer": 0,
            "feature_id": 0,
            "activation_frequency": 0.2,
            "mean_activation": 0.4,
            "max_activation": 0.8,
            "gini_coefficient": 0.6,
        },
        common
        | {
            "record_type": "macag_game1",
            "variant_name": "spline_default_gpt2",
            "variant_label": "Spline-CLT",
            "seed": 1,
            "status": "ok",
            "prompt_id": "p1",
            "family": "ioi",
            "run_name": "faithful_eps_0_10",
            "evidence_size": 3,
            "faithfulness": 0.7,
            "sufficiency": 0.6,
            "necessity": 0.8,
            "utility": 0.67,
            "oracle_calls": 20,
            "cache_hits": 4,
            "cache_size": 16,
            "evidence_nodes": ["a", "b", "c"],
        },
        common
        | {
            "record_type": "macag_game1",
            "variant_name": "linear_param_match_gpt2",
            "variant_label": "Linear CLT",
            "seed": 1,
            "status": "ok",
            "prompt_id": "p1",
            "family": "ioi",
            "run_name": "faithful_eps_0_10",
            "evidence_size": 5,
            "faithfulness": 0.5,
            "sufficiency": 0.4,
            "necessity": 0.6,
            "utility": 0.45,
            "oracle_calls": 22,
            "cache_hits": 3,
            "cache_size": 19,
            "evidence_nodes": ["a", "d", "e", "f", "g"],
        },
        common
        | {
            "record_type": "macag_error",
            "variant_name": "linear_param_match_gpt2",
            "variant_label": "Linear CLT",
            "seed": 2,
            "status": "error",
            "prompt_id": "p2",
            "family": "factual",
            "error": "empty candidate set",
        },
    ]

    aggregate = aggregate_suite_records(records, suite)
    assert aggregate["inclusion_gate"]["primary_variant"] == "spline_default_gpt2"
    assert aggregate["comparisons"]["prompt"]["top1_match_rate"]["paired_count"] == 1
    assert aggregate["comparisons"]["circuit"]["graph_replacement_score"]["paired_count"] == 1
    assert (
        aggregate["variants"]["spline_default_gpt2"]["circuit"]["overall"]["retained_error_node_fraction"]["mean"]
        == 0.4
    )
    assert aggregate["variants"]["linear_param_match_gpt2"]["diagnostics"]["error_count"] == 1

    rows = build_tables_csv_rows(aggregate)
    assert rows
    assert any(row["table"] == "Table 1" for row in rows)
    assert any(row["table"] == "Table 3" for row in rows)
    assert any(row["table"] == "Table 2" and row["metric"] == "graph_replacement_score" for row in rows)
