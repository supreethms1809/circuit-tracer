"""Tests for paper-suite config loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spline_clt.paper.config import load_suite_config


REPO_ROOT = Path(__file__).resolve().parents[1]
SUITES_DIR = REPO_ROOT / "experiments" / "paper_configs" / "suites"


def test_all_paper_suites_load_and_validate() -> None:
    expected_sizes = {
        "paper_gpt2_small.json": 20,
        "paper_gpt2_large.json": 20,
        "paper_qwen3_06b.json": 20,
        "paper_gpt2_small_dt_sweep.json": 20,
    }

    for suite_file, expected_size in expected_sizes.items():
        suite, resolved = load_suite_config(SUITES_DIR / suite_file)
        assert suite.suite_name
        assert len(suite.benchmark_entries) == expected_size
        assert resolved["benchmark_manifest_version"] == suite.benchmark_manifest_version
        assert suite.reporting.primary_variant in suite.model_variants
        assert suite.reporting.baseline_variant in suite.model_variants


def test_duplicate_prompt_ids_fail_validation(tmp_path: Path) -> None:
    suite_payload = {
        "suite_name": "bad_suite",
        "description": "Duplicate prompt ids should fail.",
        "output_root": "results/paper",
        "seeds": [1],
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
        "benchmark_manifest_version": "dup-test-v1",
        "benchmark_entries": [
            {
                "prompt_id": "dup",
                "family": "ioi",
                "prompt": "John gave Mary the book. She handed it to",
                "target_token": " Mary",
                "foil_token": " John",
                "split": "main",
                "include_macag": True,
            },
            {
                "prompt_id": "dup",
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
        },
    }
    suite_path = tmp_path / "bad_suite.json"
    suite_path.write_text(json.dumps(suite_payload))

    with pytest.raises(ValueError, match="prompt_ids"):
        load_suite_config(suite_path)
