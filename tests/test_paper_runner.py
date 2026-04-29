"""Tests for the config-driven paper runner."""

from __future__ import annotations

import json
from pathlib import Path

from spline_clt.paper.runner import PaperSuiteRunner
from spline_clt.paper.reporting import write_json, write_jsonl
from spline_clt.paper.evaluate import deterministic_sample_indices
from spline_clt.training.data import ActivationDataset
from spline_clt.training.train import split_dataset

import torch


def test_split_dataset_and_sampling_are_deterministic() -> None:
    mlp_inputs = torch.randn(10, 2, 4, 3)
    dataset = ActivationDataset(mlp_inputs=mlp_inputs, mlp_outputs=mlp_inputs.clone())

    train_a, val_a = split_dataset(dataset, val_fraction=0.2, seed=7)
    train_b, val_b = split_dataset(dataset, val_fraction=0.2, seed=7)

    assert train_a.indices == train_b.indices
    assert val_a.indices == val_b.indices
    assert deterministic_sample_indices(10, 4, seed=3) == deterministic_sample_indices(10, 4, seed=3)


def test_paper_runner_dry_run_expands_expected_jobs() -> None:
    suite_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "paper_configs"
        / "suites"
        / "paper_gpt2_small.json"
    )
    runner = PaperSuiteRunner(suite_path)
    payload = runner.dry_run()

    assert payload["suite"]["suite_name"] == "paper_gpt2_small"
    # 1 collect_dataset + 4 variants × 3 seeds × (train + evaluate + macag) + 1 report
    assert len(payload["jobs"]) == 1 + 4 * 3 * 3 + 1
    assert payload["jobs"][0]["stage"] == "collect_dataset"
    assert payload["jobs"][-1]["stage"] == "report"


def test_paper_runner_smoke_writes_full_artifact_tree(tmp_path: Path, monkeypatch) -> None:
    suite_payload = {
        "suite_name": "smoke_suite",
        "description": "smoke test",
        "output_root": str(tmp_path / "out"),
        "seeds": [1],
        "dataset": {
            "model_name": "gpt2",
            "dataset_name": "Salesforce/wikitext",
            "dataset_config": "wikitext-103-raw-v1",
            "n_tokens": 100,
            "seq_len": 16,
            "batch_size": 1,
            "cache_dir": "shared/activations",
            "val_cache_dir": "shared/activations_val",
            "val_fraction": 0.25,
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
        "benchmark_manifest_version": "smoke-v1",
        "benchmark_entries": [
            {
                "prompt_id": "p1",
                "family": "ioi",
                "prompt": "John gave Mary the book. She handed it to",
                "target_token": " Mary",
                "foil_token": " John",
                "split": "main",
                "include_macag": True,
            }
        ],
        "macag": {
            "enabled": True,
            "annotate_graphs": False,
            "game1_runs": [{"name": "faithful_eps_0_10"}],
            "game2_runs": [{"name": "contrastive_default"}],
        },
        "reporting": {
            "primary_variant": "spline_default_gpt2",
            "baseline_variant": "linear_param_match_gpt2",
            "bootstrap_samples": 20,
            "confidence_level": 0.95,
            "figure_case_study_prompt_id": "p1",
        },
    }
    suite_path = tmp_path / "smoke_suite.json"
    suite_path.write_text(json.dumps(suite_payload))

    runner = PaperSuiteRunner(suite_path)

    def fake_collect_dataset(self: PaperSuiteRunner, model_name: str):
        dataset_dir = self._dataset_dir(model_name)
        val_dataset_dir = self._val_dataset_dir(model_name)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        val_dataset_dir.mkdir(parents=True, exist_ok=True)
        # Match the new split layout so _dataset_stage_complete recognises it.
        (dataset_dir / "mlp_inputs_train.npy").write_bytes(b"")
        (dataset_dir / "mlp_outputs_train.npy").write_bytes(b"")
        (val_dataset_dir / "mlp_inputs_val.npy").write_bytes(b"")
        (val_dataset_dir / "mlp_outputs_val.npy").write_bytes(b"")
        payload = {
            "status": "completed",
            "model_name": model_name,
            "dataset_dir": str(dataset_dir),
            "val_dataset_dir": str(val_dataset_dir),
        }
        write_json(dataset_dir / "dataset_manifest.json", payload)
        return payload

    def fake_train_variant_seed(self: PaperSuiteRunner, variant_name, variant, seed):
        run_dir = self._run_dir(variant_name, seed)
        checkpoint_path = run_dir / "checkpoints" / f"{variant_name}_best"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        (checkpoint_path / "metadata.safetensors").write_bytes(b"")
        payload = {
            "status": "completed",
            "variant_name": variant_name,
            "seed": seed,
            "checkpoint_path": str(checkpoint_path),
        }
        write_json(run_dir / "train_summary.json", payload)
        self._write_run_manifest(run_dir, variant_name, variant, seed, str(checkpoint_path))
        return checkpoint_path

    def fake_evaluate_variant_seed(self: PaperSuiteRunner, variant_name, variant, seed, checkpoint_path):
        run_dir = self._run_dir(variant_name, seed)
        evaluation_dir = run_dir / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        base_record = self._record_base(
            variant_name=variant_name,
            variant=variant,
            seed=seed,
            checkpoint_path=checkpoint_path,
        )
        recon_records = [
            base_record
            | {
                "record_type": "reconstruction_sample",
                "sample_index": 0,
                "mse_total": 0.7 if variant_name == "spline_default_gpt2" else 1.0,
                "cosine_similarity": 0.8 if variant_name == "spline_default_gpt2" else 0.6,
                "relative_error": 0.2 if variant_name == "spline_default_gpt2" else 0.4,
                "active_per_pos": 4.0,
                "activation_density": 0.1,
            }
        ]
        prompt_records = [
            base_record
            | {
                "record_type": "prompt_metric",
                "status": "ok",
                "prompt_id": "p1",
                "family": "ioi",
                "prompt": "John gave Mary the book. She handed it to",
                "target_token": " Mary",
                "foil_token": " John",
                "include_macag": True,
                "graph_json_path": str(evaluation_dir / "graphs" / "p1.json"),
                "graph_pt_path": str(evaluation_dir / "graphs" / "p1.pt"),
                "active_feature_count": 5,
                "full_logit_gap": 1.0,
                "keep_only_gap": 0.8 if variant_name == "spline_default_gpt2" else 0.5,
                "remove_gap": 0.3 if variant_name == "spline_default_gpt2" else 0.6,
                "keep_only_gap_ratio": 0.8 if variant_name == "spline_default_gpt2" else 0.5,
                "gap_drop": 0.7 if variant_name == "spline_default_gpt2" else 0.4,
                "gap_drop_ratio": 0.7 if variant_name == "spline_default_gpt2" else 0.4,
                "shapley_causal_jaccard": 0.5,
                "top1_match_rate": 0.9 if variant_name == "spline_default_gpt2" else 0.6,
                "kl_divergence": 0.1 if variant_name == "spline_default_gpt2" else 0.3,
            }
        ]
        mono_records = [
            base_record
            | {
                "record_type": "monosemanticity_feature",
                "layer": 0,
                "feature_id": 0,
                "activation_frequency": 0.1,
                "mean_activation": 0.4,
                "max_activation": 1.0,
                "gini_coefficient": 0.85 if variant_name == "spline_default_gpt2" else 0.6,
            }
        ]
        (evaluation_dir / "graphs").mkdir(parents=True, exist_ok=True)
        write_jsonl(evaluation_dir / "reconstruction_records.jsonl", recon_records)
        write_jsonl(evaluation_dir / "prompt_metrics.jsonl", prompt_records)
        write_jsonl(evaluation_dir / "monosemanticity_records.jsonl", mono_records)
        summary = {
            "status": "completed",
            "variant_name": variant_name,
            "seed": seed,
            "checkpoint_path": str(checkpoint_path),
        }
        write_json(evaluation_dir / "evaluation_summary.json", summary)
        return summary

    def fake_run_macag_for_variant_seed(self: PaperSuiteRunner, variant_name, variant, seed, checkpoint_path):
        run_dir = self._run_dir(variant_name, seed)
        macag_dir = run_dir / "macag"
        macag_dir.mkdir(parents=True, exist_ok=True)
        base_record = self._record_base(
            variant_name=variant_name,
            variant=variant,
            seed=seed,
            checkpoint_path=checkpoint_path,
        )
        records = [
            base_record
            | {
                "record_type": "macag_game1",
                "status": "ok",
                "prompt_id": "p1",
                "family": "ioi",
                "run_name": "faithful_eps_0_10",
                "candidate_strategy": "graph_features",
                "candidate_count": 5,
                "evidence_size": 2 if variant_name == "spline_default_gpt2" else 4,
                "faithfulness": 0.8 if variant_name == "spline_default_gpt2" else 0.5,
                "sufficiency": 0.7,
                "necessity": 0.9,
                "utility": 0.78 if variant_name == "spline_default_gpt2" else 0.46,
                "oracle_calls": 10,
                "cache_hits": 2,
                "cache_size": 8,
                "sparsity": 0.6,
                "evidence_nodes": ["a", "b"] if variant_name == "spline_default_gpt2" else ["a", "c", "d", "e"],
                "result_json_path": str(macag_dir / "p1" / "game1.json"),
            }
        ]
        write_jsonl(macag_dir / "macag_records.jsonl", records)
        summary = {"status": "completed", "prompt_count": 1, "successful_records": 1, "error_count": 0}
        write_json(macag_dir / "macag_summary.json", summary)
        return summary

    monkeypatch.setattr(PaperSuiteRunner, "_collect_dataset", fake_collect_dataset)
    monkeypatch.setattr(PaperSuiteRunner, "_train_variant_seed", fake_train_variant_seed)
    monkeypatch.setattr(PaperSuiteRunner, "_evaluate_variant_seed", fake_evaluate_variant_seed)
    monkeypatch.setattr(PaperSuiteRunner, "_run_macag_for_variant_seed", fake_run_macag_for_variant_seed)

    result = runner.run()
    suite_root = Path(result["suite_root"])

    assert (suite_root / "resolved_config.json").exists()
    assert (suite_root / "manifest.json").exists()
    assert (suite_root / "per_example_metrics.jsonl").exists()
    assert (suite_root / "aggregate_metrics.json").exists()
    assert (suite_root / "tables.csv").exists()
    assert (suite_root / "report.md").exists()
    assert (suite_root / "figures" / "figure_manifest.json").exists()
