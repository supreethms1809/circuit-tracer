"""Config-driven paper evaluation runner."""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from attribution.shapley import shapley_logit_attribution
from eval.monosemanticity import collect_max_activating_examples
from experiments.analyze_splines import (
    collect_feature_stats,
    extract_spline_curve,
    save_curves_csv,
    top_input_dims,
)
from macag.cli.annotate_graph import annotate_graph_with_macag
from macag.factories.replacement_model import create_replacement_model_scorer
from macag.games.game1_min_faithful import solve_game1
from macag.games.game2_contrastive import solve_game2
from macag.graph import CircuitGraph
from macag.scoring import ScoringOracle
from macag.utils.metrics import metrics_to_dict
from macag.utils.supernodes import propose_supernodes, supernode_candidates
from spline_clt.kan_transcoder import load_spline_clt
from spline_clt.paper.config import BenchmarkEntry, ModelVariantConfig, SuiteConfig, load_suite_config
from spline_clt.paper.evaluate import (
    build_logit_gap_direction,
    build_prompt_graph,
    collect_prompt_cache,
    deterministic_sample_indices,
    evaluate_prompt_replacement,
    evaluate_reconstruction_samples,
    feature_node_id,
    jaccard_overlap,
    load_language_model,
    load_ranked_feature_nodes,
    parse_feature_node_id,
    replacement_logit_gap_from_subset,
)
from spline_clt.paper.reporting import (
    aggregate_suite_records,
    build_figure_manifest,
    build_report_markdown,
    build_tables_csv_rows,
    load_suite_records,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
    write_tables_csv,
)
from spline_clt.training.data import ActivationDataset, DataConfig, collect_activations
from spline_clt.training.train import TrainConfig, train


def _sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized.strip("_") or "item"


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def _device_from_name(name: str | None) -> torch.device:
    preferred = name or ("cuda" if torch.cuda.is_available() else "cpu")
    if preferred != "cpu" and not torch.cuda.is_available():
        preferred = "cpu"
    return torch.device(preferred)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit(cwd: str | Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _hardware_summary() -> dict[str, Any]:
    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0),
        }
    return {"device": "cpu"}


class PaperSuiteRunner:
    """Runs a paper evaluation suite from one resolved JSON config."""

    def __init__(self, suite_path: str | Path):
        self.suite_path = Path(suite_path).resolve()
        self.config, self.resolved_config = load_suite_config(self.suite_path)
        self.repo_root = self._find_repo_root(self.suite_path.parent)
        self.suite_root = (self.repo_root / self.config.output_root / self.config.suite_name).resolve()
        self.datasets_root = self.suite_root / self.config.dataset.cache_dir
        self.runs_root = self.suite_root / "runs"
        self.figures_root = self.suite_root / "figures"
        self.commit_hash = _git_commit(self.repo_root)
        self.hardware = _hardware_summary()

    @staticmethod
    def _find_repo_root(start: Path) -> Path:
        for candidate in [start, *start.parents]:
            if (candidate / ".git").exists():
                return candidate
            if (candidate / "pyproject.toml").exists():
                return candidate
        return start

    def validate(self) -> dict[str, Any]:
        return {
            "suite_name": self.config.suite_name,
            "suite_path": str(self.suite_path),
            "variant_names": list(self.config.model_variants.keys()),
            "seed_count": len(self.config.seeds),
            "benchmark_size": len(self.config.benchmark_entries),
            "output_root": str(self.suite_root),
        }

    def build_jobs(self) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for model_name in sorted({self._variant_model_name(variant) for variant in self.config.model_variants.values()}):
            jobs.append(
                {
                    "stage": "collect_dataset",
                    "model_name": model_name,
                    "dataset_dir": str(self._dataset_dir(model_name)),
                    "resume_ready": self._dataset_stage_complete(model_name),
                }
            )

        for variant_name, variant in self.config.model_variants.items():
            for seed in self.config.seeds:
                run_dir = self._run_dir(variant_name, seed)
                jobs.append(
                    {
                        "stage": "train",
                        "variant_name": variant_name,
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "resume_ready": self._train_stage_complete(variant_name, variant, seed),
                    }
                )
                jobs.append(
                    {
                        "stage": "evaluate",
                        "variant_name": variant_name,
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "resume_ready": (run_dir / "evaluation" / "evaluation_summary.json").exists(),
                    }
                )
                jobs.append(
                    {
                        "stage": "macag",
                        "variant_name": variant_name,
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "resume_ready": (run_dir / "macag" / "macag_summary.json").exists(),
                    }
                )
        jobs.append(
            {
                "stage": "report",
                "suite_root": str(self.suite_root),
                "resume_ready": (self.suite_root / "aggregate_metrics.json").exists(),
            }
        )
        return jobs

    def dry_run(self) -> dict[str, Any]:
        return {
            "suite": self.validate(),
            "jobs": self.build_jobs(),
        }

    def run(self) -> dict[str, Any]:
        self._prepare_suite_root()
        self._write_suite_manifest()

        if self.config.stages.collect_dataset:
            for model_name in sorted(
                {self._variant_model_name(variant) for variant in self.config.model_variants.values()}
            ):
                self._collect_dataset(model_name)

        for variant_name, variant in self.config.model_variants.items():
            for seed in self.config.seeds:
                checkpoint_path = self._train_variant_seed(variant_name, variant, seed)
                if self.config.stages.evaluate:
                    self._evaluate_variant_seed(
                        variant_name=variant_name,
                        variant=variant,
                        seed=seed,
                        checkpoint_path=checkpoint_path,
                    )
                if self.config.stages.macag and self.config.macag.enabled:
                    self._run_macag_for_variant_seed(
                        variant_name=variant_name,
                        variant=variant,
                        seed=seed,
                        checkpoint_path=checkpoint_path,
                    )

        aggregate = self._generate_reporting() if self.config.stages.report else {}
        return {
            "suite_root": str(self.suite_root),
            "aggregate_metrics_path": str(self.suite_root / "aggregate_metrics.json"),
            "aggregate": aggregate,
        }

    def _prepare_suite_root(self) -> None:
        self.suite_root.mkdir(parents=True, exist_ok=True)
        self.datasets_root.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.figures_root.mkdir(parents=True, exist_ok=True)

    def _write_suite_manifest(self) -> None:
        write_json(self.suite_root / "resolved_config.json", self.resolved_config)
        write_json(
            self.suite_root / "manifest.json",
            {
                "suite_name": self.config.suite_name,
                "suite_path": str(self.suite_path),
                "created_at": _utc_now(),
                "git_commit": self.commit_hash,
                "hardware": self.hardware,
                "benchmark_manifest_version": self.config.benchmark_manifest_version,
                "variants": list(self.config.model_variants.keys()),
                "seeds": self.config.seeds,
            },
        )

    def _variant_model_name(self, variant: ModelVariantConfig) -> str:
        return variant.model_name or self.config.dataset.model_name

    def _dataset_dir(self, model_name: str) -> Path:
        return self.datasets_root / _sanitize_name(model_name)

    def _run_dir(self, variant_name: str, seed: int) -> Path:
        return self.runs_root / _sanitize_name(variant_name) / f"seed_{seed}"

    def _variant_run_name(self, variant_name: str, variant: ModelVariantConfig, seed: int) -> str:
        base = variant.training.run_name or variant_name
        return f"{_sanitize_name(base)}_seed{seed}"

    def _best_checkpoint_dir(self, variant_name: str, variant: ModelVariantConfig, seed: int) -> Path:
        run_name = self._variant_run_name(variant_name, variant, seed)
        return self._run_dir(variant_name, seed) / "checkpoints" / f"{run_name}_best"

    def _final_checkpoint_dir(self, variant_name: str, variant: ModelVariantConfig, seed: int) -> Path:
        run_name = self._variant_run_name(variant_name, variant, seed)
        return self._run_dir(variant_name, seed) / "checkpoints" / f"{run_name}_final"

    def _dataset_stage_complete(self, model_name: str) -> bool:
        dataset_dir = self._dataset_dir(model_name)
        return (dataset_dir / "mlp_inputs.pt").exists() and (dataset_dir / "mlp_outputs.pt").exists()

    def _train_stage_complete(self, variant_name: str, variant: ModelVariantConfig, seed: int) -> bool:
        if variant.checkpoint_path:
            return Path(variant.checkpoint_path).exists()
        best_dir = self._best_checkpoint_dir(variant_name, variant, seed)
        final_dir = self._final_checkpoint_dir(variant_name, variant, seed)
        return (best_dir / "metadata.safetensors").exists() or (final_dir / "metadata.safetensors").exists()

    def _collect_dataset(self, model_name: str) -> dict[str, Any]:
        dataset_dir = self._dataset_dir(model_name)
        summary_path = dataset_dir / "dataset_manifest.json"
        if self._dataset_stage_complete(model_name):
            if summary_path.exists():
                return read_json(summary_path)
            summary = {
                "status": "resumed",
                "model_name": model_name,
                "dataset_dir": str(dataset_dir),
                "created_at": _utc_now(),
            }
            write_json(summary_path, summary)
            return summary

        dataset_dir.mkdir(parents=True, exist_ok=True)
        data_config = DataConfig(
            model_name=model_name,
            dataset_name=self.config.dataset.dataset_name,
            dataset_config=self.config.dataset.dataset_config,
            n_tokens=self.config.dataset.n_tokens,
            seq_len=self.config.dataset.seq_len,
            batch_size=self.config.dataset.batch_size,
            save_dir=str(dataset_dir),
            device=str(_device_from_name(self.config.dataset.device)),
            seed=self.config.dataset.seed,
        )
        collect_activations(data_config)
        summary = {
            "status": "completed",
            "model_name": model_name,
            "dataset_name": self.config.dataset.dataset_name,
            "dataset_config": self.config.dataset.dataset_config,
            "dataset_dir": str(dataset_dir),
            "n_tokens": self.config.dataset.n_tokens,
            "seq_len": self.config.dataset.seq_len,
            "seed": self.config.dataset.seed,
            "created_at": _utc_now(),
        }
        write_json(summary_path, summary)
        return summary

    def _resolve_checkpoint_path(
        self,
        variant_name: str,
        variant: ModelVariantConfig,
        seed: int,
    ) -> Path:
        if variant.checkpoint_path:
            return Path(variant.checkpoint_path).resolve()
        best_dir = self._best_checkpoint_dir(variant_name, variant, seed)
        final_dir = self._final_checkpoint_dir(variant_name, variant, seed)
        if (best_dir / "metadata.safetensors").exists():
            return best_dir
        return final_dir

    def _train_variant_seed(
        self,
        variant_name: str,
        variant: ModelVariantConfig,
        seed: int,
    ) -> Path:
        run_dir = self._run_dir(variant_name, seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "train_summary.json"

        if summary_path.exists():
            payload = read_json(summary_path)
            return Path(payload["checkpoint_path"])

        if variant.checkpoint_path:
            payload = {
                "status": "external_checkpoint",
                "variant_name": variant_name,
                "seed": seed,
                "checkpoint_path": str(Path(variant.checkpoint_path).resolve()),
                "created_at": _utc_now(),
            }
            write_json(summary_path, payload)
            self._write_run_manifest(run_dir, variant_name, variant, seed, payload["checkpoint_path"])
            return Path(payload["checkpoint_path"])

        dataset_dir = self._dataset_dir(self._variant_model_name(variant))
        dataset = ActivationDataset.load(
            str(dataset_dir),
            max_samples=max(self.config.dataset.max_train_samples, self.config.dataset.eval_samples + 64),
        )

        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        training = variant.training
        train_config = TrainConfig(
            n_layers=training.n_layers,
            d_model=training.d_model,
            d_transcoder=training.d_transcoder,
            encoder_type=training.encoder_type,
            grid_size=training.grid_size,
            spline_order=training.spline_order,
            learning_rate=training.learning_rate,
            warmup_steps=training.warmup_steps,
            total_steps=training.total_steps,
            batch_size=training.batch_size,
            lambda_sparsity=training.lambda_sparsity,
            c_sparsity=training.c_sparsity,
            grad_clip=training.grad_clip,
            log_every=training.log_every,
            eval_every=training.eval_every,
            save_every=training.save_every,
            checkpoint_dir=str(checkpoint_dir),
            run_name=self._variant_run_name(variant_name, variant, seed),
            update_grid_every=training.update_grid_every,
            update_grid_from=training.update_grid_from,
            reset_optimizer_every=training.reset_optimizer_every,
            data_dir=str(dataset_dir),
            device=str(_device_from_name(training.device or self.config.dataset.device)),
            dtype=training.dtype,
            seed=seed,
            val_fraction=training.val_fraction,
        )

        train(train_config, dataset=dataset)
        checkpoint_path = self._resolve_checkpoint_path(variant_name, variant, seed)
        payload = {
            "status": "completed",
            "variant_name": variant_name,
            "seed": seed,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_dir": str(checkpoint_dir),
            "created_at": _utc_now(),
        }
        write_json(summary_path, payload)
        self._write_run_manifest(run_dir, variant_name, variant, seed, str(checkpoint_path))
        return checkpoint_path

    def _write_run_manifest(
        self,
        run_dir: Path,
        variant_name: str,
        variant: ModelVariantConfig,
        seed: int,
        checkpoint_path: str,
    ) -> None:
        write_json(
            run_dir / "manifest.json",
            {
                "suite_name": self.config.suite_name,
                "variant_name": variant_name,
                "variant_label": variant.label,
                "seed": seed,
                "model_name": self._variant_model_name(variant),
                "checkpoint_path": checkpoint_path,
                "git_commit": self.commit_hash,
                "hardware": self.hardware,
                "dtype": variant.training.dtype,
                "benchmark_manifest_version": self.config.benchmark_manifest_version,
                "timestamp": _utc_now(),
            },
        )

    def _record_base(
        self,
        *,
        variant_name: str,
        variant: ModelVariantConfig,
        seed: int,
        checkpoint_path: Path,
    ) -> dict[str, Any]:
        return {
            "suite_name": self.config.suite_name,
            "variant_name": variant_name,
            "variant_label": variant.label,
            "seed": seed,
            "model_name": self._variant_model_name(variant),
            "checkpoint_path": str(checkpoint_path),
            "benchmark_manifest_version": self.config.benchmark_manifest_version,
            "dtype": variant.training.dtype,
            "git_commit": self.commit_hash,
        }

    def _evaluate_variant_seed(
        self,
        *,
        variant_name: str,
        variant: ModelVariantConfig,
        seed: int,
        checkpoint_path: Path,
    ) -> dict[str, Any]:
        run_dir = self._run_dir(variant_name, seed)
        evaluation_dir = run_dir / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        summary_path = evaluation_dir / "evaluation_summary.json"
        if summary_path.exists():
            return read_json(summary_path)

        device = _device_from_name(variant.training.device or self.config.dataset.device)
        dtype = _dtype_from_name(variant.training.dtype)
        dataset_dir = self._dataset_dir(self._variant_model_name(variant))
        dataset = ActivationDataset.load(
            str(dataset_dir),
            max_samples=max(self.config.dataset.max_train_samples, self.config.dataset.eval_samples + 64),
        )
        model = load_spline_clt(str(checkpoint_path), device=device, dtype=dtype)
        model.eval()

        base_record = self._record_base(
            variant_name=variant_name,
            variant=variant,
            seed=seed,
            checkpoint_path=checkpoint_path,
        )

        sample_indices = deterministic_sample_indices(
            total=len(dataset),
            count=self.config.dataset.eval_samples,
            seed=seed,
        )
        reconstruction_summary, reconstruction_records = evaluate_reconstruction_samples(
            model=model,
            dataset=dataset,
            device=device,
            dtype=dtype,
            sample_indices=sample_indices,
        )
        reconstruction_records = [
            base_record | record
            for record in reconstruction_records
        ]
        write_jsonl(evaluation_dir / "reconstruction_records.jsonl", reconstruction_records)

        lm = load_language_model(self._variant_model_name(variant), device=device)
        prompt_records: list[dict[str, Any]] = []
        graphs_dir = evaluation_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        for entry in self.config.benchmark_entries:
            prompt_records.append(
                self._evaluate_prompt_entry(
                    model=model,
                    lm=lm,
                    entry=entry,
                    variant_name=variant_name,
                    variant=variant,
                    seed=seed,
                    checkpoint_path=checkpoint_path,
                    graphs_dir=graphs_dir,
                )
            )
        write_jsonl(evaluation_dir / "prompt_metrics.jsonl", prompt_records)

        monosemanticity_records, monosemanticity_summary = self._collect_monosemanticity(
            model=model,
            dataset=dataset,
            base_record=base_record,
            device=device,
        )
        write_jsonl(evaluation_dir / "monosemanticity_records.jsonl", monosemanticity_records)

        spline_summary = self._collect_spline_analysis(
            model=model,
            dataset=dataset,
            evaluation_dir=evaluation_dir,
            device=device,
        )

        completed_prompts = [
            record for record in prompt_records if record.get("status", "ok") == "ok"
        ]
        prompt_metric = lambda key: (
            sum(float(record[key]) for record in completed_prompts if record.get(key) is not None)
            / max(1, sum(1 for record in completed_prompts if record.get(key) is not None))
        )
        evaluation_summary = {
            "suite_name": self.config.suite_name,
            "variant_name": variant_name,
            "seed": seed,
            "checkpoint_path": str(checkpoint_path),
            "reconstruction": reconstruction_summary,
            "prompt_metrics": {
                "count": len(completed_prompts),
                "error_count": sum(1 for record in prompt_records if record.get("status") == "error"),
                "top1_match_rate": prompt_metric("top1_match_rate"),
                "kl_divergence": prompt_metric("kl_divergence"),
            },
            "circuit_metrics": {
                "active_feature_count": prompt_metric("active_feature_count"),
                "retained_feature_node_count": prompt_metric("retained_feature_node_count"),
                "retained_error_node_count": prompt_metric("retained_error_node_count"),
                "retained_error_node_fraction": prompt_metric("retained_error_node_fraction"),
                "graph_replacement_score": prompt_metric("graph_replacement_score"),
                "graph_completeness_score": prompt_metric("graph_completeness_score"),
                "keep_only_gap_ratio": prompt_metric("keep_only_gap_ratio"),
                "gap_drop_ratio": prompt_metric("gap_drop_ratio"),
                "shapley_causal_jaccard": prompt_metric("shapley_causal_jaccard"),
            },
            "monosemanticity": monosemanticity_summary,
            "splines": spline_summary,
            "created_at": _utc_now(),
        }
        write_json(summary_path, evaluation_summary)
        del lm
        return evaluation_summary

    def _evaluate_prompt_entry(
        self,
        *,
        model: Any,
        lm: Any,
        entry: BenchmarkEntry,
        variant_name: str,
        variant: ModelVariantConfig,
        seed: int,
        checkpoint_path: Path,
        graphs_dir: Path,
    ) -> dict[str, Any]:
        base_record = self._record_base(
            variant_name=variant_name,
            variant=variant,
            seed=seed,
            checkpoint_path=checkpoint_path,
        )
        graph_slug = _sanitize_name(f"{variant_name}_{seed}_{entry.prompt_id}")
        try:
            prompt_cache = collect_prompt_cache(
                lm=lm,
                prompt=entry.prompt,
                n_layers=model.n_layers,
                feature_input_hook=model.feature_input_hook,
                feature_output_hook=model.feature_output_hook,
            )
            replacement_metrics = evaluate_prompt_replacement(model, prompt_cache, lm)
            graph_info = build_prompt_graph(
                model=model,
                lm=lm,
                prompt_cache=prompt_cache,
                graph_dir=graphs_dir,
                graph_slug=graph_slug,
                scan_id=variant.scan_id or variant_name,
                max_features=self.config.evaluation.graph.max_features,
                max_n_logits=self.config.evaluation.graph.max_n_logits,
                desired_logit_prob=self.config.evaluation.graph.desired_logit_prob,
                node_threshold=self.config.evaluation.graph.node_threshold,
                edge_threshold=self.config.evaluation.graph.edge_threshold,
            )
            target_idx, foil_idx, gap_direction = build_logit_gap_direction(
                tokenizer=lm.tokenizer,
                unembed=lm.W_U.float(),
                target_token=entry.target_token,
                foil_token=entry.foil_token,
            )
            causal_nodes = load_ranked_feature_nodes(
                Path(graph_info["graph_json_path"]),
                top_k=self.config.evaluation.circuit.top_k_features,
            )
            full_gap = replacement_logit_gap_from_subset(
                model=model,
                prompt_cache=prompt_cache,
                lm=lm,
                target_idx=target_idx,
                foil_idx=foil_idx,
                selected_node_ids=None,
            )
            keep_only_gap = replacement_logit_gap_from_subset(
                model=model,
                prompt_cache=prompt_cache,
                lm=lm,
                target_idx=target_idx,
                foil_idx=foil_idx,
                selected_node_ids=causal_nodes,
            )
            active_nodes = self._all_active_feature_node_ids(model, prompt_cache)
            remaining_nodes = sorted(set(active_nodes) - set(causal_nodes))
            remove_gap = replacement_logit_gap_from_subset(
                model=model,
                prompt_cache=prompt_cache,
                lm=lm,
                target_idx=target_idx,
                foil_idx=foil_idx,
                selected_node_ids=remaining_nodes,
            )

            shapley_jaccard = None
            if self.config.evaluation.circuit.run_shapley:
                shapley = shapley_logit_attribution(
                    model=model,
                    x_in=prompt_cache.mlp_inputs,
                    logit_target=gap_direction,
                    n_samples=self.config.evaluation.circuit.shapley_samples,
                    max_features=max(
                        self.config.evaluation.circuit.top_k_features,
                        self.config.evaluation.graph.max_features,
                    ),
                    seed=seed,
                )
                shapley_nodes = [
                    feature_node_id(
                        layer_id=int(layer_id),
                        position=int(position),
                        feature_id=int(feature_id),
                    )
                    for (layer_id, position, feature_id), _ in sorted(
                        zip(
                            shapley["active_features"].tolist(),
                            shapley["shapley_values"].abs().tolist(),
                            strict=False,
                        ),
                        key=lambda item: -item[1],
                    )[: self.config.evaluation.circuit.top_k_features]
                ]
                shapley_jaccard = jaccard_overlap(causal_nodes, shapley_nodes)

            keep_only_gap_ratio = (
                keep_only_gap / full_gap if abs(full_gap) > 1e-8 else float("nan")
            )
            gap_drop = full_gap - remove_gap
            gap_drop_ratio = gap_drop / full_gap if abs(full_gap) > 1e-8 else float("nan")

            return (
                base_record
                | {
                    "record_type": "prompt_metric",
                    "status": "ok",
                    "prompt_id": entry.prompt_id,
                    "family": entry.family,
                    "split": entry.split,
                    "prompt": entry.prompt,
                    "target_token": entry.target_token,
                    "foil_token": entry.foil_token,
                    "include_macag": entry.include_macag,
                    "graph_json_path": graph_info["graph_json_path"],
                    "graph_pt_path": graph_info["graph_pt_path"],
                    "active_feature_count": graph_info["active_feature_count"],
                    "retained_feature_node_count": graph_info["retained_feature_node_count"],
                    "retained_error_node_count": graph_info["retained_error_node_count"],
                    "retained_embedding_node_count": graph_info["retained_embedding_node_count"],
                    "retained_logit_node_count": graph_info["retained_logit_node_count"],
                    "retained_total_node_count": graph_info["retained_total_node_count"],
                    "retained_error_node_fraction": graph_info["retained_error_node_fraction"],
                    "graph_replacement_score": graph_info["graph_replacement_score"],
                    "graph_completeness_score": graph_info["graph_completeness_score"],
                    "full_logit_gap": full_gap,
                    "keep_only_gap": keep_only_gap,
                    "remove_gap": remove_gap,
                    "keep_only_gap_ratio": keep_only_gap_ratio,
                    "gap_drop": gap_drop,
                    "gap_drop_ratio": gap_drop_ratio,
                    "causal_top_k_nodes": causal_nodes,
                    "shapley_causal_jaccard": shapley_jaccard,
                }
                | replacement_metrics
            )
        except Exception as exc:
            return (
                base_record
                | {
                    "record_type": "prompt_metric",
                    "status": "error",
                    "prompt_id": entry.prompt_id,
                    "family": entry.family,
                    "split": entry.split,
                    "include_macag": entry.include_macag,
                    "error": str(exc),
                }
            )

    def _all_active_feature_node_ids(self, model: Any, prompt_cache: Any) -> list[str]:
        activations = model.encode(prompt_cache.mlp_inputs).to_sparse().coalesce()
        layer_ids, position_ids, feature_ids = activations.indices()
        return [
            feature_node_id(int(layer_id), int(position_id), int(feature_id))
            for layer_id, position_id, feature_id in zip(
                layer_ids.tolist(),
                position_ids.tolist(),
                feature_ids.tolist(),
                strict=False,
            )
        ]

    def _collect_monosemanticity(
        self,
        *,
        model: Any,
        dataset: ActivationDataset,
        base_record: dict[str, Any],
        device: torch.device,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.config.evaluation.monosemanticity.enabled:
            return [], {"status": "disabled"}
        reports = collect_max_activating_examples(
            model=model,
            dataset=dataset,
            top_n_features=self.config.evaluation.monosemanticity.n_features,
            top_k_examples=self.config.evaluation.monosemanticity.top_k_examples,
            n_samples=self.config.evaluation.monosemanticity.n_samples,
            device=device,
            dtype=torch.float32,
            seed=self.config.dataset.seed,
        )
        records = [
            base_record
            | {
                "record_type": "monosemanticity_feature",
                "layer": report.layer,
                "feature_id": report.feature_id,
                "activation_frequency": report.activation_frequency,
                "mean_activation": report.mean_activation,
                "max_activation": report.max_activation,
                "gini_coefficient": report.gini_coefficient,
            }
            for report in reports
        ]
        ginis = [record["gini_coefficient"] for record in records]
        summary = {
            "status": "completed",
            "n_features": len(records),
            "mean_gini": float(sum(ginis) / len(ginis)) if ginis else float("nan"),
            "median_gini": float(torch.tensor(ginis).median().item()) if ginis else float("nan"),
            "fraction_gini_gt_0_7": (
                float(sum(1 for value in ginis if value > 0.7) / len(ginis)) if ginis else float("nan")
            ),
            "fraction_gini_gt_0_8": (
                float(sum(1 for value in ginis if value > 0.8) / len(ginis)) if ginis else float("nan")
            ),
        }
        return records, summary

    def _collect_spline_analysis(
        self,
        *,
        model: Any,
        dataset: ActivationDataset,
        evaluation_dir: Path,
        device: torch.device,
    ) -> dict[str, Any]:
        if not self.config.evaluation.splines.enabled:
            return {"status": "disabled"}
        if model.encoder_type != "kan":
            return {"status": "skipped", "reason": "encoder_type is not kan"}

        stats = collect_feature_stats(
            model=model,
            dataset=dataset,
            device=device,
            dtype=torch.float32,
            n_samples=self.config.evaluation.splines.n_samples,
            seed=self.config.dataset.seed,
        )
        frequency = stats["activation_frequency"]
        curve_dir = evaluation_dir / "splines"
        curve_dir.mkdir(parents=True, exist_ok=True)
        flat_top = frequency.reshape(-1).topk(self.config.evaluation.splines.n_features).indices
        curve_files: list[str] = []
        for flat_idx in flat_top.tolist():
            layer_id = int(flat_idx // model.d_transcoder)
            feature_id = int(flat_idx % model.d_transcoder)
            dims = top_input_dims(
                model=model,
                layer_id=layer_id,
                feature_id=feature_id,
                top_k=self.config.evaluation.splines.top_dims,
            )
            t_values = None
            curves: dict[int, Any] = {}
            for dim in dims:
                sampled_t, sampled_curve = extract_spline_curve(
                    model=model,
                    layer_id=layer_id,
                    feature_id=feature_id,
                    input_dim=dim,
                    device=device,
                )
                if t_values is None:
                    t_values = sampled_t
                curves[dim] = sampled_curve
            if t_values is None or not curves:
                continue
            save_curves_csv(
                output_dir=str(curve_dir),
                layer_id=layer_id,
                feature_id=feature_id,
                t_vals=t_values,
                curves=curves,
            )
            curve_files.append(str(curve_dir / f"layer{layer_id}_feat{feature_id}.csv"))
        return {
            "status": "completed",
            "curve_file_count": len(curve_files),
            "curve_files": curve_files,
        }

    def _run_macag_for_variant_seed(
        self,
        *,
        variant_name: str,
        variant: ModelVariantConfig,
        seed: int,
        checkpoint_path: Path,
    ) -> dict[str, Any]:
        run_dir = self._run_dir(variant_name, seed)
        macag_dir = run_dir / "macag"
        macag_dir.mkdir(parents=True, exist_ok=True)
        summary_path = macag_dir / "macag_summary.json"
        if summary_path.exists():
            return read_json(summary_path)

        prompt_records_path = run_dir / "evaluation" / "prompt_metrics.jsonl"
        if not prompt_records_path.exists():
            raise FileNotFoundError(
                f"MACAG requires prompt evaluation artifacts at {prompt_records_path}"
            )

        prompt_records = [
            record
            for record in read_jsonl(prompt_records_path)
            if record.get("status", "ok") == "ok" and record.get("include_macag")
        ]
        macag_records: list[dict[str, Any]] = []
        for prompt_record in prompt_records:
            prompt_dir = macag_dir / _sanitize_name(str(prompt_record["prompt_id"]))
            prompt_dir.mkdir(parents=True, exist_ok=True)
            try:
                macag_records.extend(
                    self._run_macag_prompt(
                        prompt_record=prompt_record,
                        variant_name=variant_name,
                        variant=variant,
                        seed=seed,
                        checkpoint_path=checkpoint_path,
                        prompt_dir=prompt_dir,
                    )
                )
            except Exception as exc:
                macag_records.append(
                    self._record_base(
                        variant_name=variant_name,
                        variant=variant,
                        seed=seed,
                        checkpoint_path=checkpoint_path,
                    )
                    | {
                        "record_type": "macag_error",
                        "status": "error",
                        "prompt_id": prompt_record.get("prompt_id"),
                        "family": prompt_record.get("family"),
                        "error": str(exc),
                    }
                )

        write_jsonl(macag_dir / "macag_records.jsonl", macag_records)
        ok_records = [record for record in macag_records if record.get("status", "ok") == "ok"]
        summary = {
            "status": "completed",
            "prompt_count": len(prompt_records),
            "successful_records": len(ok_records),
            "error_count": sum(1 for record in macag_records if record.get("status") == "error"),
            "created_at": _utc_now(),
        }
        write_json(summary_path, summary)
        return summary

    def _run_macag_prompt(
        self,
        *,
        prompt_record: dict[str, Any],
        variant_name: str,
        variant: ModelVariantConfig,
        seed: int,
        checkpoint_path: Path,
        prompt_dir: Path,
    ) -> list[dict[str, Any]]:
        graph_json_path = Path(prompt_record["graph_json_path"])
        graph = CircuitGraph.from_json(graph_json_path)
        scorer = create_replacement_model_scorer(
            model_name=self._variant_model_name(variant),
            prompt=prompt_record["prompt"],
            graph_json=str(graph_json_path),
            target_token_by_label={
                "y": prompt_record["target_token"],
                "y_foil": prompt_record["foil_token"],
            },
            backend=self.config.macag.oracle.backend,
            score_kind=self.config.macag.oracle.score_kind,
            freeze_attention=self.config.macag.oracle.freeze_attention,
            feature_types=self.config.macag.candidate_policy.feature_types,
            model_kwargs={
                "device": str(_device_from_name(variant.training.device or self.config.dataset.device)),
                "dtype": variant.training.dtype,
            },
            local_clt_path=str(checkpoint_path),
            clt_scan=variant.scan_id or variant_name,
        )

        candidates = self._select_macag_candidates(
            graph=graph,
            graph_json_path=graph_json_path,
            prompt_dir=prompt_dir,
        )
        if not candidates:
            raise ValueError("MACAG candidate policy produced an empty candidate set.")

        base_record = self._record_base(
            variant_name=variant_name,
            variant=variant,
            seed=seed,
            checkpoint_path=checkpoint_path,
        )
        records: list[dict[str, Any]] = []

        for game1_cfg in self.config.macag.game1_runs:
            oracle = ScoringOracle(scorer, cache_enabled=self.config.macag.oracle.cache_enabled)
            result = solve_game1(
                graph=graph,
                oracle=oracle,
                target="y",
                candidates=candidates,
                alpha=game1_cfg.alpha,
                lam=game1_cfg.lam,
                budget=game1_cfg.budget,
                faithfulness_eps=game1_cfg.faithfulness_eps,
                prefilter_top_k=game1_cfg.prefilter_top_k,
                connected=game1_cfg.connected,
                min_gain=game1_cfg.min_gain,
                progress=False,
            )
            result_path = prompt_dir / f"game1_{_sanitize_name(game1_cfg.name)}.json"
            payload = {
                "game": "game1",
                "prompt_id": prompt_record["prompt_id"],
                "family": prompt_record["family"],
                "run_name": game1_cfg.name,
                "params": result.params,
                "scores": metrics_to_dict(result.metrics) | {"utility": result.utility},
                "stats": oracle.cache_stats() | {"candidate_count": result.candidate_count},
                "evidence": {
                    "E_star": sorted(str(node) for node in result.evidence),
                    "E_y": sorted(str(node) for node in result.evidence),
                },
            }
            write_json(result_path, payload)
            if self.config.macag.annotate_graphs:
                annotate_graph_with_macag(
                    graph_json_path=str(graph_json_path),
                    macag_result_json_path=str(result_path),
                    output_path=str(prompt_dir / f"game1_{_sanitize_name(game1_cfg.name)}_annotated.json"),
                    label_prefix=f"MACAG:{game1_cfg.name}",
                )
            records.append(
                base_record
                | {
                    "record_type": "macag_game1",
                    "status": "ok",
                    "prompt_id": prompt_record["prompt_id"],
                    "family": prompt_record["family"],
                    "run_name": game1_cfg.name,
                    "candidate_strategy": self.config.macag.candidate_policy.strategy,
                    "candidate_count": len(candidates),
                    "evidence_size": len(result.evidence),
                    "faithfulness": result.metrics.faithfulness_delta,
                    "sufficiency": result.metrics.sufficiency,
                    "necessity": result.metrics.necessity,
                    "utility": result.utility,
                    "oracle_calls": result.oracle_calls,
                    "cache_hits": result.cache_hits,
                    "cache_size": result.cache_size,
                    "sparsity": result.sparsity,
                    "evidence_nodes": sorted(str(node) for node in result.evidence),
                    "result_json_path": str(result_path),
                }
            )

        for game2_cfg in self.config.macag.game2_runs:
            oracle = ScoringOracle(scorer, cache_enabled=self.config.macag.oracle.cache_enabled)
            result = solve_game2(
                graph=graph,
                oracle=oracle,
                y="y",
                y_foil="y_foil",
                candidates=candidates,
                alpha=game2_cfg.alpha,
                lam=game2_cfg.lam,
                beta=game2_cfg.beta,
                abr_iters=game2_cfg.abr_iters,
                budget=game2_cfg.budget,
                connected=game2_cfg.connected,
                min_gain=game2_cfg.min_gain,
                prefilter_top_k=game2_cfg.prefilter_top_k,
                progress=False,
            )
            result_path = prompt_dir / f"game2_{_sanitize_name(game2_cfg.name)}.json"
            payload = {
                "game": "game2",
                "prompt_id": prompt_record["prompt_id"],
                "family": prompt_record["family"],
                "run_name": game2_cfg.name,
                "params": result.params,
                "scores": {
                    "y": metrics_to_dict(result.metrics_y),
                    "foil": metrics_to_dict(result.metrics_foil),
                    "utility_y": result.utility_y,
                    "utility_foil": result.utility_foil,
                    "overlap_rate": result.overlap_rate,
                },
                "stats": oracle.cache_stats(),
                "evidence": {
                    "shared": sorted(str(node) for node in result.shared),
                    "unique_y": sorted(str(node) for node in result.unique_y),
                    "unique_foil": sorted(str(node) for node in result.unique_foil),
                    "E_y": sorted(str(node) for node in result.evidence_y),
                    "E_foil": sorted(str(node) for node in result.evidence_foil),
                },
            }
            write_json(result_path, payload)
            if self.config.macag.annotate_graphs:
                annotate_graph_with_macag(
                    graph_json_path=str(graph_json_path),
                    macag_result_json_path=str(result_path),
                    output_path=str(prompt_dir / f"game2_{_sanitize_name(game2_cfg.name)}_annotated.json"),
                    label_prefix=f"MACAG:{game2_cfg.name}",
                )
            records.append(
                base_record
                | {
                    "record_type": "macag_game2",
                    "status": "ok",
                    "prompt_id": prompt_record["prompt_id"],
                    "family": prompt_record["family"],
                    "run_name": game2_cfg.name,
                    "candidate_strategy": self.config.macag.candidate_policy.strategy,
                    "candidate_count": len(candidates),
                    "evidence_size": len(result.evidence_y) + len(result.evidence_foil),
                    "faithfulness": (
                        result.metrics_y.faithfulness_delta + result.metrics_foil.faithfulness_delta
                    )
                    / 2.0,
                    "sufficiency": (result.metrics_y.sufficiency + result.metrics_foil.sufficiency) / 2.0,
                    "necessity": (result.metrics_y.necessity + result.metrics_foil.necessity) / 2.0,
                    "utility": (result.utility_y + result.utility_foil) / 2.0,
                    "overlap_rate": result.overlap_rate,
                    "oracle_calls": result.oracle_calls,
                    "cache_hits": result.cache_hits,
                    "cache_size": result.cache_size,
                    "evidence_nodes": sorted(
                        {str(node) for node in (result.evidence_y | result.evidence_foil)}
                    ),
                    "result_json_path": str(result_path),
                }
            )
        return records

    def _select_macag_candidates(
        self,
        *,
        graph: CircuitGraph,
        graph_json_path: Path,
        prompt_dir: Path,
    ) -> list[str]:
        policy = self.config.macag.candidate_policy
        if policy.strategy == "graph_features":
            return load_ranked_feature_nodes(graph_json_path, top_k=policy.top_k)

        supernodes = propose_supernodes(
            graph=graph,
            top_k=policy.top_k,
            min_group_size=policy.min_group_size,
            max_group_size=policy.max_group_size,
            min_salience=policy.min_salience,
            feature_types=policy.feature_types,
        )
        write_json(prompt_dir / "auto_supernodes.json", supernodes)
        return supernode_candidates(supernodes)

    def _generate_reporting(self) -> dict[str, Any]:
        records = load_suite_records(self.suite_root)
        write_jsonl(self.suite_root / "per_example_metrics.jsonl", records)
        aggregate = aggregate_suite_records(records, self.config)
        write_json(self.suite_root / "aggregate_metrics.json", aggregate)

        rows = build_tables_csv_rows(aggregate)
        write_tables_csv(self.suite_root / "tables.csv", rows)
        report_md = build_report_markdown(aggregate, self.config)
        (self.suite_root / "report.md").write_text(report_md)
        write_json(self.figures_root / "figure_manifest.json", build_figure_manifest(aggregate, self.config))
        return aggregate
