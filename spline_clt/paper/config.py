"""Validated JSON config loading for paper evaluation suites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, field_validator, model_validator


class StageConfig(BaseModel):
    collect_dataset: bool = True
    train: bool = True
    evaluate: bool = True
    macag: bool = True
    report: bool = True


class DatasetConfig(BaseModel):
    model_name: str = "gpt2"
    dataset_name: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    n_tokens: int = 2_000_000
    seq_len: int = 128
    batch_size: int = 32
    cache_dir: str = "shared/activations"
    #: Optional separate directory for the val split. Train and val are
    #: written to two distinct memmap files at collection time; the train
    #: cache typically lives on a per-node local NVMe (regenerable) while
    #: the val cache lives on shared/persistent storage so it can be reused
    #: across nodes and downstream evaluation. When unset, val files are
    #: written next to train files in ``cache_dir``.
    val_cache_dir: str | None = None
    val_fraction: float = 0.05
    max_train_samples: int = 10_000
    eval_samples: int = 500
    dtype: Literal["float32", "bfloat16"] = "bfloat16"
    device: str | None = None
    seed: int = 0


class TrainingSettings(BaseModel):
    n_layers: int = 12
    d_model: int = 768
    d_transcoder: int
    encoder_type: Literal["kan", "linear"]
    grid_size: int = 5
    spline_order: int = 3
    learning_rate: float = 1e-4
    warmup_steps: int = 5_000
    total_steps: int = 400_000
    batch_size: int = 32
    lambda_sparsity: float = 0.002
    c_sparsity: float = 1.0
    grad_clip: float = 1.0
    log_every: int = 200
    eval_every: int = 10_000
    save_every: int = 10_000
    update_grid_every: int = 100_000
    update_grid_from: int = 5_000
    reset_optimizer_every: int = 0
    val_fraction: float = 0.05
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int | None = None
    persistent_workers: bool = False
    dtype: Literal["float32", "bfloat16"] = "bfloat16"
    device: str | None = None
    run_name: str | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_mode: Literal["online", "offline", "disabled"] = "online"


class ModelVariantConfig(BaseModel):
    label: str
    model_name: str | None = None
    checkpoint_path: str | None = None
    scan_id: str | None = None
    training: TrainingSettings


class GraphBuildConfig(BaseModel):
    max_features: int = 64
    max_n_logits: int = 8
    desired_logit_prob: float = 0.9
    node_threshold: float = 0.8
    edge_threshold: float = 0.98
    attribution_batch_size: int = 32
    spline_attribution_method: Literal["jacobian_ablation", "shapley"] = "jacobian_ablation"
    shapley_samples_for_graph: int = 32


class CircuitEvaluationConfig(BaseModel):
    top_k_features: int = 16
    alpha: float = 0.5
    run_shapley: bool = True
    shapley_samples: int = 64


class MonosemanticityConfig(BaseModel):
    enabled: bool = True
    n_features: int = 200
    top_k_examples: int = 10
    n_samples: int = 500


class SplineAnalysisConfig(BaseModel):
    enabled: bool = True
    n_features: int = 20
    top_dims: int = 3
    n_samples: int = 500
    no_plot: bool = False


class EvaluationConfig(BaseModel):
    graph: GraphBuildConfig = Field(default_factory=GraphBuildConfig)
    circuit: CircuitEvaluationConfig = Field(default_factory=CircuitEvaluationConfig)
    monosemanticity: MonosemanticityConfig = Field(default_factory=MonosemanticityConfig)
    splines: SplineAnalysisConfig = Field(default_factory=SplineAnalysisConfig)


class BenchmarkEntry(BaseModel):
    prompt_id: str
    family: Literal["ioi", "factual", "arithmetic"]
    prompt: str
    target_token: str
    foil_token: str
    split: str = "main"
    include_macag: bool = False


class CandidatePolicyConfig(BaseModel):
    strategy: Literal["graph_features", "auto_supernodes"] = "graph_features"
    feature_types: list[str] = Field(default_factory=lambda: ["cross layer transcoder"])
    top_k: int = 40
    min_group_size: int = 2
    max_group_size: int = 8
    min_salience: float = 0.0


class MacagOracleConfig(BaseModel):
    backend: Literal["transformerlens", "nnsight"] = "transformerlens"
    score_kind: Literal["logit", "prob", "logit_gap", "negative_loss"] = "logit_gap"
    freeze_attention: bool = True
    cache_enabled: bool = True


class MacagGame1RunConfig(BaseModel):
    name: str
    alpha: float = 0.5
    lam: float = 0.01
    budget: int | None = None
    faithfulness_eps: float | None = None
    prefilter_top_k: int | None = None
    connected: bool = False
    min_gain: float = 0.0


class MacagGame2RunConfig(BaseModel):
    name: str
    alpha: float = 0.5
    lam: float = 0.01
    beta: float = 0.1
    abr_iters: int = 10
    budget: int | None = None
    connected: bool = False
    prefilter_top_k: int | None = None
    min_gain: float = 0.0


class MacagConfig(BaseModel):
    enabled: bool = True
    annotate_graphs: bool = True
    oracle: MacagOracleConfig = Field(default_factory=MacagOracleConfig)
    candidate_policy: CandidatePolicyConfig = Field(default_factory=CandidatePolicyConfig)
    game1_runs: list[MacagGame1RunConfig] = Field(default_factory=list)
    game2_runs: list[MacagGame2RunConfig] = Field(default_factory=list)


class ReportingConfig(BaseModel):
    primary_variant: str
    baseline_variant: str
    bootstrap_samples: int = 1_000
    confidence_level: float = 0.95
    figure_case_study_prompt_id: str | None = None


class WandbConfig(BaseModel):
    project: str | None = None
    entity: str | None = None
    mode: Literal["online", "offline", "disabled"] = "online"


class SuiteConfig(BaseModel):
    defaults: list[str] = Field(default_factory=list)
    suite_name: str
    description: str = ""
    output_root: str = "results/paper"
    seeds: list[int]
    stages: StageConfig = Field(default_factory=StageConfig)
    dataset: DatasetConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    model_variants: dict[str, ModelVariantConfig]
    benchmark_manifest_version: str
    benchmark_entries: list[BenchmarkEntry]
    macag: MacagConfig = Field(default_factory=MacagConfig)
    reporting: ReportingConfig
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    @field_validator("seeds")
    @classmethod
    def _validate_seeds(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("At least one seed is required.")
        if len(set(value)) != len(value):
            raise ValueError("Seeds must be unique.")
        return value

    @model_validator(mode="after")
    def _validate_manifest(self) -> "SuiteConfig":
        prompt_ids = [entry.prompt_id for entry in self.benchmark_entries]
        if len(prompt_ids) != len(set(prompt_ids)):
            raise ValueError("Benchmark prompt_ids must be unique.")
        if self.reporting.primary_variant not in self.model_variants:
            raise ValueError("reporting.primary_variant must refer to a configured model variant.")
        if self.reporting.baseline_variant not in self.model_variants:
            raise ValueError("reporting.baseline_variant must refer to a configured model variant.")
        return self


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _merge_with_defaults(path: Path, stack: set[Path]) -> Any:
    resolved = path.resolve()
    if resolved in stack:
        raise ValueError(f"Config defaults cycle detected at {resolved}")

    stack.add(resolved)
    raw = _load_json(resolved)
    merged = OmegaConf.create()
    for default_ref in raw.get("defaults", []):
        default_path = (resolved.parent / default_ref).resolve()
        merged = OmegaConf.merge(merged, _merge_with_defaults(default_path, stack))

    merged = OmegaConf.merge(
        merged,
        OmegaConf.create({key: value for key, value in raw.items() if key != "defaults"}),
    )
    stack.remove(resolved)
    return merged


def load_suite_config(path: str | Path) -> tuple[SuiteConfig, dict[str, Any]]:
    """Load, merge, and validate a paper suite config."""
    config_path = Path(path)
    merged = _merge_with_defaults(config_path, stack=set())
    resolved = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError("Resolved suite config must be a mapping.")
    suite = SuiteConfig.model_validate(resolved)
    return suite, resolved
