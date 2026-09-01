"""Config-driven paper evaluation runner."""

from __future__ import annotations

import gc
import json
import math
import os
import re
import shutil
import subprocess
import time

import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

from attribution.shapley import shapley_logit_attribution
from eval.monosemanticity import collect_max_activating_examples
from experiments.analyze_splines import (
    collect_feature_stats,
    compute_nonlinearity_score,
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
    load_replacement_model,
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
from spline_clt.training.data import (
    ActivationDataset,
    DataConfig,
    collect_activations,
    collect_activations_isolated,
)
from spline_clt.training.train import (
    TrainConfig,
    build_session,
    close_session,
    release_session_gpu,
    run_chunk,
    train,
)


def _scrub_cuda_memory(*, log_fn=None, tag: str = "") -> None:
    """Release cached blocks on every visible CUDA device (current process).

    ``empty_cache`` only affects the current device; chunked collect/train
    must scrub all devices the process may have touched. Also runs
    ``ipc_collect`` so peer-mapped blocks from NCCL/FSDP can be reclaimed
    before the next peak (full-decoder unshard ≈ 9.26 GiB).
    """
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    n = torch.cuda.device_count()
    for i in range(n):
        with torch.cuda.device(i):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    if log_fn is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        alloc = torch.cuda.memory_allocated(local_rank) / 1e9
        reserved = torch.cuda.memory_reserved(local_rank) / 1e9
        free, total = torch.cuda.mem_get_info(local_rank)
        log_fn(
            f"{tag}GPU{local_rank} allocated={alloc:.1f} GB, "
            f"reserved={reserved:.1f} GB, "
            f"free={free / 1e9:.1f}/{total / 1e9:.1f} GB"
        )


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
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    if torch.backends.mps.is_available():
        return {"device": "mps"}
    return {"device": "cpu"}


class PaperSuiteRunner:
    """Runs a paper evaluation suite from one resolved JSON config."""

    # Maps CLI stage short-names to StageConfig field names.
    _STAGE_NAME_MAP: dict[str, str] = {
        "collect": "collect_dataset",
        "train": "train",
        "evaluate": "evaluate",
        "macag": "macag",
        "report": "report",
    }

    def __init__(
        self,
        suite_path: str | Path,
        worker_id: int = 0,
        num_workers: int = 1,
        stages_override: list[str] | None = None,
    ):
        self.suite_path = Path(suite_path).resolve()
        self.config, self.resolved_config = load_suite_config(self.suite_path)
        # Allow HPC jobs to redirect large paper_eval artifacts away from shared
        # filesystems (e.g. into /cluster/.../gscratch).
        output_override = os.environ.get("PAPER_OUTPUT_ROOT")
        if output_override:
            self.config.output_root = output_override
            if isinstance(self.resolved_config, dict):
                self.resolved_config["output_root"] = output_override
        self.repo_root = self._find_repo_root(self.suite_path.parent)
        self.suite_root = (self.repo_root / self.config.output_root / self.config.suite_name).resolve()
        self.datasets_root = self.suite_root / self.config.dataset.cache_dir
        # When val_cache_dir is unset, val lives next to train (legacy layout).
        val_cache_dir = self.config.dataset.val_cache_dir or self.config.dataset.cache_dir
        self.val_datasets_root = self.suite_root / val_cache_dir
        self.runs_root = self.suite_root / "runs"
        self.figures_root = self.suite_root / "figures"
        self.commit_hash = _git_commit(self.repo_root)
        self.hardware = _hardware_summary()
        self.worker_id = worker_id
        self.num_workers = num_workers
        self._stages_override: frozenset[str] | None = (
            frozenset(stages_override) if stages_override else None
        )

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

    def _stage_enabled(self, stage_name: str) -> bool:
        """Return True if *stage_name* is enabled both in suite config and the CLI override set.

        *stage_name* must be a StageConfig field name (e.g. "collect_dataset", "train", …).
        """
        suite_flag: bool = getattr(self.config.stages, stage_name, True)
        if not suite_flag:
            return False
        if self._stages_override is None:
            return True
        # Map stage_name back to the CLI short-names used in _stages_override.
        cli_name = next(
            (k for k, v in self._STAGE_NAME_MAP.items() if v == stage_name),
            stage_name,
        )
        return cli_name in self._stages_override

    def _my_variant_seed_pairs(self) -> list[tuple[str, Any, int]]:
        """Return the variant×seed pairs assigned to this worker via round-robin sharding."""
        def _variant_order(item: tuple[str, Any]) -> tuple[int, int, str]:
            name = item[0]
            lname = name.lower()
            is_spline = "spline" in lname
            is_linear = "linear" in lname
            # Family: spline before linear (0 < 1); unknown families last.
            if is_spline:
                family = 0
            elif is_linear:
                family = 1
            else:
                family = 2
            # Variant tier: feature_match (base) before param_match (variant).
            if "feature_match" in lname:
                tier = 0
            elif "param_match" in lname:
                tier = 1
            else:
                tier = 2
            # Order per seed: spline-base, linear-base, spline-variant, linear-variant.
            return (tier, family, name)

        all_pairs: list[tuple[str, Any, int]] = [
            (variant_name, variant, seed)
            for seed in self.config.seeds
            for variant_name, variant in sorted(self.config.model_variants.items(), key=_variant_order)
        ]
        return all_pairs[self.worker_id :: self.num_workers]

    def _wait_for_suite_manifest(self, timeout_s: int = 120) -> None:
        """Wait for worker-0 to write the suite manifest (max *timeout_s* seconds)."""
        manifest_path = self.suite_root / "manifest.json"
        deadline = time.monotonic() + timeout_s
        while not manifest_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Suite manifest not found at {manifest_path} after {timeout_s}s. "
                    "Did worker-0 fail to start?"
                )
            time.sleep(2)

    def _wait_for_dataset(self, model_name: str, timeout_s: int = 7200) -> None:
        """Wait for the dataset for *model_name* to be ready (max *timeout_s* seconds)."""
        dataset_dir = self._dataset_dir(model_name)
        deadline = time.monotonic() + timeout_s
        while not self._dataset_stage_complete(model_name):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Dataset for {model_name!r} not ready in {dataset_dir} after {timeout_s}s."
                )
            time.sleep(30)

    @staticmethod
    def _log(msg: str) -> None:
        print(f"[paper-eval] {msg}", flush=True)

    @staticmethod
    def _banner(title: str) -> None:
        bar = "=" * 70
        print(f"\n{bar}\n[paper-eval] {title}\n{bar}", flush=True)

    def run(self) -> dict[str, Any]:
        self._prepare_suite_root()

        variant_seed_pairs = self._my_variant_seed_pairs()
        n_variants = len(self.config.model_variants)
        n_seeds = len(self.config.seeds)
        n_benchmarks = len(self.config.benchmark_entries)
        self._banner(
            f"Suite: {self.config.suite_name}  |  "
            f"{n_variants} variant(s) x {n_seeds} seed(s) = "
            f"{len(variant_seed_pairs)} run(s)  |  "
            f"{n_benchmarks} benchmark prompt(s)"
        )

        # Only worker 0 writes the manifest; others wait for it.
        if self.worker_id == 0:
            self._write_suite_manifest()
        else:
            self._wait_for_suite_manifest()

        unique_model_names = sorted(
            {self._variant_model_name(variant) for variant in self.config.model_variants.values()}
        )

        if self._stage_enabled("collect_dataset"):
            if self._dataset_uses_chunked_offline():
                # Chunked-offline normally relies on per-chunk writes during
                # training. But if every (variant, seed) for a given model is
                # already trained, no chunks will be produced this run, and
                # eval (which needs the val split) will fail. Detect that and
                # run a one-off collection for those models.
                eval_only_models = (
                    self._models_needing_eval_only_collection(unique_model_names)
                    if self._stage_enabled("evaluate")
                    else []
                )
                if eval_only_models:
                    if self.worker_id == 0:
                        self._log(
                            "Chunked offline + training already complete; "
                            "collecting activations so eval has a val split: "
                            f"{eval_only_models}"
                        )
                        for model_name in eval_only_models:
                            self._banner(
                                f"STAGE: Dataset collection (eval-only)  |  model={model_name}"
                            )
                            self._collect_dataset(model_name, eval_only=True)
                    else:
                        for model_name in eval_only_models:
                            self._wait_for_dataset(model_name)
                else:
                    self._log(
                        "Chunked offline collection — skipping global dataset stage "
                        "(per-chunk mmap writes during training)."
                    )
            elif self.worker_id == 0:
                for model_name in unique_model_names:
                    self._banner(f"STAGE: Dataset collection  |  model={model_name}")
                    self._collect_dataset(model_name)
            else:
                if not self._dataset_uses_chunked_offline():
                    # Defensive: dataset should already be ready (Phase 1 ran first),
                    # but poll briefly in case of NFS staleness.
                    for model_name in unique_model_names:
                        self._wait_for_dataset(model_name)

        checkpoint_paths: dict[tuple[str, int], Path] = {}

        for run_idx, (variant_name, variant, seed) in enumerate(variant_seed_pairs, 1):
            self._banner(
                f"RUN {run_idx}/{len(variant_seed_pairs)} [train]:  "
                f"variant={variant_name}  seed={seed}"
            )
            if self._stage_enabled("train"):
                self._log(f"STAGE: Training  |  variant={variant_name}  seed={seed}")
                checkpoint_path = self._train_variant_seed(variant_name, variant, seed)
            else:
                checkpoint_path = self._resolve_checkpoint_path(variant_name, variant, seed)
            checkpoint_paths[(variant_name, seed)] = checkpoint_path

        # Training is distributed (FSDP needs every rank); evaluation, MACAG, and
        # reporting are single-process (transformer_lens forward passes, graph
        # building, file writes) with no collective ops — verified. Under torchrun
        # they would otherwise run on every rank, which (a) crashes on non-zero ranks
        # because the language model loads on cuda:local_rank while inputs land on
        # cuda:0 (embed index device mismatch), and (b) races multiple ranks into the
        # same report file writes (worker_id==0 is true on every torchrun rank).
        # Only rank 0 runs them; other ranks return immediately. torchrun waits for
        # rank 0 to finish, and rank 0 issues no collectives, so early exit is safe.
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0:
            return {
                "suite_root": str(self.suite_root),
                "worker_id": self.worker_id,
                "num_workers": self.num_workers,
                "aggregate_metrics_path": str(self.suite_root / "aggregate_metrics.json"),
                "aggregate": {},
            }

        for run_idx, (variant_name, variant, seed) in enumerate(variant_seed_pairs, 1):
            self._banner(
                f"RUN {run_idx}/{len(variant_seed_pairs)} [eval/macag]:  "
                f"variant={variant_name}  seed={seed}"
            )
            checkpoint_path = checkpoint_paths[(variant_name, seed)]
            if self._stage_enabled("evaluate"):
                self._log(f"STAGE: Evaluation  |  variant={variant_name}  seed={seed}")
                self._evaluate_variant_seed(
                    variant_name=variant_name,
                    variant=variant,
                    seed=seed,
                    checkpoint_path=checkpoint_path,
                )
            if self._stage_enabled("macag") and self.config.macag.enabled:
                self._log(f"STAGE: MACAG  |  variant={variant_name}  seed={seed}")
                self._run_macag_for_variant_seed(
                    variant_name=variant_name,
                    variant=variant,
                    seed=seed,
                    checkpoint_path=checkpoint_path,
                )

        aggregate: dict[str, Any] = {}
        if self.worker_id == 0 and self._stage_enabled("report"):
            self._banner("STAGE: Report aggregation")
            aggregate = self._generate_reporting()

        self._banner("Suite complete")
        return {
            "suite_root": str(self.suite_root),
            "worker_id": self.worker_id,
            "num_workers": self.num_workers,
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

    def _val_dataset_dir(self, model_name: str) -> Path:
        return self.val_datasets_root / _sanitize_name(model_name)

    @staticmethod
    def _has_split_files(train_dir: Path, val_dir: Path) -> bool:
        return (
            (train_dir / "mlp_inputs_train.npy").exists()
            and (train_dir / "mlp_outputs_train.npy").exists()
            and (val_dir / "mlp_inputs_val.npy").exists()
            and (val_dir / "mlp_outputs_val.npy").exists()
        )

    @staticmethod
    def _npy_is_truncated(path: Path) -> bool:
        """True if a .npy is sparse/partially written.

        A crashed collection can leave a val ``.npy`` whose header declares the
        full array size (so ``st_size`` reports the whole corpus) while only a
        small prefix of windows was actually written; the unwritten tail reads
        back as zeros. Such a file passes a plain existence check but poisons
        reconstruction (target ``y=0`` -> cosine 0, ``relative_error`` explodes).
        Compare bytes actually allocated on disk (``st_blocks`` is in 512-byte
        units) against the declared size: a fully materialised, dense .npy has a
        ratio near 1.0, while the corruption observed in the field was ~0.004.
        """
        try:
            st = path.stat()
        except OSError:
            return True
        if st.st_size <= 0:
            return True
        allocated_bytes = st.st_blocks * 512
        return allocated_bytes < 0.9 * st.st_size

    def _load_train_val_datasets(
        self,
        *,
        train_dir: Path,
        val_dir: Path,
    ) -> tuple[ActivationDataset, ActivationDataset | None]:
        """Load train and val datasets respecting the on-disk layout.

        Prefers the new split layout. Falls back to the legacy single-file
        layout (train/val will then be split at training time via random_split).
        """
        if self._has_split_files(train_dir, val_dir):
            train_ds = ActivationDataset.load(str(train_dir), split="train")
            val_ds = ActivationDataset.load(str(val_dir), split="val")
            return train_ds, val_ds
        # Legacy layout: return the full dataset as "train" and let the
        # trainer's random_split carve out a val slice at runtime.
        return ActivationDataset.load(str(train_dir)), None

    def _load_val_or_legacy(
        self,
        *,
        val_dir: Path,
        train_dir: Path,
    ) -> ActivationDataset:
        """Load the val dataset, falling back to the legacy single-file dataset.

        Used by the eval stage. With split files present, returns the val split
        only. Without them, returns the whole legacy dataset (downstream
        eval samples a deterministic subset from it).
        """
        val_in = val_dir / "mlp_inputs_val.npy"
        val_out = val_dir / "mlp_outputs_val.npy"
        if val_in.exists() and val_out.exists():
            if self._npy_is_truncated(val_in) or self._npy_is_truncated(val_out):
                raise RuntimeError(
                    f"Validation activations under {val_dir} are truncated/sparse "
                    "(a crashed collection left mostly-zero rows). Refusing to "
                    "evaluate reconstruction against zeros. Delete the val .npy "
                    "files and re-run with the collect_dataset stage enabled to "
                    "regenerate the hold-out."
                )
            return ActivationDataset.load(str(val_dir), split="val")
        return ActivationDataset.load(str(train_dir))

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
        train_dir = self._dataset_dir(model_name)
        val_dir = self._val_dataset_dir(model_name)
        has_split = (
            (train_dir / "mlp_inputs_train.npy").exists()
            and (train_dir / "mlp_outputs_train.npy").exists()
            and (val_dir / "mlp_inputs_val.npy").exists()
            and (val_dir / "mlp_outputs_val.npy").exists()
        )
        # Legacy single-file layouts (kept so suites with pre-existing caches
        # don't have to re-collect). New collections always use the split layout.
        has_torch = (train_dir / "mlp_inputs.pt").exists() and (
            train_dir / "mlp_outputs.pt"
        ).exists()
        has_numpy = (train_dir / "mlp_inputs.npy").exists() and (
            train_dir / "mlp_outputs.npy"
        ).exists()
        return has_split or has_torch or has_numpy

    def _models_needing_eval_only_collection(self, model_names: list[str]) -> list[str]:
        """Models whose (variant, seed) cells are all trained but whose val split is missing.

        Used in chunked-offline mode: if no chunks will be produced this run
        (because training is already complete) and eval needs the val split,
        we must do a one-off collection.
        """
        needed: list[str] = []
        for model_name in model_names:
            variants_for_model = [
                (vn, v)
                for vn, v in self.config.model_variants.items()
                if self._variant_model_name(v) == model_name
            ]
            if not variants_for_model:
                continue
            all_trained = all(
                self._train_stage_complete(vn, v, seed)
                for (vn, v) in variants_for_model
                for seed in self.config.seeds
            )
            if not all_trained:
                continue
            # Evaluation only ever reads the VAL split (see _evaluate_variant_seed),
            # which lives on persistent shared storage. The train cache lives on
            # per-node /lscratch and is wiped between jobs, so requiring it here
            # (via _dataset_stage_complete) would trigger a full n_tokens re-collection
            # that overflows /lscratch and SIGBUSes. Only re-collect if VAL is missing.
            if self._val_split_present(model_name):
                continue
            needed.append(model_name)
        return needed

    def _val_split_present(self, model_name: str) -> bool:
        """True if the val split eval needs is already on disk AND fully written.

        A truncated/sparse val (left by a crashed re-collection) is treated as
        missing so a fresh, chunk-bounded collection regenerates it, rather than
        letting eval silently score reconstruction against zero-filled rows.
        """
        val_dir = self._val_dataset_dir(model_name)
        train_dir = self._dataset_dir(model_name)
        val_in = val_dir / "mlp_inputs_val.npy"
        val_out = val_dir / "mlp_outputs_val.npy"
        has_val_split = val_in.exists() and val_out.exists()
        if has_val_split and (
            self._npy_is_truncated(val_in) or self._npy_is_truncated(val_out)
        ):
            self._log(
                f"Val split for {model_name} at {val_dir} is truncated/sparse; "
                "treating as missing and forcing re-collection."
            )
            has_val_split = False
        # Legacy single-file layouts that _load_val_or_legacy can fall back to.
        has_legacy = (
            (train_dir / "mlp_inputs.npy").exists()
            or (train_dir / "mlp_inputs.pt").exists()
        )
        return has_val_split or has_legacy

    def _train_stage_complete(self, variant_name: str, variant: ModelVariantConfig, seed: int) -> bool:
        if variant.checkpoint_path:
            return Path(variant.checkpoint_path).exists()
        # train_summary.json is written only at line 1064, after the full
        # training loop returns. Intermediate _best/ and _final/ checkpoint
        # dirs are written each chunk by the chunked-offline trainer for
        # resume purposes — their presence does NOT mean training finished.
        return (self._run_dir(variant_name, seed) / "train_summary.json").exists()

    def _collect_dataset(self, model_name: str, *, eval_only: bool = False) -> dict[str, Any]:
        dataset_dir = self._dataset_dir(model_name)
        summary_path = dataset_dir / "dataset_manifest.json"
        if self._dataset_stage_complete(model_name):
            self._log(f"Dataset already collected for {model_name}, skipping")
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

        val_dir = self._val_dataset_dir(model_name)

        # Eval only needs the fixed val hold-out, which is materialised from the
        # FIRST chunk of the corpus. Collecting the full n_tokens in one
        # non-chunked pass would stage every token's activations on node-local
        # /lscratch and SIGBUS (gpt2-small is ~18 KB/token, so 1B tokens far
        # exceeds the disk). Cap to a single chunk so the collection reproduces
        # training's chunk-0 val deterministically (same seed, skip_items=0) and
        # fits on disk.
        n_tokens = self.config.dataset.n_tokens
        chunk_toks = self.config.dataset.collection_chunk_n_tokens
        if eval_only and chunk_toks and chunk_toks > 0:
            n_tokens = min(n_tokens, int(chunk_toks))
            # A stale val_hashes.txt (from the original training chunk 0 or a
            # crashed re-collection) marks exactly these chunk-0 windows as
            # already reserved, so dedup would drop ALL of them and write an
            # empty val. Clear it first, mirroring the training chunk-0 path, so
            # this collection re-reserves the hold-out from scratch.
            if self.config.dataset.dedup:
                stale_hashes = val_dir / "val_hashes.txt"
                if stale_hashes.exists():
                    stale_hashes.unlink()

        self._log(
            f"Collecting activations: n_tokens={n_tokens}, "
            f"seq_len={self.config.dataset.seq_len}, batch_size={self.config.dataset.batch_size}, "
            f"val_fraction={self.config.dataset.val_fraction} -> "
            f"train_dir={dataset_dir}, val_dir={val_dir}"
            + (" (eval-only, capped to one chunk)" if eval_only else "")
        )
        dataset_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        data_config = DataConfig(
            model_name=model_name,
            dataset_name=self.config.dataset.dataset_name,
            dataset_config=self.config.dataset.dataset_config,
            n_tokens=n_tokens,
            seq_len=self.config.dataset.seq_len,
            batch_size=self.config.dataset.batch_size,
            save_dir=str(dataset_dir),
            val_save_dir=str(val_dir),
            val_fraction=self.config.dataset.val_fraction,
            device=str(_device_from_name(self.config.dataset.device)),
            dtype=self.config.dataset.dtype,
            seed=self.config.dataset.seed,
            load_after_collect=False,
            dedup=self.config.dataset.dedup,
            dedup_train=self.config.dataset.dedup_train,
            val_hashes_path=str(val_dir / "val_hashes.txt"),
        )
        collect_activations(data_config)
        summary = {
            "status": "completed",
            "model_name": model_name,
            "dataset_name": self.config.dataset.dataset_name,
            "dataset_config": self.config.dataset.dataset_config,
            "dataset_dir": str(dataset_dir),
            "val_dataset_dir": str(val_dir),
            "val_fraction": self.config.dataset.val_fraction,
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

    def _dataset_uses_chunked_offline(self) -> bool:
        c = self.config.dataset.collection_chunk_n_tokens
        return c is not None and c > 0

    @staticmethod
    def _chunk_exclusive_step_bounds(total_steps: int, n_chunks: int) -> list[int]:
        """Exclusive upper bounds per chunk (chunk ci covers steps while step < bounds[ci])."""
        bounds: list[int] = []
        acc = 0
        for ci in range(n_chunks):
            lo = (total_steps * ci) // n_chunks
            hi = (total_steps * (ci + 1)) // n_chunks
            acc += hi - lo
            bounds.append(acc)
        assert acc == total_steps
        return bounds

    def _train_variant_seed_chunked_offline(
        self,
        variant_name: str,
        variant: ModelVariantConfig,
        seed: int,
        checkpoint_dir: Path,
        training,
    ) -> bool:
        """Run the chunked-offline training loop.

        Returns True iff all chunks completed normally. Returns False if the
        loop stopped early on a graceful SIGTERM shutdown (walltime/scancel),
        so the caller can avoid marking training as complete and let the next
        job resume from the last completed-chunk checkpoint.
        """
        chunk_toks = int(self.config.dataset.collection_chunk_n_tokens or 0)
        total_tokens_budget = int(self.config.dataset.n_tokens)
        n_chunks = max(1, math.ceil(total_tokens_budget / chunk_toks))
        exclusive_bounds = self._chunk_exclusive_step_bounds(training.total_steps, n_chunks)

        run_name = self._variant_run_name(variant_name, variant, seed)
        ts_path = checkpoint_dir / f"{run_name}_training_state.pt"

        cpu_resume: dict | None = None
        skip_seq = 0
        skip_items = 0
        start_ci = 0
        if ts_path.exists():
            cpu_resume = torch.load(ts_path, map_location="cpu")
            skip_seq = int(cpu_resume.get("corpus_skip_sequences", 0))
            skip_items = int(cpu_resume.get("corpus_skip_items", 0))
            start_ci = int(cpu_resume.get("chunk_index", 0))
            # Collection now resumes by item cursor (skip_items), not by re-tokenizing
            # the window prefix. A pre-migration checkpoint has consumed chunks but no
            # item cursor; resuming it as-is would re-collect from item 0 and corrupt
            # the run. Refuse and point at the one-time migration instead.
            if start_ci > 0 and skip_items <= 0 and skip_seq > 0:
                raise RuntimeError(
                    f"Chunked resume for {run_name!r} has chunk_index={start_ci} and "
                    f"corpus_skip_sequences={skip_seq} but no corpus_skip_items cursor. "
                    "This checkpoint predates the item-cursor migration. Run it once, "
                    "single-process:\n"
                    "  python -m spline_clt.training.migrate_skip_cursor \\\n"
                    f"    --state {ts_path} \\\n"
                    "    --resolved-config <output_root>/<suite>/resolved_config.json\n"
                    "then resubmit the training job."
                )

        model_name = self._variant_model_name(variant)
        dataset_dir = self._dataset_dir(model_name)
        val_dataset_dir = self._val_dataset_dir(model_name)
        world_sz = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))

        self._log(
            f"Chunked offline: n_chunks={n_chunks}, tokens_per_chunk={chunk_toks}, "
            f"budget_tokens={total_tokens_budget}, resume_chunk={start_ci}, "
            f"corpus_skip_items={skip_items}, corpus_skip_sequences={skip_seq}"
        )

        # Per-chunk session: FSDP is released before each mid-run activation
        # collect (base LM only — no FSDP needed) so Gemma can own the GPUs,
        # then rebuilt from cpu_resume / training_state for the next train
        # chunk. Holding FSDP across collect was the chunk-N+1 OOM (co-resident
        # Gemma child + flat parent allocated but collapsed free HBM).
        session = None

        for ci in range(start_ci, n_chunks):
            tokens_this = min(chunk_toks, total_tokens_budget - ci * chunk_toks)
            if tokens_this <= 0:
                break

            if world_sz > 1 and not dist.is_initialized():
                from spline_clt.training.train import _init_distributed_process_group

                # Rank 0 may spend >10 min collecting a large activation chunk
                # before other ranks reach the metadata broadcast.
                _init_distributed_process_group(timeout=timedelta(hours=4))

            # NCCL requires each rank to use its own GPU. Chunked collect runs this
            # *before* train(), which normally calls torch.cuda.set_device(local_rank).
            # _init_distributed_process_group already set the device when it ran;
            # set again here so ranks that joined an already-initialized PG still
            # bind correctly before collection.
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if world_sz > 1 and torch.cuda.is_available():
                torch.cuda.set_device(local_rank)

            # Free Spline-CLT FSDP before GPU base-LM collect. All ranks must
            # participate (barrier inside release_session_gpu).
            if session is not None:
                self._log(
                    f"Chunk {ci + 1}/{n_chunks} (rank {rank}): releasing FSDP "
                    "GPU session before isolated base-LM activation collect"
                )
                release_session_gpu(session)
                session = None
                _scrub_cuda_memory(
                    log_fn=lambda msg: self._log(
                        f"Chunk {ci + 1}/{n_chunks} pre-collect (rank {rank}): {msg}"
                    ),
                )

            # The train cache lives on node-LOCAL storage (e.g. /lscratch NVMe),
            # so on a multi-node run every node needs its own replica of the
            # chunk. Collection is fully deterministic given (seed, skip_items)
            # — HF shuffle, window chunking, and the val permutation are all
            # seeded — so each node's LOCAL_RANK==0 collects the same chunk into
            # its own local dir in parallel (no network transfer). Only global
            # rank 0 writes the shared val split and the shared val-hash file;
            # other node leaders write their (identical, unused) val slice to a
            # node-local throwaway dir. On a single node this reduces to the
            # previous rank-0-only behavior.
            is_node_leader = local_rank == 0
            shared_val_hashes = val_dataset_dir / "val_hashes.txt"
            leader_snapshot = Path(dataset_dir) / "val_hashes_snapshot.txt"

            # Fresh run (ci==0): rank 0 clears a stale val-hash file from a prior
            # run BEFORE any other node leader snapshots it, so nobody reads
            # leftover reservations. Resumes (start_ci > 0) never hit ci==0, so
            # the accumulated file is preserved.
            if rank == 0 and ci == 0 and self.config.dataset.dedup:
                if shared_val_hashes.exists():
                    shared_val_hashes.unlink()
            if world_sz > 1:
                dist.barrier()

            # Non-zero node leaders read prior val hashes from a local snapshot
            # taken now — before global rank 0 can append this chunk's new
            # hashes (that append happens at the END of rank 0's collection,
            # which starts only after the barrier below). All leaders therefore
            # see identical prior hashes and derive identical train/val masks,
            # keeping the per-node train memmaps byte-identical.
            if is_node_leader and rank != 0 and self.config.dataset.dedup:
                Path(dataset_dir).mkdir(parents=True, exist_ok=True)
                if shared_val_hashes.exists():
                    shutil.copyfile(shared_val_hashes, leader_snapshot)
                elif leader_snapshot.exists():
                    leader_snapshot.unlink()
            if world_sz > 1:
                dist.barrier()

            if rank == 0:
                chunk_val_save_dir = val_dataset_dir
                chunk_val_hashes_path = shared_val_hashes
            else:
                chunk_val_save_dir = Path(dataset_dir) / "_val_node_local"
                chunk_val_hashes_path = leader_snapshot

            n_seq_chunk = 0
            n_items_chunk = 0
            if is_node_leader:
                data_config = DataConfig(
                    model_name=model_name,
                    dataset_name=self.config.dataset.dataset_name,
                    dataset_config=self.config.dataset.dataset_config,
                    n_tokens=tokens_this,
                    seq_len=self.config.dataset.seq_len,
                    batch_size=self.config.dataset.batch_size,
                    save_dir=str(dataset_dir),
                    val_save_dir=str(chunk_val_save_dir),
                    # Materialise the fixed validation split ONCE, on the first chunk
                    # of a fresh run; every later chunk collects train-only
                    # (val_fraction=0) so it never re-opens or overwrites the val
                    # memmaps. The per-sequence split already makes the chunk-0 val
                    # sequences disjoint from chunk-0 train, and later chunks draw
                    # from later corpus items, so the frozen val stays a clean
                    # hold-out used identically by every chunk's val_eval and by the
                    # eval stage. Resumes (start_ci > 0) skip straight to the reuse
                    # branch and read the val written by the original chunk 0.
                    val_fraction=(
                        self.config.dataset.val_fraction if ci == 0 else 0.0
                    ),
                    device=str(_device_from_name(self.config.dataset.device)),
                    dtype=self.config.dataset.dtype,
                    seed=self.config.dataset.seed,
                    load_after_collect=False,
                    skip_items=skip_items,
                    # Dedupe the val holdout and keep train disjoint from it by
                    # token-window content. Val hashes persist in the (stable) val
                    # dir so later chunks (val_fraction=0) still exclude held-out
                    # windows from train.
                    dedup=self.config.dataset.dedup,
                    dedup_train=self.config.dataset.dedup_train,
                    val_hashes_path=str(chunk_val_hashes_path),
                )
                _val_mode = (
                    f"collect fixed val ({self.config.dataset.val_fraction:.0%})"
                    if ci == 0
                    else "reuse fixed val from chunk 1"
                )
                self._log(
                    f"Chunk {ci + 1}/{n_chunks} (rank {rank}, node leader): "
                    f"collect n_tokens={tokens_this}, skip_items={skip_items}, {_val_mode}"
                )
                # Isolate base-LM load/teardown in a child so its CUDA allocator
                # churn cannot fragment the parent's slabs. FSDP was released
                # above when a prior chunk left a session — GPUs are free for
                # Gemma. Pin the child to this rank's GPU only.
                coll = collect_activations_isolated(
                    data_config,
                    cuda_visible_devices=str(local_rank),
                )
                n_seq_chunk = coll.n_sequences
                n_items_chunk = coll.n_items_consumed

            if world_sz > 1:
                if torch.cuda.is_available():
                    dev = torch.device("cuda", local_rank)
                else:
                    dev = torch.device("cpu")
                buf = torch.tensor(
                    [n_seq_chunk, n_items_chunk], dtype=torch.long, device=dev
                )
                # Rank 0 is authoritative for the corpus cursors. Every node
                # leader must have consumed the same item count (collection is
                # deterministic) — verified via the train-length check below.
                dist.broadcast(buf, src=0)
                n_seq_chunk = int(buf[0].item())
                n_items_chunk = int(buf[1].item())
                dist.barrier()

            # All ranks (including non-leaders that idled through collect): scrub
            # before the first post-collect backward, which needs a contiguous
            # ~9.26 GiB slab for the full triangular W_dec unshard.
            _scrub_cuda_memory(
                log_fn=lambda msg: self._log(
                    f"Chunk {ci + 1}/{n_chunks} post-collect (rank {rank}): {msg}"
                ),
            )

            train_dataset, val_dataset = self._load_train_val_datasets(
                train_dir=dataset_dir,
                val_dir=val_dataset_dir,
            )

            if world_sz > 1:
                # Guard against stale/desynced node-local caches: every rank must
                # see the same number of train sequences for this chunk, otherwise
                # the DistributedSampler shards diverge and collectives hang (or,
                # worse, some nodes silently train on a leftover chunk from a
                # previous job on that node's local disk).
                n_local = len(train_dataset)
                len_buf = torch.tensor([n_local], dtype=torch.long, device=dev)
                dist.broadcast(len_buf, src=0)
                n_ref = int(len_buf.item())
                if n_local != n_ref:
                    raise RuntimeError(
                        f"Rank {rank}: node-local train cache at {dataset_dir} has "
                        f"{n_local} sequences for chunk {ci + 1}/{n_chunks}, but rank 0 "
                        f"collected {n_ref}. The node-local activation cache is stale or "
                        "was not collected — wipe the local cache dir on every node "
                        "(e.g. rm -rf /lscratch/*) and resubmit."
                    )

            chunk_stop = exclusive_bounds[ci]
            chunk_floor = exclusive_bounds[ci - 1] if ci > 0 else 0

            # Walltime can leave training_state with completed_step already past
            # this chunk's stop while chunk_index still points here. Collect still
            # ran so we can advance corpus cursors; skip FSDP build / run_chunk
            # (0-step run_chunk would summon before lazy_init and crash).
            if session is not None:
                current_step = int(session.global_step)
            elif cpu_resume is not None:
                current_step = int(cpu_resume["completed_step"]) + 1
            else:
                current_step = 0
            if current_step >= chunk_stop:
                self._log(
                    f"Chunk {ci + 1}/{n_chunks}: training already complete "
                    f"(step={current_step} >= stop={chunk_stop}); "
                    f"advancing corpus cursors, skipping train."
                )
                skip_seq += n_seq_chunk
                skip_items += n_items_chunk
                if cpu_resume is not None:
                    cpu_resume["corpus_skip_sequences"] = skip_seq
                    cpu_resume["corpus_skip_items"] = skip_items
                    cpu_resume["chunk_index"] = ci + 1
                if rank == 0:
                    if cpu_resume is not None:
                        torch.save(cpu_resume, ts_path)
                    elif ts_path.exists():
                        pkg = torch.load(ts_path, map_location="cpu")
                        pkg["corpus_skip_sequences"] = skip_seq
                        pkg["corpus_skip_items"] = skip_items
                        pkg["chunk_index"] = ci + 1
                        torch.save(pkg, ts_path)
                if world_sz > 1:
                    dist.barrier()
                del train_dataset, val_dataset
                continue

            if session is None:
                # Build (or rebuild after pre-collect release) the FSDP session.
                # Per-chunk bounds are set on this TrainConfig; after a mid-run
                # release, cpu_resume reloads weights + Adam from disk.
                train_config = TrainConfig(
                    n_layers=training.n_layers,
                    d_model=training.d_model,
                    d_transcoder=training.d_transcoder,
                    encoder_type=training.encoder_type,
                    grid_size=training.grid_size,
                    spline_order=training.spline_order,
                    threshold_init=training.threshold_init,
                    jumprelu_bandwidth=training.jumprelu_bandwidth,
                    activation_function=training.activation_function,
                    threshold_weight_decay=training.threshold_weight_decay,
                    threshold_adam_eps=training.threshold_adam_eps,
                    threshold_init_strategy=training.threshold_init_strategy,
                    threshold_init_target_l0=training.threshold_init_target_l0,
                    threshold_calibration_samples=training.threshold_calibration_samples,
                    threshold_calibration_values_per_sample=training.threshold_calibration_values_per_sample,
                    decoder_init_strategy=training.decoder_init_strategy,
                    decoder_calibration_samples=training.decoder_calibration_samples,
                    normalize_inputs=training.normalize_inputs,
                    normalization_samples=training.normalization_samples,
                    learning_rate=training.learning_rate,
                    optimizer=training.optimizer,
                    weight_decay=training.weight_decay,
                    adam_beta1=training.adam_beta1,
                    adam_beta2=training.adam_beta2,
                    warmup_steps=training.warmup_steps,
                    total_steps=training.total_steps,
                    batch_size=training.batch_size,
                    lambda_sparsity=training.lambda_sparsity,
                    sparsity_warmup_steps=training.sparsity_warmup_steps,
                    sparsity_decay_start=training.sparsity_decay_start,
                    lambda_sparsity_final=training.lambda_sparsity_final,
                    sparsity_l0_floor=training.sparsity_l0_floor,
                    lambda_kan_reg=training.lambda_kan_reg,
                    scale_base=training.scale_base,
                    scale_spline=training.scale_spline,
                    lr_spline_mult=training.lr_spline_mult,
                    recon_normalization=training.recon_normalization,
                    sparsity_normalization=training.sparsity_normalization,
                    recon_layer_energy_beta=training.recon_layer_energy_beta,
                    c_sparsity=training.c_sparsity,
                    grad_clip=training.grad_clip,
                    log_every=training.log_every,
                    eval_every=training.eval_every,
                    save_every=training.save_every,
                    keep_last_checkpoints=training.keep_last_checkpoints,
                    checkpoint_dir=str(checkpoint_dir),
                    run_name=run_name,
                    update_grid_every=training.update_grid_every,
                    update_grid_from=training.update_grid_from,
                    reset_optimizer_every=training.reset_optimizer_every,
                    use_fsdp=training.use_fsdp,
                    fsdp_cpu_offload=training.fsdp_cpu_offload,
                    shard_kan_encoders=training.shard_kan_encoders,
                    data_dir=str(dataset_dir),
                    val_data_dir=str(val_dataset_dir),
                    device=str(_device_from_name(training.device or self.config.dataset.device)),
                    dtype=training.dtype,
                    seed=seed,
                    val_fraction=training.val_fraction,
                    num_workers=training.num_workers,
                    pin_memory=training.pin_memory,
                    prefetch_factor=training.prefetch_factor,
                    persistent_workers=training.persistent_workers,
                    dataloader_max_host_gib=training.dataloader_max_host_gib,
                    tf32=training.tf32,
                    log_dir=str(checkpoint_dir.parent),
                    wandb_project=training.wandb_project or self.config.wandb.project,
                    wandb_entity=training.wandb_entity or self.config.wandb.entity,
                    wandb_mode=training.wandb_mode if training.wandb_project else self.config.wandb.mode,
                    wandb_run_name=f"{self.config.suite_name}/{variant_name}_seed{seed}",
                    resume_training_if_exists=training.resume_training_if_exists,
                    chunk_stop_step=chunk_stop,
                    chunk_resume_step_floor=chunk_floor,
                    corpus_skip_sequences=skip_seq,
                    corpus_skip_items=skip_items,
                    training_chunk_index=ci,
                )
                session = build_session(
                    train_config,
                    resume_payload=cpu_resume,
                    norm_dataset=train_dataset,
                )
            else:
                # Same-chunk reuse only (no collect between); update bounds.
                session.config.chunk_stop_step = chunk_stop
                session.config.chunk_resume_step_floor = chunk_floor
                session.config.corpus_skip_sequences = skip_seq
                session.config.corpus_skip_items = skip_items
                session.config.training_chunk_index = ci

            self._log(
                f"Chunk {ci + 1}/{n_chunks}: train global steps [{chunk_floor}, {chunk_stop}) "
                f"→ {chunk_stop - chunk_floor} optimizer steps "
                f"(LR schedule uses global step counter out of total_steps={training.total_steps})"
            )
            cpu_resume, shutdown = run_chunk(session, train_dataset, val_dataset)
            del train_dataset, val_dataset

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                if rank == 0:
                    # Steady-state FSDP resident size; released again before the
                    # next chunk's base-LM collect.
                    self._log(
                        f"Chunk {ci + 1}/{n_chunks} cleanup: GPU "
                        f"allocated={torch.cuda.memory_allocated() / 1e9:.1f} GB, "
                        f"reserved={torch.cuda.memory_reserved() / 1e9:.1f} GB"
                    )

            # Graceful shutdown: run_chunk() caught SIGTERM (walltime/scancel) and
            # stopped mid-chunk without writing a partial training_state. Stop the
            # chunk loop WITHOUT advancing chunk_index/corpus_skip_sequences so the
            # next job re-runs this chunk cleanly from the last completed checkpoint.
            if shutdown:
                if rank == 0:
                    self._log(
                        f"Chunk {ci + 1}/{n_chunks}: graceful shutdown (SIGTERM) — "
                        "stopping chunk loop; resuming from last completed-chunk "
                        "checkpoint on the next job."
                    )
                if world_sz > 1:
                    dist.barrier()
                # Tear down the session and report incomplete so the caller does
                # NOT write train_summary.json — the next job resumes this chunk.
                if session is not None:
                    close_session(session)
                return False

            skip_seq += n_seq_chunk
            skip_items += n_items_chunk
            if cpu_resume is not None:
                cpu_resume["corpus_skip_sequences"] = skip_seq
                cpu_resume["corpus_skip_items"] = skip_items
                cpu_resume["chunk_index"] = ci + 1
            if rank == 0:
                if cpu_resume is not None:
                    torch.save(cpu_resume, ts_path)
                elif ts_path.exists():
                    pkg = torch.load(ts_path, map_location="cpu")
                    pkg["corpus_skip_sequences"] = skip_seq
                    pkg["corpus_skip_items"] = skip_items
                    pkg["chunk_index"] = ci + 1
                    torch.save(pkg, ts_path)
            if world_sz > 1:
                dist.barrier()

        # Tear down the single per-job session once all chunks are done.
        if session is not None:
            close_session(session)
        return True

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
            self._log(f"Training already complete for {variant_name} seed={seed}, skipping")
            payload = read_json(summary_path)
            return Path(payload["checkpoint_path"])

        if variant.checkpoint_path:
            self._log(f"Using external checkpoint: {variant.checkpoint_path}")
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

        training = variant.training
        chunked_offline = self._dataset_uses_chunked_offline()
        self._log(
            f"Training {variant_name} (seed={seed}): "
            f"encoder={training.encoder_type}, d_transcoder={training.d_transcoder}, "
            f"steps={training.total_steps}, batch_size={training.batch_size}, "
            f"num_workers={training.num_workers}, pin_memory={training.pin_memory}, "
            f"dtype={training.dtype}, "
            f"dataset_name={self.config.dataset.dataset_name}, "
            f"chunked_offline={chunked_offline}, "
            f"use_fsdp={training.use_fsdp}, fsdp_cpu_offload={training.fsdp_cpu_offload}"
        )

        if chunked_offline:
            train_dataset = None
            val_dataset = None
            self._log(
                "Chunked offline collection — per-chunk mmap writes with optimizer checkpoints."
            )
        else:
            dataset_dir = self._dataset_dir(self._variant_model_name(variant))
            val_dataset_dir = self._val_dataset_dir(self._variant_model_name(variant))
            self._log(
                "Loading activation datasets in streaming mode (mmap, no RAM copy): "
                f"train={dataset_dir}, val={val_dataset_dir}"
            )
            train_dataset, val_dataset = self._load_train_val_datasets(
                train_dir=dataset_dir,
                val_dir=val_dataset_dir,
            )

        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _dataset_dir = self._dataset_dir(self._variant_model_name(variant))
        _val_dataset_dir = self._val_dataset_dir(self._variant_model_name(variant))

        training_completed = True
        if chunked_offline:
            training_completed = self._train_variant_seed_chunked_offline(
                variant_name, variant, seed, checkpoint_dir, training
            )
        else:
            train_config = TrainConfig(
                n_layers=training.n_layers,
                d_model=training.d_model,
                d_transcoder=training.d_transcoder,
                encoder_type=training.encoder_type,
                grid_size=training.grid_size,
                spline_order=training.spline_order,
                threshold_init=training.threshold_init,
                jumprelu_bandwidth=training.jumprelu_bandwidth,
                activation_function=training.activation_function,
                threshold_weight_decay=training.threshold_weight_decay,
                threshold_adam_eps=training.threshold_adam_eps,
                threshold_init_strategy=training.threshold_init_strategy,
                threshold_init_target_l0=training.threshold_init_target_l0,
                threshold_calibration_samples=training.threshold_calibration_samples,
                threshold_calibration_values_per_sample=training.threshold_calibration_values_per_sample,
                decoder_init_strategy=training.decoder_init_strategy,
                decoder_calibration_samples=training.decoder_calibration_samples,
                normalize_inputs=training.normalize_inputs,
                normalization_samples=training.normalization_samples,
                learning_rate=training.learning_rate,
                optimizer=training.optimizer,
                weight_decay=training.weight_decay,
                adam_beta1=training.adam_beta1,
                adam_beta2=training.adam_beta2,
                warmup_steps=training.warmup_steps,
                total_steps=training.total_steps,
                batch_size=training.batch_size,
                lambda_sparsity=training.lambda_sparsity,
                sparsity_warmup_steps=training.sparsity_warmup_steps,
                sparsity_decay_start=training.sparsity_decay_start,
                lambda_sparsity_final=training.lambda_sparsity_final,
                sparsity_l0_floor=training.sparsity_l0_floor,
                lambda_kan_reg=training.lambda_kan_reg,
                scale_base=training.scale_base,
                scale_spline=training.scale_spline,
                lr_spline_mult=training.lr_spline_mult,
                recon_normalization=training.recon_normalization,
                sparsity_normalization=training.sparsity_normalization,
                recon_layer_energy_beta=training.recon_layer_energy_beta,
                c_sparsity=training.c_sparsity,
                grad_clip=training.grad_clip,
                log_every=training.log_every,
                eval_every=training.eval_every,
                save_every=training.save_every,
                keep_last_checkpoints=training.keep_last_checkpoints,
                checkpoint_dir=str(checkpoint_dir),
                run_name=self._variant_run_name(variant_name, variant, seed),
                update_grid_every=training.update_grid_every,
                update_grid_from=training.update_grid_from,
                reset_optimizer_every=training.reset_optimizer_every,
                use_fsdp=training.use_fsdp,
                fsdp_cpu_offload=training.fsdp_cpu_offload,
                shard_kan_encoders=training.shard_kan_encoders,
                data_dir=str(_dataset_dir),
                val_data_dir=str(_val_dataset_dir),
                device=str(_device_from_name(training.device or self.config.dataset.device)),
                dtype=training.dtype,
                seed=seed,
                val_fraction=training.val_fraction,
                num_workers=training.num_workers,
                pin_memory=training.pin_memory,
                prefetch_factor=training.prefetch_factor,
                persistent_workers=training.persistent_workers,
                dataloader_max_host_gib=training.dataloader_max_host_gib,
                tf32=training.tf32,
                log_dir=str(run_dir),
                wandb_project=training.wandb_project or self.config.wandb.project,
                wandb_entity=training.wandb_entity or self.config.wandb.entity,
                wandb_mode=training.wandb_mode if training.wandb_project else self.config.wandb.mode,
                wandb_run_name=f"{self.config.suite_name}/{variant_name}_seed{seed}",
                resume_training_if_exists=training.resume_training_if_exists,
                chunk_stop_step=None,
                corpus_skip_sequences=0,
                training_chunk_index=0,
            )
            train(train_config, dataset=train_dataset, val_dataset=val_dataset)
        checkpoint_path = self._resolve_checkpoint_path(variant_name, variant, seed)
        if not training_completed:
            # Graceful SIGTERM (walltime/scancel) stopped chunked training early.
            # Do NOT write train_summary.json — otherwise _train_stage_complete
            # would treat this partially-trained run as finished and the next job
            # would skip training entirely. Leaving the summary absent lets the
            # next job resume from the last completed-chunk checkpoint.
            self._log(
                f"Training interrupted (graceful shutdown) for {variant_name} "
                f"seed={seed}; not writing train_summary.json so the next job resumes."
            )
            return checkpoint_path
        # Under torchrun both ranks execute here. Only rank 0 writes files to
        # avoid a simultaneous-write race (content is identical but the OS
        # interleave can corrupt JSON). Non-rank-0 ranks just return the path.
        if int(os.environ.get("RANK", "0")) == 0:
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
            self._log(f"Evaluation already complete for {variant_name} seed={seed}, skipping")
            return read_json(summary_path)

        device = _device_from_name(variant.training.device or self.config.dataset.device)
        dtype = _dtype_from_name(variant.training.dtype)
        dataset_dir = self._dataset_dir(self._variant_model_name(variant))
        val_dataset_dir = self._val_dataset_dir(self._variant_model_name(variant))

        self._log(f"Loading checkpoint: {checkpoint_path}")
        self._log(
            "Loading val activations in streaming mode (mmap, no RAM copy)"
            f" from {val_dataset_dir}"
        )
        # Evaluation only ever touches the val split — never the train split.
        # Falls back to the legacy single-file layout if split files are missing.
        dataset = self._load_val_or_legacy(val_dir=val_dataset_dir, train_dir=dataset_dir)
        model = load_spline_clt(str(checkpoint_path), device=device, dtype=dtype)
        model.eval()

        base_record = self._record_base(
            variant_name=variant_name,
            variant=variant,
            seed=seed,
            checkpoint_path=checkpoint_path,
        )

        # -- Reconstruction --
        n_eval = self.config.dataset.eval_samples
        self._log(f"Evaluating reconstruction on {n_eval} samples...")
        sample_indices = deterministic_sample_indices(
            total=len(dataset),
            count=n_eval,
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
        self._log(
            f"Reconstruction done: MSE={reconstruction_summary['mse_total']:.4f}, "
            f"cosine={reconstruction_summary['cosine_similarity']:.4f}"
        )

        # -- Prompt-level evaluation (graph + circuit faithfulness) --
        n_prompts = len(self.config.benchmark_entries)
        self._log(f"Loading language model for {n_prompts} prompt evaluations...")
        lm = load_language_model(self._variant_model_name(variant), device=device)

        # Pre-collect all prompt caches while lm is alive on GPU. After this
        # we free the full HookedTransformer (multi-GB) before loading the
        # ReplacementModel, since they don't both need to be resident.
        self._log(f"Caching activations for {n_prompts} prompts...")
        cached_entries: list[tuple[BenchmarkEntry, Any]] = []
        for prompt_idx, entry in enumerate(self.config.benchmark_entries, 1):
            self._log(
                f"  Cache {prompt_idx}/{n_prompts}: [{entry.family}] {entry.prompt_id}"
            )
            cached_entries.append(
                (
                    entry,
                    collect_prompt_cache(
                        lm=lm,
                        prompt=entry.prompt,
                        n_layers=model.n_layers,
                        feature_input_hook=model.feature_input_hook,
                        feature_output_hook=model.feature_output_hook,
                    ),
                )
            )

        # Keep only the lm pieces still needed downstream (unembed/embed,
        # final LayerNorm, tokenizer, cfg). Holding references to these
        # submodules/tensors keeps them alive after `del lm`.
        lm_handle = SimpleNamespace(
            W_U=lm.W_U,
            W_E=lm.W_E,
            b_U=getattr(lm, "b_U", None),
            ln_final=lm.ln_final,
            tokenizer=lm.tokenizer,
            cfg=lm.cfg,
        )
        del lm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._log("Loading replacement model for attribution graph construction...")
        replacement_model = load_replacement_model(
            model_name=self._variant_model_name(variant),
            transcoders=model,
            device=device,
            dtype=dtype,
        )
        prompt_records: list[dict[str, Any]] = []
        graphs_dir = evaluation_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        for prompt_idx, (entry, prompt_cache) in enumerate(cached_entries, 1):
            self._log(
                f"Prompt {prompt_idx}/{n_prompts}: "
                f"[{entry.family}] {entry.prompt_id} -- {entry.prompt!r}"
            )
            prompt_records.append(
                self._evaluate_prompt_entry(
                    model=model,
                    lm=lm_handle,
                    replacement_model=replacement_model,
                    entry=entry,
                    prompt_cache=prompt_cache,
                    variant_name=variant_name,
                    variant=variant,
                    seed=seed,
                    checkpoint_path=checkpoint_path,
                    graphs_dir=graphs_dir,
                )
            )
        write_jsonl(evaluation_dir / "prompt_metrics.jsonl", prompt_records)

        # -- Monosemanticity --
        self._log("Evaluating monosemanticity...")
        monosemanticity_records, monosemanticity_summary = self._collect_monosemanticity(
            model=model,
            dataset=dataset,
            base_record=base_record,
            device=device,
        )
        write_jsonl(evaluation_dir / "monosemanticity_records.jsonl", monosemanticity_records)

        # -- Spline analysis --
        if model.encoder_type == "kan":
            self._log("Running spline analysis...")
        spline_summary, spline_records = self._collect_spline_analysis(
            model=model,
            dataset=dataset,
            evaluation_dir=evaluation_dir,
            device=device,
            base_record=base_record,
        )
        if spline_records:
            write_jsonl(evaluation_dir / "spline_records.jsonl", spline_records)

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
                "top5_match_rate": prompt_metric("top5_match_rate"),
                "top10_match_rate": prompt_metric("top10_match_rate"),
                "kl_divergence": prompt_metric("kl_divergence"),
            },
            "circuit_metrics": {
                "active_feature_count": prompt_metric("active_feature_count"),
                "attribution_seconds": prompt_metric("attribution_seconds"),
                "attribution_peak_mem_gib": prompt_metric("attribution_peak_mem_gib"),
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
        return evaluation_summary

    def _evaluate_prompt_entry(
        self,
        *,
        model: Any,
        lm: Any,
        replacement_model: Any,
        entry: BenchmarkEntry,
        prompt_cache: Any,
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
            self._log(f"  Computing replacement fidelity...")
            replacement_metrics = evaluate_prompt_replacement(model, prompt_cache, lm)
            self._log(
                f"  Building attribution graph (max_features="
                f"{self.config.evaluation.graph.max_features})..."
            )
            graph_info = build_prompt_graph(
                replacement_model=replacement_model,
                prompt_cache=prompt_cache,
                graph_dir=graphs_dir,
                graph_slug=graph_slug,
                scan_id=variant.scan_id or variant_name,
                max_features=self.config.evaluation.graph.max_features,
                max_n_logits=self.config.evaluation.graph.max_n_logits,
                desired_logit_prob=self.config.evaluation.graph.desired_logit_prob,
                node_threshold=self.config.evaluation.graph.node_threshold,
                edge_threshold=self.config.evaluation.graph.edge_threshold,
                attribution_batch_size=self.config.evaluation.graph.attribution_batch_size,
                model=model,
                lm=lm,
                spline_attribution_method=(
                    self.config.evaluation.graph.spline_attribution_method
                ),
            )
            self._log(
                f"  Graph: {graph_info['active_feature_count']} features, "
                f"replacement={graph_info['graph_replacement_score']:.3f}"
            )
            self._log(f"  Computing circuit faithfulness (logit gap)...")
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
                self._log(
                    f"  Running Shapley attribution "
                    f"({self.config.evaluation.circuit.shapley_samples} samples)..."
                )
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
        base_record: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not self.config.evaluation.splines.enabled:
            return {"status": "disabled"}, []
        if model.encoder_type != "kan":
            return {"status": "skipped", "reason": "encoder_type is not kan"}, []

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
        records: list[dict[str, Any]] = []
        scores: list[float] = []
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

            top_dim = dims[0]
            score = compute_nonlinearity_score(t_values, curves[top_dim])
            if math.isfinite(score):
                scores.append(score)
            records.append(
                {
                    **base_record,
                    "record_type": "spline_feature",
                    "layer_id": layer_id,
                    "feature_id": feature_id,
                    "top_input_dim": int(top_dim),
                    "activation_frequency": float(frequency[layer_id, feature_id].item()),
                    "nonlinearity_score": score,
                }
            )

        finite = [s for s in scores if math.isfinite(s)]
        summary: dict[str, Any] = {
            "status": "completed",
            "curve_file_count": len(curve_files),
            "curve_files": curve_files,
            "n_features_scored": len(finite),
            "mean_nonlinearity_score": float(np.mean(finite)) if finite else float("nan"),
            "median_nonlinearity_score": float(np.median(finite)) if finite else float("nan"),
            "fraction_nonlinear_gt_0_05": (
                float(sum(1 for s in finite if s > 0.05) / len(finite))
                if finite
                else float("nan")
            ),
        }
        return summary, records

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
        self._log(f"  MACAG: {len(prompt_records)} prompts to process")
        macag_records: list[dict[str, Any]] = []
        for pi, prompt_record in enumerate(prompt_records, 1):
            prompt_dir = macag_dir / _sanitize_name(str(prompt_record["prompt_id"]))
            prompt_dir.mkdir(parents=True, exist_ok=True)
            self._log(
                f"  MACAG prompt [{pi}/{len(prompt_records)}]: "
                f"{prompt_record.get('prompt_id', '?')} "
                f"(family={prompt_record.get('family', '?')})"
            )
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

        n_game1 = len(self.config.macag.game1_runs)
        n_game2 = len(self.config.macag.game2_runs)
        self._log(
            f"    Running {n_game1} Game-1 + {n_game2} Game-2 configs  "
            f"({len(candidates)} candidates)"
        )
        for gi, game1_cfg in enumerate(self.config.macag.game1_runs, 1):
            self._log(f"    Game 1 [{gi}/{n_game1}]: {game1_cfg.name}")
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

        for gi, game2_cfg in enumerate(self.config.macag.game2_runs, 1):
            self._log(f"    Game 2 [{gi}/{n_game2}]: {game2_cfg.name}")
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
