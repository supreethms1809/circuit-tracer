"""Training loop for Spline-CLT.

Standard PyTorch training with Adam optimizer, warmup + cosine decay,
logging, and checkpointing.
"""

import contextlib
import gc
import json
import math
import os
import re
import shutil
import signal
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import Adafactor

from spline_clt.kan_transcoder import KANCrossLayerTranscoder, load_spline_clt
from spline_clt.seed import make_generator, seed_everything
from spline_clt.training.data import ActivationDataset
from spline_clt.training.loss import compute_losses, total_loss


@dataclass
class TrainConfig:
    """Training configuration."""

    # Model architecture
    n_layers: int = 12
    d_model: int = 768
    d_transcoder: int = 4096
    encoder_type: str = "kan"  # "kan" or "linear" (baseline)
    grid_size: int = 5
    spline_order: int = 3
    #: Effective initial value for the learned JumpReLU thresholds. Must be > 0
    #: (stored internally as log θ; see KANCrossLayerTranscoder.__init__). The
    #: threshold is reparametrized as θ = exp(log θ) > 0 so it can never drift
    #: negative and invert the sparsity penalty; log(threshold_init) is the init.
    threshold_init: float = 0.001
    #: Standardize the per-layer encoder input by data mean/std before the
    #: encoder. Conditions the B-spline grid and equalizes per-dim scale.
    normalize_inputs: bool = True
    #: Number of sequences sampled to estimate the normalization statistics.
    normalization_samples: int = 256

    # Training
    learning_rate: float = 1e-4
    optimizer: str = "adamw"  # "adamw" or "adafactor"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    warmup_steps: int = 1000
    total_steps: int = 50_000
    batch_size: int = 4  # sequences per batch
    lambda_sparsity: float = 0.05
    c_sparsity: float = 1.0
    #: L1 coefficient on KAN spline weights. Was hardcoded to 0.01, which pulled
    #: the spline path toward zero hard enough that the encoder degenerated to its
    #: linear/SiLU base. Ignored when encoder_type == "linear".
    lambda_kan_reg: float = 0.001
    #: "global": one Σ(ŷ-y)²/Σy² over all layers — each layer is weighted by its
    #: share of ‖y‖², which on gpt2-small is 59% layer 2 (99.5% of that being the
    #: massive-activation dim #447), leaving layers 3-9 unmodelled.
    #: "per_layer": mean of each layer's own FVU, so every layer counts equally.
    #: Both equal 1.0 at zero output, so lambda_sparsity carries over.
    recon_normalization: str = "global"
    #: Energy-tempering exponent for per-layer recon weights: w_l ~ ||y_l||^(2*beta).
    #: 0 = uniform (every layer equal), 1 = global. Use a value like 0.5 when the
    #: model has extreme per-layer variance (qwen: layers 2 & 27 hold 87% of energy,
    #: which uniform per-layer under-weights). Only used when recon_normalization="per_layer".
    recon_layer_energy_beta: float = 0.0
    #: "sum": sparsity penalty is the per-token sum over all layers (deeper models
    #: feel more pressure at fixed λ). "mean": also divide by n_layers, making λ a
    #: per-layer-per-token cost that transfers across models of different depth.
    #: Note mean(λ=x) == sum(λ=x/n_layers) exactly.
    sparsity_normalization: str = "sum"
    grad_clip: float = 1.0

    # Logging & checkpointing
    log_every: int = 100
    eval_every: int = 5000
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints"
    run_name: str = "spline_clt"
    #: Retain only the newest N periodic ``{run_name}_step{K}`` checkpoint dirs;
    #: older ones are deleted after each successful save. ``{run_name}_best`` /
    #: ``{run_name}_final`` and the dir referenced by the freshly written
    #: training_state.pt are never deleted. <= 0 keeps everything (legacy).
    keep_last_checkpoints: int = 2
    # Where to write training_records.jsonl. Defaults to checkpoint_dir's parent
    # (typically the seed_xxx run dir) so the trail lives next to the run, not
    # inside the checkpoints/ subtree.
    log_dir: str | None = None

    # Optional Weights & Biases (gated by wandb_project being set; offline-safe via wandb_mode).
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_mode: str = "online"  # "online" | "offline" | "disabled"
    wandb_run_name: str | None = None

    # Grid update
    update_grid_every: int = 10_000
    update_grid_from: int = 2000  # don't update grid in early training
    update_grid_max_samples: int = 1024  # subsample to avoid OOM (efficient-kan lstsq uses O(batch*d_model*d_transcoder))

    # Optimizer restart (linear encoder only)
    # KAN resets Adam at every grid update; set this to the same interval for
    # linear so both models get the same number of fresh optimizer starts.
    # Value of 0 disables restarts (default for KAN, which handles it internally).
    reset_optimizer_every: int = 0
    use_fsdp: bool = False
    #: When True with ``use_fsdp``, offload FSDP-managed parameters to CPU when not in use
    #: (PyTorch ``CPUOffload(offload_params=True)``). Reduces GPU VRAM; increases step latency.
    fsdp_cpu_offload: bool = False

    # Data
    data_dir: str = "data/activations"
    #: Optional separate directory holding the val split files. When set, the
    #: trainer loads ``data_dir`` (split="train") and ``val_data_dir``
    #: (split="val") instead of doing a runtime random_split. Materialises
    #: the train/val partition at collection time.
    val_data_dir: str | None = None
    device: str = "cuda"
    dtype: str = "float32"
    seed: int = 0

    # Validation. ``val_fraction`` is only consulted when no pre-split val
    # dataset is available (legacy single-dir layout); the new collection
    # path materialises the split at write time.
    val_fraction: float = 0.05
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int | None = None
    persistent_workers: bool = False
    #: Host-RAM budget for DataLoader workers staging mmap samples. Above this the
    #: trainer falls back to synchronous loading. See _apply_mmap_dataloader_guard.
    dataloader_max_host_gib: float = 64.0
    #: Allow TF32 for fp32 matmuls. The KAN encoder is deliberately fp32, and
    #: without this its GEMMs fall back to CUDA-core SIMT SGEMM — profiled at
    #: 45.6 TFLOPS vs 141.3 with TF32 (3.1x) on a GH200. TF32 rounds only the
    #: multiply inputs to a 10-bit mantissa; accumulation, storage and the Adam
    #: update stay fp32, so this is unrelated to the bf16-underflow plateau.
    #: Disabled around B-spline grid refits regardless (see _tf32_disabled).
    tf32: bool = True

    #: Resume optimizer + LR schedule step counter from ``{checkpoint_dir}/{run_name}_training_state.pt``.
    resume_training_if_exists: bool = False
    #: Exclusive upper bound on the global step counter (same semantics as ``total_steps``).
    #: Used by chunked offline collection so each segment trains only its step slice.
    chunk_stop_step: int | None = None
    #: Minimum global step index expected when entering this chunk (0 for chunk 0).
    #: Validates ``training_state.pt`` matches the chunked schedule so we never silently
    #: restart from step 0 while ``chunk_stop_step`` still reflects a later cumulative bound.
    chunk_resume_step_floor: int | None = None
    #: Corpus cursor for chunked activation collection — persisted into training state only.
    #: Legacy window-level counter; kept for record/back-compat. The active resume
    #: cursor is ``corpus_skip_items`` (item-level), which avoids re-tokenizing the
    #: consumed prefix every chunk.
    corpus_skip_sequences: int = 0
    #: Item-level corpus cursor for chunked activation collection — persisted into
    #: training state and used to resume collection at the next item boundary.
    corpus_skip_items: int = 0
    #: Current offline chunk index (0-based); persisted for chunked-collection resume.
    training_chunk_index: int = 0

    def get_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "bfloat16": torch.bfloat16}[self.dtype]


def _distributed_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def _is_main_process(rank: int) -> bool:
    return rank == 0


def _save_model_checkpoint(
    model: torch.nn.Module,
    base_model: KANCrossLayerTranscoder,
    save_path: str,
    use_fsdp: bool,
    is_main_process: bool,
) -> None:
    if use_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        dist.barrier()
        with FSDP.summon_full_params(model, writeback=False, rank0_only=True):
            if is_main_process:
                base_model.to_safetensors(save_path)
        dist.barrier()
        return
    if is_main_process:
        base_model.to_safetensors(save_path)


def _training_state_path(config: TrainConfig) -> str:
    return os.path.join(config.checkpoint_dir, f"{config.run_name}_training_state.pt")


def _prune_step_checkpoints(config: TrainConfig, is_main_process: bool) -> None:
    """Delete all but the newest ``keep_last_checkpoints`` periodic step checkpoints.

    Periodic ``{run_name}_step{K}`` dirs exist only for crash resume, and the
    freshly written training_state.pt always references the newest one, so the
    older ones are dead weight — one full model copy per ``save_every`` interval
    accumulating for the whole run. ``{run_name}_best`` and ``{run_name}_final``
    are never touched (the ``_step(\\d+)$`` pattern cannot match them). Called on
    rank 0 only, strictly AFTER the new checkpoint + training_state are on disk,
    so a crash mid-prune can never lose the current resume point.
    """
    if not is_main_process or config.keep_last_checkpoints <= 0:
        return
    pattern = re.compile(re.escape(config.run_name) + r"_step(\d+)$")
    try:
        entries = os.listdir(config.checkpoint_dir)
    except FileNotFoundError:
        return
    step_dirs: list[tuple[int, str]] = []
    for name in entries:
        match = pattern.fullmatch(name)
        full = os.path.join(config.checkpoint_dir, name)
        if match and os.path.isdir(full):
            step_dirs.append((int(match.group(1)), full))
    step_dirs.sort()
    for _, path in step_dirs[: -config.keep_last_checkpoints]:
        shutil.rmtree(path, ignore_errors=True)
        print(f"[train] Pruned old step checkpoint: {path}")


def _recursive_cpu_clone(obj: Any) -> Any:
    """Deep-copy nested structures, moving tensors to CPU (detached clones)."""

    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _recursive_cpu_clone(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_recursive_cpu_clone(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_recursive_cpu_clone(v) for v in obj)
    return obj


def _staging_copy_training_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """RAM blob safe to retain across chunked ``train()`` calls (optimizer off-GPU)."""

    out = dict(payload)
    for key in ("optimizer_state_dict", "optimizer_local_state_dict"):
        if key in out and out[key] is not None:
            out[key] = _recursive_cpu_clone(out[key])
    return out


def _fsdp_optim_state_dict_ctx(model_for_train: torch.nn.Module):
    """Context for rank0-only, CPU-offloaded FSDP optim state dict gather/load.

    Halves host RAM during chunked-resume by keeping the full optimizer state
    on rank 0 only. ``optim_state_dict_to_load`` inside this context will
    broadcast from rank 0 to other ranks during the load, so we don't need to
    handle the broadcast manually.
    """
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullOptimStateDictConfig,
        StateDictType,
    )

    return FSDP.state_dict_type(
        model_for_train,
        StateDictType.FULL_STATE_DICT,
        optim_state_dict_config=FullOptimStateDictConfig(
            rank0_only=True, offload_to_cpu=True
        ),
    )


def _gather_optimizer_state(
    model_for_train: torch.nn.Module,
    optimizer_main: torch.optim.Optimizer,
    optimizer_local: torch.optim.Optimizer | None,
    *,
    use_fsdp: bool,
    rank: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    """Return (main_sd, local_sd).

    Under FSDP we use ``rank0_only=True``: the full ``main_sd`` lives on rank 0
    only (CPU-offloaded); non-zero ranks get an empty dict. ``optim_state_dict_to_load``
    handles the inverse broadcast on resume.

    ``local_sd`` covers KAN encoder params (replicated, FSDP-ignored). Because
    encoder gradients are explicitly all-reduced each step, the per-rank
    ``optimizer_local.state_dict()`` is bit-identical, so we likewise keep only
    rank 0's copy.
    """
    if use_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with _fsdp_optim_state_dict_ctx(model_for_train):
            main_sd = FSDP.optim_state_dict(model_for_train, optimizer_main)
    else:
        main_sd = optimizer_main.state_dict()
    if optimizer_local is not None and rank == 0:
        local_sd = optimizer_local.state_dict()
    else:
        local_sd = None
    return main_sd, local_sd


def _load_optimizer_from_training_state(
    model_for_train: torch.nn.Module,
    optimizer_main: torch.optim.Optimizer,
    optimizer_local: torch.optim.Optimizer | None,
    main_sd: dict[str, torch.Tensor] | None,
    local_sd: dict[str, torch.Tensor] | None,
    *,
    use_fsdp: bool,
    distributed: bool,
) -> None:
    if main_sd is not None or use_fsdp:
        # Always enter the rank0_only context under FSDP — rank 0 has main_sd,
        # other ranks have an empty dict and rely on the in-context broadcast.
        if use_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            with _fsdp_optim_state_dict_ctx(model_for_train):
                to_load = FSDP.optim_state_dict_to_load(
                    model_for_train,
                    optimizer_main,
                    main_sd if main_sd is not None else {},
                )
            optimizer_main.load_state_dict(to_load)
            del to_load
        elif main_sd is not None:
            optimizer_main.load_state_dict(main_sd)
    if optimizer_local is not None:
        # Local state lives only on rank 0 in the staged payload; broadcast it
        # to every rank before load. Using broadcast_object_list keeps the
        # nested-tensor structure intact.
        if distributed:
            payload = [local_sd] if dist.get_rank() == 0 else [None]
            dist.broadcast_object_list(payload, src=0)
            local_sd = payload[0]
        if local_sd is not None:
            optimizer_local.load_state_dict(local_sd)


def _all_reduce_encoder_grads(
    encoder_params: list[torch.nn.Parameter],
    world_size: int,
) -> None:
    """Average KAN encoder gradients across ranks.

    KAN encoders are FSDP-ignored (kept fp32 to protect the B-spline grid),
    so FSDP does not all-reduce their gradients. Without this sync, each rank
    would step its replicated encoder copy on local data only and silently
    diverge between ``update_grid`` broadcasts.
    """
    if world_size <= 1:
        return
    for p in encoder_params:
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


def _set_lr(optimizers: list[torch.optim.Optimizer | None], lr: float) -> None:
    for opt in optimizers:
        if opt is None:
            continue
        for pg in opt.param_groups:
            pg["lr"] = lr


def _save_training_state(
    *,
    path: str,
    model_for_train: torch.nn.Module,
    optimizer_main: torch.optim.Optimizer,
    optimizer_local: torch.optim.Optimizer | None,
    completed_step: int,
    best_val_loss: float,
    model_checkpoint_dir: str,
    corpus_skip_sequences: int,
    corpus_skip_items: int,
    chunk_index: int,
    wandb_run_id: str | None,
    use_fsdp: bool,
    is_main_process: bool,
    distributed: bool,
) -> dict[str, Any]:
    """Persist Adam/Adafactor state and LR schedule counter (via completed_step).

    All ranks participate in FSDP optimizer state aggregation; only rank 0 writes the file.

    Returns:
        The payload dict written (same structure as ``torch.load`` would recover).
    """
    rank = int(os.environ.get("RANK", "0"))
    main_sd, local_sd = _gather_optimizer_state(
        model_for_train,
        optimizer_main,
        optimizer_local,
        use_fsdp=use_fsdp,
        rank=rank,
    )
    payload: dict[str, Any] = {
        "completed_step": int(completed_step),
        "best_val_loss": float(best_val_loss),
        "optimizer_state_dict": main_sd,
        "optimizer_local_state_dict": local_sd,
        "model_checkpoint_dir": os.path.abspath(model_checkpoint_dir),
        "corpus_skip_sequences": int(corpus_skip_sequences),
        "corpus_skip_items": int(corpus_skip_items),
        "chunk_index": int(chunk_index),
    }
    if wandb_run_id is not None:
        payload["wandb_run_id"] = str(wandb_run_id)
    if distributed:
        dist.barrier()
    if is_main_process:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(payload, path)
    if distributed:
        dist.barrier()
    return payload


def get_lr(step: int, config: TrainConfig) -> float:
    """Compute learning rate with linear warmup + cosine decay."""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / max(
        1, config.total_steps - config.warmup_steps
    )
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def create_optimizer(
    params,
    config: TrainConfig,
    lr: float,
) -> torch.optim.Optimizer:
    """Build optimizer according to config.

    Adafactor is supported for very large models where Adam-family optimizer
    states exceed device memory.
    """
    params = list(params)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2),
        )
    if config.optimizer == "adafactor":
        return Adafactor(
            params,
            lr=lr,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def _build_optimizers(
    model: KANCrossLayerTranscoder,
    model_for_train: torch.nn.Module,
    config: TrainConfig,
    lr: float,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer | None]:
    """Build (optimizer_main, optimizer_local).

    For ``encoder_type == "kan"`` the encoder params (replicated, fp32,
    FSDP-ignored) are split into a separate optimizer so their Adam state
    can be saved/restored across chunk boundaries without going through
    FSDP's optim_state_dict (which only covers FSDP-managed params). Also
    makes the grid-update optimizer reset narrower (decoder Adam state is
    preserved across grid updates).
    """
    if config.encoder_type == "kan":
        encoder_params = list(model.encoders.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        other_params = [
            p for p in model_for_train.parameters() if id(p) not in encoder_ids
        ]
        optimizer_main = create_optimizer(other_params, config, lr=lr)
        optimizer_local = create_optimizer(encoder_params, config, lr=lr)
        return optimizer_main, optimizer_local
    return create_optimizer(model_for_train.parameters(), config, lr=lr), None


# ---------------------------------------------------------------------------
# Cooperative shutdown (SIGTERM / SIGINT)
# ---------------------------------------------------------------------------
# Slurm delivers SIGTERM to the batch shell 120s before walltime (the sbatch uses
# --signal=B:TERM@120), and `scancel` sends SIGTERM too. If that signal interrupts
# a rank while it is blocked inside an NCCL collective or a CUDA driver call, the
# process wedges in uninterruptible (D-state) sleep, becomes unkillable (even by
# SIGKILL), holds the GPU, and forces a node reboot. To avoid that we catch the
# signal cooperatively: set a flag, finish the current step at a collective-safe
# point, checkpoint, and exit cleanly between collectives.
#
# Caveat: a Python signal handler only runs between bytecode ops on the main
# thread, so a SIGTERM that lands *while* the main thread is inside a C++ NCCL
# collective is deferred until that call returns. The NCCL watchdog
# (TORCH_NCCL_ASYNC_ERROR_HANDLING / heartbeat monitor, set in the sbatch) is the
# backstop for that residual case. Handler + watchdog are complementary.
_SHUTDOWN_REQUESTED = False


def _install_shutdown_handler() -> None:
    """Install SIGTERM/SIGINT handlers that request a graceful stop.

    No-op (best effort) if not called from the main thread, e.g. under some test
    runners where ``signal.signal`` raises ``ValueError``.
    """
    def _handler(signum, _frame):  # noqa: ANN001
        global _SHUTDOWN_REQUESTED
        _SHUTDOWN_REQUESTED = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


def shutdown_requested() -> bool:
    """Local view of the cooperative-shutdown flag (this process only)."""
    return _SHUTDOWN_REQUESTED


def _shutdown_requested_global(distributed: bool, device: torch.device) -> bool:
    """Collective-agreed shutdown: True if ANY rank caught SIGTERM/SIGINT.

    Must be called by all ranks together (it issues an all_reduce) so the whole
    group leaves the training loop in lockstep — otherwise one rank breaks while
    a peer enters the next collective and hangs.
    """
    local = 1.0 if _SHUTDOWN_REQUESTED else 0.0
    if not distributed:
        return local > 0.0
    flag = torch.tensor([local], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return flag.item() > 0.0


@dataclass
class TrainSession:
    """Persistent per-job training state for build-once chunked training.

    Built once via :func:`build_session` and reused across :func:`run_chunk`
    calls so the model + FSDP wrapper + optimizers + wandb run are NOT rebuilt
    every chunk — the per-chunk rebuild is what made GPU memory grow ~12 GB per
    chunk until the card saturated and steps thrashed. Mutable fields
    (optimizers, ``global_step``, ``best_val_loss``, pending optimizer state,
    NaN counter) are written back by :func:`run_chunk` each chunk.
    """

    config: TrainConfig
    rank: int
    world_size: int
    local_rank: int
    distributed: bool
    device: torch.device
    dtype: torch.dtype
    model_build_dtype: torch.dtype
    is_main_process: bool
    model: KANCrossLayerTranscoder
    model_for_train: torch.nn.Module
    optimizer: torch.optim.Optimizer
    optimizer_local: torch.optim.Optimizer | None
    encoder_params: list
    encoder_id_set: set
    encoder_needs_manual_reduce: bool
    wandb_run: Any
    wandb_run_id_for_state: str | None
    log_file: Any
    ts_path: str
    global_step: int
    best_val_loss: float
    pending_optimizer_sd: dict | None = None
    pending_optimizer_local_sd: dict | None = None
    n_nan_consecutive: int = 0


def _resolve_device(config: TrainConfig, distributed: bool, local_rank: int) -> torch.device:
    if config.device:
        if config.device == "cuda":
            return torch.device("cuda", local_rank if distributed else 0)
        return torch.device(config.device)
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank if distributed else 0)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _enable_tf32(config: TrainConfig, is_main_process: bool) -> None:
    """Route fp32 matmuls through TF32 tensor cores."""
    if not config.tf32 or not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if is_main_process:
        print("[train] TF32 enabled for fp32 matmuls (KAN encoder).")


@contextlib.contextmanager
def _tf32_disabled():
    """Force true fp32 matmuls inside the block.

    ``update_grid`` refits the B-spline coefficients by least squares against the
    repositioned knots. That solve is ill-conditioned enough that a 10-bit
    mantissa is not obviously safe, and it runs a few times per job, so pay full
    fp32 for it.
    """
    prev_mm = torch.backends.cuda.matmul.allow_tf32
    prev_prec = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_mm
        torch.set_float32_matmul_precision(prev_prec)


def _apply_mmap_dataloader_guard(config: TrainConfig, probe: Any) -> None:
    """Bound DataLoader worker memory for mmap-backed streaming datasets.

    Each in-flight sample is ``2 * n_layers * seq_len * d_model`` bf16 elements
    (23 MB at gpt2-large shapes), and a worker holds ``prefetch_factor *
    batch_size`` of them in /dev/shm.

    ``num_workers > 0`` is opt-in and genuinely risky here, because the estimate
    below is a *lower bound* on what the cgroup is charged:

    * mmap page cache. Workers stream a ~687 GiB activation chunk off /lscratch;
      those pages are charged to the step's cgroup. They are reclaimable in
      isolation.
    * ``pin_memory`` buffers are page-locked and therefore **not** reclaimable, so
      under pressure the kernel OOM-kills the step rather than evicting cache.

    Together those sank a 4-worker/pinned gpt2-large run at MaxRSS 1.17 TB on a
    1.24 TiB node. If you enable workers, leave ``pin_memory=False``.

    ``probe`` may be a Subset (legacy random_split) without ``mlp_inputs`` — guard
    with getattr so only true mmap datasets trip it.
    """
    mlp = getattr(probe, "mlp_inputs", None)
    if not isinstance(mlp, np.ndarray):
        return
    if config.num_workers <= 0:
        config.pin_memory = False
        config.prefetch_factor = None
        config.persistent_workers = False
        return

    if config.pin_memory:
        print(
            "[train] mmap streaming dataset: pin_memory=True is unsafe here "
            "(page-locked buffers cannot be reclaimed against the mmap page cache); "
            "forcing pin_memory=False."
        )
        config.pin_memory = False

    sample_bytes = 2 * int(np.prod(mlp.shape[1:])) * 2  # mlp_inputs + mlp_outputs, bf16
    prefetch = config.prefetch_factor or 2
    # Workers each stage `prefetch` batches through /dev/shm.
    in_flight = config.num_workers * prefetch * config.batch_size
    est_gib = sample_bytes * in_flight / 2**30

    if est_gib > config.dataloader_max_host_gib:
        print(
            f"[train] mmap streaming dataset: estimated DataLoader host RAM "
            f"{est_gib:.1f} GiB > dataloader_max_host_gib="
            f"{config.dataloader_max_host_gib} GiB; falling back to synchronous "
            f"loading (num_workers=0)."
        )
        config.num_workers = 0
        config.pin_memory = False
        config.prefetch_factor = None
        config.persistent_workers = False
    else:
        print(
            f"[train] mmap streaming dataset: num_workers={config.num_workers}, "
            f"prefetch_factor={prefetch}, pin_memory={config.pin_memory} "
            f"(~{est_gib:.1f} GiB host RAM in flight)"
        )
        config.prefetch_factor = prefetch
        config.persistent_workers = config.persistent_workers and config.num_workers > 0


def build_session(
    config: TrainConfig,
    *,
    resume_payload: dict[str, Any] | None = None,
    norm_dataset: ActivationDataset | None = None,
) -> TrainSession:
    """Build model + FSDP wrap + optimizers + wandb run ONCE per job.

    The chunked paper runner calls this once and then :func:`run_chunk` per
    chunk, so the heavy objects are not rebuilt each chunk. On a cold start
    (no ``resume_payload`` / no on-disk ``training_state.pt``) and when
    ``config.normalize_inputs`` is set, ``norm_dataset`` is required to estimate
    input normalization before the FSDP wrap. On resume, normalization is loaded
    from the checkpoint and ``norm_dataset`` is unused.

    The on-disk checkpoint format read here (``training_state.pt`` + the
    safetensors model dir) is identical to the legacy per-chunk path, so existing
    runs resume unchanged.
    """
    rank, world_size, local_rank = _distributed_env()
    distributed = world_size > 1
    _enable_tf32(config, _is_main_process(rank))
    if config.use_fsdp and not distributed:
        raise ValueError(
            "use_fsdp=True requires distributed launch (e.g. torchrun --nproc_per_node=2 ...)."
        )
    if distributed and not config.use_fsdp:
        raise ValueError(
            "Distributed training requires use_fsdp=True. DDP and other distributed "
            "wrappers are not supported by this trainer."
        )
    if config.fsdp_cpu_offload and not config.use_fsdp:
        raise ValueError("fsdp_cpu_offload=True requires use_fsdp=True.")
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(hours=4))

    device = _resolve_device(config, distributed, local_rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if config.fsdp_cpu_offload and device.type != "cuda":
        raise ValueError("fsdp_cpu_offload=True requires a CUDA training device.")
    dtype = config.get_dtype()
    # Seed before model init so cold-start threshold/spline init is deterministic.
    seed_everything(config.seed + config.training_chunk_index)
    is_main_process = _is_main_process(rank)

    # Catch SIGTERM/SIGINT cooperatively so a walltime/scancel signal checkpoints
    # and exits between collectives instead of wedging a rank into a D-state orphan.
    _install_shutdown_handler()

    ts_path = _training_state_path(config)
    training_pkg: dict[str, Any] | None = None
    if resume_payload is not None:
        training_pkg = resume_payload
    elif config.resume_training_if_exists and os.path.isfile(ts_path):
        training_pkg = torch.load(ts_path, map_location="cpu")

    # Under FSDP we build the model in float32 so FSDP keeps an fp32 master copy
    # of every managed parameter (decoder, biases, thresholds) and the optimizer
    # state stays fp32. MixedPrecision below casts to bfloat16 only for compute.
    # Without this the model lives in bfloat16 with bfloat16 Adam state, and any
    # update smaller than ~2^-8 relative to the weight rounds away — the late
    # training plateau. (KAN encoders are already pinned to fp32 internally.)
    model_build_dtype = torch.float32 if config.use_fsdp else dtype

    # Create model
    model = KANCrossLayerTranscoder(
        n_layers=config.n_layers,
        d_model=config.d_model,
        d_transcoder=config.d_transcoder,
        encoder_type=config.encoder_type,
        grid_size=config.grid_size,
        spline_order=config.spline_order,
        activation_function="jump_relu",
        threshold_init=config.threshold_init,
        device=device,
        dtype=model_build_dtype,
    )

    if training_pkg is not None:
        recovered = load_spline_clt(
            training_pkg["model_checkpoint_dir"],
            device=device,
            dtype=model_build_dtype,
        )
        model.load_state_dict(recovered.state_dict())
        del recovered
    elif config.normalize_inputs:
        # Fresh start: estimate per-layer input normalization from a sample of
        # the training data and bake it into the model buffers BEFORE the FSDP
        # wrap. Deterministic + seeded, so every rank computes identical stats.
        if norm_dataset is None:
            raise ValueError(
                "build_session(norm_dataset=...) is required on a cold start when "
                "config.normalize_inputs=True."
            )
        from spline_clt.training.data import compute_input_normalization

        if is_main_process:
            print(
                f"[train] Estimating input normalization from "
                f"{config.normalization_samples} sequences..."
            )
        norm_mean, norm_std = compute_input_normalization(
            norm_dataset,
            n_layers=config.n_layers,
            d_model=config.d_model,
            n_sequences=config.normalization_samples,
            seed=config.seed,
        )
        model.set_input_normalization(norm_mean, norm_std)
        if is_main_process:
            print(
                f"[train] Input normalization set "
                f"(mean abs-range [{norm_mean.abs().min():.3g}, {norm_mean.abs().max():.3g}], "
                f"std range [{norm_std.min():.3g}, {norm_std.max():.3g}])"
            )

    model_for_train: torch.nn.Module = model
    if config.use_fsdp:
        from torch.distributed.fsdp import (
            CPUOffload,
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )

        mp_policy = None
        if dtype == torch.bfloat16:
            # param_dtype=bf16: cast the fp32 master weights to bf16 for compute.
            # reduce_dtype=fp32: accumulate the gradient all-reduce in fp32 so
            #   small gradients don't vanish in the cross-rank reduction.
            # buffer_dtype=fp32: keep the input-normalization mean/std buffers in
            #   fp32 (their values can be large; bf16 would lose precision).
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )
        # KAN encoders are intentionally kept in fp32 for numerical stability.
        # Exclude them from FSDP flattening to avoid mixed-dtype flat parameter errors.
        ignored_modules = list(model.encoders) if config.encoder_type == "kan" else None
        cpu_offload = CPUOffload(offload_params=True) if config.fsdp_cpu_offload else None
        model_for_train = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            ignored_modules=ignored_modules,
            device_id=device if device.type == "cuda" else None,
            cpu_offload=cpu_offload,
        )

    # Optimizer (NOT LBFGS — doesn't scale)
    optimizer, optimizer_local = _build_optimizers(
        model, model_for_train, config, lr=config.learning_rate
    )
    encoder_params = (
        list(model.encoders.parameters()) if config.encoder_type == "kan" else []
    )
    encoder_id_set = {id(p) for p in encoder_params}
    # KAN encoders are FSDP-ignored so FSDP doesn't all-reduce their grads.
    # Single-GPU has nothing to reduce. Only the FSDP path needs the explicit sync.
    encoder_needs_manual_reduce = bool(encoder_params) and config.use_fsdp
    # Defer restoring optimizer state until after the first backward when resuming.
    # Loading Adam buffers before forward overlaps FSDP's temporary full-param unshard
    # (~10+ GiB) with optimizer state GPU memory; cold starts avoid that peak because
    # Adam allocates moments lazily on first step().
    pending_optimizer_sd: dict | None = None
    pending_optimizer_local_sd: dict | None = None
    if training_pkg is not None:
        osd = training_pkg.get("optimizer_state_dict")
        if osd is not None:
            pending_optimizer_sd = osd
        local_osd = training_pkg.get("optimizer_local_state_dict")
        if local_osd is not None:
            pending_optimizer_local_sd = local_osd
        elif config.encoder_type == "kan" and is_main_process:
            print(
                "[train] WARNING: resume payload has no optimizer_local_state_dict; "
                "KAN encoder Adam state will start fresh. This is expected only when "
                "loading a pre-fix checkpoint."
            )

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    log_dir = config.log_dir or os.path.dirname(os.path.abspath(config.checkpoint_dir))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training_records.jsonl")
    log_file = open(log_path, "a", buffering=1) if is_main_process else None

    resume_wandb_id: str | None = None
    if training_pkg is not None:
        rid = training_pkg.get("wandb_run_id")
        if rid is not None:
            resume_wandb_id = str(rid)

    wandb_run = None
    if is_main_process and config.wandb_project and config.wandb_mode != "disabled":
        try:
            import wandb  # type: ignore

            init_kw: dict = {
                "project": config.wandb_project,
                "entity": config.wandb_entity,
                "name": config.wandb_run_name or config.run_name,
                "mode": config.wandb_mode,  # type: ignore[arg-type]
                "dir": log_dir,
                "config": {k: getattr(config, k) for k in config.__dataclass_fields__},
            }
            if resume_wandb_id:
                init_kw["id"] = resume_wandb_id
                init_kw["resume"] = "allow"
            wandb_run = wandb.init(**init_kw)
            if resume_wandb_id:
                print(f"[train] Resuming Weights & Biases run id={resume_wandb_id}")
        except Exception as exc:  # pragma: no cover - wandb is optional
            print(f"[train] wandb init failed ({exc!r}); continuing with jsonl only")
            wandb_run = None

    wandb_run_id_for_state: str | None = None
    if wandb_run is not None:
        wandb_run_id_for_state = wandb.run.id

    step = training_pkg["completed_step"] + 1 if training_pkg is not None else 0
    best_val_loss = (
        float(training_pkg["best_val_loss"])
        if training_pkg is not None
        else float("inf")
    )

    floor = config.chunk_resume_step_floor
    if floor is None:
        floor = 0
    elif floor > 0:
        if training_pkg is None:
            raise RuntimeError(
                f"[train] chunk_resume_step_floor={floor} requires training_state.pt "
                "when continuing chunked offline training."
            )
        resumed_step = int(training_pkg["completed_step"]) + 1
        if resumed_step < floor:
            raise RuntimeError(
                f"[train] training_state.pt would resume at global step {resumed_step}, "
                f"but chunk_resume_step_floor={floor}. Refusing to re-run earlier chunks "
                "(optimizer/checkpoint likely stale — delete checkpoints or fix training_state.pt)."
            )

    # Drop the training_state blob after copying fields; optimizer tensors stay in
    # pending_optimizer_sd on CPU until the first backward (see above).
    if training_pkg is not None:
        del training_pkg
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return TrainSession(
        config=config,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        distributed=distributed,
        device=device,
        dtype=dtype,
        model_build_dtype=model_build_dtype,
        is_main_process=is_main_process,
        model=model,
        model_for_train=model_for_train,
        optimizer=optimizer,
        optimizer_local=optimizer_local,
        encoder_params=encoder_params,
        encoder_id_set=encoder_id_set,
        encoder_needs_manual_reduce=encoder_needs_manual_reduce,
        wandb_run=wandb_run,
        wandb_run_id_for_state=wandb_run_id_for_state,
        log_file=log_file,
        ts_path=ts_path,
        global_step=step,
        best_val_loss=best_val_loss,
        pending_optimizer_sd=pending_optimizer_sd,
        pending_optimizer_local_sd=pending_optimizer_local_sd,
        n_nan_consecutive=0,
    )


def run_chunk(
    session: TrainSession,
    train_dataset: Any,
    val_dataset: Any,
) -> tuple[dict[str, Any] | None, bool]:
    """Run one chunk's step range on a persistent :class:`TrainSession`.

    Per-chunk step bounds, chunk index and corpus-skip bookkeeping come from
    ``session.config`` (the caller sets ``chunk_stop_step`` /
    ``chunk_resume_step_floor`` / ``training_chunk_index`` /
    ``corpus_skip_sequences`` before each call). Mutated state is written back to
    ``session``. Returns ``(cpu_resume_payload, graceful_shutdown)`` where
    ``cpu_resume_payload`` is a RAM-safe staging copy of the end-of-chunk
    training_state, or ``{"_shutdown_requested": True}`` when SIGTERM was caught.
    """
    config = session.config
    device = session.device
    dtype = session.dtype
    distributed = session.distributed
    rank = session.rank
    world_size = session.world_size
    is_main_process = session.is_main_process
    model = session.model
    model_for_train = session.model_for_train
    optimizer = session.optimizer
    optimizer_local = session.optimizer_local
    encoder_params = session.encoder_params
    encoder_id_set = session.encoder_id_set
    encoder_needs_manual_reduce = session.encoder_needs_manual_reduce
    model_build_dtype = session.model_build_dtype
    wandb_run = session.wandb_run
    wandb_run_id_for_state = session.wandb_run_id_for_state
    log_file = session.log_file
    ts_path = session.ts_path
    step = session.global_step
    best_val_loss = session.best_val_loss
    n_nan_consecutive = session.n_nan_consecutive
    pending_optimizer_sd = session.pending_optimizer_sd
    pending_optimizer_local_sd = session.pending_optimizer_local_sd

    # Advance the global RNG with chunk index so per-chunk operations
    # (e.g. update_grid subsampling) don't repeat the same draws every chunk.
    seed_everything(config.seed + config.training_chunk_index)

    _apply_mmap_dataloader_guard(config, train_dataset if train_dataset is not None else val_dataset)

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.seed,
            drop_last=True,
        )

    train_loader_kwargs = {
        "dataset": train_dataset,
        "batch_size": config.batch_size,
        "shuffle": train_sampler is None,
        "sampler": train_sampler,
        "drop_last": True,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "persistent_workers": config.persistent_workers and config.num_workers > 0,
        "generator": make_generator(config.seed),
    }
    if config.num_workers > 0 and config.prefetch_factor is not None:
        train_loader_kwargs["prefetch_factor"] = config.prefetch_factor
    train_loader = DataLoader(**train_loader_kwargs)

    train_iter = iter(train_loader)
    _NAN_RECOVERY_THRESHOLD = 100  # reload best ckpt after this many consecutive NaN steps

    upper_bound = config.total_steps
    if config.chunk_stop_step is not None:
        upper_bound = min(upper_bound, config.chunk_stop_step)

    # Reshuffle each time the loader is re-iterated; otherwise StopIteration
    # within a chunk would replay the same shuffle order. Combine the global
    # chunk index with an intra-chunk pass counter so the seed never repeats.
    sampler_pass = 0

    def _bump_sampler_epoch() -> None:
        nonlocal sampler_pass
        if train_sampler is not None:
            # Large multiplier prevents collisions across chunks; the chunk
            # index alone is reused by future chunks' first pass.
            train_sampler.set_epoch(config.training_chunk_index * 100_000 + sampler_pass)
            sampler_pass += 1

    _bump_sampler_epoch()

    if is_main_process and config.chunk_stop_step is not None:
        span = max(0, upper_bound - step)
        print(
            f"[train] Chunk LR segment: global steps step ∈ [{step}, {upper_bound}) "
            f"({span} optimizer steps; total_steps={config.total_steps})"
        )

    pbar = tqdm(
        total=config.total_steps,
        initial=min(step, config.total_steps),
        desc="Training",
        disable=not is_main_process,
    )
    graceful_shutdown = False
    while step < upper_bound:
        # Cooperative shutdown check at a collective-safe point (no collective in
        # flight). All ranks agree via all_reduce so the group leaves together.
        if _shutdown_requested_global(distributed, device):
            graceful_shutdown = True
            if is_main_process:
                pbar.write(
                    f"Step {step}: SIGTERM/SIGINT received — stopping after "
                    f"{step} steps; preserving last completed-chunk checkpoint."
                )
            break

        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            _bump_sampler_epoch()
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Shape: (batch, n_layers, seq_len, d_model) → (n_layers, batch*seq_len, d_model)
        x_in = batch["mlp_inputs"].to(device=device, dtype=dtype)
        y_true = batch["mlp_outputs"].to(device=device, dtype=dtype)

        # Reshape: merge batch and seq_len dimensions
        b, n_l, s, d = x_in.shape
        x_in = x_in.permute(1, 0, 2, 3).reshape(n_l, b * s, d)
        y_true = y_true.permute(1, 0, 2, 3).reshape(n_l, b * s, d)

        should_update_grid = (
            config.encoder_type == "kan"
            and config.update_grid_every > 0
            and step >= config.update_grid_from
            and step % config.update_grid_every == 0
        )
        grid_update_inputs: list[torch.Tensor] | None = None

        # Update learning rate
        lr = get_lr(step, config)
        _set_lr([optimizer, optimizer_local], lr)

        # Forward + loss
        optimizer.zero_grad()
        if optimizer_local is not None:
            optimizer_local.zero_grad()
        # Every metric is a .item() off a device tensor, and the L0 mask is a full
        # pass over (n_layers, n_pos, d_transcoder). Only pay for it on log steps.
        want_metrics = step % config.log_every == 0
        recon_per_layer = config.recon_normalization == "per_layer"
        sparsity_per_layer_mean = config.sparsity_normalization == "mean"
        if config.use_fsdp:
            # FSDP only unshards flat parameters inside module.forward() hooks,
            # so we must enter through model_for_train(x_in) rather than calling
            # encode/decode_dense directly. KAN encoder params are accessed
            # unwrapped in compute_losses() only for the explicit spline regularizer.
            activations, y_hat, dec_norms = model_for_train(x_in)
            loss, metrics = compute_losses(
                activations, y_hat, dec_norms, model, y_true,
                lambda_sparsity=config.lambda_sparsity,
                c_sparsity=config.c_sparsity,
                lambda_kan_reg=config.lambda_kan_reg,
                compute_metrics=want_metrics,
                recon_per_layer=recon_per_layer,
                sparsity_per_layer_mean=sparsity_per_layer_mean,
                recon_layer_energy_beta=config.recon_layer_energy_beta,
            )
        else:
            loss, metrics = total_loss(
                model_for_train, x_in, y_true,
                lambda_sparsity=config.lambda_sparsity,
                c_sparsity=config.c_sparsity,
                lambda_kan_reg=config.lambda_kan_reg,
                compute_metrics=want_metrics,
                recon_per_layer=recon_per_layer,
                sparsity_per_layer_mean=sparsity_per_layer_mean,
                recon_layer_energy_beta=config.recon_layer_energy_beta,
            )

        # NaN guard: skip update; after _NAN_RECOVERY_THRESHOLD consecutive NaN steps,
        # reload the best checkpoint and halve the peak learning rate.
        # This handles the case where a grid update permanently destabilizes training
        # (Adam's accumulated momentum becomes invalid after knot repositioning and
        # can drive parameters into inf territory within a few hundred steps).
        # Collective: each rank sees a different x_in shard so loss can be NaN on
        # one rank and finite on another. If any rank is non-finite, ALL must skip
        # together — otherwise the finite rank enters loss.backward() and FSDP's
        # reduce-scatter blocks waiting for the rank that took the `continue` path.
        local_finite = 1.0 if loss.isfinite().item() else 0.0
        if distributed:
            finite_t = torch.tensor([local_finite], device=device)
            dist.all_reduce(finite_t, op=dist.ReduceOp.MIN)
            is_global_nan = finite_t.item() == 0.0
        else:
            is_global_nan = local_finite == 0.0

        if is_global_nan:
            n_nan_consecutive += 1
            if n_nan_consecutive == 1:
                if is_main_process:
                    pbar.write(f"Step {step}: non-finite loss on at least one rank, skipping update")
            if n_nan_consecutive >= _NAN_RECOVERY_THRESHOLD:
                best_path = os.path.join(config.checkpoint_dir, f"{config.run_name}_best")
                if os.path.isdir(best_path):
                    if is_main_process:
                        pbar.write(
                            f"  {_NAN_RECOVERY_THRESHOLD} consecutive NaN steps — "
                            f"reloading best checkpoint and halving peak LR "
                            f"({config.learning_rate:.2e} → {config.learning_rate/2:.2e})"
                        )
                    if config.use_fsdp:
                        # load_state_dict on the unwrapped model doesn't reach FSDP
                        # flat params; summon with writeback=True so the copy lands
                        # in the sharded buffers.
                        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                        with FSDP.summon_full_params(model_for_train, writeback=True):
                            recovered = load_spline_clt(best_path, device=device, dtype=model_build_dtype)
                            model.load_state_dict(recovered.state_dict())
                            del recovered
                    else:
                        recovered = load_spline_clt(best_path, device=device, dtype=model_build_dtype)
                        model.load_state_dict(recovered.state_dict())
                        del recovered
                    config.learning_rate /= 2
                    optimizer, optimizer_local = _build_optimizers(
                        model, model_for_train, config, lr=config.learning_rate
                    )
                    pending_optimizer_sd = None
                    pending_optimizer_local_sd = None
                else:
                    if is_main_process:
                        pbar.write(
                            f"  {_NAN_RECOVERY_THRESHOLD} consecutive NaN steps but no best "
                            f"checkpoint at {best_path!r} — resetting NaN counter and continuing"
                        )
                n_nan_consecutive = 0
            step += 1
            pbar.update(1)
            continue

        n_nan_consecutive = 0

        # Diagnostic: how much is the B-spline path actually contributing vs the
        # linear/SiLU base? Computed on rank 0 only at log steps (encoders are
        # FSDP-ignored and replicated, so this needs no collective) and injected
        # into the metrics that get logged to jsonl/W&B. x_in is still alive here
        # (it is deleted after backward below).
        if config.encoder_type == "kan" and is_main_process and want_metrics:
            metrics["stats/spline_contribution_frac"] = model.spline_contribution_fraction(x_in)

        # Backward
        loss.backward()
        if should_update_grid and rank == 0:
            # Only rank 0 fits the new grid; updated grid + spline params are
            # broadcast to other ranks below. This guarantees identical encoders
            # across ranks (the previous "every rank fits its own grid then we
            # broadcast only parameters()" pattern silently left each rank with
            # a different B-spline grid buffer, since `grid` is a buffer, not a
            # parameter).
            max_n = config.update_grid_max_samples
            grid_update_inputs = []
            with torch.no_grad():
                for layer_id in range(model.n_layers):
                    x_layer = x_in[layer_id].detach()
                    if x_layer.shape[0] > max_n:
                        idx = torch.randperm(x_layer.shape[0], device=x_layer.device)[:max_n]
                        x_layer = x_layer[idx]
                    grid_update_inputs.append(x_layer.clone().contiguous())
        # Drop graph roots and wide activations before restoring Adam/adafactor tensors from
        # CPU — deferred resume loads optimizer state here and load_state_dict allocates on
        # GPU while backward leaves allocator slabs fragmented unless refs are cleared first.
        # Steady-state steps don't need the heavy gc/sync/empty_cache; only the resume step
        # and grid-update steps do. Keep the cheap `del` cleanup unconditional.
        del loss
        if config.use_fsdp:
            del activations, y_hat, dec_norms
        del x_in, y_true
        needs_heavy_cleanup = (
            pending_optimizer_sd is not None
            or pending_optimizer_local_sd is not None
            or should_update_grid
        )
        if needs_heavy_cleanup:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        # All-reduce KAN encoder gradients across ranks BEFORE loading any
        # deferred Adam state. The encoders are FSDP-ignored, so without this
        # explicit sync each rank would step its replicated copy on local data
        # only and silently diverge between grid-update broadcasts.
        if encoder_needs_manual_reduce:
            _all_reduce_encoder_grads(encoder_params, world_size)

        if pending_optimizer_sd is not None or pending_optimizer_local_sd is not None:
            _load_optimizer_from_training_state(
                model_for_train,
                optimizer,
                optimizer_local,
                pending_optimizer_sd,
                pending_optimizer_local_sd,
                use_fsdp=config.use_fsdp,
                distributed=distributed,
            )
            pending_optimizer_sd = None
            pending_optimizer_local_sd = None

        if config.grad_clip > 0:
            if config.use_fsdp:
                # Joint norm across FSDP-managed and encoder-local params, then a
                # single scaling factor applied to all grads — equivalent to a
                # global clip_grad_norm_ over the full parameter set.
                #
                # FSDP.clip_grad_norm_(inf) returns the FSDP-side norm (after the
                # internal all-reduce) without rescaling, since clip_coef=inf
                # clamps to 1.0 → multiplicative no-op. Encoder grads were just
                # all-reduced above so their norm is identical on every rank.
                fsdp_total = model_for_train.clip_grad_norm_(float("inf"))
                enc_grads = [p.grad for p in encoder_params if p.grad is not None]
                if enc_grads:
                    # Under fsdp_cpu_offload, fsdp_total can land on CPU while
                    # encoder grads (FSDP-ignored) live on GPU. Reduce the norm
                    # on the FSDP norm's device, but cast `coef` per-grad below.
                    enc_norm_sq = torch.stack(
                        [g.detach().pow(2).sum().to(fsdp_total.device) for g in enc_grads]
                    ).sum()
                    combined = (fsdp_total.pow(2) + enc_norm_sq).sqrt()
                else:
                    combined = fsdp_total
                coef = (config.grad_clip / (combined + 1e-6)).clamp(max=1.0)
                for g in enc_grads:
                    g.mul_(coef.to(device=g.device, dtype=g.dtype))
                for p in model_for_train.parameters():
                    if id(p) in encoder_id_set or p.grad is None:
                        continue
                    p.grad.mul_(coef.to(device=p.grad.device, dtype=p.grad.dtype))
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        if optimizer_local is not None:
            optimizer_local.step()

        # Logging
        if want_metrics:
            metrics["lr"] = lr
            rel_fro = metrics.get("reconstruction/rel_fro_error", float("nan"))
            l0_tok = metrics.get("stats/l0_active_features_per_token", float("nan"))
            if is_main_process:
                # loss == recon + sparse + kan_reg. Print all three addends, or the
                # displayed terms won't reconcile with the total.
                pbar.set_postfix({
                    "loss": f"{metrics['loss/total']:.4f}",
                    "recon": f"{metrics['loss/reconstruction']:.4f}",
                    "sparse": f"{metrics['loss/sparsity']:.4f}",
                    "kan_reg": f"{metrics['loss/kan_regularization']:.4f}",
                    "rel_fro": f"{rel_fro:.3f}",
                    "l0_tok": f"{l0_tok:.1f}",
                    "act_lp": f"{metrics['stats/active_features_per_pos']:.1f}",
                    "lr": f"{lr:.2e}",
                })
            record = {"record_type": "train_step", "step": step, **{k: float(v) for k, v in metrics.items()}}
            if log_file is not None:
                log_file.write(json.dumps(record) + "\n")
            if wandb_run is not None:
                wandb_run.log({k: v for k, v in metrics.items()}, step=step)

        # Grid update (adapt B-spline knots to data distribution) — KAN only
        # Subsample to avoid OOM: efficient-kan's curve2coeff builds (batch, d_model, d_transcoder) intermediates
        if should_update_grid:
            # Save a pre-update snapshot so NaN recovery always has a clean state.
            pre_update_path = os.path.join(
                config.checkpoint_dir, f"{config.run_name}_pre_grid{step}"
            )
            _save_model_checkpoint(
                model=model_for_train,
                base_model=model,
                save_path=pre_update_path,
                use_fsdp=config.use_fsdp,
                is_main_process=is_main_process,
            )

            with torch.no_grad(), _tf32_disabled():
                if rank == 0:
                    assert grid_update_inputs is not None
                    for layer_id, x_layer in enumerate(grid_update_inputs):
                        model.encoders[layer_id].update_grid(x_layer)
                if distributed:
                    # KAN encoders are FSDP-ignored so each rank holds its own
                    # unsharded copy. Broadcast both parameters AND buffers from
                    # rank 0 — `grid` is registered as a buffer (efficient_kan)
                    # and update_grid mutates it in place. Broadcasting only
                    # parameters() leaves every other rank with rank-0's spline
                    # weights but its own (stale, locally-fit) grid, which then
                    # silently corrupts forward outputs across ranks.
                    for layer_id in range(model.n_layers):
                        for param in model.encoders[layer_id].parameters():
                            dist.broadcast(param.data, src=0)
                        for buf in model.encoders[layer_id].buffers():
                            dist.broadcast(buf.data, src=0)
            if rank == 0:
                del grid_update_inputs

            # Reset Adam state after grid update: the accumulated first/second moments
            # correspond to old spline knot positions and will drive the new parameters
            # in the wrong direction, causing the inf spiral seen in practice. With the
            # split optimizer we reset only the encoder optimizer, preserving decoder
            # Adam state across grid updates.
            optimizer_local = create_optimizer(
                model.encoders.parameters(), config, lr=lr
            )
            pending_optimizer_local_sd = None
            if is_main_process:
                pbar.write(
                    f"Step {step}: grid updated and encoder optimizer state reset "
                    f"(pre-update checkpoint saved to {pre_update_path})"
                )

        # Optimizer restart for linear encoder — mirrors KAN's grid-update resets
        # so both models receive the same number of fresh Adam starts during training.
        if (
            config.encoder_type != "kan"
            and config.reset_optimizer_every > 0
            and step > 0
            and step % config.reset_optimizer_every == 0
        ):
            optimizer = create_optimizer(model_for_train.parameters(), config, lr=lr)
            pending_optimizer_sd = None
            if is_main_process:
                pbar.write(f"Step {step}: optimizer state reset (reset_optimizer_every={config.reset_optimizer_every})")

        # Evaluation
        if step > 0 and step % config.eval_every == 0:
            val_loss = float("nan")
            if config.use_fsdp:
                # summon_full_params must be entered by ALL ranks (even with
                # rank0_only=True it issues collective calls). Run eval on rank 0
                # inside the context so it sees the full parameters, then broadcast
                # val_loss so all ranks can agree on is_best without a barrier mismatch.
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                with FSDP.summon_full_params(model_for_train, writeback=False, rank0_only=True):
                    if is_main_process:
                        val_loss = evaluate(model, val_dataset, config, device, dtype)
                        pbar.write(f"Step {step}: val_loss={val_loss:.4f}")
                val_loss_t = torch.tensor(val_loss, device=device)
                dist.broadcast(val_loss_t, src=0)
                val_loss = val_loss_t.item()
            else:
                if is_main_process:
                    val_loss = evaluate(model, val_dataset, config, device, dtype)
                    pbar.write(f"Step {step}: val_loss={val_loss:.4f}")

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                _save_model_checkpoint(
                    model=model_for_train,
                    base_model=model,
                    save_path=os.path.join(config.checkpoint_dir, f"{config.run_name}_best"),
                    use_fsdp=config.use_fsdp,
                    is_main_process=is_main_process,
                )
                if is_main_process:
                    pbar.write(f"  New best model saved (val_loss={val_loss:.4f})")

            if log_file is not None:
                log_file.write(json.dumps({
                    "record_type": "val_eval",
                    "step": step,
                    "val_loss": float(val_loss),
                    "best_val_loss": float(best_val_loss),
                    "is_best": bool(is_best),
                }) + "\n")
            if wandb_run is not None:
                wandb_run.log({"val/loss": val_loss, "val/best_loss": best_val_loss}, step=step)

        # Checkpoint
        if step > 0 and step % config.save_every == 0:
            ckpt_dir = os.path.join(config.checkpoint_dir, f"{config.run_name}_step{step}")
            _save_model_checkpoint(
                model=model_for_train,
                base_model=model,
                save_path=ckpt_dir,
                use_fsdp=config.use_fsdp,
                is_main_process=is_main_process,
            )
            _save_training_state(
                path=ts_path,
                model_for_train=model_for_train,
                optimizer_main=optimizer,
                optimizer_local=optimizer_local,
                completed_step=step,
                best_val_loss=best_val_loss,
                model_checkpoint_dir=ckpt_dir,
                corpus_skip_sequences=config.corpus_skip_sequences,
                corpus_skip_items=config.corpus_skip_items,
                chunk_index=config.training_chunk_index,
                wandb_run_id=wandb_run_id_for_state,
                use_fsdp=config.use_fsdp,
                is_main_process=is_main_process,
                distributed=distributed,
            )
            # New checkpoint + training_state are safely on disk — drop stale
            # step checkpoints (rank 0 only, no collectives involved).
            _prune_step_checkpoints(config, is_main_process)

        step += 1
        pbar.update(1)

    pbar.close()

    # Write mutated state back to the session for the next chunk's run_chunk call.
    session.global_step = step
    session.best_val_loss = best_val_loss
    session.optimizer = optimizer
    session.optimizer_local = optimizer_local
    session.n_nan_consecutive = n_nan_consecutive
    session.pending_optimizer_sd = pending_optimizer_sd
    session.pending_optimizer_local_sd = pending_optimizer_local_sd

    if graceful_shutdown:
        # SIGTERM mid-chunk: do NOT overwrite the last completed-chunk checkpoint
        # with a partial-chunk step/optimizer (that would desync the runner's
        # per-chunk corpus_skip_sequences bookkeeping). The barrier keeps both
        # ranks together before close_session.
        if distributed and dist.is_initialized():
            dist.barrier()
        return {"_shutdown_requested": True}, True

    # Save end-of-chunk model + training_state (the cross-job resume point).
    final_dir = os.path.join(config.checkpoint_dir, f"{config.run_name}_final")
    _save_model_checkpoint(
        model=model_for_train,
        base_model=model,
        save_path=final_dir,
        use_fsdp=config.use_fsdp,
        is_main_process=is_main_process,
    )
    completed_last = step - 1
    final_payload = _save_training_state(
        path=ts_path,
        model_for_train=model_for_train,
        optimizer_main=optimizer,
        optimizer_local=optimizer_local,
        completed_step=completed_last,
        best_val_loss=best_val_loss,
        model_checkpoint_dir=final_dir,
        corpus_skip_sequences=config.corpus_skip_sequences,
        corpus_skip_items=config.corpus_skip_items,
        chunk_index=config.training_chunk_index,
        wandb_run_id=wandb_run_id_for_state,
        use_fsdp=config.use_fsdp,
        is_main_process=is_main_process,
        distributed=distributed,
    )
    # training_state now points at _final; mid-chunk step checkpoints are stale.
    _prune_step_checkpoints(config, is_main_process)
    if distributed and dist.is_initialized():
        # Barrier so all ranks finish the chunk together before the caller
        # collects the next chunk / closes the session.
        dist.barrier()
    return _staging_copy_training_payload(final_payload), False


def close_session(session: TrainSession) -> None:
    """Tear down a persistent session: close the jsonl log, finish the wandb run.

    Do NOT destroy the process group: re-initialising on the same port while the
    OS holds it in TIME_WAIT hangs the next launch. The PG is cleaned up when the
    torchrun-launched processes exit.
    """
    if session.log_file is not None:
        session.log_file.close()
    if session.wandb_run is not None:
        session.wandb_run.finish()
    if session.distributed and dist.is_initialized():
        dist.barrier()


def train(
    config: TrainConfig,
    dataset: ActivationDataset | None = None,
    val_dataset: ActivationDataset | None = None,
    *,
    resume_payload: dict[str, Any] | None = None,
    return_resume_payload: bool = False,
) -> KANCrossLayerTranscoder | tuple[KANCrossLayerTranscoder, dict[str, Any]]:
    """Train a Spline-CLT model in a single session (standalone / non-chunked).

    Thin wrapper: ``build_session`` → one ``run_chunk`` over the full step range
    → ``close_session``. The chunked paper runner instead builds the session once
    and calls :func:`run_chunk` per chunk, so the model/optimizers are not rebuilt
    every chunk. The on-disk checkpoint format is identical either way.

    Loading priority for the train/val datasets:
      1. Both ``dataset`` and ``val_dataset`` provided → use as-is, no split.
      2. Only ``dataset`` provided → runtime ``random_split`` (legacy).
      3. Neither provided and ``config.val_data_dir`` set → load split files
         (``mlp_inputs_train.npy``/``mlp_inputs_val.npy``) from the two dirs.
      4. Otherwise → load legacy single-file layout from ``config.data_dir``
         and ``random_split`` it.

    Returns:
        Trained ``KANCrossLayerTranscoder``, or ``(model, cpu_resume_payload)``
        when ``return_resume_payload`` is True.
    """
    # Offline path: load from pre-collected mmap / in-RAM activation files.
    if dataset is None and val_dataset is None:
        if config.val_data_dir:
            dataset = ActivationDataset.load(config.data_dir, split="train")
            val_dataset = ActivationDataset.load(config.val_data_dir, split="val")
        else:
            dataset = ActivationDataset.load(config.data_dir)

    # mmap guard on the FULL dataset (a post-split Subset lacks ``mlp_inputs``).
    _apply_mmap_dataloader_guard(config, dataset if dataset is not None else val_dataset)

    # Train/val split: pre-split when both datasets are available, otherwise
    # fall back to a deterministic random_split over the single dataset.
    if val_dataset is not None and dataset is not None:
        train_dataset = dataset
    else:
        assert dataset is not None
        train_dataset, val_dataset = split_dataset(
            dataset=dataset,
            val_fraction=config.val_fraction,
            seed=config.seed,
        )

    session = build_session(config, resume_payload=resume_payload, norm_dataset=train_dataset)
    payload, _graceful = run_chunk(session, train_dataset, val_dataset)
    close_session(session)

    if return_resume_payload:
        # payload is the staging copy on normal completion, or the shutdown
        # sentinel on a graceful stop.
        return session.model, payload if payload is not None else {}
    return session.model


def split_dataset(
    dataset: ActivationDataset,
    val_fraction: float,
    seed: int,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Create a deterministic train/validation split."""
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    return torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=make_generator(seed),
    )


@torch.no_grad()
def evaluate(
    model: KANCrossLayerTranscoder,
    val_dataset: torch.utils.data.Dataset,
    config: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """Evaluate model on validation set.

    Returns:
        Average reconstruction loss on validation data.
    """
    from spline_clt.training.loss import reconstruction_loss

    model.eval()
    val_loader_kwargs = {
        "dataset": val_dataset,
        "batch_size": config.batch_size,
        "shuffle": False,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "persistent_workers": config.persistent_workers and config.num_workers > 0,
    }
    if config.num_workers > 0 and config.prefetch_factor is not None:
        val_loader_kwargs["prefetch_factor"] = config.prefetch_factor
    val_loader = DataLoader(**val_loader_kwargs)

    total_loss_val = 0.0
    n_batches = 0

    for batch in val_loader:
        x_in = batch["mlp_inputs"].to(device=device, dtype=dtype)
        y_true = batch["mlp_outputs"].to(device=device, dtype=dtype)

        b, n_l, s, d = x_in.shape
        x_in = x_in.permute(1, 0, 2, 3).reshape(n_l, b * s, d)
        y_true = y_true.permute(1, 0, 2, 3).reshape(n_l, b * s, d)

        activations = model.encode(x_in)
        y_hat = model.decode_dense(activations, input_acts=x_in)
        # Must match the training objective, or best-checkpoint selection optimises
        # a different loss than the one being minimised.
        loss = reconstruction_loss(
            y_hat, y_true, per_layer=config.recon_normalization == "per_layer"
        )
        total_loss_val += loss.item()
        n_batches += 1

    model.train()
    return total_loss_val / max(1, n_batches)


def load_config(path: str) -> TrainConfig:
    """Load training config from YAML file."""
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)
    return TrainConfig(**data)
