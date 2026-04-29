"""Training loop for Spline-CLT.

Standard PyTorch training with Adam optimizer, warmup + cosine decay,
logging, and checkpointing.
"""

import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Adafactor

from spline_clt.kan_transcoder import KANCrossLayerTranscoder
from spline_clt.seed import make_generator, seed_everything
from spline_clt.training.data import ActivationDataset
from spline_clt.training.loss import total_loss


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
    grad_clip: float = 1.0

    # Logging & checkpointing
    log_every: int = 100
    eval_every: int = 5000
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints"
    run_name: str = "spline_clt"
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

    def get_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "bfloat16": torch.bfloat16}[self.dtype]


def get_lr(step: int, config: TrainConfig) -> float:
    """Compute learning rate with linear warmup + cosine decay."""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / max(
        1, config.total_steps - config.warmup_steps
    )
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def create_optimizer(
    model: KANCrossLayerTranscoder,
    config: TrainConfig,
    lr: float,
) -> torch.optim.Optimizer:
    """Build optimizer according to config.

    Adafactor is supported for very large models where Adam-family optimizer
    states exceed device memory.
    """
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2),
        )
    if config.optimizer == "adafactor":
        return Adafactor(
            model.parameters(),
            lr=lr,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def train(
    config: TrainConfig,
    dataset: ActivationDataset | None = None,
    val_dataset: ActivationDataset | None = None,
) -> KANCrossLayerTranscoder:
    """Train a Spline-CLT model.

    Loading priority for the train/val datasets:
      1. Both ``dataset`` and ``val_dataset`` provided → use as-is, no split.
      2. Only ``dataset`` provided → runtime ``random_split`` (legacy).
      3. Neither provided and ``config.val_data_dir`` set → load split files
         (``mlp_inputs_train.npy``/``mlp_inputs_val.npy``) from the two dirs.
      4. Otherwise → load legacy single-file layout from ``config.data_dir``
         and ``random_split`` it.

    Args:
        config: Training configuration.
        dataset: Pre-loaded train (or full) activation dataset.
        val_dataset: Pre-loaded val activation dataset. If both ``dataset``
            and ``val_dataset`` are provided, no runtime split is performed.

    Returns:
        Trained KANCrossLayerTranscoder model.
    """
    if config.device:
        device = torch.device(config.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dtype = config.get_dtype()
    seed_everything(config.seed)

    # Load data — split-aware path takes priority.
    if dataset is None and val_dataset is None:
        if config.val_data_dir:
            dataset = ActivationDataset.load(config.data_dir, split="train")
            val_dataset = ActivationDataset.load(config.val_data_dir, split="val")
        else:
            dataset = ActivationDataset.load(config.data_dir)

    # Safety guard for mmap-backed activation streaming datasets:
    # multiprocessing + prefetch + pin_memory can spike host RAM (/dev/shm) and
    # get the process OOM-killed before step 0, especially with large activation
    # samples. Force conservative DataLoader settings in this mode.
    probe = dataset if dataset is not None else val_dataset
    assert probe is not None
    if isinstance(probe.mlp_inputs, np.ndarray):
        if config.num_workers > 0 or config.pin_memory or config.prefetch_factor is not None:
            print(
                "[train] mmap streaming dataset detected; forcing "
                "num_workers=0, pin_memory=False, prefetch_factor=None, "
                "persistent_workers=False to avoid host OOM."
            )
        config.num_workers = 0
        config.pin_memory = False
        config.prefetch_factor = None
        config.persistent_workers = False

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

    train_loader_kwargs = {
        "dataset": train_dataset,
        "batch_size": config.batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "persistent_workers": config.persistent_workers and config.num_workers > 0,
        "generator": make_generator(config.seed),
    }
    if config.num_workers > 0 and config.prefetch_factor is not None:
        train_loader_kwargs["prefetch_factor"] = config.prefetch_factor
    train_loader = DataLoader(**train_loader_kwargs)

    # Create model
    model = KANCrossLayerTranscoder(
        n_layers=config.n_layers,
        d_model=config.d_model,
        d_transcoder=config.d_transcoder,
        encoder_type=config.encoder_type,
        grid_size=config.grid_size,
        spline_order=config.spline_order,
        activation_function="jump_relu",
        device=device,
        dtype=dtype,
    )

    # Optimizer (NOT LBFGS — doesn't scale)
    optimizer = create_optimizer(model, config, lr=config.learning_rate)


    # Training loop
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    log_dir = config.log_dir or os.path.dirname(os.path.abspath(config.checkpoint_dir))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training_records.jsonl")
    log_file = open(log_path, "a", buffering=1)

    wandb_run = None
    if config.wandb_project and config.wandb_mode != "disabled":
        try:
            import wandb  # type: ignore

            wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name or config.run_name,
                mode=config.wandb_mode,  # type: ignore[arg-type]
                dir=log_dir,
                config={k: getattr(config, k) for k in config.__dataclass_fields__},
                reinit=True,
            )
        except Exception as exc:  # pragma: no cover - wandb is optional
            print(f"[train] wandb init failed ({exc!r}); continuing with jsonl only")
            wandb_run = None

    step = 0
    best_val_loss = float("inf")
    train_iter = iter(train_loader)
    n_nan_consecutive = 0          # consecutive non-finite loss steps
    _NAN_RECOVERY_THRESHOLD = 100  # reload best ckpt after this many consecutive NaN steps

    pbar = tqdm(total=config.total_steps, desc="Training")
    while step < config.total_steps:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Shape: (batch, n_layers, seq_len, d_model) → (n_layers, batch*seq_len, d_model)
        x_in = batch["mlp_inputs"].to(device=device, dtype=dtype)
        y_true = batch["mlp_outputs"].to(device=device, dtype=dtype)

        # Reshape: merge batch and seq_len dimensions
        b, n_l, s, d = x_in.shape
        x_in = x_in.permute(1, 0, 2, 3).reshape(n_l, b * s, d)
        y_true = y_true.permute(1, 0, 2, 3).reshape(n_l, b * s, d)

        # Update learning rate
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward + loss
        optimizer.zero_grad()
        loss, metrics = total_loss(
            model, x_in, y_true,
            lambda_sparsity=config.lambda_sparsity,
            c_sparsity=config.c_sparsity,
        )

        # NaN guard: skip update; after _NAN_RECOVERY_THRESHOLD consecutive NaN steps,
        # reload the best checkpoint and halve the peak learning rate.
        # This handles the case where a grid update permanently destabilizes training
        # (Adam's accumulated momentum becomes invalid after knot repositioning and
        # can drive parameters into inf territory within a few hundred steps).
        if not loss.isfinite():
            n_nan_consecutive += 1
            if n_nan_consecutive == 1:
                pbar.write(f"Step {step}: non-finite loss ({loss.item():.4f}), skipping update")
            if n_nan_consecutive >= _NAN_RECOVERY_THRESHOLD:
                best_path = os.path.join(config.checkpoint_dir, f"{config.run_name}_best")
                if os.path.isdir(best_path):
                    pbar.write(
                        f"  {_NAN_RECOVERY_THRESHOLD} consecutive NaN steps — "
                        f"reloading best checkpoint and halving peak LR "
                        f"({config.learning_rate:.2e} → {config.learning_rate/2:.2e})"
                    )
                    from spline_clt.kan_transcoder import load_spline_clt as _load_for_recovery
                    recovered = _load_for_recovery(best_path, device=device, dtype=dtype)
                    model.load_state_dict(recovered.state_dict())
                    del recovered
                    config.learning_rate /= 2
                    optimizer = create_optimizer(model, config, lr=config.learning_rate)
                else:
                    pbar.write(
                        f"  {_NAN_RECOVERY_THRESHOLD} consecutive NaN steps but no best "
                        f"checkpoint at {best_path!r} — resetting NaN counter and continuing"
                    )
                n_nan_consecutive = 0
            step += 1
            pbar.update(1)
            continue

        n_nan_consecutive = 0

        # Backward
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        # Logging
        if step % config.log_every == 0:
            metrics["lr"] = lr
            rel_fro = metrics.get("reconstruction/rel_fro_error", float("nan"))
            l0_tok = metrics.get("stats/l0_active_features_per_token", float("nan"))
            pbar.set_postfix({
                "loss": f"{metrics['loss/total']:.4f}",
                "recon": f"{metrics['loss/reconstruction']:.4f}",
                "sparse": f"{metrics['loss/sparsity']:.4f}",
                "rel_fro": f"{rel_fro:.3f}",
                "l0_tok": f"{l0_tok:.1f}",
                "act_lp": f"{metrics['stats/active_features_per_pos']:.1f}",
                "lr": f"{lr:.2e}",
            })
            record = {"record_type": "train_step", "step": step, **{k: float(v) for k, v in metrics.items()}}
            log_file.write(json.dumps(record) + "\n")
            if wandb_run is not None:
                wandb_run.log({k: v for k, v in metrics.items()}, step=step)

        # Grid update (adapt B-spline knots to data distribution) — KAN only
        # Subsample to avoid OOM: efficient-kan's curve2coeff builds (batch, d_model, d_transcoder) intermediates
        if (
            config.encoder_type == "kan"
            and config.update_grid_every > 0
            and step >= config.update_grid_from
            and step % config.update_grid_every == 0
        ):
            # Save a pre-update snapshot so NaN recovery always has a clean state.
            pre_update_path = os.path.join(
                config.checkpoint_dir, f"{config.run_name}_pre_grid{step}"
            )
            model.to_safetensors(pre_update_path)

            with torch.no_grad():
                max_n = config.update_grid_max_samples
                for layer_id in range(model.n_layers):
                    x_layer = x_in[layer_id]
                    if x_layer.shape[0] > max_n:
                        idx = torch.randperm(x_layer.shape[0], device=x_layer.device)[:max_n]
                        x_layer = x_layer[idx]
                    model.encoders[layer_id].update_grid(x_layer)

            # Reset Adam state after grid update: the accumulated first/second moments
            # correspond to old spline knot positions and will drive the new parameters
            # in the wrong direction, causing the inf spiral seen in practice.
            optimizer = create_optimizer(model, config, lr=lr)
            pbar.write(
                f"Step {step}: grid updated and optimizer state reset "
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
            optimizer = create_optimizer(model, config, lr=lr)
            pbar.write(f"Step {step}: optimizer state reset (reset_optimizer_every={config.reset_optimizer_every})")

        # Evaluation
        if step > 0 and step % config.eval_every == 0:
            val_loss = evaluate(model, val_dataset, config, device, dtype)
            pbar.write(f"Step {step}: val_loss={val_loss:.4f}")

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                model.to_safetensors(
                    os.path.join(config.checkpoint_dir, f"{config.run_name}_best")
                )
                pbar.write(f"  New best model saved (val_loss={val_loss:.4f})")

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
            model.to_safetensors(
                os.path.join(config.checkpoint_dir, f"{config.run_name}_step{step}")
            )

        step += 1
        pbar.update(1)

    pbar.close()

    # Save final model
    model.to_safetensors(
        os.path.join(config.checkpoint_dir, f"{config.run_name}_final")
    )

    log_file.close()
    if wandb_run is not None:
        wandb_run.finish()

    return model


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
        loss = reconstruction_loss(y_hat, y_true)
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
