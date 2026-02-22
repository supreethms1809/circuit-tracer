"""Training loop for KAN-CLT.

Standard PyTorch training with Adam optimizer, warmup + cosine decay,
logging, and checkpointing.
"""

import math
import os
import time
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from kan_clt.kan_transcoder import KANCrossLayerTranscoder
from kan_clt.training.data import ActivationDataset
from kan_clt.training.loss import total_loss


@dataclass
class TrainConfig:
    """Training configuration."""

    # Model architecture
    n_layers: int = 12
    d_model: int = 768
    d_transcoder: int = 4096
    grid_size: int = 5
    spline_order: int = 3

    # Training
    learning_rate: float = 1e-4
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
    run_name: str = "kan_clt"

    # Grid update
    update_grid_every: int = 10_000
    update_grid_from: int = 2000  # don't update grid in early training

    # Data
    data_dir: str = "data/activations"
    device: str = "cuda"
    dtype: str = "float32"

    # Validation
    val_fraction: float = 0.05

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


def train(
    config: TrainConfig,
    dataset: ActivationDataset | None = None,
) -> KANCrossLayerTranscoder:
    """Train a KAN-CLT model.

    Args:
        config: Training configuration.
        dataset: Pre-loaded activation dataset. If None, loads from config.data_dir.

    Returns:
        Trained KANCrossLayerTranscoder model.
    """
    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu"
    )
    dtype = config.get_dtype()

    # Load data
    if dataset is None:
        dataset = ActivationDataset.load(config.data_dir)

    # Train/val split
    n_val = max(1, int(len(dataset) * config.val_fraction))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    # Create model
    model = KANCrossLayerTranscoder(
        n_layers=config.n_layers,
        d_model=config.d_model,
        d_transcoder=config.d_transcoder,
        grid_size=config.grid_size,
        spline_order=config.spline_order,
        activation_function="jump_relu",
        device=device,
        dtype=dtype,
    )

    # Optimizer (NOT LBFGS — doesn't scale)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    step = 0
    best_val_loss = float("inf")
    train_iter = iter(train_loader)

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

        # NaN guard: skip update if loss is non-finite (can happen after grid update)
        if not loss.isfinite():
            pbar.write(f"Step {step}: non-finite loss ({loss.item():.4f}), skipping update")
            step += 1
            pbar.update(1)
            continue

        # Backward
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        # Logging
        if step % config.log_every == 0:
            metrics["lr"] = lr
            pbar.set_postfix({
                "loss": f"{metrics['loss/total']:.4f}",
                "recon": f"{metrics['loss/reconstruction']:.4f}",
                "sparse": f"{metrics['loss/sparsity']:.4f}",
                "active": f"{metrics['stats/active_features_per_pos']:.1f}",
                "lr": f"{lr:.2e}",
            })

        # Grid update (adapt B-spline knots to data distribution)
        if (
            config.update_grid_every > 0
            and step >= config.update_grid_from
            and step % config.update_grid_every == 0
        ):
            with torch.no_grad():
                for layer_id in range(model.n_layers):
                    model.encoders[layer_id].update_grid(x_in[layer_id])

        # Evaluation
        if step > 0 and step % config.eval_every == 0:
            val_loss = evaluate(model, val_dataset, config, device, dtype)
            pbar.write(f"Step {step}: val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.to_safetensors(
                    os.path.join(config.checkpoint_dir, f"{config.run_name}_best")
                )
                pbar.write(f"  New best model saved (val_loss={val_loss:.4f})")

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

    return model


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
    from kan_clt.training.loss import reconstruction_loss

    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

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
