"""Shared harness for from-scratch spline vs linear probes.

Forces identical JumpReLU θ and decoder weights across arms so init-scale
asymmetries cannot explain the gap. Custom train loop (not paper-eval).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.probes import PROBES_ROOT
from spline_clt.kan_transcoder import KANCrossLayerTranscoder
from spline_clt.seed import seed_everything
from spline_clt.training.data import ActivationDataset, compute_input_normalization
from spline_clt.training.loss import compute_decoder_norms, compute_losses
from spline_clt.training.train import (
    calibrate_decoder_scale_from_data,
    initialize_thresholds_from_data,
)

ArmKind = Literal["linear", "silu_base", "kan", "spline_only", "spline_only_nogrid"]


@dataclass
class ProbeTrainConfig:
    """Short locked-scale probe training config."""

    n_layers: int = 12
    d_model: int = 768
    d_transcoder: int = 2048
    grid_size: int = 5
    spline_order: int = 3
    learning_rate: float = 5e-5
    total_steps: int = 2000
    warmup_steps: int = 200
    batch_size: int = 8
    lambda_sparsity: float = 0.005
    c_sparsity: float = 1.0
    #: Linear ramp 0 → lambda_sparsity over this many steps; 0 = full λ from step 0.
    sparsity_warmup_steps: int = 0
    #: Cosine decay of λ starts at this step; 0 disables.
    sparsity_decay_start: int = 0
    #: Floor for late λ decay when sparsity_decay_start > 0.
    lambda_sparsity_final: float = 0.0
    lambda_kan_reg: float = 0.0
    grad_clip: float = 1.0
    log_every: int = 50
    seed: int = 101
    device: str = "cuda"
    dtype: str = "float32"
    max_samples: int = 512
    val_samples: int = 64
    calibration_samples: int = 16
    target_l0: float = 32.0
    jumprelu_bandwidth: float = 0.1  # absolute fallback; see bandwidth_frac
    bandwidth_frac: float = 0.1  # STE half-width as fraction of mean θ
    #: If set, overrides ``bandwidth_frac`` after θ calibration (paper-style absolute bw).
    absolute_jumprelu_bandwidth: float | None = None
    activation_function: str = "jump_relu"  # or "relu"
    theta_delay_frac: float = 0.0  # hold θ at floor for this fraction of steps
    theta_delay_floor: float = 1e-4
    use_threshold_adam_group: bool = False  # True in Probe G (wd=0, tiny eps)
    lock_threshold: bool = True
    lock_decoder: bool = True
    freeze_encoder: bool = False
    freeze_decoder: bool = False
    # B-spline knot adaptation (critical for spline-only; optional otherwise).
    update_grid_every: int = 0
    update_grid_from: int = 200
    update_grid_max_pos: int = 256
    data_dir: str = (
        "/gscratch/ssuresh/shared/activations/"
        "paper_r3/gpt2_small_linear_feature_match/val/gpt2"
    )
    recon_normalization: str = "per_layer"
    sparsity_normalization: str = "mean"
    # Tier-1 routing knobs (KAN only). Defaults preserve efficient-kan behaviour.
    scale_base: float = 1.0
    scale_spline: float = 1.0
    #: Multiply spline-path param LR relative to ``learning_rate`` (base stays 1×).
    lr_spline_mult: float = 1.0


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_probe_dataset(cfg: ProbeTrainConfig) -> tuple[ActivationDataset, ActivationDataset]:
    """Load a small RAM slice from an existing val activation cache."""
    full = ActivationDataset.load(cfg.data_dir, max_samples=cfg.max_samples + cfg.val_samples, split="val")
    n = len(full)
    n_train = min(cfg.max_samples, max(1, n - min(cfg.val_samples, n // 5)))
    n_val = min(cfg.val_samples, n - n_train)
    train = ActivationDataset(full.mlp_inputs[:n_train], full.mlp_outputs[:n_train])
    val = ActivationDataset(
        full.mlp_inputs[n_train : n_train + n_val],
        full.mlp_outputs[n_train : n_train + n_val],
    )
    return train, val


def build_model(arm: ArmKind, cfg: ProbeTrainConfig, device: torch.device) -> KANCrossLayerTranscoder:
    encoder_type = "linear" if arm == "linear" else "kan"
    model = KANCrossLayerTranscoder(
        n_layers=cfg.n_layers,
        d_model=cfg.d_model,
        d_transcoder=cfg.d_transcoder,
        encoder_type=encoder_type,
        grid_size=cfg.grid_size,
        spline_order=cfg.spline_order,
        activation_function=cfg.activation_function,
        threshold_init=0.01,
        jumprelu_bandwidth=cfg.jumprelu_bandwidth,
        device=device,
        dtype=_dtype(cfg.dtype),
        scale_base=cfg.scale_base,
        scale_spline=cfg.scale_spline,
    )
    if arm == "silu_base":
        zero_and_freeze_spline_path(model)
    elif arm in ("spline_only", "spline_only_nogrid"):
        zero_and_freeze_base_path(model)
    return model


def _kan_param_buckets(
    model: KANCrossLayerTranscoder,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split trainable params into (base_path, spline_path, other)."""
    base_params: list[torch.nn.Parameter] = []
    spline_params: list[torch.nn.Parameter] = []
    other_params: list[torch.nn.Parameter] = []
    if model.encoder_type != "kan":
        return [], [], [p for p in model.parameters() if p.requires_grad]

    spline_ids: set[int] = set()
    base_ids: set[int] = set()
    for enc in model.encoders:
        kl = enc.kan_linear
        if kl.base_weight.requires_grad:
            base_params.append(kl.base_weight)
            base_ids.add(id(kl.base_weight))
        if kl.spline_weight.requires_grad:
            spline_params.append(kl.spline_weight)
            spline_ids.add(id(kl.spline_weight))
        if getattr(kl, "enable_standalone_scale_spline", False) and kl.spline_scaler.requires_grad:
            spline_params.append(kl.spline_scaler)
            spline_ids.add(id(kl.spline_scaler))

    for p in model.parameters():
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in base_ids or pid in spline_ids:
            continue
        other_params.append(p)
    return base_params, spline_params, other_params


def build_optimizer(model: KANCrossLayerTranscoder, cfg: ProbeTrainConfig) -> torch.optim.AdamW:
    """AdamW with optional higher LR on the B-spline path (``lr_spline_mult``)."""
    from circuit_tracer.transcoder.activation_functions import JumpReLU

    base_params, spline_params, other_params = _kan_param_buckets(model)
    use_split = (
        model.encoder_type == "kan"
        and abs(cfg.lr_spline_mult - 1.0) > 1e-12
        and len(spline_params) > 0
    )

    threshold_param = None
    if (
        isinstance(model.activation_function, JumpReLU)
        and model.activation_function.threshold.requires_grad
    ):
        threshold_param = model.activation_function.threshold

    if use_split:
        groups: list[dict[str, Any]] = []
        if base_params:
            groups.append(
                {
                    "params": base_params,
                    "lr": cfg.learning_rate,
                    "lr_mult": 1.0,
                    "weight_decay": 0.01,
                }
            )
        groups.append(
            {
                "params": spline_params,
                "lr": cfg.learning_rate * cfg.lr_spline_mult,
                "lr_mult": float(cfg.lr_spline_mult),
                "weight_decay": 0.01,
            }
        )
        if threshold_param is not None and cfg.use_threshold_adam_group:
            thr_id = id(threshold_param)
            other_params = [p for p in other_params if id(p) != thr_id]
            if other_params:
                groups.append(
                    {
                        "params": other_params,
                        "lr": cfg.learning_rate,
                        "lr_mult": 1.0,
                        "weight_decay": 0.01,
                    }
                )
            groups.append(
                {
                    "params": [threshold_param],
                    "lr": cfg.learning_rate,
                    "lr_mult": 1.0,
                    "weight_decay": 0.0,
                    "eps": 1e-15,
                }
            )
        elif other_params:
            groups.append(
                {
                    "params": other_params,
                    "lr": cfg.learning_rate,
                    "lr_mult": 1.0,
                    "weight_decay": 0.01,
                }
            )
        return torch.optim.AdamW(groups, betas=(0.9, 0.95))

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters for optimizer")
    if cfg.use_threshold_adam_group and threshold_param is not None:
        thr_id = id(threshold_param)
        other = [p for p in params if id(p) != thr_id]
        return torch.optim.AdamW(
            [
                {
                    "params": other,
                    "lr": cfg.learning_rate,
                    "lr_mult": 1.0,
                    "weight_decay": 0.01,
                },
                {
                    "params": [threshold_param],
                    "lr": cfg.learning_rate,
                    "lr_mult": 1.0,
                    "weight_decay": 0.0,
                    "eps": 1e-15,
                },
            ],
            betas=(0.9, 0.95),
        )
    return torch.optim.AdamW(
        [{"params": params, "lr": cfg.learning_rate, "lr_mult": 1.0, "weight_decay": 0.01}],
        betas=(0.9, 0.95),
    )


def zero_and_freeze_spline_path(model: KANCrossLayerTranscoder) -> None:
    """SiLU + base_weight only; B-spline path hard-off."""
    if model.encoder_type != "kan":
        return
    for enc in model.encoders:
        kl = enc.kan_linear
        with torch.no_grad():
            kl.spline_weight.zero_()
        kl.spline_weight.requires_grad_(False)
        if getattr(kl, "enable_standalone_scale_spline", False):
            kl.spline_scaler.requires_grad_(False)


def zero_and_freeze_base_path(model: KANCrossLayerTranscoder) -> None:
    """B-spline path only; SiLU base hard-off (thesis-critical ablation).

    Mathematically valid: each edge is φ(x)=Σ c_k B_k(x). The SiLU base in
    efficient-kan is optional residual capacity, not required by KAN theory.
    """
    if model.encoder_type != "kan":
        return
    for enc in model.encoders:
        kl = enc.kan_linear
        with torch.no_grad():
            kl.base_weight.zero_()
        kl.base_weight.requires_grad_(False)


def zero_spline_path_eval(model: KANCrossLayerTranscoder) -> dict[str, torch.Tensor]:
    """Zero spline weights for eval; return backups to restore."""
    backups: dict[str, torch.Tensor] = {}
    if model.encoder_type != "kan":
        return backups
    for i, enc in enumerate(model.encoders):
        kl = enc.kan_linear
        backups[f"{i}"] = kl.spline_weight.detach().clone()
        with torch.no_grad():
            kl.spline_weight.zero_()
    return backups


def restore_spline_path(model: KANCrossLayerTranscoder, backups: dict[str, torch.Tensor]) -> None:
    if model.encoder_type != "kan":
        return
    for i, enc in enumerate(model.encoders):
        key = f"{i}"
        if key in backups:
            with torch.no_grad():
                enc.kan_linear.spline_weight.copy_(backups[key])


def _preact_std_per_layer(
    model: KANCrossLayerTranscoder,
    dataset: ActivationDataset,
    n_sequences: int,
    seed: int,
) -> torch.Tensor:
    """Per-layer std of encoder preactivations on a fixed sample."""
    g = torch.Generator().manual_seed(seed)
    n = min(n_sequences, len(dataset))
    idx = torch.randperm(len(dataset), generator=g)[:n].tolist()
    device = model.b_enc.device
    # Accumulate on GPU; one sequence at a time but vectorized per layer via encode path.
    sums = torch.zeros(model.n_layers, device=device, dtype=torch.float64)
    sumsq = torch.zeros(model.n_layers, device=device, dtype=torch.float64)
    count = torch.zeros(model.n_layers, device=device, dtype=torch.float64)
    with torch.no_grad():
        for i in idx:
            x = dataset[i]["mlp_inputs"].to(device=device, dtype=torch.float32)
            for layer_id in range(model.n_layers):
                # Cap positions — full seq B-spline expand is the slow path.
                xl = x[layer_id]
                if xl.shape[0] > 64:
                    xl = xl[:64]
                p = model.encode_layer(
                    xl, layer_id, apply_activation_function=False
                ).float()
                sums[layer_id] += p.sum().double()
                sumsq[layer_id] += p.pow(2).sum().double()
                count[layer_id] += p.numel()
    mean = sums / count.clamp_min(1.0)
    var = (sumsq / count.clamp_min(1.0)) - mean.pow(2)
    return var.clamp_min(0.0).sqrt().float().cpu()


def _scale_encoder_weights(model: KANCrossLayerTranscoder, scales: torch.Tensor) -> None:
    """Multiply each layer's encoder weights by ``scales[layer]`` (preact rescale)."""
    with torch.no_grad():
        for layer_id, s in enumerate(scales.tolist()):
            if abs(s - 1.0) < 1e-8:
                continue
            enc = model.encoders[layer_id]
            if model.encoder_type == "linear":
                enc.W_enc.mul_(s)
            else:
                # Scale whichever encoder path is still live.
                if enc.kan_linear.base_weight.requires_grad:
                    enc.kan_linear.base_weight.mul_(s)
                if enc.kan_linear.spline_weight.requires_grad:
                    enc.kan_linear.spline_weight.mul_(s)
                    if getattr(enc.kan_linear, "enable_standalone_scale_spline", False):
                        enc.kan_linear.spline_scaler.mul_(s)
            model.b_enc[layer_id].mul_(s)


def _activation_l0(
    model: KANCrossLayerTranscoder,
    dataset: ActivationDataset,
    n_sequences: int,
    seed: int,
) -> float:
    g = torch.Generator().manual_seed(seed)
    n = min(n_sequences, len(dataset))
    idx = torch.randperm(len(dataset), generator=g)[:n].tolist()
    device = model.b_enc.device
    active = 0.0
    positions = 0.0
    with torch.no_grad():
        for i in idx:
            x = dataset[i]["mlp_inputs"].to(device=device, dtype=torch.float32)
            # Cap length for speed
            if x.shape[1] > 64:
                x = x[:, :64]
            acts = model.encode(x)
            active += float((acts > 0).sum().item())
            positions += float(acts.shape[0] * acts.shape[1])  # layers * pos
    return active / max(positions, 1.0)


def apply_shared_init(
    reference: KANCrossLayerTranscoder,
    target: KANCrossLayerTranscoder,
    dataset: ActivationDataset,
    cfg: ProbeTrainConfig,
) -> dict[str, Any]:
    """Lock input-norm, preact scale, θ, and W_dec across arms.

    Copying linear θ onto a raw KAN is fatal: KAN preacts are ~4× smaller, so
    every feature dies (act_lp=0, loss stuck at 1.0). We first rescale the
    target encoder so its preact std matches the reference layer-wise, *then*
    copy θ and W_dec.
    """
    info: dict[str, Any] = {}
    n_cal = min(4, cfg.calibration_samples)
    print(
        f"[lock] matching preact scale ({target.encoder_type}, n_cal={n_cal})...",
        flush=True,
    )
    with torch.no_grad():
        target.set_input_normalization(
            reference.enc_input_mean.detach().clone(),
            reference.enc_input_std.detach().clone(),
        )

    ref_std = _preact_std_per_layer(reference, dataset, n_cal, cfg.seed + 11)
    print(f"[lock] ref preact std mean={float(ref_std.mean()):.4f}", flush=True)
    tgt_std = _preact_std_per_layer(target, dataset, n_cal, cfg.seed + 11)
    print(f"[lock] tgt preact std mean={float(tgt_std.mean()):.4f}", flush=True)
    scales = (ref_std / tgt_std.clamp_min(1e-6)).clamp(1e-3, 1e3)
    _scale_encoder_weights(target, scales)
    info["preact_scale_factors"] = scales.tolist()
    info["ref_preact_std"] = ref_std.tolist()
    info["tgt_preact_std_before"] = tgt_std.tolist()
    info["tgt_preact_std_after"] = _preact_std_per_layer(
        target, dataset, n_cal, cfg.seed + 11
    ).tolist()
    print(
        f"[lock] scale factors median={float(scales.median()):.3f} "
        f"after_std={float(torch.tensor(info['tgt_preact_std_after']).mean()):.4f}",
        flush=True,
    )

    with torch.no_grad():
        target.activation_function.threshold.copy_(
            reference.activation_function.threshold.detach()
        )
        for i in range(target.n_layers):
            target.W_dec[i].copy_(reference.W_dec[i].detach())
        target.b_dec.copy_(reference.b_dec.detach())

        theta = reference.effective_threshold.detach()
        bw = float((0.1 * theta.mean()).clamp_min(1e-4).item())
        target.activation_function.bandwidth = bw
        reference.activation_function.bandwidth = bw
        info["locked_bandwidth"] = bw
        info["locked_theta_mean"] = float(theta.mean().item())
        info["locked_theta_per_layer"] = [
            float(t.mean().item()) for t in theta.reshape(reference.n_layers, -1)
        ]
        info["decoder_col_norm_mean"] = [
            float(w.norm(dim=-1).max(dim=-1).values.mean().item()) for w in reference.W_dec
        ]

    l0 = _activation_l0(target, dataset, n_cal, cfg.seed + 17)
    info["post_lock_act_lp"] = l0
    print(f"[lock] post-lock act_lp={l0:.3f} (need >= 1)", flush=True)
    if l0 < 1.0:
        raise RuntimeError(
            f"Locked-scale init left target nearly dead (act_lp={l0:.4f}). "
            "Preact rescale failed to align encoder scales with reference θ."
        )
    return info


def calibrate_reference(
    model: KANCrossLayerTranscoder,
    dataset: ActivationDataset,
    cfg: ProbeTrainConfig,
) -> dict[str, Any]:
    """Run input-norm + data-quantile θ + data_scaled decoder on the reference arm."""
    seed_everything(cfg.seed)
    info: dict[str, Any] = {}
    mean, std = compute_input_normalization(
        dataset,
        n_layers=model.n_layers,
        d_model=model.d_model,
        n_sequences=cfg.calibration_samples,
        seed=cfg.seed,
    )
    model.set_input_normalization(mean, std)
    from circuit_tracer.transcoder.activation_functions import JumpReLU

    if isinstance(model.activation_function, JumpReLU):
        thresholds = initialize_thresholds_from_data(
            model,
            dataset,
            target_l0=cfg.target_l0,
            n_sequences=cfg.calibration_samples,
            values_per_sample=32768,
            seed=cfg.seed,
        )
        info["thresholds"] = thresholds.detach().cpu().tolist()
        info["theta_mean"] = float(model.effective_threshold.mean().item())
    else:
        info["thresholds"] = None
        info["theta_mean"] = None
    scales = calibrate_decoder_scale_from_data(
        model,
        dataset,
        n_sequences=cfg.calibration_samples,
        seed=cfg.seed,
    )
    info["decoder_scales"] = scales.detach().cpu().tolist() if isinstance(scales, torch.Tensor) else list(scales)
    return info


def set_trainable(model: KANCrossLayerTranscoder, cfg: ProbeTrainConfig, arm: ArmKind) -> None:
    for p in model.parameters():
        p.requires_grad_(True)
    from circuit_tracer.transcoder.activation_functions import JumpReLU

    if cfg.lock_threshold and isinstance(model.activation_function, JumpReLU):
        model.activation_function.threshold.requires_grad_(False)
    if cfg.lock_decoder or cfg.freeze_decoder:
        for w in model.W_dec:
            w.requires_grad_(False)
        model.b_dec.requires_grad_(False)
    if cfg.freeze_encoder:
        for enc in model.encoders:
            for p in enc.parameters():
                p.requires_grad_(False)
        model.b_enc.requires_grad_(False)
    if arm == "silu_base":
        zero_and_freeze_spline_path(model)
    elif arm in ("spline_only", "spline_only_nogrid"):
        zero_and_freeze_base_path(model)


def lr_at_step(step: int, cfg: ProbeTrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * float(step + 1) / float(max(1, cfg.warmup_steps))
    t = (step - cfg.warmup_steps) / float(max(1, cfg.total_steps - cfg.warmup_steps))
    return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


def lambda_at_step(step: int, cfg: ProbeTrainConfig) -> float:
    """Match ``spline_clt.training.train.sparsity_lambda_at_step`` for probes."""
    from spline_clt.training.train import sparsity_lambda_at_step

    return sparsity_lambda_at_step(
        step,
        lambda_sparsity=cfg.lambda_sparsity,
        total_steps=cfg.total_steps,
        sparsity_warmup_steps=cfg.sparsity_warmup_steps,
        sparsity_decay_start=cfg.sparsity_decay_start,
        lambda_sparsity_final=cfg.lambda_sparsity_final,
    )


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype):
    # Collate gives (B, n_layers, seq, d). Model expects (n_layers, n_pos, d) with
    # positions flattened across batch.
    x = batch["mlp_inputs"].to(device=device, dtype=dtype)
    y = batch["mlp_outputs"].to(device=device, dtype=dtype)
    if x.dim() == 4:
        b, n_layers, seq, d = x.shape
        x = x.permute(1, 0, 2, 3).reshape(n_layers, b * seq, d)
        y = y.permute(1, 0, 2, 3).reshape(n_layers, b * seq, d)
    return x, y


@torch.no_grad()
def eval_metrics(
    model: KANCrossLayerTranscoder,
    dataset: ActivationDataset,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int = 8,
    batch_size: int = 4,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    totals: dict[str, float] = {}
    n = 0
    model.eval()
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x, y = batch_to_device(batch, device, dtype)
        acts, y_hat, dec_norms = model(x)
        _, metrics = compute_losses(
            acts,
            y_hat,
            dec_norms,
            model,
            y,
            lambda_sparsity=0.0,
            c_sparsity=1.0,
            lambda_kan_reg=0.0,
            compute_metrics=True,
            recon_per_layer=True,
            sparsity_per_layer_mean=True,
        )
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        n += 1
    model.train()
    return {k: v / max(1, n) for k, v in totals.items()}


def grad_path_norms(model: KANCrossLayerTranscoder) -> dict[str, float]:
    """Sum of ||grad|| for encoder path groups."""
    from circuit_tracer.transcoder.activation_functions import JumpReLU

    out = {
        "grad/base_weight": 0.0,
        "grad/spline_weight": 0.0,
        "grad/encoder_other": 0.0,
        "grad/decoder": 0.0,
        "grad/threshold": 0.0,
        "grad/b_enc": 0.0,
    }
    if model.encoder_type == "kan":
        for enc in model.encoders:
            kl = enc.kan_linear
            if kl.base_weight.grad is not None:
                out["grad/base_weight"] += float(kl.base_weight.grad.detach().norm().item())
            if kl.spline_weight.grad is not None:
                out["grad/spline_weight"] += float(kl.spline_weight.grad.detach().norm().item())
    else:
        for enc in model.encoders:
            if enc.W_enc.grad is not None:
                out["grad/encoder_other"] += float(enc.W_enc.grad.detach().norm().item())
    for w in model.W_dec:
        if w.grad is not None:
            out["grad/decoder"] += float(w.grad.detach().norm().item())
    if isinstance(model.activation_function, JumpReLU):
        thr = model.activation_function.threshold
        if thr.grad is not None:
            out["grad/threshold"] = float(thr.grad.detach().norm().item())
    if model.b_enc.grad is not None:
        out["grad/b_enc"] = float(model.b_enc.grad.detach().norm().item())
    return out


def preact_moments(
    model: KANCrossLayerTranscoder, x: torch.Tensor
) -> dict[str, float]:
    from circuit_tracer.transcoder.activation_functions import JumpReLU

    with torch.no_grad():
        preacts = []
        for layer_id in range(model.n_layers):
            p = model.encode_layer(x[layer_id], layer_id, apply_activation_function=False)
            preacts.append(p.float())
        pcat = torch.stack(preacts, dim=0)  # (n_layers, n_pos, d_transcoder)
        out = {
            "preact/mean": float(pcat.mean().item()),
            "preact/std": float(pcat.std().item()),
            "preact/abs_mean": float(pcat.abs().mean().item()),
        }
        if not isinstance(model.activation_function, JumpReLU):
            out["preact/frac_above_theta"] = float((pcat > 0).float().mean().item())
            out["theta/mean"] = 0.0
            return out
        theta = model.effective_threshold.float()
        # effective_threshold is (n_layers, 1, d_transcoder); align for broadcast
        while theta.dim() < pcat.dim():
            theta = theta.unsqueeze(1)
        if theta.shape[-1] != pcat.shape[-1] and theta.shape[-1] == 1:
            pass
        elif theta.shape[0] == pcat.shape[0] and theta.numel() == pcat.shape[0]:
            # per-layer scalar fallback
            theta = theta.reshape(pcat.shape[0], 1, 1)
        out["preact/frac_above_theta"] = float((pcat > theta).float().mean().item())
        out["theta/mean"] = float(model.effective_threshold.float().mean().item())
        return out


def train_arm(
    arm: ArmKind,
    cfg: ProbeTrainConfig,
    out_dir: Path,
    reference_state: dict[str, Any] | None = None,
    reference_model: KANCrossLayerTranscoder | None = None,
) -> dict[str, Any]:
    """Train one arm under locked scales; write records + checkpoint."""
    from circuit_tracer.transcoder.activation_functions import JumpReLU

    print(f"[train_arm] start arm={arm} out={out_dir}", flush=True)
    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dtype = _dtype(cfg.dtype)
    print(f"[train_arm] loading dataset from {cfg.data_dir}", flush=True)
    train_ds, val_ds = load_probe_dataset(cfg)
    print(f"[train_arm] n_train={len(train_ds)} n_val={len(val_ds)}", flush=True)
    out_dir = ensure_dir(out_dir)

    print(f"[train_arm] building model arm={arm}", flush=True)
    model = build_model(arm, cfg, device)

    init_info: dict[str, Any]
    if reference_model is not None:
        print(f"[train_arm] apply_shared_init from reference → {arm}", flush=True)
        # Copy locks from an already-calibrated reference.
        init_info = apply_shared_init(reference_model, model, train_ds, cfg)
        init_info["init_mode"] = "copied_from_reference"
    elif reference_state is not None:
        # Rebuild reference from state dict path — unused currently.
        init_info = {"init_mode": "state", **reference_state}
    else:
        # This arm IS the reference: calibrate then keep.
        print(f"[train_arm] calibrate_reference for {arm}", flush=True)
        init_info = calibrate_reference(model, train_ds, cfg)
        init_info["init_mode"] = "calibrated_reference"
        if isinstance(model.activation_function, JumpReLU):
            theta_mean = float(model.effective_threshold.mean().item())
            if cfg.absolute_jumprelu_bandwidth is not None:
                bw = float(cfg.absolute_jumprelu_bandwidth)
                init_info["bandwidth_mode"] = "absolute"
            else:
                bw = max(float(cfg.bandwidth_frac) * theta_mean, 1e-4)
                init_info["bandwidth_mode"] = "frac"
                init_info["bandwidth_frac"] = cfg.bandwidth_frac
            model.activation_function.bandwidth = bw
            init_info["bandwidth"] = bw
            init_info["theta_mean"] = theta_mean

    # Optional θ delay: hold gate open (θ≈floor) then restore calibrated θ.
    delay_steps = int(cfg.theta_delay_frac * cfg.total_steps)
    saved_log_theta: torch.Tensor | None = None
    if (
        delay_steps > 0
        and isinstance(model.activation_function, JumpReLU)
    ):
        saved_log_theta = model.activation_function.threshold.detach().clone()
        with torch.no_grad():
            model.activation_function.threshold.fill_(math.log(cfg.theta_delay_floor))
        init_info["theta_delay_steps"] = delay_steps
        init_info["theta_delay_floor"] = cfg.theta_delay_floor
        print(
            f"[train_arm] θ delay: floor={cfg.theta_delay_floor} for {delay_steps} steps",
            flush=True,
        )

    print(f"[train_arm] entering train loop arm={arm}", flush=True)
    set_trainable(model, cfg, arm)
    # During delay, freeze θ so it stays at floor until restore.
    if saved_log_theta is not None:
        model.activation_function.threshold.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params and saved_log_theta is None:
        raise RuntimeError(f"No trainable parameters for arm={arm}")

    opt = build_optimizer(model, cfg)
    print(
        f"[train_arm] scale_base={cfg.scale_base} scale_spline={cfg.scale_spline} "
        f"lr_spline_mult={cfg.lr_spline_mult} λ_kan={cfg.lambda_kan_reg}",
        flush=True,
    )

    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    records_path = out_dir / "training_records.jsonl"
    if records_path.exists():
        records_path.unlink()

    step = 0
    data_iter = iter(loader)
    model.train()
    torch.set_float32_matmul_precision("high")
    theta_restored = saved_log_theta is None

    while step < cfg.total_steps:
        if (
            not theta_restored
            and saved_log_theta is not None
            and step == delay_steps
        ):
            with torch.no_grad():
                model.activation_function.threshold.copy_(saved_log_theta)
            if not cfg.lock_threshold:
                model.activation_function.threshold.requires_grad_(True)
                # Rebuild optimizer so θ enters the right group after delay.
                opt = build_optimizer(model, cfg)
            theta_restored = True
            print(
                f"[train_arm] restored calibrated θ at step {step}; "
                f"meanθ={float(model.effective_threshold.mean().item()):.4f}",
                flush=True,
            )

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x, y = batch_to_device(batch, device, dtype)
        lr = lr_at_step(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr * float(g.get("lr_mult", 1.0))

        opt.zero_grad(set_to_none=True)
        acts, y_hat, dec_norms = model(x)
        lam = lambda_at_step(step, cfg)
        loss, metrics = compute_losses(
            acts,
            y_hat,
            dec_norms,
            model,
            y,
            lambda_sparsity=lam,
            c_sparsity=cfg.c_sparsity,
            lambda_kan_reg=(
                cfg.lambda_kan_reg if arm in ("kan", "spline_only", "spline_only_nogrid") else 0.0
            ),
            compute_metrics=(step % cfg.log_every == 0),
            recon_per_layer=True,
            sparsity_per_layer_mean=True,
        )
        loss.backward()
        grad_stats = grad_path_norms(model) if step % cfg.log_every == 0 else {}
        clip_params = [p for p in model.parameters() if p.requires_grad]
        if cfg.grad_clip > 0 and clip_params:
            torch.nn.utils.clip_grad_norm_(clip_params, cfg.grad_clip)
        opt.step()

        # Adapt B-spline knots to the data (needed when SiLU base is off).
        if (
            model.encoder_type == "kan"
            and cfg.update_grid_every > 0
            and step >= cfg.update_grid_from
            and step % cfg.update_grid_every == 0
        ):
            with torch.no_grad():
                for layer_id in range(model.n_layers):
                    x_layer = model._normalize_input(x[layer_id], layer_id)
                    if x_layer.shape[0] > cfg.update_grid_max_pos:
                        x_layer = x_layer[: cfg.update_grid_max_pos]
                    model.encoders[layer_id].update_grid(x_layer)
            print(f"[train_arm] update_grid at step {step}", flush=True)
            # Re-freeze base if spline-only (update_grid shouldn't touch it, but be safe)
            if arm in ("spline_only", "spline_only_nogrid"):
                zero_and_freeze_base_path(model)
        if step % cfg.log_every == 0:
            record = {
                "step": step,
                "arm": arm,
                "lr": lr,
                "lambda_sparsity_eff": lam,
                "loss/total": float(loss.detach().item()),
                **metrics,
                **grad_stats,
                **preact_moments(model, x),
            }
            if model.encoder_type == "kan":
                record["stats/spline_contribution_frac"] = model.spline_contribution_fraction(x)
            with records_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(
                f"[{arm}] step {step}/{cfg.total_steps} "
                f"loss={record['loss/total']:.4f} "
                f"rel_fro={record.get('reconstruction/rel_fro_error', float('nan')):.3f} "
                f"act_lp={record.get('stats/active_features_per_pos', float('nan')):.2f} "
                f"L0={record.get('stats/l0_active_features_per_token', float('nan')):.1f} "
                f"λ={lam:.2e}",
                flush=True,
            )
        step += 1

    val = eval_metrics(model, val_ds, device, dtype, batch_size=cfg.batch_size)
    # Base-only eval for KAN arms
    base_only = None
    if model.encoder_type == "kan" and arm == "kan":
        backups = zero_spline_path_eval(model)
        base_only = eval_metrics(model, val_ds, device, dtype, batch_size=cfg.batch_size)
        restore_spline_path(model, backups)

    ckpt_dir = ensure_dir(out_dir / "checkpoint")
    model.to_safetensors(str(ckpt_dir))

    summary = {
        "arm": arm,
        "config": asdict(cfg),
        "init": init_info,
        "final_val": val,
        "base_only_val": base_only,
        "checkpoint": str(ckpt_dir),
        "records": str(records_path),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "device": str(device),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {"summary": summary, "model": model}
