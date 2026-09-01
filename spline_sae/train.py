"""Online activation streaming + Spline-SAE / sparse transcoder training.

One job does: load LM → stream tokens → train SAE → write metrics/checkpoint.
Designed for 1-GPU Slurm baseline runs (linear replicate + kan analog).

SAE mode: encode and reconstruct the same hook (``hook_name``).
Transcoder mode: set ``target_hook_name`` (e.g. resid_pre → attn_out).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

import torch
import yaml
from torch.optim import Adam

from spline_clt.seed import seed_everything
from spline_sae.loss import compute_sae_losses
from spline_sae.model import SplineSAE


@dataclass
class TrainConfig:
    model_name: str
    d_model: int
    hook_name: str
    layer: int
    d_sae: int
    # If set and different from hook_name: encode hook_name, reconstruct target.
    target_hook_name: str = ""
    # If True (transcoder only): reconstruct (target - input), e.g. layer write
    # Δ = resid_post - resid_pre, so the skip connection does not dominate NMSE.
    reconstruct_delta: bool = False
    encoder_type: Literal["kan", "linear"] = "linear"
    activation: Literal["jumprelu", "topk", "relu", "base_jump"] = "jumprelu"
    decoder_type: Literal["linear", "mlp", "linear_mlp", "kan"] = "linear"
    decoder_hidden: int = 0  # 0 → d_model (mlp) or decoder_bot (kan)
    decoder_bot: int = 512  # KAN decoder bottleneck width
    topk_k: int = 32
    threshold_init: float = 0.01
    jumprelu_bandwidth: float = 0.001
    grid_size: int = 5
    spline_order: int = 3
    scale_base: float = 1.0
    scale_spline: float = 1.0
    lambda_sparsity: float = 1e-3
    c_sparsity: float = 1.0
    lambda_nl_gap: float = 0.0
    lambda_frac_hinge: float = 0.0
    frac_target: float = 0.35
    freeze_base_after: int = 0  # 0 = never; else freeze KAN base_weight at this step
    target_l0: float = 0.0  # 0 = off; soft-adapt lambda_sparsity toward this L0
    l0_adapt_every: int = 200
    l0_adapt_rate: float = 0.05
    learning_rate: float = 3e-4
    lr_spline_mult: float = 1.0
    n_steps: int = 20_000
    batch_tokens: int = 4096
    seq_len: int = 1024
    dataset: str = "monology/pile-uncopyrighted"
    dataset_config: str | None = None
    text_column: str = "text"
    seed: int = 101
    log_every: int = 50
    eval_every: int = 500
    eval_batches: int = 8
    dtype: str = "bfloat16"
    device: str = "cuda"
    output_dir: str = ""
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_target_hook(self) -> str:
        return self.target_hook_name or self.hook_name

    @property
    def is_transcoder(self) -> bool:
        return bool(self.target_hook_name) and self.target_hook_name != self.hook_name

    @staticmethod
    def from_yaml(path: Path) -> "TrainConfig":
        raw = yaml.safe_load(path.read_text())
        known = {f.name for f in TrainConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in raw.items() if k in known and k != "extras"}
        cfg = TrainConfig(**kwargs)
        cfg.extras = {k: v for k, v in raw.items() if k not in known}
        return cfg


def _load_lm(model_name: str, device: str, dtype: torch.dtype):
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _text_stream(dataset: str, dataset_config: str | None, text_column: str, seed: int) -> Iterator[str]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"path": dataset, "split": "train", "streaming": True}
    if dataset_config:
        kwargs["name"] = dataset_config
    ds = load_dataset(**kwargs)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    for row in ds:
        text = row.get(text_column) or row.get("content") or ""
        if isinstance(text, str) and text.strip():
            yield text


def _layer_index(hook_name: str) -> int | None:
    try:
        return int(hook_name.split(".")[1])
    except (IndexError, ValueError):
        return None


@torch.no_grad()
def _activation_batches(
    model,
    hook_name: str,
    texts: Iterator[str],
    seq_len: int,
    batch_tokens: int,
    device: str,
    target_hook_name: str = "",
    reconstruct_delta: bool = False,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(x_in, y_out)`` chunks of ~batch_tokens.

    When ``reconstruct_delta`` and a distinct target hook are set, ``y_out`` is
    ``target - input`` (layer residual write).
    """
    target = target_hook_name or hook_name
    paired = target != hook_name
    if reconstruct_delta and not paired:
        raise ValueError("reconstruct_delta=True requires a distinct target_hook_name")
    buf_x: list[torch.Tensor] = []
    buf_y: list[torch.Tensor] = []
    n_buf = 0
    tokenizer = model.tokenizer
    hooks = [hook_name] if not paired else [hook_name, target]
    stop_idxs = [i for i in (_layer_index(h) for h in hooks) if i is not None]
    stop = (max(stop_idxs) + 1) if stop_idxs else None

    for text in texts:
        toks = tokenizer.encode(text, add_special_tokens=False)
        if not toks:
            continue
        for start in range(0, len(toks), seq_len):
            window = toks[start : start + seq_len]
            if len(window) < 16:
                continue
            ids = torch.tensor([window], device=device)
            _, cache = model.run_with_cache(
                ids, names_filter=hooks, stop_at_layer=stop
            )
            xin = cache[hook_name][0].detach()
            yout = cache[target][0].detach() if paired else xin
            if xin.shape[0] > 1:
                xin = xin[1:]
                if paired:
                    yout = yout[1:]
            xin = xin.reshape(-1, xin.shape[-1])
            yout = yout.reshape(-1, yout.shape[-1]) if paired else xin
            if paired and xin.shape[0] != yout.shape[0]:
                n = min(xin.shape[0], yout.shape[0])
                xin, yout = xin[:n], yout[:n]
            if reconstruct_delta:
                yout = yout - xin
            buf_x.append(xin)
            buf_y.append(yout if paired else xin)
            n_buf += xin.shape[0]
            del cache
            while n_buf >= batch_tokens:
                cat_x = torch.cat(buf_x, dim=0)
                cat_y = torch.cat(buf_y, dim=0)
                yield cat_x[:batch_tokens].contiguous(), cat_y[:batch_tokens].contiguous()
                rem_x = cat_x[batch_tokens:]
                rem_y = cat_y[batch_tokens:]
                buf_x = [rem_x] if rem_x.numel() else []
                buf_y = [rem_y] if rem_y.numel() else []
                n_buf = rem_x.shape[0] if rem_x.numel() else 0


def _build_optimizer(model: SplineSAE, cfg: TrainConfig) -> Adam:
    if model.encoder_type != "kan" or cfg.lr_spline_mult == 1.0:
        return Adam(model.parameters(), lr=cfg.learning_rate)
    base_params: list[torch.nn.Parameter] = []
    spline_params: list[torch.nn.Parameter] = []
    other: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "base_weight" in name:
            base_params.append(p)
        elif "spline_weight" in name or "spline_scaler" in name:
            spline_params.append(p)
        else:
            other.append(p)
    groups = [
        {"params": other, "lr": cfg.learning_rate},
        {"params": base_params, "lr": cfg.learning_rate},
        {"params": spline_params, "lr": cfg.learning_rate * cfg.lr_spline_mult, "lr_mult": cfg.lr_spline_mult},
    ]
    return Adam([g for g in groups if g["params"]], lr=cfg.learning_rate)


@torch.no_grad()
def _eval_loop(
    model: SplineSAE,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    cfg: TrainConfig,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    n = 0
    for x, y in batches:
        _, metrics = compute_sae_losses(
            model,
            x,
            y=y,
            lambda_sparsity=cfg.lambda_sparsity,
            c_sparsity=cfg.c_sparsity,
            lambda_nl_gap=cfg.lambda_nl_gap,
            lambda_frac_hinge=cfg.lambda_frac_hinge,
            frac_target=cfg.frac_target,
            compute_metrics=True,
        )
        for k, v in metrics.items():
            if isinstance(v, float) and math.isfinite(v):
                totals[k] = totals.get(k, 0.0) + v
        n += 1
    if n == 0:
        return {}
    return {k: v / n for k, v in totals.items()}


def train(cfg: TrainConfig) -> Path:
    seed_everything(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.dtype]

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(json.dumps({**asdict(cfg), "device_resolved": device}, indent=2) + "\n")

    print(f"[spline_sae] loading LM {cfg.model_name} on {device}", flush=True)
    lm = _load_lm(cfg.model_name, device=device, dtype=dtype)
    mode = "transcoder" if cfg.is_transcoder else "sae"
    tgt = f"Δ({cfg.resolved_target_hook}-{cfg.hook_name})" if cfg.reconstruct_delta else cfg.resolved_target_hook
    print(
        f"[spline_sae] LM ready d_model={cfg.d_model} mode={mode} "
        f"in={cfg.hook_name} out={tgt}",
        flush=True,
    )

    sae = SplineSAE(
        d_model=cfg.d_model,
        d_sae=cfg.d_sae,
        encoder_type=cfg.encoder_type,
        activation=cfg.activation,
        decoder_type=cfg.decoder_type,
        decoder_hidden=(cfg.decoder_hidden or None),
        decoder_bot=cfg.decoder_bot,
        topk_k=cfg.topk_k,
        threshold_init=cfg.threshold_init,
        jumprelu_bandwidth=cfg.jumprelu_bandwidth,
        grid_size=cfg.grid_size,
        spline_order=cfg.spline_order,
        scale_base=cfg.scale_base,
        scale_spline=cfg.scale_spline,
    ).to(device=device, dtype=torch.float32)
    opt = _build_optimizer(sae, cfg)
    lambda_s = float(cfg.lambda_sparsity)
    base_frozen = False

    texts = _text_stream(cfg.dataset, cfg.dataset_config, cfg.text_column, cfg.seed)
    act_iter = _activation_batches(
        lm,
        cfg.hook_name,
        texts,
        cfg.seq_len,
        cfg.batch_tokens,
        device,
        target_hook_name=cfg.target_hook_name,
        reconstruct_delta=cfg.reconstruct_delta,
    )

    eval_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    metrics_path = out / "train_metrics.jsonl"
    best_nmse = float("inf")
    t0 = time.time()

    print(
        f"[spline_sae] training {cfg.encoder_type}/{cfg.activation}/{cfg.decoder_type} "
        f"d_sae={cfg.d_sae} steps={cfg.n_steps} scale_base={cfg.scale_base} "
        f"lr_spline_mult={cfg.lr_spline_mult} freeze_base_after={cfg.freeze_base_after} "
        f"target_l0={cfg.target_l0}",
        flush=True,
    )
    for step in range(1, cfg.n_steps + 1):
        if (
            not base_frozen
            and cfg.freeze_base_after > 0
            and step >= cfg.freeze_base_after
            and sae.encoder_type == "kan"
        ):
            n_frozen = 0
            for name, p in sae.named_parameters():
                if "base_weight" in name:
                    p.requires_grad_(False)
                    n_frozen += 1
            base_frozen = True
            opt = _build_optimizer(sae, cfg)
            print(f"[spline_sae] froze {n_frozen} base_weight tensors at step {step}", flush=True)

        x, y = next(act_iter)
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        if len(eval_batches) < cfg.eval_batches:
            eval_batches.append((x.detach().clone(), y.detach().clone()))

        sae.train()
        loss, metrics = compute_sae_losses(
            sae,
            x,
            y=y,
            lambda_sparsity=lambda_s,
            c_sparsity=cfg.c_sparsity,
            lambda_nl_gap=cfg.lambda_nl_gap,
            lambda_frac_hinge=cfg.lambda_frac_hinge,
            frac_target=cfg.frac_target,
            compute_metrics=(step % cfg.log_every == 0 or step == 1),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()

        if (
            cfg.target_l0 > 0
            and step % cfg.l0_adapt_every == 0
            and "stats/l0" in metrics
        ):
            ratio = metrics["stats/l0"] / cfg.target_l0
            factor = ratio ** cfg.l0_adapt_rate
            factor = max(0.5, min(2.0, factor))
            lambda_s = float(max(1e-8, lambda_s * factor))
            metrics["stats/lambda_sparsity"] = lambda_s

        if step % cfg.log_every == 0 or step == 1:
            row = {
                "step": step,
                "seconds": time.time() - t0,
                "stats/lambda_sparsity": lambda_s,
                "stats/base_frozen": base_frozen,
                **metrics,
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"step {step}/{cfg.n_steps} nmse={metrics.get('loss/nmse', float('nan')):.4f} "
                f"l0={metrics.get('stats/l0', float('nan')):.2f} "
                f"frac={metrics.get('stats/spline_contribution_frac', float('nan'))} "
                f"act_frac={metrics.get('stats/act_frac', float('nan'))} "
                f"gap={metrics.get('stats/recon_gap', float('nan'))} "
                f"λs={lambda_s:.3g}",
                flush=True,
            )

        if step % cfg.eval_every == 0 or step == cfg.n_steps:
            sae.eval()
            cfg_lambda = cfg.lambda_sparsity
            cfg.lambda_sparsity = lambda_s
            ev = _eval_loop(sae, eval_batches, cfg)
            cfg.lambda_sparsity = cfg_lambda
            ev_row = {
                "step": step,
                "split": "eval_holdout",
                "stats/lambda_sparsity": lambda_s,
                "stats/base_frozen": base_frozen,
                **ev,
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(ev_row) + "\n")
            nmse_v = ev.get("loss/nmse", float("inf"))
            print(f"[eval] step {step} nmse={nmse_v:.4f} l0={ev.get('stats/l0', float('nan')):.2f}", flush=True)
            ckpt = {
                "step": step,
                "config": asdict(cfg),
                "state_dict": {k: v.detach().cpu() for k, v in sae.state_dict().items()},
                "eval": ev,
            }
            torch.save(ckpt, out / "last.pt")
            if nmse_v < best_nmse:
                best_nmse = nmse_v
                torch.save(ckpt, out / "best.pt")
                (out / "best_eval.json").write_text(json.dumps(ev_row, indent=2) + "\n")

    summary = {
        "best_nmse": best_nmse,
        "n_steps": cfg.n_steps,
        "seconds": time.time() - t0,
        "output_dir": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[spline_sae] done best_nmse={best_nmse:.4f} → {out}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Train Spline-SAE baseline / analog")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="")
    p.add_argument("--encoder-type", type=str, default="", choices=["", "kan", "linear"])
    p.add_argument("--n-steps", type=int, default=0)
    p.add_argument("--lambda-sparsity", type=float, default=-1.0)
    p.add_argument("--lambda-nl-gap", type=float, default=-1.0)
    p.add_argument("--scale-base", type=float, default=-1.0)
    p.add_argument("--lr-spline-mult", type=float, default=-1.0)
    args = p.parse_args()

    cfg = TrainConfig.from_yaml(Path(args.config))
    if args.encoder_type:
        cfg.encoder_type = args.encoder_type  # type: ignore[assignment]
    if args.n_steps > 0:
        cfg.n_steps = args.n_steps
    if args.lambda_sparsity >= 0:
        cfg.lambda_sparsity = args.lambda_sparsity
    if args.lambda_nl_gap >= 0:
        cfg.lambda_nl_gap = args.lambda_nl_gap
    if args.scale_base >= 0:
        cfg.scale_base = args.scale_base
    if args.lr_spline_mult >= 0:
        cfg.lr_spline_mult = args.lr_spline_mult
    if args.output_dir:
        cfg.output_dir = args.output_dir
    elif not cfg.output_dir:
        tag = f"{cfg.model_name.split('/')[-1]}_l{cfg.layer}_{cfg.encoder_type}_{cfg.activation}"
        cfg.output_dir = f"/gscratch/ssuresh/results/spline_sae/{tag}"

    train(cfg)


if __name__ == "__main__":
    main()
