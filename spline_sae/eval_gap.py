"""Evaluate recon_gap on finished Spline-SAE checkpoints.

For each KAN checkpoint:
  gap = NMSE(decode(base_only)) - NMSE(decode(full))
gap > 0 ⇒ splines help reconstruction on the sampled activations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spline_sae.loss import nmse, recon_gap_metric
from spline_sae.model import SplineSAE
from spline_sae.train import TrainConfig, _activation_batches, _load_lm, _text_stream


def _build_from_ckpt(ckpt: dict, device: str) -> SplineSAE:
    cfg = ckpt["config"]
    model = SplineSAE(
        d_model=cfg["d_model"],
        d_sae=cfg["d_sae"],
        encoder_type=cfg["encoder_type"],
        activation=cfg["activation"],
        topk_k=cfg.get("topk_k", 32),
        threshold_init=cfg.get("threshold_init", 0.01),
        jumprelu_bandwidth=cfg.get("jumprelu_bandwidth", 0.001),
        grid_size=cfg.get("grid_size", 5),
        spline_order=cfg.get("spline_order", 3),
        scale_base=cfg.get("scale_base", 1.0),
        scale_spline=cfg.get("scale_spline", 1.0),
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device=device, dtype=torch.float32).eval()


@torch.no_grad()
def eval_checkpoint(
    ckpt_path: Path,
    n_batches: int = 16,
    batch_tokens: int = 2048,
    seed: int = 123,
    device: str = "cuda",
) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_raw = ckpt["config"]
    # Minimal TrainConfig-like access
    model_name = cfg_raw["model_name"]
    hook = cfg_raw["hook_name"]
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        cfg_raw.get("dtype", "bfloat16")
    ]
    print(f"[gap-eval] loading LM {model_name}", flush=True)
    lm = _load_lm(model_name, device=device, dtype=dtype)
    sae = _build_from_ckpt(ckpt, device)

    texts = _text_stream(
        cfg_raw.get("dataset", "monology/pile-uncopyrighted"),
        cfg_raw.get("dataset_config"),
        cfg_raw.get("text_column", "text"),
        seed,
    )
    act_iter = _activation_batches(
        lm, hook, texts, cfg_raw.get("seq_len", 1024), batch_tokens, device
    )

    gaps = []
    nmse_full = []
    nmse_base = []
    l0s = []
    fracs = []
    for i in range(n_batches):
        x = next(act_iter).to(device=device, dtype=torch.float32)
        stats = recon_gap_metric(sae, x)
        y, a, _ = sae(x)
        gaps.append(stats["recon_gap"])
        nmse_full.append(stats["nmse_full"])
        nmse_base.append(stats["nmse_base_only"])
        l0s.append(float((a > 0).float().sum(dim=-1).mean().item()))
        if sae.encoder_type == "kan":
            fracs.append(sae.spline_contribution_fraction(x))
        print(
            f"  batch {i+1}/{n_batches} nmse_full={stats['nmse_full']:.4f} "
            f"nmse_base={stats['nmse_base_only']:.4f} gap={stats['recon_gap']:+.4f}",
            flush=True,
        )

    def mean(xs: list[float]) -> float:
        return float(sum(xs) / max(len(xs), 1))

    return {
        "checkpoint": str(ckpt_path),
        "model_name": model_name,
        "encoder_type": cfg_raw["encoder_type"],
        "activation": cfg_raw["activation"],
        "n_batches": n_batches,
        "batch_tokens": batch_tokens,
        "nmse_full_mean": mean(nmse_full),
        "nmse_base_only_mean": mean(nmse_base),
        "recon_gap_mean": mean(gaps),
        "l0_mean": mean(l0s),
        "spline_frac_mean": mean(fracs) if fracs else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Paths to best.pt (or directories containing best.pt)",
    )
    p.add_argument("--n-batches", type=int, default=16)
    p.add_argument("--batch-tokens", type=int, default=2048)
    p.add_argument("--output", type=str, default="/gscratch/ssuresh/results/spline_sae/gap_eval.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for raw in args.checkpoints:
        path = Path(raw)
        if path.is_dir():
            path = path / "best.pt"
        print(f"\n=== {path} ===", flush=True)
        results.append(eval_checkpoint(path, n_batches=args.n_batches, batch_tokens=args.batch_tokens, device=device))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, indent=2) + "\n")
    print(f"\nWrote {out}", flush=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
