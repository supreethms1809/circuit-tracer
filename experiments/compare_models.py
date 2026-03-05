#!/usr/bin/env python3
"""Compare Spline-CLT vs linear CLT baseline on reconstruction and sparsity metrics.

Loads two trained checkpoints and evaluates both on the same held-out data,
printing a side-by-side comparison table.

Usage:
    python experiments/compare_models.py \
        --kan-checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
        --linear-checkpoint checkpoints/gpt2_small_linear/linear_clt_gpt2_best \
        --data-dir data/activations \
        --n-samples 200
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.nn import functional as F
from tqdm import tqdm

from spline_clt.kan_transcoder import KANCrossLayerTranscoder, load_spline_clt
from spline_clt.seed import seed_everything
from spline_clt.training.data import ActivationDataset


@torch.no_grad()
def evaluate_model(
    model: KANCrossLayerTranscoder,
    dataset: ActivationDataset,
    device: torch.device,
    dtype: torch.dtype,
    n_samples: int = 200,
    seed: int | None = None,
) -> dict[str, float | list[float]]:
    """Evaluate reconstruction quality and sparsity on held-out samples.

    Args:
        model: Trained Spline-CLT or linear CLT.
        dataset: Activation dataset (full, not split).
        device: Compute device.
        dtype: Data dtype.
        n_samples: Number of sequences to evaluate.

    Returns:
        Dict of metrics: mse_total, mse_per_layer, cosine_sim, rel_error,
        active_per_pos, activation_density, n_params.
    """
    model.eval()
    if seed is not None:
        seed_everything(seed)
    n_samples = min(n_samples, len(dataset))
    indices = torch.randperm(len(dataset))[:n_samples]

    n_layers = model.n_layers
    d_model = model.d_model

    mse_per_layer = [0.0] * n_layers
    cos_sim_total = 0.0
    rel_error_total = 0.0
    active_counts = []
    densities = []

    for idx in tqdm(indices, desc=f"Evaluating {model.encoder_type}", leave=False):
        sample = dataset[int(idx)]
        x_in = sample["mlp_inputs"].to(device=device, dtype=dtype)   # (n_layers, seq, d)
        y_true = sample["mlp_outputs"].to(device=device, dtype=dtype)

        activations = model.encode(x_in)       # (n_layers, seq, d_transcoder)
        y_hat = model.decode_dense(activations, input_acts=x_in)

        # Per-layer MSE
        for l in range(n_layers):
            mse = ((y_hat[l].float() - y_true[l].float()) ** 2).mean().item()
            mse_per_layer[l] += mse

        # Cosine similarity (all layers, all positions, flattened)
        cos = F.cosine_similarity(
            y_hat.reshape(-1, d_model).float(),
            y_true.reshape(-1, d_model).float(),
            dim=-1,
        ).mean().item()
        cos_sim_total += cos

        # Relative error
        err = (y_hat.float() - y_true.float()).norm() / y_true.float().norm().clamp(min=1e-8)
        rel_error_total += err.item()

        # Sparsity
        active = (activations > 0).float()
        active_counts.append(active.sum(dim=-1).mean().item())
        densities.append(active.mean().item())

    mse_per_layer = [m / n_samples for m in mse_per_layer]
    n_params = sum(p.numel() for p in model.parameters())

    return {
        "mse_total": sum(mse_per_layer) / n_layers,
        "mse_per_layer": mse_per_layer,
        "cosine_similarity": cos_sim_total / n_samples,
        "relative_error": rel_error_total / n_samples,
        "active_per_pos": sum(active_counts) / n_samples,
        "activation_density": sum(densities) / n_samples,
        "n_params": n_params,
    }


def fmt(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}"


def print_comparison(
    kan_metrics: dict,
    lin_metrics: dict,
    kan_label: str = "Spline-CLT",
    lin_label: str = "Linear CLT",
) -> None:
    """Print a formatted side-by-side comparison table."""
    col_w = 14
    header = f"{'Metric':<28} {kan_label:>{col_w}} {lin_label:>{col_w}} {'Δ (KAN-Lin)':>{col_w}}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    rows = [
        ("MSE (total)", "mse_total", 6),
        ("Cosine similarity", "cosine_similarity", 4),
        ("Relative error", "relative_error", 4),
        ("Active features/pos", "active_per_pos", 2),
        ("Activation density", "activation_density", 6),
        ("Parameters", "n_params", 0),
    ]

    for label, key, dec in rows:
        k_val = kan_metrics[key]
        l_val = lin_metrics[key]
        if key == "n_params":
            k_str = f"{k_val:,}"
            l_str = f"{l_val:,}"
            d_str = f"{k_val - l_val:+,}"
        else:
            k_str = fmt(k_val, dec)
            l_str = fmt(l_val, dec)
            d_str = fmt(k_val - l_val, dec)
        print(f"{label:<28} {k_str:>{col_w}} {l_str:>{col_w}} {d_str:>{col_w}}")

    print("=" * len(header))

    # Per-layer MSE breakdown
    print("\nPer-layer MSE:")
    n_layers = len(kan_metrics["mse_per_layer"])
    print(f"  {'Layer':<6} {kan_label:>12} {lin_label:>12} {'Δ':>10}")
    for l in range(n_layers):
        k = kan_metrics["mse_per_layer"][l]
        lv = lin_metrics["mse_per_layer"][l]
        print(f"  {l:<6} {k:>12.6f} {lv:>12.6f} {k-lv:>+10.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Spline-CLT vs linear CLT baseline")
    parser.add_argument(
        "--kan-checkpoint", type=str, default=None,
        help="Path to Spline-CLT checkpoint directory",
    )
    parser.add_argument(
        "--linear-checkpoint", type=str, default=None,
        help="Path to linear CLT checkpoint directory",
    )
    parser.add_argument("--data-dir", type=str, default="data/activations")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "bfloat16"])
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"Device: {device} | dtype: {dtype}")
    print(f"Loading dataset from {args.data_dir}...")
    dataset = ActivationDataset.load(args.data_dir, max_samples=args.n_samples + 50)
    print(f"  {len(dataset)} samples loaded")

    models: dict[str, KANCrossLayerTranscoder] = {}

    for label, ckpt_path in [("Spline-CLT", args.kan_checkpoint), ("Linear CLT", args.linear_checkpoint)]:
        if ckpt_path is None:
            print(f"\n[{label}] No checkpoint provided — skipping.")
            continue
        if not os.path.isdir(ckpt_path):
            print(f"\n[{label}] Checkpoint not found: {ckpt_path} — skipping.")
            continue
        print(f"\nLoading {label} from {ckpt_path}...")
        model = load_spline_clt(ckpt_path, device=device, dtype=dtype)
        model.eval()
        n_p = sum(p.numel() for p in model.parameters())
        print(f"  encoder_type={model.encoder_type}, {n_p:,} parameters")
        models[label] = model

    if len(models) == 0:
        print("\nNo checkpoints loaded. Pass --kan-checkpoint and/or --linear-checkpoint.")
        return

    metrics = {}
    for label, model in models.items():
        print(f"\nEvaluating {label} on {args.n_samples} samples...")
        m = evaluate_model(model, dataset, device, dtype, n_samples=args.n_samples)
        metrics[label] = m
        print(f"  MSE={m['mse_total']:.6f}  cos_sim={m['cosine_similarity']:.4f}  "
              f"active/pos={m['active_per_pos']:.2f}  params={m['n_params']:,}")

    if "Spline-CLT" in metrics and "Linear CLT" in metrics:
        print_comparison(metrics["Spline-CLT"], metrics["Linear CLT"])
    else:
        for label, m in metrics.items():
            print(f"\n{label}:")
            for k, v in m.items():
                if k != "mse_per_layer":
                    print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
