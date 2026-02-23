#!/usr/bin/env python3
"""Spline shape analysis for KAN-CLT encoders.

Extracts and visualizes the learned B-spline functions for the most active
features. Useful for understanding what nonlinear patterns the KAN encoder
has learned to detect.

For each top feature we plot:
    - The effective transfer function KAN_enc(e_i * t) for the most
      influential input dimensions (found by looking at base_weight magnitude).
    - Histogram of activations across the validation dataset.

Usage:
    python experiments/analyze_splines.py \
        --checkpoint checkpoints/gpt2_small/kan_clt_gpt2_best \
        --data-dir data/activations \
        --n-features 20 \
        --output-dir results/splines \
        [--no-plot]        # skip matplotlib, just save CSV
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from kan_clt.kan_transcoder import KANCrossLayerTranscoder, load_kan_clt
from kan_clt.training.data import ActivationDataset


@torch.no_grad()
def collect_feature_stats(
    model: KANCrossLayerTranscoder,
    dataset: ActivationDataset,
    device: torch.device,
    dtype: torch.dtype,
    n_samples: int = 500,
) -> dict[str, torch.Tensor]:
    """Collect per-feature activation statistics across dataset.

    Returns:
        Dict with:
            mean_activation: (n_layers, d_transcoder) mean activation values.
            activation_frequency: (n_layers, d_transcoder) fraction of positions active.
            max_activation: (n_layers, d_transcoder) max observed activation.
    """
    n_layers = model.n_layers
    d_transcoder = model.d_transcoder
    n_samples = min(n_samples, len(dataset))

    sum_act = torch.zeros(n_layers, d_transcoder)
    sum_freq = torch.zeros(n_layers, d_transcoder)
    max_act = torch.zeros(n_layers, d_transcoder)
    n_pos_total = 0

    indices = torch.randperm(len(dataset))[:n_samples]
    for idx in indices:
        sample = dataset[int(idx)]
        x_in = sample["mlp_inputs"].to(device=device, dtype=dtype)
        acts = model.encode(x_in).float().cpu()  # (n_layers, seq_len, d_transcoder)

        n_pos = acts.shape[1]
        sum_act += acts.sum(dim=1)
        sum_freq += (acts > 0).float().sum(dim=1)
        max_act = torch.maximum(max_act, acts.max(dim=1).values)
        n_pos_total += n_pos

    return {
        "mean_activation": sum_act / n_pos_total,
        "activation_frequency": sum_freq / n_pos_total,
        "max_activation": max_act,
    }


@torch.no_grad()
def extract_spline_curve(
    model: KANCrossLayerTranscoder,
    layer_id: int,
    feature_id: int,
    input_dim: int,
    n_points: int = 200,
    x_range: tuple[float, float] = (-5.0, 5.0),
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the effective spline transfer function for one (feature, input_dim) pair.

    Creates probe inputs: e_j * t where e_j is the j-th standard basis vector and
    t sweeps over x_range. Returns (t_values, feature_output).

    Args:
        model: KAN-CLT model (encoder_type must be "kan").
        layer_id: Which encoder layer to probe.
        feature_id: Which output feature to trace.
        input_dim: Which input dimension to sweep.
        n_points: Number of sample points.
        x_range: Range of input values to sweep.

    Returns:
        (t_values, output_values) — both shape (n_points,) as numpy arrays.
    """
    if model.encoder_type != "kan":
        raise ValueError("Spline analysis only applies to KAN encoders.")

    if device is None:
        device = next(model.parameters()).device

    encoder = model.encoders[layer_id]
    d_model = model.d_model

    t = torch.linspace(x_range[0], x_range[1], n_points, device=device)
    # Probe inputs: zero everywhere except input_dim = t
    x_probe = torch.zeros(n_points, d_model, device=device)
    x_probe[:, input_dim] = t

    out = encoder(x_probe)  # (n_points, d_transcoder)
    return t.cpu().numpy(), out[:, feature_id].cpu().numpy()


def top_input_dims(
    model: KANCrossLayerTranscoder,
    layer_id: int,
    feature_id: int,
    top_k: int = 5,
) -> list[int]:
    """Find the input dimensions with the largest base_weight for a feature.

    Uses the base_weight (linear part of the KAN) to identify most influential dims.

    Args:
        model: KAN-CLT model.
        layer_id: Encoder layer.
        feature_id: Feature index.
        top_k: Number of top input dims to return.

    Returns:
        List of input dimension indices, sorted by weight magnitude descending.
    """
    encoder = model.encoders[layer_id]
    # base_weight: (n_features, d_model)
    base_w = encoder.kan_linear.base_weight[feature_id].abs()
    return base_w.topk(top_k).indices.tolist()


def save_curves_csv(
    output_dir: str,
    layer_id: int,
    feature_id: int,
    t_vals: np.ndarray,
    curves: dict[int, np.ndarray],
) -> None:
    """Save spline curves to CSV for downstream analysis."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"layer{layer_id}_feat{feature_id}.csv")
    dim_keys = sorted(curves.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t"] + [f"dim_{d}" for d in dim_keys])
        for i, t in enumerate(t_vals):
            writer.writerow([f"{t:.4f}"] + [f"{curves[d][i]:.6f}" for d in dim_keys])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze KAN spline shapes")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to KAN-CLT checkpoint directory")
    parser.add_argument("--data-dir", type=str, default="data/activations")
    parser.add_argument("--n-features", type=int, default=20,
                        help="Number of top features (by activation frequency) to analyze")
    parser.add_argument("--n-samples", type=int, default=300,
                        help="Dataset samples for computing feature statistics")
    parser.add_argument("--top-dims", type=int, default=3,
                        help="Number of top input dimensions to plot per feature")
    parser.add_argument("--output-dir", type=str, default="results/splines")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip matplotlib plotting, only save CSV files")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Loading checkpoint from {args.checkpoint}...")
    model = load_kan_clt(args.checkpoint, device=device, dtype=torch.float32)
    model.eval()

    if model.encoder_type != "kan":
        print(f"encoder_type={model.encoder_type!r} — spline analysis only applies to KAN encoders.")
        return

    print(f"  n_layers={model.n_layers}, d_model={model.d_model}, "
          f"d_transcoder={model.d_transcoder}, grid_size={model.grid_size}")

    print(f"\nLoading dataset from {args.data_dir}...")
    dataset = ActivationDataset.load(args.data_dir, max_samples=args.n_samples + 50)
    print(f"  {len(dataset)} samples loaded")

    print(f"\nCollecting feature statistics across {args.n_samples} sequences...")
    stats = collect_feature_stats(model, dataset, device, torch.float32, n_samples=args.n_samples)
    freq = stats["activation_frequency"]  # (n_layers, d_transcoder)
    mean_act = stats["mean_activation"]

    os.makedirs(args.output_dir, exist_ok=True)

    # Save summary CSV
    summary_path = os.path.join(args.output_dir, "feature_summary.csv")
    print(f"\nSaving feature summary to {summary_path}...")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "feature", "activation_frequency", "mean_activation", "max_activation"])
        for l in range(model.n_layers):
            for feat in range(model.d_transcoder):
                writer.writerow([
                    l, feat,
                    f"{freq[l, feat].item():.6f}",
                    f"{mean_act[l, feat].item():.6f}",
                    f"{stats['max_activation'][l, feat].item():.6f}",
                ])

    # Find top features across all layers by activation frequency
    flat_freq = freq.reshape(-1)
    top_indices = flat_freq.topk(args.n_features).indices
    top_features = [(int(idx // model.d_transcoder), int(idx % model.d_transcoder))
                    for idx in top_indices]

    print(f"\nAnalyzing top {args.n_features} most active features...")

    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            HAS_MPL = True
        except ImportError:
            print("  matplotlib not available — saving CSV only.")
            HAS_MPL = False
    else:
        HAS_MPL = False

    for rank, (layer_id, feature_id) in enumerate(top_features):
        f_freq = freq[layer_id, feature_id].item()
        print(f"  [{rank+1:2d}] Layer {layer_id}, Feature {feature_id:4d}  "
              f"freq={f_freq:.4f}", end="")

        # Find top input dimensions
        try:
            top_dims = top_input_dims(model, layer_id, feature_id, top_k=args.top_dims)
        except Exception as e:
            print(f" (skip: {e})")
            continue

        # Extract spline curves
        curves: dict[int, np.ndarray] = {}
        try:
            for dim in top_dims:
                t_vals, curve = extract_spline_curve(
                    model, layer_id, feature_id, dim, device=device
                )
                curves[dim] = curve
        except Exception as e:
            print(f" (curve error: {e})")
            continue

        # Save CSV
        save_curves_csv(args.output_dir, layer_id, feature_id, t_vals, curves)
        print(f"  top_dims={top_dims}")

        if HAS_MPL:
            fig, ax = plt.subplots(figsize=(8, 4))
            for dim, curve in curves.items():
                ax.plot(t_vals, curve, label=f"dim {dim}")
            ax.axhline(0, color="k", linewidth=0.5)
            ax.set_xlabel("Input value (along basis dimension)")
            ax.set_ylabel("KAN encoder output (pre-JumpReLU)")
            ax.set_title(f"Layer {layer_id}, Feature {feature_id}  [freq={f_freq:.3f}]")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig_path = os.path.join(args.output_dir, f"layer{layer_id}_feat{feature_id}.png")
            fig.savefig(fig_path, dpi=100)
            plt.close(fig)

    print(f"\nResults saved to {args.output_dir}/")
    print("  feature_summary.csv — activation stats for all features")
    print("  layer*_feat*.csv    — spline curves (t, dim_0, dim_1, ...)")
    if HAS_MPL:
        print("  layer*_feat*.png    — spline curve plots")


if __name__ == "__main__":
    main()
