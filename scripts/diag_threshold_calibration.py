"""Smoke-test data-quantile JumpReLU initialization on real activations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rebuttal_eval.common import load_val_dataset, sample_indices
from spline_clt.kan_transcoder import KANCrossLayerTranscoder
from spline_clt.seed import seed_everything
from spline_clt.training.data import compute_input_normalization
from spline_clt.training.train import initialize_thresholds_from_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-type", choices=("linear", "kan"), required=True)
    parser.add_argument("--activation-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-l0", type=float, default=32.0)
    parser.add_argument("--calibration-samples", type=int, default=16)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--values-per-sample", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    dataset = load_val_dataset(args.activation_dir)
    model = KANCrossLayerTranscoder(
        n_layers=12,
        d_model=768,
        d_transcoder=12288,
        encoder_type=args.encoder_type,
        grid_size=5,
        spline_order=3,
        activation_function="jump_relu",
        threshold_init=0.01,
        jumprelu_bandwidth=0.001,
        device=device,
        dtype=torch.float32,
    )
    norm_mean, norm_std = compute_input_normalization(
        dataset,
        n_layers=model.n_layers,
        d_model=model.d_model,
        n_sequences=args.calibration_samples,
        seed=args.seed,
    )
    model.set_input_normalization(norm_mean, norm_std)
    thresholds = initialize_thresholds_from_data(
        model,
        dataset,
        target_l0=args.target_l0,
        n_sequences=args.calibration_samples,
        values_per_sample=args.values_per_sample,
        seed=args.seed,
    )

    indices = sample_indices(len(dataset), args.eval_samples, args.seed + 10_000)
    active = torch.zeros(model.n_layers, device=device, dtype=torch.float64)
    positions = 0
    with torch.inference_mode():
        for index in indices:
            x = dataset[index]["mlp_inputs"].to(device=device, dtype=torch.float32)
            activations = model.encode(x)
            active += (activations > 0).sum(dim=(1, 2), dtype=torch.float64)
            positions += activations.shape[1]
    l0_per_layer = active / positions

    payload = {
        "encoder_type": args.encoder_type,
        "target_l0": args.target_l0,
        "calibration_samples": args.calibration_samples,
        "eval_samples": len(indices),
        "values_per_sample": args.values_per_sample,
        "layer_thresholds": thresholds.tolist(),
        "l0_per_layer": l0_per_layer.cpu().tolist(),
        "l0_layer_token": float(l0_per_layer.mean().cpu()),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.encoder_type}_threshold_calibration.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
