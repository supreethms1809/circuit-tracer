"""Compare gate margins and L0 in fp32 versus bf16 on a fixed holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rebuttal_eval.common import load_transcoder, load_val_dataset, sample_indices


def measure(
    checkpoint: str,
    activation_dir: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
    n_samples: int,
    seed: int,
) -> dict:
    """Measure preactivation margins and active counts at one precision."""
    model = load_transcoder(checkpoint, device=device, dtype=dtype)
    dataset = load_val_dataset(activation_dir)
    indices = sample_indices(len(dataset), n_samples, seed)

    active = torch.zeros(model.n_layers, device=device, dtype=torch.float64)
    positions = torch.zeros(model.n_layers, device=device, dtype=torch.float64)
    preact_sum = torch.zeros(model.n_layers, device=device, dtype=torch.float64)
    preact_sq_sum = torch.zeros(model.n_layers, device=device, dtype=torch.float64)
    preact_count = torch.zeros(model.n_layers, device=device, dtype=torch.float64)
    windows = (0.001, 0.01, 0.1)
    near = {
        width: torch.zeros(model.n_layers, device=device, dtype=torch.float64)
        for width in windows
    }

    with torch.inference_mode():
        for index in indices:
            x = dataset[index]["mlp_inputs"].to(device=device, dtype=dtype)
            for layer_id in range(model.n_layers):
                z = model.encode_layer(
                    x[layer_id], layer_id, apply_activation_function=False
                ).float()
                theta = model.effective_threshold[layer_id].float()
                margin = z - theta
                active[layer_id] += (margin > 0).sum(dtype=torch.float64)
                positions[layer_id] += z.shape[0]
                preact_sum[layer_id] += z.sum(dtype=torch.float64)
                preact_sq_sum[layer_id] += z.square().sum(dtype=torch.float64)
                preact_count[layer_id] += z.numel()
                for width in windows:
                    near[width][layer_id] += (
                        margin.abs() < width
                    ).sum(dtype=torch.float64)

    mean = preact_sum / preact_count
    variance = (preact_sq_sum / preact_count - mean.square()).clamp_min(0)
    l0_per_layer = active / positions
    result = {
        "dtype": str(dtype).removeprefix("torch."),
        "n_samples": len(indices),
        "l0_per_layer": l0_per_layer.cpu().tolist(),
        "l0_layer_token": float(l0_per_layer.mean().cpu()),
        "preactivation_mean_per_layer": mean.cpu().tolist(),
        "preactivation_std_per_layer": variance.sqrt().cpu().tolist(),
        "near_gate_fraction": {
            str(width): (near[width] / preact_count).cpu().tolist()
            for width in windows
        },
        "threshold_mean": float(model.effective_threshold.float().mean().cpu()),
        "threshold_median": float(model.effective_threshold.float().median().cpu()),
        "threshold_unique": int(model.effective_threshold.float().unique().numel()),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--activation-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "activation_dir": args.activation_dir,
        "float32": measure(
            args.checkpoint,
            args.activation_dir,
            dtype=torch.float32,
            device=device,
            n_samples=args.n_samples,
            seed=args.seed,
        ),
        "bfloat16": measure(
            args.checkpoint,
            args.activation_dir,
            dtype=torch.bfloat16,
            device=device,
            n_samples=args.n_samples,
            seed=args.seed,
        ),
    }
    payload["l0_bf16_minus_fp32"] = (
        payload["bfloat16"]["l0_layer_token"]
        - payload["float32"]["l0_layer_token"]
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.label}_gate_margins.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
