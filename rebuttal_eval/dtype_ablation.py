"""REQ-8: is float32 necessary at inference, or only at training?

Three conditions on one checkpoint, identical val subset:

- fp32:               whole model float32 (reference).
- bf16_body:          production inference config — transcoder body bf16, KAN
                      spline core pinned fp32 (kan_transcoder.py design).
- bf16_spline_weights: fp32 compute path, but every KAN encoder parameter
                      (base_weight, spline_weight, spline_scaler) round-tripped
                      through bf16 — i.e. what serving the spline core in bf16
                      precision would do to the numbers, without altering the
                      pinned-fp32 compute path. (True all-bf16 execution is not
                      supported by the module: the encoder forward hard-casts
                      to fp32; that pin exists for training-time grid updates.)

Emits per-condition reconstruction metrics and max abs deltas vs fp32.

Usage:
  python -m rebuttal_eval.dtype_ablation --checkpoint <ckpt> \
      --activation-dir <val_dir> --out-dir <dir> [--n-samples 128]
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import torch
from torch.nn import functional as F

from rebuttal_eval.common import (
    Provenance,
    emit,
    fmt,
    git_sha,
    load_transcoder,
    load_val_dataset,
    sample_indices,
)


@torch.no_grad()
def eval_condition(model, dataset, indices, device) -> dict[str, float]:
    mse_sum = 0.0
    cos_sum = 0.0
    sse = 0.0
    ssy = 0.0
    l0_sum = 0.0
    n = 0
    for idx in indices:
        sample = dataset[idx]
        x_in = sample["mlp_inputs"].to(device=device, dtype=model.b_dec.dtype)
        y_true = sample["mlp_outputs"].to(device=device, dtype=torch.float32)
        activations = model.encode(x_in)
        y_hat = model.decode_dense(activations, input_acts=x_in).float()
        mse_sum += ((y_hat - y_true) ** 2).mean().item()
        cos_sum += F.cosine_similarity(
            y_hat.reshape(-1, model.d_model), y_true.reshape(-1, model.d_model), dim=-1
        ).mean().item()
        sse += ((y_hat - y_true) ** 2).sum().item()
        ssy += (y_true**2).sum().item()
        l0_sum += (activations > 0).float().sum(dim=-1).mean().item()
        n += 1
    return {
        "mse_mean": mse_sum / n,
        "cosine_similarity": cos_sum / n,
        "nmse": sse / max(ssy, 1e-16),
        "l0_active_per_pos": l0_sum / n,
    }


def _roundtrip_spline_weights_bf16(model) -> int:
    """Round-trip every KAN encoder parameter through bf16; returns count."""
    changed = 0
    for encoder in model.encoders:
        kan_linear = getattr(encoder, "kan_linear", None)
        if kan_linear is None:
            continue
        for parameter in kan_linear.parameters():
            parameter.data = parameter.data.to(torch.bfloat16).to(parameter.dtype)
            changed += parameter.numel()
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--activation-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    dataset = load_val_dataset(args.activation_dir, split=args.split)
    indices = sample_indices(len(dataset), args.n_samples, args.seed)

    conditions: dict[str, dict[str, Any]] = {}

    model = load_transcoder(args.checkpoint, device=device, dtype=torch.float32)
    encoder_type = model.encoder_type
    conditions["fp32"] = eval_condition(model, dataset, indices, device)

    if encoder_type == "kan":
        changed = _roundtrip_spline_weights_bf16(model)
        conditions["bf16_spline_weights"] = eval_condition(
            model, dataset, indices, device
        )
        conditions["bf16_spline_weights"]["params_roundtripped"] = changed
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = load_transcoder(args.checkpoint, device=device, dtype=torch.bfloat16)
    conditions["bf16_body"] = eval_condition(model, dataset, indices, device)
    del model

    reference = conditions["fp32"]
    for name, row in conditions.items():
        row["max_abs_delta_vs_fp32"] = (
            max(
                abs(row[key] - reference[key])
                for key in ("mse_mean", "cosine_similarity", "nmse")
            )
            if name != "fp32"
            else 0.0
        )

    payload = {
        "git_sha": git_sha(),
        "checkpoint": args.checkpoint,
        "encoder_type": encoder_type,
        "n_samples": len(indices),
        "seed": args.seed,
        "conditions": conditions,
    }
    lines = [
        f"# fp32-vs-bf16 inference ablation (REQ-8) — {encoder_type}",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- {len(indices)} val samples (seed {args.seed}), git {git_sha()}",
        "",
        "| Condition | MSE | Cos | NMSE | L0 | max |Δ| vs fp32 |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in conditions.items():
        lines.append(
            f"| {name} | {fmt(row['mse_mean'])} | {fmt(row['cosine_similarity'])} "
            f"| {fmt(row['nmse'])} | {fmt(row['l0_active_per_pos'])} "
            f"| {fmt(row['max_abs_delta_vs_fp32'], 3)} |"
        )
    name = f"dtype_ablation_{args.label}" if args.label else "dtype_ablation"
    emit(args.out_dir, name, payload, "\n".join(lines) + "\n")

    provenance = Provenance(
        script="rebuttal_eval.dtype_ablation", checkpoint=args.checkpoint,
        seed=args.seed,
    )
    for cond, row in conditions.items():
        provenance.record("REQ-8", f"{cond}.nmse", row["nmse"])
    provenance.write(args.out_dir)
    print(f"wrote {name} for {len(conditions)} conditions -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
