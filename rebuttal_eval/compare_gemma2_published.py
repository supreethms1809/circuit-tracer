"""Compare a local Gemma-2 Spline-CLT checkpoint to published mntss linear CLT.

Runs both models on the **same** validation activation mmap subset and reports
side-by-side reconstruction + sparsity metrics (rel_fro, cosine, L0/layer/tok).

Published baseline: ``mntss/clt-gemma-2-2b-426k`` (d_t=16384, ReLU).

Usage:
  python -m rebuttal_eval.compare_gemma2_published \\
      --spline-checkpoint <ckpt_dir> \\
      --activation-dir <val/.../google_gemma-2-2b> \\
      --out-dir results/rebuttal/gemma2_vs_published \\
      [--hub-id mntss/clt-gemma-2-2b-426k] \\
      [--n-samples 256] [--seed 101] [--device cuda]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.nn import functional as F

from circuit_tracer.transcoder.cross_layer_transcoder import load_clt
from rebuttal_eval.common import (
    Provenance,
    emit,
    fmt,
    git_sha,
    load_transcoder,
    load_val_dataset,
    sample_indices,
)


def _rel_fro(y_hat: torch.Tensor, y_true: torch.Tensor) -> float:
    sse = float((y_hat - y_true).square().sum().item())
    ssy = float(y_true.square().sum().item())
    return math.sqrt(sse / max(ssy, 1e-16))


def _l0_per_layer_tok(activations: torch.Tensor) -> float:
    """Mean over (layer, position) of #{features > 0}."""
    return float((activations > 0).float().sum(dim=-1).mean().item())


def _score_predictor(
    indices: list[int],
    dataset,
    device: torch.device,
    predict_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, Any]:
    """predict_fn(x) -> (y_hat, activations), both float32 on device."""
    cos_vals: list[float] = []
    rel_pp: list[float] = []
    sample_rel: list[float] = []
    l0_sum = 0.0
    n_pos_total = 0
    sse = 0.0
    ssy = 0.0

    for idx in indices:
        sample = dataset[idx]
        x_in = sample["mlp_inputs"].to(device=device, dtype=torch.float32)
        y_true = sample["mlp_outputs"].to(device=device, dtype=torch.float32)
        n_pos = int(y_true.shape[1])
        with torch.no_grad():
            y_hat, acts = predict_fn(x_in)
            y_hat = y_hat.float()
            acts = acts.float()
        cos = F.cosine_similarity(y_hat, y_true, dim=-1)
        err = (y_hat - y_true).norm(dim=-1)
        y_n = y_true.norm(dim=-1).clamp(min=1e-8)
        cos_vals.extend(cos.reshape(-1).cpu().tolist())
        rel_pp.extend((err / y_n).reshape(-1).cpu().tolist())
        sample_rel.append(_rel_fro(y_hat, y_true))
        l0_sum += _l0_per_layer_tok(acts) * n_pos
        n_pos_total += n_pos
        sse += float((y_hat - y_true).square().sum().item())
        ssy += float(y_true.square().sum().item())

    cos_arr = np.asarray(cos_vals, dtype=np.float64)
    return {
        "cos_mean": float(cos_arr.mean()),
        "frac_cos_negative": float((cos_arr < 0).mean()),
        "frac_cos_above_0p9": float((cos_arr > 0.9).mean()),
        "rel_err_per_position_mean": float(np.mean(rel_pp)),
        "rel_err_paper_form_mean": float(np.mean(sample_rel)),
        "rel_err_global_frobenius": float(math.sqrt(sse / max(ssy, 1e-16))),
        "l0_active_per_layer_tok": float(l0_sum / max(n_pos_total, 1)),
        "n_samples": len(indices),
        "n_pos_total": n_pos_total,
    }


def _spline_predict(model) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    def _fn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        acts = model.encode(x)
        y_hat = model.decode_dense(acts, input_acts=x)
        return y_hat, acts

    return _fn


def _hub_predict(clt) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    def _fn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Hub CLT is typically bf16; metrics cast to float32 in the scorer.
        x_h = x.to(dtype=clt.b_enc.dtype)
        acts = clt.encode(x_h)
        y_hat = clt.forward(x_h)
        return y_hat, acts

    return _fn


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    dataset = load_val_dataset(args.activation_dir, split=args.split)
    indices = sample_indices(len(dataset), args.n_samples, args.seed)

    spline = load_transcoder(args.spline_checkpoint, device=device, dtype=torch.float32)
    spline_stats = _score_predictor(indices, dataset, device, _spline_predict(spline))
    del spline
    if device.type == "cuda":
        torch.cuda.empty_cache()

    hub_path = snapshot_download(args.hub_id, local_files_only=not args.allow_hub_download)
    hub = load_clt(
        hub_path,
        device=device,
        dtype=torch.bfloat16,
        lazy_decoder=True,
        lazy_encoder=False,
    )
    hub.eval()
    hub_stats = _score_predictor(indices, dataset, device, _hub_predict(hub))
    del hub
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def _delta(key: str) -> float:
        return float(spline_stats[key] - hub_stats[key])

    payload: dict[str, Any] = {
        "meta": {
            "script": "rebuttal_eval.compare_gemma2_published",
            "git_sha": git_sha(),
            "spline_checkpoint": str(args.spline_checkpoint),
            "hub_id": args.hub_id,
            "hub_snapshot": hub_path,
            "activation_dir": str(args.activation_dir),
            "n_samples": args.n_samples,
            "seed": args.seed,
            "device": str(device),
            "label": args.label,
            "published_l0_layer_tok_banked": 12.6,
        },
        "spline": {
            **spline_stats,
            "encoder_type": "kan",
            "d_transcoder": "from_checkpoint",
        },
        "published_linear": {
            **hub_stats,
            "encoder_type": "linear_relu",
            "d_transcoder": 16384,
            "hub_id": args.hub_id,
        },
        "delta_spline_minus_published": {
            "cos_mean": _delta("cos_mean"),
            "rel_err_paper_form_mean": _delta("rel_err_paper_form_mean"),
            "rel_err_global_frobenius": _delta("rel_err_global_frobenius"),
            "l0_active_per_layer_tok": _delta("l0_active_per_layer_tok"),
        },
    }
    return payload


def _render_markdown(payload: dict[str, Any]) -> str:
    s = payload["spline"]
    p = payload["published_linear"]
    d = payload["delta_spline_minus_published"]
    m = payload["meta"]
    lines = [
        "# Gemma-2 Spline-CLT vs published mntss linear CLT",
        "",
        f"- spline: `{m['spline_checkpoint']}`",
        f"- published: `{m['hub_id']}` (d_t=16384)",
        f"- activation val: `{m['activation_dir']}`",
        f"- n_samples={m['n_samples']}, seed={m['seed']}",
        f"- banked published L0/layer/tok (wikitext): {m['published_l0_layer_tok_banked']}",
        "",
        "| Metric | Spline (ours) | Published linear | Δ (ours − pub) |",
        "|---|---:|---:|---:|",
        f"| cos_mean | {fmt(s['cos_mean'])} | {fmt(p['cos_mean'])} | {fmt(d['cos_mean'])} |",
        f"| rel_err (paper form) | {fmt(s['rel_err_paper_form_mean'])} | {fmt(p['rel_err_paper_form_mean'])} | {fmt(d['rel_err_paper_form_mean'])} |",
        f"| rel_err (global Fro) | {fmt(s['rel_err_global_frobenius'])} | {fmt(p['rel_err_global_frobenius'])} | {fmt(d['rel_err_global_frobenius'])} |",
        f"| L0 / layer / tok | {fmt(s['l0_active_per_layer_tok'])} | {fmt(p['l0_active_per_layer_tok'])} | {fmt(d['l0_active_per_layer_tok'])} |",
        f"| frac cos < 0 | {fmt(s['frac_cos_negative'])} | {fmt(p['frac_cos_negative'])} | — |",
        f"| frac cos > 0.9 | {fmt(s['frac_cos_above_0p9'])} | {fmt(p['frac_cos_above_0p9'])} | — |",
        "",
        "Lower rel_err / higher cos is better reconstruction. L0 is compared on the "
        "same activation cache (not wikitext); the banked 12.6 figure is the "
        "published sparsity anchor from `measure_ref_clt_l0`.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spline-checkpoint", required=True)
    parser.add_argument("--activation-dir", required=True)
    parser.add_argument("--hub-id", default="mntss/clt-gemma-2-2b-426k")
    parser.add_argument("--split", default="val")
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--label", default="gemma2_dt6144_vs_mntss")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--allow-hub-download",
        action="store_true",
        help="Permit HF download if the hub snapshot is not cached locally.",
    )
    args = parser.parse_args(argv)

    payload = run_compare(args)
    name = f"compare_gemma2_published_{args.label}"
    emit(args.out_dir, name, payload, _render_markdown(payload))

    provenance = Provenance(
        script="rebuttal_eval.compare_gemma2_published",
        checkpoint=str(args.spline_checkpoint),
        seed=args.seed,
    )
    for arm in ("spline", "published_linear"):
        for key in (
            "cos_mean",
            "rel_err_paper_form_mean",
            "l0_active_per_layer_tok",
        ):
            provenance.record(arm, key, payload[arm][key])
    provenance.write(args.out_dir)

    print(_render_markdown(payload))
    print(f"wrote {name}.json/.md to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
