"""Blocking validation checks §4.1–4.3 from paper_rebuttal_todo.md (REQ-6).

One model pass over a deterministic val subset per checkpoint, emitting:

§4.1  per-position cosine distribution (deciles, frac<0, frac>0.9, joint with
      ||y|| deciles) and the rel_err >= 1 - cos consistency check, computed in
      BOTH forms: per-position rel_err (the form the inequality strictly
      applies to) and the paper's per-sample global-Frobenius rel_err.
§4.2  metric mask identity report (all metrics share the full tensor; BOS/
      position-0 handling stated), plus a position-0-excluded recomputation.
§4.3  trivial baselines on the same mask/samples: y_hat = 0, per-layer dataset
      mean, b_dec-only (all features zeroed), and the actual model; variance
      explained vs the mean predictor; per-layer ||b_dec|| vs feature
      contribution norm split.

Exit status is nonzero iff the per-position §4.1 inequality fails (spec
§0.3.5: stop and report, do not work around).

Usage:
  python -m rebuttal_eval.check_reconstruction \
      --checkpoint <ckpt_dir> --activation-dir <val_dir> --out-dir <dir> \
      [--n-samples 256] [--seed 101] [--device cuda] [--label spline_fm]

  python -m rebuttal_eval.check_reconstruction --compare old.json new.json \
      --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
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

TOL = 1e-3


def _deciles(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, q / 10)) for q in range(11)]


def _row_stats(y_hat: torch.Tensor, y_true: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-(layer, position) cosine and relative error for one sample.

    Both tensors are (n_layers, n_pos, d_model); outputs are (n_layers, n_pos).
    """
    cos = F.cosine_similarity(y_hat, y_true, dim=-1)
    err_norm = (y_hat - y_true).norm(dim=-1)
    y_norm = y_true.norm(dim=-1)
    rel = err_norm / y_norm.clamp(min=1e-8)
    return {"cos": cos, "rel": rel, "y_norm": y_norm}


class _PredictorAccumulator:
    """Accumulates per-row stats and sums for one predictor across samples."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.cos_rows: list[np.ndarray] = []      # each (n_layers, n_pos)
        self.rel_rows: list[np.ndarray] = []
        self.sse_per_layer = np.zeros(n_layers, dtype=np.float64)
        self.n_elems_per_layer = np.zeros(n_layers, dtype=np.float64)
        self.sample_rel_fro: list[float] = []     # paper-form rel err per sample

    def add(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> None:
        stats = _row_stats(y_hat, y_true)
        self.cos_rows.append(stats["cos"].cpu().numpy().astype(np.float32))
        self.rel_rows.append(stats["rel"].cpu().numpy().astype(np.float32))
        err_sq = (y_hat - y_true).square()
        self.sse_per_layer += err_sq.sum(dim=(1, 2)).double().cpu().numpy()
        self.n_elems_per_layer += float(y_true.shape[1] * y_true.shape[2])
        sse = float(err_sq.sum().item())
        ssy = float(y_true.square().sum().item())
        self.sample_rel_fro.append((sse / max(ssy, 1e-16)) ** 0.5)

    def summary(
        self, ssy_per_layer: np.ndarray, ss_centered_per_layer: np.ndarray
    ) -> dict[str, Any]:
        cos = np.concatenate([r.reshape(-1) for r in self.cos_rows])
        rel = np.concatenate([r.reshape(-1) for r in self.rel_rows])
        mse_per_layer = self.sse_per_layer / self.n_elems_per_layer
        var_explained_layer = 1.0 - self.sse_per_layer / np.maximum(
            ss_centered_per_layer, 1e-16
        )
        return {
            "cos_mean": float(cos.mean()),
            "rel_err_per_position_mean": float(rel.mean()),
            "rel_err_global_frobenius": float(
                (self.sse_per_layer.sum() / max(ssy_per_layer.sum(), 1e-16)) ** 0.5
            ),
            "rel_err_paper_form_mean_over_samples": float(
                np.mean(self.sample_rel_fro)
            ),
            "mse_mean": float(mse_per_layer.mean()),
            "mse_per_layer": [float(v) for v in mse_per_layer],
            "variance_explained_aggregate": float(
                1.0 - self.sse_per_layer.sum() / max(ss_centered_per_layer.sum(), 1e-16)
            ),
            "variance_explained_per_layer": [float(v) for v in var_explained_layer],
        }


def run_checks(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    model = load_transcoder(args.checkpoint, device=device, dtype=torch.float32)
    dataset = load_val_dataset(args.activation_dir, split=args.split)
    indices = sample_indices(len(dataset), args.n_samples, args.seed)
    n_layers = model.n_layers

    # Pass 1 (no model): per-layer dataset mean over the identical sample
    # subset, plus sums of squares for variance-explained denominators.
    mean_sum: torch.Tensor | None = None
    ssy_per_layer = np.zeros(n_layers, dtype=np.float64)
    n_rows_per_layer = 0
    for idx in indices:
        y = dataset[idx]["mlp_outputs"].to(device=device, dtype=torch.float32)
        mean_sum = y.sum(dim=1) if mean_sum is None else mean_sum + y.sum(dim=1)
        ssy_per_layer += y.square().sum(dim=(1, 2)).double().cpu().numpy()
        n_rows_per_layer += y.shape[1]
    assert mean_sum is not None, "empty sample subset"
    y_mean = mean_sum / n_rows_per_layer  # (n_layers, d_model)

    # Pass 2: model forward + all predictors on the same rows.
    predictors = {
        "zero": _PredictorAccumulator(n_layers),
        "dataset_mean": _PredictorAccumulator(n_layers),
        "b_dec_only": _PredictorAccumulator(n_layers),
        "model": _PredictorAccumulator(n_layers),
    }
    ss_centered_per_layer = np.zeros(n_layers, dtype=np.float64)
    y_norm_rows: list[np.ndarray] = []
    l0_total = 0.0
    feature_contrib_norm_sum = np.zeros(n_layers, dtype=np.float64)
    n_pos_total = 0
    has_skip = model.W_skip is not None

    b_dec = model.b_dec.detach()  # (n_layers, d_model)
    for idx in indices:
        sample = dataset[idx]
        x_in = sample["mlp_inputs"].to(device=device, dtype=torch.float32)
        y_true = sample["mlp_outputs"].to(device=device, dtype=torch.float32)
        n_pos = y_true.shape[1]

        with torch.no_grad():
            activations = model.encode(x_in)
            y_hat = model.decode_dense(activations, input_acts=x_in)

        y_mean_exp = y_mean[:, None, :].expand_as(y_true)
        b_dec_exp = b_dec[:, None, :].expand_as(y_true)
        predictors["zero"].add(torch.zeros_like(y_true), y_true)
        predictors["dataset_mean"].add(y_mean_exp, y_true)
        predictors["b_dec_only"].add(b_dec_exp, y_true)
        predictors["model"].add(y_hat, y_true)

        ss_centered_per_layer += (
            (y_true - y_mean_exp).square().sum(dim=(1, 2)).double().cpu().numpy()
        )
        y_norm_rows.append(y_true.norm(dim=-1).cpu().numpy().astype(np.float32))
        l0_total += float((activations > 0).float().sum(dim=-1).mean().item()) * n_pos
        # Feature contribution isolated by subtracting the b_dec (+skip) floor.
        floor = b_dec_exp + (x_in @ model.W_skip if has_skip else 0.0)
        feature_contrib_norm_sum += (
            (y_hat - floor).norm(dim=-1).sum(dim=1).double().cpu().numpy()
        )
        n_pos_total += n_pos

    model_acc = predictors["model"]
    cos_all = np.concatenate([r.reshape(-1) for r in model_acc.cos_rows])
    rel_all = np.concatenate([r.reshape(-1) for r in model_acc.rel_rows])
    y_norm_all = np.concatenate([r.reshape(-1) for r in y_norm_rows])
    cos_by_layer = np.concatenate(model_acc.cos_rows, axis=1)  # (n_layers, total_pos)

    # §4.1 inequalities. The spec bound E[rel] >= 1 - E[cos] is only valid
    # when every per-position cosine is non-negative (1 - c is the lower
    # convex envelope of sqrt(1 - c^2) on [0, 1] only). The universally valid
    # per-position bound is rel >= sqrt(1 - c^2), and additionally rel >= 1
    # wherever c < 0. A metric bug exists iff the universal bound fails, or
    # the spec bound fails with (essentially) no negative cosines present;
    # a spec-bound failure WITH negative cosines is the anti-aligned
    # undertrained-model diagnosis, not a bug.
    cos_mean = float(cos_all.mean())
    rel_pp_mean = float(rel_all.mean())
    rel_paper_mean = float(np.mean(model_acc.sample_rel_fro))
    frac_neg = float((cos_all < 0).mean())
    universal_bound = np.sqrt(np.clip(1.0 - cos_all**2, 0.0, None))
    universal_bound = np.where(cos_all < 0, np.maximum(universal_bound, 1.0), universal_bound)
    universal_ok = rel_pp_mean >= float(universal_bound.mean()) - TOL
    spec_bound_ok = rel_pp_mean >= (1.0 - cos_mean) - TOL
    per_position_ok = universal_ok and (spec_bound_ok or frac_neg > 1e-4)
    paper_form_ok = rel_paper_mean >= (1.0 - cos_mean) - TOL

    # Joint distribution: cosine stats binned by ||y|| decile.
    decile_edges = np.quantile(y_norm_all, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(decile_edges[1:-1], y_norm_all), 0, 9)
    joint = [
        {
            "y_norm_decile": bin_id,
            "y_norm_range": [float(decile_edges[bin_id]), float(decile_edges[bin_id + 1])],
            "cos_mean": float(cos_all[bins == bin_id].mean()),
            "frac_cos_negative": float((cos_all[bins == bin_id] < 0).mean()),
        }
        for bin_id in range(10)
    ]

    # §4.2: position-0-excluded recomputation (drop first position per window).
    cos_no0 = np.concatenate([r[:, 1:].reshape(-1) for r in model_acc.cos_rows])
    rel_no0 = np.concatenate([r[:, 1:].reshape(-1) for r in model_acc.rel_rows])

    ssy = ssy_per_layer
    ssc = ss_centered_per_layer
    payload: dict[str, Any] = {
        "meta": {
            "checkpoint": str(args.checkpoint),
            "activation_dir": str(args.activation_dir),
            "label": args.label,
            "encoder_type": model.encoder_type,
            "n_layers": n_layers,
            "d_transcoder": model.d_transcoder,
            "d_model": model.d_model,
            "n_samples": len(indices),
            "n_positions_total": n_pos_total,
            "seed": args.seed,
            "git_sha": git_sha(),
            "l0_active_per_pos": l0_total / max(n_pos_total, 1),
            "has_skip": has_skip,
        },
        "section_4_1": {
            "cos_deciles": _deciles(cos_all),
            "frac_cos_negative": float((cos_all < 0).mean()),
            "frac_cos_above_0p9": float((cos_all > 0.9).mean()),
            "cos_mean": cos_mean,
            "rel_err_per_position_mean": rel_pp_mean,
            "rel_err_paper_form_mean_over_samples": rel_paper_mean,
            "inequality_per_position": {
                "lhs_rel_err_mean": rel_pp_mean,
                "rhs_one_minus_cos_mean": 1.0 - cos_mean,
                "rhs_universal_bound_mean": float(universal_bound.mean()),
                "spec_bound_pass": bool(spec_bound_ok),
                "universal_bound_pass": bool(universal_ok),
                "frac_cos_negative": frac_neg,
                "pass": bool(per_position_ok),
                "note": (
                    "spec bound (1 - E[cos]) is only valid when all per-position "
                    "cosines are >= 0; a spec-bound failure with negative "
                    "cosines present indicates anti-aligned reconstructions "
                    "(undertrained model), not a metric bug. Blocking iff the "
                    "universal bound E[max(sqrt(1-c^2), 1_{c<0})] fails, or the "
                    "spec bound fails with no negative cosines."
                ),
            },
            "inequality_paper_form": {
                "lhs_rel_err_mean": rel_paper_mean,
                "rhs_one_minus_cos_mean": 1.0 - cos_mean,
                "pass": bool(paper_form_ok),
                "note": (
                    "The paper's rel_err is a per-sample global Frobenius ratio "
                    "(norm-weighted quadratic mean over positions) while cos_mean "
                    "is position-uniform; the bound only strictly applies to the "
                    "per-position form, so a failure here alone indicates "
                    "norm-weighting, not necessarily a metric bug."
                ),
            },
            "per_layer": [
                {
                    "layer": layer_id,
                    "cos_mean": float(cos_by_layer[layer_id].mean()),
                    "frac_cos_negative": float((cos_by_layer[layer_id] < 0).mean()),
                    "frac_cos_above_0p9": float((cos_by_layer[layer_id] > 0.9).mean()),
                }
                for layer_id in range(n_layers)
            ],
            "joint_cos_by_y_norm_decile": joint,
        },
        "section_4_2": {
            "mask_definition": (
                "All metrics (MSE, relative error, cosine) are computed over the "
                "identical full (n_layers, n_pos, d_model) tensor per sample: no "
                "mask, no padding (fixed-length token windows), position 0 "
                "included. Windows are contiguous corpus slices without a "
                "prepended BOS token; position 0 is the attention-sink position "
                "of each window."
            ),
            "n_rows": int(cos_all.size),
            "position_0_excluded": {
                "cos_mean": float(cos_no0.mean()),
                "frac_cos_negative": float((cos_no0 < 0).mean()),
                "rel_err_per_position_mean": float(rel_no0.mean()),
            },
        },
        "section_4_3": {
            "predictors": {
                name: acc.summary(ssy, ssc) for name, acc in predictors.items()
            },
            "contribution_split_per_layer": [
                {
                    "layer": layer_id,
                    "b_dec_norm": float(b_dec[layer_id].norm().item()),
                    "mean_feature_contribution_norm": float(
                        feature_contrib_norm_sum[layer_id] / max(n_pos_total, 1)
                    ),
                }
                for layer_id in range(n_layers)
            ],
        },
        "status": "PASS" if per_position_ok else "FAIL",
    }
    return payload


def _render_markdown(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    s41 = payload["section_4_1"]
    s42 = payload["section_4_2"]
    s43 = payload["section_4_3"]
    lines = [
        f"# Reconstruction checks — {meta['label'] or meta['encoder_type']}",
        "",
        f"- checkpoint: `{meta['checkpoint']}`",
        f"- encoder: {meta['encoder_type']}, d_t={meta['d_transcoder']}, "
        f"L={meta['n_layers']}, d_model={meta['d_model']}",
        f"- samples: {meta['n_samples']} (seed {meta['seed']}), "
        f"positions: {meta['n_positions_total']}, "
        f"L0 (active/pos): {fmt(meta['l0_active_per_pos'])}",
        f"- git: {meta['git_sha']}  |  status: **{payload['status']}**",
        "",
        "## §4.1 Cosine / relative-error consistency",
        "",
        f"- cos mean: {fmt(s41['cos_mean'])}  |  frac cos<0: "
        f"{fmt(s41['frac_cos_negative'])}  |  frac cos>0.9: "
        f"{fmt(s41['frac_cos_above_0p9'])}",
        f"- cos deciles: {', '.join(fmt(v, 3) for v in s41['cos_deciles'])}",
        f"- per-position: rel_err {fmt(s41['inequality_per_position']['lhs_rel_err_mean'])} "
        f"vs spec bound 1-cos {fmt(s41['inequality_per_position']['rhs_one_minus_cos_mean'])} "
        f"({'ok' if s41['inequality_per_position']['spec_bound_pass'] else 'violated'}), "
        f"vs universal bound {fmt(s41['inequality_per_position']['rhs_universal_bound_mean'])} "
        f"({'ok' if s41['inequality_per_position']['universal_bound_pass'] else 'violated'}) "
        f"-> {'PASS' if s41['inequality_per_position']['pass'] else 'FAIL'}",
        f"- paper form (per-sample Frobenius): rel_err "
        f"{fmt(s41['inequality_paper_form']['lhs_rel_err_mean'])} "
        f">= 1-cos {fmt(s41['inequality_paper_form']['rhs_one_minus_cos_mean'])} "
        f"-> {'PASS' if s41['inequality_paper_form']['pass'] else 'FAIL'}",
        "",
        "| ||y|| decile | cos mean | frac cos<0 |",
        "|---|---|---|",
    ]
    for row in s41["joint_cos_by_y_norm_decile"]:
        lines.append(
            f"| {row['y_norm_decile']} "
            f"[{fmt(row['y_norm_range'][0], 3)}, {fmt(row['y_norm_range'][1], 3)}] "
            f"| {fmt(row['cos_mean'])} | {fmt(row['frac_cos_negative'])} |"
        )
    lines += [
        "",
        "| layer | cos mean | frac<0 | frac>0.9 |",
        "|---|---|---|---|",
    ]
    for row in s41["per_layer"]:
        lines.append(
            f"| {row['layer']} | {fmt(row['cos_mean'])} | "
            f"{fmt(row['frac_cos_negative'])} | {fmt(row['frac_cos_above_0p9'])} |"
        )
    lines += [
        "",
        "## §4.2 Metric mask identity",
        "",
        s42["mask_definition"],
        "",
        f"- N rows: {s42['n_rows']}",
        f"- position 0 excluded: cos mean {fmt(s42['position_0_excluded']['cos_mean'])}, "
        f"frac<0 {fmt(s42['position_0_excluded']['frac_cos_negative'])}, "
        f"per-pos rel err {fmt(s42['position_0_excluded']['rel_err_per_position_mean'])}",
        "",
        "## §4.3 Trivial baselines (identical mask and samples)",
        "",
        "| Predictor | MSE | Rel err (global) | Rel err (per-pos) | Cos | VarExpl |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("zero", "dataset_mean", "b_dec_only", "model"):
        row = s43["predictors"][name]
        lines.append(
            f"| {name} | {fmt(row['mse_mean'])} | "
            f"{fmt(row['rel_err_global_frobenius'])} | "
            f"{fmt(row['rel_err_per_position_mean'])} | {fmt(row['cos_mean'])} | "
            f"{fmt(row['variance_explained_aggregate'])} |"
        )
    lines += [
        "",
        "| layer | ||b_dec|| | mean ||feature contribution|| |",
        "|---|---|---|",
    ]
    for row in s43["contribution_split_per_layer"]:
        lines.append(
            f"| {row['layer']} | {fmt(row['b_dec_norm'])} | "
            f"{fmt(row['mean_feature_contribution_norm'])} |"
        )
    return "\n".join(lines) + "\n"


def _render_compare(old: dict[str, Any], new: dict[str, Any]) -> str:
    """Submitted-vs-converged delta table (§4.1 compare mode)."""
    def pick(payload: dict[str, Any]) -> dict[str, float]:
        s41 = payload["section_4_1"]
        s43 = payload["section_4_3"]["predictors"]["model"]
        return {
            "cos_mean": s41["cos_mean"],
            "frac_cos_negative": s41["frac_cos_negative"],
            "frac_cos_above_0p9": s41["frac_cos_above_0p9"],
            "rel_err_paper_form": s41["rel_err_paper_form_mean_over_samples"],
            "mse_mean": s43["mse_mean"],
            "l0": payload["meta"]["l0_active_per_pos"],
        }

    old_stats, new_stats = pick(old), pick(new)
    lines = [
        "# Submitted vs converged — reconstruction diagnosis",
        "",
        f"- submitted: `{old['meta']['checkpoint']}`",
        f"- converged: `{new['meta']['checkpoint']}`",
        "",
        "| Metric | Submitted | Converged |",
        "|---|---|---|",
    ]
    for key in old_stats:
        lines.append(f"| {key} | {fmt(old_stats[key])} | {fmt(new_stats[key])} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--activation-dir")
    parser.add_argument("--split", default="val")
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--label", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--compare", nargs=2, metavar=("OLD_JSON", "NEW_JSON"),
        help="Render a submitted-vs-converged delta from two result JSONs",
    )
    args = parser.parse_args(argv)

    if args.compare:
        old = json.loads(Path(args.compare[0]).read_text())
        new = json.loads(Path(args.compare[1]).read_text())
        emit(args.out_dir, "compare_submitted_vs_converged", {"old": old["meta"], "new": new["meta"]},
             _render_compare(old, new))
        print(f"wrote compare table to {args.out_dir}")
        return 0

    if not args.checkpoint or not args.activation_dir:
        parser.error("--checkpoint and --activation-dir are required unless --compare")

    payload = run_checks(args)
    name = f"check_reconstruction_{args.label}" if args.label else "check_reconstruction"
    emit(args.out_dir, name, payload, _render_markdown(payload))

    provenance = Provenance(
        script="rebuttal_eval.check_reconstruction",
        checkpoint=str(args.checkpoint),
        seed=args.seed,
    )
    for key in ("cos_mean", "frac_cos_negative", "rel_err_paper_form_mean_over_samples"):
        provenance.record("4.1", key, payload["section_4_1"][key])
    for pred, row in payload["section_4_3"]["predictors"].items():
        provenance.record("4.3", f"{pred}.mse_mean", row["mse_mean"])
    provenance.write(args.out_dir)

    print(f"[{payload['status']}] wrote {name}.json/.md to {args.out_dir}")
    if payload["status"] == "FAIL":
        print(
            "BLOCKING FAILURE §4.1: per-position rel_err mean violates "
            "rel_err >= 1 - cos. Stop and report (spec §0.3.5); do not format "
            "tables on top of this.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
