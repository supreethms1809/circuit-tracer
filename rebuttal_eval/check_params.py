"""§4.4 parameter-count reconciliation (also REQ-11 pairing evidence).

Reads checkpoint safetensors directly (no model construction, CPU-only) and
reconciles actual parameter counts against two analytic encoder formulas:

- three-term (base weight + spline coefficients + per-edge spline scaler):
    enc_per_feature = d_model * (2 + G + k)
- two-term (paper Eq. 13, no scaler):
    enc_per_feature = d_model * (1 + G + k)

Buffers (KAN grid, enc_input_mean/std) are excluded from parameter counts.
Linear checkpoints reconcile against enc_per_feature = d_model.

Usage:
  python -m rebuttal_eval.check_params --checkpoints CKPT [CKPT ...] \
      --out-dir <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from safetensors import safe_open

from rebuttal_eval.common import Provenance, emit, git_sha

#: state-dict key suffix -> reporting bucket. Keys not matched here are
#: reported under "unrecognized" so nothing is silently dropped.
_BUFFER_SUFFIXES = ("grid",)


def _numel(shape: list[int]) -> int:
    n = 1
    for dim in shape:
        n *= dim
    return n


def inspect_checkpoint(ckpt_dir: Path) -> dict[str, Any]:
    meta_path = ckpt_dir / "metadata.safetensors"
    with safe_open(str(meta_path), framework="pt") as handle:
        keys = handle.keys()
        meta = {
            "n_layers": int(handle.get_tensor("n_layers").item()),
            "d_transcoder": int(handle.get_tensor("d_transcoder").item()),
            "d_model": int(handle.get_tensor("d_model").item()),
            "grid_size": int(handle.get_tensor("grid_size").item()),
            "spline_order": int(handle.get_tensor("spline_order").item()),
            "encoder_type": (
                "linear"
                if "encoder_type_linear" in keys
                and bool(handle.get_tensor("encoder_type_linear").item())
                else "kan"
            ),
        }

    buckets: dict[str, int] = {
        "base_weight": 0,
        "spline_weight": 0,
        "spline_scaler": 0,
        "W_enc": 0,
        "W_dec": 0,
        "b_enc": 0,
        "b_dec": 0,
        "threshold": 0,
        "buffers_excluded": 0,
        "unrecognized": 0,
    }
    unrecognized_keys: list[str] = []
    n_layers = meta["n_layers"]
    for layer_id in range(n_layers):
        for file_name in (f"encoder_{layer_id}.safetensors", f"W_dec_{layer_id}.safetensors"):
            with safe_open(str(ckpt_dir / file_name), framework="pt") as handle:
                for key in handle.keys():
                    numel = _numel(handle.get_slice(key).get_shape())
                    leaf = key.split(".")[-1]
                    if leaf in _BUFFER_SUFFIXES or leaf.startswith("enc_input_"):
                        buckets["buffers_excluded"] += numel
                    elif leaf in ("base_weight", "spline_weight", "spline_scaler", "W_enc"):
                        buckets[leaf] += numel
                    elif key.startswith("b_enc_"):
                        buckets["b_enc"] += numel
                    elif key.startswith("b_dec_"):
                        buckets["b_dec"] += numel
                    elif key.startswith("enc_input_"):
                        buckets["buffers_excluded"] += numel
                    elif key.startswith("threshold_"):
                        buckets["threshold"] += numel
                    elif key.startswith("W_dec_"):
                        buckets["W_dec"] += numel
                    else:
                        buckets["unrecognized"] += numel
                        unrecognized_keys.append(key)

    actual_total = sum(
        count for name, count in buckets.items()
        if name not in ("buffers_excluded", "unrecognized")
    ) + buckets["unrecognized"]
    scaler_enabled = buckets["spline_scaler"] > 0

    L = n_layers
    d_model, d_t = meta["d_model"], meta["d_transcoder"]
    G, k = meta["grid_size"], meta["spline_order"]
    decoder = (L * (L + 1) // 2) * d_model * d_t
    biases = L * d_t * 2 + L * d_model  # b_enc + threshold, b_dec
    if meta["encoder_type"] == "kan":
        encoder_3term = L * d_t * d_model * (2 + G + k)
        encoder_2term = L * d_t * d_model * (1 + G + k)
    else:
        encoder_3term = encoder_2term = L * d_t * d_model

    analytic_3 = encoder_3term + decoder + biases
    analytic_2 = encoder_2term + decoder + biases
    reconciles = (
        "3-term" if actual_total == analytic_3
        else "2-term" if actual_total == analytic_2
        else "neither"
    )
    return {
        "checkpoint": str(ckpt_dir),
        "meta": meta,
        "buckets": buckets,
        "unrecognized_keys": unrecognized_keys,
        "actual_total_params": actual_total,
        "analytic_3term": analytic_3,
        "analytic_2term_eq13": analytic_2,
        "delta_3term": actual_total - analytic_3,
        "delta_2term": actual_total - analytic_2,
        "spline_scaler_enabled": scaler_enabled,
        "reconciles_with": reconciles,
    }


def _render_markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "# §4.4 Parameter-count reconciliation",
        "",
        f"git: {git_sha()}. Buffers (KAN grid, input-normalization stats) are "
        "excluded from all parameter counts.",
        "",
        "| Checkpoint | Enc | d_t | actual | analytic 3-term | analytic 2-term (Eq.13) "
        "| scaler | reconciles |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for res in results:
        meta = res["meta"]
        lines.append(
            f"| `{Path(res['checkpoint']).name}` | {meta['encoder_type']} "
            f"| {meta['d_transcoder']:,} | {res['actual_total_params']:,} "
            f"| {res['analytic_3term']:,} | {res['analytic_2term_eq13']:,} "
            f"| {'yes' if res['spline_scaler_enabled'] else 'no'} "
            f"| **{res['reconciles_with']}** |"
        )
    lines += ["", "## Component breakdown", ""]
    bucket_names = [
        "base_weight", "spline_weight", "spline_scaler", "W_enc",
        "W_dec", "b_enc", "b_dec", "threshold",
    ]
    lines.append("| Checkpoint | " + " | ".join(bucket_names) + " |")
    lines.append("|---" * (len(bucket_names) + 1) + "|")
    for res in results:
        lines.append(
            f"| `{Path(res['checkpoint']).name}` | "
            + " | ".join(f"{res['buckets'][b]:,}" for b in bucket_names)
            + " |"
        )
    mismatches = [r for r in results if r["reconciles_with"] == "neither"]
    if mismatches:
        lines += ["", "## Mismatches (investigate before quoting Eq. 13)", ""]
        for res in mismatches:
            lines.append(
                f"- `{res['checkpoint']}`: delta vs 3-term "
                f"{res['delta_3term']:+,}, vs 2-term {res['delta_2term']:+,}; "
                f"unrecognized keys: {res['unrecognized_keys'] or 'none'}"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    results = [inspect_checkpoint(Path(c)) for c in args.checkpoints]
    payload = {"git_sha": git_sha(), "results": results}
    emit(args.out_dir, "check_params", payload, _render_markdown(results))

    provenance = Provenance(script="rebuttal_eval.check_params")
    for res in results:
        provenance.record(
            "4.4", "actual_total_params", res["actual_total_params"],
            checkpoint=res["checkpoint"],
        )
        provenance.record(
            "4.4", "reconciles_with", res["reconciles_with"],
            checkpoint=res["checkpoint"],
        )
    provenance.write(args.out_dir)

    for res in results:
        print(
            f"{Path(res['checkpoint']).name}: {res['actual_total_params']:,} params, "
            f"scaler={'yes' if res['spline_scaler_enabled'] else 'no'}, "
            f"reconciles with {res['reconciles_with']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
