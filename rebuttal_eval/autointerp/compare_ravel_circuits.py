"""Compare RAVEL circuit structure: spline vs linear/hub for the same prompts.

Feature ID spaces are NOT aligned across different d_transcoder models, so
this never reports a cross-model Jaccard on (layer, feature) IDs. Instead it
compares per-prompt graph composition and (when available) autointerp score
distributions.

Usage:
  python -m rebuttal_eval.autointerp.compare_ravel_circuits \\
      --spline-graphs <dir> --linear-graphs <dir> \\
      --out-dir <dir> \\
      [--spline-autointerp <dir>] [--linear-autointerp <dir>]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _mean(vals: list[float]) -> float:
    return statistics.fmean(vals) if vals else float("nan")


def _std(vals: list[float]) -> float:
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def _prompt_id_from_graph_name(name: str) -> str | None:
    # e.g. r5b_b3_..._101_ravel_aarau_country_0.json -> ravel_aarau_country_0
    stem = Path(name).stem
    if "ravel_" not in stem:
        return None
    return stem[stem.index("ravel_") :]


def _graph_stats(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    nodes = payload.get("nodes", [])
    n_feat = sum(1 for n in nodes if n.get("feature_type") == "cross layer transcoder")
    n_err = sum(1 for n in nodes if n.get("feature_type") == "mlp reconstruction error")
    expl = n_feat + n_err
    feats = {
        (int(n["layer"]), int(n["feature"]))
        for n in nodes
        if n.get("feature_type") == "cross layer transcoder"
        and n.get("layer") is not None
        and n.get("feature") is not None
    }
    return {
        "n_feature_nodes": n_feat,
        "n_error_nodes": n_err,
        "error_fraction": (n_err / expl) if expl else float("nan"),
        "unique_features": feats,
        "n_unique_features": len(feats),
    }


def _load_autointerp_scores(out_dir: Path | None) -> dict[str, float]:
    if out_dir is None or not out_dir.exists():
        return {}
    scores_path = out_dir / "feature_scores.jsonl"
    if not scores_path.exists():
        return {}
    det_vals: list[float] = []
    fuzz_vals: list[float] = []
    with scores_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            det = row.get("detection_accuracy", row.get("detection"))
            fuzz = row.get("fuzzing_accuracy", row.get("fuzzing"))
            if det is not None and det == det:
                det_vals.append(float(det))
            if fuzz is not None and fuzz == fuzz:
                fuzz_vals.append(float(fuzz))
    return {
        "n_scored": float(max(len(det_vals), len(fuzz_vals))),
        "detection_mean": _mean(det_vals),
        "detection_std": _std(det_vals),
        "fuzzing_mean": _mean(fuzz_vals),
        "fuzzing_std": _std(fuzz_vals),
    }


def _index_graphs(graphs_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in graphs_dir.glob("*.json"):
        if path.name == "graph-metadata.json":
            continue
        pid = _prompt_id_from_graph_name(path.name)
        if pid:
            out[pid] = path
    return out


def compare(
    spline_graphs: Path,
    linear_graphs: Path,
    out_dir: Path,
    spline_autointerp: Path | None = None,
    linear_autointerp: Path | None = None,
) -> dict[str, Any]:
    sp_idx = _index_graphs(spline_graphs)
    lin_idx = _index_graphs(linear_graphs)
    shared = sorted(set(sp_idx) & set(lin_idx))

    rows: list[dict[str, Any]] = []
    for pid in shared:
        sp = _graph_stats(sp_idx[pid])
        lin = _graph_stats(lin_idx[pid])
        rows.append(
            {
                "prompt_id": pid,
                "spline_n_feature_nodes": sp["n_feature_nodes"],
                "linear_n_feature_nodes": lin["n_feature_nodes"],
                "spline_n_error_nodes": sp["n_error_nodes"],
                "linear_n_error_nodes": lin["n_error_nodes"],
                "spline_error_fraction": sp["error_fraction"],
                "linear_error_fraction": lin["error_fraction"],
                "spline_n_unique_features": sp["n_unique_features"],
                "linear_n_unique_features": lin["n_unique_features"],
                "feature_node_ratio_spline_over_linear": (
                    sp["n_feature_nodes"] / lin["n_feature_nodes"]
                    if lin["n_feature_nodes"]
                    else float("nan")
                ),
            }
        )

    def col(key: str) -> list[float]:
        return [float(r[key]) for r in rows if r[key] == r[key]]

    # Union feature counts within each model (not cross-aligned).
    sp_all: set[tuple[int, int]] = set()
    lin_all: set[tuple[int, int]] = set()
    for pid in shared:
        sp_all |= _graph_stats(sp_idx[pid])["unique_features"]
        lin_all |= _graph_stats(lin_idx[pid])["unique_features"]

    summary = {
        "n_shared_prompts": len(shared),
        "n_spline_only_prompts": len(set(sp_idx) - set(lin_idx)),
        "n_linear_only_prompts": len(set(lin_idx) - set(sp_idx)),
        "spline_unique_features_union": len(sp_all),
        "linear_unique_features_union": len(lin_all),
        "per_prompt_means": {
            "spline_n_feature_nodes": _mean(col("spline_n_feature_nodes")),
            "linear_n_feature_nodes": _mean(col("linear_n_feature_nodes")),
            "spline_n_error_nodes": _mean(col("spline_n_error_nodes")),
            "linear_n_error_nodes": _mean(col("linear_n_error_nodes")),
            "spline_error_fraction": _mean(col("spline_error_fraction")),
            "linear_error_fraction": _mean(col("linear_error_fraction")),
            "feature_node_ratio_spline_over_linear": _mean(
                col("feature_node_ratio_spline_over_linear")
            ),
        },
        "autointerp": {
            "spline": _load_autointerp_scores(spline_autointerp),
            "linear": _load_autointerp_scores(linear_autointerp),
        },
        "note": (
            "Cross-model (layer, feature) Jaccard is undefined: spline d_t=6144 "
            "vs hub linear d_t=16384 use different feature dictionaries. Compare "
            "per-prompt node composition and autointerp score distributions."
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_prompt.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    (out_dir / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")

    sp_ai = summary["autointerp"]["spline"]
    lin_ai = summary["autointerp"]["linear"]
    lines = [
        "# Gemma-2 RAVEL circuit compare: spline dt6144 vs hub linear 426k",
        "",
        f"- Shared prompts: {summary['n_shared_prompts']}",
        f"- Spline unique features (union): {summary['spline_unique_features_union']}",
        f"- Hub unique features (union): {summary['linear_unique_features_union']}",
        "",
        "## Per-prompt graph composition (means)",
        "",
        "| Arm | Feat nodes | Err nodes | Err frac |",
        "|---|---:|---:|---:|",
        (
            f"| Spline | {summary['per_prompt_means']['spline_n_feature_nodes']:.1f} "
            f"| {summary['per_prompt_means']['spline_n_error_nodes']:.1f} "
            f"| {summary['per_prompt_means']['spline_error_fraction']:.3f} |"
        ),
        (
            f"| Hub linear | {summary['per_prompt_means']['linear_n_feature_nodes']:.1f} "
            f"| {summary['per_prompt_means']['linear_n_error_nodes']:.1f} "
            f"| {summary['per_prompt_means']['linear_error_fraction']:.3f} |"
        ),
        "",
        (
            "Mean feat-node ratio (spline/linear): "
            f"{summary['per_prompt_means']['feature_node_ratio_spline_over_linear']:.2f}"
        ),
        "",
        "## Autointerp scores (graph-targeted features)",
        "",
        "| Arm | N scored | Detection | Fuzzing |",
        "|---|---:|---:|---:|",
        (
            f"| Spline | {sp_ai.get('n_scored', float('nan')):.0f} "
            f"| {sp_ai.get('detection_mean', float('nan')):.3f} "
            f"| {sp_ai.get('fuzzing_mean', float('nan')):.3f} |"
        ),
        (
            f"| Hub linear | {lin_ai.get('n_scored', float('nan')):.0f} "
            f"| {lin_ai.get('detection_mean', float('nan')):.3f} "
            f"| {lin_ai.get('fuzzing_mean', float('nan')):.3f} |"
        ),
        "",
        summary["note"],
        "",
    ]
    (out_dir / "comparison.md").write_text("\n".join(lines))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spline-graphs", type=Path, required=True)
    parser.add_argument("--linear-graphs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spline-autointerp", type=Path, default=None)
    parser.add_argument("--linear-autointerp", type=Path, default=None)
    args = parser.parse_args()
    summary = compare(
        spline_graphs=args.spline_graphs,
        linear_graphs=args.linear_graphs,
        out_dir=args.out_dir,
        spline_autointerp=args.spline_autointerp,
        linear_autointerp=args.linear_autointerp,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
