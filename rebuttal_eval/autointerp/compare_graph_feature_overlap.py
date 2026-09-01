#!/usr/bin/env python3
"""Per-prompt RAVEL graph overlap: spline vs hub linear.

Raw (layer, feature) IDs are NOT aligned across d_t=6144 vs d_t=16384, so ID
Jaccard is reported only as a diagnostic (expected ~0). The meaningful axes
are:
  - layer-set Jaccard (which layers appear in the pruned graph)
  - exact normalized-clerp Jaccard (identical autointerp labels)
  - bag-of-words Jaccard on clerp text (softer semantic overlap)
  - top-k-by-influence versions of the above
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

_SCORE_SUFFIX = re.compile(
    r"\s*\((?:det|fuzz|detection|fuzzing)=[^)]*\)\s*$", re.IGNORECASE
)
_WORD = re.compile(r"[a-z0-9]+")


def _mean(vals: list[float]) -> float:
    return statistics.fmean(vals) if vals else float("nan")


def _prompt_id(name: str) -> str | None:
    stem = Path(name).stem
    if "ravel_" not in stem:
        return None
    return stem[stem.index("ravel_") :]


def _norm_clerp(text: str) -> str:
    text = (text or "").strip()
    text = _SCORE_SUFFIX.sub("", text).strip()
    return re.sub(r"\s+", " ", text).lower()


def _bow(text: str) -> frozenset[str]:
    stop = {
        "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "with",
        "is", "are", "was", "were", "be", "by", "as", "at", "from", "that",
        "this", "it", "its", "pattern", "involves", "often", "sometimes",
        "such", "like", "followed", "preceded", "indicating",
    }
    return frozenset(w for w in _WORD.findall(text.lower()) if w not in stop and len(w) > 2)


def _jaccard(a: set[Any], b: set[Any]) -> float:
    if not a and not b:
        return float("nan")
    union = a | b
    return len(a & b) / len(union) if union else float("nan")


def _load_feat_nodes(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    nodes = []
    for node in payload.get("nodes", []):
        if node.get("feature_type") != "cross layer transcoder":
            continue
        layer = node.get("layer")
        feat = node.get("feature")
        if layer is None or feat is None:
            continue
        nodes.append(
            {
                "layer": int(layer),
                "feature": int(feat),
                "influence": float(node.get("influence") or 0.0),
                "clerp": _norm_clerp(str(node.get("clerp") or "")),
            }
        )
    return nodes


def _index(graphs_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in graphs_dir.glob("*.json"):
        if path.name in {"graph-metadata.json", "clerp_writeback_summary.json"}:
            continue
        pid = _prompt_id(path.name)
        if pid:
            out[pid] = path
    return out


def _sets(nodes: list[dict[str, Any]], top_k: int | None = None) -> dict[str, set[Any]]:
    ordered = sorted(nodes, key=lambda n: (-n["influence"], n["layer"], n["feature"]))
    if top_k is not None:
        ordered = ordered[:top_k]
    ids = {(n["layer"], n["feature"]) for n in ordered}
    layers = {n["layer"] for n in ordered}
    clerps = {n["clerp"] for n in ordered if n["clerp"]}
    words: set[str] = set()
    for n in ordered:
        if n["clerp"]:
            words |= _bow(n["clerp"])
    return {"ids": ids, "layers": layers, "clerps": clerps, "words": words}


def compare_prompt(
    sp_nodes: list[dict[str, Any]],
    lin_nodes: list[dict[str, Any]],
    top_k: int = 32,
) -> dict[str, Any]:
    sp_all = _sets(sp_nodes)
    lin_all = _sets(lin_nodes)
    sp_top = _sets(sp_nodes, top_k=top_k)
    lin_top = _sets(lin_nodes, top_k=top_k)
    return {
        "spline_n_features": len(sp_all["ids"]),
        "linear_n_features": len(lin_all["ids"]),
        "spline_n_layers": len(sp_all["layers"]),
        "linear_n_layers": len(lin_all["layers"]),
        "id_jaccard": _jaccard(sp_all["ids"], lin_all["ids"]),
        "layer_jaccard": _jaccard(sp_all["layers"], lin_all["layers"]),
        "clerp_exact_jaccard": _jaccard(sp_all["clerps"], lin_all["clerps"]),
        "clerp_bow_jaccard": _jaccard(sp_all["words"], lin_all["words"]),
        "n_shared_exact_clerps": len(sp_all["clerps"] & lin_all["clerps"]),
        "topk_id_jaccard": _jaccard(sp_top["ids"], lin_top["ids"]),
        "topk_layer_jaccard": _jaccard(sp_top["layers"], lin_top["layers"]),
        "topk_clerp_exact_jaccard": _jaccard(sp_top["clerps"], lin_top["clerps"]),
        "topk_clerp_bow_jaccard": _jaccard(sp_top["words"], lin_top["words"]),
        "topk_n_shared_exact_clerps": len(sp_top["clerps"] & lin_top["clerps"]),
        "shared_exact_clerps_sample": sorted(sp_all["clerps"] & lin_all["clerps"])[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spline-graphs",
        type=Path,
        default=Path(
            "/gscratch/ssuresh/results/paper_r5b_b3/ravel_eval_suite_r5b_b3_gemma2_2b_spline_dt6144/"
            "runs/r5b_b3_gemma2_2b_spline_dt6144/seed_101/evaluation/graphs_autointerp"
        ),
    )
    parser.add_argument(
        "--linear-graphs",
        type=Path,
        default=Path(
            "/gscratch/ssuresh/results/paper/ravel_eval_suite_v3_hub_gemma2/"
            "runs/hub_gemma2_426k/seed_101/evaluation/graphs_autointerp"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/gscratch/ssuresh/results/paper_r5b_b3/ravel_circuit_feature_overlap_gemma2"
        ),
    )
    parser.add_argument("--top-k", type=int, default=32)
    args = parser.parse_args()

    sp_idx = _index(args.spline_graphs)
    lin_idx = _index(args.linear_graphs)
    shared = sorted(set(sp_idx) & set(lin_idx))
    rows: list[dict[str, Any]] = []
    for i, pid in enumerate(shared, 1):
        row = {"prompt_id": pid, **compare_prompt(
            _load_feat_nodes(sp_idx[pid]),
            _load_feat_nodes(lin_idx[pid]),
            top_k=args.top_k,
        )}
        rows.append(row)
        if i % 50 == 0 or i == len(shared):
            print(f"compared {i}/{len(shared)}", flush=True)

    def col(key: str) -> list[float]:
        return [float(r[key]) for r in rows if r.get(key) == r.get(key)]

    summary = {
        "n_shared_prompts": len(shared),
        "top_k": args.top_k,
        "means": {
            key: _mean(col(key))
            for key in (
                "spline_n_features",
                "linear_n_features",
                "id_jaccard",
                "layer_jaccard",
                "clerp_exact_jaccard",
                "clerp_bow_jaccard",
                "n_shared_exact_clerps",
                "topk_id_jaccard",
                "topk_layer_jaccard",
                "topk_clerp_exact_jaccard",
                "topk_clerp_bow_jaccard",
                "topk_n_shared_exact_clerps",
            )
        },
        "frac_prompts_any_exact_clerp_overlap": (
            sum(1 for r in rows if r["n_shared_exact_clerps"] > 0) / len(rows)
            if rows else float("nan")
        ),
        "frac_prompts_any_topk_exact_clerp_overlap": (
            sum(1 for r in rows if r["topk_n_shared_exact_clerps"] > 0) / len(rows)
            if rows else float("nan")
        ),
        "note": (
            "ID Jaccard across spline vs hub is not interpretable: different "
            "feature dictionaries (d_t=6144 vs 16384). Prefer layer Jaccard and "
            "clerp/label overlap."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "per_prompt_overlap.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    (args.out_dir / "overlap_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    m = summary["means"]
    md = [
        "# Per-prompt RAVEL graph feature overlap: spline vs hub linear",
        "",
        f"- Shared prompts: {summary['n_shared_prompts']}",
        f"- Top-k (by influence): {args.top_k}",
        "",
        "## Mean overlap metrics",
        "",
        "| Metric | All retained features | Top-k |",
        "|---|---:|---:|",
        f"| Feature-ID Jaccard (not aligned) | {m['id_jaccard']:.4f} | {m['topk_id_jaccard']:.4f} |",
        f"| Layer-set Jaccard | {m['layer_jaccard']:.4f} | {m['topk_layer_jaccard']:.4f} |",
        f"| Exact clerp/label Jaccard | {m['clerp_exact_jaccard']:.4f} | {m['topk_clerp_exact_jaccard']:.4f} |",
        f"| Clerp bag-of-words Jaccard | {m['clerp_bow_jaccard']:.4f} | {m['topk_clerp_bow_jaccard']:.4f} |",
        f"| # shared exact clerps | {m['n_shared_exact_clerps']:.2f} | {m['topk_n_shared_exact_clerps']:.2f} |",
        "",
        (
            f"Prompts with ≥1 exact shared clerp (all features): "
            f"{summary['frac_prompts_any_exact_clerp_overlap']:.1%}"
        ),
        (
            f"Prompts with ≥1 exact shared clerp (top-{args.top_k}): "
            f"{summary['frac_prompts_any_topk_exact_clerp_overlap']:.1%}"
        ),
        "",
        f"Mean retained feature nodes: spline {m['spline_n_features']:.1f} vs hub {m['linear_n_features']:.1f}",
        "",
        summary["note"],
        "",
    ]
    (args.out_dir / "overlap_summary.md").write_text("\n".join(md))
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_dir / 'overlap_summary.md'}")


if __name__ == "__main__":
    main()
