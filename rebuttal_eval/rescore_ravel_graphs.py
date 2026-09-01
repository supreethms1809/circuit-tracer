#!/usr/bin/env python3
"""Re-score RAVEL graph .pt files with reconstruction_fve=None.

Hub runs zeroed graph_replacement_score via FVE scaling when NMSE>=1.
This recomputes raw replacement + completeness from saved graphs and
aggregates node-composition metrics from the existing prompt jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from circuit_tracer.graph import Graph, compute_graph_scores


def _mean(vals: list[float]) -> float:
    return statistics.fmean(vals) if vals else float("nan")


def _std(vals: list[float]) -> float:
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def _load_prompt_rows(jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("record_type") in {None, "prompt", "prompt_metric"}:
                if row.get("status", "ok").startswith("ok") or "graph" in (
                    str(row.get("graph_pt_path", "")) + str(row.get("graph_json_path", ""))
                ):
                    rows.append(row)
    return rows


def _graph_pt_for_row(row: dict[str, Any]) -> Path | None:
    if row.get("graph_pt_path"):
        return Path(row["graph_pt_path"])
    if row.get("graph_json_path"):
        return Path(row["graph_json_path"]).with_suffix(".pt")
    return None


def rescore_suite(
    *,
    label: str,
    jsonl_path: Path,
    out_jsonl: Path,
) -> dict[str, Any]:
    rows = _load_prompt_rows(jsonl_path)
    out_rows: list[dict[str, Any]] = []
    raw_repl: list[float] = []
    raw_comp: list[float] = []
    reported_repl: list[float] = []
    err_frac: list[float] = []
    err_count: list[float] = []
    feat_count: list[float] = []
    active: list[float] = []
    missing = 0

    for i, row in enumerate(rows, 1):
        pt = _graph_pt_for_row(row)
        if pt is None or not pt.exists():
            missing += 1
            continue
        graph = Graph.from_pt(str(pt))
        repl, comp = compute_graph_scores(graph, reconstruction_fve=None)
        reported = row.get("graph_replacement_score")
        out = {
            "prompt_id": row.get("prompt_id"),
            "family": row.get("family"),
            "variant_name": row.get("variant_name") or label,
            "graph_pt_path": str(pt),
            "graph_replacement_score_raw": repl,
            "graph_completeness_score_raw": comp,
            "graph_replacement_score_reported": reported,
            "graph_completeness_score_reported": row.get("graph_completeness_score"),
            "retained_error_node_fraction": row.get("retained_error_node_fraction"),
            "retained_error_node_count": row.get("retained_error_node_count"),
            "retained_feature_node_count": row.get("retained_feature_node_count"),
            "active_feature_count": row.get("active_feature_count"),
            "top1_match_rate": row.get("top1_match_rate"),
            "kl_divergence": row.get("kl_divergence"),
        }
        out_rows.append(out)
        raw_repl.append(float(repl))
        raw_comp.append(float(comp))
        if reported is not None and reported == reported:
            reported_repl.append(float(reported))
        for key, bucket in (
            ("retained_error_node_fraction", err_frac),
            ("retained_error_node_count", err_count),
            ("retained_feature_node_count", feat_count),
            ("active_feature_count", active),
        ):
            val = row.get(key)
            if val is not None and val == val:
                bucket.append(float(val))
        if i % 50 == 0 or i == len(rows):
            print(f"[{label}] {i}/{len(rows)} rescored", flush=True)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as fh:
        for row in out_rows:
            fh.write(json.dumps(row) + "\n")

    implied_fve = []
    for row in out_rows:
        raw = row["graph_replacement_score_raw"]
        rep = row["graph_replacement_score_reported"]
        if raw and rep is not None and rep == rep and float(raw) > 1e-12:
            implied_fve.append(float(rep) / float(raw))

    summary = {
        "label": label,
        "n_prompts": len(out_rows),
        "n_missing_graphs": missing,
        "graph_replacement_score_raw": {
            "mean": _mean(raw_repl),
            "std": _std(raw_repl),
            "min": min(raw_repl) if raw_repl else float("nan"),
            "max": max(raw_repl) if raw_repl else float("nan"),
        },
        "graph_completeness_score_raw": {
            "mean": _mean(raw_comp),
            "std": _std(raw_comp),
        },
        "graph_replacement_score_reported": {
            "mean": _mean(reported_repl),
            "std": _std(reported_repl),
            "zero_fraction": (
                sum(1 for v in reported_repl if v == 0.0) / len(reported_repl)
                if reported_repl
                else float("nan")
            ),
        },
        "implied_fve_from_reported_over_raw": {
            "mean": _mean(implied_fve),
            "std": _std(implied_fve),
            "n": len(implied_fve),
        },
        "retained_error_node_fraction": {"mean": _mean(err_frac), "std": _std(err_frac)},
        "retained_error_node_count": {"mean": _mean(err_count), "std": _std(err_count)},
        "retained_feature_node_count": {"mean": _mean(feat_count), "std": _std(feat_count)},
        "active_feature_count": {"mean": _mean(active), "std": _std(active)},
        "source_jsonl": str(jsonl_path),
        "out_jsonl": str(out_jsonl),
        "note": (
            "graph_replacement_score_raw uses compute_graph_scores(..., "
            "reconstruction_fve=None). Reported hub zeros are FVE-scaled artifacts."
        ),
    }
    return summary


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/gscratch/ssuresh/results/paper_r5b_b3/ravel_rescored_raw_replacement"
        ),
    )
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    suites = [
        (
            "hub_gemma2_426k",
            Path(
                "/gscratch/ssuresh/results/paper/ravel_eval_suite_v3_hub_gemma2/"
                "runs/hub_gemma2_426k/seed_101/evaluation/prompt_metrics.jsonl"
            ),
        ),
        (
            "spline_dt6144",
            Path(
                "/gscratch/ssuresh/results/paper_r5b_b3/"
                "ravel_eval_suite_r5b_b3_gemma2_2b_spline_dt6144/per_example_metrics.jsonl"
            ),
        ),
    ]

    summaries: list[dict[str, Any]] = []
    for label, jsonl_path in suites:
        if not jsonl_path.exists():
            raise FileNotFoundError(jsonl_path)
        summary = rescore_suite(
            label=label,
            jsonl_path=jsonl_path,
            out_jsonl=out_dir / f"{label}_rescored.jsonl",
        )
        summaries.append(summary)
        (out_dir / f"{label}_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)

    payload = {
        "suites": summaries,
        "note": (
            "Corrected RAVEL graph scores with reconstruction_fve=None. "
            "Do not compare reported hub replacement (all zeros) to spline."
        ),
    }
    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# RAVEL graph re-score (raw replacement, FVE=None)",
        "",
        "| Suite | N | Raw repl | Reported repl | Raw complete | Err frac | Err n | Feat n | Active | Implied FVE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['label']} | {s['n_prompts']} "
            f"| {_fmt(s['graph_replacement_score_raw']['mean'])} "
            f"| {_fmt(s['graph_replacement_score_reported']['mean'])} "
            f"| {_fmt(s['graph_completeness_score_raw']['mean'])} "
            f"| {_fmt(s['retained_error_node_fraction']['mean'])} "
            f"| {_fmt(s['retained_error_node_count']['mean'], 1)} "
            f"| {_fmt(s['retained_feature_node_count']['mean'], 1)} "
            f"| {_fmt(s['active_feature_count']['mean'], 1)} "
            f"| {_fmt(s['implied_fve_from_reported_over_raw']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "- Raw replacement: `compute_graph_scores(graph, reconstruction_fve=None)`.",
            "- Hub reported replacement was zeroed by FVE scaling (NMSE≥1 → FVE=0).",
            "- Error-fraction gap is mostly retained feature count, not error count.",
            "",
        ]
    )
    (out_dir / "comparison.md").write_text("\n".join(lines))
    print(f"wrote {out_dir / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
