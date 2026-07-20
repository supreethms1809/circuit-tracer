#!/usr/bin/env python
"""Faithfulness-vs-size curves per method from a MACAG baselines sweep (B3.1).

Walks ``<root>/<clt>/<slug>/macag_baselines.json``, aggregates each method's
prefix-cumulative raw faithfulness at every realized evidence size k (mean with a
bootstrap CI over prompts), and emits:

- ``<out-dir>/curves.csv``  — clt,task,method,k,n,mean_faith,lo,hi
- ``<out-dir>/auc.csv``     — per-method mean AUC (from the stored
  ``auc_raw_faithfulness``) with CI
- ``<out-dir>/curve_<clt>_<task>.png`` — one panel per (clt, task) with CI bands
  and Game 1's mean own-|E*| stop marker

ACDC has no ranked prefixes; its points come from ``comparison.faithfulness_at_k``
(post budget-matching) or ``methods.acdc.best_by_size`` on older runs, capped at
``--max-k`` so an uncapped k≈200 sweep does not stretch the axis.

Note: early-stopping methods (Game 1) do not realize every k, so per-point ``n``
varies — it is recorded in curves.csv and reflected in the CI width.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from spline_clt.paper.reporting import bootstrap_mean_ci

BENCH = "macag/data/acdc_benchmark_prompts.json"
CLTS = ["gemma2-426k", "gemma2-2.5M", "llama32-524k"]
METHODS = ("game1", "shapley", "eap", "influence", "acdc")

# Fixed entity -> categorical slot (validated set; order = CVD-safe slot order).
# Colors follow the method identity, never panel-local rank, so a panel missing
# a method never repaints the others.
METHOD_COLORS = {
    "game1": "#2a78d6",  # blue
    "eap": "#1baf7a",  # aqua
    "influence": "#eda100",  # yellow
    "shapley": "#4a3aa7",  # violet
    "acdc": "#e34948",  # red
}
METHOD_LABELS = {
    "game1": "MACAG Game 1",
    "shapley": "MC Shapley (gold)",
    "eap": "EAP (graph-derived)",
    "influence": "top-k influence",
    "acdc": "ACDC (τ-prune)",
}
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MUTED = "#898781"
INK = "#0b0b0b"


def slug_to_task(bench_path: str) -> dict[str, str]:
    d = json.load(open(bench_path))
    return {it["id"]: task for task, items in d["tasks"].items() for it in items}


def _method_points(payload: dict[str, Any], method: str, max_k: int) -> dict[int, float]:
    """{k: faith} for one method from one run, capped at max_k."""
    at_k = (payload.get("comparison", {}).get("faithfulness_at_k", {})).get(method)
    if at_k is None and method == "acdc":
        best = (payload.get("methods", {}).get("acdc", {}) or {}).get("best_by_size", {})
        at_k = {size: block["scores"]["faithfulness"] for size, block in best.items()}
    if not at_k:
        return {}
    points: dict[int, float] = {}
    for k_str, faith in at_k.items():
        k = int(k_str)
        if 1 <= k <= max_k and isinstance(faith, (int, float)):
            points[k] = float(faith)
    return points


def collect(
    root: Path, task_of: dict[str, str], methods: tuple[str, ...], max_k_arg: int
) -> tuple[dict, dict, dict, int]:
    """Per-(clt,task,method,k) faith lists, per-method AUC lists, game1 stop sizes."""
    curve_vals: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    auc_vals: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    stop_sizes: dict[tuple[str, str], list[int]] = defaultdict(list)
    max_k = max_k_arg
    for tag in CLTS:
        base = root / tag
        if not base.is_dir():
            continue
        for slug_dir in sorted(base.iterdir()):
            path = slug_dir / "macag_baselines.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text())
            budget = int(payload.get("params", {}).get("budget", 8))
            if max_k_arg <= 0:
                max_k = max(max_k, budget)
            task = task_of.get(slug_dir.name, "?")
            for method in methods:
                for k, faith in _method_points(payload, method, max_k or budget).items():
                    curve_vals[(tag, task, method, k)].append(faith)
                auc = payload.get("comparison", {}).get("auc_raw_faithfulness", {}).get(method)
                if isinstance(auc, (int, float)):
                    auc_vals[(tag, task, method)].append(float(auc))
            g1_results = payload.get("methods", {}).get("game1", {}).get("results") or {}
            if g1_results:
                stop_sizes[(tag, task)].append(max(int(k) for k in g1_results))
    return curve_vals, auc_vals, stop_sizes, max_k


def _plot_panel(
    out_path: Path,
    clt: str,
    task: str,
    rows: list[dict[str, Any]],
    stop_mean: Optional[float],
    methods: tuple[str, ...],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)

    plotted = False
    for method in methods:
        pts = sorted((r["k"], r["mean_faith"], r["lo"], r["hi"]) for r in rows if r["method"] == method)
        if not pts:
            continue
        ks = [p[0] for p in pts]
        means = [p[1] for p in pts]
        los = [p[2] for p in pts]
        his = [p[3] for p in pts]
        color = METHOD_COLORS.get(method, MUTED)
        ax.fill_between(ks, los, his, color=color, alpha=0.15, linewidth=0)
        ax.plot(
            ks, means, color=color, linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1, label=METHOD_LABELS.get(method, method),
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return

    if stop_mean is not None:
        ax.axvline(
            stop_mean, color=METHOD_COLORS["game1"], linewidth=1.2, linestyle=(0, (4, 3)),
            label="Game 1 mean own |E*|",
        )

    ax.set_xlabel("evidence size k", color=MUTED, fontsize=10)
    ax.set_ylabel("raw faithfulness (logit-gap units)", color=MUTED, fontsize=10)
    ax.set_title(f"{clt} — {task}", color=INK, fontsize=11, loc="left")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend = ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="best")
    for line in legend.get_lines():
        line.set_linewidth(2)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/macag_acdc")
    ap.add_argument("--bench", default=BENCH, help="prompt JSON for slug->task labels")
    ap.add_argument("--out-dir", default=None, help="default: <root>/curves")
    ap.add_argument("--methods", default=",".join(METHODS),
                    help=f"comma list from {METHODS}")
    ap.add_argument("--bootstrap-samples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-k", type=int, default=0,
                    help="cap on plotted k (0 = the runs' budget; caps uncapped ACDC sizes)")
    ap.add_argument("--no-plots", action="store_true", help="CSVs only (no matplotlib)")
    args = ap.parse_args(argv)

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    task_of = slug_to_task(args.bench)

    curve_vals, auc_vals, stop_sizes, _ = collect(root, task_of, methods, args.max_k)
    if not curve_vals:
        raise SystemExit(f"no macag_baselines.json found under {root}")

    curve_rows: list[dict[str, Any]] = []
    for (tag, task, method, k), values in sorted(curve_vals.items()):
        lo, hi = bootstrap_mean_ci(values, args.bootstrap_samples, 0.95, args.seed)
        curve_rows.append({
            "clt": tag, "task": task, "method": method, "k": k, "n": len(values),
            "mean_faith": sum(values) / len(values), "lo": lo, "hi": hi,
        })
    with (out_dir / "curves.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        writer.writeheader()
        writer.writerows(curve_rows)

    auc_rows: list[dict[str, Any]] = []
    for (tag, task, method), values in sorted(auc_vals.items()):
        lo, hi = bootstrap_mean_ci(values, args.bootstrap_samples, 0.95, args.seed)
        auc_rows.append({
            "clt": tag, "task": task, "method": method, "n": len(values),
            "mean_auc": sum(values) / len(values), "lo": lo, "hi": hi,
        })
    if auc_rows:
        with (out_dir / "auc.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(auc_rows[0].keys()))
            writer.writeheader()
            writer.writerows(auc_rows)

    n_png = 0
    if not args.no_plots:
        panels = sorted({(r["clt"], r["task"]) for r in curve_rows})
        for tag, task in panels:
            rows = [r for r in curve_rows if r["clt"] == tag and r["task"] == task]
            stops = stop_sizes.get((tag, task))
            stop_mean = sum(stops) / len(stops) if stops else None
            safe_task = "".join(c if c.isalnum() or c in "-_" else "_" for c in task)
            out_path = out_dir / f"curve_{tag}_{safe_task}.png"
            _plot_panel(out_path, tag, task, rows, stop_mean, methods)
            if out_path.is_file():
                n_png += 1

    print(f"wrote {out_dir / 'curves.csv'} ({len(curve_rows)} rows), "
          f"{out_dir / 'auc.csv'} ({len(auc_rows)} rows), {n_png} PNG panel(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
