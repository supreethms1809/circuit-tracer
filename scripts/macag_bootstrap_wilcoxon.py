#!/usr/bin/env python
"""Bootstrap CIs + paired Wilcoxon tests over a MACAG baselines sweep root.

Reads ``<root>/baselines.csv`` (written by ``experiments/analyze_macag_baselines.py``)
and ``<root>/summary.csv`` (for the ``pref`` target-preferred filter, joined on
``(clt, slug)``), and reports, for Game 1 vs each baseline selector:

- mean faithfulness with a 95% bootstrap CI over prompts (the prompt is the
  resampling unit — the selectors are deterministic given a fixed graph+oracle),
- paired Wilcoxon signed-rank tests with Holm correction across the method
  family and the matched-pairs rank-biserial effect size,
- win/loss/tie counts, and
- the Shapley-vs-Game-1 oracle-cost ratio with a bootstrap CI.

Faithfulness comparisons use the budget-matched column (``faith_budget_*``),
falling back to the method's own-k faithfulness (``faith_*``) when the method
stopped before the budget (Game 1's early stop selects nothing further, so its
final-set faithfulness IS its value at the budget). ACDC rows are included but
its mean selected size is reported alongside — pre budget-matching (roadmap
Phase 3 ``matched_k``) its k is not comparable and its p-values should be read
with that caveat.

Outputs a markdown report and a long-format CSV next to the input root.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy.stats import rankdata, wilcoxon

from spline_clt.paper.reporting import bootstrap_mean_ci

BASELINE_METHODS = ("influence", "eap", "shapley", "acdc")
METRIC_FAMILIES = ("faith_budget", "fpf", "auc")


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_rows(root: Path) -> list[dict[str, Any]]:
    """Load baselines.csv rows, annotated with ``pref`` from summary.csv."""
    baselines_path = root / "baselines.csv"
    if not baselines_path.is_file():
        raise SystemExit(f"missing {baselines_path}; run analyze_macag_baselines.py first")
    with baselines_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    pref_by_key: dict[tuple[str, str], bool] = {}
    summary_path = root / "summary.csv"
    if summary_path.is_file():
        with summary_path.open(newline="") as f:
            for srow in csv.DictReader(f):
                key = (srow.get("clt", ""), srow.get("slug", ""))
                pref_by_key[key] = str(srow.get("pref", "")).strip().lower() == "true"

    for row in rows:
        row["pref"] = pref_by_key.get((row.get("clt", ""), row.get("slug", "")))
    return rows


def metric_value(row: dict[str, Any], family: str, method: str) -> Optional[float]:
    if family == "faith_budget":
        at_budget = _to_float(row.get(f"faith_budget_{method}"))
        return at_budget if at_budget is not None else _to_float(row.get(f"faith_{method}"))
    return _to_float(row.get(f"{family}_{method}"))


def paired_values(
    rows: Sequence[dict[str, Any]], family: str, method: str
) -> tuple[list[float], list[float]]:
    """Aligned (game1, method) value pairs, dropping rows where either is missing."""
    g1_vals: list[float] = []
    m_vals: list[float] = []
    for row in rows:
        g1 = metric_value(row, family, "game1")
        other = metric_value(row, family, method)
        if g1 is None or other is None:
            continue
        g1_vals.append(g1)
        m_vals.append(other)
    return g1_vals, m_vals


def wilcoxon_paired(deltas: Sequence[float]) -> dict[str, float]:
    """Two-sided Wilcoxon p + matched-pairs rank-biserial r on paired deltas.

    The effect size is hand-rolled from signed ranks (r = (W+ − W−)/(W+ + W−))
    so it does not depend on which statistic convention the installed scipy
    returns. Zero deltas are dropped for r (and handled by pratt for p).
    """
    n = len(deltas)
    nonzero = [d for d in deltas if d != 0.0]
    if not nonzero:
        return {"p": 1.0, "rank_biserial": 0.0, "n": float(n)}
    ranks = rankdata([abs(d) for d in nonzero])
    w_plus = float(sum(r for r, d in zip(ranks, nonzero) if d > 0))
    w_minus = float(sum(r for r, d in zip(ranks, nonzero) if d < 0))
    r_rb = (w_plus - w_minus) / (w_plus + w_minus)
    result: Any = wilcoxon(deltas, zero_method="pratt", alternative="two-sided")
    return {"p": float(result.pvalue), "rank_biserial": r_rb, "n": float(n)}


def holm_correct(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down correction; returns adjusted p per key (monotone, capped at 1)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for i, (key, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running_max = max(running_max, adj)
        adjusted[key] = running_max
    return adjusted


def _mean_ci(
    values: Sequence[float], samples: int, confidence: float, seed: int
) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    lo, hi = bootstrap_mean_ci(values, samples, confidence, seed)
    return float(np.mean(values)), lo, hi


def method_table(
    rows: Sequence[dict[str, Any]],
    *,
    family: str,
    samples: int,
    confidence: float,
    seed: int,
) -> list[dict[str, Any]]:
    """One record per method (game1 first) for a metric family over ``rows``."""
    records: list[dict[str, Any]] = []

    g1_all = [v for v in (metric_value(r, family, "game1") for r in rows) if v is not None]
    mean, lo, hi = _mean_ci(g1_all, samples, confidence, seed)
    records.append(
        {"method": "game1", "n": len(g1_all), "mean": mean, "lo": lo, "hi": hi,
         "mean_k": float(np.mean([v for v in (_to_float(r.get("k_game1")) for r in rows) if v]))
         if rows else float("nan")}
    )

    raw_p: dict[str, float] = {}
    stats_by_method: dict[str, dict[str, Any]] = {}
    for method in BASELINE_METHODS:
        g1_vals, m_vals = paired_values(rows, family, method)
        deltas = [g - m for g, m in zip(g1_vals, m_vals)]
        mean, lo, hi = _mean_ci(m_vals, samples, confidence, seed)
        ks = [v for v in (_to_float(r.get(f"k_{method}")) for r in rows) if v]
        rec: dict[str, Any] = {
            "method": method,
            "n": len(m_vals),
            "mean": mean,
            "lo": lo,
            "hi": hi,
            "mean_k": float(np.mean(ks)) if ks else float("nan"),
            "median_delta": float(np.median(deltas)) if deltas else float("nan"),
            "wins": sum(1 for d in deltas if d > 0),
            "losses": sum(1 for d in deltas if d < 0),
            "ties": sum(1 for d in deltas if d == 0),
        }
        if deltas:
            test = wilcoxon_paired(deltas)
            rec["p_raw"] = test["p"]
            rec["rank_biserial"] = test["rank_biserial"]
            raw_p[method] = test["p"]
        stats_by_method[method] = rec

    adjusted = holm_correct(raw_p) if raw_p else {}
    for method in BASELINE_METHODS:
        rec = stats_by_method[method]
        if method in adjusted:
            rec["p_holm"] = adjusted[method]
        records.append(rec)
    return records


def cost_ratio_record(
    rows: Sequence[dict[str, Any]], *, samples: int, confidence: float, seed: int
) -> dict[str, Any]:
    ratios: list[float] = []
    cheaper = 0
    for row in rows:
        shapley = _to_float(row.get("oracle_shapley"))
        game1 = _to_float(row.get("oracle_game1"))
        if shapley and game1 and game1 > 0:
            ratios.append(shapley / game1)
            if shapley > game1:
                cheaper += 1
    mean, lo, hi = _mean_ci(ratios, samples, confidence, seed)
    return {"method": "shapley", "n": len(ratios), "mean": mean, "lo": lo, "hi": hi,
            "wins": cheaper, "losses": len(ratios) - cheaper, "ties": 0}


def _fmt(value: Any, nd: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value != value:  # NaN
            return "-"
        if 0 < abs(value) < 1e-3:
            return f"{value:.1e}"
        return f"{value:.{nd}f}"
    return str(value)


def render_markdown(blocks: dict[tuple[str, str], list[dict[str, Any]]]) -> str:
    lines: list[str] = ["# MACAG baselines: bootstrap CIs + paired Wilcoxon", ""]
    columns = ("method", "n", "mean_k", "mean", "lo", "hi", "median_delta",
               "wins", "losses", "ties", "p_raw", "p_holm", "rank_biserial")
    for (filt, family), records in blocks.items():
        lines.append(f"## {family} — filter: {filt}")
        lines.append("")
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "|".join("---" for _ in columns) + "|")
        for rec in records:
            lines.append("| " + " | ".join(_fmt(rec.get(c)) for c in columns) + " |")
        lines.append("")
    lines.append("Notes: comparisons are Game 1 minus method, paired per prompt; "
                 "Holm correction spans the four baselines within each block. "
                 "`cost_ratio` rows report oracle_shapley/oracle_game1. Check "
                 "`mean_k` before reading ACDC p-values — pre budget-matching its "
                 "selected size is not comparable.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="Sweep root containing baselines.csv (+ summary.csv for pref)")
    ap.add_argument("--bootstrap-samples", type=int, default=10_000)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", type=Path, default=None,
                    help="Output path prefix (default <root>/bootstrap_wilcoxon)")
    args = ap.parse_args(argv)

    rows = load_rows(args.root)
    if not rows:
        raise SystemExit(f"no rows in {args.root}/baselines.csv")
    have_pref = any(row.get("pref") is not None for row in rows)

    filters: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [("all", lambda _r: True)]
    if have_pref:
        filters.append(("pref_only", lambda r: r.get("pref") is True))

    kwargs = {"samples": args.bootstrap_samples, "confidence": args.confidence,
              "seed": args.seed}
    blocks: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for filt_name, keep in filters:
        subset = [r for r in rows if keep(r)]
        if not subset:
            continue
        for family in METRIC_FAMILIES:
            blocks[(filt_name, family)] = method_table(subset, family=family, **kwargs)
        blocks[(filt_name, "cost_ratio")] = [cost_ratio_record(subset, **kwargs)]

    markdown = render_markdown(blocks)
    print(markdown)

    out_prefix = args.out_prefix or (args.root / "bootstrap_wilcoxon")
    md_path = Path(f"{out_prefix}.md")
    csv_path = Path(f"{out_prefix}.csv")
    md_path.write_text(markdown)

    csv_columns = ["filter", "metric", "method", "n", "mean_k", "mean", "lo", "hi",
                   "median_delta", "wins", "losses", "ties", "p_raw", "p_holm",
                   "rank_biserial"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        for (filt_name, family), records in blocks.items():
            for rec in records:
                writer.writerow({"filter": filt_name, "metric": family, **rec})
    print(f"wrote {md_path}\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
