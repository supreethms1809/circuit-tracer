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

Faithfulness comparisons use the budget-matched column (``faith_budget_*``)
only — rows missing an at-budget value are dropped rather than silently
compared at unequal set sizes. Prompt is the resampling unit: when the same
prompt appears under multiple CLTs, bootstrap/Wilcoxon average within prompt
first so CLT×prompt rows are not treated as independent.

A10: pooled blocks still do that prompt-collapsing average, AND every
(clt, task) stratum gets its own full block set (same families, same tests,
Holm within each block). Strata never average across CLTs, so the
Gemma-426k / Gemma-2.5M prompt overlap cannot dilute a CLT-specific effect.
Which family/stratum is confirmatory is D7 — this script reports all of them
with n visible so the choice is auditable.

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
        # Equal-k only: do not fall back to own-k (unequal sizes).
        return _to_float(row.get(f"faith_budget_{method}"))
    return _to_float(row.get(f"{family}_{method}"))


def _prompt_key(row: dict[str, Any]) -> str:
    """Resampling unit: prefer slug (prompt id), fall back to slug|task."""
    slug = str(row.get("slug") or "").strip()
    if slug:
        return slug
    return f"{row.get('task', '')}|{row.get('clt', '')}"


def collapse_to_prompts(
    rows: Sequence[dict[str, Any]], family: str, method: str
) -> list[float]:
    """Average metric within prompt across CLTs, then return one value per prompt."""
    by_prompt: dict[str, list[float]] = {}
    for row in rows:
        value = metric_value(row, family, method)
        if value is None:
            continue
        by_prompt.setdefault(_prompt_key(row), []).append(value)
    return [float(np.mean(vals)) for vals in by_prompt.values()]


def paired_values(
    rows: Sequence[dict[str, Any]], family: str, method: str
) -> tuple[list[float], list[float]]:
    """Aligned (game1, method) pairs after collapsing to the prompt unit."""
    g1_by: dict[str, list[float]] = {}
    m_by: dict[str, list[float]] = {}
    for row in rows:
        key = _prompt_key(row)
        g1 = metric_value(row, family, "game1")
        other = metric_value(row, family, method)
        if g1 is not None:
            g1_by.setdefault(key, []).append(g1)
        if other is not None:
            m_by.setdefault(key, []).append(other)
    g1_vals: list[float] = []
    m_vals: list[float] = []
    for key in sorted(set(g1_by) & set(m_by)):
        g1_vals.append(float(np.mean(g1_by[key])))
        m_vals.append(float(np.mean(m_by[key])))
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

    g1_all = collapse_to_prompts(rows, family, "game1")
    mean, lo, hi = _mean_ci(g1_all, samples, confidence, seed)
    k_by_prompt: dict[str, list[float]] = {}
    for row in rows:
        k_val = _to_float(row.get("k_game1"))
        if k_val is not None:
            k_by_prompt.setdefault(_prompt_key(row), []).append(k_val)
    mean_k = (
        float(np.mean([float(np.mean(v)) for v in k_by_prompt.values()]))
        if k_by_prompt
        else float("nan")
    )
    records.append(
        {"method": "game1", "n": len(g1_all), "mean": mean, "lo": lo, "hi": hi,
         "mean_k": mean_k}
    )

    raw_p: dict[str, float] = {}
    stats_by_method: dict[str, dict[str, Any]] = {}
    for method in BASELINE_METHODS:
        g1_vals, m_vals = paired_values(rows, family, method)
        deltas = [g - m for g, m in zip(g1_vals, m_vals)]
        mean, lo, hi = _mean_ci(m_vals, samples, confidence, seed)
        ks_by: dict[str, list[float]] = {}
        for row in rows:
            k_val = _to_float(row.get(f"k_{method}"))
            if k_val is not None:
                # Include explicit zeros (empty selections).
                ks_by.setdefault(_prompt_key(row), []).append(k_val)
        ks = [float(np.mean(v)) for v in ks_by.values()]
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
                 "selected size is not comparable. `clt=*,task=*` blocks are "
                 "per-stratum (no cross-CLT averaging); pooled `all` blocks average "
                 "shared slugs across CLTs first.")
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

    # A10: per-(clt, task) strata. Same blocks as pooled; the resampling unit
    # stays the prompt (slug), and rows are already one-per-(clt, slug) so no
    # cross-CLT averaging happens inside a stratum.
    strata = sorted({(str(r.get("clt", "")), str(r.get("task", ""))) for r in rows})
    for clt, task in strata:
        in_stratum = [r for r in rows
                      if str(r.get("clt", "")) == clt and str(r.get("task", "")) == task]
        for filt_name, keep in filters:
            subset = [r for r in in_stratum if keep(r)]
            if not subset:
                continue
            label = f"clt={clt},task={task}"
            if filt_name != "all":
                label += f"+{filt_name}"
            for family in METRIC_FAMILIES:
                blocks[(label, family)] = method_table(subset, family=family, **kwargs)
            blocks[(label, "cost_ratio")] = [cost_ratio_record(subset, **kwargs)]

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
