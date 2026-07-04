#!/usr/bin/env python
"""Combine per-seed MACAG sweep roots into cross-seed CSVs.

Each seeded replicate of a sweep writes its analyzer CSVs into its own root
(e.g. ``results/macag_mib_seed0`` .. ``_seed2``). This script concatenates the
per-root CSVs with a ``seed`` column and, for row-level CSVs that carry
``clt``/``task`` columns, writes a cross-seed summary (mean/std/n over the
per-seed cell means) so seed variability can be read off directly.

Usage:
  python scripts/macag_combine_seeds.py \
      results/macag_mib_seed0 results/macag_mib_seed1 results/macag_mib_seed2 \
      --out results/macag_mib_seeds
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Analyzer outputs eligible for concatenation (missing files are skipped).
CSV_NAMES = [
    "summary.csv",
    "summary_agg.csv",
    "baselines.csv",
    "abr_vs_fp.csv",
    "frozen_vs_unfrozen.csv",
    "frozen_vs_unfrozen_agg.csv",
    "bootstrap_wilcoxon.csv",
    "curves/curves.csv",
    "curves/auc.csv",
]


def infer_seed(root: Path) -> int:
    match = re.search(r"seed[_-]?(\d+)$", root.name)
    if not match:
        raise SystemExit(
            f"Cannot infer seed from root name {root.name!r}; expected a '*seed<N>' suffix."
        )
    return int(match.group(1))


def summarize(df: pd.DataFrame) -> pd.DataFrame | None:
    """Mean/std/n across seeds of the per-seed (clt, task) means."""
    keys = [k for k in ("clt", "task") if k in df.columns]
    if not keys or "seed" not in df.columns:
        return None
    numeric = df.select_dtypes("number").columns.difference(["seed"])
    if numeric.empty:
        return None
    per_seed = df.groupby(keys + ["seed"], as_index=False)[list(numeric)].mean()
    agg = per_seed.groupby(keys)[list(numeric)].agg(["mean", "std", "count"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    return agg.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path, help="per-seed sweep roots (*_seed<N>)")
    ap.add_argument("--out", type=Path, required=True, help="output directory for combined CSVs")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name in CSV_NAMES:
        frames = []
        for root in args.roots:
            path = root / name
            if not path.is_file():
                continue
            df = pd.read_csv(path)
            df.insert(0, "seed", infer_seed(root))
            frames.append(df)
        if not frames:
            print(f"skip {name}: not present in any root")
            continue
        combined = pd.concat(frames, ignore_index=True)
        stem = name.replace("/", "_").removesuffix(".csv")
        out_path = args.out / f"{stem}_allseeds.csv"
        combined.to_csv(out_path, index=False)
        print(f"wrote {out_path} ({len(combined)} rows, seeds={sorted(combined.seed.unique())})")

        summary = summarize(combined)
        if summary is not None:
            sum_path = args.out / f"{stem}_seed_summary.csv"
            summary.to_csv(sum_path, index=False)
            print(f"wrote {sum_path} ({len(summary)} cells)")


if __name__ == "__main__":
    main()
