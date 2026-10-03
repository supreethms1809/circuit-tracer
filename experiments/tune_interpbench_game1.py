#!/usr/bin/env python
"""Grid-search InterpBench Game1 knobs for gold-circuit F1 (fast, no Shapley).

Runs a small prompt subset for each (alpha, lam, fill_budget) and writes a
summary CSV ranked by mean logit_gap F1. Used to pick gold-aligned defaults
before the full 3-seed campaign.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument(
        "--alphas",
        default="0.0,0.25,0.5,1.0",
        help="Necessity-heavy (low) vs sufficiency-heavy (high).",
    )
    ap.add_argument("--lams", default="0.0,0.02")
    ap.add_argument("--fill-budgets", default="0,1", help="0/1 flags")
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    lams = [float(x) for x in args.lams.split(",") if x.strip()]
    fills = [bool(int(x)) for x in args.fill_budgets.split(",") if x.strip()]

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_root / "tune_summary.csv"
    rows: list[dict[str, object]] = []

    for alpha in alphas:
        for lam in lams:
            for fill in fills:
                tag = f"a{alpha:g}_l{lam:g}_fill{int(fill)}"
                out_dir = args.out_root / tag
                out_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    "experiments/run_interpbench_macag.py",
                    "--limit",
                    str(args.limit),
                    "--device",
                    args.device,
                    "--budget",
                    "4",
                    "--alpha",
                    str(alpha),
                    "--lam",
                    str(lam),
                    "--score-kinds",
                    "logit_gap",
                    "--skip-shapley",
                    "--seed",
                    "0",
                    "--out-dir",
                    str(out_dir),
                ]
                cmd.append("--fill-budget" if fill else "--no-fill-budget")
                print(f"\n>>> TUNE {tag}", flush=True)
                log_path = out_dir / "tune.log"
                with log_path.open("w") as log:
                    rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
                csv_path = out_dir / "interpbench_macag.csv"
                if rc != 0 or not csv_path.is_file():
                    print(f"FAIL {tag} rc={rc}", flush=True)
                    rows.append(
                        {
                            "tag": tag,
                            "alpha": alpha,
                            "lam": lam,
                            "fill_budget": fill,
                            "n": 0,
                            "mean_f1": float("nan"),
                            "mean_precision": float("nan"),
                            "mean_recall": float("nan"),
                            "mean_size": float("nan"),
                            "rc": rc,
                        }
                    )
                    continue
                with csv_path.open() as f:
                    data = list(csv.DictReader(f))
                f1s = [float(r["logit_gap_f1"]) for r in data]
                ps = [float(r["logit_gap_precision"]) for r in data]
                rs = [float(r["logit_gap_recall"]) for r in data]
                sz = [float(r["logit_gap_evidence_size"]) for r in data]
                row = {
                    "tag": tag,
                    "alpha": alpha,
                    "lam": lam,
                    "fill_budget": fill,
                    "n": len(data),
                    "mean_f1": sum(f1s) / len(f1s),
                    "mean_precision": sum(ps) / len(ps),
                    "mean_recall": sum(rs) / len(rs),
                    "mean_size": sum(sz) / len(sz),
                    "rc": rc,
                }
                rows.append(row)
                print(
                    f"OK {tag}: F1={row['mean_f1']:.3f} P={row['mean_precision']:.3f} "
                    f"R={row['mean_recall']:.3f} |E*|={row['mean_size']:.2f}",
                    flush=True,
                )

    rows.sort(key=lambda r: (-(r["mean_f1"] if r["mean_f1"] == r["mean_f1"] else -1),))
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n===== ranked tune summary -> {summary_path} =====")
    for r in rows[:8]:
        print(
            f"  {r['tag']}: F1={r['mean_f1']:.3f} P={r['mean_precision']:.3f} "
            f"R={r['mean_recall']:.3f} |E*|={r['mean_size']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
