#!/usr/bin/env python3
"""B1: long-form CSV from paper_final MACAG JSONs (eval_plan §8.2 B1)."""

from __future__ import annotations

import argparse
from pathlib import Path

from macag.analysis.b1_rows import iter_campaign_rows, write_csv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--campaign-root",
        type=Path,
        default=Path("/gscratch/ssuresh/paper_final/macag"),
        help="Root with <clt>/<slug>/logit_gap/*.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("eval_plan/b1_fixture/macag_b1_per_leg_rows.csv"),
    )
    ap.add_argument("--score-subdir", default="logit_gap")
    ap.add_argument("--clts", nargs="*", default=None, help="Optional CLT tag filter")
    args = ap.parse_args()

    rows = iter_campaign_rows(
        args.campaign_root,
        score_subdir=args.score_subdir,
        clt_tags=args.clts,
    )
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
