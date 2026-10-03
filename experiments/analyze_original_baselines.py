#!/usr/bin/env python
"""Aggregate original-pipeline baselines (size / logit_gap / KL / cost).

Reads ``macag_original_baselines.json`` under a MIB seed root. Does **not**
compute Jaccard vs Game 1.

Example::

    python experiments/analyze_original_baselines.py \\
      --root /gscratch/$USER/macag_mib_tmlr250v3_h200/macag_mib_seed0 \\
      --out results/original_baselines.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _rows_for_payload(clt: str, slug: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    g1 = (payload.get("comparison") or {}).get("game1_ref") or {}
    for method, entry in (payload.get("methods") or {}).items():
        if not isinstance(entry, dict):
            continue
        scores = entry.get("scores") or {}
        stats = entry.get("selection_stats") or {}
        matched = entry.get("matched_k") if method == "acdc_edge" else None
        row = {
            "clt": clt,
            "slug": slug,
            "method": method,
            "status": entry.get("status", "ok"),
            "size": entry.get("size"),
            "logit_gap": scores.get("logit_gap"),
            "kl": scores.get("kl"),
            "model_forwards": stats.get("model_forwards"),
            "model_backwards": stats.get("model_backwards"),
            "wall_s": stats.get("wall_s"),
            "matched_achieved_k": (matched or {}).get("achieved_k") if isinstance(matched, dict) else None,
            "matched_exact": (matched or {}).get("exact") if isinstance(matched, dict) else None,
            "game1_size_nodes": g1.get("size_nodes"),
            "game1_logit_gap_faithfulness": g1.get("logit_gap_faithfulness"),
            "game1_kl": g1.get("kl"),
        }
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--score-kind", default="logit_gap")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in sorted(args.root.glob(f"*/*/{args.score_kind}/macag_original_baselines.json")):
        clt = path.parents[2].name
        slug = path.parents[1].name
        payload = json.loads(path.read_text())
        rows.extend(_rows_for_payload(clt, slug, payload))
    # Also accept flat layout without score_kind dir.
    for path in sorted(args.root.glob("*/*/macag_original_baselines.json")):
        if args.score_kind in path.parts:
            continue
        clt = path.parents[1].name
        slug = path.parent.name
        payload = json.loads(path.read_text())
        rows.extend(_rows_for_payload(clt, slug, payload))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clt",
        "slug",
        "method",
        "status",
        "size",
        "logit_gap",
        "kl",
        "model_forwards",
        "model_backwards",
        "wall_s",
        "matched_achieved_k",
        "matched_exact",
        "game1_size_nodes",
        "game1_logit_gap_faithfulness",
        "game1_kl",
    ]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
