#!/usr/bin/env python3
"""Compare node-threshold Game 1 runs (0.85/0.90/0.95, edge=1.0) to unfiltered and 0.8/0.98."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from experiments.analyze_game1_pruned08 import (
    LEGS,
    _load,
    _row,
    _specs,
)

NODE_TAGS = (
    ("n085_e100", 0.85),
    ("n090_e100", 0.90),
    ("n095_e100", 0.95),
)


def main() -> int:
    user = os.environ.get("USER", "ssuresh")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path(f"/gscratch/{user}/macag_mib_v4_game1_node_thr_sweep"),
    )
    parser.add_argument(
        "--pruned08-root",
        type=Path,
        default=Path(f"/gscratch/{user}/macag_mib_v4_game1_pruned08"),
    )
    args = parser.parse_args()

    rows: list[dict] = []
    missing: list[str] = []
    for spec in _specs(user):
        baseline_path = (
            spec["source_root"]
            / spec["clt_tag"]
            / spec["slug"]
            / "logit_gap"
            / "macag_game1.json"
        )
        if not baseline_path.is_file():
            raise FileNotFoundError(baseline_path)
        baseline = _load(baseline_path)

        def add(label: str, path: Path, pool: set[str] | None) -> None:
            exists = path.is_file()
            if not exists:
                missing.append(str(path))
            payload = _load(path) if exists else None
            for leg in LEGS:
                rows.append(
                    _row(
                        spec=spec,
                        label=label,
                        leg=leg,
                        payload=payload,
                        baseline=baseline,
                        pool=pool,
                        missing=not exists,
                    )
                )

        add("unfiltered", baseline_path, pool=None)
        pruned08_stats = (
            args.pruned08_root
            / spec["clt_tag"]
            / spec["slug"]
            / "graphs"
            / "prune_stats.json"
        )
        pruned08_pool = (
            set(_load(pruned08_stats).get("feature_node_ids") or [])
            if pruned08_stats.is_file()
            else None
        )
        add(
            "n080_e098",
            args.pruned08_root
            / spec["clt_tag"]
            / spec["slug"]
            / "pruned08"
            / "logit_gap"
            / "macag_game1.json",
            pruned08_pool,
        )
        for tag, _thr in NODE_TAGS:
            stats_path = (
                args.sweep_root
                / spec["clt_tag"]
                / spec["slug"]
                / tag
                / "graphs"
                / "prune_stats.json"
            )
            pool = (
                set(_load(stats_path).get("feature_node_ids") or [])
                if stats_path.is_file()
                else None
            )
            add(
                tag,
                args.sweep_root
                / spec["clt_tag"]
                / spec["slug"]
                / tag
                / "logit_gap"
                / "macag_game1.json",
                pool,
            )

    payload = {
        "sweep_root": str(args.sweep_root),
        "pruned08_root": str(args.pruned08_root),
        "missing": missing,
        "rows": rows,
    }
    args.sweep_root.mkdir(parents=True, exist_ok=True)
    json_path = args.sweep_root / "comparison.json"
    csv_path = args.sweep_root / "comparison.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    flat_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"shared", "unfiltered_only", "run_only"}
        }
        for row in rows
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    if missing:
        print(f"missing {len(missing)} artifact(s)")
    for spec in _specs(user):
        print(f"\n{spec['model_key']}")
        for row in rows:
            if row["model_key"] != spec["model_key"] or row["missing"]:
                continue
            ret = row["faithfulness_retention"]
            ret_s = "   n/a" if ret is None else f"{100 * ret:5.1f}%"
            jac = row["jaccard_unfiltered"]
            rec = row["recall_unfiltered"]
            pool = row["unfiltered_e_star_in_pool"]
            pool_s = (
                "-"
                if pool is None
                else f"{pool}/{row['unfiltered_e_star_size']}"
            )
            print(
                f"  {row['run']:<12} {row['leg']:<9} "
                f"|E*|={str(row['selected_size'] or '-'):>2} "
                f"n={str(row['n_candidates'] or '-'):>5} "
                f"Fret={ret_s} jac={0 if jac is None else jac:.2f} "
                f"rec={0 if rec is None else rec:.2f} pool={pool_s}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
