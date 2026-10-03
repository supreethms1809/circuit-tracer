#!/usr/bin/env python3
"""Compare 0.8-pruned Game 1 E* sets to unfiltered and all prefilter runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

PREFILTER_VALUES = (10, 20, 50, 100, 500, 1000)
UNFROZEN_LADDER = (1500, 2000, 2500, 3000)
LEGS = ("frozen", "unfrozen")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _block(payload: dict[str, Any], leg: str) -> dict[str, Any]:
    block = payload.get(leg)
    return block if isinstance(block, dict) else payload


def _evidence(payload: dict[str, Any], leg: str) -> set[str]:
    return {
        str(node)
        for node in (_block(payload, leg).get("evidence") or {}).get("E_star") or []
    }


def _faithfulness(payload: dict[str, Any], leg: str) -> float | None:
    value = (_block(payload, leg).get("scores") or {}).get("faithfulness")
    return None if value is None else float(value)


def _n_candidates(payload: dict[str, Any], leg: str) -> int | None:
    stats = _block(payload, leg).get("stats") or {}
    for key in ("prefiltered_candidate_count", "total_candidates"):
        if stats.get(key) is not None:
            return int(stats[key])
    return None


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _recall(reference: set[str], observed: set[str]) -> float:
    return len(reference & observed) / len(reference) if reference else 1.0


def _specs(user: str) -> list[dict[str, Any]]:
    return [
        {
            "model_key": "h200",
            "clt_tag": "gemma2-426k",
            "slug": "mib_gemma2_ioi_0000",
            "source_root": Path(f"/gscratch/{user}/macag_mib_tmlr250v4_h200/macag_mib_seed0"),
        },
        {
            "model_key": "g25m",
            "clt_tag": "gemma2-2.5M",
            "slug": "mib_gemma2_ioi_0000",
            "source_root": Path(
                f"/gscratch/{user}/macag_mib_tmlr250v4_gemma25m/macag_mib_seed0"
            ),
        },
        {
            "model_key": "llama",
            "clt_tag": "llama32-524k",
            "slug": "mib_llama3_ioi_0000",
            "source_root": Path(f"/gscratch/{user}/macag_mib_tmlr250v4_llama/macag_mib_seed0"),
        },
    ]


def _row(
    *,
    spec: dict[str, Any],
    label: str,
    leg: str,
    payload: dict[str, Any] | None,
    baseline: dict[str, Any],
    pool: set[str] | None,
    missing: bool,
) -> dict[str, Any]:
    base_e = _evidence(baseline, leg)
    base_f = _faithfulness(baseline, leg)
    pool_hits = None if pool is None else sum(node in pool for node in base_e)
    if missing or payload is None:
        return {
            "model_key": spec["model_key"],
            "clt_tag": spec["clt_tag"],
            "slug": spec["slug"],
            "run": label,
            "leg": leg,
            "missing": True,
            "n_candidates": None,
            "selected_size": None,
            "faithfulness": None,
            "faithfulness_retention": None,
            "jaccard_unfiltered": None,
            "recall_unfiltered": None,
            "unfiltered_e_star_in_pool": pool_hits,
            "unfiltered_e_star_size": len(base_e),
            "shared": None,
            "unfiltered_only": None,
            "run_only": None,
        }
    ev = _evidence(payload, leg)
    faith = _faithfulness(payload, leg)
    retention = (
        None if base_f in (None, 0) or faith is None else float(faith) / float(base_f)
    )
    return {
        "model_key": spec["model_key"],
        "clt_tag": spec["clt_tag"],
        "slug": spec["slug"],
        "run": label,
        "leg": leg,
        "missing": False,
        "n_candidates": _n_candidates(payload, leg),
        "selected_size": len(ev),
        "faithfulness": faith,
        "faithfulness_retention": retention,
        "jaccard_unfiltered": _jaccard(base_e, ev),
        "recall_unfiltered": _recall(base_e, ev),
        "unfiltered_e_star_in_pool": pool_hits,
        "unfiltered_e_star_size": len(base_e),
        "shared": sorted(base_e & ev),
        "unfiltered_only": sorted(base_e - ev),
        "run_only": sorted(ev - base_e),
    }


def main() -> int:
    user = os.environ.get("USER", "ssuresh")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pruned-root",
        type=Path,
        default=Path(f"/gscratch/{user}/macag_mib_v4_game1_pruned08"),
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path(f"/gscratch/{user}/macag_mib_v4_game1_prefilter_sweep"),
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in _specs(user):
        baseline_path = (
            spec["source_root"] / spec["clt_tag"] / spec["slug"] / "logit_gap" / "macag_game1.json"
        )
        if not baseline_path.is_file():
            raise FileNotFoundError(baseline_path)
        baseline = _load(baseline_path)
        stats_path = (
            args.pruned_root / spec["clt_tag"] / spec["slug"] / "graphs" / "prune_stats.json"
        )
        pool: set[str] | None = None
        if stats_path.is_file():
            pool = set(_load(stats_path).get("feature_node_ids") or [])

        def add(label: str, path: Path, *, attach_pool: bool = False) -> None:
            exists = path.is_file()
            if not exists:
                missing.append(str(path))
            payload = _load(path) if exists else None
            for leg in LEGS:
                if label.startswith("unfrozen_pf") and leg != "unfrozen":
                    continue
                rows.append(
                    _row(
                        spec=spec,
                        label=label,
                        leg=leg,
                        payload=payload,
                        baseline=baseline,
                        pool=pool if attach_pool else None,
                        missing=not exists,
                    )
                )

        add("unfiltered", baseline_path, attach_pool=True)
        add(
            "pruned08",
            args.pruned_root
            / spec["clt_tag"]
            / spec["slug"]
            / "pruned08"
            / "logit_gap"
            / "macag_game1.json",
            attach_pool=True,
        )
        for k in PREFILTER_VALUES:
            add(
                f"pf{k}",
                args.sweep_root
                / spec["clt_tag"]
                / spec["slug"]
                / f"pf{k}"
                / "logit_gap"
                / "macag_game1.json",
            )
        for k in UNFROZEN_LADDER:
            add(
                f"unfrozen_pf{k}",
                args.sweep_root
                / spec["clt_tag"]
                / spec["slug"]
                / f"unfrozen_pf{k}"
                / "logit_gap"
                / "macag_game1.json",
            )

    payload = {
        "pruned_root": str(args.pruned_root),
        "sweep_root": str(args.sweep_root),
        "missing": missing,
        "rows": rows,
    }
    args.pruned_root.mkdir(parents=True, exist_ok=True)
    json_path = args.pruned_root / "comparison.json"
    csv_path = args.pruned_root / "comparison.csv"
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
    interesting = {"unfiltered", "pruned08"} | {f"pf{k}" for k in PREFILTER_VALUES}
    for spec in _specs(user):
        print(f"\n{spec['model_key']}")
        for row in rows:
            if row["model_key"] != spec["model_key"] or row["missing"]:
                continue
            if row["run"] not in interesting:
                continue
            ret = row["faithfulness_retention"]
            ret_s = "   n/a" if ret is None else f"{100 * ret:5.1f}%"
            jac = row["jaccard_unfiltered"]
            rec = row["recall_unfiltered"]
            jac_s = " n/a" if jac is None else f"{jac:.2f}"
            rec_s = " n/a" if rec is None else f"{rec:.2f}"
            print(
                f"  {row['run']:<16} {row['leg']:<9} "
                f"|E*|={str(row['selected_size'] or '-'):>2} "
                f"n={str(row['n_candidates'] or '-'):>5} "
                f"Fret={ret_s} jac={jac_s} rec={rec_s}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
