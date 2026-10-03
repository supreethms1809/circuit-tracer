#!/usr/bin/env python3
"""Compare one-prompt Game 1 prefilter sweeps against completed v4 runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PREFILTER_VALUES = (10, 20, 50, 100, 500, 1000)
LEGS = ("frozen", "unfrozen")


@dataclass(frozen=True)
class ModelSpec:
    """Locations and labels for one matched v4 prompt."""

    model_key: str
    clt_tag: str
    slug: str
    source_root: Path

    @property
    def source_kind_dir(self) -> Path:
        return self.source_root / self.clt_tag / self.slug / "logit_gap"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _evidence(block: dict[str, Any]) -> set[str]:
    return set((block.get("evidence") or {}).get("E_star") or [])


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _recall(reference: set[str], observed: set[str]) -> float:
    return len(reference & observed) / len(reference) if reference else 1.0


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _metric(scores: dict[str, Any], key: str) -> float | None:
    value = scores.get(key)
    return float(value) if value is not None else None


def _row(
    *,
    spec: ModelSpec,
    label: str,
    prefilter_top_k: int | None,
    leg: str,
    block: dict[str, Any],
    baseline_block: dict[str, Any],
    wall_seconds: int | None,
) -> dict[str, Any]:
    scores = block.get("scores") or {}
    stats = block.get("stats") or {}
    kl_scores = block.get("kl_faithfulness") or {}
    baseline_scores = baseline_block.get("scores") or {}
    baseline_kl = baseline_block.get("kl_faithfulness") or {}
    evidence = _evidence(block)
    baseline_evidence = _evidence(baseline_block)

    logit_f = _metric(scores, "faithfulness")
    baseline_logit_f = _metric(baseline_scores, "faithfulness")
    kl_f_norm = _metric(kl_scores, "faithfulness_normalized")
    baseline_kl_f_norm = _metric(baseline_kl, "faithfulness_normalized")
    kl_norm_delta = (
        kl_f_norm - baseline_kl_f_norm
        if kl_f_norm is not None and baseline_kl_f_norm is not None
        else None
    )
    logit_retention = _safe_ratio(logit_f, baseline_logit_f)
    quality_pass = (
        prefilter_top_k is None
        or (
            logit_retention is not None
            and logit_retention >= 0.90
            and kl_norm_delta is not None
            and kl_norm_delta >= -0.05
        )
    )

    return {
        "model_key": spec.model_key,
        "clt_tag": spec.clt_tag,
        "slug": spec.slug,
        "run": label,
        "prefilter_top_k": prefilter_top_k,
        "leg": leg,
        "selected_size": len(evidence),
        "total_candidates": stats.get("total_candidates"),
        "candidate_pool": stats.get("prefiltered_candidate_count"),
        "iterations": stats.get("iterations"),
        "oracle_calls": stats.get("oracle_calls"),
        "wall_seconds": wall_seconds,
        "evidence_jaccard_unfiltered": _jaccard(baseline_evidence, evidence),
        "evidence_recall_unfiltered": _recall(baseline_evidence, evidence),
        "logit_faithfulness": logit_f,
        "logit_faithfulness_normalized": _metric(
            scores, "faithfulness_normalized"
        ),
        "logit_sufficiency": _metric(scores, "sufficiency"),
        "logit_necessity": _metric(scores, "necessity"),
        "logit_sufficiency_normalized": _metric(
            scores, "sufficiency_normalized"
        ),
        "logit_necessity_normalized": _metric(
            scores, "necessity_normalized"
        ),
        "utility": _metric(scores, "utility"),
        "kl_faithfulness": _metric(kl_scores, "faithfulness"),
        "kl_faithfulness_normalized": kl_f_norm,
        "logit_faithfulness_retention": logit_retention,
        "kl_faithfulness_normalized_delta": kl_norm_delta,
        "quality_pass": quality_pass,
    }


def _model_rows(
    spec: ModelSpec,
    sweep_root: Path,
    *,
    require_complete: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    baseline_path = spec.source_kind_dir / "macag_game1.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"missing baseline {baseline_path}")
    baseline = _load(baseline_path)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for leg in LEGS:
        baseline_block = baseline.get(leg) or {}
        rows.append(
            _row(
                spec=spec,
                label="unfiltered",
                prefilter_top_k=None,
                leg=leg,
                block=baseline_block,
                baseline_block=baseline_block,
                wall_seconds=None,
            )
        )

    for prefilter_top_k in PREFILTER_VALUES:
        kind_dir = (
            sweep_root
            / spec.clt_tag
            / spec.slug
            / f"pf{prefilter_top_k}"
            / "logit_gap"
        )
        game1_path = kind_dir / "macag_game1.json"
        meta_path = kind_dir / "run_metadata.json"
        if not game1_path.is_file():
            missing.append(str(game1_path))
            continue
        game1 = _load(game1_path)
        metadata = _load(meta_path) if meta_path.is_file() else {}
        wall_seconds = metadata.get("game1_wall_seconds")
        for leg in LEGS:
            rows.append(
                _row(
                    spec=spec,
                    label=f"pf{prefilter_top_k}",
                    prefilter_top_k=prefilter_top_k,
                    leg=leg,
                    block=game1.get(leg) or {},
                    baseline_block=baseline.get(leg) or {},
                    wall_seconds=wall_seconds,
                )
            )

    if missing and require_complete:
        raise FileNotFoundError("missing sweep artifacts:\n" + "\n".join(missing))
    return rows, missing


def _recommendation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    filtered = [row for row in rows if row["prefilter_top_k"] is not None]
    expected_cells = len({(row["model_key"], row["leg"]) for row in rows})
    by_k: dict[int, list[dict[str, Any]]] = {}
    for row in filtered:
        by_k.setdefault(int(row["prefilter_top_k"]), []).append(row)

    diagnostics = []
    recommended = None
    for prefilter_top_k in PREFILTER_VALUES:
        group = by_k.get(prefilter_top_k, [])
        complete = len(group) == expected_cells
        pass_all = complete and all(bool(row["quality_pass"]) for row in group)
        diagnostics.append(
            {
                "prefilter_top_k": prefilter_top_k,
                "complete_cells": len(group),
                "expected_cells": expected_cells,
                "pass_all_cells": pass_all,
                "min_logit_faithfulness_retention": min(
                    (
                        row["logit_faithfulness_retention"]
                        for row in group
                        if row["logit_faithfulness_retention"] is not None
                    ),
                    default=None,
                ),
                "min_kl_faithfulness_normalized_delta": min(
                    (
                        row["kl_faithfulness_normalized_delta"]
                        for row in group
                        if row["kl_faithfulness_normalized_delta"] is not None
                    ),
                    default=None,
                ),
                "median_oracle_calls": (
                    sorted(
                        int(row["oracle_calls"])
                        for row in group
                        if row["oracle_calls"] is not None
                    )[len(group) // 2]
                    if group
                    else None
                ),
            }
        )
        if recommended is None and pass_all:
            recommended = prefilter_top_k

    return {
        "rule": (
            "smallest k with >=90% raw logit-faithfulness retention and no more "
            "than 0.05 absolute loss in normalized KL faithfulness for every "
            "model/freeze leg"
        ),
        "recommended_prefilter_top_k": recommended,
        "diagnostics": diagnostics,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path(
            f"/gscratch/{os.environ.get('USER', 'ssuresh')}/"
            "macag_mib_v4_game1_prefilter_sweep"
        ),
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    user = os.environ.get("USER", "ssuresh")
    specs = (
        ModelSpec(
            "h200",
            "gemma2-426k",
            "mib_gemma2_ioi_0000",
            Path(f"/gscratch/{user}/macag_mib_tmlr250v4_h200/macag_mib_seed0"),
        ),
        ModelSpec(
            "g25m",
            "gemma2-2.5M",
            "mib_gemma2_ioi_0000",
            Path(
                f"/gscratch/{user}/macag_mib_tmlr250v4_gemma25m/"
                "macag_mib_seed0"
            ),
        ),
        ModelSpec(
            "llama",
            "llama32-524k",
            "mib_llama3_ioi_0000",
            Path(f"/gscratch/{user}/macag_mib_tmlr250v4_llama/macag_mib_seed0"),
        ),
    )

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in specs:
        model_rows, model_missing = _model_rows(
            spec, args.sweep_root, require_complete=args.require_complete
        )
        rows.extend(model_rows)
        missing.extend(model_missing)

    payload = {
        "sweep_root": str(args.sweep_root),
        "prefilter_values": list(PREFILTER_VALUES),
        "models": [
            {
                "model_key": spec.model_key,
                "clt_tag": spec.clt_tag,
                "slug": spec.slug,
                "unfiltered_run_dir": str(spec.source_kind_dir),
            }
            for spec in specs
        ],
        "missing": missing,
        "recommendation": _recommendation(rows),
        "rows": rows,
    }
    args.sweep_root.mkdir(parents=True, exist_ok=True)
    json_path = args.sweep_root / "comparison.json"
    csv_path = args.sweep_root / "comparison.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(csv_path, rows)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    if missing:
        print(f"missing {len(missing)} run(s)")
    recommendation = payload["recommendation"]["recommended_prefilter_top_k"]
    print(f"recommended_prefilter_top_k={recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
