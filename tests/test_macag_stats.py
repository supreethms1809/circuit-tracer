"""Tests for the MACAG statistics tooling (bootstrap/Wilcoxon script + flip-rate CIs)."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location(
    "macag_bootstrap_wilcoxon", REPO_ROOT / "scripts" / "macag_bootstrap_wilcoxon.py"
)
assert _SPEC is not None and _SPEC.loader is not None
stats_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["macag_bootstrap_wilcoxon"] = stats_mod
_SPEC.loader.exec_module(stats_mod)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_root(tmp_path: Path, n: int = 8, shift: float = 2.0) -> Path:
    """baselines.csv where game1 beats influence by exactly ``shift`` per prompt."""
    baselines = []
    summary = []
    for i in range(n):
        base = 1.0 + 0.1 * i
        row = {"clt": "cltA", "task": "toy", "slug": f"p{i:02d}", "budget": 4}
        for method, value in (
            ("game1", base + shift),
            ("influence", base),
            ("eap", base + 0.5),
            ("shapley", base + shift - 0.5),
            ("acdc", base + 1.0),
        ):
            row[f"faith_{method}"] = value
            row[f"faith_budget_{method}"] = value
            row[f"k_{method}"] = 4
            row[f"fpf_{method}"] = value / 4
            row[f"auc_{method}"] = value
            row[f"oracle_{method}"] = 4000 if method == "shapley" else 100
        baselines.append(row)
        summary.append({"clt": "cltA", "task": "toy", "slug": f"p{i:02d}",
                        "pref": "True" if i % 2 == 0 else "False"})
    _write_csv(tmp_path / "baselines.csv", baselines)
    _write_csv(tmp_path / "summary.csv", summary)
    return tmp_path


def test_holm_correct_monotone_and_geq_raw() -> None:
    raw = {"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5}
    adjusted = stats_mod.holm_correct(raw)
    for key, p in raw.items():
        assert adjusted[key] >= p
    ordered = [adjusted[k] for k, _ in sorted(raw.items(), key=lambda kv: kv[1])]
    assert ordered == sorted(ordered)
    assert adjusted["a"] == pytest.approx(0.004)  # 4 * 0.001


def test_wilcoxon_paired_direction_and_effect() -> None:
    out = stats_mod.wilcoxon_paired([1.0, 2.0, 1.5, 0.5, 3.0, 2.5])
    assert out["p"] < 0.05
    assert out["rank_biserial"] == pytest.approx(1.0)  # all deltas positive
    zero = stats_mod.wilcoxon_paired([0.0, 0.0])
    assert zero["p"] == 1.0
    assert zero["rank_biserial"] == 0.0


def test_bootstrap_wilcoxon_end_to_end(tmp_path: Path) -> None:
    root = _synthetic_root(tmp_path)
    assert stats_mod.main(["--root", str(root), "--bootstrap-samples", "200",
                           "--seed", "0"]) == 0
    csv_path = root / "bootstrap_wilcoxon.csv"
    md_path = root / "bootstrap_wilcoxon.md"
    assert md_path.is_file()
    with csv_path.open(newline="") as f:
        records = list(csv.DictReader(f))

    faith_all = {r["method"]: r for r in records
                 if r["filter"] == "all" and r["metric"] == "faith_budget"}
    # game1 mean = mean(base) + 2.0 with base = 1.0..1.7 -> mean 1.35
    assert float(faith_all["game1"]["mean"]) == pytest.approx(3.35)
    assert float(faith_all["influence"]["mean"]) == pytest.approx(1.35)
    # CI brackets the mean
    assert float(faith_all["game1"]["lo"]) <= 3.35 <= float(faith_all["game1"]["hi"])
    # game1 beats influence on every prompt
    assert int(faith_all["influence"]["wins"]) == 8
    assert int(faith_all["influence"]["losses"]) == 0
    assert float(faith_all["influence"]["median_delta"]) == pytest.approx(2.0)
    assert float(faith_all["influence"]["p_holm"]) >= float(faith_all["influence"]["p_raw"])
    # pref filter block exists with n=4 rows contributing
    faith_pref = {r["method"]: r for r in records
                  if r["filter"] == "pref_only" and r["metric"] == "faith_budget"}
    assert int(faith_pref["game1"]["n"]) == 4
    # cost ratio: 4000/100 = 40 for every prompt
    cost = [r for r in records if r["metric"] == "cost_ratio" and r["filter"] == "all"]
    assert float(cost[0]["mean"]) == pytest.approx(40.0)


def test_bootstrap_wilcoxon_deterministic(tmp_path: Path) -> None:
    root = _synthetic_root(tmp_path)
    stats_mod.main(["--root", str(root), "--bootstrap-samples", "100", "--seed", "7"])
    first = (root / "bootstrap_wilcoxon.csv").read_text()
    stats_mod.main(["--root", str(root), "--bootstrap-samples", "100", "--seed", "7"])
    assert (root / "bootstrap_wilcoxon.csv").read_text() == first


def _fake_baselines_payload(faith_by_method: dict[str, dict[str, float]], budget: int = 3) -> dict:
    payload = {
        "params": {"budget": budget},
        "comparison": {
            "faithfulness_at_k": faith_by_method,
            "auc_raw_faithfulness": {
                m: sum(v.values()) / len(v) for m, v in faith_by_method.items() if v
            },
        },
        "methods": {
            "game1": {
                "results": {k: {"evidence": [], "scores": {"faithfulness": f}}
                            for k, f in faith_by_method.get("game1", {}).items()}
            },
            "acdc": {
                "best_by_size": {
                    "12": {"evidence": [], "scores": {"faithfulness": 99.0}, "tau": 0.1}
                }
            },
        },
    }
    return payload


def test_plot_faithfulness_curves_end_to_end(tmp_path: Path) -> None:
    from experiments.plot_faithfulness_curves import main as curves_main

    bench = {"tasks": {"toy": [{"id": "p00"}, {"id": "p01"}]}}
    bench_path = tmp_path / "bench.json"
    bench_path.write_text(json.dumps(bench))

    root = tmp_path / "sweep"
    for slug, shift in (("p00", 0.0), ("p01", 1.0)):
        run_dir = root / "gemma2-426k" / slug
        run_dir.mkdir(parents=True)
        payload = _fake_baselines_payload({
            "game1": {"1": 2.0 + shift, "2": 4.0 + shift},
            "influence": {"1": 0.5 + shift, "2": 1.0 + shift, "3": 1.5 + shift},
        })
        (run_dir / "macag_baselines.json").write_text(json.dumps(payload))

    assert curves_main([
        "--root", str(root), "--bench", str(bench_path),
        "--bootstrap-samples", "100", "--seed", "0",
    ]) == 0

    with (root / "curves" / "curves.csv").open(newline="") as f:
        rows = {(r["method"], r["k"]): r for r in csv.DictReader(f)}
    assert float(rows[("game1", "1")]["mean_faith"]) == pytest.approx(2.5)
    assert float(rows[("game1", "2")]["mean_faith"]) == pytest.approx(4.5)
    assert int(rows[("influence", "3")]["n"]) == 2
    # ACDC's uncapped size-12 point is excluded by the budget cap (max_k=3)
    assert ("acdc", "12") not in rows
    lo, hi = float(rows[("game1", "1")]["lo"]), float(rows[("game1", "1")]["hi"])
    assert lo <= 2.5 <= hi

    with (root / "curves" / "auc.csv").open(newline="") as f:
        auc = {r["method"]: r for r in csv.DictReader(f)}
    assert float(auc["game1"]["mean_auc"]) == pytest.approx((3.0 + 4.0) / 2)

    pngs = list((root / "curves").glob("curve_*.png"))
    assert len(pngs) == 1 and pngs[0].stat().st_size > 0


def test_aggregate_flip_stats_exact_rates() -> None:
    from experiments.analyze_acdc_frozen_vs_unfrozen import aggregate_flip_stats

    def row(slug: str, flip: bool, range_f: float, range_u: float) -> dict:
        return {"clt": "cltA", "task": "indirect_object_identification", "slug": slug,
                "pref": True, "range_flip": flip, "range_f": range_f, "range_u": range_u,
                "upstream_f": 1, "upstream_u": 2}

    rows = [row("p0", True, -1.0, 2.0), row("p1", True, -0.5, 1.0),
            row("p2", False, 0.5, 1.0), row("p3", False, 1.0, -1.0)]
    records = aggregate_flip_stats(rows, samples=200, seed=0)
    assert len(records) == 1
    rec = records[0]
    assert rec["task"] == "IOI"
    assert rec["n"] == 4
    assert rec["flip_rate"] == pytest.approx(0.5)
    assert 0.0 <= rec["flip_lo"] <= 0.5 <= rec["flip_hi"] <= 1.0
    assert rec["negrange_f_rate"] == pytest.approx(0.5)
    assert rec["negrange_u_rate"] == pytest.approx(0.25)
    assert rec["recon_fail_f"] == 2 and rec["recon_fail_u"] == 1


def test_aggregate_flip_stats_degenerate_all_true() -> None:
    from experiments.analyze_acdc_frozen_vs_unfrozen import aggregate_flip_stats

    rows = [{"clt": "cltA", "task": "docstring_completion", "slug": f"p{i}",
             "pref": True, "range_flip": True, "range_f": -1.0, "range_u": 1.0,
             "upstream_f": 0, "upstream_u": 0} for i in range(3)]
    rec = aggregate_flip_stats(rows, samples=100, seed=0)[0]
    assert rec["flip_rate"] == pytest.approx(1.0)
    assert rec["flip_lo"] == pytest.approx(1.0)
    assert rec["flip_hi"] == pytest.approx(1.0)
    # non-pref rows are excluded entirely
    rows_nopref = [dict(r, pref=False) for r in rows]
    assert aggregate_flip_stats(rows_nopref, samples=10, seed=0) == []
