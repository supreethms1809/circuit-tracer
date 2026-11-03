"""B1 row extractor against frozen fixture JSONs."""

from __future__ import annotations

from pathlib import Path

from macag.analysis.b1_rows import extract_run_dir, k_grid_continuous
from macag.utils.final_protocol import primary_leg

FIXTURE = Path(__file__).resolve().parents[1] / "eval_plan" / "b1_fixture"


def test_primary_leg_degenerate_gemma_ioi_fixture():
    rows = extract_run_dir(
        FIXTURE / "degenerate_gemma_ioi",
        clt="gemma2-426k",
        slug="mib_gemma2_ioi_0071",
        score_subdir=".",
    )
    assert rows
    pl = {r["leg"] for r in rows if r["is_primary_leg"]}
    assert pl == {"unfrozen"}
    g1_rows = [r for r in rows if r["method"] == "game1"]
    frozen_g1 = next(r for r in g1_rows if r["leg"] == "frozen")
    assert frozen_g1["leg_degenerate"] is True
    assert frozen_g1["matched_k_status"] == "degenerate_leg"
    unf_inf = next(
        r for r in rows if r["leg"] == "unfrozen" and r["method"] == "influence"
    )
    assert unf_inf["eval_k"] == 3
    assert unf_inf["capped_f"] is not None
    assert unf_inf["kl_faithfulness"] is not None
    assert unf_inf["check_kl_k_eq_game1"] is True
    assert unf_inf["k_grid_continuous"] is True


def test_llama_ioi_fixture_primary_frozen():
    rows = extract_run_dir(
        FIXTURE / "llama_ioi",
        clt="llama32-524k",
        slug="mib_llama3_ioi_0020",
        score_subdir=".",
    )
    pl_legs = {r["leg"] for r in rows if r["is_primary_leg"]}
    assert pl_legs == {"frozen"}
    assert primary_leg(
        __import__("json").loads((FIXTURE / "llama_ioi" / "macag_game1.json").read_text())
    ) == ("frozen", "frozen")


def test_k_grid_continuous_detects_gap():
    ok, _ = k_grid_continuous({0: {}, 1: {}, 2: {}}, budget=2)
    assert ok
    bad, note = k_grid_continuous({0: {}, 2: {}}, budget=2)
    assert not bad
    assert "missing_k_1" in note
