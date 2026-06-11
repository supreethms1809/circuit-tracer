from __future__ import annotations

import pytest

from macag.cli.annotate_graph import _build_groups, _build_groups_for_result


def test_build_groups_accepts_legacy_game1_list_evidence() -> None:
    pinned, supernodes = _build_groups(
        macag_result={
            "game": "game1",
            "evidence": ["1_2_3", "4_5_6"],
        },
        label_prefix="MACAG:test",
    )

    assert pinned == ["1_2_3", "4_5_6"]
    assert supernodes == [["MACAG:test:E_star", "1_2_3", "4_5_6"]]


def _dual_result() -> dict:
    return {
        "game": "game1",
        "freeze_mode": "both",
        "frozen": {"evidence": {"E_star": ["1_2_3"]}},
        "unfrozen": {"evidence": {"E_star": ["1_2_3", "4_5_6"]}},
    }


def test_annotate_dual_result_annotates_both_legs() -> None:
    pinned, supernodes = _build_groups_for_result(
        macag_result=_dual_result(), label_prefix="MACAG", freeze_select="both"
    )
    # Pins are the dedup'd union across legs.
    assert pinned == ["1_2_3", "4_5_6"]
    assert supernodes == [
        ["MACAG:frozen:E_star", "1_2_3"],
        ["MACAG:unfrozen:E_star", "1_2_3", "4_5_6"],
    ]


def test_annotate_dual_result_select_single_leg() -> None:
    pinned, supernodes = _build_groups_for_result(
        macag_result=_dual_result(), label_prefix="MACAG", freeze_select="frozen"
    )
    assert pinned == ["1_2_3"]
    assert supernodes == [["MACAG:frozen:E_star", "1_2_3"]]


def test_annotate_dual_result_missing_leg_raises() -> None:
    broken = _dual_result()
    del broken["unfrozen"]
    with pytest.raises(ValueError, match="unfrozen"):
        _build_groups_for_result(
            macag_result=broken, label_prefix="MACAG", freeze_select="both"
        )


def test_annotate_single_mode_result_ignores_freeze_select() -> None:
    single = {"game": "game1", "evidence": {"E_star": ["1_2_3"]}}
    pinned, supernodes = _build_groups_for_result(
        macag_result=single, label_prefix="MACAG", freeze_select="frozen"
    )
    assert pinned == ["1_2_3"]
    assert supernodes == [["MACAG:E_star", "1_2_3"]]
