from __future__ import annotations

from macag.cli.annotate_graph import _build_groups


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
