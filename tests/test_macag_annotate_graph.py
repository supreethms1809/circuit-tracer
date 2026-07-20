from __future__ import annotations

import json
from pathlib import Path

import pytest

from macag.cli.annotate_graph import (
    _build_groups,
    _build_groups_for_result,
    _default_macag_output_path,
    _macag_title_prefix,
    annotate_graph_with_macag,
)


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


def test_macag_title_prefix_from_game() -> None:
    assert _macag_title_prefix({"game": "game1"}, "MACAG") == "MACAG G1"
    assert _macag_title_prefix({"game": "game2"}, "MACAG") == "MACAG G2"
    assert _macag_title_prefix({"game": "game2"}, "MACAG:g2") == "MACAG G2"


def test_macag_title_prefix_merges_multi_game() -> None:
    merged = _macag_title_prefix(
        {"game": "game1"},
        "MACAG:g1",
        existing_prefix="MACAG G2",
    )
    assert merged == "MACAG G1+G2"


def test_default_macag_output_path_prepends_slug() -> None:
    path = _default_macag_output_path(Path("/tmp/graphs/ioi_01.json"))
    assert path.name == "macag-ioi_01.json"


def test_annotate_graph_sets_title_prefix_and_slug(tmp_path) -> None:
    graph_path = tmp_path / "ioi_01.json"
    graph_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "slug": "ioi_01",
                    "scan": "demo-scan",
                    "transcoder_list": [],
                    "prompt_tokens": ["<bos>", "hi"],
                    "prompt": "<bos>hi",
                    "node_threshold": 0.8,
                    "schema_version": 1,
                },
                "qParams": {},
                "nodes": [],
                "links": [],
            }
        )
    )
    macag_path = tmp_path / "macag_game1.json"
    macag_path.write_text(
        json.dumps(
            {
                "game": "game1",
                "evidence": {"E_star": ["1_2_3"]},
            }
        )
    )

    output = annotate_graph_with_macag(
        graph_json_path=graph_path,
        macag_result_json_path=macag_path,
        output_path=None,
        label_prefix="MACAG:g1",
    )

    assert output.name == "macag-ioi_01.json"
    payload = json.loads(output.read_text())
    assert payload["metadata"]["slug"] == "macag-ioi_01"
    assert payload["metadata"]["title_prefix"] == "MACAG G1"

    meta_index = json.loads((tmp_path / "graph-metadata.json").read_text())
    annotated = next(entry for entry in meta_index["graphs"] if entry["slug"] == "macag-ioi_01")
    assert annotated["title_prefix"] == "MACAG G1"
