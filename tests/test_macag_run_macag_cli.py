"""CLI-level tests for run_macag Game 1 freeze modes and output schemas.

Uses toy oracles and a freeze-aware dataclass scorer (via a monkeypatched
factory loader) so everything runs without a model or GPU.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import macag.cli.run_macag as run_macag
from macag.scoring import ScoringOracle


@dataclass
class _FreezeSwitchingScorer:
    """Toy additive scorer whose weight table depends on freeze_attention."""

    frozen_weights: dict[str, float] = field(default_factory=dict)
    unfrozen_weights: dict[str, float] = field(default_factory=dict)
    base: float = 0.0
    freeze_attention: bool = True

    def _weights(self) -> dict[str, float]:
        return self.frozen_weights if self.freeze_attention else self.unfrozen_weights

    def score_all(self, target: Any) -> float:
        return self.base + float(sum(self._weights().values()))

    def score_empty(self, target: Any) -> float:
        return self.base

    def score_keep_only(self, nodes: set[Any], target: Any) -> float:
        return self.base + float(sum(self._weights().get(node, 0.0) for node in nodes))

    def score_remove(self, nodes: set[Any], target: Any) -> float:
        return self.score_all(target) - float(
            sum(self._weights().get(node, 0.0) for node in nodes)
        )


def _write_graph(tmp_path: Path, node_ids: list[str]) -> Path:
    nodes = [
        {"node_id": node_id, "feature_type": "cross layer transcoder"}
        for node_id in node_ids
    ]
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"nodes": nodes, "edges": []}))
    return path


def _write_toy_oracle(tmp_path: Path, weights: dict[str, float]) -> Path:
    path = tmp_path / "toy.json"
    path.write_text(json.dumps({"weights_by_target": {"y": weights}}))
    return path


def _base_args(graph: Path, output: Path) -> list[str]:
    return [
        "game1",
        "--graph-json", str(graph),
        "--target", "y",
        "--output-json", str(output),
        "--no-progress",
    ]


def test_cli_game1_single_mode_output_schema_unchanged(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path, ["a", "b"])
    toy = _write_toy_oracle(tmp_path, {"a": 2.0, "b": 1.0})
    output = tmp_path / "out.json"

    assert run_macag.main(_base_args(graph, output) + ["--toy-oracle-json", str(toy)]) == 0
    payload = json.loads(output.read_text())
    # Exact pre-dual top-level schema: no freeze_mode key, no leg sub-dicts.
    assert list(payload.keys()) == [
        "input_id", "target", "foil", "game", "params", "evidence", "scores", "stats",
    ]
    assert payload["params"]["stop_metric"] == "normalized"  # unset default, frozen mode
    assert "freeze_attention" not in payload["params"]
    assert payload["evidence"]["E_star"] == ["a", "b"]


def test_cli_game1_freeze_mode_both_rejects_toy_oracle(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path, ["a"])
    toy = _write_toy_oracle(tmp_path, {"a": 1.0})
    args = _base_args(graph, tmp_path / "out.json") + [
        "--toy-oracle-json", str(toy), "--freeze-mode", "both",
    ]
    with pytest.raises(ValueError, match="toy"):
        run_macag.main(args)


def test_cli_game1_freeze_mode_unfrozen_rejects_toy_oracle(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path, ["a"])
    toy = _write_toy_oracle(tmp_path, {"a": 1.0})
    args = _base_args(graph, tmp_path / "out.json") + [
        "--toy-oracle-json", str(toy), "--freeze-mode", "unfrozen",
    ]
    with pytest.raises(ValueError, match="toy"):
        run_macag.main(args)


def test_cli_game1_both_rejects_explicit_normalized_stop(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path, ["a"])
    toy = _write_toy_oracle(tmp_path, {"a": 1.0})
    args = _base_args(graph, tmp_path / "out.json") + [
        "--toy-oracle-json", str(toy),
        "--freeze-mode", "both",
        "--stop-metric", "normalized",
    ]
    with pytest.raises(ValueError, match="normalized"):
        run_macag.main(args)


def _patch_factory(monkeypatch: pytest.MonkeyPatch, scorer: _FreezeSwitchingScorer) -> None:
    def _fake_load_factory(import_path: str) -> Any:
        def _factory(**kwargs: Any) -> ScoringOracle:
            return ScoringOracle(backend=scorer, cache_enabled=True)

        return _factory

    monkeypatch.setattr(run_macag, "_load_factory", _fake_load_factory)


def test_cli_game1_both_end_to_end_with_freeze_aware_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _write_graph(tmp_path, ["a", "b"])
    output = tmp_path / "out.json"
    # Frozen range -3 (< 0), unfrozen range 5 (>= 0): the attention-mediation flip.
    scorer = _FreezeSwitchingScorer(
        frozen_weights={"a": -2.0, "b": -1.0},
        unfrozen_weights={"a": 3.0, "b": 2.0},
    )
    _patch_factory(monkeypatch, scorer)

    args = _base_args(graph, output) + [
        "--oracle-factory", "macag.fake:factory",
        "--freeze-mode", "both",
    ]
    assert run_macag.main(args) == 0
    payload = json.loads(output.read_text())

    assert payload["freeze_mode"] == "both"
    assert payload["game"] == "game1"
    assert payload["params"]["stop_metric"] == "raw_relative"
    assert payload["params"]["matched"] is True
    for leg, freeze in (("frozen", True), ("unfrozen", False)):
        leg_payload = payload[leg]
        # Each leg carries exactly the single-mode payload minus the envelope.
        assert list(leg_payload.keys()) == ["params", "evidence", "scores", "stats"]
        assert leg_payload["params"]["freeze_attention"] is freeze
        assert leg_payload["params"]["stop_metric"] == "raw_relative"
    # Frozen leg: all weights negative -> no improving candidate -> empty set.
    assert payload["frozen"]["evidence"]["E_star"] == []
    assert payload["unfrozen"]["evidence"]["E_star"] == ["a", "b"]

    diagnostic = payload["attention_mediation"]
    assert diagnostic["verdict"] == "attention_mediated"
    assert diagnostic["range_flip"] is True
    assert diagnostic["range_frozen"] == pytest.approx(-3.0)
    assert diagnostic["range_unfrozen"] == pytest.approx(5.0)
    assert diagnostic["evidence_size_frozen"] == 0
    assert diagnostic["evidence_size_unfrozen"] == 2


def test_cli_game1_unfrozen_mode_derives_unfrozen_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _write_graph(tmp_path, ["a", "b"])
    output = tmp_path / "out.json"
    scorer = _FreezeSwitchingScorer(
        frozen_weights={"a": -2.0, "b": -1.0},
        unfrozen_weights={"a": 3.0, "b": 2.0},
    )
    _patch_factory(monkeypatch, scorer)

    args = _base_args(graph, output) + [
        "--oracle-factory", "macag.fake:factory",
        "--freeze-mode", "unfrozen",
    ]
    assert run_macag.main(args) == 0
    payload = json.loads(output.read_text())
    # Single-mode schema (no freeze_mode envelope), scored under unfrozen weights,
    # with the unfrozen stop-metric default resolved.
    assert "freeze_mode" not in payload
    assert payload["params"]["stop_metric"] == "raw_relative"
    assert payload["evidence"]["E_star"] == ["a", "b"]
    assert payload["scores"]["recoverable_range"] == pytest.approx(5.0)


def test_cli_game1_frozen_mode_warns_on_unfrozen_built_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    graph = _write_graph(tmp_path, ["a"])
    output = tmp_path / "out.json"
    scorer = _FreezeSwitchingScorer(
        frozen_weights={"a": 1.0},
        unfrozen_weights={"a": 2.0},
        freeze_attention=False,  # kwargs-built unfrozen oracle, no --freeze-mode flag
    )
    _patch_factory(monkeypatch, scorer)

    args = _base_args(graph, output) + ["--oracle-factory", "macag.fake:factory"]
    with caplog.at_level(logging.WARNING, logger="macag.cli.run_macag"):
        assert run_macag.main(args) == 0
    assert any("UNFROZEN" in record.message for record in caplog.records)
    # The oracle is honored, not silently re-frozen.
    payload = json.loads(output.read_text())
    assert payload["scores"]["recoverable_range"] == pytest.approx(2.0)
