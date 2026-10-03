"""Tests for offline Neuronpedia-style graph JSON pruning."""

from __future__ import annotations

from pathlib import Path

from experiments.prune_graph_json import prune_graph_file, prune_graph_payload


def test_prune_keeps_mass_prefix_and_hubs() -> None:
    payload = {
        "metadata": {"slug": "toy", "node_threshold": 1.0},
        "qParams": {},
        "nodes": [
            {"node_id": "E_1_0", "feature_type": "embedding", "influence_raw": 0.0},
            {"node_id": "L_1_0", "feature_type": "logit", "influence_raw": 0.0},
            {"node_id": "0_1_1", "feature_type": "cross layer transcoder", "influence_raw": 5.0},
            {"node_id": "0_2_1", "feature_type": "cross layer transcoder", "influence_raw": 4.0},
            {"node_id": "0_3_1", "feature_type": "cross layer transcoder", "influence_raw": 0.5},
            {"node_id": "0_4_1", "feature_type": "cross layer transcoder", "influence_raw": 0.1},
            {
                "node_id": "err_1",
                "feature_type": "mlp reconstruction error",
                "influence_raw": 0.0,
            },
        ],
        "links": [
            {"source": "E_1_0", "target": "0_1_1", "weight": 1.0},
            {"source": "0_1_1", "target": "0_2_1", "weight": 1.0},
            {"source": "0_2_1", "target": "L_1_0", "weight": 1.0},
            {"source": "E_1_0", "target": "0_3_1", "weight": 0.01},
            {"source": "0_3_1", "target": "L_1_0", "weight": 0.01},
            {"source": "E_1_0", "target": "0_4_1", "weight": 0.001},
            {"source": "0_4_1", "target": "L_1_0", "weight": 0.001},
            {"source": "err_1", "target": "L_1_0", "weight": 0.0001},
        ],
    }
    pruned, stats = prune_graph_payload(payload, node_threshold=0.8, edge_threshold=1.0)
    kept = {node["node_id"] for node in pruned["nodes"]}
    assert "E_1_0" in kept
    assert "L_1_0" in kept
    assert "0_1_1" in kept
    assert "0_2_1" in kept
    assert "0_4_1" not in kept
    assert stats["n_features_before"] == 4
    assert stats["n_features_after"] == 2


def test_prune_writes_compact_json(tmp_path: Path) -> None:
    source = tmp_path / "src.json"
    dest = tmp_path / "dst.json"
    stats_path = tmp_path / "stats.json"
    source.write_text(
        """
        {
          "metadata": {"slug": "toy"},
          "nodes": [
            {"node_id": "E_1_0", "feature_type": "embedding", "influence_raw": 0.0},
            {"node_id": "L_1_0", "feature_type": "logit", "influence_raw": 0.0},
            {"node_id": "0_1_1", "feature_type": "cross layer transcoder", "influence_raw": 1.0}
          ],
          "links": [
            {"source": "E_1_0", "target": "0_1_1", "weight": 1.0},
            {"source": "0_1_1", "target": "L_1_0", "weight": 1.0}
          ]
        }
        """
    )
    stats = prune_graph_file(
        source, dest, node_threshold=0.8, edge_threshold=0.98, stats_path=stats_path
    )
    assert dest.is_file()
    assert stats["n_features_after"] == 1
    assert stats_path.is_file()
