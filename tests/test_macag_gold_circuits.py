"""Tests for (layer-band, token-role) gold-circuit scoring + AUROC helpers (B4.1)."""

from __future__ import annotations

import pytest

from macag.eval.gold_circuits import (
    IOI_GOLD,
    assign_token_roles,
    average_precision,
    binary_auroc,
    score_evidence_against_gold,
)
from macag.graph import CircuitGraph

# 12-layer graph so IOI_GOLD's GPT-2-small-derived bands can be tested at the
# published layers directly. Prompt: <bos> When John and Mary went ... to  (END)
PROMPT_TOKENS = ["<bos>", "When", " John", " and", " Mary", " went", " to", " the",
                 " store", ",", " John", " gave", " a", " drink", " to"]
IOI_METADATA = {"S1": "John", "IO": "Mary", "S2_corrupted": "David"}
S1_POS, IO_POS, S2_POS, END_POS = 2, 4, 10, len(PROMPT_TOKENS) - 1


def _graph(nodes: dict[str, tuple[int, int]]) -> CircuitGraph:
    """nodes: node_id -> (layer, ctx_idx); ids follow {layer}_{feature}_{ctx}."""
    metadata = {
        node: {"feature_type": "cross layer transcoder", "layer": str(layer), "ctx_idx": ctx}
        for node, (layer, ctx) in nodes.items()
    }
    # Depth anchor so n_layers infers to 12 regardless of evidence layers.
    metadata["11_0_1"] = {"feature_type": "cross layer transcoder", "layer": "11", "ctx_idx": 1}
    return CircuitGraph(nodes=list(metadata), node_metadata=metadata)


def test_assign_token_roles_from_metadata() -> None:
    roles = assign_token_roles(PROMPT_TOKENS, IOI_METADATA)
    assert roles[S1_POS] == "S1"
    assert roles[S2_POS] == "S2"
    assert roles[IO_POS] == "IO"
    assert roles[END_POS] == "END"
    assert len(roles) == 4


def test_assign_token_roles_fallback_without_metadata() -> None:
    roles = assign_token_roles(PROMPT_TOKENS, None)
    # "When" is capitalized but not space-prefixed; " John" duplicates -> S;
    # " Mary" unique -> IO.
    assert roles[S1_POS] == "S1"
    assert roles[S2_POS] == "S2"
    assert roles[IO_POS] == "IO"
    assert roles[END_POS] == "END"


def test_assign_token_roles_fallback_partial_on_ambiguity() -> None:
    tokens = ["<bos>", " Anna", " met", " Ben", " and", " Carl", "."]
    roles = assign_token_roles(tokens, None)  # no duplicated name
    assert roles == {len(tokens) - 1: "END"}


def test_score_evidence_exact_precision_recall() -> None:
    # name_mover hit: layer 9 at END; s_inhibition hit: layer 8 at END;
    # duplicate_token hit: layer 0 at S2; miss: layer 5 at IO (no component
    # reads IO); miss: unresolvable role position.
    graph = _graph({
        "9_1_14": (9, END_POS),
        "8_2_14": (8, END_POS),
        "0_3_10": (0, S2_POS),
        "5_4_4": (5, IO_POS),
    })
    roles = assign_token_roles(PROMPT_TOKENS, IOI_METADATA)
    evidence = ["9_1_14", "8_2_14", "0_3_10", "5_4_4"]
    score = score_evidence_against_gold(graph, evidence, roles, IOI_GOLD, n_layers=12)
    assert score.n_scored == 4
    assert score.n_matched == 3
    assert score.precision == pytest.approx(3 / 4)
    # components hit: name_mover, s_inhibition, duplicate_token (layer 0 at S2
    # also falls in induction's widened band? induction band starts at 0.15;
    # 0/11 = 0.0 -> no). 3 of 4 components.
    assert score.component_hits["name_mover"] >= 1
    assert score.component_hits["s_inhibition"] >= 1
    assert score.component_hits["duplicate_token"] == 1
    assert score.component_hits["induction"] == 0
    assert score.recall == pytest.approx(3 / 4)
    assert score.f1 == pytest.approx(0.75)


def test_score_tolerates_unresolvable_nodes_and_missing_roles() -> None:
    graph = _graph({"9_1_14": (9, END_POS)})
    graph.add_node("E_5_0", metadata={"feature_type": "embedding", "layer": "E"})
    roles = {END_POS: "END"}
    score = score_evidence_against_gold(
        graph, ["9_1_14", "E_5_0", "missing_node"], roles, IOI_GOLD, n_layers=12
    )
    assert score.n_evidence == 3
    # E_5_0 has layer "E" but parses via the {layer}_{feature}_{pos} convention?
    # "E" is not an int -> unresolvable layer -> excluded from n_scored.
    assert score.n_scored == 1
    assert score.precision == pytest.approx(1.0)


def test_score_requires_layer_depth() -> None:
    graph = CircuitGraph(nodes=["x"], node_metadata={"x": {}})
    with pytest.raises(ValueError, match="n_layers"):
        score_evidence_against_gold(graph, ["x"], {}, IOI_GOLD)


def test_binary_auroc_exact_and_ties() -> None:
    # perfect separation
    assert binary_auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == pytest.approx(1.0)
    # perfect inversion
    assert binary_auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == pytest.approx(0.0)
    # all tied -> 0.5
    assert binary_auroc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == pytest.approx(0.5)
    # hand-computed mixed case: scores 3>2>1, labels T,F,T -> AUROC = 0.5
    assert binary_auroc([3.0, 2.0, 1.0], [True, False, True]) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="positive"):
        binary_auroc([1.0, 2.0], [False, False])


def test_average_precision_exact() -> None:
    # ranked: T, F, T -> AP = (1/1 + 2/3) / 2
    assert average_precision([3.0, 2.0, 1.0], [True, False, True]) == pytest.approx(
        (1.0 + 2.0 / 3.0) / 2.0
    )
    assert average_precision([1.0, 2.0], [True, True]) == pytest.approx(1.0)
