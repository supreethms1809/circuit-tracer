"""Tests for the autointerp explanation/scoring plumbing (no LLM, no GPU)."""

import json
import random

from rebuttal_eval.autointerp.explain import explain_features, highlight_window
from rebuttal_eval.autointerp.graph_features import (
    annotate_graph_clerps,
    extract_features_from_graphs_dir,
    format_clerp,
    load_feature_list,
    parse_feature_node_id,
    write_clerps_to_graphs,
)
from rebuttal_eval.autointerp.prompts import parse_yes_no
from rebuttal_eval.autointerp.score import _decoy_highlight, score_all
from rebuttal_eval.autointerp.run import aggregate


class KeywordBackend:
    """Fake scorer: answers YES iff the keyword appears in the user prompt."""

    model = "fake-scorer"

    def __init__(self, keyword: str):
        self.keyword = keyword
        self.calls = 0

    def complete(self, system, user, max_tokens=256, temperature=0.0):
        self.calls += 1
        if "One-sentence description" in user:
            return f"fires on the token {self.keyword}"
        # Detection/fuzzing: judge only the excerpt, not the description line
        # (which always contains the keyword).
        excerpt = user.split("Excerpt", 1)[-1]
        if "marked tokens" in system:
            return "YES" if f"<<{self.keyword}>>" in excerpt else "NO"
        return "YES" if self.keyword in excerpt else "NO"


def _make_collection():
    # Windows: half contain " cat", half don't.
    windows = []
    for i in range(20):
        if i % 2 == 0:
            windows.append([" the", " cat", " sat"])
        else:
            windows.append([" a", " dog", " ran"])
    cat_windows = [i for i in range(20) if i % 2 == 0]
    dog_windows = [i for i in range(20) if i % 2 == 1]
    feature = {
        "layer": 0,
        "feature": 7,
        "activation_frequency": 0.1,
        "top_examples": [
            {"window_id": wid, "max_act": 2.0, "token_acts": [0.0, 2.0, 0.0]}
            for wid in cat_windows[:5]
        ],
        "scoring_activating": [
            {"window_id": wid, "max_act": 2.0, "token_acts": [0.0, 2.0, 0.0]}
            for wid in cat_windows[5:]
        ],
        "scoring_non_activating": dog_windows,
    }
    return {
        "meta": {
            "checkpoint": "ckpt",
            "base_model": "gpt2",
            "encoder_type": "kan",
            "l0_estimate_active_per_pos": 100.0,
            "dead_feature_fraction": 0.05,
            "sampling": "uniform random over alive features",
            "corpus": "wikitext val+test",
            "explain_score_shards": "disjoint by window parity",
        },
        "features": [feature],
        "windows": windows,
    }


def test_highlight_window_marks_active_tokens():
    rendered = highlight_window([" the", " cat", " sat"], [0.0, 2.0, 0.1])
    assert "<< cat>>" in rendered
    assert "<< the>>" not in rendered
    # 0.1 is below 30% of max 2.0 -> not highlighted
    assert "<< sat>>" not in rendered


def test_decoy_highlight_avoids_active_tokens():
    rng = random.Random(0)
    rendered = _decoy_highlight([" the", " cat", " sat"], [0.0, 2.0, 0.0], rng)
    assert "<< cat>>" not in rendered
    assert "<<" in rendered


def test_parse_yes_no():
    assert parse_yes_no("YES") is True
    assert parse_yes_no("no.") is False
    assert parse_yes_no(" Yes, it fires") is True
    assert parse_yes_no("maybe") is None


def test_end_to_end_scoring_with_keyword_backend(tmp_path):
    collection = _make_collection()
    backend = KeywordBackend(" cat")
    explanations = explain_features(
        collection, backend, tmp_path / "explanations.jsonl"
    )
    assert explanations["0:7"] == "fires on the token  cat"

    results = score_all(
        collection, explanations, backend, tmp_path / "scores.jsonl", seed=1
    )
    row = results[0]
    # Perfect keyword scorer on a keyword feature: full marks both metrics.
    assert row["detection_accuracy"] == 1.0
    assert row["fuzzing_accuracy"] == 1.0

    agg = aggregate(results, collection["meta"], backend.model)
    assert agg["detection_accuracy_mean"] == 1.0
    assert agg["controls"]["l0_active_per_pos"] == 100.0

    # Resumability: second call reads the JSONL cache, no new LLM calls.
    calls_before = backend.calls
    score_all(collection, explanations, backend, tmp_path / "scores.jsonl", seed=1)
    assert backend.calls == calls_before


def _toy_graph(slug: str = "ravel_city") -> dict:
    return {
        "metadata": {"slug": slug, "prompt": "Aarau is in"},
        "nodes": [
            {
                "node_id": "3_42_5",
                "feature": 999,  # Cantor pairing; must NOT be used as feat idx
                "layer": "3",
                "feature_type": "cross layer transcoder",
                "clerp": "",
            },
            {
                "node_id": "3_42_6",  # same feature, different position
                "feature": 999,
                "layer": "3",
                "feature_type": "cross layer transcoder",
                "clerp": "",
            },
            {
                "node_id": "1_7_2",
                "feature": 111,
                "layer": "1",
                "feature_type": "cross layer transcoder",
                "clerp": "",
            },
            {
                "node_id": "0_3_2",
                "feature": -1,
                "layer": "3",
                "feature_type": "mlp reconstruction error",
                "clerp": "",
            },
            {
                "node_id": "13_502_7",
                "feature": 502,
                "layer": "13",
                "feature_type": "logit",
                "clerp": 'Output " France" (p=0.4)',
            },
        ],
    }


def test_parse_feature_node_id():
    assert parse_feature_node_id("3_42_5") == (3, 42)
    assert parse_feature_node_id("E_1_0") is None
    assert parse_feature_node_id("bad") is None


def test_extract_unique_features_ignores_cantor_and_non_features(tmp_path):
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    (graphs / "a.json").write_text(json.dumps(_toy_graph("a")))
    (graphs / "b.json").write_text(json.dumps(_toy_graph("b")))
    payload = extract_features_from_graphs_dir(graphs)
    assert payload["n_graphs"] == 2
    assert payload["n_unique_features"] == 2
    pairs = {(r["layer"], r["feature"]) for r in payload["features"]}
    assert pairs == {(1, 7), (3, 42)}
    # Cantor value 999 must not appear as a feature index
    assert 999 not in {r["feature"] for r in payload["features"]}


def test_load_feature_list_formats(tmp_path):
    path = tmp_path / "features.json"
    path.write_text(json.dumps({"features": [{"layer": 0, "feature": 1}, [2, 3], {"layer": 0, "feature": 1}]}))
    assert load_feature_list(path) == [(0, 1), (2, 3)]


def test_write_clerps_and_autointerp_scores(tmp_path):
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    (graphs / "g.json").write_text(json.dumps(_toy_graph()))
    out = tmp_path / "annotated"
    explanations = {
        "3:42": "fires on city names",
        "1:7": "fires on country tokens",
    }
    scores = {
        "3:42": {
            "layer": 3,
            "feature": 42,
            "detection_accuracy": 0.8,
            "fuzzing_accuracy": 0.7,
            "detection_n": 10,
            "fuzzing_n": 10,
        }
    }
    totals = write_clerps_to_graphs(
        graphs, explanations, out, scores, include_scores_in_clerp=True
    )
    assert totals["named"] == 3  # two 3:42 positions + one 1:7
    assert totals["missing_explanation"] == 0
    annotated = json.loads((out / "g.json").read_text())
    by_id = {n["node_id"]: n for n in annotated["nodes"]}
    assert by_id["3_42_5"]["clerp"] == "fires on city names (det=0.80, fuzz=0.70)"
    assert by_id["3_42_5"]["autointerp"]["detection_accuracy"] == 0.8
    assert by_id["1_7_2"]["clerp"] == "fires on country tokens"
    # error / logit nodes untouched
    assert by_id["0_3_2"]["clerp"] == ""
    assert by_id["13_502_7"]["clerp"].startswith("Output")


def test_format_clerp_optional_scores():
    assert format_clerp("city", {"detection_accuracy": 0.5}, include_scores=False) == "city"
    assert format_clerp("city", {"detection_accuracy": 0.5}, include_scores=True) == (
        "city (det=0.50)"
    )


def test_annotate_skips_missing_explanations():
    graph = _toy_graph()
    counts = annotate_graph_clerps(graph, {"3:42": "city"})
    assert counts["named"] == 2
    assert counts["missing_explanation"] == 1


def test_aggregate_explicit_feature_list_policy():
    collection = _make_collection()
    collection["meta"]["feature_list"] = "feature_list.json"
    collection["meta"]["n_features_targeted"] = 1
    results = [{
        "detection_accuracy": 0.5,
        "fuzzing_accuracy": 0.25,
    }]
    agg = aggregate(results, collection["meta"], "scorer")
    assert agg["n_features_targeted"] == 1
    assert "explicit feature list" in agg["controls"]["dead_features_excluded"]
    assert agg["controls"]["feature_list"] == "feature_list.json"
