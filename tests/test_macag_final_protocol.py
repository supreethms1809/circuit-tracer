"""Final-campaign protocol helpers: matched k, random-draw means, subsets, assignment."""

from __future__ import annotations

import json

from macag.baselines.floors import select_random_draws
from macag.cli.run_baselines import main as run_baselines_main
from macag.utils.final_protocol import (
    acdc_target_for_leg,
    assign_stages,
    assign_subset_flags,
    matched_k_read,
    mean_prefix_scores,
)


def _game1(frozen_k: int, unfrozen_k: int, unfrozen_degenerate: bool = False) -> dict:
    def leg(k: int, degenerate: bool) -> dict:
        return {
            "degenerate": degenerate,
            "evidence": {"E_star": [f"n{i}" for i in range(k)]},
        }

    return {
        "freeze_mode": "both",
        "frozen": leg(frozen_k, False),
        "unfrozen": leg(unfrozen_k, unfrozen_degenerate),
    }


def test_acdc_target_skips_degenerate_and_over_budget():
    game1 = _game1(4, 3, unfrozen_degenerate=True)
    frozen = acdc_target_for_leg(game1, "frozen", budget=8, explicit_k=None)
    assert frozen == {"status": "ok", "target_k": 4, "search": True}
    unfrozen = acdc_target_for_leg(game1, "unfrozen", budget=8, explicit_k=None)
    assert unfrozen["status"] == "degenerate_leg"
    assert unfrozen["search"] is False
    over = acdc_target_for_leg(_game1(20, 1), "frozen", budget=8, explicit_k=None)
    assert over["status"] == "unavailable_at_k"
    assert over["target_k"] == 20
    assert over["search"] is False


def test_explicit_k_does_not_override_game1():
    spec = acdc_target_for_leg(_game1(4, 1), "frozen", budget=8, explicit_k=2)
    assert spec["target_k"] == 4


def test_matched_read_does_not_fall_back_to_budget():
    results = {8: {"evidence": ["a"], "scores": {"faithfulness": 1.0}}}
    spec = {"status": "unavailable_at_k", "target_k": 20}
    read = matched_k_read(results, spec)
    assert read["status"] == "unavailable_at_k"
    assert read["evidence"] is None
    ok = matched_k_read({4: {"evidence": ["b"], "scores": {"faithfulness": 0.5}}}, {"status": "ok", "target_k": 4})
    assert ok["evidence"] == ["b"]


def test_random_draws_are_seeded_and_scores_average():
    draws = select_random_draws(["a", "b", "c", "d"], seed=0, n_draws=10)
    assert draws.params["seeds"] == list(range(10))
    again = select_random_draws(["a", "b", "c", "d"], seed=0, n_draws=10)
    assert draws.extras["draw_rankings"] == again.extras["draw_rankings"]
    per_draw = [
        {1: {"evidence": ["a"], "scores": {"faithfulness": 0.0}}},
        {1: {"evidence": ["b"], "scores": {"faithfulness": 2.0}}},
    ]
    mean, sd = mean_prefix_scores(per_draw)
    assert mean[1]["scores"]["faithfulness"] == 1.0
    assert sd["1"]["faithfulness"] > 0


def test_subset_flags_are_nested_prefixes():
    tasks = {
        "ioi": [{"id": f"ioi-{i}"} for i in range(25)],
        "mcqa": [{"id": f"mcqa-{i}"} for i in range(25)],
        "arc_easy": [{"id": f"arc-{i}"} for i in range(25)],
    }
    flags = assign_subset_flags(tasks, seed=0)
    ioi_s7 = [pid for pid, names in flags.items() if pid.startswith("ioi-") and "s7" in names]
    ioi_s2 = [pid for pid, names in flags.items() if pid.startswith("ioi-") and "s2" in names]
    assert len(ioi_s7) == 2
    assert len(ioi_s2) == 20
    assert set(ioi_s7) <= set(ioi_s2)
    assert sum("s1" in names for names in flags.values()) == 20
    assert sum("s8" in names for names in flags.values()) == 10


def test_assignment_respects_stage_order_and_covers_every_task():
    cells = [
        {"cell_id": "slow", "cost": {"graph": 1, "game1_baselines": 50, "game2_b0": 5, "game2_b0p2": 5}},
        {"cell_id": "fast", "cost": 1},
    ]
    assigned = assign_stages(cells, n_workers=2)
    flat = [item for queue in assigned for item in queue]
    assert len(flat) == 8
    by_cell: dict[str, list[tuple[str, float]]] = {}
    for item in flat:
        by_cell.setdefault(item["cell_id"], []).append((item["stage"], item["start"]))
    for cell_id, stages in by_cell.items():
        order = [name for name, _ in sorted(stages, key=lambda pair: pair[1])]
        assert order == ["graph", "game1_baselines", "game2_b0", "game2_b0p2"]
    # The long Game 1 task is not piled onto a worker that is still in the graph
    # of the same cell: its start is at least the graph cost.
    slow_g1 = next(item for item in flat if item["cell_id"] == "slow" and item["stage"] == "game1_baselines")
    assert slow_g1["start"] >= 1


def test_harness_marks_over_budget_and_averages_random_draws(tmp_path):
    graph = {
        "nodes": [
            {"node_id": "a", "feature_type": "cross layer transcoder", "layer": "0", "ctx_idx": 1, "influence": 3.0},
            {"node_id": "b", "feature_type": "cross layer transcoder", "layer": "1", "ctx_idx": 1, "influence": 1.0},
        ],
        "links": [
            {"source": "a", "target": "b", "weight": 1.0},
        ],
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph))
    toy = tmp_path / "toy.json"
    toy.write_text(json.dumps({"weights_by_target": {"y": {"a": 3.0, "b": 1.0}}, "base_by_target": {"y": 0.0}}))
    game1 = {
        "freeze_mode": "both",
        "frozen": {"degenerate": False, "evidence": {"E_star": ["a", "b", "c", "d", "e"]}},
        "unfrozen": {"degenerate": True, "evidence": {"E_star": ["a"]}},
    }
    game1_path = tmp_path / "game1.json"
    game1_path.write_text(json.dumps(game1))
    out = tmp_path / "baselines.json"
    assert run_baselines_main(
        [
            "--graph-json", str(graph_path),
            "--target", "y",
            "--budget", "2",
            "--toy-oracle-json", str(toy),
            "--methods", "random,acdc",
            "--random-draws", "4",
            "--acdc-taus", "0.5",
            "--acdc-target-from-game1", str(game1_path),
            "--legs", "both",
            "--no-connected",
            "--no-progress",
            "--output-json", str(out),
        ]
    ) == 0
    payload = json.loads(out.read_text())
    assert payload["stats"]["wall_seconds"]["total"] >= 0
    random_block = payload["methods"]["random"]
    assert random_block["params"]["n_draws"] == 4
    assert len(random_block["extras"]["random_draw_results"]) == 4
    assert "scores_sd" in random_block["extras"]
    # Toy oracles have no freeze, so per-leg blocks are not invented.
    assert "legs" not in random_block
    matched = payload["methods"]["acdc"]["matched_k"]
    assert matched["status"] == "unavailable_at_k"
    assert matched["target_k"] == 5
    assert matched["evidence"] is None
