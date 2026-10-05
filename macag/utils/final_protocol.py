"""Helpers for the final MACAG paper protocol (eval plan Group A).

Pure functions: matched-k status, random-draw aggregation, subset flags, and
longest-first stage assignment. Oracle I/O stays in the CLIs.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

# One seeded shuffle per task; subset flags are prefixes of that shuffle.
# S1 is 20 cells drawn round-robin across tasks from the same shuffle.
SUBSET_PREFIXES: dict[str, dict[str, int]] = {
    "p3": {"ioi": 20, "mcqa": 20},
    "s2": {"ioi": 20},
    "s3": {"ioi": 10},
    "s4": {"ioi": 10},
    "s6": {"ioi": 10},
    "s7": {"ioi": 2},
    "s8": {"mcqa": 10},
}
S1_CELLS = 20

# Rankings that do not depend on freeze. Re-score them per leg; do not re-select.
LEG_INDEPENDENT_METHODS = frozenset({"influence", "eap", "eap_syed", "random"})


def leg_block(game1: Mapping[str, Any] | None, leg: str) -> Mapping[str, Any]:
    """Frozen or unfrozen Game 1 leg, or the whole payload for a single-mode run."""
    if not game1:
        return {}
    if game1.get("freeze_mode") == "both":
        block = game1.get(leg)
        return block if isinstance(block, Mapping) else {}
    return game1


def leg_is_degenerate(game1: Mapping[str, Any] | None, leg: str) -> bool:
    block = leg_block(game1, leg)
    if not block:
        return False
    if "degenerate" in block:
        return bool(block["degenerate"])
    return bool((block.get("scores") or {}).get("is_degenerate", False))


def leg_evidence_size(game1: Mapping[str, Any] | None, leg: str) -> int | None:
    """|E*| for one leg. None when that leg is absent. 0 is a real empty selection."""
    block = leg_block(game1, leg)
    if not block:
        return None
    evidence = (block.get("evidence") or {}).get("E_star")
    if evidence is None:
        evidence = (block.get("evidence") or {}).get("E_y") or []
    return len(evidence)


def acdc_target_for_leg(
    game1: Mapping[str, Any] | None,
    leg: str,
    budget: int,
    explicit_k: int | None,
) -> dict[str, Any]:
    """Decide whether ACDC should search a size on this leg.

    A Game 1 file wins over ``explicit_k``. Degenerate legs are skipped.
    ``|E*| > budget`` is ``unavailable_at_k`` and must not fall back to the
    report budget.
    """
    if game1 is not None:
        if leg_block(game1, leg) == {} and game1.get("freeze_mode") == "both":
            return {"status": "no_game1", "target_k": None, "search": False}
        if leg_is_degenerate(game1, leg):
            return {"status": "degenerate_leg", "target_k": None, "search": False}
        k = leg_evidence_size(game1, leg)
        if k is None:
            return {"status": "no_game1", "target_k": None, "search": False}
        if k > budget:
            return {"status": "unavailable_at_k", "target_k": k, "search": False}
        return {"status": "ok", "target_k": k, "search": True}
    if explicit_k is None:
        return {"status": "not_requested", "target_k": None, "search": False}
    k = budget if explicit_k == -1 else int(explicit_k)
    if k > budget:
        return {"status": "unavailable_at_k", "target_k": k, "search": False}
    return {"status": "ok", "target_k": k, "search": True}


def matched_k_read(results: Mapping[Any, Any] | None, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Point a prefix curve at Game 1's |E*|, or mark the cell unavailable."""
    status = spec.get("status", "no_game1")
    if status != "ok":
        return {
            "status": "no_game1" if status == "not_requested" else status,
            "target_k": spec.get("target_k"),
            "evidence": None,
            "scores": None,
        }
    k = int(spec["target_k"])
    results = results or {}
    entry = results.get(k)
    if entry is None:
        entry = results.get(str(k))
    if not entry:
        return {"status": "unavailable_at_k", "target_k": k, "evidence": None, "scores": None}
    return {
        "status": "ok",
        "target_k": k,
        "evidence": entry.get("evidence"),
        "scores": entry.get("scores"),
    }


def mean_prefix_scores(
    draws: Sequence[Mapping[int, Mapping[str, Any]]],
) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, float]]]:
    """Per-k mean scores across random draws, plus per-k score SDs.

    Evidence is the first draw's prefix (the mean is over scores, not sets).
    """
    if not draws:
        return {}, {}
    keys = sorted({int(k) for draw in draws for k in draw})
    mean_results: dict[int, dict[str, Any]] = {}
    sd_by_k: dict[str, dict[str, float]] = {}
    for k in keys:
        score_lists: dict[str, list[float]] = {}
        evidence = None
        for draw in draws:
            entry = draw.get(k) or draw.get(str(k)) or {}
            if evidence is None:
                evidence = entry.get("evidence")
            scores = entry.get("scores") or {}
            for name, value in scores.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    score_lists.setdefault(name, []).append(float(value))
        mean_scores = {name: sum(vals) / len(vals) for name, vals in score_lists.items()}
        sd_scores: dict[str, float] = {}
        for name, vals in score_lists.items():
            if len(vals) < 2:
                sd_scores[name] = 0.0
                continue
            mu = mean_scores[name]
            sd_scores[name] = (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        mean_results[k] = {"evidence": evidence, "scores": mean_scores}
        sd_by_k[str(k)] = sd_scores
    return mean_results, sd_by_k


def assign_subset_flags(
    tasks: Mapping[str, Sequence[Mapping[str, Any]]],
    seed: int,
) -> dict[str, list[str]]:
    """Mark nested subset membership from a second shuffle (seed + 1).

    Returns ``id -> subset names``. Every prompt stays in the full Phase P set;
    flags only name the smaller sensitivity slices.
    """
    rng_seed = seed + 1
    # Shuffle within each model, not across models. Gemma and Llama each keep
    # their own 20-of-25 P3 slice. Entries without mib_model stay one group.
    by_task: dict[tuple[str, str], list[str]] = {}
    for task, entries in tasks.items():
        grouped: dict[str, list[str]] = {}
        for entry in entries:
            model = str(entry.get("mib_model") or "")
            grouped.setdefault(model, []).append(str(entry["id"]))
        for model, ids in grouped.items():
            rng = random.Random(f"{rng_seed}:{model}:{task}")
            rng.shuffle(ids)
            by_task[(model, task)] = ids
    flags: dict[str, list[str]] = {pid: [] for ids in by_task.values() for pid in ids}
    for name, limits in SUBSET_PREFIXES.items():
        for task, n in limits.items():
            for (model, group_task), ids in by_task.items():
                if group_task != task:
                    continue
                for pid in ids[:n]:
                    flags[pid].append(name)
    # S1: 20 prompts, round-robin across model × task.
    order = sorted(by_task)
    picked: list[str] = []
    cursors = {task: 0 for task in order}
    while len(picked) < S1_CELLS and any(cursors[t] < len(by_task[t]) for t in order):
        for task in order:
            i = cursors[task]
            if i >= len(by_task[task]):
                continue
            picked.append(by_task[task][i])
            cursors[task] = i + 1
            if len(picked) >= S1_CELLS:
                break
    for pid in picked:
        flags[pid].append("s1")
    return flags


def assign_stages(
    cells: Sequence[Mapping[str, Any]],
    n_workers: int,
    stages: Sequence[str] = ("graph", "game1_baselines", "game2_b0", "game2_b0p2"),
) -> list[list[dict[str, Any]]]:
    """Longest-first list scheduling of (cell, stage) tasks.

    ``cells`` items need ``cell_id`` and optional ``cost`` (a number or a map
    of stage -> cost). A stage is ready only after the previous stage of the
    same cell has been scheduled to finish no later than this task's start.
    """
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    tasks: list[dict[str, Any]] = []
    for cell in cells:
        cell_id = str(cell["cell_id"])
        cost_map = cell.get("cost")
        prev = None
        for stage in stages:
            if isinstance(cost_map, Mapping):
                cost = float(cost_map.get(stage, 1.0))
            elif cost_map is None:
                cost = 1.0
            else:
                cost = float(cost_map)
            tasks.append(
                {
                    "cell_id": cell_id,
                    "stage": stage,
                    "cost": cost,
                    "depends_on": prev,
                }
            )
            prev = (cell_id, stage)
    # Priority: longer tasks first; stable by cell then stage index.
    stage_index = {name: i for i, name in enumerate(stages)}
    pending = sorted(
        tasks,
        key=lambda t: (-t["cost"], t["cell_id"], stage_index[t["stage"]]),
    )
    finish: dict[tuple[str, str], float] = {}
    worker_free = [0.0] * n_workers
    assigned: list[list[dict[str, Any]]] = [[] for _ in range(n_workers)]
    while pending:
        ready_idx = None
        for i, task in enumerate(pending):
            dep = task["depends_on"]
            if dep is None or dep in finish:
                ready_idx = i
                break
        if ready_idx is None:
            raise RuntimeError("stage assignment stalled; a dependency is missing")
        task = pending.pop(ready_idx)
        dep = task["depends_on"]
        ready_at = 0.0 if dep is None else finish[dep]
        worker = min(range(n_workers), key=lambda w: max(worker_free[w], ready_at))
        start = max(worker_free[worker], ready_at)
        end = start + task["cost"]
        worker_free[worker] = end
        finish[(task["cell_id"], task["stage"])] = end
        assigned[worker].append(
            {
                "cell_id": task["cell_id"],
                "stage": task["stage"],
                "cost": task["cost"],
                "start": start,
                "end": end,
            }
        )
    return assigned
