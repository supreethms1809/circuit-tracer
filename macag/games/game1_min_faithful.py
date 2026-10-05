"""Game 1: minimal faithful evidence set via greedy hill-climb."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from macag.graph import CircuitGraph, NodeId, grow_connected_frontier
from macag.scoring import ScoringOracle, TargetId
from macag.utils.metrics import (
    FaithfulnessMetrics,
    compute_faithfulness_metrics,
    dedupe_preserve_order,
    game1_utility,
    sparsity,
)
from macag.nvtx import nvtx_range

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

LOGGER = logging.getLogger(__name__)

GAME1_CKPT_SCHEMA = "macag_game1_ckpt_v1"


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(dest)


def _load_game1_checkpoint(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    dest = Path(path)
    if not dest.is_file():
        return None
    data = json.loads(dest.read_text())
    if data.get("schema") != GAME1_CKPT_SCHEMA:
        LOGGER.warning("Ignoring game1 checkpoint with unknown schema at %s", dest)
        return None
    return data


def game1_checkpoint_progress(ckpt: Mapping[str, Any]) -> tuple[int, int, int]:
    """Comparable (frozen_done, frozen_|E|, current_leg_|E|) for harvest-vs-ckpt."""
    frozen_done = 1 if ckpt.get("frozen_done") else 0
    frozen_n = len(ckpt.get("frozen_order") or [])
    selected_n = len(ckpt.get("selected_order") or [])
    leg = str(ckpt.get("leg") or "single")
    if not frozen_done and leg != "unfrozen":
        frozen_n = max(frozen_n, selected_n)
        selected_n = 0
    return (frozen_done, frozen_n, selected_n)


def _empty_game1_ckpt() -> dict[str, Any]:
    return {
        "schema": GAME1_CKPT_SCHEMA,
        "freeze_mode": "single",
        "leg": "single",
        "frozen_done": False,
        "frozen_order": [],
        "selected_order": [],
        "first_faith_gain": None,
        "harvested": False,
    }


def checkpoint_from_game1_logs(log_paths: Sequence[str | Path]) -> dict[str, Any] | None:
    """Rebuild a Game 1 checkpoint from ``Game1 added node=`` lines in solver logs.

    Dual-freeze logs emit a frozen leg then an unfrozen leg. A 24h kill often
    happens after the frozen greedy has grown |E| into the teens but before
    ``macag_game1.json`` is written (that file is only emitted after BOTH legs).
    Harvesting the slug log recovers that frozen prefix so the next submit does
    not restart from |E|=0.
    """
    import re

    added = re.compile(r"Game1 added node=(\S+) gain=([-\d.eE]+) \|E\|=(\d+)")
    best: dict[str, Any] | None = None
    best_key = (-1, -1, -1)

    def consider(current: dict[str, Any]) -> None:
        nonlocal best, best_key
        if not current["selected_order"] and not current["frozen_order"] and not current["frozen_done"]:
            return
        key = game1_checkpoint_progress(current)
        if key > best_key:
            best_key = key
            best = dict(current)

    for path in log_paths:
        p = Path(path)
        if not p.is_file():
            continue
        current = _empty_game1_ckpt()
        current["harvested"] = True
        with p.open(errors="replace") as handle:
            for line in handle:
                if "Game1 dual: frozen leg starting" in line:
                    consider(current)
                    current = _empty_game1_ckpt()
                    current["harvested"] = True
                    current["freeze_mode"] = "both"
                    current["leg"] = "frozen"
                elif "Game1 start:" in line:
                    # Dual freeze logs a Game1 start per leg; do not treat that as a
                    # new single-mode run or we drop the frozen prefix.
                    if current["freeze_mode"] == "both" and current["leg"] in ("frozen", "unfrozen"):
                        continue
                    consider(current)
                    current = _empty_game1_ckpt()
                    current["harvested"] = True
                    current["freeze_mode"] = "single"
                    current["leg"] = "single"
                elif "Game1 dual: unfrozen leg starting" in line:
                    if not current["frozen_order"]:
                        current["frozen_order"] = list(current["selected_order"])
                    current["frozen_done"] = True
                    current["selected_order"] = []
                    current["first_faith_gain"] = None
                    current["leg"] = "unfrozen"
                    current["freeze_mode"] = "both"
                elif "Game1 finished:" in line:
                    if current["leg"] == "frozen":
                        current["frozen_done"] = True
                        current["frozen_order"] = list(current["selected_order"])
                elif "Game1 dual finished" in line:
                    current["frozen_done"] = True
                    if not current["frozen_order"] and current["leg"] != "unfrozen":
                        current["frozen_order"] = list(current["selected_order"])
                else:
                    match = added.search(line)
                    if not match:
                        continue
                    node = match.group(1)
                    size = int(match.group(3))
                    seq = current["selected_order"]
                    if len(seq) + 1 == size:
                        seq.append(node)
                    elif size <= len(seq):
                        seq[:] = seq[: size - 1] + [node]
                    else:
                        seq.append(node)
                    if current["leg"] == "frozen":
                        current["frozen_order"] = list(seq)
        consider(current)
    return best


def maybe_write_harvested_game1_checkpoint(
    dest: str | Path,
    log_paths: Sequence[str | Path],
) -> dict[str, Any] | None:
    """Write a harvested Game 1 ckpt only if it is strictly ahead of any existing file."""
    harvested = checkpoint_from_game1_logs(log_paths)
    if harvested is None:
        return None
    existing = _load_game1_checkpoint(dest)
    if existing is not None and game1_checkpoint_progress(existing) >= game1_checkpoint_progress(harvested):
        return existing
    _atomic_write_json(dest, harvested)
    return harvested


CandidatePrefilter = Callable[
    [CircuitGraph, ScoringOracle, TargetId, Sequence[NodeId], float, float, int],
    list[NodeId],
]


def _sort_key(node: NodeId) -> str:
    return str(node)


def prefilter_candidates(
    graph: CircuitGraph,
    oracle: ScoringOracle,
    target: TargetId,
    candidates: Sequence[NodeId],
    alpha: float,
    lam: float,
    top_k: int,
    connected: bool = False,
    progress: bool = True,
    log_every: int = 50,
    cap_sufficiency: bool = True,
    cap_necessity: bool = True,
) -> list[NodeId]:
    """Rank candidates by singleton gain and keep the top-k.

    The ranking is always computed so the output is deterministic regardless of
    `top_k` (L3). When `connected` is set, the retained pool is grown as a
    connected frontier instead of a raw rank truncation so the connected greedy is
    not handed a disconnected pool (L2).
    """
    if top_k <= 0:
        return []

    ranking: list[tuple[float, NodeId]] = []
    iterator: Sequence[NodeId] | Any = candidates
    if progress and tqdm is not None:
        iterator = tqdm(
            candidates,
            desc="Game1 prefilter",
            leave=True,
        )

    for idx, node in enumerate(iterator, start=1):
        if not graph.has_node(node):
            continue
        singleton = {node}
        metrics = compute_faithfulness_metrics(
            oracle=oracle,
            target=target,
            nodes=singleton,
            alpha=alpha,
            cap_sufficiency=cap_sufficiency,
            cap_necessity=cap_necessity,
        )
        utility = game1_utility(metrics.faithfulness_delta, size=1, lam=lam)
        ranking.append((utility, node))
        if progress and tqdm is None and log_every > 0 and idx % log_every == 0:
            LOGGER.info("Game1 prefilter evaluated %d/%d candidates", idx, len(candidates))

    ranking.sort(key=lambda item: (-item[0], _sort_key(item[1])))
    ranked_nodes = [node for _, node in ranking]
    if top_k >= len(ranked_nodes):
        return ranked_nodes
    if connected:
        return grow_connected_frontier(graph, ranked_nodes, top_k)
    return ranked_nodes[:top_k]


@dataclass
class EvidenceSetResult:
    evidence: set[NodeId]
    induced_subgraph: CircuitGraph
    metrics: FaithfulnessMetrics
    utility: float
    selected_order: list[NodeId]
    iterations: int
    candidate_count: int
    total_candidates: int
    params: dict[str, Any]
    oracle_calls: int
    cache_hits: int
    cache_size: int
    sparsity: float


def result_from_selected_order(
    graph: CircuitGraph,
    oracle: ScoringOracle,
    target: TargetId,
    selected_order: Sequence[NodeId],
    *,
    alpha: float,
    lam: float,
    total_candidates: int,
    candidate_count: int,
    params: Mapping[str, Any],
    cap_sufficiency: bool = True,
    cap_necessity: bool = True,
) -> EvidenceSetResult:
    """Rebuild Game 1 metrics for an already-chosen evidence order (no greedy)."""
    oracle.reset_stats()
    selected = {node for node in selected_order if graph.has_node(node)}
    order = [node for node in selected_order if node in selected]
    metrics = compute_faithfulness_metrics(
        oracle=oracle,
        target=target,
        nodes=selected,
        alpha=alpha,
        cap_sufficiency=cap_sufficiency,
        cap_necessity=cap_necessity,
    )
    utility = game1_utility(faithfulness_delta=metrics.faithfulness_delta, size=len(selected), lam=lam)
    stats = oracle.cache_stats()
    return EvidenceSetResult(
        evidence=selected,
        induced_subgraph=graph.subgraph(selected),
        metrics=metrics,
        utility=utility,
        selected_order=order,
        iterations=len(order),
        candidate_count=candidate_count,
        total_candidates=total_candidates,
        params=dict(params),
        oracle_calls=stats["oracle_calls"],
        cache_hits=stats["cache_hits"],
        cache_size=stats["cache_size"],
        sparsity=sparsity(selected_size=len(selected), total_size=max(1, total_candidates)),
    )


def solve_game1(
    graph: CircuitGraph,
    oracle: ScoringOracle,
    target: TargetId,
    candidates: Sequence[NodeId] | None = None,
    alpha: float = 0.5,
    lam: float = 0.01,
    budget: int | None = None,
    faithfulness_eps: float | None = None,
    stop_metric: str = "normalized",
    prefilter_top_k: int | None = None,
    prefilter_fn: CandidatePrefilter | None = None,
    connected: bool = False,
    min_gain: float = 0.0,
    fill_budget: bool = False,
    progress: bool = True,
    log_every: int = 50,
    checkpoint_path: str | Path | None = None,
    resume_selected_order: Sequence[NodeId] | None = None,
    checkpoint_meta: Mapping[str, Any] | None = None,
    cap_sufficiency: bool = True,
    cap_necessity: bool = True,
) -> EvidenceSetResult:
    """Greedy hill-climb solver for Game 1.

    `stop_metric` controls how `faithfulness_eps` is interpreted:
      * "normalized" (default): stop when faithfulness_delta_normalized >= 1 - eps.
        This divides by recoverable_range (all - empty), which is the correct
        error-floor-aware target with frozen attention but goes DEGENERATE when
        the range collapses toward zero/negative (e.g. unfrozen attention),
        producing spurious early/late stops.
      * "raw_relative": denominator-free diminishing-returns stop. Stop before
        adding a node whose marginal raw faithfulness gain (the alpha-mixed
        faithfulness_delta increase, EXCLUDING the lam size penalty) is
        < eps * (the first feature's faithfulness gain). Stable regardless of the
        error floor, so it is the correct choice when recoverable_range is
        unreliable (unfrozen attention). Selection itself still maximizes the
        lam-penalized utility; only the stop test is penalty-free.

    When ``fill_budget`` is True and ``budget`` is set, the solver keeps adding the
    best remaining candidate until ``|E|=budget`` even if its utility gain is
    non-positive (still prefers the least-negative gain). Useful when a
    same-size comparison against a known gold set is required.

    ``checkpoint_path`` writes ``macag_game1_ckpt_v1`` after each added node so a
    24h wall-clock kill can resume the greedy instead of restarting from |E|=0.
    Dual-freeze orchestration passes ``checkpoint_meta`` (leg / frozen_order).
    """
    if stop_metric not in ("normalized", "raw_relative"):
        raise ValueError("stop_metric must be 'normalized' or 'raw_relative'.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    if lam < 0.0:
        raise ValueError("lam must be non-negative.")
    if budget is not None and budget < 0:
        raise ValueError("budget must be non-negative when provided.")
    if min_gain < 0.0:
        raise ValueError("min_gain must be non-negative.")
    if fill_budget and budget is None:
        raise ValueError("fill_budget requires a finite budget.")

    # Per-solve stats: without this, reusing one oracle across several solves
    # (Game 1 + Game 2 on the same scorer, notebooks) reports cumulative counts.
    oracle.reset_stats()

    candidate_pool = dedupe_preserve_order(candidates if candidates is not None else graph.nodes())
    candidate_pool = [node for node in candidate_pool if graph.has_node(node)]
    # Full candidate count before any prefilter, so reported sparsity reflects the
    # true graph rather than the (possibly much smaller) prefiltered pool (I4).
    total_candidates = len(candidate_pool)
    if progress:
        LOGGER.info(
            "Game1 start: candidates=%d budget=%s alpha=%.3f lambda=%.4f",
            len(candidate_pool),
            budget,
            alpha,
            lam,
        )

    with nvtx_range("game1.solve"):
        return _solve_game1_body(
            graph=graph,
            oracle=oracle,
            target=target,
            candidate_pool=candidate_pool,
            total_candidates=total_candidates,
            alpha=alpha,
            lam=lam,
            budget=budget,
            faithfulness_eps=faithfulness_eps,
            stop_metric=stop_metric,
            prefilter_top_k=prefilter_top_k,
            prefilter_fn=prefilter_fn,
            connected=connected,
            min_gain=min_gain,
            fill_budget=fill_budget,
            progress=progress,
            log_every=log_every,
            checkpoint_path=checkpoint_path,
            resume_selected_order=resume_selected_order,
            checkpoint_meta=checkpoint_meta,
            cap_sufficiency=cap_sufficiency,
            cap_necessity=cap_necessity,
        )


def _solve_game1_body(
    graph: CircuitGraph,
    oracle: ScoringOracle,
    target: TargetId,
    candidate_pool: list[NodeId],
    total_candidates: int,
    alpha: float,
    lam: float,
    budget: int | None,
    faithfulness_eps: float | None,
    stop_metric: str,
    prefilter_top_k: int | None,
    prefilter_fn: CandidatePrefilter | None,
    connected: bool,
    min_gain: float,
    fill_budget: bool,
    progress: bool,
    log_every: int,
    checkpoint_path: str | Path | None,
    resume_selected_order: Sequence[NodeId] | None,
    checkpoint_meta: Mapping[str, Any] | None,
    cap_sufficiency: bool = True,
    cap_necessity: bool = True,
) -> EvidenceSetResult:
    if prefilter_top_k is not None:
        if prefilter_fn:
            candidate_pool = prefilter_fn(
                graph, oracle, target, candidate_pool, alpha, lam, prefilter_top_k
            )
        else:
            candidate_pool = prefilter_candidates(
                graph,
                oracle,
                target,
                candidate_pool,
                alpha,
                lam,
                prefilter_top_k,
                connected,
                progress=progress,
                log_every=log_every,
                cap_sufficiency=cap_sufficiency,
                cap_necessity=cap_necessity,
            )

    utility_cache: dict[frozenset[NodeId], float] = {}
    metric_cache: dict[frozenset[NodeId], FaithfulnessMetrics] = {}

    def evaluate(nodes: set[NodeId]) -> tuple[float, FaithfulnessMetrics]:
        key = frozenset(nodes)
        if key in utility_cache:
            return utility_cache[key], metric_cache[key]
        with nvtx_range("game1.evaluate"):
            metrics = compute_faithfulness_metrics(
                oracle=oracle,
                target=target,
                nodes=nodes,
                alpha=alpha,
                cap_sufficiency=cap_sufficiency,
                cap_necessity=cap_necessity,
            )
            utility = game1_utility(faithfulness_delta=metrics.faithfulness_delta, size=len(nodes), lam=lam)
            utility_cache[key] = utility
            metric_cache[key] = metrics
            return utility, metrics

    ckpt = _load_game1_checkpoint(checkpoint_path)
    meta = dict(checkpoint_meta or {})
    requested_leg = str(meta.get("leg") or (ckpt or {}).get("leg") or "single")
    freeze_mode = str(meta.get("freeze_mode") or (ckpt or {}).get("freeze_mode") or "single")
    frozen_done = bool(meta.get("frozen_done", (ckpt or {}).get("frozen_done", False)))
    frozen_order = list(meta.get("frozen_order") or (ckpt or {}).get("frozen_order") or [])
    first_faith_gain: float | None = None
    resume_order: list[NodeId] = []
    if resume_selected_order is not None:
        resume_order = [node for node in resume_selected_order if graph.has_node(node)]
        if ckpt is not None and ckpt.get("first_faith_gain") is not None:
            first_faith_gain = float(ckpt["first_faith_gain"])
    elif ckpt is not None:
        ckpt_leg = str(ckpt.get("leg") or "single")
        if requested_leg == "unfrozen":
            if ckpt_leg == "unfrozen":
                resume_order = [node for node in (ckpt.get("selected_order") or []) if graph.has_node(node)]
        elif requested_leg == "frozen":
            source = ckpt.get("selected_order") or ckpt.get("frozen_order") or []
            if not ckpt.get("frozen_done"):
                resume_order = [node for node in source if graph.has_node(node)]
        else:
            resume_order = [node for node in (ckpt.get("selected_order") or []) if graph.has_node(node)]
        if ckpt.get("first_faith_gain") is not None:
            first_faith_gain = float(ckpt["first_faith_gain"])
        ckpt_params = ckpt.get("params") or {}
        if ckpt_params:
            mismatches = []
            if ckpt_params.get("alpha") is not None and abs(float(ckpt_params["alpha"]) - alpha) > 1e-12:
                mismatches.append("alpha")
            if ckpt_params.get("lambda") is not None and abs(float(ckpt_params["lambda"]) - lam) > 1e-12:
                mismatches.append("lambda")
            if ckpt_params.get("stop_metric") not in (None, stop_metric):
                mismatches.append("stop_metric")
            if (
                ckpt_params.get("cap_sufficiency") is not None
                and bool(ckpt_params.get("cap_sufficiency")) != bool(cap_sufficiency)
            ):
                mismatches.append("cap_sufficiency")
            if mismatches:
                LOGGER.warning(
                    "Game1 checkpoint param mismatch at %s (%s); resuming anyway",
                    checkpoint_path,
                    ", ".join(mismatches),
                )

    selected_order: list[NodeId] = list(resume_order)
    selected: set[NodeId] = set(selected_order)
    iterations = len(selected_order)
    # Raw marginal faithfulness_delta gain (lam-free) of the first added node;
    # the raw_relative stop is defined on faithfulness gains, NOT utility gains,
    # so lam does not distort the eps-relative test.
    if first_faith_gain is None and selected_order:
        _, empty_metrics = evaluate(set())
        _, first_metrics = evaluate({selected_order[0]})
        first_faith_gain = first_metrics.faithfulness_delta - empty_metrics.faithfulness_delta

    def persist() -> None:
        if checkpoint_path is None:
            return
        order = list(selected_order)
        payload = {
            "schema": GAME1_CKPT_SCHEMA,
            "freeze_mode": freeze_mode,
            "leg": requested_leg,
            "frozen_done": frozen_done,
            "frozen_order": list(frozen_order) if requested_leg != "frozen" else list(order),
            "selected_order": order,
            "first_faith_gain": first_faith_gain,
            "params": {
                "alpha": alpha,
                "lambda": lam,
                "budget": budget,
                "faithfulness_eps": faithfulness_eps,
                "stop_metric": stop_metric,
                "prefilter_top_k": prefilter_top_k,
                "connected": connected,
                "min_gain": min_gain,
                "fill_budget": fill_budget,
                "cap_sufficiency": cap_sufficiency,
            },
        }
        _atomic_write_json(checkpoint_path, payload)

    if selected_order:
        if progress:
            LOGGER.info(
                "Game1 resume: leg=%s |E|=%d frozen_done=%s",
                requested_leg,
                len(selected_order),
                frozen_done,
            )
        persist()

    while True:
        if budget is not None and len(selected) >= budget:
            break

        current_utility, current_metrics = evaluate(selected)
        best_node: NodeId | None = None
        # Under fill_budget, track the best gain even if negative so we can still
        # grow to |E|=budget for same-size gold comparisons.
        best_gain = float("-inf") if fill_budget else min_gain
        best_faith_gain = 0.0

        iterator: Sequence[NodeId] | Any = candidate_pool
        if progress and tqdm is not None:
            iterator = tqdm(
                candidate_pool,
                desc=f"Game1 sweep |E|={len(selected)}",
                leave=False,
            )

        with nvtx_range(f"game1.step.|E|={len(selected)}"):
            for idx, node in enumerate(iterator, start=1):
                if node in selected:
                    continue
                trial = set(selected)
                trial.add(node)
                if connected and len(trial) > 1 and not graph.connected_through(trial):
                    continue
                trial_utility, trial_metrics = evaluate(trial)
                gain = trial_utility - current_utility
                if gain > best_gain:
                    best_gain = gain
                    best_node = node
                    best_faith_gain = trial_metrics.faithfulness_delta - current_metrics.faithfulness_delta
                elif gain == best_gain and best_node is not None and _sort_key(node) < _sort_key(best_node):
                    best_node = node
                    best_faith_gain = trial_metrics.faithfulness_delta - current_metrics.faithfulness_delta
                if progress and tqdm is None and log_every > 0 and idx % log_every == 0:
                    LOGGER.info("Game1 evaluated %d/%d candidates", idx, len(candidate_pool))

        if best_node is None:
            if progress:
                LOGGER.info("Game1 no improving candidate found (|E|=%d)", len(selected))
            break

        # Denominator-free diminishing-returns stop (before adding): the best
        # available feature contributes less than `eps` of the top feature's raw
        # marginal faithfulness gain. Uses lam-free faithfulness deltas so the
        # sparsity penalty cannot distort the eps-relative test; stable when
        # recoverable_range is unreliable. Disabled under fill_budget so we can
        # still reach a gold-matched set size.
        if (
            not fill_budget
            and faithfulness_eps is not None
            and stop_metric == "raw_relative"
            and first_faith_gain is not None
            and first_faith_gain > 0.0
            and best_faith_gain < faithfulness_eps * first_faith_gain
        ):
            if progress:
                LOGGER.info(
                    "Game1 raw_relative stop: best_faith_gain=%.6f < eps*first_faith_gain=%.6f (|E|=%d)",
                    best_faith_gain,
                    faithfulness_eps * first_faith_gain,
                    len(selected),
                )
            break

        selected.add(best_node)
        selected_order.append(best_node)
        iterations += 1
        if first_faith_gain is None:
            first_faith_gain = best_faith_gain
        if progress:
            LOGGER.info("Game1 added node=%s gain=%.6f |E|=%d", best_node, best_gain, len(selected))
        persist()

        if (
            not fill_budget
            and faithfulness_eps is not None
            and stop_metric == "normalized"
        ):
            _, metrics = evaluate(selected)
            # Stop on the SAME alpha-mixed objective the solver optimizes, in its
            # normalized (error-floor-aware) form (C2). faithfulness_delta_normalized
            # is in [0, 1]; reaching >= 1 - eps means the evidence recovers all but
            # `eps` of the achievable faithfulness. The previous condition tested only
            # the raw sufficiency gap, which is inconsistent when alpha != 1.
            # A2'': when R <= 0 the normalized ratio is >= 1 for any non-empty set
            # (sufficiency <= R < 0 divided by R), so it must NOT trigger the stop
            # — that would certify a singleton as "recovering everything".
            if metrics.is_degenerate:
                if progress:
                    LOGGER.warning(
                        "Game1 normalized stop skipped: recoverable_range=%.6f <= 0 "
                        "(degenerate cell, A2''); capped sufficiency cannot "
                        "discriminate here — use the unfrozen leg / necessity only.",
                        metrics.recoverable_range,
                    )
            elif metrics.faithfulness_delta_normalized >= 1.0 - faithfulness_eps:
                if progress:
                    LOGGER.info(
                        "Game1 reached faithfulness_eps=%.6f (normalized delta=%.6f)",
                        faithfulness_eps,
                        metrics.faithfulness_delta_normalized,
                    )
                break

    final_utility, final_metrics = evaluate(selected)
    stats = oracle.cache_stats()
    if progress:
        LOGGER.info(
            "Game1 finished: iterations=%d oracle_calls=%d cache_hits=%d",
            iterations,
            stats["oracle_calls"],
            stats["cache_hits"],
        )
    return EvidenceSetResult(
        evidence=selected,
        induced_subgraph=graph.subgraph(selected),
        metrics=final_metrics,
        utility=final_utility,
        selected_order=selected_order,
        iterations=iterations,
        candidate_count=len(candidate_pool),
        total_candidates=total_candidates,
        params={
            "alpha": alpha,
            "lambda": lam,
            "budget": budget,
            "faithfulness_eps": faithfulness_eps,
            "stop_metric": stop_metric,
            "prefilter_top_k": prefilter_top_k,
            "connected": connected,
            "min_gain": min_gain,
            "fill_budget": fill_budget,
            "cap_sufficiency": cap_sufficiency,
            "cap_necessity": cap_necessity,
        },
        oracle_calls=stats["oracle_calls"],
        cache_hits=stats["cache_hits"],
        cache_size=stats["cache_size"],
        sparsity=sparsity(selected_size=len(selected), total_size=max(1, total_candidates)),
    )
