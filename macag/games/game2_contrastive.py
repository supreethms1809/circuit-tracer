"""Game 2: contrastive two-agent evidence allocation via ABR or fictitious play."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import threading
from typing import Any, Callable, Mapping, Sequence

from macag.graph import CircuitGraph, NodeId, grow_connected_frontier
from macag.scoring import ScoringOracle, TargetId
from macag.utils.metrics import (
    FaithfulnessMetrics,
    compute_faithfulness_metrics,
    dedupe_preserve_order,
    game2_utility,
    overlap_rate,
    sparsity,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

LOGGER = logging.getLogger(__name__)

GAME2_CKPT_SCHEMA = "macag_game2_ckpt_v1"


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(dest)


def _load_game2_checkpoint(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    dest = Path(path)
    if not dest.is_file():
        return None
    data = json.loads(dest.read_text())
    if data.get("schema") != GAME2_CKPT_SCHEMA:
        LOGGER.warning("Ignoring game2 checkpoint with unknown schema at %s", dest)
        return None
    return data


def _sort_key(node: NodeId) -> str:
    return str(node)


def _overlap_weight(nodes: set[NodeId], other_weights: Mapping[NodeId, float]) -> float:
    """Expected overlap of `nodes` with the opponent.

    `other_weights` maps node -> inclusion weight in [0, 1]. Hard (ABR) opponents
    use weight 1.0 per selected node; fictitious-play opponents use empirical
    inclusion frequencies.
    """
    return sum(other_weights.get(node, 0.0) for node in nodes)


def _prefilter_with_overlap_penalty(
    graph: CircuitGraph,
    oracle: ScoringOracle,
    target: TargetId,
    fixed_other_weights: Mapping[NodeId, float],
    candidates: Sequence[NodeId],
    alpha: float,
    lam: float,
    beta: float,
    top_k: int,
    connected: bool = False,
) -> list[NodeId]:
    if top_k <= 0:
        return []

    ranking: list[tuple[float, NodeId]] = []
    for node in candidates:
        if not graph.has_node(node):
            continue
        singleton = {node}
        metrics = compute_faithfulness_metrics(oracle=oracle, target=target, nodes=singleton, alpha=alpha)
        utility = game2_utility(
            faithfulness_delta=metrics.faithfulness_delta,
            size=1,
            overlap_weight=_overlap_weight(singleton, fixed_other_weights),
            lam=lam,
            beta=beta,
        )
        ranking.append((utility, node))
    ranking.sort(key=lambda item: (-item[0], _sort_key(item[1])))
    ranked_nodes = [node for _, node in ranking]
    if top_k >= len(ranked_nodes):
        return ranked_nodes
    if connected:
        return grow_connected_frontier(graph, ranked_nodes, top_k)
    return ranked_nodes[:top_k]


def _best_response(
    graph: CircuitGraph,
    oracle: ScoringOracle,
    target: TargetId,
    fixed_other_weights: Mapping[NodeId, float],
    candidates: Sequence[NodeId],
    alpha: float,
    lam: float,
    beta: float,
    budget: int | None,
    connected: bool,
    min_gain: float,
    prefilter_top_k: int | None,
    progress: bool,
    log_every: int,
    progress_desc: str,
    initial_selected: set[NodeId] | None = None,
    on_add: Callable[[set[NodeId]], None] | None = None,
) -> set[NodeId]:
    candidate_pool = list(candidates)
    if prefilter_top_k is not None:
        candidate_pool = _prefilter_with_overlap_penalty(
            graph=graph,
            oracle=oracle,
            target=target,
            fixed_other_weights=fixed_other_weights,
            candidates=candidate_pool,
            alpha=alpha,
            lam=lam,
            beta=beta,
            top_k=prefilter_top_k,
            connected=connected,
        )

    utility_cache: dict[frozenset[NodeId], float] = {}

    def evaluate(nodes: set[NodeId]) -> float:
        key = frozenset(nodes)
        if key in utility_cache:
            return utility_cache[key]
        metrics = compute_faithfulness_metrics(oracle=oracle, target=target, nodes=nodes, alpha=alpha)
        utility = game2_utility(
            faithfulness_delta=metrics.faithfulness_delta,
            size=len(nodes),
            overlap_weight=_overlap_weight(nodes, fixed_other_weights),
            lam=lam,
            beta=beta,
        )
        utility_cache[key] = utility
        return utility

    selected: set[NodeId] = set(initial_selected or ())
    sweep = 0
    while True:
        sweep += 1
        if budget is not None and len(selected) >= budget:
            break

        current_utility = evaluate(selected)
        best_node: NodeId | None = None
        best_gain = min_gain

        iterator: Sequence[NodeId] | Any = candidate_pool
        if progress and tqdm is not None:
            iterator = tqdm(
                candidate_pool,
                desc=f"{progress_desc} sweep={sweep} |E|={len(selected)}",
                leave=False,
            )

        for idx, node in enumerate(iterator, start=1):
            if node in selected:
                continue
            trial = set(selected)
            trial.add(node)
            if connected and len(trial) > 1 and not graph.connected_through(trial):
                continue
            gain = evaluate(trial) - current_utility
            if gain > best_gain:
                best_gain = gain
                best_node = node
            elif gain == best_gain and best_node is not None and _sort_key(node) < _sort_key(best_node):
                best_node = node
            if progress and tqdm is None and log_every > 0 and idx % log_every == 0:
                LOGGER.info(
                    "[%s] sweep %d evaluated %d/%d candidates",
                    progress_desc,
                    sweep,
                    idx,
                    len(candidate_pool),
                )

        if best_node is None:
            if progress:
                LOGGER.info(
                    "[%s] no improving candidate found (|E|=%d)",
                    progress_desc,
                    len(selected),
                )
            break
        selected.add(best_node)
        if on_add is not None:
            on_add(set(selected))
        if progress:
            LOGGER.info(
                "[%s] added node=%s gain=%.6f |E|=%d",
                progress_desc,
                best_node,
                best_gain,
                len(selected),
            )

    return selected


@dataclass
class ContrastiveEvidenceResult:
    evidence_y: set[NodeId]
    evidence_foil: set[NodeId]
    shared: set[NodeId]
    unique_y: set[NodeId]
    unique_foil: set[NodeId]
    metrics_y: FaithfulnessMetrics
    metrics_foil: FaithfulnessMetrics
    utility_y: float
    utility_foil: float
    overlap_rate: float
    iterations: int
    converged: bool
    # Round whose joint allocation is returned (best combined hard-overlap
    # utility). 0 means the initial empty allocation beat every solver round —
    # i.e. penalties outweighed faithfulness in all realized joint allocations.
    best_iteration: int
    params: dict[str, Any]
    oracle_calls: int
    cache_hits: int
    cache_size: int
    total_candidates: int
    sparsity_y: float
    sparsity_foil: float
    # Fictitious play only: empirical inclusion frequency per node across rounds
    # (soft evidence membership). Empty for the ABR solver.
    node_frequencies_y: dict[str, float] = field(default_factory=dict)
    node_frequencies_foil: dict[str, float] = field(default_factory=dict)


def _empirical_frequencies(counts: Mapping[NodeId, int], rounds: int) -> dict[NodeId, float]:
    if rounds <= 0:
        return {}
    return {node: count / rounds for node, count in counts.items()}


def _max_frequency_change(
    old: Mapping[NodeId, float], new: Mapping[NodeId, float]
) -> float:
    keys = set(old) | set(new)
    if not keys:
        return 0.0
    return max(abs(new.get(key, 0.0) - old.get(key, 0.0)) for key in keys)


def solve_game2(
    graph: CircuitGraph,
    oracle: ScoringOracle,
    y: TargetId,
    y_foil: TargetId,
    candidates: Sequence[NodeId] | None = None,
    alpha: float = 0.5,
    lam: float = 0.01,
    beta: float = 0.1,
    abr_iters: int = 10,
    budget: int | None = None,
    connected: bool = False,
    min_gain: float = 0.0,
    prefilter_top_k: int | None = None,
    solver: str = "abr",
    fp_tol: float = 1e-3,
    progress: bool = True,
    log_every: int = 50,
    checkpoint_path: str | Path | None = None,
    parallel_agents: bool = True,
) -> ContrastiveEvidenceResult:
    """Contrastive evidence allocation via ABR or fictitious play.

    `solver` selects the update rule (both use Jacobi-style simultaneous updates
    capped at `abr_iters` rounds):
      * "abr" (default): each agent best-responds to the opponent's LAST evidence
        set. Can 2-cycle; mitigated by best-iterate tracking.
      * "fp" (fictitious play): each agent best-responds to the opponent's
        EMPIRICAL HISTORY of evidence sets. Because the opponent only enters the
        utility through the overlap penalty, the expected utility against the
        empirical mixture is exact: beta * sum_{n in E} p_t(n), where p_t(n) is
        the fraction of past rounds the opponent included node n. No extra
        oracle calls are needed. Stops early when the empirical frequencies of
        both agents change by less than `fp_tol` (or the best responses repeat).

    Within each round the two best-responses are independent given the frozen
    opponent, so they run concurrently (`parallel_agents=True`) and only
    communicate at the Jacobi barrier: join sets, score the joint allocation,
    update FP frequencies, test convergence, then persist. Mid-round resume
    with ``phase=foil`` stays sequential so a completed y-side is not recomputed.
    On one GPU the shared ReplacementModel is serialized inside ScoringOracle;
    the barrier is still required for correctness (independent jobs with no
    barrier would be β=0, two Game 1s).

    Reported metrics/utilities always use the HARD overlap of the returned joint
    allocation, so ABR and FP results are directly comparable.
    """
    if solver not in ("abr", "fp"):
        raise ValueError("solver must be 'abr' or 'fp'.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    if lam < 0.0:
        raise ValueError("lam must be non-negative.")
    if beta < 0.0:
        raise ValueError("beta must be non-negative.")
    if abr_iters <= 0:
        raise ValueError("abr_iters must be a positive integer.")
    if budget is not None and budget < 0:
        raise ValueError("budget must be non-negative when provided.")
    if min_gain < 0.0:
        raise ValueError("min_gain must be non-negative.")
    if fp_tol < 0.0:
        raise ValueError("fp_tol must be non-negative.")

    # Per-solve stats: without this, reusing one oracle across several solves
    # (Game 1 + Game 2 on the same scorer, notebooks) reports cumulative counts.
    oracle.reset_stats()

    candidate_pool = dedupe_preserve_order(candidates if candidates is not None else graph.nodes())
    candidate_pool = [node for node in candidate_pool if graph.has_node(node)]
    # Full candidate count before per-agent prefilters, so reported sparsity reflects
    # the true graph rather than the prefiltered pool (I4).
    total_candidates = len(candidate_pool)
    if progress:
        LOGGER.info(
            "Game2 start: solver=%s candidates=%d max_iters=%d budget=%s "
            "alpha=%.3f lambda=%.4f beta=%.4f parallel_agents=%s",
            solver,
            len(candidate_pool),
            abr_iters,
            budget,
            alpha,
            lam,
            beta,
            parallel_agents,
        )

    def evaluate_allocation(
        e_y: set[NodeId], e_foil: set[NodeId]
    ) -> tuple[FaithfulnessMetrics, FaithfulnessMetrics, float, float, float]:
        """Score a joint allocation: per-agent metrics, utilities, combined utility."""
        shared_nodes = e_y & e_foil
        m_y = compute_faithfulness_metrics(oracle=oracle, target=y, nodes=e_y, alpha=alpha)
        m_foil = compute_faithfulness_metrics(oracle=oracle, target=y_foil, nodes=e_foil, alpha=alpha)
        u_y = game2_utility(
            faithfulness_delta=m_y.faithfulness_delta,
            size=len(e_y),
            overlap_weight=float(len(shared_nodes)),
            lam=lam,
            beta=beta,
        )
        u_foil = game2_utility(
            faithfulness_delta=m_foil.faithfulness_delta,
            size=len(e_foil),
            overlap_weight=float(len(shared_nodes)),
            lam=lam,
            beta=beta,
        )
        return m_y, m_foil, u_y, u_foil, u_y + u_foil

    evidence_y: set[NodeId] = set()
    evidence_foil: set[NodeId] = set()
    converged = False
    iterations = 0
    solver_label = solver.upper()

    # Fictitious play state: per-agent inclusion counts across past rounds and the
    # empirical frequencies derived from them. Round 1 responds to an empty
    # history, exactly matching ABR round 1.
    counts_y: dict[NodeId, int] = {}
    counts_foil: dict[NodeId, int] = {}
    freq_y: dict[NodeId, float] = {}
    freq_foil: dict[NodeId, float] = {}

    # Best-iterate tracking (C3): neither ABR nor FP is guaranteed to converge to a
    # pure equilibrium (ABR can 2-cycle), so retain the highest combined-utility
    # joint allocation seen across all rounds and return that rather than the final
    # iterate. The combined utility always uses the HARD overlap so ABR and FP runs
    # stay directly comparable.
    best_eval = evaluate_allocation(evidence_y, evidence_foil)
    best_evidence_y, best_evidence_foil = set(evidence_y), set(evidence_foil)
    best_combined = best_eval[4]
    best_iteration = 0

    ckpt = _load_game2_checkpoint(checkpoint_path)
    start_iter = 1
    start_phase = "y"
    resume_next_y: set[NodeId] = set()
    resume_next_foil: set[NodeId] = set()
    if ckpt is not None:
        start_iter = max(1, int(ckpt.get("iteration") or 1))
        start_phase = str(ckpt.get("phase") or "y")
        evidence_y = set(ckpt.get("frozen_y") or [])
        evidence_foil = set(ckpt.get("frozen_foil") or [])
        resume_next_y = set(ckpt.get("next_y") or [])
        resume_next_foil = set(ckpt.get("next_foil") or [])
        best_evidence_y = set(ckpt.get("best_y") or best_evidence_y)
        best_evidence_foil = set(ckpt.get("best_foil") or best_evidence_foil)
        best_iteration = int(ckpt.get("best_iteration") or 0)
        if ckpt.get("best_combined") is not None:
            best_combined = float(ckpt["best_combined"])
            best_eval = evaluate_allocation(best_evidence_y, best_evidence_foil)
        if progress:
            LOGGER.info(
                "Game2 resume: iter=%d phase=%s |frozen_y|=%d |frozen_foil|=%d |next_y|=%d |next_foil|=%d",
                start_iter,
                start_phase,
                len(evidence_y),
                len(evidence_foil),
                len(resume_next_y),
                len(resume_next_foil),
            )

    progress_y: set[NodeId] = set(resume_next_y)
    progress_foil: set[NodeId] = set(resume_next_foil)
    persist_lock = threading.Lock()

    def persist(
        iteration: int,
        phase: str,
        *,
        next_y: set[NodeId] | None = None,
        next_foil: set[NodeId] | None = None,
    ) -> None:
        if checkpoint_path is None:
            return
        with persist_lock:
            if next_y is not None:
                progress_y.clear()
                progress_y.update(next_y)
            if next_foil is not None:
                progress_foil.clear()
                progress_foil.update(next_foil)
            _atomic_write_json(
                checkpoint_path,
                {
                    "schema": GAME2_CKPT_SCHEMA,
                    "solver": solver,
                    "abr_iters": abr_iters,
                    "iteration": iteration,
                    "phase": phase,
                    "frozen_y": sorted(evidence_y, key=_sort_key),
                    "frozen_foil": sorted(evidence_foil, key=_sort_key),
                    "next_y": sorted(progress_y, key=_sort_key),
                    "next_foil": sorted(progress_foil, key=_sort_key),
                    "best_y": sorted(best_evidence_y, key=_sort_key),
                    "best_foil": sorted(best_evidence_foil, key=_sort_key),
                    "best_combined": best_combined,
                    "best_iteration": best_iteration,
                },
            )

    def _agent_best_response(
        target: TargetId,
        opponent_weights: Mapping[NodeId, float],
        seed: set[NodeId],
        progress_desc: str,
        on_add: Callable[[set[NodeId]], None] | None,
    ) -> set[NodeId]:
        return _best_response(
            graph=graph,
            oracle=oracle,
            target=target,
            fixed_other_weights=opponent_weights,
            candidates=candidate_pool,
            alpha=alpha,
            lam=lam,
            beta=beta,
            budget=budget,
            connected=connected,
            min_gain=min_gain,
            prefilter_top_k=prefilter_top_k,
            progress=progress,
            log_every=log_every,
            progress_desc=progress_desc,
            initial_selected=seed,
            on_add=on_add,
        )

    for iteration in range(start_iter, abr_iters + 1):
        if progress:
            LOGGER.info(
                "%s iteration %d/%d start: |E_y|=%d |E_foil|=%d",
                solver_label,
                iteration,
                abr_iters,
                len(evidence_y),
                len(evidence_foil),
            )
        # Symmetric (Jacobi) update (C3): both agents best-respond to the SAME frozen
        # opponent from the previous round, removing the Gauss-Seidel asymmetry where
        # the foil saw the freshly-updated y but y saw a stale foil.
        if solver == "fp":
            # Best-respond to the opponent's empirical mixture of past sets.
            opponent_for_y: dict[NodeId, float] = dict(freq_foil)
            opponent_for_foil: dict[NodeId, float] = dict(freq_y)
        else:
            # ABR: hard 0/1 weights from the opponent's last iterate.
            opponent_for_y = {node: 1.0 for node in evidence_foil}
            opponent_for_foil = {node: 1.0 for node in evidence_y}

        skip_y = iteration == start_iter and start_phase == "foil"
        resume_this_round = iteration == start_iter
        y_seed = (
            resume_next_y
            if resume_this_round and start_phase in {"y", "both"}
            else set()
        )
        foil_seed = (
            resume_next_foil
            if resume_this_round and start_phase in {"foil", "both"}
            else set()
        )
        # Parallelize only when both agents still have to run this round.
        parallel_this_round = bool(parallel_agents) and not skip_y

        if skip_y:
            next_y = set(resume_next_y)
            persist(iteration, "foil", next_y=next_y, next_foil=foil_seed)
            next_foil = _agent_best_response(
                y_foil,
                opponent_for_foil,
                foil_seed,
                f"{solver_label}[{iteration}] foil",
                lambda selected, it=iteration: persist(it, "foil", next_foil=selected),
            )
        elif parallel_this_round:
            if progress:
                LOGGER.info(
                    "%s iteration %d: parallel y || foil; Jacobi barrier after both finish",
                    solver_label,
                    iteration,
                )
            persist(iteration, "both", next_y=y_seed, next_foil=foil_seed)
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="game2") as pool:
                fut_y = pool.submit(
                    _agent_best_response,
                    y,
                    opponent_for_y,
                    y_seed,
                    f"{solver_label}[{iteration}] y",
                    lambda selected, it=iteration: persist(it, "both", next_y=selected),
                )
                fut_foil = pool.submit(
                    _agent_best_response,
                    y_foil,
                    opponent_for_foil,
                    foil_seed,
                    f"{solver_label}[{iteration}] foil",
                    lambda selected, it=iteration: persist(it, "both", next_foil=selected),
                )
                next_y = fut_y.result()
                next_foil = fut_foil.result()
            if progress:
                LOGGER.info(
                    "%s iteration %d: barrier — exchanging |E_y|=%d |E_foil|=%d",
                    solver_label,
                    iteration,
                    len(next_y),
                    len(next_foil),
                )
        else:
            persist(iteration, "y", next_y=y_seed, next_foil=set())
            next_y = _agent_best_response(
                y,
                opponent_for_y,
                y_seed,
                f"{solver_label}[{iteration}] y",
                lambda selected, it=iteration: persist(it, "y", next_y=selected),
            )
            persist(iteration, "foil", next_y=next_y, next_foil=foil_seed)
            next_foil = _agent_best_response(
                y_foil,
                opponent_for_foil,
                foil_seed,
                f"{solver_label}[{iteration}] foil",
                lambda selected, it=iteration: persist(it, "foil", next_foil=selected),
            )

        iterations = iteration

        cur_eval = evaluate_allocation(next_y, next_foil)
        if cur_eval[4] > best_combined:
            best_combined = cur_eval[4]
            best_eval = cur_eval
            best_evidence_y, best_evidence_foil = set(next_y), set(next_foil)
            best_iteration = iteration

        if progress:
            LOGGER.info(
                "%s iteration %d done: next |E_y|=%d next |E_foil|=%d combined_utility=%.6f",
                solver_label,
                iteration,
                len(next_y),
                len(next_foil),
                cur_eval[4],
            )

        if solver == "fp":
            for node in next_y:
                counts_y[node] = counts_y.get(node, 0) + 1
            for node in next_foil:
                counts_foil[node] = counts_foil.get(node, 0) + 1
            new_freq_y = _empirical_frequencies(counts_y, iteration)
            new_freq_foil = _empirical_frequencies(counts_foil, iteration)
            freq_change = max(
                _max_frequency_change(freq_y, new_freq_y),
                _max_frequency_change(freq_foil, new_freq_foil),
            )
            freq_y = new_freq_y
            freq_foil = new_freq_foil
            if next_y == evidence_y and next_foil == evidence_foil:
                converged = True
                if progress:
                    LOGGER.info("FP best responses repeated at iteration %d", iteration)
            elif iteration > 1 and freq_change < fp_tol:
                converged = True
                if progress:
                    LOGGER.info(
                        "FP empirical frequencies stabilized at iteration %d (max change %.6f < fp_tol %.6f)",
                        iteration,
                        freq_change,
                        fp_tol,
                    )
            if converged:
                evidence_y = next_y
                evidence_foil = next_foil
                break
        elif next_y == evidence_y and next_foil == evidence_foil:
            converged = True
            if progress:
                LOGGER.info("ABR converged at iteration %d", iteration)
            break

        evidence_y = next_y
        evidence_foil = next_foil
        persist(iteration + 1, "y", next_y=set(), next_foil=set())

    # Return the best joint allocation seen, not the final iterate (C3).
    evidence_y = best_evidence_y
    evidence_foil = best_evidence_foil
    metrics_y, metrics_foil, utility_y, utility_foil, _ = best_eval
    if best_iteration == 0 and iterations > 0 and (next_y or next_foil):
        LOGGER.warning(
            "Game2 returning the initial EMPTY allocation: solver rounds produced "
            "non-empty evidence, but every round's joint allocation had combined "
            "utility <= 0 under the hard-overlap evaluation (lam=%.4f, beta=%.4f "
            "penalties outweighed realized faithfulness). Consider lowering lam/beta "
            "or inspecting per-round logs.",
            lam,
            beta,
        )

    shared = evidence_y & evidence_foil
    unique_y = evidence_y - evidence_foil
    unique_foil = evidence_foil - evidence_y

    stats = oracle.cache_stats()
    if progress:
        LOGGER.info(
            "Game2 finished: solver=%s iterations=%d converged=%s oracle_calls=%d cache_hits=%d",
            solver,
            iterations,
            converged,
            stats["oracle_calls"],
            stats["cache_hits"],
        )
    return ContrastiveEvidenceResult(
        evidence_y=evidence_y,
        evidence_foil=evidence_foil,
        shared=shared,
        unique_y=unique_y,
        unique_foil=unique_foil,
        metrics_y=metrics_y,
        metrics_foil=metrics_foil,
        utility_y=utility_y,
        utility_foil=utility_foil,
        overlap_rate=overlap_rate(evidence_y, evidence_foil),
        iterations=iterations,
        converged=converged,
        best_iteration=best_iteration,
        params={
            "alpha": alpha,
            "lambda": lam,
            "beta": beta,
            "abr_iters": abr_iters,
            "budget": budget,
            "connected": connected,
            "prefilter_top_k": prefilter_top_k,
            "min_gain": min_gain,
            "solver": solver,
            "fp_tol": fp_tol if solver == "fp" else None,
            "parallel_agents": parallel_agents,
        },
        oracle_calls=stats["oracle_calls"],
        cache_hits=stats["cache_hits"],
        cache_size=stats["cache_size"],
        total_candidates=total_candidates,
        sparsity_y=sparsity(selected_size=len(evidence_y), total_size=max(1, total_candidates)),
        sparsity_foil=sparsity(selected_size=len(evidence_foil), total_size=max(1, total_candidates)),
        node_frequencies_y={str(node): freq for node, freq in sorted(freq_y.items(), key=lambda kv: str(kv[0]))},
        node_frequencies_foil={str(node): freq for node, freq in sorted(freq_foil.items(), key=lambda kv: str(kv[0]))},
    )


def checkpoint_from_abr_logs(
    log_paths: Sequence[str | Path],
    *,
    solver: str = "abr",
    abr_iters: int = 4,
) -> dict[str, Any] | None:
    """Rebuild an ABR checkpoint from ``added node=`` lines in solver logs.

    Jacobi iteration 1 best-responds to the empty opponent, so Austin and foil
    sequences from different 24h shards of the same config can be merged by
    taking the longest add-list per side.
    """
    import re

    pat = re.compile(
        r"\[ABR\[(\d+)\] (y|foil)\] added node=(\S+) gain=([-\d.eE]+) \|E\|=(\d+)"
    )
    y_by_iter: dict[int, list[str]] = {}
    foil_by_iter: dict[int, list[str]] = {}
    for path in log_paths:
        p = Path(path)
        if not p.is_file():
            continue
        y_cur: dict[int, list[str]] = {}
        foil_cur: dict[int, list[str]] = {}
        with p.open(errors="replace") as handle:
            for line in handle:
                match = pat.search(line)
                if not match:
                    continue
                iteration = int(match.group(1))
                side = match.group(2)
                node = match.group(3)
                size = int(match.group(5))
                bucket = y_cur if side == "y" else foil_cur
                seq = bucket.setdefault(iteration, [])
                if len(seq) + 1 == size:
                    seq.append(node)
                elif size <= len(seq):
                    seq[:] = seq[: size - 1] + [node]
                else:
                    seq.append(node)
        for iteration, seq in y_cur.items():
            if len(seq) > len(y_by_iter.get(iteration, [])):
                y_by_iter[iteration] = seq
        for iteration, seq in foil_cur.items():
            if len(seq) > len(foil_by_iter.get(iteration, [])):
                foil_by_iter[iteration] = seq
    if not y_by_iter and not foil_by_iter:
        return None
    max_iter = max([*y_by_iter, *foil_by_iter])
    # Resume the latest incomplete Jacobi round (iter 1 if that is all we have).
    next_y = list(y_by_iter.get(max_iter, []))
    next_foil = list(foil_by_iter.get(max_iter, []))
    phase = "foil" if next_foil or max_iter in foil_by_iter else "y"
    frozen_y: list[str] = list(y_by_iter.get(max_iter - 1, [])) if max_iter > 1 else []
    frozen_foil: list[str] = list(foil_by_iter.get(max_iter - 1, [])) if max_iter > 1 else []
    return {
        "schema": GAME2_CKPT_SCHEMA,
        "solver": solver,
        "abr_iters": abr_iters,
        "iteration": max_iter,
        "phase": phase,
        "frozen_y": frozen_y,
        "frozen_foil": frozen_foil,
        "next_y": next_y,
        "next_foil": next_foil,
        "best_y": next_y,
        "best_foil": next_foil,
        "best_combined": None,
        "best_iteration": 0,
        "harvested": True,
    }
