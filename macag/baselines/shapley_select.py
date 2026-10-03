"""B2.2 — Monte-Carlo Shapley (and Banzhaf) over the MACAG intervention oracle.

This is the gold-standard credit reference of macag.md §3.6 / §A.4: the value
function is the SAME coalitional v(S) the games optimize — the alpha-mixed
faithfulness over keep_only/remove ablations through the oracle — NOT the
spline-CLT reconstruction game in attribution/shapley.py, which measures a
different v on a different object and must not be wrapped here.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from macag.baselines.common import SelectionResult, coalition_value, ranking_from_scores
from macag.graph import NodeId
from macag.scoring import ScoringOracle, TargetId
from macag.utils.metrics import dedupe_preserve_order

LOGGER = logging.getLogger(__name__)
SHAPLEY_CKPT_SCHEMA = "macag_shapley_ckpt_v1"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload) + "\n")
    tmp.replace(path)


def _rng_state_to_json(state: tuple[Any, ...]) -> list[Any]:
    version, ints, gauss = state
    return [version, list(ints), gauss]


def _rng_state_from_json(blob: Sequence[Any]) -> tuple[Any, ...]:
    version, ints, gauss = blob
    return (version, tuple(ints), gauss)


@dataclass
class ShapleyEstimate:
    """Per-node credit estimates over the MACAG coalitional game."""

    values: dict[NodeId, float]
    std_errors: dict[NodeId, float]
    samples_per_node: int
    estimator: str
    seed: int
    base_value: float
    grand_value: float
    # sum(values) - (grand_value - base_value); each sampled Shapley ordering
    # telescopes exactly, so this is an implementation invariant, not convergence.
    efficiency_gap: float | None
    params: dict[str, object] = field(default_factory=dict)

    def ranking(self) -> list[NodeId]:
        return ranking_from_scores(self.values)


def _finalize(
    sums: dict[NodeId, float],
    pool: list[NodeId],
    count: int,
    estimator: str,
    seed: int,
    base_value: float,
    grand_value: float,
    params: dict[str, object],
    *,
    se_sums: dict[NodeId, float],
    se_sumsqs: dict[NodeId, float],
    independent_draws: int,
) -> ShapleyEstimate:
    """Finalize means and standard errors.

    ``count`` is the number of orderings/samples accumulated into sums.
    ``se_sums`` / ``se_sumsqs`` accumulate one value per independent RNG draw.
    For antithetic Shapley that value is the mean marginal across a permutation
    and its reverse, preserving their covariance in the standard error.
    """
    n_se = independent_draws
    values: dict[NodeId, float] = {}
    std_errors: dict[NodeId, float] = {}
    for node in pool:
        mean = sums[node] / count
        if n_se > 1:
            draw_mean = se_sums[node] / n_se
            centered_ss = se_sumsqs[node] - n_se * draw_mean * draw_mean
            sample_variance = max(0.0, centered_ss / (n_se - 1))
            std_errors[node] = math.sqrt(sample_variance / n_se)
        else:
            std_errors[node] = float("nan")  # undefined for a single draw
        values[node] = mean
    # Every sampled Shapley permutation telescopes, so this is a structural
    # implementation check—not a convergence diagnostic. Banzhaf has no
    # efficiency axiom and therefore does not report the field.
    efficiency_gap = (
        sum(values.values()) - (grand_value - base_value)
        if estimator == "shapley"
        else None
    )
    return ShapleyEstimate(
        values=values,
        std_errors=std_errors,
        samples_per_node=count,
        estimator=estimator,
        seed=seed,
        base_value=base_value,
        grand_value=grand_value,
        efficiency_gap=efficiency_gap,
        params=dict(params) | {
            "independent_draws": n_se,
            "uncertainty_unit": (
                "antithetic_pair_mean"
                if bool(params.get("antithetic"))
                else "independent_sample"
            ),
        },
    )


def estimate_shapley(
    oracle: ScoringOracle,
    target: TargetId,
    candidates: Sequence[NodeId],
    alpha: float = 0.5,
    permutations: int = 64,
    seed: int = 0,
    antithetic: bool = True,
    progress: bool = False,
    checkpoint_path: str | Path | None = None,
) -> ShapleyEstimate:
    """Permutation-sampling Shapley estimator with optional antithetic pairing.

    Each permutation walks the pool front-to-back, charging every node its
    marginal v-gain at the coalition built so far: |C| coalition evaluations
    per permutation, each costing at most two fresh oracle calls (keep_only and
    remove; all/empty are cached after the first). Antithetic pairing runs the
    reversed permutation alongside, which cancels order noise for near-additive
    v. Deterministic for a fixed seed. ``checkpoint_path`` writes after every
    antithetic pair so a 24h kill can resume without rerunning finished perms.
    """
    if permutations <= 0:
        raise ValueError("permutations must be positive.")
    if antithetic and permutations % 2:
        raise ValueError("Antithetic Shapley requires an even number of permutations.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    pool = dedupe_preserve_order(candidates)
    if not pool:
        raise ValueError("Shapley estimation needs a non-empty candidate pool.")

    rng = random.Random(seed)
    sums: dict[NodeId, float] = {node: 0.0 for node in pool}
    se_sums: dict[NodeId, float] = {node: 0.0 for node in pool}
    se_sumsqs: dict[NodeId, float] = {node: 0.0 for node in pool}
    base_value = coalition_value(oracle, target, set(), alpha)

    runs = 0
    independent_draws = 0
    ckpt_file = Path(checkpoint_path) if checkpoint_path else None
    if ckpt_file is not None and ckpt_file.is_file():
        data = json.loads(ckpt_file.read_text())
        if data.get("schema") == SHAPLEY_CKPT_SCHEMA and data.get("pool") == list(pool):
            rng.setstate(_rng_state_from_json(data["rng_state"]))
            sums = {node: float(data["sums"][node]) for node in pool}
            se_sums = {node: float(data["se_sums"][node]) for node in pool}
            se_sumsqs = {node: float(data["se_sumsqs"][node]) for node in pool}
            runs = int(data["runs"])
            independent_draws = int(data["independent_draws"])
            LOGGER.info("Shapley resume: %d/%d permutations", runs, permutations)
        else:
            LOGGER.warning("Ignoring Shapley checkpoint at %s (schema/pool mismatch)", ckpt_file)

    def persist() -> None:
        if ckpt_file is None:
            return
        _atomic_write_json(
            ckpt_file,
            {
                "schema": SHAPLEY_CKPT_SCHEMA,
                "seed": seed,
                "permutations": permutations,
                "antithetic": antithetic,
                "alpha": alpha,
                "pool": list(pool),
                "runs": runs,
                "independent_draws": independent_draws,
                "rng_state": _rng_state_to_json(rng.getstate()),
                "sums": sums,
                "se_sums": se_sums,
                "se_sumsqs": se_sumsqs,
            },
        )

    while runs < permutations:
        permutation = rng.sample(pool, len(pool))
        batch = [permutation]
        if antithetic and runs + 1 < permutations:
            batch.append(list(reversed(permutation)))
        draw_marginals: list[dict[NodeId, float]] = []
        for ordering in batch:
            previous = base_value
            coalition: set[NodeId] = set()
            ordering_marginals: dict[NodeId, float] = {}
            for node in ordering:
                coalition.add(node)
                value = coalition_value(oracle, target, coalition, alpha)
                marginal = value - previous
                sums[node] += marginal
                ordering_marginals[node] = marginal
                previous = value
            draw_marginals.append(ordering_marginals)
            runs += 1
        for node in pool:
            draw_value = sum(item[node] for item in draw_marginals) / len(draw_marginals)
            se_sums[node] += draw_value
            se_sumsqs[node] += draw_value * draw_value
        independent_draws += 1
        persist()
        if progress and runs % 8 == 0:
            LOGGER.info("Shapley: %d/%d permutations", runs, permutations)

    grand_value = coalition_value(oracle, target, set(pool), alpha)
    return _finalize(
        sums,
        pool,
        runs,
        estimator="shapley",
        seed=seed,
        base_value=base_value,
        grand_value=grand_value,
        params={"alpha": alpha, "permutations": runs, "antithetic": antithetic},
        se_sums=se_sums,
        se_sumsqs=se_sumsqs,
        independent_draws=independent_draws,
    )


def estimate_banzhaf(
    oracle: ScoringOracle,
    target: TargetId,
    candidates: Sequence[NodeId],
    alpha: float = 0.5,
    samples: int = 64,
    seed: int = 0,
    progress: bool = False,
) -> ShapleyEstimate:
    """Monte-Carlo Banzhaf: marginals against uniformly random coalitions.

    Per sample, draws S by independent fair coin flips and charges every node
    v(S + node) - v(S - node). Less order-sensitive than Shapley for strongly
    interacting features (macag.md §3.6); the natural second gold reference.
    """
    if samples <= 0:
        raise ValueError("samples must be positive.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    pool = dedupe_preserve_order(candidates)
    if not pool:
        raise ValueError("Banzhaf estimation needs a non-empty candidate pool.")

    rng = random.Random(seed)
    sums: dict[NodeId, float] = {node: 0.0 for node in pool}
    sumsqs: dict[NodeId, float] = {node: 0.0 for node in pool}
    base_value = coalition_value(oracle, target, set(), alpha)

    for run in range(1, samples + 1):
        sample = {node for node in pool if rng.random() < 0.5}
        for node in pool:
            with_node = sample | {node}
            without_node = sample - {node}
            marginal = coalition_value(oracle, target, with_node, alpha) - coalition_value(
                oracle, target, without_node, alpha
            )
            sums[node] += marginal
            sumsqs[node] += marginal * marginal
        if progress and run % 8 == 0:
            LOGGER.info("Banzhaf: %d/%d samples", run, samples)

    grand_value = coalition_value(oracle, target, set(pool), alpha)
    return _finalize(
        sums,
        pool,
        samples,
        estimator="banzhaf",
        seed=seed,
        base_value=base_value,
        grand_value=grand_value,
        params={"alpha": alpha, "samples": samples},
        se_sums=sums,
        se_sumsqs=sumsqs,
        independent_draws=samples,
    )


def select_top_shapley(
    oracle: ScoringOracle,
    target: TargetId,
    candidates: Sequence[NodeId],
    alpha: float = 0.5,
    permutations: int = 64,
    seed: int = 0,
    antithetic: bool = True,
    estimator: str = "shapley",
    progress: bool = False,
    checkpoint_path: str | Path | None = None,
) -> SelectionResult:
    """Rank candidates by estimated Shapley (or Banzhaf) value, best first."""
    if estimator == "shapley":
        estimate = estimate_shapley(
            oracle,
            target,
            candidates,
            alpha=alpha,
            permutations=permutations,
            seed=seed,
            antithetic=antithetic,
            progress=progress,
            checkpoint_path=checkpoint_path,
        )
    elif estimator == "banzhaf":
        estimate = estimate_banzhaf(
            oracle, target, candidates, alpha=alpha, samples=permutations, seed=seed, progress=progress
        )
    else:
        raise ValueError("estimator must be 'shapley' or 'banzhaf'.")

    ranking = estimate.ranking()
    intervals: dict[NodeId, tuple[float, float]] = {}
    for node, value in estimate.values.items():
        se = estimate.std_errors[node]
        intervals[node] = (
            (value - 1.96 * se, value + 1.96 * se)
            if math.isfinite(se)
            else (float("-inf"), float("inf"))
        )
    rank_stability: dict[str, dict[str, object]] = {}
    for k in range(1, len(ranking) + 1):
        lowers = sorted((lo for lo, _ in intervals.values()), reverse=True)
        kth_lower = lowers[k - 1]
        possible = {node for node, (_, hi) in intervals.items() if hi >= kth_lower}
        uppers = sorted((hi for _, hi in intervals.values()), reverse=True)
        next_upper = uppers[k] if k < len(uppers) else float("-inf")
        definite = {node for node, (lo, _) in intervals.items() if lo > next_upper}
        selected = set(ranking[:k])
        rank_stability[str(k)] = {
            "stable": definite == selected == possible,
            "definite_top_k_count": len(definite),
            "possible_top_k_count": len(possible),
        }

    return SelectionResult(
        method=estimator,
        ranking=ranking,
        scores=dict(estimate.values),
        params=dict(estimate.params) | {"seed": seed},
        extras={
            "std_errors": {str(node): err for node, err in estimate.std_errors.items()},
            "samples_per_node": estimate.samples_per_node,
            "base_value": estimate.base_value,
            "grand_value": estimate.grand_value,
            "efficiency_gap": estimate.efficiency_gap,
            "efficiency_gap_interpretation": (
                "structural_telescope_check_not_convergence"
                if estimator == "shapley"
                else "not_applicable_to_banzhaf"
            ),
            "rank_stability_95": rank_stability,
        },
    )
