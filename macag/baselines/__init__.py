"""Baseline selectors for the MACAG head-to-head evaluation (roadmap Phase 2).

Every baseline ranks or selects feature nodes under (an approximation of) the
same coalitional value function v(S) used by the games (macag.md §3.0), and is
scored through the same ScoringOracle, so only the selection rule differs:

- influence.py       (B2.1) top-k by the graph's own `influence` metric
- shapley_select.py  (B2.2) Monte-Carlo Shapley/Banzhaf over the MACAG oracle
- eap.py             (B2.3) EAP / attribution-patching node scores from graph edges
- acdc_prune.py      (B2.4) ported ACDC top-down threshold pruning
- bruteforce.py      (B3.2) exact best size-k subset, for the greedy optimality gap

The head-to-head harness is `python -m macag.cli.run_baselines`.
"""

from macag.baselines.common import (
    SelectionResult,
    coalition_value,
    jaccard,
    precision_at_k,
    ranking_from_scores,
    spearman_rank_correlation,
)
from macag.baselines.influence import select_top_influence
from macag.baselines.eap import compute_eap_node_scores, select_top_eap
from macag.baselines.shapley_select import (
    ShapleyEstimate,
    estimate_banzhaf,
    estimate_shapley,
    select_top_shapley,
)
from macag.baselines.acdc_prune import ACDCPruneResult, acdc_prune, acdc_tau_sweep
from macag.baselines.bruteforce import BruteForceResult, best_subset_bruteforce

__all__ = [
    "SelectionResult",
    "coalition_value",
    "jaccard",
    "precision_at_k",
    "ranking_from_scores",
    "spearman_rank_correlation",
    "select_top_influence",
    "compute_eap_node_scores",
    "select_top_eap",
    "ShapleyEstimate",
    "estimate_shapley",
    "estimate_banzhaf",
    "select_top_shapley",
    "ACDCPruneResult",
    "acdc_prune",
    "acdc_tau_sweep",
    "BruteForceResult",
    "best_subset_bruteforce",
]
