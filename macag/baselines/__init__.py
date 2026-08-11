"""Baseline selectors for the MACAG head-to-head evaluation (roadmap Phase 2).

Legacy IDs ``eap`` / ``acdc`` are stable (do not rename — historical JSON).
Explicit aliases and paper-faithful methods are documented in
``macag/docs/baseline_method_map.md``.

- influence.py       top-k by graph ``influence``
- eap.py             graph path-effect (alias: eap_graph) — NOT Syed EAP
- eap_syed.py        Syed/Nanda attribution patching on CLT feature nodes
- shapley_select.py  Monte-Carlo Shapley/Banzhaf over the MACAG oracle
- acdc_prune.py      ported ACDC node τ-prune (alias: acdc_ported)
- acdc_native.py     same τ-prune rule on native heads/MLPs
- bruteforce.py      exact best size-k subset

The head-to-head harness is ``python -m macag.cli.run_baselines``.
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
from macag.baselines.eap_syed import compute_syed_eap_node_scores, select_top_eap_syed
from macag.baselines.shapley_select import (
    ShapleyEstimate,
    estimate_banzhaf,
    estimate_shapley,
    select_top_shapley,
)
from macag.baselines.acdc_prune import ACDCPruneResult, acdc_prune, acdc_tau_sweep
from macag.baselines.acdc_native import (
    build_native_component_oracle,
    run_acdc_native,
)
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
    "compute_syed_eap_node_scores",
    "select_top_eap_syed",
    "ShapleyEstimate",
    "estimate_shapley",
    "estimate_banzhaf",
    "select_top_shapley",
    "ACDCPruneResult",
    "acdc_prune",
    "acdc_tau_sweep",
    "build_native_component_oracle",
    "run_acdc_native",
    "BruteForceResult",
    "best_subset_bruteforce",
]
