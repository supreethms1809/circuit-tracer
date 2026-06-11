"""Utility helpers for MACAG."""

from macag.utils.attention_mediation import (
    VERDICT_ATTENTION_MEDIATED,
    VERDICT_FEATURE_MEDIATED,
    VERDICT_INDETERMINATE,
    AttentionMediationDiagnostic,
    compute_attention_mediation_diagnostic,
)
from macag.utils.metrics import (
    FaithfulnessMetrics,
    compute_faithfulness_metrics,
    dedupe_preserve_order,
    game1_utility,
    game2_utility,
    metrics_to_dict,
    overlap_rate,
    sparsity,
)
from macag.utils.supernodes import (
    merge_supernodes_into_payload,
    node_salience,
    propose_supernodes,
    rank_nodes_by_salience,
    supernode_candidates,
)

__all__ = [
    "AttentionMediationDiagnostic",
    "compute_attention_mediation_diagnostic",
    "VERDICT_ATTENTION_MEDIATED",
    "VERDICT_FEATURE_MEDIATED",
    "VERDICT_INDETERMINATE",
    "FaithfulnessMetrics",
    "compute_faithfulness_metrics",
    "dedupe_preserve_order",
    "game1_utility",
    "game2_utility",
    "metrics_to_dict",
    "overlap_rate",
    "sparsity",
    "merge_supernodes_into_payload",
    "node_salience",
    "propose_supernodes",
    "rank_nodes_by_salience",
    "supernode_candidates",
]
