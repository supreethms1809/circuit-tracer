"""Gold-circuit validation at the (layer-band, token-role) level (roadmap B4.1).

CLT features are not attention heads, so MACAG evidence cannot be scored against
the published IOI circuit head-for-head. This module validates at the coarser
level the roadmap prescribes: does the selected evidence *read from the layers
and token positions* where the known IOI components live?

Two caveats are load-bearing and stated up front:

- **The layer bands are a judgment call.** ``IOI_GOLD`` expresses the published
  GPT-2-small components (duplicate-token heads a0.h1/a0.h10/a3.h0, induction
  a5.h5/a6.h9, S-inhibition a7.h3/a7.h9/a8.h6/a8.h10, name-movers a9.h6/a9.h9/
  a10.h0; Wang et al., 2023) as *depth fractions* of a 12-layer model, widened
  to tolerate cross-architecture drift. They are module constants — tune, don't
  hardcode downstream.
- **Recall is component-level.** A CLT may legitimately represent one head's
  computation with several features (or one feature for several heads), so
  feature-level recall is undefined; we report the fraction of gold *components*
  hit by at least one evidence node, and node-level precision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from macag.graph import CircuitGraph, NodeId
from macag.utils.attention_mediation import _infer_depth_and_final_pos, _node_layer_pos

LOGGER = logging.getLogger(__name__)

ROLE_S1 = "S1"
ROLE_S2 = "S2"
ROLE_IO = "IO"
ROLE_END = "END"


@dataclass(frozen=True)
class GoldComponent:
    """One known circuit component class as a (layer-band, token-role) region.

    ``layer_band`` is an inclusive range of ``layer / (n_layers - 1)`` depth
    fractions; ``token_roles`` are the prompt roles the component reads at.
    """

    name: str
    layer_band: tuple[float, float]
    token_roles: frozenset[str]


# Published GPT-2-small IOI layers as depth fractions of (n_layers-1)=11, each
# band widened by ~0.1 on both sides: duplicate-token L0/L3 (0.0-0.27),
# induction L5/L6 (0.45-0.55), S-inhibition L7/L8 (0.64-0.73), name-mover
# L9/L10 (0.82-0.91). Duplicate-token/induction read at S2; S-inhibition and
# name-movers write/read at END.
IOI_GOLD: tuple[GoldComponent, ...] = (
    GoldComponent("duplicate_token", (0.0, 0.35), frozenset({ROLE_S2})),
    GoldComponent("induction", (0.15, 0.65), frozenset({ROLE_S2})),
    GoldComponent("s_inhibition", (0.50, 0.85), frozenset({ROLE_END})),
    GoldComponent("name_mover", (0.70, 1.00), frozenset({ROLE_END})),
)


def _name_positions(prompt_tokens: Sequence[str], name: str) -> list[int]:
    stripped = name.strip()
    return [i for i, tok in enumerate(prompt_tokens) if tok.strip() == stripped]


def assign_token_roles(
    prompt_tokens: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> dict[int, str]:
    """Map ctx_idx -> role in {S1, S2, IO, END} for an IOI prompt.

    Primary path uses manifest metadata (``{"S1": "John", "IO": "Mary", ...}``,
    as carried by ``macag/data/acdc_benchmark_prompts.json``). Fallback (MIB IOI
    prompts carry no role metadata) infers roles from the token list: among
    space-prefixed capitalized tokens, the duplicated one is the subject
    (S1/S2 by occurrence order) and the unique one is IO. The fallback is
    heuristic — multi-token names, third names, or capitalized non-names break
    it — so it returns a *partial* map and downstream scoring tolerates missing
    roles; report coverage via ``roles_resolved``.
    """
    roles: dict[int, str] = {}
    if prompt_tokens:
        roles[len(prompt_tokens) - 1] = ROLE_END

    if metadata and metadata.get("S1"):
        s_positions = _name_positions(prompt_tokens, str(metadata["S1"]))
        if s_positions:
            roles[s_positions[0]] = ROLE_S1
        if len(s_positions) > 1:
            roles[s_positions[1]] = ROLE_S2
        io_name = metadata.get("IO")
        if io_name:
            io_positions = _name_positions(prompt_tokens, str(io_name))
            if io_positions:
                roles[io_positions[0]] = ROLE_IO
        return roles

    # Fallback: name-shaped tokens are space-prefixed, capitalized, alphabetic.
    candidates: dict[str, list[int]] = {}
    for i, tok in enumerate(prompt_tokens):
        stripped = tok.strip()
        if (
            tok[:1] == " "
            and stripped
            and stripped[0].isupper()
            and stripped.isalpha()
        ):
            candidates.setdefault(stripped, []).append(i)
    duplicated = [name for name, pos in candidates.items() if len(pos) == 2]
    unique = [name for name, pos in candidates.items() if len(pos) == 1]
    if len(duplicated) == 1:
        s_pos = candidates[duplicated[0]]
        roles[s_pos[0]] = ROLE_S1
        roles[s_pos[1]] = ROLE_S2
        if len(unique) == 1:
            roles[candidates[unique[0]][0]] = ROLE_IO
    else:
        LOGGER.debug(
            "IOI role fallback could not identify a unique duplicated name "
            "(duplicated=%s unique=%s); returning partial roles.",
            duplicated,
            unique,
        )
    return roles


@dataclass
class GoldScore:
    """Precision/recall of an evidence set against a gold component spec."""

    precision: float
    recall: float
    f1: float
    n_evidence: int
    n_scored: int  # evidence nodes with resolvable (layer, ctx)
    n_matched: int
    component_hits: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "n_evidence": self.n_evidence,
            "n_scored": self.n_scored,
            "n_matched": self.n_matched,
            "component_hits": dict(self.component_hits),
        }


def score_evidence_against_gold(
    graph: CircuitGraph,
    evidence: Sequence[NodeId],
    roles: Mapping[int, str],
    gold: Sequence[GoldComponent] = IOI_GOLD,
    n_layers: int | None = None,
) -> GoldScore:
    """Node-level precision / component-level recall of evidence vs the gold spec.

    A node matches a component iff its depth fraction ``layer/(n_layers-1)``
    falls in the component's band AND its position's role is one the component
    reads at. Nodes with unresolvable (layer, ctx) are excluded from precision's
    denominator (reported as ``n_scored``).
    """
    if n_layers is None:
        n_layers, _final = _infer_depth_and_final_pos(graph)
    if not n_layers or n_layers < 2:
        raise ValueError("Could not infer n_layers (>= 2) from the graph; pass n_layers.")

    component_hits = {component.name: 0 for component in gold}
    n_scored = 0
    n_matched = 0
    for node in evidence:
        layer, pos = _node_layer_pos(graph, node)
        if layer is None or pos is None:
            continue
        n_scored += 1
        fraction = layer / (n_layers - 1)
        role = roles.get(pos)
        matched = False
        for component in gold:
            if role in component.token_roles and (
                component.layer_band[0] <= fraction <= component.layer_band[1]
            ):
                component_hits[component.name] += 1
                matched = True
        if matched:
            n_matched += 1

    precision = n_matched / n_scored if n_scored else 0.0
    recall = (
        sum(1 for hits in component_hits.values() if hits > 0) / len(gold) if gold else 0.0
    )
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return GoldScore(
        precision=precision,
        recall=recall,
        f1=f1,
        n_evidence=len(evidence),
        n_scored=n_scored,
        n_matched=n_matched,
        component_hits=component_hits,
    )


def binary_auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Rank-based AUROC (Mann–Whitney) of continuous scores vs binary labels.

    Average ranks on ties; raises if either class is empty (AUROC undefined).
    Used for InterpBench node-level validation — upstream MIB's
    ``evaluate_area_under_roc`` is edge-level only.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length.")
    n_pos = sum(1 for label in labels if label)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs at least one positive and one negative label.")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed average rank of the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    rank_sum_pos = sum(rank for rank, label in zip(ranks, labels) if label)
    u_statistic = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u_statistic / (n_pos * n_neg)


def average_precision(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Average precision (area under the precision-recall curve, step-wise)."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length.")
    n_pos = sum(1 for label in labels if label)
    if n_pos == 0:
        raise ValueError("average_precision needs at least one positive label.")
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    hits = 0
    total = 0.0
    for rank, idx in enumerate(order, start=1):
        if labels[idx]:
            hits += 1
            total += hits / rank
    return total / n_pos
