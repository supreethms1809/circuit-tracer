"""Tests for the answer_span score kind (multi-token target/foil spans, B0.1)."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from macag.factories.replacement_model import resolve_target_to_token_span
from macag.scoring import (
    ReplacementModelInterventionScorer,
    ScoringOracle,
    compute_span_logprob,
)

VOCAB = 6
PROMPT_IDS = torch.tensor([5, 1, 2])  # id 5 acts as the special/BOS token


class _SpanMockModel:
    """Mock whose last-position logits depend on how many nodes are ablated.

    All positions share one logits row per (n_interventions) key so span
    log-probs are exactly computable in the tests.
    """

    def __init__(self, rows_by_ablation: dict[int, torch.Tensor]) -> None:
        self.rows_by_ablation = rows_by_ablation
        self.calls: list[dict[str, Any]] = []

    def ensure_tokenized(self, prompt: Any) -> torch.Tensor:
        if isinstance(prompt, torch.Tensor):
            return prompt
        return PROMPT_IDS.clone()

    def feature_intervention(
        self, prompt: Any, interventions: list[Any], **kwargs: Any
    ) -> tuple[torch.Tensor, None]:
        tokens = self.ensure_tokenized(prompt)
        self.calls.append({"len": int(tokens.shape[0]), "n_ablate": len(interventions)})
        row = self.rows_by_ablation[len(interventions)]
        logits = row.expand(int(tokens.shape[0]), VOCAB).clone()
        return logits.unsqueeze(0), None


def _scorer(
    rows_by_ablation: dict[int, torch.Tensor],
    spans: dict[str, list[int]],
    score_kind: str = "answer_span",
) -> tuple[ReplacementModelInterventionScorer, _SpanMockModel]:
    model = _SpanMockModel(rows_by_ablation)
    scorer = ReplacementModelInterventionScorer(
        model=model,
        prompt="test",
        node_to_intervention={"a": (0, 0, 1), "b": (0, 0, 2)},
        target_to_logit_idx={"y": spans["y"][0], "y_foil": spans["y_foil"][0]},
        score_kind=score_kind,  # type: ignore[arg-type]
        foil_by_target={"y": "y_foil", "y_foil": "y"},
        target_span_ids_by_label=spans if score_kind == "answer_span" else None,
    )
    return scorer, model


def test_compute_span_logprob_exact() -> None:
    # seq logits: rows for positions 0..3; span scored at prompt_len-1 + i
    logits = torch.zeros(1, 4, VOCAB)
    logits[0, 2] = torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    logits[0, 3] = torch.tensor([0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    expected = (
        torch.log_softmax(logits[0, 2], dim=-1)[1] + torch.log_softmax(logits[0, 3], dim=-1)[2]
    )
    assert compute_span_logprob(logits, [1, 2], prompt_len=3) == pytest.approx(
        float(expected), abs=1e-6
    )


def test_single_token_span_equals_logit_gap() -> None:
    rows = {
        0: torch.tensor([0.0, 3.0, 1.0, 0.0, 0.5, 0.0]),
        1: torch.tensor([0.0, 2.0, 1.5, 0.0, 0.5, 0.0]),
        2: torch.tensor([0.0, 0.5, 2.5, 0.0, 0.5, 0.0]),
    }
    spans = {"y": [1], "y_foil": [2]}
    span_scorer, _ = _scorer(rows, spans, score_kind="answer_span")
    gap_scorer, _ = _scorer(rows, spans, score_kind="logit_gap")
    for mode in ("score_all", "score_empty"):
        span_score = getattr(span_scorer, mode)("y")
        gap_score = getattr(gap_scorer, mode)("y")
        # log_softmax[t] - log_softmax[f] == logit[t] - logit[f] (normalizer cancels)
        assert span_score == pytest.approx(gap_score, abs=1e-5)


def test_shared_first_token_disambiguated_by_span() -> None:
    """The B0.1 acceptance criterion in miniature.

    Target span [3, 1] and foil span [3, 2] share the first token (id 3), so a
    first-token logit gap is exactly 0; the answer-span gap sees the second
    token and is nonzero.
    """
    rows = {0: torch.tensor([0.0, 4.0, 1.0, 2.0, 0.0, 0.0])}
    spans = {"y": [3, 1], "y_foil": [3, 2]}
    scorer, _ = _scorer(rows, spans)
    span_gap = scorer.score_all("y")
    # shared first token contributes identically to both spans and cancels:
    # gap = logprob(1) - logprob(2) at the second span position
    row_logprobs = torch.log_softmax(rows[0], dim=-1)
    assert span_gap == pytest.approx(float(row_logprobs[1] - row_logprobs[2]), abs=1e-5)
    assert span_gap != 0.0

    gap_scorer, _ = _scorer(rows, {"y": [3], "y_foil": [3]}, score_kind="logit_gap")
    assert gap_scorer.score_all("y") == pytest.approx(0.0, abs=1e-6)


def test_interventions_and_forward_accounting() -> None:
    rows = {
        0: torch.tensor([0.0, 3.0, 1.0, 0.0, 0.0, 0.0]),
        1: torch.tensor([0.0, 2.0, 1.2, 0.0, 0.0, 0.0]),
        2: torch.tensor([0.0, 1.0, 1.4, 0.0, 0.0, 0.0]),
    }
    spans = {"y": [1, 1], "y_foil": [2, 2]}
    scorer, model = _scorer(rows, spans)
    oracle = ScoringOracle(backend=scorer, cache_enabled=True)

    all_score = oracle.all("y")
    empty = oracle.empty("y")
    keep_a = oracle.keep_only({"a"}, "y")
    assert all_score > keep_a > empty  # fewer ablations -> higher target preference

    # each logical score ran two forwards (target + foil continuation)
    stats = oracle.cache_stats()
    assert scorer.forward_count == 2 * stats["oracle_calls"]

    # inputs include the span: prompt (3 tokens) + span (2 tokens)
    assert all(call["len"] == 5 for call in model.calls)

    # cache hit does not add forwards
    before = scorer.forward_count
    oracle.keep_only({"a"}, "y")
    assert scorer.forward_count == before
    assert oracle.cache_stats()["cache_hits"] >= 1


def test_answer_span_requires_spans_and_foil() -> None:
    rows = {0: torch.tensor([0.0] * VOCAB)}
    with pytest.raises(ValueError, match="target_span_ids_by_label"):
        ReplacementModelInterventionScorer(
            model=_SpanMockModel(rows),
            prompt="test",
            node_to_intervention={"a": (0, 0, 1)},
            target_to_logit_idx={"y": 1},
            score_kind="answer_span",
        )
    scorer, _ = _scorer(rows, {"y": [1], "y_foil": [2]})
    object.__setattr__(scorer, "foil_by_target", None)
    with pytest.raises(ValueError, match="foil"):
        scorer.score_all("y")


class _FakeTokenizer:
    TABLE = {" 42": [10, 2], " 40": [10, 0], " Austin": [7], "42": [4, 2]}

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": self.TABLE.get(text, [1])}


def test_resolve_target_to_token_span() -> None:
    spans = resolve_target_to_token_span(
        _FakeTokenizer(),
        {"y": " 42", "y_foil": " 40", "single": " Austin", "by_id": "id:9", "as_int": 3},
    )
    assert spans["y"] == [10, 2]
    assert spans["y_foil"] == [10, 0]
    assert spans["single"] == [7]
    assert spans["by_id"] == [9]
    assert spans["as_int"] == [3]
