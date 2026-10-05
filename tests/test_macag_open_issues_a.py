"""Tests for the MACAG open-issues list, code-fix half (A1-A11, 2026-10-05).

Design items D1-D10 are deliberately NOT asserted here — these tests pin the
mechanism (wiring, marks, fallbacks, provenance) while leaving recipe/paper
choices open.
"""

from __future__ import annotations

import csv
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ------------------------------------------------------------------ A6: provenance
def test_provenance_keys_and_never_raises() -> None:
    from macag.utils.provenance import code_provenance

    prov = code_provenance()
    assert set(prov) == {"git_commit", "git_dirty"}
    # Inside the repo the commit resolves to a 40-hex sha.
    assert isinstance(prov["git_commit"], str) and len(prov["git_commit"]) == 40
    assert prov["git_dirty"] in (True, False)
    # Outside any checkout: Nones, no raise.
    outside = code_provenance("/definitely/not/a/repo")
    assert outside == {"git_commit": None, "git_dirty": None}


# ------------------------------------------------- A3: matched-k oversized reads
def test_prefix_at_k_unavailable_when_no_prefix_fits() -> None:
    from macag.kl_rescore import _acdc_best_at_k, _prefix_at_k

    results = {
        "4": {"evidence": ["a", "b", "c", "d"]},
        "8": {"evidence": ["a", "b", "c", "d", "e", "f", "g", "h"]},
    }
    # No realized prefix <= 2: unavailable, NOT the oversized k=4 set.
    assert _prefix_at_k(results, 2) == ([], None)
    assert _prefix_at_k({}, 8) == ([], None)
    # Normal reads unchanged.
    evidence, own_k = _prefix_at_k(results, 8)
    assert own_k == 8 and len(evidence) == 8
    evidence, own_k = _prefix_at_k(results, 6)
    assert own_k == 4 and len(evidence) == 4


def test_acdc_best_at_k_never_oversized() -> None:
    from macag.kl_rescore import _acdc_best_at_k

    entry = {"best_by_size": {"4": {"evidence": ["a"] * 4}, "8": {"evidence": ["a"] * 8}}}
    assert _acdc_best_at_k(entry, 2) is None
    assert _acdc_best_at_k(entry, 8)["evidence"] == ["a"] * 8
    assert _acdc_best_at_k(entry, 6)["evidence"] == ["a"] * 4
    assert _acdc_best_at_k({}, 8) is None


# ------------------------------------------------- A2: degenerate-leg marking
def _dual_payload(frozen_degenerate: bool, frozen_e_star: list[str]) -> dict:
    return {
        "freeze_mode": "both",
        "frozen": {
            "degenerate": frozen_degenerate,
            "evidence": {"E_star": frozen_e_star},
            "scores": {"is_degenerate": frozen_degenerate},
        },
        "unfrozen": {
            "degenerate": False,
            "evidence": {"E_star": ["x", "y"]},
            "scores": {"is_degenerate": False},
        },
    }


def test_matched_k_degenerate_leg_is_undefined() -> None:
    from macag.kl_rescore import game1_leg_degenerate, game1_matched_k, game1_matched_k_status

    payload = _dual_payload(True, [])
    assert game1_leg_degenerate(payload) is True
    assert game1_matched_k(payload) is None
    assert game1_matched_k_status(payload) == "degenerate_leg"


def test_matched_k_genuinely_empty_leg_is_zero() -> None:
    from macag.kl_rescore import game1_matched_k, game1_matched_k_status

    payload = _dual_payload(False, [])
    assert game1_matched_k(payload) == 0
    assert game1_matched_k_status(payload) == "ok"


def test_matched_k_normal_and_missing() -> None:
    from macag.kl_rescore import game1_leg_degenerate, game1_matched_k, game1_matched_k_status

    assert game1_matched_k(_dual_payload(False, ["a", "b", "c"])) == 3
    assert game1_matched_k_status(_dual_payload(False, ["a"])) == "ok"
    assert game1_matched_k(None) is None
    assert game1_matched_k_status(None) == "no_game1"
    assert game1_leg_degenerate(None) is None


def test_matched_k_legacy_fallback_without_leg_flag() -> None:
    from macag.kl_rescore import game1_leg_degenerate, game1_matched_k

    # Outputs written before the A2 leg mark: fall back to nested scores.
    legacy = {
        "freeze_mode": "both",
        "frozen": {"evidence": {"E_star": []}, "scores": {"is_degenerate": True}},
    }
    assert game1_leg_degenerate(legacy) is True
    assert game1_matched_k(legacy) is None


class _StubOracle:
    """Minimal ScoringOracle surface for rescore arithmetic (no model)."""

    def clear_cache(self) -> None:
        return None

    def reset_stats(self) -> None:
        return None

    def cache_stats(self) -> dict[str, int]:
        return {"oracle_calls": 4, "cache_hits": 0, "cache_size": 0}

    def all(self, target) -> float:
        return 10.0

    def empty(self, target) -> float:
        return 2.0

    def keep_only(self, nodes, target) -> float:
        return 2.0 + len(nodes)

    def remove(self, nodes, target) -> float:
        return 10.0 - len(nodes)


def test_rescore_baselines_records_matched_k_status() -> None:
    from macag.kl_rescore import rescore_baselines

    payload = {
        "params": {"budget": 8},
        "methods": {
            "influence": {
                "ranking": ["a", "b", "c"],
                "results": {
                    "0": {"evidence": [], "scores": {"faithfulness": 0.0}},
                    "1": {"evidence": ["a"], "scores": {"faithfulness": 1.0}},
                    "2": {"evidence": ["a", "b"], "scores": {"faithfulness": 2.0}},
                },
            }
        },
    }
    out = rescore_baselines(
        payload, _StubOracle(), matched_k=2, matched_k_status="degenerate_leg"
    )
    assert out["matched_k"] == 2
    assert out["matched_k_status"] == "degenerate_leg"
    assert out["methods"]["influence"]["evidence_size"] == 2


# ------------------------------------------------- A5: Track B token resolution
def test_original_baselines_prefers_kwargs_index_map() -> None:
    from macag.cli.run_original_baselines import _resolve_tokens

    kwargs = {
        "target_token_by_label": {"y": " D", "y_foil": " A"},
        "foil_by_target": {"y": "y_foil"},
        "target_to_logit_idx": {"y": 608, "y_foil": 586},
    }
    assert _resolve_tokens(kwargs) == (" D", " A", 608, 586)


class _StubTokenizer:
    """Bare vs spaced single tokens, like Gemma MCQA labels."""

    _IDS = {"A": 1, "B": 2, " A": 101, " B": 102, " D": 104}
    _STRS = {v: k for k, v in _IDS.items()}

    def __call__(self, text: str, add_special_tokens: bool = False):
        class _Enc:
            pass

        enc = _Enc()
        enc.input_ids = [self._IDS[text]]
        return enc

    def decode(self, ids: list[int]) -> str:
        return self._STRS[ids[0]]


class _StubModel:
    tokenizer = _StubTokenizer()


def test_original_baselines_falls_back_through_track_a() -> None:
    # No target_to_logit_idx in kwargs: must route through Track A's
    # leading-space correction (bare 'A' -> spaced id), not bare to_tokens().
    from macag.cli.run_original_baselines import _resolve_missing_indices, _resolve_tokens

    kwargs = {
        "target_token_by_label": {"y": " D", "y_foil": "A"},
        "foil_by_target": {"y": "y_foil"},
    }
    target_token, foil_token, target_idx, foil_idx = _resolve_tokens(kwargs)
    assert (target_token, foil_token, target_idx, foil_idx) == (" D", "A", -1, None)
    target_idx, foil_idx = _resolve_missing_indices(
        kwargs, _StubTokenizer(), "y", target_token, foil_token, target_idx, foil_idx
    )
    assert target_idx == 104
    assert foil_idx == 101  # spaced ' A', not bare-'A' id 1


# ------------------------------------------------- A4/A9: builder + committed file
def test_label_token_convention_with_stub_tokenizer() -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    from build_mib_benchmark_prompts import _label_token

    assert _label_token(_StubTokenizer(), "A") == " A"

    def _enc(ids: list[int]):
        class _E:
            pass

        e = _E()
        e.input_ids = ids
        return e

    _DECODE = {11: "1", 14: "4", 0: " "}

    class _DigitTokenizer:
        def __call__(self, text: str, add_special_tokens: bool = False):
            # Digits: spaced form is two tokens (space + digit).
            if text == " 1":
                return _enc([0, 11])
            return _enc([{"1": 11, "4": 14}[text]])

        def decode(self, ids: list[int]) -> str:
            return "".join(_DECODE[i] for i in ids)

    assert _label_token(_DigitTokenizer(), "1") == "1"

    class _EmptyTokenizer:
        def __call__(self, text: str, add_special_tokens: bool = False):
            return _enc([])

        def decode(self, ids: list[int]) -> str:
            raise AssertionError("must raise before decode")

    with pytest.raises(ValueError):
        _label_token(_EmptyTokenizer(), "??")


def test_builder_accepts_sample_and_seed() -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    from build_mib_benchmark_prompts import export_prompts

    sig = inspect.signature(export_prompts)
    assert "sample" in sig.parameters and "seed" in sig.parameters


def test_builder_random_sampling_end_to_end_with_stubs(monkeypatch) -> None:
    """Seeded subset without HF data: stub the dataset + tokenizer imports."""
    import sys
    import types

    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    import build_mib_benchmark_prompts as builder

    rows = [
        {
            "prompt": f"prompt {i} A B C D",
            "choices": {"label": ["A", "B", "C", "D"]},
            "answerKey": i % 4,
            "symbol_counterfactual": {"prompt": f"cf {i}"},
        }
        for i in range(20)
    ]

    class _FakeDS:
        def __init__(self, *args, **kwargs) -> None:
            self.dataset = rows

        def __len__(self) -> int:
            return len(rows)

    stub_mod = types.ModuleType("MIB_circuit_track")
    stub_ds = types.ModuleType("MIB_circuit_track.dataset")
    stub_ds.HFEAPDataset = _FakeDS
    monkeypatch.setitem(sys.modules, "MIB_circuit_track", stub_mod)
    monkeypatch.setitem(sys.modules, "MIB_circuit_track.dataset", stub_ds)

    class _FakeTok:
        def __init__(self) -> None:
            self._reg: dict[int, str] = {}

        def __call__(self, text: str, add_special_tokens: bool = False):
            i = abs(hash(text)) % 1000
            self._reg[i] = text

            class _Enc:
                pass

            e = _Enc()
            e.input_ids = [i]
            return e

        def decode(self, ids) -> str:
            return self._reg[ids[0]]

    import transformers

    class _FakeTokenizerCls:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return _FakeTok()

    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeTokenizerCls)

    out = builder.export_prompts(
        models=["gemma2"],
        tasks=["mcqa"],
        split="validation",
        limit_per_task=5,
        task_limits=None,
        counterfactual_type=None,
        sample="random",
        seed=0,
    )
    got = out["tasks"]["mcqa"]
    assert len(got) == 5
    import random as _random

    assert [r["metadata"]["index"] for r in got] == sorted(
        _random.Random(0).sample(range(20), 5)
    )
    assert all(r["metadata"]["sample_seed"] == 0 for r in got)
    assert len({r["id"] for r in got}) == 5

    with pytest.raises(ValueError):
        builder.export_prompts(
            models=["gemma2"],
            tasks=["mcqa"],
            split="validation",
            limit_per_task=None,
            task_limits=None,
            counterfactual_type=None,
            sample="random",
            seed=None,
        )


def test_committed_prompts_have_no_bare_tokens() -> None:
    payload = json.loads((REPO_ROOT / "macag" / "data" / "mib_benchmark_prompts.json").read_text())
    for task, rows in payload["tasks"].items():
        for row in rows:
            for key in ("correct_token", "incorrect_token"):
                assert row[key][:1].isspace(), (task, row["id"], key, row[key])
    assert "token_convention" in payload["benchmarks_info"]
    assert payload["benchmarks_info"]["sampling"] == {"method": "first", "seed": None}


# ------------------------------------------------- A11: influence objective label
def test_influence_params_label_objective_without_changing_ranking() -> None:
    from macag.baselines.influence import select_top_influence
    from macag.graph import CircuitGraph

    graph = CircuitGraph(
        nodes=["a", "b"],
        node_metadata={
            "a": {"influence_raw": 6.0},
            "b": {"influence_raw": 4.0},
        },
    )
    result = select_top_influence(graph, ["a", "b"])
    assert result.ranking[:2] == ["a", "b"]
    assert "target-foil gap" in result.params["objective"]


# ------------------------------------------------- A10: bootstrap strata smoke
def test_bootstrap_emits_per_clt_task_strata(tmp_path) -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from macag_bootstrap_wilcoxon import main as bootstrap_main

    root = tmp_path / "sweep"
    root.mkdir()
    cols = [
        "clt", "task", "slug", "k_game1", "k_influence",
        "faith_budget_game1", "faith_budget_influence",
        "fpf_game1", "fpf_influence", "auc_game1", "auc_influence",
    ]
    rows = []
    for clt in ("gemma2-426k", "gemma2-2.5M"):
        for i in range(6):
            slug = f"mib_gemma2_mcqa_{i:04d}"
            rows.append({
                "clt": clt, "task": "mcqa", "slug": slug,
                "k_game1": 4, "k_influence": 8,
                "faith_budget_game1": 5.0 + i * 0.1,
                "faith_budget_influence": 1.0 + i * 0.05,
                "fpf_game1": 1.2, "fpf_influence": 0.2,
                "auc_game1": 4.0, "auc_influence": 0.8,
            })
    with (root / "baselines.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    with (root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clt", "slug", "pref"])
        writer.writeheader()
        for clt in ("gemma2-426k", "gemma2-2.5M"):
            for i in range(6):
                writer.writerow({"clt": clt, "slug": f"mib_gemma2_mcqa_{i:04d}", "pref": "true"})

    assert bootstrap_main([
        "--root", str(root), "--bootstrap-samples", "200", "--seed", "0",
    ]) == 0
    out_csv = root / "bootstrap_wilcoxon.csv"
    assert out_csv.is_file()
    with out_csv.open(newline="") as f:
        filters = {row["filter"] for row in csv.DictReader(f)}
    # Pooled blocks preserved; per-(clt, task) strata added (no cross-CLT averaging).
    assert "all" in filters and "pref_only" in filters
    assert "clt=gemma2-426k,task=mcqa" in filters
    assert "clt=gemma2-2.5M,task=mcqa" in filters
    assert "clt=gemma2-426k,task=mcqa+pref_only" in filters


# ------------------------------------------------- A1: pipeline game2 wiring
def test_pipeline_rejects_non_onesided_game2_score_kind() -> None:
    proc = subprocess.run(
        ["bash", "scripts/run_macag_pipeline.sh", "--game2-score-kind", "logit_gap"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 2
    assert "one-sided" in proc.stderr
