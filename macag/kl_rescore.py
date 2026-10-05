"""Post-hoc rescoring of saved MACAG evidence under an alternate oracle.

The canonical instance is **KL faithfulness** (``KL_SPEC``): re-score every
stored evidence set under ``score_kind="kl_divergence"`` so the evaluation
metric is not Game 1's selection objective. The machinery is parametrized by a
:class:`RescoreSpec`, so other selection-independent checks (e.g. the alternate
foil of ``macag.cli.rescore_altfoil``) reuse the same walk/merge/embed logic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from macag.factories.replacement_model import create_replacement_model_oracle
from macag.scoring import ScoringOracle, TargetId
from macag.utils.metrics import compute_faithfulness_metrics, metrics_to_dict

LOGGER = logging.getLogger(__name__)

DEFAULT_TARGET = "y"
KL_OUTPUT_NAME = "macag_kl_faithfulness.json"
ORACLE_KWARGS_NAME = "oracle_kwargs.json"


@dataclass(frozen=True)
class RescoreSpec:
    """One rescoring flavor: which oracle to rebuild and where results land.

    ``kwargs_overrides`` are merged over the stored ``oracle_kwargs.json``;
    ``transform`` (applied after the merge) covers overrides that must read the
    stored kwargs (e.g. substituting one label inside ``target_token_by_label``).
    """

    output_name: str  # per-run sidecar JSON filename
    embed_key: str  # key embedded into macag_game{1,2}.json / macag_baselines.json
    score_label: str  # key of the rescored block inside the sidecar payload
    kwargs_overrides: Mapping[str, Any] = field(default_factory=dict)
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None


KL_SPEC = RescoreSpec(
    output_name=KL_OUTPUT_NAME,
    embed_key="kl_faithfulness",
    score_label="kl_divergence",
    kwargs_overrides={"score_kind": "kl_divergence"},
)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def build_oracle_with_overrides(
    kwargs: Mapping[str, Any],
    spec: RescoreSpec,
    *,
    freeze_attention: bool | None = None,
) -> ScoringOracle:
    """Rebuild a scoring oracle from saved kwargs with the spec's overrides."""
    merged = dict(kwargs)
    merged.update(spec.kwargs_overrides)
    if spec.transform is not None:
        merged = spec.transform(merged)
    if freeze_attention is not None:
        merged["freeze_attention"] = freeze_attention
    built = create_replacement_model_oracle(**merged)
    return built.oracle


def build_kl_oracle(
    kwargs: Mapping[str, Any],
    *,
    freeze_attention: bool | None = None,
) -> ScoringOracle:
    """Build a KL-divergence oracle from saved replacement-model kwargs."""
    return build_oracle_with_overrides(kwargs, KL_SPEC, freeze_attention=freeze_attention)


def _alpha_from_payload(payload: Mapping[str, Any], default: float = 0.5) -> float:
    params = payload.get("params") or {}
    if "alpha" in params:
        return float(params["alpha"])
    return default


def _rescore_nodes(
    oracle: ScoringOracle,
    *,
    target: TargetId,
    nodes: Sequence[str],
    alpha: float,
    cap_sufficiency: bool = True,
    cap_necessity: bool = True,
) -> dict[str, Any]:
    metrics = compute_faithfulness_metrics(
        oracle=oracle,
        target=target,
        nodes=set(nodes),
        alpha=alpha,
        cap_sufficiency=cap_sufficiency,
        cap_necessity=cap_necessity,
    )
    return {
        "evidence": list(nodes),
        "scores": metrics_to_dict(metrics),
        "oracle_calls": oracle.cache_stats()["oracle_calls"],
    }


def _faith_at_own_k(results: Mapping[str, Any], budget: int) -> tuple[list[str], int | None]:
    if not results:
        return [], None
    ks = sorted(int(k) for k in results)
    capped = [k for k in ks if k <= budget]
    own_k = max(capped) if capped else min(ks)
    entry = results.get(str(own_k), {})
    evidence = entry.get("evidence") or []
    return list(evidence), own_k


def _acdc_evidence_for_budget(entry: Mapping[str, Any], budget: int) -> tuple[list[str], float | None, dict[str, Any]]:
    """Prefer matched_k evidence; never fall back to an oversized sweep set.

    Returns (evidence, logit_gap_faithfulness, meta).
    """
    matched = entry.get("matched_k")
    if isinstance(matched, Mapping) and matched.get("status") != "unavailable":
        evidence = matched.get("evidence")
        ach = matched.get("achieved_k")
        if evidence is not None and ach is not None and int(ach) <= budget:
            scores = matched.get("scores") or {}
            faith = scores.get("faithfulness")
            if faith is None and "value" in matched:
                faith = matched.get("value")
            return (
                list(evidence or []),
                float(faith) if isinstance(faith, (int, float)) else None,
                {
                    "source": "matched_k",
                    "achieved_k": ach,
                    "exact": matched.get("exact"),
                },
            )

    best_by_size = entry.get("best_by_size") or {}
    best = best_by_size.get(str(budget))
    if best is None:
        return [], None, {"source": "missing_budget_size"}
    scores = best.get("scores") or {}
    faith = scores.get("faithfulness", best.get("value"))
    return (
        list(best.get("evidence") or []),
        float(faith) if isinstance(faith, (int, float)) else None,
        {"source": "best_by_size", "size": budget},
    )


def rescore_game1_leg(
    leg: Mapping[str, Any],
    oracle: ScoringOracle,
    *,
    target: TargetId = DEFAULT_TARGET,
    score_label: str = "kl_divergence",
    cap_sufficiency: bool = True,
    cap_necessity: bool = True,
) -> dict[str, Any]:
    evidence = leg.get("evidence", {}).get("E_star") or []
    alpha = _alpha_from_payload(leg)
    logit_scores = leg.get("scores") or {}
    kl_block = _rescore_nodes(
        oracle,
        target=target,
        nodes=evidence,
        alpha=alpha,
        cap_sufficiency=cap_sufficiency,
        cap_necessity=cap_necessity,
    )
    return {
        "evidence_size": len(evidence),
        "alpha": alpha,
        "cap_sufficiency": cap_sufficiency,
        "cap_necessity": cap_necessity,
        "logit_gap": {
            "faithfulness": logit_scores.get("faithfulness"),
            "sufficiency": logit_scores.get("sufficiency"),
            "recoverable_range": logit_scores.get("recoverable_range"),
        },
        score_label: kl_block["scores"],
        "oracle_calls": kl_block["oracle_calls"],
    }


def rescore_game2(
    payload: Mapping[str, Any],
    oracle: ScoringOracle,
    *,
    target: TargetId = DEFAULT_TARGET,
    foil: TargetId = "y_foil",
    score_label: str = "kl_divergence",
    cap_sufficiency: bool = True,
    cap_necessity: bool = True,
) -> dict[str, Any]:
    alpha = _alpha_from_payload(payload)
    evidence = payload.get("evidence") or {}
    e_y = evidence.get("E_y") or []
    e_foil = evidence.get("E_foil") or []
    scores = payload.get("scores") or {}
    kl_y = _rescore_nodes(
        oracle, target=target, nodes=e_y, alpha=alpha,
        cap_sufficiency=cap_sufficiency, cap_necessity=cap_necessity,
    )
    oracle.clear_cache()
    oracle.reset_stats()
    kl_foil = _rescore_nodes(
        oracle, target=foil, nodes=e_foil, alpha=alpha,
        cap_sufficiency=cap_sufficiency, cap_necessity=cap_necessity,
    )
    return {
        "alpha": alpha,
        "cap_sufficiency": cap_sufficiency,
        "cap_necessity": cap_necessity,
        "logit_gap": {
            "target_faithfulness": (scores.get("target") or {}).get("faithfulness"),
            "foil_faithfulness": (scores.get("foil") or {}).get("faithfulness"),
            "overlap_rate": scores.get("overlap_rate"),
        },
        score_label: {
            "target": kl_y["scores"],
            "foil": kl_foil["scores"],
            "oracle_calls": kl_y["oracle_calls"] + kl_foil["oracle_calls"],
        },
    }


def _prefix_at_k(results: Mapping[str, Any], k: int) -> tuple[list[str], int | None]:
    """Largest realized prefix size <= k (A5 matched-k read).

    Ranked methods only populate keys up to their selected size; taking the
    largest key <= k scores every method at Game 1's |E*| instead of at the
    global budget.
    """
    if not results:
        return [], None
    ks = sorted(int(x) for x in results)
    capped = [x for x in ks if x <= k]
    own_k = max(capped) if capped else min(ks)
    entry = results.get(str(own_k), {})
    return list(entry.get("evidence") or []), own_k


def _acdc_best_at_k(entry: Mapping[str, Any], k: int) -> dict[str, Any] | None:
    best_by_size = entry.get("best_by_size") or {}
    if not best_by_size:
        return None
    if str(k) in best_by_size:
        return best_by_size[str(k)]
    sizes = sorted(int(s) for s in best_by_size)
    capped = [s for s in sizes if s <= k]
    pick = max(capped) if capped else min(sizes)
    return best_by_size[str(pick)]


def game1_matched_k(game1_payload: Mapping[str, Any] | None) -> int | None:
    """Game 1's selected size |E*| for matched-k reads (A5).

    Dual-freeze outputs use the frozen leg (the leg the baseline harness scores:
    the pipeline's frozen oracle kwargs). Single-mode outputs use E_star.
    """
    if not game1_payload:
        return None
    if game1_payload.get("freeze_mode") == "both":
        frozen = game1_payload.get("frozen") or {}
        evidence = (frozen.get("evidence") or {}).get("E_star") or []
        return len(evidence) or None
    evidence = (game1_payload.get("evidence") or {}).get("E_star") or []
    return len(evidence) or None


def rescore_baselines(
    payload: Mapping[str, Any],
    oracle: ScoringOracle,
    *,
    target: TargetId = DEFAULT_TARGET,
    score_label: str = "kl_divergence",
    matched_k: int | None = None,
    cap_sufficiency: bool = True,
    cap_necessity: bool = True,
) -> dict[str, Any]:
    alpha = _alpha_from_payload(payload)
    budget = int(payload.get("params", {}).get("budget", 8))
    # A5: score every method at Game 1's |E*| when known, not at the global
    # budget. Falls back to the legacy at-budget read when matched_k is None.
    eval_k = matched_k if matched_k is not None else budget
    methods_out: dict[str, Any] = {}
    for method, entry in (payload.get("methods") or {}).items():
        if method in ("acdc", "acdc_native"):
            # A5: prefer the matched-k read at Game 1's |E*| when known;
            # otherwise use the budget-capped helper (never oversized sets).
            best = _acdc_best_at_k(entry, eval_k) if matched_k is not None else None
            if best is not None:
                evidence = best.get("evidence") or []
                logit_faith = (best.get("scores") or {}).get("faithfulness")
            else:
                evidence, logit_faith, meta = _acdc_evidence_for_budget(entry, budget)
                if not evidence and meta.get("source") == "missing_budget_size":
                    LOGGER.warning(
                        "Skipping KL rescore for %s: no matched_k and no best_by_size[%s]",
                        method,
                        budget,
                    )
                    continue
        else:
            if matched_k is not None:
                evidence, own_k = _prefix_at_k(entry.get("results") or {}, matched_k)
            else:
                evidence, own_k = _faith_at_own_k(entry.get("results") or {}, budget)
            logit_faith = None
            if own_k is not None:
                logit_faith = (
                    (entry.get("results") or {}).get(str(own_k), {}).get("scores") or {}
                ).get("faithfulness")
            # Preserve empty selections as valid k=0 (faithfulness ~ 0).
            if not evidence and own_k is None and not (entry.get("results") or {}):
                ranking = entry.get("ranking") or []
                if ranking == [] or entry.get("extras", {}).get("stopped_early"):
                    evidence = []
                    logit_faith = 0.0
        if method not in ("acdc", "acdc_native") and not evidence and logit_faith is None:
            continue
        oracle.clear_cache()
        oracle.reset_stats()
        kl_block = _rescore_nodes(
            oracle, target=target, nodes=evidence, alpha=alpha,
            cap_sufficiency=cap_sufficiency, cap_necessity=cap_necessity,
        )
        methods_out[method] = {
            "evidence_size": len(evidence),
            "logit_gap_faithfulness": logit_faith,
            score_label: kl_block["scores"],
            "oracle_calls": kl_block["oracle_calls"],
        }
    out: dict[str, Any] = {
        "alpha": alpha,
        "budget": budget,
        "methods": methods_out,
        "cap_sufficiency": cap_sufficiency,
        "cap_necessity": cap_necessity,
    }
    if matched_k is not None:
        out["matched_k"] = matched_k
        out["eval_k"] = eval_k
    return out


def merge_kl_into_outputs(
    run_dir: str | Path,
    kl_payload: Mapping[str, Any],
    *,
    spec: RescoreSpec = KL_SPEC,
) -> None:
    """Embed rescored blocks into the saved game/baseline JSON files."""
    root = Path(run_dir)
    label = spec.score_label

    g1_path = root / "macag_game1.json"
    kl_g1 = kl_payload.get("game1") or {}
    # P1-7 cap provenance: sibling key naming the capping convention behind the
    # embedded KL numbers (block-level flags win; else the sidecar top level).
    top_caps = {
        "cap_sufficiency": kl_payload.get("cap_sufficiency", True),
        "cap_necessity": kl_payload.get("cap_necessity", True),
    }

    def _caps_of(block: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "cap_sufficiency": block.get("cap_sufficiency", top_caps["cap_sufficiency"]),
            "cap_necessity": block.get("cap_necessity", top_caps["cap_necessity"]),
        }

    if g1_path.is_file() and kl_g1:
        g1 = load_json(g1_path)
        if g1.get("freeze_mode") == "both":
            for leg_name in ("frozen", "unfrozen"):
                if leg_name in kl_g1 and leg_name in g1:
                    g1[leg_name][spec.embed_key] = kl_g1[leg_name][label]
                    g1[leg_name][spec.embed_key + "_caps"] = _caps_of(kl_g1[leg_name])
        elif "single" in kl_g1:
            g1[spec.embed_key] = kl_g1["single"][label]
            g1[spec.embed_key + "_caps"] = _caps_of(kl_g1["single"])
        g1_path.write_text(json.dumps(g1, indent=2) + "\n")

    g2_path = root / "macag_game2.json"
    kl_g2 = kl_payload.get("game2") or {}
    if g2_path.is_file() and kl_g2:
        g2 = load_json(g2_path)
        g2[spec.embed_key] = kl_g2.get(label)
        g2[spec.embed_key + "_caps"] = _caps_of(kl_g2)
        g2_path.write_text(json.dumps(g2, indent=2) + "\n")

    bl_path = root / "macag_baselines.json"
    kl_bl = kl_payload.get("baselines") or {}
    kl_methods = kl_bl.get("methods") or {}
    if bl_path.is_file() and kl_methods:
        bl = load_json(bl_path)
        for method, block in kl_methods.items():
            entry = (bl.get("methods") or {}).get(method)
            if entry is None:
                continue
            entry[spec.embed_key] = block.get(label)
            entry[spec.embed_key + "_caps"] = _caps_of(kl_bl)
            # A5 provenance: which k the KL read used (matched-k vs budget).
            # Sibling keys only; the scores shape above is unchanged for readers.
            if "evidence_size" in block:
                entry[spec.embed_key + "_evidence_size"] = block["evidence_size"]
            if kl_bl.get("matched_k") is not None:
                entry[spec.embed_key + "_matched_k"] = kl_bl["matched_k"]
        bl_path.write_text(json.dumps(bl, indent=2) + "\n")


def read_game1_kl_faith(run_dir: str | Path, leg: str = "frozen") -> float | None:
    """Read Game 1 KL faithfulness from embedded game JSON or the KL sidecar."""
    root = Path(run_dir)
    g1_path = root / "macag_game1.json"
    if g1_path.is_file():
        g1 = load_json(g1_path)
        if g1.get("freeze_mode") == "both":
            block = g1.get(leg) or {}
        else:
            block = g1
        kl = block.get("kl_faithfulness") or {}
        faith = kl.get("faithfulness")
        if isinstance(faith, (int, float)):
            return float(faith)

    sidecar = root / KL_OUTPUT_NAME
    if not sidecar.is_file():
        return None
    payload = load_json(sidecar)
    kl_g1 = payload.get("game1") or {}
    block = kl_g1.get("single") if "single" in kl_g1 else kl_g1.get(leg)
    if not block:
        return None
    faith = (block.get("kl_divergence") or {}).get("faithfulness")
    return float(faith) if isinstance(faith, (int, float)) else None


def rescore_run_dir(
    run_dir: str | Path,
    *,
    target: TargetId = DEFAULT_TARGET,
    foil: TargetId = "y_foil",
    force: bool = False,
    spec: RescoreSpec = KL_SPEC,
    cap_sufficiency: bool = True,
    cap_necessity: bool = True,
) -> Path | None:
    """Re-score saved MACAG outputs in ``run_dir``; write the spec's sidecar JSON."""
    root = Path(run_dir)
    out_path = root / spec.output_name
    label = spec.score_label
    if out_path.is_file() and not force:
        output = load_json(out_path)
        merge_kl_into_outputs(root, output, spec=spec)
        LOGGER.debug("skip rescore %s (exists); refreshed embedded blocks", out_path)
        return out_path

    kwargs_path = root / ORACLE_KWARGS_NAME
    if not kwargs_path.is_file():
        LOGGER.warning("skip %s: missing %s", root, ORACLE_KWARGS_NAME)
        return None

    kwargs = load_json(kwargs_path)
    output: dict[str, Any] = {
        "run_dir": str(root),
        "score_kind": label,
        "target": target,
        "foil": foil,
        # P1-7: cap provenance — the rescore recomputes capped metrics, so the
        # flags must travel with the numbers (defaults match the games).
        "cap_sufficiency": cap_sufficiency,
        "cap_necessity": cap_necessity,
    }

    g1_path = root / "macag_game1.json"
    if g1_path.is_file():
        g1 = load_json(g1_path)
        output["game1"] = {}
        if g1.get("freeze_mode") == "both":
            for leg_name, freeze in (("frozen", True), ("unfrozen", False)):
                leg = g1.get(leg_name)
                if not leg:
                    continue
                oracle = build_oracle_with_overrides(kwargs, spec, freeze_attention=freeze)
                output["game1"][leg_name] = rescore_game1_leg(
                    leg, oracle, target=target, score_label=label,
                    cap_sufficiency=cap_sufficiency, cap_necessity=cap_necessity,
                )
        else:
            freeze = bool(kwargs.get("freeze_attention", True))
            oracle = build_oracle_with_overrides(kwargs, spec, freeze_attention=freeze)
            output["game1"]["single"] = rescore_game1_leg(
                g1, oracle, target=target, score_label=label,
                cap_sufficiency=cap_sufficiency, cap_necessity=cap_necessity,
            )

    g2_path = root / "macag_game2.json"
    if g2_path.is_file():
        g2 = load_json(g2_path)
        freeze = bool(kwargs.get("freeze_attention", True))
        oracle = build_oracle_with_overrides(kwargs, spec, freeze_attention=freeze)
        output["game2"] = rescore_game2(
            g2, oracle, target=target, foil=foil, score_label=label,
            cap_sufficiency=cap_sufficiency, cap_necessity=cap_necessity,
        )

    bl_path = root / "macag_baselines.json"
    if bl_path.is_file():
        bl = load_json(bl_path)
        freeze = bool(kwargs.get("freeze_attention", True))
        oracle = build_oracle_with_overrides(kwargs, spec, freeze_attention=freeze)
        # A5: match baselines to Game 1's |E*| (frozen leg) when a Game 1 output
        # exists; otherwise fall back to the legacy at-budget read.
        g1_for_k = load_json(g1_path) if g1_path.is_file() else None
        output["baselines"] = rescore_baselines(
            bl, oracle, target=target, score_label=label,
            matched_k=game1_matched_k(g1_for_k),
            cap_sufficiency=cap_sufficiency, cap_necessity=cap_necessity,
        )

    if "game1" not in output and "game2" not in output and "baselines" not in output:
        LOGGER.warning("skip %s: no game/baseline JSON to rescore", root)
        return None

    out_path.write_text(json.dumps(output, indent=2) + "\n")
    merge_kl_into_outputs(root, output, spec=spec)
    return out_path


def rescore_tree(
    root: str | Path,
    *,
    clt_tags: Sequence[str] | None = None,
    force: bool = False,
    spec: RescoreSpec = KL_SPEC,
    spec_for_run: Callable[[Path], RescoreSpec | None] | None = None,
    cap_sufficiency: bool = True,
    cap_necessity: bool = True,
) -> list[Path]:
    """Walk ``root/<clt>/<slug>/`` and rescore every completed run.

    ``spec_for_run`` overrides ``spec`` per run directory (returning ``None``
    skips that run) — used by alt-foil rescoring, where the substituted foil
    token differs per prompt.
    """
    base = Path(root)
    written: list[Path] = []
    if not base.is_dir():
        return written

    for clt_dir in sorted(base.iterdir()):
        if not clt_dir.is_dir():
            continue
        if clt_tags and clt_dir.name not in clt_tags:
            continue
        for slug_dir in sorted(clt_dir.iterdir()):
            if not slug_dir.is_dir():
                continue
            # Dual-utility layout: <clt>/<slug>/<score_kind>/macag_game1.json
            kind_dirs = [
                p for p in sorted(slug_dir.iterdir())
                if p.is_dir() and (
                    (p / "macag_game1.json").is_file() or (p / "macag_baselines.json").is_file()
                )
            ]
            run_dirs = kind_dirs if kind_dirs else [slug_dir]
            for run_dir in run_dirs:
                if not (run_dir / "macag_game1.json").is_file() and not (
                    run_dir / "macag_baselines.json"
                ).is_file():
                    continue
                run_spec: RescoreSpec | None = spec
                if spec_for_run is not None:
                    run_spec = spec_for_run(run_dir)
                    if run_spec is None:
                        LOGGER.info("skip %s: no rescore spec for this run", run_dir)
                        continue
                path = rescore_run_dir(
                    run_dir,
                    force=force,
                    spec=run_spec,
                    cap_sufficiency=cap_sufficiency,
                    cap_necessity=cap_necessity,
                )
                if path is not None:
                    written.append(path)
    return written


def altfoil_spec(foil_token: str) -> RescoreSpec:
    """Rescore under the stored score_kind but with an alternate foil token.

    The stored ``target_token_by_label['y_foil']`` is replaced; everything else
    (score_kind — logit_gap or answer_span — freeze convention, prompt, graph)
    stays as the original run recorded it, so the delta isolates the foil choice
    (the §11.3 "single foil" threat).
    """

    def transform(kwargs: dict[str, Any]) -> dict[str, Any]:
        labels = dict(kwargs.get("target_token_by_label") or {})
        if "y_foil" not in labels:
            raise ValueError(
                "stored oracle kwargs have no target_token_by_label['y_foil'] to substitute"
            )
        labels["y_foil"] = foil_token
        out = dict(kwargs)
        out["target_token_by_label"] = labels
        return out

    return RescoreSpec(
        output_name="macag_altfoil_faithfulness.json",
        embed_key="altfoil_faithfulness",
        score_label="altfoil",
        transform=transform,
    )
