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
) -> dict[str, Any]:
    metrics = compute_faithfulness_metrics(
        oracle=oracle,
        target=target,
        nodes=set(nodes),
        alpha=alpha,
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


def rescore_game1_leg(
    leg: Mapping[str, Any],
    oracle: ScoringOracle,
    *,
    target: TargetId = DEFAULT_TARGET,
    score_label: str = "kl_divergence",
) -> dict[str, Any]:
    evidence = leg.get("evidence", {}).get("E_star") or []
    alpha = _alpha_from_payload(leg)
    logit_scores = leg.get("scores") or {}
    kl_block = _rescore_nodes(oracle, target=target, nodes=evidence, alpha=alpha)
    return {
        "evidence_size": len(evidence),
        "alpha": alpha,
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
) -> dict[str, Any]:
    alpha = _alpha_from_payload(payload)
    evidence = payload.get("evidence") or {}
    e_y = evidence.get("E_y") or []
    e_foil = evidence.get("E_foil") or []
    scores = payload.get("scores") or {}
    kl_y = _rescore_nodes(oracle, target=target, nodes=e_y, alpha=alpha)
    oracle.clear_cache()
    oracle.reset_stats()
    kl_foil = _rescore_nodes(oracle, target=foil, nodes=e_foil, alpha=alpha)
    return {
        "alpha": alpha,
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


def rescore_baselines(
    payload: Mapping[str, Any],
    oracle: ScoringOracle,
    *,
    target: TargetId = DEFAULT_TARGET,
    score_label: str = "kl_divergence",
) -> dict[str, Any]:
    alpha = _alpha_from_payload(payload)
    budget = int(payload.get("params", {}).get("budget", 8))
    methods_out: dict[str, Any] = {}
    for method, entry in (payload.get("methods") or {}).items():
        if method == "acdc":
            best = (entry.get("best_by_size") or {}).get(str(budget))
            if best is None:
                sizes = sorted(int(s) for s in (entry.get("best_by_size") or {}))
                if not sizes:
                    continue
                best = entry["best_by_size"][str(max(sizes))]
            evidence = best.get("evidence") or []
            logit_faith = (best.get("scores") or {}).get("faithfulness")
        else:
            evidence, own_k = _faith_at_own_k(entry.get("results") or {}, budget)
            logit_faith = None
            if own_k is not None:
                logit_faith = (
                    (entry.get("results") or {}).get(str(own_k), {}).get("scores") or {}
                ).get("faithfulness")
        if not evidence:
            continue
        oracle.clear_cache()
        oracle.reset_stats()
        kl_block = _rescore_nodes(oracle, target=target, nodes=evidence, alpha=alpha)
        methods_out[method] = {
            "evidence_size": len(evidence),
            "logit_gap_faithfulness": logit_faith,
            score_label: kl_block["scores"],
            "oracle_calls": kl_block["oracle_calls"],
        }
    return {"alpha": alpha, "budget": budget, "methods": methods_out}


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
    if g1_path.is_file() and kl_g1:
        g1 = load_json(g1_path)
        if g1.get("freeze_mode") == "both":
            for leg_name in ("frozen", "unfrozen"):
                if leg_name in kl_g1 and leg_name in g1:
                    g1[leg_name][spec.embed_key] = kl_g1[leg_name][label]
        elif "single" in kl_g1:
            g1[spec.embed_key] = kl_g1["single"][label]
        g1_path.write_text(json.dumps(g1, indent=2) + "\n")

    g2_path = root / "macag_game2.json"
    kl_g2 = kl_payload.get("game2") or {}
    if g2_path.is_file() and kl_g2:
        g2 = load_json(g2_path)
        g2[spec.embed_key] = kl_g2.get(label)
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
                    leg, oracle, target=target, score_label=label
                )
        else:
            freeze = bool(kwargs.get("freeze_attention", True))
            oracle = build_oracle_with_overrides(kwargs, spec, freeze_attention=freeze)
            output["game1"]["single"] = rescore_game1_leg(
                g1, oracle, target=target, score_label=label
            )

    g2_path = root / "macag_game2.json"
    if g2_path.is_file():
        g2 = load_json(g2_path)
        freeze = bool(kwargs.get("freeze_attention", True))
        oracle = build_oracle_with_overrides(kwargs, spec, freeze_attention=freeze)
        output["game2"] = rescore_game2(g2, oracle, target=target, foil=foil, score_label=label)

    bl_path = root / "macag_baselines.json"
    if bl_path.is_file():
        bl = load_json(bl_path)
        freeze = bool(kwargs.get("freeze_attention", True))
        oracle = build_oracle_with_overrides(kwargs, spec, freeze_attention=freeze)
        output["baselines"] = rescore_baselines(bl, oracle, target=target, score_label=label)

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
            if not (slug_dir / "macag_game1.json").is_file() and not (
                slug_dir / "macag_baselines.json"
            ).is_file():
                continue
            run_spec: RescoreSpec | None = spec
            if spec_for_run is not None:
                run_spec = spec_for_run(slug_dir)
                if run_spec is None:
                    LOGGER.info("skip %s: no rescore spec for this run", slug_dir)
                    continue
            path = rescore_run_dir(slug_dir, force=force, spec=run_spec)
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
