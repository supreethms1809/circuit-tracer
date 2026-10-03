#!/usr/bin/env python3
"""Rescore saved Dallas–Austin evidence without re-searching.

Writes a sidecar JSON. Does not overwrite Game 1 / Game 2 / baseline artifacts.

Modes:
  tokens     last-token argmax, P(Austin), P(Texas), post-hoc KL vs clean
  ablation   same sets under ablation_mode=mean and corrupted (zero is the original)
  both       tokens + ablation (default)

Usage:
  python experiments/dallas_rescore_sets.py \\
    --outdir /gscratch/$USER/macag_dallas_austin_llama/llama32-524k/dallas-austin
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

from macag.factories.replacement_model import create_replacement_model_oracle
from macag.scoring import (
    ScoringOracle,
    _last_token_logits,
    compute_kl_score,
    derive_oracle_with_freeze,
)
from macag.utils.metrics import compute_faithfulness_metrics, metrics_to_dict

LOGGER = logging.getLogger(__name__)

DEFAULT_KIND = "logit_gap"
TARGET = "y"


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _collect_sets(kind_dir: Path) -> dict[str, dict[str, Any]]:
    """Named evidence sets from tagged Game 1 / Game 2 JSON (frozen E* by default)."""
    sets: dict[str, dict[str, Any]] = {}
    for path in sorted(kind_dir.glob("macag_game1*.json")):
        if path.name.endswith(".ckpt.json") or "_compare.json" in path.name or "_pool.json" in path.name:
            continue
        payload = _load(path)
        alpha = float((payload.get("params") or {}).get("alpha", 0.5))
        if payload.get("freeze_mode") == "both":
            for leg in ("frozen", "unfrozen"):
                block = payload.get(leg) or {}
                evidence = list((block.get("evidence") or {}).get("E_star") or [])
                if not evidence:
                    continue
                key = f"game1/{path.stem}/{leg}"
                sets[key] = {
                    "source": str(path),
                    "kind": "game1",
                    "leg": leg,
                    "alpha": float((block.get("params") or {}).get("alpha", alpha)),
                    "freeze_attention": leg == "frozen",
                    "nodes": evidence,
                    "selection_scores": block.get("scores"),
                }
        else:
            evidence = list((payload.get("evidence") or {}).get("E_star") or [])
            if evidence:
                sets[f"game1/{path.stem}"] = {
                    "source": str(path),
                    "kind": "game1",
                    "leg": "single",
                    "alpha": alpha,
                    "freeze_attention": True,
                    "nodes": evidence,
                    "selection_scores": payload.get("scores"),
                }

    for path in sorted(kind_dir.glob("macag_game2*.json")):
        if path.name.endswith(".ckpt.json"):
            continue
        payload = _load(path)
        ev = payload.get("evidence") or {}
        alpha = float((payload.get("params") or {}).get("alpha", 0.5))
        freeze = bool((payload.get("params") or {}).get("freeze_attention", True))
        for side, key_name in (("E_y", "y"), ("E_foil", "foil")):
            nodes = list(ev.get(side) or [])
            if not nodes:
                continue
            sets[f"game2/{path.stem}/{key_name}"] = {
                "source": str(path),
                "kind": "game2",
                "leg": key_name,
                "alpha": alpha,
                "freeze_attention": freeze,
                "nodes": nodes,
                "selection_scores": payload.get("scores"),
            }
    return sets


def _backend(oracle: ScoringOracle) -> Any:
    return oracle.backend


def _interventions_for(backend: Any, nodes: set[str], mode: str) -> list[Any]:
    universe = set(backend.intervention_universe())
    node_set = set(nodes) & universe
    if mode == "remove":
        ablate = node_set
    elif mode == "keep_only":
        ablate = universe - node_set
    elif mode == "empty":
        ablate = universe
    elif mode == "all":
        ablate = set()
    else:
        raise ValueError(f"unknown mode {mode}")
    return backend._ablation_interventions(ablate)


def _token_report(
    backend: Any,
    interventions: list[Any],
    *,
    target_idx: int,
    foil_idx: int | None,
    tokenizer: Any,
    ref_logits: Any,
) -> dict[str, Any]:
    logits = backend._run_logits(interventions)
    token_logits = _last_token_logits(logits)
    probs = torch.softmax(token_logits.float(), dim=-1)
    argmax_id = int(token_logits.argmax().item())
    decode = tokenizer.decode([argmax_id])
    austin_p = float(probs[target_idx].item())
    texas_p = float(probs[foil_idx].item()) if foil_idx is not None else None
    kl = float(compute_kl_score(ref_logits, logits))
    gap = None
    if foil_idx is not None:
        gap = float((token_logits[target_idx] - token_logits[foil_idx]).item())
    return {
        "argmax_id": argmax_id,
        "argmax": decode,
        "p_austin": austin_p,
        "p_texas": texas_p,
        "logit_gap": gap,
        "kl_vs_clean": -kl,  # compute_kl_score returns -KL; report +KL
        "kl_score": kl,
    }


def _score_set(oracle: ScoringOracle, nodes: list[str], alpha: float) -> dict[str, Any]:
    metrics = compute_faithfulness_metrics(
        oracle=oracle, target=TARGET, nodes=set(nodes), alpha=alpha
    )
    return metrics_to_dict(metrics)


def _build_oracle(kwargs: dict[str, Any], ablation_mode: str | None = None) -> ScoringOracle:
    merged = dict(kwargs)
    if ablation_mode is not None:
        merged["ablation_mode"] = ablation_mode
    built = create_replacement_model_oracle(**merged)
    if built.candidates and hasattr(built.oracle.backend, "restrict_universe"):
        built.oracle.backend.restrict_universe(set(built.candidates))
        built.oracle.clear_cache()
    return built.oracle


def _drop_oracle(oracle: ScoringOracle | None) -> None:
    if oracle is None:
        return
    backend = getattr(oracle, "backend", None)
    if backend is not None and hasattr(backend, "model"):
        backend.model = None
    del oracle
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--score-kind", default=DEFAULT_KIND)
    parser.add_argument("--mode", choices=("tokens", "ablation", "both"), default="both")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Default: <kind_dir>/macag_set_rescore.json",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    kind_dir = args.outdir / args.score_kind
    kwargs_path = kind_dir / "oracle_kwargs.json"
    if not kwargs_path.is_file():
        raise SystemExit(f"missing {kwargs_path}")
    out_path = args.output_json or (kind_dir / "macag_set_rescore.json")
    if out_path.is_file() and not args.force:
        LOGGER.info("skip (exists %s); pass --force to overwrite", out_path)
        return 0

    kwargs = _load(kwargs_path)
    sets = _collect_sets(kind_dir)
    if not sets:
        raise SystemExit(f"no Game 1/2 JSON under {kind_dir}")
    LOGGER.info(">>> %d evidence sets under %s", len(sets), kind_dir)

    payload: dict[str, Any] = {
        "outdir": str(args.outdir),
        "score_kind": args.score_kind,
        "mode": args.mode,
        "n_sets": len(sets),
        "sets": {},
    }
    for name, spec in sets.items():
        payload["sets"][name] = {
            "source": spec["source"],
            "kind": spec["kind"],
            "leg": spec["leg"],
            "alpha": float(spec["alpha"]),
            "freeze_attention": bool(spec["freeze_attention"]),
            "nodes": spec["nodes"],
            "size": len(spec["nodes"]),
        }

    want_tokens = args.mode in ("tokens", "both")
    want_ablation = args.mode in ("ablation", "both")
    modes = ["zero"]
    if want_ablation:
        modes.extend(["mean", "corrupted"])

    for abl_mode in modes:
        LOGGER.info(">>> loading oracle ablation_mode=%s", abl_mode)
        oracle = _build_oracle(kwargs, None if abl_mode == "zero" else abl_mode)
        backend0 = _backend(oracle)
        tokenizer = target_idx = foil_idx = ref_logits = None
        if want_tokens and abl_mode == "zero":
            tokenizer = backend0.model.tokenizer
            target_idx = int(backend0.target_to_logit_idx[TARGET])
            foil_label = (backend0.foil_by_target or {}).get(TARGET) or backend0.default_foil
            foil_idx = (
                int(backend0.target_to_logit_idx[foil_label]) if foil_label is not None else None
            )
            ref_logits = backend0._reference_logits()

        freeze_oracles: dict[bool, ScoringOracle] = {}
        for name, spec in sets.items():
            freeze = bool(spec["freeze_attention"])
            if freeze not in freeze_oracles:
                freeze_oracles[freeze] = derive_oracle_with_freeze(
                    oracle, freeze_attention=freeze
                )
            leg_oracle = freeze_oracles[freeze]
            row = payload["sets"][name]
            LOGGER.info(
                ">>> %s ablation=%s |E|=%d freeze=%s",
                name, abl_mode, len(spec["nodes"]), freeze,
            )
            if want_ablation:
                row.setdefault("ablation", {})[abl_mode] = _score_set(
                    leg_oracle, spec["nodes"], float(spec["alpha"])
                )
            if want_tokens and abl_mode == "zero":
                backend = _backend(leg_oracle)
                token_block = {}
                for mode in ("all", "keep_only", "remove"):
                    interventions = _interventions_for(backend, set(spec["nodes"]), mode)
                    token_block[mode] = _token_report(
                        backend,
                        interventions,
                        target_idx=target_idx,
                        foil_idx=foil_idx,
                        tokenizer=tokenizer,
                        ref_logits=ref_logits,
                    )
                row["tokens"] = token_block
                LOGGER.info(
                    "    clean argmax=%r  remove argmax=%r  keep argmax=%r  "
                    "remove P(Austin)=%.4f KL=%.3f",
                    token_block["all"]["argmax"],
                    token_block["remove"]["argmax"],
                    token_block["keep_only"]["argmax"],
                    token_block["remove"]["p_austin"],
                    token_block["remove"]["kl_vs_clean"],
                )

        freeze_oracles.clear()
        _drop_oracle(oracle)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    LOGGER.info(">>> wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
