#!/usr/bin/env python3
"""Dallas–Austin AtP counterfactual diagnostic.

Recomputes feature AtP (``eap_syed``) and native-edge EAP on the existing
Dallas graph. Houston is in Texas, so ``a_corr − a_clean`` on a Texas/capital
feature can be ~0 even if the implementation is correct.

If the Game 1 hub stays AtP=0 on Houston but moves when the city is not in
Texas, the Dallas failure is the counterfactual, not a bug. If the hub stays
0 on every equal-length corrupt prompt, the decoder-AtP path is the suspect.

Does not overwrite Dallas method JSON. Writes a diagnostic sidecar.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

from macag.baselines.eap_syed import select_top_eap_syed
from macag.factories.replacement_model import create_replacement_model_scorer

LOGGER = logging.getLogger(__name__)

HUBS = (
    "0_25454_10",
    "15_16228_10",
    "9_25557_10",
    "3_18384_10",
    "0_12390_10",
)

CANDIDATE_CITIES = (
    "Houston",  # original Dallas corrupt (still Texas)
    "Boston",
    "Denver",
    "Miami",
    "Chicago",
    "Seattle",
    "Atlanta",
    "Detroit",
    "Paris",
    "London",
)


def _load_kwargs(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _template(city: str) -> str:
    return f"Fact: The capital of the state containing {city} is"


def _summarize_atp(
    ranking: list[str],
    abs_scores: dict[str, float],
    signed: dict[str, float],
    *,
    a_clean: float | None,
    a_corr: float | None,
) -> dict[str, Any]:
    rank = {node: i + 1 for i, node in enumerate(ranking)}
    vals = list(abs_scores.values())
    n_zero = sum(1 for v in vals if v == 0.0)
    hubs = {}
    for node in HUBS:
        hubs[node] = {
            "rank": rank.get(node),
            "abs": abs_scores.get(node),
            "signed": signed.get(node, signed.get(str(node))),
        }
    return {
        "n_candidates": len(ranking),
        "n_exact_zero": n_zero,
        "max_abs": max(vals) if vals else None,
        "top5": ranking[:5],
        "top5_abs": [abs_scores[n] for n in ranking[:5]],
        "hubs": hubs,
        "hub_activation": {"a_clean": a_clean, "a_corr": a_corr, "delta": None if a_clean is None or a_corr is None else a_corr - a_clean},
    }


def _hub_activation(model: Any, tokens: torch.Tensor, spec: tuple[int, int, int]) -> float:
    with torch.inference_mode():
        _logits, acts = model.get_activations(tokens, sparse=False)
    layer, pos, feat = spec
    return float(acts[layer, pos, feat].item())


def _feature_atp_for_corrupt(
    scorer: Any,
    *,
    prompt: str,
    corrupted_prompt: str,
    candidates: list[str],
    target_logit: int,
    foil_logit: int | None,
) -> dict[str, Any]:
    model = scorer.model
    spec = scorer.node_to_intervention[HUBS[0]]
    layer, pos, feat = int(spec[0]), int(spec[1]), int(spec[2])
    clean_tokens = model.ensure_tokenized(prompt)
    corr_tokens = model.ensure_tokenized(corrupted_prompt)
    a_clean = _hub_activation(model, clean_tokens, (layer, pos, feat))
    a_corr = _hub_activation(model, corr_tokens, (layer, pos, feat))
    result = select_top_eap_syed(
        model,
        prompt=prompt,
        corrupted_prompt=corrupted_prompt,
        node_to_intervention=scorer.node_to_intervention,
        candidates=candidates,
        target_logit_idx=target_logit,
        foil_logit_idx=foil_logit,
        freeze_attention=bool(getattr(scorer, "freeze_attention", True)),
        use_absolute=True,
    )
    signed = (result.extras or {}).get("signed_scores") or {}
    summary = _summarize_atp(
        result.ranking,
        result.scores,
        signed,
        a_clean=a_clean,
        a_corr=a_corr,
    )
    summary["seq_len_clean"] = int(clean_tokens.numel())
    summary["seq_len_corrupt"] = int(corr_tokens.numel())
    summary["grad_path"] = (result.params or {}).get("grad_path")
    summary["missing_grad_count"] = (result.extras or {}).get("missing_grad_count")
    return summary


def _native_eap(kwargs: dict[str, Any], corrupted_prompt: str) -> dict[str, Any]:
    from macag.baselines.original.eap_edge import run_eap_edge
    from macag.cli.run_original_baselines import _load_native_model, _resolve_tokens

    model = _load_native_model(str(kwargs["model_name"]), kwargs.get("model_kwargs"))
    target_token, foil_token, target_idx, foil_idx = _resolve_tokens(kwargs, "y")
    payload = run_eap_edge(
        model,
        clean_prompt=str(kwargs["prompt"]),
        corrupted_prompt=corrupted_prompt,
        target_token=target_token,
        foil_token=foil_token,
        k=None,
        target_idx=target_idx if target_idx >= 0 else None,
        foil_idx=foil_idx,
    )
    sweep = payload.get("sweep") or []
    compact = []
    for row in sweep:
        scores = row.get("scores") or {}
        compact.append(
            {
                "size": row.get("size") or row.get("realized_edge_count"),
                "logit_gap": scores.get("logit_gap"),
                "kl": scores.get("kl"),
            }
        )
    return {
        "status": payload.get("status"),
        "n_graph_edges": (payload.get("selection_stats") or {}).get("n_graph_edges"),
        "wall_s": (payload.get("selection_stats") or {}).get("wall_s"),
        "sweep": compact,
        "top5": (payload.get("ranking") or [])[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-kwargs-file",
        type=Path,
        default=Path(
            "/gscratch/ssuresh/macag_dallas_austin_llama/llama32-524k/dallas-austin/"
            "kl_divergence/oracle_kwargs.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "/gscratch/ssuresh/macag_dallas_austin_llama/llama32-524k/dallas-austin/"
            "diagnostics/atp_counterfactual.json"
        ),
    )
    parser.add_argument("--native-eap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-cities", type=int, default=6)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    kwargs = _load_kwargs(args.oracle_kwargs_file)
    prompt = str(kwargs["prompt"])
    original_corrupt = str(kwargs.get("corrupted_prompt") or _template("Houston"))

    LOGGER.info("loading ReplacementModel from %s", args.oracle_kwargs_file)
    scorer = create_replacement_model_scorer(**kwargs)
    model = scorer.model
    candidates = sorted(str(n) for n in scorer.node_to_intervention)
    target_logit = int(scorer.target_to_logit_idx["y"])
    foil_label = (scorer.foil_by_target or {}).get("y")
    foil_logit = (
        int(scorer.target_to_logit_idx[foil_label])
        if foil_label is not None and foil_label in scorer.target_to_logit_idx
        else None
    )
    clean_len = int(model.ensure_tokenized(prompt).numel())
    LOGGER.info("clean seq_len=%d n_candidates=%d target=%s foil=%s", clean_len, len(candidates), target_logit, foil_logit)

    matched: list[tuple[str, str]] = []
    seen: set[str] = set()
    for city in CANDIDATE_CITIES:
        text = original_corrupt if city == "Houston" else _template(city)
        if text in seen:
            continue
        n = int(model.ensure_tokenized(text).numel())
        LOGGER.info("tokenize city=%s seq_len=%d match=%s", city, n, n == clean_len)
        if n == clean_len:
            matched.append((city, text))
            seen.add(text)
        if len(matched) >= args.max_cities:
            break
    if not any(c == "Houston" for c, _ in matched):
        matched.insert(0, ("Houston", original_corrupt))

    feature_atp: dict[str, Any] = {}
    for city, corrupt in matched:
        LOGGER.info("feature AtP corrupt=%s", city)
        feature_atp[city] = _feature_atp_for_corrupt(
            scorer,
            prompt=prompt,
            corrupted_prompt=corrupt,
            candidates=candidates,
            target_logit=target_logit,
            foil_logit=foil_logit,
        )
        hub = feature_atp[city]["hubs"][HUBS[0]]
        LOGGER.info(
            "  hub rank=%s abs=%s delta_a=%s top5=%s",
            hub["rank"],
            hub["abs"],
            feature_atp[city]["hub_activation"]["delta"],
            feature_atp[city]["top5"],
        )

    native: dict[str, Any] = {}
    if args.native_eap:
        native_cities = [pair for pair in matched if pair[0] in ("Houston", "Boston", "Denver", "Paris")]
        if len(native_cities) < 2:
            native_cities = matched[:2]
        for city, corrupt in native_cities:
            LOGGER.info("native eap_edge corrupt=%s", city)
            native[city] = _native_eap(kwargs, corrupt)
            LOGGER.info("  sweep[0:3]=%s", (native[city].get("sweep") or [])[:3])

    houston = feature_atp.get("Houston") or {}
    others = {k: v for k, v in feature_atp.items() if k != "Houston"}
    hub_h = ((houston.get("hubs") or {}).get(HUBS[0]) or {}).get("abs")
    moved = []
    still_zero = []
    for city, summary in others.items():
        abs_s = ((summary.get("hubs") or {}).get(HUBS[0]) or {}).get("abs")
        if abs_s is None:
            continue
        if hub_h is not None and abs_s > 0.0 and abs_s > (hub_h or 0.0):
            moved.append({"city": city, "hub_abs": abs_s, "hub_rank": summary["hubs"][HUBS[0]]["rank"]})
        if abs_s == 0.0:
            still_zero.append(city)

    if hub_h == 0.0 and moved:
        verdict = (
            "counterfactual: hub AtP is 0 on Houston (Texas) and nonzero on a "
            "non-Texas equal-length city — Dallas AtP failure is the corrupt pair, not a bug"
        )
    elif hub_h == 0.0 and others and all(
        ((s.get("hubs") or {}).get(HUBS[0]) or {}).get("abs") == 0.0 for s in others.values()
    ):
        verdict = (
            "implementation_suspect: hub AtP stays 0 on every equal-length corrupt "
            "prompt — decoder-AtP path or hub mapping needs a closer look"
        )
    else:
        verdict = "inconclusive: inspect per-city hub abs/rank in feature_atp"

    payload = {
        "prompt": prompt,
        "original_corrupt": original_corrupt,
        "clean_seq_len": clean_len,
        "target_logit_idx": target_logit,
        "foil_logit_idx": foil_logit,
        "verdict": verdict,
        "hub_moved_on": moved,
        "hub_still_zero_on": still_zero,
        "feature_atp": feature_atp,
        "native_eap_edge": native,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    LOGGER.info("wrote %s", args.output_json)
    LOGGER.info("verdict: %s", verdict)
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
