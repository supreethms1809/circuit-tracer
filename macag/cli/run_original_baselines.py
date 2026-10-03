"""Run original-pipeline baselines (native-edge EAP / ACDC) into a sidecar JSON.

Writes ``macag_original_baselines.json`` — never merge into rematch
``macag_baselines.json`` (different ID universe; no Jaccard).

Example::

    python -m macag.cli.run_original_baselines \\
      --oracle-kwargs-file run/logit_gap/oracle_kwargs.json \\
      --input-id mib_gemma2_ioi_0000 \\
      --budget 8 \\
      --methods eap_edge,acdc_edge \\
      --output-json run/logit_gap/macag_original_baselines.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping

LOGGER = logging.getLogger(__name__)

KNOWN_METHODS = ("eap_edge", "acdc_edge")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _load_native_model(model_name: str, model_kwargs: Mapping[str, Any] | None) -> Any:
    import torch
    from transformer_lens import HookedTransformer

    # auto-circuit's PatchWrapper einsum mixes mask + activations; bf16 models
    # (Gemma-2 / Llama-3 defaults) raise RuntimeError. Force float32 for this track.
    kw = dict(model_kwargs or {})
    kw.pop("torch_dtype", None)
    kw["dtype"] = torch.float32
    model = HookedTransformer.from_pretrained(model_name, **kw)
    model.cfg.use_attn_result = True
    return model


def _resolve_tokens(kwargs: Mapping[str, Any], target_label: str = "y") -> tuple[str, str | None, int, int | None]:
    token_by_label = kwargs.get("target_token_by_label") or {}
    foil_by_target = kwargs.get("foil_by_target") or {}
    target_map = kwargs.get("target_to_logit_idx") or {}
    if target_label not in token_by_label:
        raise ValueError(f"oracle kwargs missing target_token_by_label[{target_label!r}]")
    target_token = str(token_by_label[target_label])
    target_idx = int(target_map[target_label]) if target_label in target_map else None
    foil_label = foil_by_target.get(target_label)
    foil_token = str(token_by_label[foil_label]) if foil_label in token_by_label else None
    foil_idx = int(target_map[foil_label]) if foil_label in target_map else None
    return target_token, foil_token, target_idx if target_idx is not None else -1, foil_idx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-kwargs-file", type=Path, required=True)
    parser.add_argument("--input-id", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--budget",
        type=int,
        default=0,
        help="Top-k for budget-matched mode. 0 = native (EAP ranking+sweep; ACDC τ circuit when --acdc-target-k 0).",
    )
    parser.add_argument("--methods", default="eap_edge,acdc_edge")
    parser.add_argument(
        "--acdc-taus",
        default="0.1",
        help="Comma τ list, or 'default'/'auto'/empty for auto-circuit's built-in τ grid.",
    )
    parser.add_argument("--acdc-metric", choices=("logit_gap", "kl", "kl_div", "mse"), default="kl")
    parser.add_argument(
        "--acdc-target-k",
        type=int,
        default=0,
        help="0 = native ACDC τ-survivors (no budget). -1 = match --budget. >0 = explicit k.",
    )
    parser.add_argument("--model-name", default=None, help="Override model_name from oracle kwargs.")
    parser.add_argument("--clean-prompt", default=None)
    parser.add_argument("--corrupted-prompt", default=None)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    kwargs = _load_json(args.oracle_kwargs_file)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in KNOWN_METHODS]
    if unknown:
        raise SystemExit(f"Unknown original methods {unknown}; known={KNOWN_METHODS}")

    model_name = args.model_name or kwargs.get("model_name")
    if not model_name:
        raise SystemExit("model_name required (CLI or oracle kwargs)")
    clean = args.clean_prompt or kwargs.get("prompt")
    corrupt = args.corrupted_prompt or kwargs.get("corrupted_prompt")
    if not clean or not corrupt:
        raise SystemExit("clean + corrupted prompts required")

    target_token, foil_token, target_idx, foil_idx = _resolve_tokens(kwargs)
    if target_idx < 0:
        from macag.baselines.original.metrics import resolve_token_index

        # Resolve after model load.
        target_idx = -1

    model_kwargs = dict(kwargs.get("model_kwargs") or {})
    if args.device:
        model_kwargs["device"] = args.device
    model = _load_native_model(str(model_name), model_kwargs)

    if target_idx < 0:
        from macag.baselines.original.metrics import resolve_token_index

        target_idx = resolve_token_index(model, target_token)
        if foil_token is not None and foil_idx is None:
            foil_idx = resolve_token_index(model, foil_token)

    method_blocks: dict[str, Any] = {}
    if "eap_edge" in methods:
        from macag.baselines.original.eap_edge import run_eap_edge

        eap_k = args.budget if args.budget > 0 else None
        method_blocks["eap_edge"] = run_eap_edge(
            model,
            clean_prompt=str(clean),
            corrupted_prompt=str(corrupt),
            target_token=target_token,
            foil_token=foil_token,
            k=eap_k,
            target_idx=target_idx,
            foil_idx=foil_idx,
        )

    if "acdc_edge" in methods:
        from macag.baselines.original.acdc_edge import run_acdc_edge

        target_k = None
        if args.acdc_target_k != 0:
            if args.budget <= 0 and args.acdc_target_k == -1:
                raise SystemExit(
                    "acdc-target-k=-1 requires --budget > 0; use --acdc-target-k 0 for native τ circuit"
                )
            target_k = args.budget if args.acdc_target_k == -1 else args.acdc_target_k
        taus_raw = (args.acdc_taus or "").strip().lower()
        if taus_raw in ("", "default", "auto"):
            taus = None
        else:
            taus = _parse_float_list(args.acdc_taus)
        method_blocks["acdc_edge"] = run_acdc_edge(
            model,
            clean_prompt=str(clean),
            corrupted_prompt=str(corrupt),
            target_token=target_token,
            foil_token=foil_token,
            taus=taus,
            metric=args.acdc_metric,
            target_k=target_k,
            target_idx=target_idx,
            foil_idx=foil_idx,
        )

    # Optional Game 1 reference metrics from sibling files (size/KL only).
    game1_ref = None
    run_dir = args.oracle_kwargs_file.parent
    g1_path = run_dir / "macag_game1.json"
    kl_path = run_dir / "macag_kl_faithfulness.json"
    if g1_path.is_file():
        g1 = _load_json(g1_path)
        evidence = ((g1.get("evidence") or {}).get("E_star")) or []
        scores = g1.get("scores") or {}
        game1_ref = {
            "size_nodes": len(evidence),
            "logit_gap_faithfulness": scores.get("faithfulness"),
            "note": "CLT feature-node Game 1; size units differ from edge circuits.",
        }
        if kl_path.is_file():
            kl_payload = _load_json(kl_path)
            # Prefer game1 KL if present.
            g1_kl = (kl_payload.get("methods") or {}).get("game1") or kl_payload.get("game1")
            if isinstance(g1_kl, dict):
                game1_ref["kl"] = g1_kl.get("kl") or g1_kl.get("kl_divergence")

    payload = {
        "schema_version": 1,
        "track": "original_pipeline",
        "game": "original_baselines",
        "input_id": args.input_id,
        "experiment_identity": {
            "schema_version": 1,
            "track": "original_pipeline",
            "input_id": args.input_id,
            "model_name": str(model_name),
            "clean_prompt_sha256": _sha256_text(str(clean)),
            "corrupted_prompt_sha256": _sha256_text(str(corrupt)),
            "budget": args.budget,
            "methods": methods,
        },
        "methods": method_blocks,
        "comparison": {
            "notes": {
                "eap_edge": "Native output-edge AtP; not Jaccard-comparable to CLT Game1.",
                "acdc_edge": "Corrupt-patch τ-prune on output edges; pipeline metrics only.",
            },
            "game1_ref": game1_ref,
        },
        "params": {
            "budget": args.budget,
            "acdc_taus": args.acdc_taus,
            "acdc_metric": args.acdc_metric,
            "acdc_target_k": args.acdc_target_k,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    LOGGER.info("Wrote %s", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
