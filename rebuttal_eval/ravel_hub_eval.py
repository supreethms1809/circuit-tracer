"""RAVEL replacement-fidelity eval for published hub CLTs (mntss/*).

Runs the same prompt metrics as the paper-eval RAVEL suites (top-1, KL,
attribution graphs, circuit keep-only / gap-drop) against HuggingFace CLTs
already cached under ``HF_HOME``, without going through ``load_spline_clt``
or requiring a val activation cache.

Usage:
  python -m rebuttal_eval.ravel_hub_eval \\
      --suite experiments/paper_configs/suites/paper_v3_ravel_hub_llama32.json \\
      --out-dir /gscratch/ssuresh/results/paper/ravel_eval_suite_v3_hub_llama32

  # or directly:
  python -m rebuttal_eval.ravel_hub_eval \\
      --model meta-llama/Llama-3.2-1B \\
      --transcoder-set mntss/clt-llama-3.2-1b-524k \\
      --prompts-from experiments/paper_configs/suites/paper_v3_ravel_dtsweep_spline_dt768.json \\
      --out-dir results/rebuttal/ravel_hub_llama32
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from huggingface_hub import snapshot_download

from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.transcoder.cross_layer_transcoder import load_clt
from rebuttal_eval.common import Provenance, emit, fmt, git_sha
from spline_clt.paper.evaluate import (
    build_logit_gap_direction,
    build_prompt_graph,
    collect_prompt_cache,
    evaluate_prompt_replacement,
    load_ranked_feature_nodes,
    parse_feature_node_id,
    replacement_logits_from_reconstruction,
)
from spline_clt.seed import seed_everything


def _load_language_model_local(
    hub_name: str, device: torch.device, dtype: torch.dtype
) -> Any:
    """Load HookedTransformer from a warm HF cache without network calls.

    Keeps the hub *name* for TransformerLens config mapping, but feeds weights
    from the local snapshot (same pattern as ``measure_ref_clt_l0``).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer

    model_path = str(_local_snapshot(hub_name))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, local_files_only=True
    )
    lm = HookedTransformer.from_pretrained(
        hub_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        dtype=dtype,
        local_files_only=True,
    )
    lm.eval()
    return lm


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


@torch.no_grad()
def _logit_gap_from_subset(
    model: Any,
    prompt_cache: Any,
    lm: Any,
    target_idx: int,
    foil_idx: int,
    selected_node_ids: list[str] | None,
) -> float:
    """Target−foil gap using dense encode + sparse decode (hub CLT compatible)."""
    baseline = model.encode(prompt_cache.mlp_inputs)
    if selected_node_ids is not None:
        masked = torch.zeros_like(baseline)
        for node_id in selected_node_ids:
            triple = parse_feature_node_id(node_id)
            if triple is None:
                continue
            layer_id, position, feature_id = triple
            if (
                0 <= layer_id < masked.shape[0]
                and 0 <= position < masked.shape[1]
                and 0 <= feature_id < masked.shape[2]
            ):
                masked[layer_id, position, feature_id] = baseline[
                    layer_id, position, feature_id
                ]
    else:
        masked = baseline
    reconstruction = model.decode(masked.to_sparse(), input_acts=prompt_cache.mlp_inputs)
    logits = replacement_logits_from_reconstruction(prompt_cache, reconstruction, lm)
    final = logits[-1].float()
    return float((final[target_idx] - final[foil_idx]).item())


def _all_active_feature_node_ids(model: Any, prompt_cache: Any) -> list[str]:
    acts = model.encode(prompt_cache.mlp_inputs)
    active = (acts > 0).nonzero(as_tuple=False)
    return [f"{int(l)}_{int(f)}_{int(p)}" for l, p, f in active.tolist()]


def _load_suite(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_refs_from_suite(suite: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract hub CLT variants from a suite JSON."""
    refs: list[dict[str, Any]] = []
    for name, variant in suite.get("model_variants", {}).items():
        transcoder_set = variant.get("transcoder_set")
        checkpoint = variant.get("checkpoint_path")
        model_name = variant.get("model_name") or suite.get("dataset", {}).get("model_name")
        if not transcoder_set and not checkpoint:
            continue
        refs.append(
            {
                "label": name,
                "display": variant.get("label", name),
                "model": model_name,
                "transcoder_set": transcoder_set,
                "checkpoint_path": checkpoint,
                "scan_id": variant.get("scan_id") or name,
                "n_layers": (variant.get("training") or {}).get("n_layers"),
                "d_transcoder": (variant.get("training") or {}).get("d_transcoder"),
                "d_model": (variant.get("training") or {}).get("d_model"),
            }
        )
    return refs


def _local_snapshot(repo_id: str) -> Path:
    """Resolve a hub repo to its local snapshot dir (no network)."""
    return Path(snapshot_download(repo_id, local_files_only=True))


def _load_transcoder(
    *,
    transcoder_set: str | None,
    checkpoint_path: str | None,
    scan_id: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    if checkpoint_path:
        path = Path(checkpoint_path)
    else:
        assert transcoder_set is not None
        path = _local_snapshot(transcoder_set)
    return load_clt(
        str(path),
        feature_input_hook="hook_resid_mid",
        feature_output_hook="hook_mlp_out",
        scan=scan_id,
        device=device,
        dtype=dtype,
        lazy_decoder=True,
        lazy_encoder=False,
    )


def evaluate_ref(
    *,
    ref: dict[str, Any],
    entries: list[dict[str, Any]],
    out_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    graph_cfg: dict[str, Any],
    circuit_cfg: dict[str, Any],
    seed: int,
    skip_graphs: bool,
) -> dict[str, Any]:
    seed_everything(seed)
    label = ref["label"]
    run_dir = out_dir / "runs" / label / f"seed_{seed}" / "evaluation"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "evaluation_summary.json"
    if summary_path.exists():
        print(f"skip {label}: {summary_path} exists")
        return json.loads(summary_path.read_text())

    print(f"[{label}] loading CLT + base model {ref['model']} (local cache)")
    model = _load_transcoder(
        transcoder_set=ref.get("transcoder_set"),
        checkpoint_path=ref.get("checkpoint_path"),
        scan_id=ref["scan_id"],
        device=device,
        dtype=dtype,
    )
    model.eval()
    # Compat shims used by a few evaluate helpers / reporting.
    if not hasattr(model, "encoder_type"):
        model.encoder_type = "linear"  # type: ignore[attr-defined]
    if not hasattr(model, "device"):
        model.device = device  # type: ignore[attr-defined]

    lm = _load_language_model_local(ref["model"], device=device, dtype=dtype)
    print(f"[{label}] caching {len(entries)} prompts")
    cached: list[tuple[dict[str, Any], Any]] = []
    for i, entry in enumerate(entries, 1):
        if i % 50 == 0 or i == 1:
            print(f"  cache {i}/{len(entries)}: {entry['prompt_id']}")
        cached.append(
            (
                entry,
                collect_prompt_cache(
                    lm=lm,
                    prompt=entry["prompt"],
                    n_layers=model.n_layers,
                    feature_input_hook=model.feature_input_hook,
                    feature_output_hook=model.feature_output_hook,
                ),
            )
        )

    lm_handle = SimpleNamespace(
        W_U=lm.W_U,
        W_E=lm.W_E,
        b_U=getattr(lm, "b_U", None),
        ln_final=lm.ln_final,
        tokenizer=lm.tokenizer,
        cfg=lm.cfg,
    )
    del lm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    replacement_model = None
    if not skip_graphs:
        print(f"[{label}] loading ReplacementModel for attribution")
        # Rebuild HF weights + tokenizer from the same local snapshot (lm was
        # deleted to free VRAM). Tokenizer must be passed explicitly —
        # HookedTransformer otherwise calls AutoTokenizer.from_pretrained on
        # the hub *name*, which 429s / fails under HF_HUB_OFFLINE.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = str(_local_snapshot(ref["model"]))
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
        replacement_model = ReplacementModel.from_pretrained_and_transcoders(
            model_name=ref["model"],
            transcoders=model,
            backend="transformerlens",
            device=device,
            dtype=dtype,
            hf_model=hf_model,
            tokenizer=tokenizer,
            local_files_only=True,
        )

    graphs_dir = run_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for i, (entry, prompt_cache) in enumerate(cached, 1):
        print(f"[{label}] prompt {i}/{len(entries)}: {entry['prompt_id']}")
        record: dict[str, Any] = {
            "record_type": "prompt",
            "variant_name": label,
            "seed": seed,
            "prompt_id": entry["prompt_id"],
            "family": entry.get("family"),
            "prompt": entry["prompt"],
            "target_token": entry["target_token"],
            "foil_token": entry["foil_token"],
            "transcoder_set": ref.get("transcoder_set"),
            "checkpoint_path": ref.get("checkpoint_path"),
            "status": "ok",
        }
        try:
            replacement = evaluate_prompt_replacement(model, prompt_cache, lm_handle)
            record.update(replacement)

            if skip_graphs or replacement_model is None:
                record["status"] = "ok_no_graph"
            else:
                slug = _sanitize(f"{label}_{seed}_{entry['prompt_id']}")
                graph_info = build_prompt_graph(
                    replacement_model=replacement_model,
                    prompt_cache=prompt_cache,
                    graph_dir=graphs_dir,
                    graph_slug=slug,
                    scan_id=ref["scan_id"],
                    max_features=int(graph_cfg.get("max_features", 7500)),
                    max_n_logits=int(graph_cfg.get("max_n_logits", 10)),
                    desired_logit_prob=float(graph_cfg.get("desired_logit_prob", 0.99)),
                    node_threshold=float(graph_cfg.get("node_threshold", 0.8)),
                    edge_threshold=float(graph_cfg.get("edge_threshold", 0.98)),
                    attribution_batch_size=int(
                        graph_cfg.get("attribution_batch_size", 256)
                    ),
                    model=model,
                    lm=lm_handle,
                )
                record.update(
                    {
                        "active_feature_count": graph_info["active_feature_count"],
                        "retained_feature_node_count": graph_info[
                            "retained_feature_node_count"
                        ],
                        "retained_error_node_count": graph_info[
                            "retained_error_node_count"
                        ],
                        "retained_error_node_fraction": graph_info[
                            "retained_error_node_fraction"
                        ],
                        "graph_replacement_score": graph_info["graph_replacement_score"],
                        "graph_completeness_score": graph_info[
                            "graph_completeness_score"
                        ],
                        "graph_json_path": graph_info["graph_json_path"],
                    }
                )

                target_idx, foil_idx, _ = build_logit_gap_direction(
                    tokenizer=lm_handle.tokenizer,
                    unembed=lm_handle.W_U.float(),
                    target_token=entry["target_token"],
                    foil_token=entry["foil_token"],
                )
                causal_nodes = load_ranked_feature_nodes(
                    Path(graph_info["graph_json_path"]),
                    top_k=int(circuit_cfg.get("top_k_features", 32)),
                )
                full_gap = _logit_gap_from_subset(
                    model, prompt_cache, lm_handle, target_idx, foil_idx, None
                )
                keep_only_gap = _logit_gap_from_subset(
                    model, prompt_cache, lm_handle, target_idx, foil_idx, causal_nodes
                )
                active_nodes = _all_active_feature_node_ids(model, prompt_cache)
                remaining = sorted(set(active_nodes) - set(causal_nodes))
                remove_gap = _logit_gap_from_subset(
                    model, prompt_cache, lm_handle, target_idx, foil_idx, remaining
                )
                denom = abs(full_gap) if abs(full_gap) > 1e-8 else float("nan")
                record["keep_only_gap_ratio"] = (
                    keep_only_gap / denom if denom == denom else float("nan")
                )
                record["gap_drop_ratio"] = (
                    (full_gap - remove_gap) / denom if denom == denom else float("nan")
                )
                record["full_gap"] = full_gap
                record["keep_only_gap"] = keep_only_gap
                record["remove_gap"] = remove_gap
        except Exception as exc:  # noqa: BLE001 — per-prompt resilience
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR: {record['error']}", file=sys.stderr)
        records.append(record)

    metrics_path = run_dir / "prompt_metrics.jsonl"
    with metrics_path.open("w") as fh:
        for row in records:
            fh.write(json.dumps(row, default=str) + "\n")

    ok = [r for r in records if r.get("status", "").startswith("ok")]
    def avg(key: str) -> float | None:
        vals = [float(r[key]) for r in ok if r.get(key) is not None and r[key] == r[key]]
        return _mean(vals)

    summary = {
        "suite_name": out_dir.name,
        "variant_name": label,
        "display_label": ref.get("display"),
        "seed": seed,
        "model": ref["model"],
        "transcoder_set": ref.get("transcoder_set"),
        "checkpoint_path": ref.get("checkpoint_path"),
        "n_layers": model.n_layers,
        "d_transcoder": model.d_transcoder,
        "d_model": model.d_model,
        "git_sha": git_sha(),
        "prompt_metrics": {
            "count": len(ok),
            "error_count": sum(1 for r in records if r.get("status") == "error"),
            "top1_match_rate": avg("top1_match_rate"),
            "top5_match_rate": avg("top5_match_rate"),
            "top10_match_rate": avg("top10_match_rate"),
            "kl_divergence": avg("kl_divergence"),
        },
        "circuit_metrics": {
            "active_feature_count": avg("active_feature_count"),
            "retained_feature_node_count": avg("retained_feature_node_count"),
            "retained_error_node_count": avg("retained_error_node_count"),
            "retained_error_node_fraction": avg("retained_error_node_fraction"),
            "graph_replacement_score": avg("graph_replacement_score"),
            "graph_completeness_score": avg("graph_completeness_score"),
            "keep_only_gap_ratio": avg("keep_only_gap_ratio"),
            "gap_drop_ratio": avg("gap_drop_ratio"),
        },
        "note": (
            "Published hub CLT reference eval. Reconstruction / monosemanticity "
            "skipped (no val activation cache). Compare top-1 / KL / graph metrics "
            "to gpt2-small Spline-CLT RAVEL suites."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# RAVEL hub CLT reference evaluation",
        "",
        f"- git: {payload.get('git_sha')}",
        "",
        "| Label | Base | CLT | N | Top-1 | Top-5 | Top-10 | KL | Graph repl | Keep-only |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("variants", []):
        pm = row.get("prompt_metrics", {})
        cm = row.get("circuit_metrics", {})
        lines.append(
            f"| {row.get('display_label') or row.get('variant_name')} "
            f"| {row.get('model')} | `{row.get('transcoder_set') or row.get('checkpoint_path')}` "
            f"| {pm.get('count')} "
            f"| {fmt(pm.get('top1_match_rate'), 3)} "
            f"| {fmt(pm.get('top5_match_rate'), 3)} "
            f"| {fmt(pm.get('top10_match_rate'), 3)} "
            f"| {fmt(pm.get('kl_divergence'), 3)} "
            f"| {fmt(cm.get('graph_replacement_score'), 3)} "
            f"| {fmt(cm.get('keep_only_gap_ratio'), 3)} |"
        )
    lines.append("")
    lines.append(payload.get("note", ""))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="",
                        help="suite JSON with model_variants[].transcoder_set")
    parser.add_argument("--prompts-from", default="",
                        help="suite JSON used only for benchmark_entries")
    parser.add_argument("--model", default="")
    parser.add_argument("--transcoder-set", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # float32 default: evaluate_prompt_replacement mixes CLT outputs with
    # float residual/unembed math; bf16 CLT weights raise dtype mismatches
    # on Llama/Gemma ReplacementModel paths.
    parser.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--skip-graphs", action="store_true",
                        help="replacement fidelity only (faster)")
    parser.add_argument("--max-prompts", type=int, default=0,
                        help="optional cap for smoke tests")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    device = torch.device(args.device)

    graph_cfg: dict[str, Any] = {}
    circuit_cfg: dict[str, Any] = {}
    entries: list[dict[str, Any]] = []
    refs: list[dict[str, Any]] = []

    if args.suite:
        suite = _load_suite(Path(args.suite))
        entries = list(suite.get("benchmark_entries", []))
        refs = _resolve_refs_from_suite(suite)
        graph_cfg = suite.get("evaluation", {}).get("graph", {})
        circuit_cfg = suite.get("evaluation", {}).get("circuit", {})
        (out_dir / "resolved_config.json").write_text(json.dumps(suite, indent=2))
    else:
        if not (args.model and args.transcoder_set and args.prompts_from):
            parser.error(
                "provide --suite, or (--model --transcoder-set --prompts-from)"
            )
        prompts_suite = _load_suite(Path(args.prompts_from))
        entries = list(prompts_suite.get("benchmark_entries", []))
        graph_cfg = prompts_suite.get("evaluation", {}).get("graph", {})
        circuit_cfg = prompts_suite.get("evaluation", {}).get("circuit", {})
        label = args.label or args.transcoder_set.replace("/", "_")
        refs = [
            {
                "label": label,
                "display": label,
                "model": args.model,
                "transcoder_set": args.transcoder_set,
                "checkpoint_path": None,
                "scan_id": label,
            }
        ]

    if args.max_prompts > 0:
        entries = entries[: args.max_prompts]
    if not entries:
        raise SystemExit("no benchmark_entries to evaluate")
    if not refs:
        raise SystemExit("no hub CLT variants found (need transcoder_set or checkpoint_path)")

    summaries = []
    for ref in refs:
        summaries.append(
            evaluate_ref(
                ref=ref,
                entries=entries,
                out_dir=out_dir,
                device=device,
                dtype=dtype,
                graph_cfg=graph_cfg,
                circuit_cfg=circuit_cfg,
                seed=args.seed,
                skip_graphs=args.skip_graphs,
            )
        )

    payload = {
        "git_sha": git_sha(),
        "variants": summaries,
        "note": (
            "Published mntss CLT RAVEL reference. Compare top-1/KL against "
            "gpt2-small Spline/Linear RAVEL suite aggregates — not a "
            "disentanglement (cause/isolate) score."
        ),
    }
    (out_dir / "aggregate_metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    emit(out_dir, "ravel_hub_eval", payload, _render_md(payload))

    provenance = Provenance(
        script="rebuttal_eval.ravel_hub_eval",
        checkpoint=",".join(
            str(r.get("transcoder_set") or r.get("checkpoint_path")) for r in refs
        ),
        seed=args.seed,
    )
    for row in summaries:
        provenance.record(
            "2.4",
            f"{row['variant_name']}.top1",
            row["prompt_metrics"]["top1_match_rate"],
        )
        provenance.record(
            "2.4",
            f"{row['variant_name']}.kl",
            row["prompt_metrics"]["kl_divergence"],
        )
    provenance.write(out_dir)
    print(f"done -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
