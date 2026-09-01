"""Measure L0 of published reference linear CLTs on natural text.

Gives the sparsity target for a healthy linear baseline (HANDOFF §D.1).
Convention matches ``check_reconstruction``: L0/layer/token =
mean over (layer, position) of #{features > 0}. Also reports the
summed-over-layers total (per-token L0) and per-layer breakdown.

Usage:
  python -m rebuttal_eval.measure_ref_clt_l0 \\
      --model meta-llama/Llama-3.2-1B \\
      --transcoder-set mntss/clt-llama-3.2-1b-524k \\
      --out-dir results/rebuttal/ref_clt_l0 \\
      [--n-sequences 64] [--max-seq-len 128] [--seed 101]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from huggingface_hub import snapshot_download
from safetensors import safe_open

from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.transcoder.cross_layer_transcoder import load_clt
from rebuttal_eval.common import Provenance, emit, fmt, git_sha
import yaml

# Offline-friendly: arrow file already on disk under HF datasets cache.
_WIKITEXT_TEST_ARROW = (
    "/gscratch/ssuresh/datasets/Salesforce___wikitext/"
    "wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/"
    "wikitext-test.arrow"
)

DEFAULT_REFS = [
    {
        "label": "llama32_524k",
        "model": "meta-llama/Llama-3.2-1B",
        "transcoder_set": "mntss/clt-llama-3.2-1b-524k",
    },
    {
        "label": "gemma2_426k",
        "model": "google/gemma-2-2b",
        "transcoder_set": "mntss/clt-gemma-2-2b-426k",
    },
]


def _load_wikitext_texts(n_sequences: int, seed: int, min_chars: int = 80) -> list[str]:
    """Deterministic sample of long wikitext-2 test lines (offline arrow)."""
    arrow = Path(os.environ.get("WIKITEXT_TEST_ARROW", _WIKITEXT_TEST_ARROW))
    if not arrow.is_file():
        raise FileNotFoundError(
            f"wikitext test arrow not found at {arrow}; set WIKITEXT_TEST_ARROW"
        )
    ds = Dataset.from_file(str(arrow))
    candidates = [t.strip() for t in ds["text"] if len(t.strip()) >= min_chars]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))
    return [candidates[i] for i in order[:n_sequences]]


def _threshold_stats(transcoder_set: str) -> dict[str, Any]:
    """Per-layer threshold uniqueness / range from the HF snapshot (CPU)."""
    path = Path(snapshot_download(transcoder_set, local_files_only=True))
    per_layer: list[dict[str, Any]] = []
    n_layers = len(list(path.glob("W_enc_*.safetensors")))
    for layer_id in range(n_layers):
        enc = path / f"W_enc_{layer_id}.safetensors"
        with safe_open(enc, framework="pt") as handle:
            keys = list(handle.keys())
            thr_keys = [k for k in keys if "thresh" in k.lower()]
            if not thr_keys:
                per_layer.append({"layer": layer_id, "has_threshold": False})
                continue
            thr = handle.get_tensor(thr_keys[0]).float()
            per_layer.append(
                {
                    "layer": layer_id,
                    "has_threshold": True,
                    "key": thr_keys[0],
                    "n_features": int(thr.numel()),
                    "n_unique": int(thr.unique().numel()),
                    "min": float(thr.min()),
                    "median": float(thr.median()),
                    "max": float(thr.max()),
                    "frac_gt_0p01": float((thr > 0.01).float().mean()),
                }
            )
    n_with = sum(1 for row in per_layer if row.get("has_threshold"))
    return {
        "snapshot": str(path),
        "n_layers": n_layers,
        "n_layers_with_threshold": n_with,
        "activation": "jump_relu" if n_with == n_layers else (
            "relu_or_mixed" if n_with else "relu_no_threshold"
        ),
        "per_layer": per_layer,
    }


def _local_snapshot(repo_id: str) -> str:
    """Resolve a hub id to its on-disk snapshot (no download if cached)."""
    return snapshot_download(repo_id, local_files_only=True)


def _measure_one(
    model_name: str,
    transcoder_set: str,
    texts: list[str],
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    skip_bos: bool,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM

    # Keep the hub *name* for TransformerLens config mapping, but feed weights
    # from the local snapshot so nothing needs network under a warm cache.
    model_path = _local_snapshot(model_name)
    transcoder_path = _local_snapshot(transcoder_set)
    cfg_path = Path(transcoder_path) / "config.yaml"
    hooks = {"feature_input_hook": "hook_resid_mid", "feature_output_hook": "hook_mlp_out"}
    if cfg_path.is_file():
        with open(cfg_path) as handle:
            loaded = yaml.safe_load(handle) or {}
        hooks["feature_input_hook"] = loaded.get(
            "feature_input_hook", hooks["feature_input_hook"]
        )
        hooks["feature_output_hook"] = loaded.get(
            "feature_output_hook", hooks["feature_output_hook"]
        )
    print(
        f"[ref_clt_l0] local model={model_path}\n"
        f"[ref_clt_l0] local transcoder={transcoder_path}",
        flush=True,
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        local_files_only=True,
    )
    clt = load_clt(
        transcoder_path,
        device=device,
        dtype=dtype,
        lazy_encoder=False,
        lazy_decoder=True,
        feature_input_hook=hooks["feature_input_hook"],
        feature_output_hook=hooks["feature_output_hook"],
    )
    model = ReplacementModel.from_pretrained_and_transcoders(
        model_name,
        clt,
        device=device,
        dtype=dtype,
        hf_model=hf_model,
        local_files_only=True,
    )
    n_layers = int(model.transcoders.n_layers)
    d_t = int(model.transcoders.d_transcoder)

    # Accumulators: per-layer sum of active counts and position counts.
    active_sum = np.zeros(n_layers, dtype=np.float64)
    active_sum_no0 = np.zeros(n_layers, dtype=np.float64)
    n_pos = 0
    n_pos_no0 = 0
    seq_lens: list[int] = []

    for text in texts:
        tokens = model.ensure_tokenized(text)
        if tokens.numel() > max_seq_len:
            tokens = tokens[:max_seq_len]
        tokens = tokens.to(device)
        seq_lens.append(int(tokens.numel()))

        with torch.inference_mode():
            _, acts = model.get_activations(tokens, apply_activation_function=True)
        # acts: (n_layers, n_pos, d_transcoder)
        active = (acts > 0).float().sum(dim=-1)  # (n_layers, n_pos)
        active_cpu = active.float().cpu().numpy()
        active_sum += active_cpu.sum(axis=1)
        n_pos += active_cpu.shape[1]
        if active_cpu.shape[1] > 1:
            active_sum_no0 += active_cpu[:, 1:].sum(axis=1)
            n_pos_no0 += active_cpu.shape[1] - 1

        del acts, active
        if device.type == "cuda":
            torch.cuda.empty_cache()

    l0_per_layer = (active_sum / max(n_pos, 1)).tolist()
    l0_per_layer_no0 = (active_sum_no0 / max(n_pos_no0, 1)).tolist()
    l0_layer_token = float(np.mean(l0_per_layer))
    l0_total_token = float(np.sum(l0_per_layer))
    density = l0_layer_token / max(d_t, 1)

    result = {
        "model": model_name,
        "transcoder_set": transcoder_set,
        "n_layers": n_layers,
        "d_transcoder": d_t,
        "d_model": int(model.cfg.d_model),
        "n_sequences": len(texts),
        "n_positions_total": n_pos,
        "mean_seq_len": float(np.mean(seq_lens)) if seq_lens else 0.0,
        "skip_bos_reported_separately": True,
        "l0_per_layer_per_token": l0_layer_token,
        "l0_total_per_token": l0_total_token,
        "activation_density_per_layer": density,
        "l0_per_layer": l0_per_layer,
        "l0_per_layer_per_token_excl_pos0": float(np.mean(l0_per_layer_no0)),
        "l0_total_per_token_excl_pos0": float(np.sum(l0_per_layer_no0)),
        "l0_per_layer_excl_pos0": l0_per_layer_no0,
        "convention": (
            "l0_per_layer_per_token = mean over (layer, pos) of #{a>0}; "
            "matches check_reconstruction.l0_active_per_pos. "
            "l0_total_per_token = sum over layers (autointerp convention)."
        ),
    }
    if skip_bos:
        result["primary"] = "excl_pos0"
    else:
        result["primary"] = "incl_pos0"

    # Free GPU before next model.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _markdown(rows: list[dict[str, Any]], our_linear: dict[str, float] | None) -> str:
    lines = [
        "# Published linear CLT L0 anchors",
        "",
        "L0/layer/token = mean over (layer, position) of active features "
        "(same convention as `check_reconstruction`). "
        "L0/token (total) = sum over layers.",
        "",
        "| Source | Base | d_t | Layers | Act | L0/layer/tok | L0/tok total | Density/layer |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        thr = row.get("threshold_stats", {})
        act = thr.get("activation", "?")
        lines.append(
            f"| {row['label']} | {row['model']} | {row['d_transcoder']} | "
            f"{row['n_layers']} | {act} | "
            f"{fmt(row['l0_per_layer_per_token'], 3)} | "
            f"{fmt(row['l0_total_per_token'], 3)} | "
            f"{fmt(row['activation_density_per_layer'], 4)} |"
        )
    def _intish(value: float | int | None) -> str:
        if value is None:
            return "NOT FOUND"
        return str(int(value))

    if our_linear:
        lines.append(
            f"| OUR linear FM (v3 s101, val) | gpt2 | "
            f"{_intish(our_linear.get('d_transcoder'))} | "
            f"{_intish(our_linear.get('n_layers'))} | jump_relu (frozen θ) | "
            f"{fmt(our_linear.get('l0_per_layer_per_token'), 3)} | "
            f"{fmt(our_linear.get('l0_total_per_token'), 3)} | "
            f"{fmt(our_linear.get('density'), 4)} |"
        )
        lines.append(
            f"| OUR spline FM (v3 s101, val) | gpt2 | "
            f"{_intish(our_linear.get('spline_d_transcoder'))} | "
            f"{_intish(our_linear.get('n_layers'))} | jump_relu | "
            f"{fmt(our_linear.get('spline_l0'), 3)} | "
            f"{fmt(our_linear.get('spline_l0_total'), 3)} | "
            f"{fmt(our_linear.get('spline_density'), 4)} |"
        )
    lines += [
        "",
        "### Gap vs published",
        "",
    ]
    if our_linear and rows:
        ref = rows[0]
        ours = our_linear["l0_per_layer_per_token"]
        target = ref["l0_per_layer_per_token"]
        lines.append(
            f"Our linear FM is **{ours / max(target, 1e-9):.1f}× denser** than "
            f"{ref['label']} on L0/layer/token "
            f"({fmt(ours, 3)} vs {fmt(target, 3)}). "
            f"Published JumpReLU CLTs land near L0/layer ≈ {fmt(target, 3)}; "
            f"that is the sparsity regime the linear baseline should reach."
        )
    lines += [
        "",
        "### Threshold heterogeneity (CPU snapshot)",
        "",
        "| Source | Layers w/ θ | n_unique range | median θ range |",
        "|---|---:|---|---|",
    ]
    for row in rows:
        thr = row.get("threshold_stats", {})
        layers = [x for x in thr.get("per_layer", []) if x.get("has_threshold")]
        if not layers:
            lines.append(f"| {row['label']} | 0 / {thr.get('n_layers', '?')} | n/a (ReLU) | n/a |")
            continue
        nun = [x["n_unique"] for x in layers]
        med = [x["median"] for x in layers]
        lines.append(
            f"| {row['label']} | {len(layers)} / {thr['n_layers']} | "
            f"{min(nun)}–{max(nun)} | {fmt(min(med), 3)}–{fmt(max(med), 3)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _load_our_linear_anchor() -> dict[str, float] | None:
    """Pull banked check_reconstruction L0 for the comparison row."""
    root = Path("results/rebuttal/check_reconstruction")
    linear = root / "check_reconstruction_gpt2s_v3_linear_fm_s101.json"
    spline = root / "check_reconstruction_gpt2s_v3_spline_fm_s101.json"
    if not linear.is_file():
        return None
    lin = json.loads(linear.read_text())["meta"]
    out: dict[str, float] = {
        "l0_per_layer_per_token": float(lin["l0_active_per_pos"]),
        "l0_total_per_token": float(lin["l0_active_per_pos"]) * float(lin["n_layers"]),
        "d_transcoder": float(lin["d_transcoder"]),
        "n_layers": float(lin["n_layers"]),
        "density": float(lin["l0_active_per_pos"]) / float(lin["d_transcoder"]),
    }
    if spline.is_file():
        sp = json.loads(spline.read_text())["meta"]
        out["spline_l0"] = float(sp["l0_active_per_pos"])
        out["spline_l0_total"] = float(sp["l0_active_per_pos"]) * float(sp["n_layers"])
        out["spline_d_transcoder"] = float(sp["d_transcoder"])
        out["spline_density"] = float(sp["l0_active_per_pos"]) / float(sp["d_transcoder"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Override: single HF model id")
    parser.add_argument("--transcoder-set", default=None, help="Override: single CLT repo")
    parser.add_argument("--label", default=None, help="Override label for single run")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-sequences", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument(
        "--skip-gemma",
        action="store_true",
        help="Only run llama (cheapest / has JumpReLU thresholds)",
    )
    parser.add_argument(
        "--primary-excl-bos",
        action="store_true",
        help="Treat excl-pos0 numbers as the headline (still report both)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    texts = _load_wikitext_texts(args.n_sequences, args.seed)

    if args.model and args.transcoder_set:
        refs = [
            {
                "label": args.label or "custom",
                "model": args.model,
                "transcoder_set": args.transcoder_set,
            }
        ]
    else:
        refs = list(DEFAULT_REFS)
        if args.skip_gemma:
            refs = [r for r in refs if "llama" in r["label"]]

    provenance = Provenance(
        script="rebuttal_eval.measure_ref_clt_l0",
        seed=args.seed,
    )
    rows: list[dict[str, Any]] = []

    for ref in refs:
        print(
            f"[ref_clt_l0] measuring {ref['label']}: {ref['model']} + "
            f"{ref['transcoder_set']} on {len(texts)} seqs",
            flush=True,
        )
        thr = _threshold_stats(ref["transcoder_set"])
        measured = _measure_one(
            model_name=ref["model"],
            transcoder_set=ref["transcoder_set"],
            texts=texts,
            max_seq_len=args.max_seq_len,
            device=device,
            dtype=dtype,
            skip_bos=args.primary_excl_bos,
        )
        row = {"label": ref["label"], **measured, "threshold_stats": thr}
        rows.append(row)
        provenance.record(
            "ref_clt_l0",
            f"{ref['label']}.l0_per_layer_per_token",
            measured["l0_per_layer_per_token"],
            checkpoint=ref["transcoder_set"],
        )
        provenance.record(
            "ref_clt_l0",
            f"{ref['label']}.l0_total_per_token",
            measured["l0_total_per_token"],
            checkpoint=ref["transcoder_set"],
        )
        # Per-ref JSON for partial resume / inspection.
        emit(
            args.out_dir,
            f"ref_clt_l0_{ref['label']}",
            row,
            _markdown([row], None),
        )

    our = _load_our_linear_anchor()
    payload = {
        "meta": {
            "n_sequences": args.n_sequences,
            "max_seq_len": args.max_seq_len,
            "seed": args.seed,
            "dtype": args.dtype,
            "device": str(device),
            "corpus": "wikitext-2-raw-v1 test (long lines)",
            "git_sha": git_sha(),
        },
        "refs": rows,
        "our_linear_fm_v3_s101": our,
    }
    md = _markdown(rows, our)
    emit(args.out_dir, "ref_clt_l0", payload, md)
    provenance.write(args.out_dir)
    print(md)
    print(f"[ref_clt_l0] wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
