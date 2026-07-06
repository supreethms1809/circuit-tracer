#!/usr/bin/env python
"""Run MACAG on InterpBench IOI and validate against the known circuit (B4.1b).

InterpBench (mib-bench/interpbench) is a 6-layer/4-head model whose IOI
ground-truth circuit is known exactly (nodes ``m0, a1.h1, a2.h1, a4.h1``). This
runner ablates **native components** (heads/MLPs) through
``macag.scoring_components.HookedComponentInterventionScorer`` — the same
four-mode oracle contract the CLT games use — and reports:

- node-level **AUROC / average precision** of MC-Shapley per-component credit
  against the ground-truth ``in_graph`` flags (upstream MIB's AUROC is
  edge-level only), and
- set-level **precision/recall/F1** of Game 1's selected evidence vs the gold
  node set (both sides are components here, so recall is well-defined —
  unlike the CLT-feature case in ``analyze_gold_circuits.py``).

Outputs ``<out-dir>/interpbench_macag.csv`` + an aggregate with bootstrap CIs.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

from macag.baselines.shapley_select import estimate_shapley
from macag.eval.gold_circuits import average_precision, binary_auroc
from macag.games.game1_min_faithful import solve_game1
from macag.graph import CircuitGraph
from macag.scoring import ScoringOracle
from macag.scoring_components import (
    HookedComponentInterventionScorer,
    component_universe,
    load_gold_component_nodes,
)
from spline_clt.paper.reporting import bootstrap_mean_ci


def load_interpbench_model(device: str = "cuda") -> Any:
    """Load the InterpBench IOI model from the mib-bench hub checkpoint.

    Adapted from external/MIB-circuit-track/run_attribution.py::load_interpbench_model
    (MIB circuit track), parametrized over device instead of hardcoding cuda.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    hf_cfg = hf_hub_download("mib-bench/interpbench", filename="ll_model_cfg.pkl")
    hf_model = hf_hub_download(
        "mib-bench/interpbench", subfolder="ioi_all_splits", filename="ll_model_100_100_80.pth"
    )
    cfg_dict = pickle.load(open(hf_cfg, "rb"))
    if isinstance(cfg_dict, dict):
        cfg = HookedTransformerConfig.from_dict(cfg_dict)
    else:
        assert isinstance(cfg_dict, HookedTransformerConfig)
        cfg = cfg_dict
    cfg.device = device
    # Evaluation-mode hook config (training used a different one).
    cfg.use_hook_mlp_in = True
    cfg.use_attn_result = True
    cfg.use_split_qkv_input = True

    model = HookedTransformer(cfg)
    model.load_state_dict(torch.load(hf_model, map_location=device))
    return model


def load_gold_nodes() -> set[str]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("mib-bench/interpbench", filename="interpbench_graph.json")
    return load_gold_component_nodes(json.loads(Path(path).read_text()))


def iter_ioi_prompts(split: str, limit: int) -> list[dict[str, str]]:
    from datasets import load_dataset

    rows = load_dataset("mib-bench/ioi", split=split)
    out: list[dict[str, str]] = []
    for i, row in enumerate(rows):
        if limit and len(out) >= limit:
            break
        meta = row.get("metadata") or {}
        io_name, subject = meta.get("indirect_object"), meta.get("subject")
        if not io_name or not subject or io_name == subject:
            continue
        out.append(
            {
                "id": f"interpbench_ioi_{i:04d}",
                "prompt": row["prompt"],
                "target": f" {io_name}",
                "foil": f" {subject}",
            }
        )
    return out


def _single_token_id(tokenizer: Any, text: str) -> int | None:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return int(ids[0]) if len(ids) == 1 else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=10, help="number of IOI prompts (0 = all)")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--budget", type=int, default=4, help="Game 1 budget (|gold| = 4)")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--shapley-permutations", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap-samples", type=int, default=10_000)
    ap.add_argument("--out-dir", type=Path, default=Path("results/interpbench_macag"))
    args = ap.parse_args(argv)

    model = load_interpbench_model(device=args.device)
    if model.tokenizer is None:
        raise SystemExit("InterpBench model loaded without a tokenizer; cannot score prompts.")
    gold = load_gold_nodes()
    universe = component_universe(int(model.cfg.n_layers), int(model.cfg.n_heads))
    graph = CircuitGraph(nodes=list(universe))
    print(f"InterpBench: {len(universe)} components, gold circuit = {sorted(gold)}")

    prompts = iter_ioi_prompts(args.split, args.limit)
    if not prompts:
        raise SystemExit("no usable IOI prompts (need single-token IO/subject names)")

    rows: list[dict[str, Any]] = []
    for item in prompts:
        target_id = _single_token_id(model.tokenizer, item["target"])
        foil_id = _single_token_id(model.tokenizer, item["foil"])
        if target_id is None or foil_id is None:
            print(f"skip {item['id']}: multi-token name")
            continue
        scorer = HookedComponentInterventionScorer(
            model=model,
            prompt=item["prompt"],
            target_to_logit_idx={"y": target_id, "y_foil": foil_id},
            score_kind="logit_gap",
            foil_by_target={"y": "y_foil", "y_foil": "y"},
        )
        oracle = ScoringOracle(backend=scorer, cache_enabled=True)
        baseline_gap = oracle.all("y")

        game1 = solve_game1(
            graph, oracle, "y", candidates=universe, alpha=args.alpha, lam=args.lam,
            budget=args.budget, stop_metric="raw_relative", progress=False,
        )
        evidence = {str(node) for node in game1.evidence}
        overlap = evidence & gold
        precision = len(overlap) / len(evidence) if evidence else 0.0
        recall = len(overlap) / len(gold) if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        shapley = estimate_shapley(
            oracle, "y", universe, alpha=args.alpha,
            permutations=args.shapley_permutations, seed=args.seed, progress=False,
        )
        scores = [shapley.values.get(node, 0.0) for node in universe]
        labels = [node in gold for node in universe]
        auroc = binary_auroc(scores, labels)
        ap_score = average_precision(scores, labels)

        stats = oracle.cache_stats()
        rows.append({
            "slug": item["id"], "target_preferred": baseline_gap > 0,
            "baseline_gap": baseline_gap,
            "evidence_size": len(evidence), "evidence": " ".join(sorted(evidence)),
            "precision": precision, "recall": recall, "f1": f1,
            "shapley_auroc": auroc, "shapley_ap": ap_score,
            "oracle_calls": stats["oracle_calls"],
        })
        print(f"{item['id']}: gap={baseline_gap:+.2f} E*={sorted(evidence)} "
              f"P={precision:.2f} R={recall:.2f} AUROC={auroc:.3f}")

    if not rows:
        raise SystemExit("no prompts scored")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "interpbench_macag.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n===== InterpBench aggregate (n={len(rows)}) =====")
    for key in ("precision", "recall", "f1", "shapley_auroc", "shapley_ap"):
        values = [float(row[key]) for row in rows]
        lo, hi = bootstrap_mean_ci(values, args.bootstrap_samples, 0.95, args.seed)
        print(f"  {key:14} {sum(values) / len(values):.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
