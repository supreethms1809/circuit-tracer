#!/usr/bin/env python
"""Run MACAG on InterpBench IOI and validate against the known circuit (B4.1b).

InterpBench (mib-bench/interpbench) is a 6-layer/4-head model whose IOI
ground-truth circuit is known exactly (nodes ``m0, a1.h1, a2.h1, a4.h1``). This
runner ablates **native components** (heads/MLPs) through
``macag.scoring_components.HookedComponentInterventionScorer``.

Evaluation protocol (gold *presence* in the freely selected faithful set):
  1. Run Game 1 on ``logit_gap`` with **no hard budget** by default (``--budget 0``).
     The set grows while faithfulness/necessity utility still improves, then stops.
  2. ``final_set = E* ∩ gold`` — gold nodes that appear among those selected features.
  3. Leave-one-out importance inside ``E*``: ``I(i) = faith(E*) - faith(E* \\ {i})``.
  4. Optional MC-Shapley over the full component universe for ranking AUROC/AP.
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
from macag.utils.metrics import compute_faithfulness_metrics


def load_interpbench_model(device: str = "cuda") -> Any:
    """Load the InterpBench IOI model from the mib-bench hub checkpoint."""
    import torch
    from huggingface_hub import hf_hub_download
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from transformers import AutoTokenizer

    print("InterpBench: resolving checkpoint files (local cache only)...", flush=True)
    hf_cfg = hf_hub_download(
        "mib-bench/interpbench", filename="ll_model_cfg.pkl", local_files_only=True
    )
    hf_model = hf_hub_download(
        "mib-bench/interpbench",
        subfolder="ioi_all_splits",
        filename="ll_model_100_100_80.pth",
        local_files_only=True,
    )
    cfg_dict = pickle.load(open(hf_cfg, "rb"))
    if isinstance(cfg_dict, dict):
        cfg = HookedTransformerConfig.from_dict(cfg_dict)
    else:
        assert isinstance(cfg_dict, HookedTransformerConfig)
        cfg = cfg_dict
    cfg.device = device
    cfg.use_hook_mlp_in = True
    cfg.use_attn_result = True
    cfg.use_split_qkv_input = True
    if hasattr(cfg, "tokenizer_name"):
        cfg.tokenizer_name = None

    print("InterpBench: loading GPT-2 tokenizer (local cache only)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"InterpBench: building HookedTransformer on {device}...", flush=True)
    model = HookedTransformer(cfg, tokenizer=tokenizer)
    print("InterpBench: loading state dict...", flush=True)
    model.load_state_dict(torch.load(hf_model, map_location=device, weights_only=False))
    print("InterpBench: model ready.", flush=True)
    return model


def load_gold_nodes() -> set[str]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "mib-bench/interpbench", filename="interpbench_graph.json", local_files_only=True
    )
    return load_gold_component_nodes(json.loads(Path(path).read_text()))


def iter_ioi_prompts(split: str, limit: int) -> list[dict[str, str]]:
    from datasets import load_dataset

    print(f"InterpBench: loading mib-bench/ioi ({split}) from local cache...", flush=True)
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
    ids = tokenizer.encode(text, add_special_tokens=False)
    return int(ids[0]) if len(ids) == 1 else None


def leave_one_out_importance(
    oracle: ScoringOracle,
    target: str,
    evidence: set[str],
    alpha: float,
) -> dict[str, float]:
    """Importance of each node to the selected set via leave-one-out faith drop."""
    if not evidence:
        return {}
    full = compute_faithfulness_metrics(oracle, target, set(evidence), alpha).faithfulness_delta
    out: dict[str, float] = {}
    for node in sorted(evidence):
        reduced = set(evidence) - {node}
        reduced_faith = (
            compute_faithfulness_metrics(oracle, target, reduced, alpha).faithfulness_delta
            if reduced
            else 0.0
        )
        out[str(node)] = float(full - reduced_faith)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=10, help="number of IOI prompts (0 = all)")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--budget",
        type=int,
        default=0,
        help="Game 1 budget cap. 0 = uncapped (grow while utility improves).",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Faithfulness mix: alpha*sufficiency + (1-alpha)*necessity.",
    )
    ap.add_argument(
        "--lam",
        type=float,
        default=0.0,
        help="Sparsity penalty on |E|. Keep 0 for free faith/necessity selection.",
    )
    ap.add_argument(
        "--fill-budget",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force |E*|=budget even with non-positive gains (off for free selection).",
    )
    ap.add_argument(
        "--faithfulness-eps",
        type=float,
        default=None,
        help="Optional raw_relative early stop (fraction of first-feature faith gain). "
        "Default None = stop only when no positive utility gain remains.",
    )
    ap.add_argument("--shapley-permutations", type=int, default=64)
    ap.add_argument(
        "--skip-shapley",
        action="store_true",
        help="Skip MC-Shapley ranking (faster Game1-only runs).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap-samples", type=int, default=10_000)
    ap.add_argument("--out-dir", type=Path, default=Path("results/interpbench_macag"))
    ap.add_argument(
        "--score-kinds",
        default="logit_gap",
        help="Selection utilities (default: logit_gap only).",
    )
    args = ap.parse_args(argv)
    score_kinds = [k.strip() for k in args.score_kinds.split(",") if k.strip()]
    budget = None if args.budget <= 0 else args.budget
    if args.fill_budget and budget is None:
        raise SystemExit("--fill-budget requires a positive --budget")

    model = load_interpbench_model(device=args.device)
    if model.tokenizer is None:
        raise SystemExit("InterpBench model loaded without a tokenizer; cannot score prompts.")
    gold = load_gold_nodes()
    universe = component_universe(int(model.cfg.n_layers), int(model.cfg.n_heads))
    graph = CircuitGraph(nodes=list(universe))
    print(f"InterpBench: {len(universe)} components, gold circuit = {sorted(gold)}")
    print(
        f"InterpBench: score_kinds={score_kinds} connected=False prefilter=off "
        f"budget={'uncapped' if budget is None else budget} alpha={args.alpha} "
        f"lam={args.lam} fill_budget={args.fill_budget} "
        f"faithfulness_eps={args.faithfulness_eps} skip_shapley={args.skip_shapley} "
        f"protocol=free_select+gold_in_set+loo_importance"
    )

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

        row: dict[str, Any] = {
            "slug": item["id"],
            "target_preferred": None,
            "baseline_gap": None,
        }
        for score_kind in score_kinds:
            scorer = HookedComponentInterventionScorer(
                model=model,
                prompt=item["prompt"],
                target_to_logit_idx={"y": target_id, "y_foil": foil_id},
                score_kind=score_kind,  # type: ignore[arg-type]
                foil_by_target={"y": "y_foil", "y_foil": "y"},
            )
            oracle = ScoringOracle(backend=scorer, cache_enabled=True)
            baseline = oracle.all("y")
            if score_kind == score_kinds[0]:
                row["target_preferred"] = baseline > 0
                row["baseline_gap"] = baseline

            game1 = solve_game1(
                graph, oracle, "y", candidates=universe, alpha=args.alpha, lam=args.lam,
                budget=budget, stop_metric="raw_relative",
                faithfulness_eps=args.faithfulness_eps,
                connected=False, prefilter_top_k=None, fill_budget=args.fill_budget,
                progress=False,
            )
            evidence = {str(node) for node in game1.evidence}
            final_set = evidence & gold
            missed_gold = gold - evidence

            importance = leave_one_out_importance(oracle, "y", evidence, args.alpha)
            gold_importance = {n: importance[n] for n in sorted(final_set)}
            nongold_importance = {
                n: importance[n] for n in sorted(evidence - gold) if n in importance
            }
            total_imp = sum(importance.values())
            gold_imp_sum = sum(gold_importance.values())
            gold_imp_share = (gold_imp_sum / total_imp) if abs(total_imp) > 1e-12 else 0.0

            gold_recall = len(final_set) / len(gold) if gold else 0.0
            gold_hit = 1.0 if final_set else 0.0
            n_gold = len(final_set)
            precision = len(final_set) / len(evidence) if evidence else 0.0
            f1 = (
                2 * precision * gold_recall / (precision + gold_recall)
                if (precision + gold_recall)
                else 0.0
            )

            shapley_by_node: dict[str, float] = {}
            auroc = None
            ap_score = None
            if not args.skip_shapley:
                shapley = estimate_shapley(
                    oracle, "y", universe, alpha=args.alpha,
                    permutations=args.shapley_permutations, seed=args.seed, progress=False,
                )
                shapley_by_node = {str(n): float(shapley.values.get(n, 0.0)) for n in universe}
                scores = [shapley_by_node[n] for n in universe]
                labels = [n in gold for n in universe]
                auroc = binary_auroc(scores, labels)
                ap_score = average_precision(scores, labels)
            gold_shapley = {n: shapley_by_node.get(n, 0.0) for n in sorted(final_set)}

            stats = oracle.cache_stats()
            prefix = score_kind
            row[f"{prefix}_evidence_size"] = len(evidence)
            row[f"{prefix}_evidence"] = " ".join(sorted(evidence))
            row[f"{prefix}_final_set"] = " ".join(sorted(final_set))
            row[f"{prefix}_missed_gold"] = " ".join(sorted(missed_gold))
            row[f"{prefix}_n_gold_recovered"] = n_gold
            row[f"{prefix}_gold_hit"] = gold_hit
            row[f"{prefix}_gold_recall"] = gold_recall
            row[f"{prefix}_precision"] = precision
            row[f"{prefix}_recall"] = gold_recall
            row[f"{prefix}_f1"] = f1
            row[f"{prefix}_gold_importance_sum"] = gold_imp_sum
            row[f"{prefix}_gold_importance_mean"] = (
                gold_imp_sum / n_gold if n_gold else 0.0
            )
            row[f"{prefix}_gold_importance_share"] = gold_imp_share
            row[f"{prefix}_importance_json"] = json.dumps(importance, sort_keys=True)
            row[f"{prefix}_gold_importance_json"] = json.dumps(gold_importance, sort_keys=True)
            row[f"{prefix}_nongold_importance_json"] = json.dumps(
                nongold_importance, sort_keys=True
            )
            row[f"{prefix}_gold_shapley_json"] = json.dumps(gold_shapley, sort_keys=True)
            row[f"{prefix}_shapley_auroc"] = auroc
            row[f"{prefix}_shapley_ap"] = ap_score
            row[f"{prefix}_oracle_calls"] = stats["oracle_calls"]
            row[f"{prefix}_baseline"] = baseline

            auroc_s = "skipped" if auroc is None else f"{auroc:.3f}"
            print(
                f"{item['id']} [{score_kind}]: gap={baseline:+.2f} "
                f"|E*|={len(evidence)} final={sorted(final_set)} "
                f"hit={int(gold_hit)} R={gold_recall:.2f} "
                f"gold_imp_share={gold_imp_share:.2f} AUROC={auroc_s}",
                flush=True,
            )

        primary = score_kinds[0]
        for key in (
            "evidence_size",
            "evidence",
            "final_set",
            "missed_gold",
            "n_gold_recovered",
            "gold_hit",
            "gold_recall",
            "precision",
            "recall",
            "f1",
            "gold_importance_sum",
            "gold_importance_mean",
            "gold_importance_share",
            "shapley_auroc",
            "shapley_ap",
            "oracle_calls",
        ):
            row[key] = row[f"{primary}_{key}"]
        rows.append(row)

    if not rows:
        raise SystemExit("no prompts scored")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "interpbench_macag.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n===== InterpBench aggregate (n={len(rows)}) =====")
    from spline_clt.paper.reporting import bootstrap_mean_ci

    metric_keys = [
        "gold_hit",
        "gold_recall",
        "n_gold_recovered",
        "gold_importance_mean",
        "gold_importance_share",
        "precision",
        "f1",
        "shapley_auroc",
        "shapley_ap",
        "evidence_size",
    ]
    for score_kind in score_kinds:
        print(f"--- {score_kind} ---")
        for key in metric_keys:
            col = f"{score_kind}_{key}"
            values: list[float] = []
            for row in rows:
                raw = row.get(col)
                if raw is None or raw == "":
                    continue
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    continue
                if val == val:
                    values.append(val)
            if not values:
                print(f"  {key:22} n/a (skipped)")
                continue
            lo, hi = bootstrap_mean_ci(values, args.bootstrap_samples, 0.95, args.seed)
            print(f"  {key:22} {sum(values) / len(values):.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
