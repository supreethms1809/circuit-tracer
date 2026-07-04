#!/usr/bin/env python
"""Export MIB-bench HuggingFace prompts into MACAG benchmark JSON format.

Produces ``macag/data/mib_benchmark_prompts.json`` with the same per-prompt
schema as ``acdc_benchmark_prompts.json``, plus ``mib_model`` / ``mib_split`` /
``mib_task`` metadata so ``scripts/run_macag_mib.sh`` can route each prompt to
the matching hub CLT (gemma2 prompts -> gemma2 CLTs only).

Example:
  conda run -n ct python experiments/build_mib_benchmark_prompts.py \\
    --models gemma2 --tasks ioi mcqa --split validation --limit-per-task 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MIB_DIR = REPO_ROOT / "external" / "MIB-circuit-track"
DEFAULT_OUT = REPO_ROOT / "macag" / "data" / "mib_benchmark_prompts.json"

# MIB leaderboard cells with a matching public hub CLT in run_macag_mib.sh.
MIB_MODEL_TO_HF = {
    "gemma2": "google/gemma-2-2b",
}

MIB_TASK_TO_HF = {
    "ioi": "ioi",
    "mcqa": "copycolors_mcqa",
    "arithmetic_addition": "arithmetic_addition",
    "arithmetic_subtraction": "arithmetic_subtraction",
    "arc_easy": "arc_easy",
    "arc_challenge": "arc_challenge",
}

# gemma2 MIB cells (from MIB COL_MAPPING).
GEMMA2_TASKS = ("ioi", "mcqa", "arc_easy")


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([token_id])


def _ioi_tokens(tokenizer: Any, row: dict[str, Any]) -> tuple[str, str]:
    meta = row["metadata"]
    correct = tokenizer(f" {meta['indirect_object']}", add_special_tokens=False).input_ids[0]
    incorrect = tokenizer(f" {meta['subject']}", add_special_tokens=False).input_ids[0]
    return _decode_token(tokenizer, correct), _decode_token(tokenizer, incorrect)


def _mcqa_tokens(tokenizer: Any, row: dict[str, Any], counterfactual_col: dict[str, Any]) -> tuple[str, str]:
    correct = tokenizer(row["choices"]["label"][row["answerKey"]], add_special_tokens=False).input_ids[0]
    incorrect_ans = str(counterfactual_col["choices"]["label"][counterfactual_col["answerKey"]])
    incorrect = tokenizer(incorrect_ans, add_special_tokens=False).input_ids[0]
    return _decode_token(tokenizer, correct), _decode_token(tokenizer, incorrect)


def _arithmetic_tokens(tokenizer: Any, row: dict[str, Any]) -> tuple[str, str]:
    correct = tokenizer(str(row["label"]), add_special_tokens=False).input_ids[0]
    incorrect = tokenizer(str(row["random_counterfactual"]["label"]), add_special_tokens=False).input_ids[0]
    return _decode_token(tokenizer, correct), _decode_token(tokenizer, incorrect)


def parse_task_limits(pairs: list[str]) -> dict[str, int]:
    """Parse repeated ``TASK=N`` overrides (e.g. ``--task-limit ioi=500``)."""
    limits: dict[str, int] = {}
    for pair in pairs:
        task, sep, value = pair.partition("=")
        if not sep or not task or not value.isdigit():
            raise ValueError(f"--task-limit expects TASK=N, got {pair!r}")
        limits[task] = int(value)
    return limits


def export_prompts(
    *,
    models: list[str],
    tasks: list[str],
    split: str,
    limit_per_task: int | None,
    task_limits: dict[str, int] | None,
    counterfactual_type: str | None,
) -> dict[str, Any]:
    sys.path.insert(0, str(MIB_DIR))
    from MIB_circuit_track.dataset import HFEAPDataset  # noqa: WPS433
    from transformers import AutoTokenizer

    out_tasks: dict[str, list[dict[str, Any]]] = {}
    for model in models:
        if model not in MIB_MODEL_TO_HF:
            raise ValueError(f"Unsupported mib_model={model!r}; known: {sorted(MIB_MODEL_TO_HF)}")
        tokenizer = AutoTokenizer.from_pretrained(MIB_MODEL_TO_HF[model])
        for task in tasks:
            if model == "gemma2" and task not in GEMMA2_TASKS:
                print(f"skip {model}/{task}: no gemma2 MIB leaderboard cell")
                continue
            if task not in MIB_TASK_TO_HF:
                raise ValueError(f"Unknown task {task!r}")
            hf_url = f"mib-bench/{MIB_TASK_TO_HF[task]}"
            ds = HFEAPDataset(
                hf_url,
                tokenizer,
                split=split,
                task=task,
                model_name=model,
                counterfactual_type=counterfactual_type,
            )
            n = len(ds)
            limit = (task_limits or {}).get(task, limit_per_task)
            if limit is not None:
                n = min(n, limit)
            bucket = f"{task}"
            out_tasks.setdefault(bucket, [])
            for i in range(n):
                row = ds.dataset[i]
                if task == "ioi":
                    cf_col = counterfactual_type or "s2_io_flip_counterfactual"
                    clean = row["prompt"]
                    corrupted = row[cf_col]["prompt"]
                    correct_tok, incorrect_tok = _ioi_tokens(tokenizer, row)
                elif task in ("mcqa", "arc_easy", "arc_challenge"):
                    cf_type = counterfactual_type or "symbol_counterfactual"
                    cf_col = row[cf_type]
                    clean = row["prompt"]
                    corrupted = cf_col["prompt"]
                    correct_tok, incorrect_tok = _mcqa_tokens(tokenizer, row, cf_col)
                elif task.startswith("arithmetic"):
                    clean = row["prompt"]
                    corrupted = row["random_counterfactual"]["prompt"]
                    correct_tok, incorrect_tok = _arithmetic_tokens(tokenizer, row)
                else:
                    raise ValueError(task)

                slug = f"mib_{model}_{task}_{i:04d}"
                out_tasks[bucket].append(
                    {
                        "id": slug,
                        "mib_model": model,
                        "mib_task": task,
                        "mib_split": split,
                        "clean_prompt": clean,
                        "corrupted_prompt": corrupted,
                        "correct_token": correct_tok,
                        "incorrect_token": incorrect_tok,
                        "metadata": {
                            "source": "mib-bench",
                            "hf_dataset": hf_url,
                            "index": i,
                        },
                    }
                )
            print(f"exported {n} prompts for {model}/{task} ({split})")
    return {
        "benchmarks_info": {
            "description": (
                "Prompts from the MIB circuit-localization benchmark "
                "(mib-bench/* on HuggingFace), exported for MACAG Game 1/2 evaluation. "
                "Each prompt is tokenizer-aligned to its mib_model; run only on matching CLTs."
            ),
            "usage": (
                "Same schema as acdc_benchmark_prompts.json. "
                "Use clean_prompt for attribution; score logit gap between "
                "correct_token and incorrect_token. Route via mib_model in run_macag_mib.sh."
            ),
            "mib_model_to_clt_tags": {
                "gemma2": ["gemma2-426k", "gemma2-2.5M"],
            },
        },
        "tasks": out_tasks,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--models", nargs="+", default=["gemma2"])
    ap.add_argument("--tasks", nargs="+", default=list(GEMMA2_TASKS))
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--limit-per-task", type=int, default=None,
                    help="cap examples per (model, task); default = full filtered split")
    ap.add_argument("--task-limit", action="append", default=[], metavar="TASK=N",
                    help="per-task cap overriding --limit-per-task, e.g. --task-limit ioi=500 "
                         "(repeatable; tasks not listed fall back to --limit-per-task / full split)")
    ap.add_argument("--counterfactual-type", default=None,
                    help="IOI/MCQA counterfactual column (MIB default per task if omitted)")
    args = ap.parse_args()

    if not MIB_DIR.is_dir():
        raise SystemExit(f"MIB repo missing at {MIB_DIR}; run external/setup_mib.sh")

    payload = export_prompts(
        models=args.models,
        tasks=args.tasks,
        split=args.split,
        limit_per_task=args.limit_per_task,
        task_limits=parse_task_limits(args.task_limit),
        counterfactual_type=args.counterfactual_type,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
        f.write("\n")
    n_total = sum(len(v) for v in payload["tasks"].values())
    print(f"wrote {n_total} prompts -> {args.out}")


if __name__ == "__main__":
    main()
