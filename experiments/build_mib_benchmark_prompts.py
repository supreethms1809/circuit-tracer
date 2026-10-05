#!/usr/bin/env python
"""Export MIB-bench HuggingFace prompts into MACAG benchmark JSON format.

Produces ``macag/data/mib_benchmark_prompts.json`` with the same per-prompt
schema as ``acdc_benchmark_prompts.json``, plus ``mib_model`` / ``mib_split`` /
``mib_task`` metadata so ``scripts/run_macag_mib.sh`` can route each prompt to
the matching hub CLT (gemma2 prompts -> gemma2 CLTs, llama3 prompts -> the
llama32-524k CLT only).

Example:
  conda run -n ct python experiments/build_mib_benchmark_prompts.py \\
    --models gemma2 --tasks ioi mcqa --split validation --limit-per-task 10

  conda run -n ct python experiments/build_mib_benchmark_prompts.py \\
    --models llama3 --tasks ioi mcqa arithmetic_addition arithmetic_subtraction \\
        arc_easy arc_challenge --split validation --limit-per-task 10
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MIB_DIR = REPO_ROOT / "external" / "MIB-circuit-track"
DEFAULT_OUT = REPO_ROOT / "macag" / "data" / "mib_benchmark_prompts.json"

# MIB leaderboard cells with a matching public hub CLT in run_macag_mib.sh.
# NOTE: the MIB leaderboard's "llama3" cells were collected against
# meta-llama/Llama-3.1-8B, but the only public CLT we have is
# mntss/clt-llama-3.2-1b-524k (meta-llama/Llama-3.2-1B). We tokenize/route
# against the 3.2-1B model here since that's the model MACAG can actually
# score; this reuses the MIB prompt text/labels but is not a reproduction of
# the official Llama leaderboard cell (different model size).
MIB_MODEL_TO_HF = {
    "gemma2": "google/gemma-2-2b",
    "llama3": "meta-llama/Llama-3.2-1B",
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

# llama3 MIB cells (from MIB COL_MAPPING).
LLAMA3_TASKS = ("ioi", "mcqa", "arithmetic_addition", "arithmetic_subtraction", "arc_easy", "arc_challenge")

MODEL_TASKS = {
    "gemma2": GEMMA2_TASKS,
    "llama3": LLAMA3_TASKS,
}


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([token_id])


def _label_token(tokenizer: Any, label: str) -> str:
    """Space-prefixed single-token form of a choice label (A4).

    Mirrors Track A's ``_prefer_leading_space_token`` (macag/factories/
    replacement_model.py) so the builder emits exactly what the scoring path
    resolves: prefer ``' X'`` when it is a single token, keep the bare label
    otherwise (e.g. digits, where ``' 1'`` is typically two tokens). Both the
    stored target and foil go through this, so the JSON is self-consistent.
    """
    bare_ids = tokenizer(str(label), add_special_tokens=False).input_ids
    spaced_ids = tokenizer(f" {label}", add_special_tokens=False).input_ids
    if len(spaced_ids) == 1 and (len(bare_ids) != 1 or spaced_ids[0] != bare_ids[0]):
        return _decode_token(tokenizer, spaced_ids[0])
    if not bare_ids:
        raise ValueError(f"label {label!r} tokenized to an empty sequence")
    return _decode_token(tokenizer, bare_ids[0])


def _ioi_tokens(tokenizer: Any, row: dict[str, Any]) -> tuple[str, str]:
    meta = row["metadata"]
    correct = tokenizer(f" {meta['indirect_object']}", add_special_tokens=False).input_ids[0]
    incorrect = tokenizer(f" {meta['subject']}", add_special_tokens=False).input_ids[0]
    return _decode_token(tokenizer, correct), _decode_token(tokenizer, incorrect)


def _normalize_answer_index(answer_key: Any, n_choices: int) -> int:
    """Normalize MIB answerKey (int or digit-string) to a choice index."""
    try:
        idx = int(answer_key)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unrecognized answerKey {answer_key!r}") from exc
    if not 0 <= idx < n_choices:
        raise ValueError(f"answerKey {idx} out of range for {n_choices} choices")
    return idx


def _mcqa_tokens(tokenizer: Any, row: dict[str, Any]) -> tuple[str, str]:
    """Foil = a WRONG option from the CLEAN prompt (logit-gap convention).

    MIB's own dataloader uses the symbol-counterfactual column's correct answer
    as the "incorrect" index for clean-vs-corrupted patching. That token (e.g.
    '4' on an 'A-D' prompt) is not an option in the clean prompt, so scoring a
    clean-prompt logit gap against it is invalid (A0). Here both tokens are
    clean-prompt options: the correct label and the next wrong label
    (round-robin, so foils stay balanced across the benchmark instead of
    collapsing onto 'A').
    """
    labels = row["choices"]["label"]
    correct_idx = _normalize_answer_index(row["answerKey"], len(labels))
    foil_idx = (correct_idx + 1) % len(labels)
    correct = _label_token(tokenizer, str(labels[correct_idx]))
    incorrect = _label_token(tokenizer, str(labels[foil_idx]))
    return correct, incorrect


def _arithmetic_tokens(tokenizer: Any, row: dict[str, Any]) -> tuple[str, str]:
    correct = _label_token(tokenizer, str(row["label"]))
    incorrect = _label_token(tokenizer, str(row["random_counterfactual"]["label"]))
    return correct, incorrect


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
    sample: str = "first",
    seed: int | None = None,
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
            if task not in MODEL_TASKS.get(model, MIB_TASK_TO_HF.keys()):
                print(f"skip {model}/{task}: no {model} MIB leaderboard cell")
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
            # A9: first-N (legacy default) or a seeded random subset. Random
            # sampling requires a seed and a limit; sampled dataset indices go
            # into slugs/metadata so subsets are stable and auditable. Whether
            # the paper rematches on a random subset is D6 — this only adds
            # the option.
            if sample == "random":
                if seed is None:
                    raise ValueError("--sample random requires --seed")
                if limit is None:
                    raise ValueError("--sample random requires --limit-per-task or --task-limit")
                idxs = sorted(random.Random(seed).sample(range(len(ds)), min(limit, len(ds))))
            elif sample == "first":
                idxs = list(range(min(n, limit) if limit is not None else n))
            else:
                raise ValueError(f"Unknown --sample {sample!r}; choose first|random")
            bucket = f"{task}"
            out_tasks.setdefault(bucket, [])
            for i in idxs:
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
                    correct_tok, incorrect_tok = _mcqa_tokens(tokenizer, row)
                    if correct_tok == incorrect_tok:
                        raise ValueError(f"foil equals target for {task} row {i}")
                    # A4: spacing-agnostic — stored tokens are space-prefixed
                    # (' A') while prompts list options as '\nA. ...'; compare
                    # the stripped label (same strength as the legacy check).
                    foil_label = incorrect_tok.strip() or incorrect_tok
                    if foil_label not in clean and f" {foil_label}" not in clean:
                        raise ValueError(
                            f"foil {incorrect_tok!r} not an option in the clean prompt "
                            f"({task} row {i}; clean starts {clean[:80]!r})"
                        )
                elif task.startswith("arithmetic"):
                    clean = row["prompt"]
                    corrupted = row["random_counterfactual"]["prompt"]
                    correct_tok, incorrect_tok = _arithmetic_tokens(tokenizer, row)
                else:
                    raise ValueError(task)

                slug = f"mib_{model}_{task}_{i:04d}"
                entry_metadata: dict[str, Any] = {
                    "source": "mib-bench",
                    "hf_dataset": hf_url,
                    "index": i,
                    "split_size": n,
                }
                if sample == "random":
                    entry_metadata["sample_seed"] = seed
                    entry_metadata["sample_method"] = "random"
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
                        "metadata": entry_metadata,
                    }
                )
            print(f"exported {len(idxs)} prompts for {model}/{task} ({split})")
    return {
        "benchmarks_info": {
            "description": (
                "Prompts from the MIB circuit-localization benchmark "
                "(mib-bench/* on HuggingFace), exported for MACAG Game 1/2 evaluation. "
                "Each prompt is tokenizer-aligned to its mib_model; run only on matching CLTs."
            ),
            # A4: the stored token convention, matching Track A's resolution
            # (macag/factories/replacement_model.py::_prefer_leading_space_token).
            "token_convention": (
                "correct_token/incorrect_token are single-token strings in "
                "space-prefixed form (e.g. ' D'). Bare choice labels are stored "
                "spaced, except digits whose spaced form is multi-token (kept bare)."
            ),
            "sampling": {"method": sample, "seed": seed},
            "usage": (
                "Same schema as acdc_benchmark_prompts.json. "
                "Use clean_prompt for attribution; score logit gap between "
                "correct_token and incorrect_token. Route via mib_model in run_macag_mib.sh."
            ),
            "mib_model_to_clt_tags": {
                "gemma2": ["gemma2-426k", "gemma2-2.5M"],
                "llama3": ["llama32-524k"],
            },
        },
        "tasks": out_tasks,
    }


def _next_option_foil(correct_token: str, prompt: str) -> str:
    """Next clean-prompt option after the correct label, space-prefixed.

    Matches ``_mcqa_tokens`` on letter options (A–D). Checked against the
    tokenizer-built campaign file: 200/200 foils agree.
    """
    import re

    labels = re.findall(r"(?m)^([A-Z])\.", prompt)
    letter = correct_token.strip()
    if letter not in labels:
        raise ValueError(
            f"correct token {correct_token!r} is not an option label in the prompt"
        )
    nxt = labels[(labels.index(letter) + 1) % len(labels)]
    return f" {nxt}"


def resample_from_manifest(
    manifest_path: Path,
    *,
    seed: int,
    task_limits: dict[str, int],
    models: list[str],
    tasks: list[str],
) -> dict[str, Any]:
    """Seeded draw from the exported full manifest, with clean-prompt foils.

    The full v1 export stored MCQA/ARC foils as counterfactual digits. Those
    are replaced with the next wrong option. IOI name tokens are already the
    clean-prompt subject and indirect object. The same seed is reconstructed
    per (model, task), matching ``export_prompts``.
    """
    manifest = json.loads(manifest_path.read_text())
    out_tasks: dict[str, list[dict[str, Any]]] = {}
    split_sizes: dict[str, int] = {}
    for model in models:
        for task in tasks:
            rows = [
                row for row in manifest["tasks"].get(task, [])
                if row.get("mib_model") == model
            ]
            if not rows:
                print(f"skip {model}/{task}: not in {manifest_path.name}")
                continue
            limit = task_limits.get(task)
            if limit is None:
                raise ValueError(f"--from-manifest needs --task-limit for {task}")
            split_sizes[f"{model}/{task}"] = len(rows)
            idxs = sorted(random.Random(seed).sample(range(len(rows)), min(limit, len(rows))))
            bucket = out_tasks.setdefault(task, [])
            for i in idxs:
                row = dict(rows[i])
                meta = dict(row.get("metadata") or {})
                meta["split_size"] = len(rows)
                meta["sample_seed"] = seed
                meta["sample_method"] = "random"
                meta["source_manifest"] = manifest_path.name
                if task in ("mcqa", "arc_easy", "arc_challenge"):
                    row["incorrect_token"] = _next_option_foil(
                        row["correct_token"], row["clean_prompt"]
                    )
                    meta["foil_rule"] = "next_wrong_option"
                row["metadata"] = meta
                bucket.append(row)
            print(f"sampled {len(idxs)}/{len(rows)} for {model}/{task} seed={seed}")
    return {
        "benchmarks_info": {
            "description": (
                "Seeded random MIB validation subset for the final MACAG paper. "
                "25 IOI + 25 MCQA + 25 ARC-Easy per mib_model."
            ),
            "token_convention": (
                "correct_token/incorrect_token are single-token strings in "
                "space-prefixed form (e.g. ' D'). MCQA/ARC foils are the next "
                "wrong option in the clean prompt, not the MIB counterfactual digit."
            ),
            "sampling": {
                "method": "random",
                "seed": seed,
                "task_limits": task_limits,
                "source_manifest": manifest_path.name,
                "split_sizes": split_sizes,
                "note": (
                    "MCQA validation in the manifest has 50 rows, so 25 is half "
                    "that split. IOI is 500 and ARC-Easy is 570 in the same export."
                ),
            },
            "usage": (
                "Same schema as acdc_benchmark_prompts.json. "
                "Route via mib_model in run_macag_mib.sh."
            ),
            "mib_model_to_clt_tags": {
                "gemma2": ["gemma2-426k", "gemma2-2.5M"],
                "llama3": ["llama32-524k"],
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
    ap.add_argument("--sample", choices=("first", "random"), default="first",
                    help="row selection within each split: first-N (legacy) or seeded "
                         "random subset (A9; requires --seed and a limit)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for --sample random")
    ap.add_argument(
        "--from-manifest",
        type=Path,
        default=None,
        help="Resample an already-exported full manifest instead of HuggingFace. "
        "Rewrites MCQA/ARC foils to the next clean-prompt option.",
    )
    ap.add_argument(
        "--mark-subsets",
        action="store_true",
        help="Tag Phase-S slices from a second shuffle (seed+1). "
        "P3 is 20 IOI+20 MCQA; S1 is 20 cells round-robin; S2 20 IOI; "
        "S3/S4/S6 10 IOI; S7 2 IOI; S8 10 MCQA.",
    )
    args = ap.parse_args()

    if args.from_manifest is not None:
        if args.seed is None:
            raise SystemExit("--from-manifest requires --seed")
        payload = resample_from_manifest(
            args.from_manifest,
            seed=args.seed,
            task_limits=parse_task_limits(args.task_limit),
            models=args.models,
            tasks=args.tasks,
        )
    elif not MIB_DIR.is_dir():
        raise SystemExit(f"MIB repo missing at {MIB_DIR}; run external/setup_mib.sh")
    else:
        payload = export_prompts(
            models=args.models,
            tasks=args.tasks,
            split=args.split,
            limit_per_task=args.limit_per_task,
            task_limits=parse_task_limits(args.task_limit),
            counterfactual_type=args.counterfactual_type,
            sample=args.sample,
            seed=args.seed,
        )
    if args.mark_subsets:
        if args.seed is None:
            raise SystemExit("--mark-subsets requires --seed")
        from macag.utils.final_protocol import SUBSET_PREFIXES, assign_subset_flags

        flags = assign_subset_flags(payload["tasks"], args.seed)
        for entries in payload["tasks"].values():
            for entry in entries:
                entry["subsets"] = flags.get(entry["id"], [])
        payload["benchmarks_info"]["subsets"] = {
            "shuffle_seed": args.seed + 1,
            "prefixes": SUBSET_PREFIXES,
            "s1_cells": 20,
            "note": (
                "Flags are prefixes of one seeded shuffle per mib_model and task, "
                "not a cost ranking. P3 is 20 IOI + 20 MCQA within each model's 25. "
                "S1 is 20 prompts round-robin across model and task. "
                "MCQA validation has 50 rows, so 25 is half that split."
            ),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
        f.write("\n")
    n_total = sum(len(v) for v in payload["tasks"].values())
    print(f"wrote {n_total} prompts -> {args.out}")


if __name__ == "__main__":
    main()
