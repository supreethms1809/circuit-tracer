#!/usr/bin/env python3
"""S8 alternate MCQA foil: the highest-logit wrong option on the clean prompt.

One forward per prompt. The stored foil stays space-prefixed when that form
is a single token. Writes a sibling JSON and sets metadata.foil_mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformer_lens import HookedTransformer


def _single_token(tokenizer, text: str) -> str | None:
    spaced = text if text.startswith(" ") else f" {text}"
    ids = tokenizer.encode(spaced, add_special_tokens=False)
    if len(ids) == 1:
        return spaced
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 1:
        return text
    return None


def alternate_foil(model: HookedTransformer, prompt: str, correct: str, options: list[str]) -> str:
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)[0, -1].float()
    correct_id = model.to_single_token(correct)
    best_text = None
    best_logit = None
    for option in options:
        if option.strip() == correct.strip():
            continue
        token = _single_token(model.tokenizer, option.strip())
        if token is None:
            continue
        idx = model.to_single_token(token)
        if idx == correct_id:
            continue
        value = float(logits[idx])
        if best_logit is None or value > best_logit:
            best_logit = value
            best_text = token
    if best_text is None:
        raise ValueError(f"no single-token wrong option for prompt starting {prompt[:60]!r}")
    return best_text


def _options_from_prompt(prompt: str) -> list[str]:
    labels = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if len(stripped) >= 2 and stripped[1] == "." and stripped[0].isalpha():
            labels.append(stripped[0])
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True, help="HookedTransformer name, e.g. gemma-2-2b")
    parser.add_argument("--task", default="mcqa")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    payload = json.loads(args.prompts.read_text())
    model = HookedTransformer.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    kept = 0
    for entry in payload.get("tasks", {}).get(args.task, []):
        if kept >= args.limit:
            break
        options = _options_from_prompt(entry["clean_prompt"])
        foil = alternate_foil(model, entry["clean_prompt"], entry["correct_token"], options)
        entry["incorrect_token"] = foil
        entry.setdefault("metadata", {})["foil_mode"] = "highest_logit_wrong_option"
        kept += 1
    payload.setdefault("benchmarks_info", {})["foil_mode"] = "highest_logit_wrong_option"
    args.output.write_text(json.dumps(payload, indent=4) + "\n")
    print(f"rewrote {kept} {args.task} foils -> {args.output}")


if __name__ == "__main__":
    main()
