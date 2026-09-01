"""Detection and fuzzing scoring for feature explanations (REQ-4, §3)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from rebuttal_eval.autointerp.llm_backends import LLMBackend
from rebuttal_eval.autointerp.explain import highlight_window
from rebuttal_eval.autointerp.prompts import (
    DETECTION_SYSTEM,
    DETECTION_USER_TEMPLATE,
    FUZZING_SYSTEM,
    FUZZING_USER_TEMPLATE,
    parse_yes_no,
)


def _decoy_highlight(
    tokens: list[str], token_acts: list[float], rng: random.Random
) -> str:
    """Highlight the same NUMBER of tokens, but deliberately wrong ones."""
    n_tokens = len(tokens)
    genuinely_active = {
        i for i, act in enumerate(token_acts[:n_tokens]) if act > 0
    }
    candidates = [i for i in range(n_tokens) if i not in genuinely_active]
    n_marks = max(1, min(len(genuinely_active) or 1, len(candidates)))
    marked = set(rng.sample(candidates, n_marks)) if candidates else set()
    return "".join(
        f"<<{token}>>" if i in marked else token for i, token in enumerate(tokens)
    )


def score_feature(
    feature: dict[str, Any],
    explanation: str,
    windows: list[list[str]],
    backend: LLMBackend,
    rng: random.Random,
    n_detection: int = 10,
    n_fuzzing: int = 10,
) -> dict[str, Any]:
    """Score one feature; returns per-feature accuracy stats."""
    activating = feature["scoring_activating"]
    non_activating = feature["scoring_non_activating"]

    # Detection: balanced held-out set, unmarked contexts.
    n_each = min(n_detection // 2, len(activating), len(non_activating))
    detection_correct = 0
    detection_total = 0
    if n_each > 0 and explanation:
        trials = [
            ("".join(windows[ex["window_id"]]), True)
            for ex in rng.sample(activating, n_each)
        ] + [
            ("".join(windows[wid]), False)
            for wid in rng.sample(non_activating, n_each)
        ]
        rng.shuffle(trials)
        for context, truth in trials:
            answer = parse_yes_no(
                backend.complete(
                    DETECTION_SYSTEM,
                    DETECTION_USER_TEMPLATE.format(
                        explanation=explanation, context=context
                    ),
                    max_tokens=5,
                )
            )
            if answer is not None:
                detection_total += 1
                detection_correct += int(answer == truth)

    # Fuzzing: same activating contexts, genuine vs decoy highlighting.
    n_fuzz_each = min(n_fuzzing // 2, len(activating))
    fuzzing_correct = 0
    fuzzing_total = 0
    if n_fuzz_each > 0 and explanation:
        chosen = rng.sample(activating, n_fuzz_each)
        trials = [
            (
                highlight_window(windows[ex["window_id"]], ex["token_acts"]),
                True,
            )
            for ex in chosen
        ] + [
            (
                _decoy_highlight(windows[ex["window_id"]], ex["token_acts"], rng),
                False,
            )
            for ex in chosen
        ]
        rng.shuffle(trials)
        for context, truth in trials:
            answer = parse_yes_no(
                backend.complete(
                    FUZZING_SYSTEM,
                    FUZZING_USER_TEMPLATE.format(
                        explanation=explanation, context=context
                    ),
                    max_tokens=5,
                )
            )
            if answer is not None:
                fuzzing_total += 1
                fuzzing_correct += int(answer == truth)

    return {
        "layer": feature["layer"],
        "feature": feature["feature"],
        "activation_frequency": feature["activation_frequency"],
        "explanation": explanation,
        "detection_accuracy": (
            detection_correct / detection_total if detection_total else None
        ),
        "detection_n": detection_total,
        "fuzzing_accuracy": (
            fuzzing_correct / fuzzing_total if fuzzing_total else None
        ),
        "fuzzing_n": fuzzing_total,
    }


def score_all(
    collection: dict[str, Any],
    explanations: dict[str, str],
    backend: LLMBackend,
    results_path: str | Path,
    seed: int,
    n_detection: int = 10,
    n_fuzzing: int = 10,
) -> list[dict[str, Any]]:
    """Score every feature; JSONL-cached per feature for resumability."""
    results_file = Path(results_path)
    done: dict[str, dict[str, Any]] = {}
    if results_file.exists():
        for line in results_file.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[f"{row['layer']}:{row['feature']}"] = row

    windows = collection["windows"]
    results: list[dict[str, Any]] = []
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "a") as handle:
        for index, feature in enumerate(collection["features"]):
            key = f"{feature['layer']}:{feature['feature']}"
            if key in done:
                results.append(done[key])
                continue
            rng = random.Random(seed * 100_003 + index)
            row = score_feature(
                feature,
                explanations.get(key, ""),
                windows,
                backend,
                rng,
                n_detection=n_detection,
                n_fuzzing=n_fuzzing,
            )
            results.append(row)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
    return results
