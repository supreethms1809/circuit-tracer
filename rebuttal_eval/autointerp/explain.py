"""Generate one-sentence feature explanations from top-activating contexts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rebuttal_eval.autointerp.llm_backends import LLMBackend
from rebuttal_eval.autointerp.prompts import EXPLAIN_SYSTEM, EXPLAIN_USER_TEMPLATE


def highlight_window(
    tokens: list[str], token_acts: list[float], threshold_frac: float = 0.3
) -> str:
    """Render a window with activating tokens wrapped in << >>.

    Tokens at or above `threshold_frac` of the window's max activation are
    highlighted, so weak incidental activations don't drown the pattern.
    """
    max_act = max(token_acts) if token_acts else 0.0
    cutoff = max_act * threshold_frac
    parts = []
    for token, act in zip(tokens, token_acts):
        parts.append(f"<<{token}>>" if max_act > 0 and act >= cutoff and act > 0 else token)
    return "".join(parts)


def explain_features(
    collection: dict[str, Any],
    backend: LLMBackend,
    cache_path: str | Path,
    max_examples: int = 10,
) -> dict[str, str]:
    """Return {\"layer:feature\": explanation}; JSONL-cached for resume."""
    cache_file = Path(cache_path)
    cache: dict[str, str] = {}
    if cache_file.exists():
        for line in cache_file.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["key"]] = row["explanation"]

    windows = collection["windows"]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "a") as handle:
        for feature in collection["features"]:
            key = f"{feature['layer']}:{feature['feature']}"
            if key in cache:
                continue
            examples = feature["top_examples"][:max_examples]
            if not examples:
                cache[key] = ""
                handle.write(json.dumps({"key": key, "explanation": ""}) + "\n")
                continue
            rendered = "\n---\n".join(
                highlight_window(windows[ex["window_id"]], ex["token_acts"])
                for ex in examples
            )
            explanation = backend.complete(
                EXPLAIN_SYSTEM,
                EXPLAIN_USER_TEMPLATE.format(examples=rendered),
                max_tokens=100,
            ).strip()
            cache[key] = explanation
            handle.write(json.dumps({"key": key, "explanation": explanation}) + "\n")
            handle.flush()
    return cache
