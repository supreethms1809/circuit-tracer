"""Prompt templates for auto-interp explanation and scoring."""

from __future__ import annotations

EXPLAIN_SYSTEM = (
    "You are analyzing a neuron (feature) inside a language model. You will "
    "see text excerpts where the feature activates; activating tokens are "
    "wrapped in << >> delimiters, with higher activation meaning stronger "
    "response. Describe, in ONE short sentence, the specific pattern the "
    "feature responds to. Be concrete (tokens, syntax, topic, position). Do "
    "not mention neurons, features, or activations in the sentence itself."
)

EXPLAIN_USER_TEMPLATE = (
    "Excerpts (activating tokens marked with << >>):\n\n{examples}\n\n"
    "One-sentence description of the pattern:"
)

DETECTION_SYSTEM = (
    "You are evaluating whether a description of a language-model feature "
    "matches a text excerpt. The feature fires on specific tokens matching "
    "the description. Answer with exactly YES if the feature would fire "
    "somewhere in the excerpt, or NO if it would not. Answer only YES or NO."
)

DETECTION_USER_TEMPLATE = (
    "Feature description: {explanation}\n\nExcerpt:\n{context}\n\n"
    "Would the feature fire in this excerpt? Answer YES or NO:"
)

FUZZING_SYSTEM = (
    "You are evaluating whether the marked tokens in a text excerpt match a "
    "feature description. Tokens wrapped in << >> are claimed to be where "
    "the feature fires. Answer with exactly YES if the marked tokens match "
    "the description, or NO if the marking looks wrong (the description may "
    "match the excerpt elsewhere, but the marked tokens are not the ones "
    "described). Answer only YES or NO."
)

FUZZING_USER_TEMPLATE = (
    "Feature description: {explanation}\n\nExcerpt with marked tokens:\n"
    "{context}\n\nDo the marked tokens match the description? Answer YES or NO:"
)


def parse_yes_no(text: str) -> bool | None:
    """Parse a YES/NO answer; None when the reply is unparseable."""
    cleaned = text.strip().upper()
    if cleaned.startswith("YES"):
        return True
    if cleaned.startswith("NO"):
        return False
    if "YES" in cleaned and "NO" not in cleaned:
        return True
    if "NO" in cleaned and "YES" not in cleaned:
        return False
    return None
