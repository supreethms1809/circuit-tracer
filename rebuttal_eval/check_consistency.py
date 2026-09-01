"""§4.6 numerical consistency: prose numbers vs table numbers.

Scans markdown files for numbers appearing in prose and in tables, and
reports any value that appears with inconsistent precision/rounding (the
submitted paper shipped 1.903 vs 1.90/1.91, 44.7 vs 44.72, 81.53 vs 81.68,
feeding a Clarity score of 1). Run over the final rebuttal drafts before
pasting to OpenReview.

Usage:
  python -m rebuttal_eval.check_consistency FILE.md [FILE.md ...]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

_NUMBER = re.compile(r"(?<![\w.])(\d+\.\d+)(?![\w.])")


def scan(paths: list[str]) -> dict[str, list[tuple[str, str]]]:
    """Group numbers by integer-part+2-decimal prefix; flag divergent forms."""
    groups: dict[str, set[str]] = defaultdict(set)
    locations: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path in paths:
        text = Path(path).read_text()
        for match in _NUMBER.finditer(text):
            value = match.group(1)
            prefix = value[: value.index(".") + 3]
            groups[prefix].add(value)
            locations[value].append((path, prefix))

    conflicts: dict[str, list[tuple[str, str]]] = {}
    for prefix, forms in groups.items():
        if len(forms) > 1:
            conflicts[prefix] = [
                (form, ", ".join(sorted({p for p, _ in locations[form]})))
                for form in sorted(forms)
            ]
    return conflicts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args(argv)

    conflicts = scan(args.files)
    if not conflicts:
        print("no prose/table numeric disagreements found")
        return 0
    print(f"{len(conflicts)} potential disagreement group(s):")
    for prefix, forms in sorted(conflicts.items()):
        print(f"  ~{prefix}:")
        for form, sources in forms:
            print(f"    {form}  ({sources})")
    print(
        "\nReview each group: same quantity rendered at different precision "
        "must be unified before pasting (spec §4.6)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
