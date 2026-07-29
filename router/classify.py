# router/classify.py
"""Rule-based task classification.

Task difficulty is NOT predictable from prompt text (measured r=+0.24 for
length, 36-125x variance within a category). What is predictable is the kind of
operation, which is all this classifies.
"""

from __future__ import annotations

from dataclasses import dataclass

from router.config import ClassSpec, Settings

UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class Classification:
    cls: str
    confidence: float
    source: str


def classify(
    prompt: str, classes: dict[str, ClassSpec], settings: Settings
) -> Classification:
    matched: dict[str, list[str]] = {}
    for name, spec in classes.items():
        hits = [p.pattern for p in spec.patterns if p.search(prompt)]
        if hits:
            matched[name] = hits

    if not matched:
        return Classification(UNCLASSIFIED, settings.confidence_none, "none")

    # Most matching patterns wins; config order breaks ties because dicts
    # preserve insertion order and load_classes preserves YAML order.
    winner = max(matched, key=lambda n: len(matched[n]))
    confidence = (
        settings.confidence_single
        if len(matched) == 1
        else settings.confidence_multi
    )
    return Classification(winner, confidence, f"rule:{matched[winner][0]}")
