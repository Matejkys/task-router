# tests/test_classify.py
import json
import time
from pathlib import Path

from router.classify import UNCLASSIFIED, classify
from router.config import load_classes, load_settings

REPO = Path(__file__).resolve().parents[1]
CLASSES = load_classes(REPO / "config/classes.yaml", None)
SETTINGS = load_settings(REPO / "config/settings.yaml")

# Floors differ because the two fixtures are different instruments: the example
# set deliberately includes prompts the shipped rules miss, the local set is the
# hand-labelled corpus whose measured baseline was 87.5%.
LOCAL_FLOOR = 0.85
EXAMPLE_FLOOR = 0.70


def test_single_match_is_confident():
    r = classify(
        "Resolve issue https://example.com/org/repo/issues/1088", CLASSES, SETTINGS
    )
    assert r.cls == "resolve"
    assert r.confidence == SETTINGS.confidence_single
    assert r.source.startswith("rule:")


def test_no_match_is_unclassified():
    r = classify("hmm", CLASSES, SETTINGS)
    assert r.cls == UNCLASSIFIED
    assert r.confidence == SETTINGS.confidence_none
    assert r.source == "none"


def test_ambiguous_prompt_scores_below_threshold():
    # Mentions both a release (mechanical) and implementing (feature).
    r = classify("Implement the release automation for the deploy", CLASSES, SETTINGS)
    assert r.confidence == SETTINGS.confidence_multi
    assert r.confidence < SETTINGS.confidence_threshold, "must invite refinement"
    assert r.cls in {"mechanical", "feature"}


def test_review_of_a_pull_request_is_pr_review():
    r = classify(
        "Please review https://example.com/org/repo/pull/1056", CLASSES, SETTINGS
    )
    assert r.cls == "pr_review"


def test_golden_set_coverage_does_not_regress():
    local = REPO / "tests/fixtures/golden.local.jsonl"
    example = REPO / "tests/fixtures/golden.example.jsonl"
    using_local = local.exists()
    path = local if using_local else example
    floor = LOCAL_FLOOR if using_local else EXAMPLE_FLOOR

    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert rows, f"empty fixture: {path}"
    misses = [
        (r["prompt"], r["class"], classify(r["prompt"], CLASSES, SETTINGS).cls)
        for r in rows
        if classify(r["prompt"], CLASSES, SETTINGS).cls != r["class"]
    ]
    coverage = (len(rows) - len(misses)) / len(rows)
    print(f"\ngolden fixture: {path.name}  coverage={coverage:.1%} "
          f"({len(rows) - len(misses)}/{len(rows)})")
    for prompt, expected, got in misses:
        print(f"  MISS want={expected:12} got={got:12} {prompt[:60]}")
    assert coverage >= floor, (
        f"coverage {coverage:.1%} below floor {floor:.0%} using {path.name}"
    )


def test_rules_path_is_under_100ms():
    prompt = "Resolve issue https://example.com/org/repo/issues/1088 " * 20
    start = time.perf_counter()
    for _ in range(100):
        classify(prompt, CLASSES, SETTINGS)
    per_call_ms = (time.perf_counter() - start) / 100 * 1000
    assert per_call_ms < 100, f"{per_call_ms:.1f} ms per call"
