# router/report.py
"""Recompute per-class reality from telemetry and propose budget changes.

Proposes only. The config stays reviewable, because a router that retunes
itself silently detunes itself.
"""

from __future__ import annotations

import statistics as st
from collections import defaultdict
from pathlib import Path

from router import paths, telemetry
from router.config import ClassSpec, Settings, load_classes, load_settings


def _latest_per_session(rows: list[dict]) -> list[dict]:
    """Collapse a session's many per-turn rows to its fullest one.

    The Stop hook fires once per assistant turn, so one session appends many
    rows under one session_id with growing out_tokens. Counting each as a sample
    would let a chatty session dominate calibration, so keep only the row with
    the highest out_tokens per session_id — the final, fullest snapshot. A row
    without a session_id is treated as its own session.
    """
    latest: dict[str, dict] = {}
    for i, row in enumerate(rows):
        key = row.get("session_id") or f"__no_session_{i}"
        tokens = row.get("outcome", {}).get("out_tokens", 0)
        held = latest.get(key)
        if held is None or tokens >= held.get("outcome", {}).get("out_tokens", 0):
            latest[key] = row
    return list(latest.values())


def summarise(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in _latest_per_session(rows):
        grouped[row.get("class", "unclassified")].append(row)

    out: dict[str, dict] = {}
    for cls, items in grouped.items():
        tokens = sorted(i.get("outcome", {}).get("out_tokens", 0) for i in items)
        idx = min(len(tokens) - 1, int(0.75 * len(tokens)))
        out[cls] = {
            "n": len(items),
            "median": int(st.median(tokens)),
            "p75": tokens[idx],
            "exceeded": sum(
                1 for i in items if i.get("outcome", {}).get("exceeded")
            ),
            "overrides": sum(1 for i in items if i.get("overrides")),
        }
    return out


def propose(
    summary: dict[str, dict], classes: dict[str, ClassSpec], settings: Settings
) -> list[str]:
    lines: list[str] = []
    for cls, stats in sorted(summary.items()):
        spec = classes.get(cls)
        if spec is None or stats["n"] < settings.min_samples_to_propose:
            continue
        if stats["exceeded"] / stats["n"] >= settings.exceeded_ratio_threshold:
            lines.append(
                f"{cls}: budget_soft {spec.budget_soft} exceeded in "
                f"{stats['exceeded']}/{stats['n']} sessions; observed p75 is "
                f"{stats['p75']}. Consider raising it to {stats['p75']}."
            )
        if stats["overrides"] / stats["n"] >= settings.override_ratio_threshold:
            lines.append(
                f"{cls}: overridden in {stats['overrides']}/{stats['n']} "
                f"sessions. Suggest revisiting sub_model={spec.sub_model} - "
                f"frequent overrides mean the mandate is wrong, not the agent."
            )
    return lines


def main() -> None:
    settings = load_settings(paths.SETTINGS_YAML)
    path = paths.telemetry_override() or settings.telemetry_path
    rows = telemetry.read_all(Path(path))
    summary = summarise(rows)
    print(f"{'class':14} {'n':>4} {'median':>10} {'p75':>10} {'over':>5} {'ovr':>5}")
    print("-" * 54)
    for cls, s in sorted(summary.items()):
        print(f"{cls:14} {s['n']:>4} {s['median']:>10,} {s['p75']:>10,} "
              f"{s['exceeded']:>5} {s['overrides']:>5}")
    classes = load_classes(paths.CLASSES_YAML, paths.CLASSES_LOCAL_YAML)
    suggestions = propose(summary, classes, settings)
    if suggestions:
        print("\nSuggestions (apply by hand):")
        for line in suggestions:
            print(f"  - {line}")
    else:
        print(
            f"\nNo suggestions: fewer than {settings.min_samples_to_propose} "
            f"sessions per class."
        )


if __name__ == "__main__":
    main()
