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
    would let a chatty session dominate calibration, so keep only one row per
    session_id — the final, fullest snapshot. A row without a session_id is
    treated as its own session.

    Rows now carry a `ts`, and where both rows have one the latest `ts` wins
    rather than the highest out_tokens: usage/cost totals are cumulative but
    out_tokens can legitimately reset when a transcript is compacted, and only
    the latest row holds the full usage block. The old max-out_tokens rule is
    kept as the tiebreak for the older, ts-less rows already in telemetry, and a
    ts-bearing row always beats a ts-less one (it is from the newer, richer
    format for the same session).
    """
    latest: dict[str, dict] = {}
    for i, row in enumerate(rows):
        key = row.get("session_id") or f"__no_session_{i}"
        held = latest.get(key)
        if held is None or _supersedes(row, held):
            latest[key] = row
    return list(latest.values())


def _supersedes(row: dict, held: dict) -> bool:
    ts, held_ts = row.get("ts"), held.get("ts")
    if ts and held_ts:
        return ts >= held_ts
    if ts or held_ts:
        return bool(ts)
    tokens = row.get("outcome", {}).get("out_tokens", 0)
    return tokens >= held.get("outcome", {}).get("out_tokens", 0)


def _percentile(values: list[float], q: float) -> float:
    idx = min(len(values) - 1, int(q * len(values)))
    return values[idx]


def summarise(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in _latest_per_session(rows):
        grouped[row.get("class", "unclassified")].append(row)

    out: dict[str, dict] = {}
    for cls, items in grouped.items():
        tokens = sorted(i.get("outcome", {}).get("out_tokens", 0) for i in items)
        out[cls] = {
            "n": len(items),
            "median": int(st.median(tokens)),
            "p75": int(_percentile(tokens, 0.75)),
            "exceeded": sum(
                1 for i in items if i.get("outcome", {}).get("exceeded")
            ),
            "overrides": sum(1 for i in items if i.get("overrides")),
            **_cost_stats(items),
        }
    return out


def _cost_stats(items: list[dict]) -> dict:
    """Cost figures over the sessions that actually carry a usage block.

    Rows predate the usage block or lost it to a pricing failure, so n_cost is
    reported separately from n: a class's cost stats speak for n_cost sessions,
    not for all of them. `main_share` is main-loop cost as a percentage of
    main+sub — the number the Fable-plans-cheaper hypothesis turns on.
    """
    costs: list[float] = []
    main_sum = 0.0
    total_sum = 0.0
    for item in items:
        block = item.get("outcome", {}).get("usage") or {}
        cost = block.get("cost_usd")
        if not isinstance(cost, dict):
            continue
        costs.append(float(cost.get("total") or 0.0))
        main_sum += float(cost.get("main") or 0.0)
        total_sum += float(cost.get("total") or 0.0)
    if not costs:
        return {"n_cost": 0, "cost_median": None, "cost_p75": None,
                "main_share": None}
    costs.sort()
    return {
        "n_cost": len(costs),
        "cost_median": st.median(costs),
        "cost_p75": _percentile(costs, 0.75),
        "main_share": (main_sum / total_sum * 100) if total_sum else None,
    }


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


def _money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:,.2f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}%"


def main() -> None:
    settings = load_settings(paths.SETTINGS_YAML)
    path = paths.telemetry_override() or settings.telemetry_path
    rows = telemetry.read_all(Path(path))
    summary = summarise(rows)
    print(f"{'class':14} {'n':>4} {'median':>10} {'p75':>10} {'over':>5} "
          f"{'ovr':>5} {'n$':>4} {'med$':>9} {'p75$':>9} {'main%':>6}")
    print("-" * 84)
    for cls, s in sorted(summary.items()):
        print(f"{cls:14} {s['n']:>4} {s['median']:>10,} {s['p75']:>10,} "
              f"{s['exceeded']:>5} {s['overrides']:>5} {s['n_cost']:>4} "
              f"{_money(s['cost_median']):>9} {_money(s['cost_p75']):>9} "
              f"{_pct(s['main_share']):>6}")
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
