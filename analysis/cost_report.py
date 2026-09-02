#!/usr/bin/env python3
# analysis/cost_report.py
"""Estimate USD cost of Claude Code sessions from local transcripts.

Scans ~/.claude/projects/*/*.jsonl (root configurable via
config/settings.yaml's `analysis` section), reads each session's main
transcript plus its subagents/*.jsonl files, and reports cost and behavioural
metrics grouped by era (dominant main-loop model), model, router class, or
week. Never prints raw prompt text.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from router import paths, telemetry
from router.config import Settings, load_pricing, load_settings
from router.report import _latest_per_session
from router.usage import Totals, UsageRecord, aggregate, cost_usd, read_usage, subagent_transcripts


# ---------------------------------------------------------------------------
# Transcript parsing helpers (pure; take already-loaded records where possible)
# ---------------------------------------------------------------------------

def parse_records(path: Path) -> list[dict]:
    """Read a JSONL transcript, skipping unparsable lines. Missing file -> []."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    records = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


def scan_sessions(root: Path, since_days: int | None = None, since_date: datetime | None = None) -> list[Path]:
    """List *.jsonl transcripts under root/*/*.jsonl, filtered by mtime."""
    if since_date is not None:
        cutoff = since_date
    elif since_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    else:
        cutoff = None

    found = []
    if not root.is_dir():
        return found
    for path in root.glob("*/*.jsonl"):
        if cutoff is not None:
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            except OSError:
                continue
            if mtime < cutoff:
                continue
        found.append(path)
    return found


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def dominant_model(records: list[dict]) -> str | None:
    """Most common model among non-sidechain assistant turns."""
    counts: Counter[str] = Counter()
    for rec in records:
        if rec.get("type") != "assistant" or rec.get("isSidechain"):
            continue
        model = (rec.get("message") or {}).get("model")
        if model:
            counts[model] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _is_real_user_text(rec: dict) -> str | None:
    """Return the text of a genuine (non-tool-result) user message, or None."""
    if rec.get("type") != "user" or rec.get("isSidechain"):
        return None
    content = (rec.get("message") or {}).get("content")
    if isinstance(content, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return None
        text = " ".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    else:
        text = content
    if not isinstance(text, str) or not text.strip():
        return None
    return text.strip()


def count_user_turns(records: list[dict], interrupt_marker: str = "[Request interrupted") -> int:
    count = 0
    for rec in records:
        text = _is_real_user_text(rec)
        if text is None or text.startswith(interrupt_marker):
            continue
        count += 1
    return count


def count_interrupts(records: list[dict], interrupt_marker: str = "[Request interrupted") -> int:
    count = 0
    for rec in records:
        text = _is_real_user_text(rec)
        if text is not None and text.startswith(interrupt_marker):
            count += 1
    return count


def count_dispatches(records: list[dict], dispatch_tools: frozenset[str]) -> int:
    count = 0
    for rec in records:
        if rec.get("type") != "assistant":
            continue
        for block in (rec.get("message") or {}).get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") in dispatch_tools:
                count += 1
    return count


def first_user_text(records: list[dict]) -> str | None:
    for rec in records:
        text = _is_real_user_text(rec)
        if text is not None:
            return text
    return None


def is_internal_session(text: str | None, marker: str) -> bool:
    return bool(text) and marker in text


def session_span_minutes(records: list[dict]) -> float:
    timestamps = [t for t in (_parse_ts(r.get("timestamp")) for r in records) if t is not None]
    if len(timestamps) < 2:
        return 0.0
    return round((max(timestamps) - min(timestamps)).total_seconds() / 60, 2)


def session_telemetry_classes(rows: list[dict]) -> dict[str, str]:
    """session_id -> router class, from the fullest (latest) telemetry row."""
    return {
        row["session_id"]: row.get("class", "unclassified")
        for row in _latest_per_session(rows)
        if row.get("session_id")
    }


# ---------------------------------------------------------------------------
# Per-session row construction
# ---------------------------------------------------------------------------

def build_session_row(
    transcript: Path,
    *,
    pricing: dict,
    zero_cost_models: frozenset[str],
    dispatch_tools: frozenset[str],
    telemetry_classes: dict[str, str],
    internal_marker: str,
    interrupt_marker: str = "[Request interrupted",
    default_cache_ttl: str = "5m",
) -> tuple[dict | None, str]:
    """Build one session's metrics row.

    Returns (row, reason). reason is "ok" on success; on exclusion row is
    None and reason is "internal" (the router's own `claude -p` classifier
    session) or "empty" (no assistant records anywhere, main or subagent --
    not worth a row, and not an "internal" exclusion either)."""
    records = parse_records(transcript)
    if not records:
        return None, "empty"

    first_text = first_user_text(records)
    if is_internal_session(first_text, internal_marker):
        return None, "internal"

    main_records = [r for r in records if not r.get("isSidechain")]

    main_usage, _ = read_usage(transcript, 0, default_cache_ttl=default_cache_ttl)
    main_only = [u for u in main_usage if not u.is_sidechain]
    sub_from_main = [u for u in main_usage if u.is_sidechain]

    sub_usage: list[UsageRecord] = list(sub_from_main)
    for sub_path in subagent_transcripts(transcript):
        recs, _ = read_usage(sub_path, 0, default_cache_ttl=default_cache_ttl)
        sub_usage.extend(recs)

    if not main_only and not sub_usage:
        return None, "empty"

    main_totals = aggregate(main_only)
    sub_totals = aggregate(sub_usage)

    main_cost = cost_usd(main_totals, pricing, zero_cost_models)
    sub_cost = cost_usd(sub_totals, pricing, zero_cost_models)

    combined_totals: dict[str, Totals] = dict(main_totals)
    for model, totals in sub_totals.items():
        combined_totals[model] = combined_totals.get(model, Totals()) + totals

    unpriced = {**main_cost.unpriced}
    for model, totals in sub_cost.unpriced.items():
        unpriced[model] = unpriced.get(model, Totals()) + totals

    era = dominant_model(main_records)
    session_id = transcript.stem
    start_ts = None
    for r in records:
        t = _parse_ts(r.get("timestamp"))
        if t is not None:
            start_ts = t
            break

    row = {
        "session_id": session_id,
        "file": str(transcript),
        "era": era,
        "class": telemetry_classes.get(session_id, "unclassified"),
        "start": start_ts.isoformat() if start_ts else None,
        "span_min": session_span_minutes(records),
        "cost_main": round(main_cost.total, 4),
        "cost_sub": round(sub_cost.total, 4),
        "cost_total": round(main_cost.total + sub_cost.total, 4),
        "user_turns": count_user_turns(records, interrupt_marker),
        "interrupts": count_interrupts(records, interrupt_marker),
        "dispatches": count_dispatches(main_records, dispatch_tools),
        "totals_by_model": {m: asdict(t) for m, t in combined_totals.items()},
        "unpriced": {m: asdict(t) for m, t in unpriced.items()},
    }
    return row, "ok"


# ---------------------------------------------------------------------------
# Grouping and summarising
# ---------------------------------------------------------------------------

def group_key(row: dict, by: str) -> str:
    if by in ("era", "model"):
        return row.get("era") or "unknown"
    if by == "class":
        return row.get("class") or "unclassified"
    if by == "week":
        start = _parse_ts(row.get("start"))
        if start is None:
            return "unknown"
        iso = start.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    raise ValueError(f"unknown grouping: {by}")


def _median(values: list[float]) -> float:
    return float(st.median(values)) if values else 0.0


def _p75(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(0.75 * len(ordered)))
    return ordered[idx]


def summarize_group(rows: list[dict]) -> dict:
    """Per-group medians/p75 and cost shares. Shares and cost-per-turn are
    `None` ("n/a") rather than 0%/100% when the group's total cost is zero --
    a share computed over a zero denominator is not a real ratio."""
    n = len(rows)
    total_costs = [r["cost_total"] for r in rows]
    main_sum = sum(r["cost_main"] for r in rows)
    sub_sum = sum(r["cost_sub"] for r in rows)
    denom = main_sum + sub_sum
    user_turns = [r["user_turns"] for r in rows]
    cost_per_turn = [
        (r["cost_total"] / r["user_turns"]) for r in rows if r["user_turns"] > 0
    ]
    return {
        "n": n,
        "median_cost": _median(total_costs),
        "p75_cost": _p75(total_costs),
        "main_share_pct": (100 * main_sum / denom) if denom else None,
        "sub_share_pct": (100 * sub_sum / denom) if denom else None,
        "median_cost_per_user_turn": _median(cost_per_turn) if denom else None,
        "median_interrupts": _median([r["interrupts"] for r in rows]),
        "median_dispatches": _median([r["dispatches"] for r in rows]),
        "median_user_turns": _median(user_turns),
    }


def group_rows(rows: list[dict], by: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row, by)].append(row)
    return dict(grouped)


def totals_by_model_across(rows: list[dict]) -> dict[str, Totals]:
    out: dict[str, Totals] = {}
    for row in rows:
        for model, tdict in row["totals_by_model"].items():
            out[model] = out.get(model, Totals()) + Totals(**tdict)
    return out


def unpriced_by_model_across(rows: list[dict]) -> dict[str, Totals]:
    out: dict[str, Totals] = {}
    for row in rows:
        for model, tdict in row["unpriced"].items():
            out[model] = out.get(model, Totals()) + Totals(**tdict)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _collect_rows(settings: Settings, pricing_models: dict, zero_cost_models: frozenset, since_days: int | None, since_date):
    telemetry_rows = telemetry.read_all(settings.analysis_telemetry_path)
    telemetry_classes = session_telemetry_classes(telemetry_rows)

    transcripts = scan_sessions(
        settings.analysis_projects_root, since_days=since_days, since_date=since_date
    )
    rows = []
    excluded_internal = 0
    skipped_empty = 0
    for transcript in transcripts:
        row, reason = build_session_row(
            transcript,
            pricing=pricing_models,
            zero_cost_models=zero_cost_models,
            dispatch_tools=settings.dispatch_tools,
            telemetry_classes=telemetry_classes,
            internal_marker=settings.analysis_internal_session_marker,
        )
        if row is None:
            if reason == "internal":
                excluded_internal += 1
            else:
                skipped_empty += 1
            continue
        rows.append(row)
    return rows, excluded_internal, skipped_empty


def _fmt_pct(value: float | None) -> str:
    return "   n/a" if value is None else f"{value:5.1f}%"


def _fmt_cost(value: float | None) -> str:
    return "     n/a" if value is None else f"{value:9.3f}"


def _print_group_table(rows: list[dict], by: str) -> None:
    groups = group_rows(rows, by)
    header = (
        f"{'group':20} {'n':>4} {'median$':>10} {'p75$':>10} "
        f"{'main%':>7} {'sub%':>7} {'$/turn':>9} {'med_int':>8} {'med_disp':>9} {'med_turns':>10}"
    )
    print(header)
    print("-" * len(header))
    for key in sorted(groups):
        s = summarize_group(groups[key])
        print(
            f"{key:20} {s['n']:>4} {s['median_cost']:>10.2f} {s['p75_cost']:>10.2f} "
            f"{_fmt_pct(s['main_share_pct'])} {_fmt_pct(s['sub_share_pct'])} "
            f"{_fmt_cost(s['median_cost_per_user_turn'])} {s['median_interrupts']:>8.1f} "
            f"{s['median_dispatches']:>9.1f} {s['median_user_turns']:>10.1f}"
        )


def _print_model_cost_table(all_totals: dict[str, Totals], pricing_models: dict, zero_cost_models: frozenset) -> None:
    breakdown = cost_usd(all_totals, pricing_models, zero_cost_models)
    grand_total = breakdown.total
    header = (
        f"{'model':30} {'input':>12} {'output':>12} {'cache_r':>12} {'cache_w5m':>12} "
        f"{'cache_w1h':>12} {'calls':>7} {'cost$':>10} {'share%':>8}"
    )
    print(header)
    print("-" * len(header))
    for model, t in sorted(all_totals.items()):
        cost = breakdown.per_model.get(model)
        share = (100 * cost / grand_total) if cost is not None and grand_total else None
        cost_str = f"{cost:10.2f}" if cost is not None else f"{'n/a':>10}"
        share_str = f"{share:7.1f}%" if share is not None else f"{'n/a':>8}"
        print(
            f"{model:30} {t.input_tokens:>12,} {t.output_tokens:>12,} {t.cache_read:>12,} "
            f"{t.cache_write_5m:>12,} {t.cache_write_1h:>12,} {t.calls:>7,} {cost_str} {share_str}"
        )


def _print_human(
    rows: list[dict],
    by: str,
    all_totals: dict[str, Totals],
    unpriced: dict[str, Totals],
    excluded_internal: int,
    skipped_empty: int,
    pricing_models: dict,
    zero_cost_models: frozenset,
):
    print(
        f"Sessions analysed: {len(rows)} "
        f"(excluded {excluded_internal} internal router sessions, "
        f"skipped empty transcripts: {skipped_empty})\n"
    )

    if by == "model":
        # --by model shows only the per-model cost breakdown: a session-level
        # grouping table would just repeat --by era's rows under a
        # different label, since "model" and "era" are the same signal here.
        _print_model_cost_table(all_totals, pricing_models, zero_cost_models)
    else:
        _print_group_table(rows, by)
        print("\nTotals by model:")
        _print_model_cost_table(all_totals, pricing_models, zero_cost_models)

    if unpriced:
        names = ", ".join(sorted(unpriced))
        print(f"\nUnpriced models (tokens not costed): {names}")
    else:
        print("\nUnpriced models: none")


def main(argv: list[str] | None = None) -> None:
    settings = load_settings(paths.SETTINGS_YAML)
    # Same accessor the Stop hook uses, so TASK_ROUTER_PRICING points both
    # at one table -- live telemetry and this retroactive report must never
    # price the same tokens differently.
    pricing = load_pricing(paths.pricing_yaml())
    pricing_models = {name: asdict(p) for name, p in pricing.models.items()}

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=settings.analysis_default_days)
    parser.add_argument("--since", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--by", choices=["era", "model", "class", "week"], default="era")
    args = parser.parse_args(argv)

    since_date = None
    since_days = args.days
    if args.since:
        since_date = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        since_days = None

    rows, excluded_internal, skipped_empty = _collect_rows(
        settings, pricing_models, pricing.zero_cost_models, since_days, since_date
    )
    all_totals = totals_by_model_across(rows)
    unpriced = unpriced_by_model_across(rows)

    if args.json:
        groups = group_rows(rows, args.by)
        model_costs = cost_usd(all_totals, pricing_models, pricing.zero_cost_models)
        out = {
            "sessions": rows,
            "excluded_internal": excluded_internal,
            "skipped_empty": skipped_empty,
            "groups": {k: summarize_group(v) for k, v in groups.items()},
            "totals_by_model": {m: asdict(t) for m, t in all_totals.items()},
            "cost_by_model": model_costs.per_model,
            "unpriced": {m: asdict(t) for m, t in unpriced.items()},
        }
        print(json.dumps(out, indent=1))
    else:
        _print_human(
            rows, args.by, all_totals, unpriced, excluded_internal, skipped_empty,
            pricing_models, pricing.zero_cost_models,
        )


if __name__ == "__main__":
    main()
