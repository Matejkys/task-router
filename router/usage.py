# router/usage.py
"""Pure, fail-safe token-usage extraction from Claude Code transcripts.

Mirrors router/budget.py's offset-aware reading so Part 2 (incremental cost
tracking) can reuse it. Bad lines are skipped; missing files yield empty
results. No config value (pricing, TTL defaults) is hardcoded here -- callers
pass it in from router/config.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class UsageRecord:
    model: str
    timestamp: str | None
    is_sidechain: bool
    input_tokens: int
    cache_read: int
    cache_write_5m: int
    cache_write_1h: int
    output_tokens: int


def _cache_write_split(usage: dict, default_cache_ttl: str) -> tuple[int, int]:
    creation = usage.get("cache_creation")
    if isinstance(creation, dict):
        return (
            creation.get("ephemeral_5m_input_tokens") or 0,
            creation.get("ephemeral_1h_input_tokens") or 0,
        )
    flat = usage.get("cache_creation_input_tokens") or 0
    if not flat:
        return 0, 0
    if default_cache_ttl == "1h":
        return 0, flat
    return flat, 0


def read_usage(
    path: Path, offset: int = 0, default_cache_ttl: str = "5m"
) -> tuple[list[UsageRecord], int]:
    """Read assistant usage records from `path` starting at byte `offset`.

    Returns (records, new_offset). A missing or unreadable file yields
    ([], 0). A file that shrank since `offset` (compaction, rewrite) is
    re-read from the start, matching router/budget.py.
    """
    try:
        size = path.stat().st_size
        start = 0 if offset > size else offset
        with path.open("r", errors="replace") as fh:
            fh.seek(start)
            data = fh.read()
    except OSError:
        return [], 0

    records: list[UsageRecord] = []
    for line in data.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("type") != "assistant":
            continue
        message = rec.get("message") or {}
        usage = message.get("usage") or {}
        cache_write_5m, cache_write_1h = _cache_write_split(usage, default_cache_ttl)
        records.append(
            UsageRecord(
                model=message.get("model") or "?",
                timestamp=rec.get("timestamp"),
                is_sidechain=bool(rec.get("isSidechain")),
                input_tokens=usage.get("input_tokens") or 0,
                cache_read=usage.get("cache_read_input_tokens") or 0,
                cache_write_5m=cache_write_5m,
                cache_write_1h=cache_write_1h,
                output_tokens=usage.get("output_tokens") or 0,
            )
        )

    return records, size


def subagent_transcripts(transcript_path: Path) -> list[Path]:
    """Subagent transcripts live at <dir>/<session_id>/subagents/*.jsonl,
    where session_id is the main transcript's filename stem."""
    session_dir = transcript_path.parent / transcript_path.stem / "subagents"
    if not session_dir.is_dir():
        return []
    return sorted(session_dir.glob("*.jsonl"))


@dataclass(frozen=True)
class Totals:
    input_tokens: int = 0
    cache_read: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0
    output_tokens: int = 0
    calls: int = 0

    def __add__(self, other: "Totals") -> "Totals":
        return Totals(
            input_tokens=self.input_tokens + other.input_tokens,
            cache_read=self.cache_read + other.cache_read,
            cache_write_5m=self.cache_write_5m + other.cache_write_5m,
            cache_write_1h=self.cache_write_1h + other.cache_write_1h,
            output_tokens=self.output_tokens + other.output_tokens,
            calls=self.calls + other.calls,
        )


def aggregate(records: list[UsageRecord]) -> dict[str, Totals]:
    out: dict[str, Totals] = {}
    for r in records:
        cur = out.get(r.model, Totals())
        out[r.model] = cur + Totals(
            input_tokens=r.input_tokens,
            cache_read=r.cache_read,
            cache_write_5m=r.cache_write_5m,
            cache_write_1h=r.cache_write_1h,
            output_tokens=r.output_tokens,
            calls=1,
        )
    return out


@dataclass
class CostBreakdown:
    per_model: dict[str, float] = field(default_factory=dict)
    total: float = 0.0
    unpriced: dict[str, Totals] = field(default_factory=dict)


def _model_cost(totals: Totals, price: dict) -> float:
    per_million = 1_000_000
    return (
        totals.input_tokens * price["input"]
        + totals.output_tokens * price["output"]
        + totals.cache_read * price["cache_read"]
        + totals.cache_write_5m * price["cache_write_5m"]
        + totals.cache_write_1h * price["cache_write_1h"]
    ) / per_million


def cost_usd(
    totals_by_model: dict[str, Totals],
    pricing: dict,
    zero_cost_models: frozenset[str] = frozenset(),
) -> CostBreakdown:
    """Compute USD cost per model. `pricing` maps model -> dict with keys
    input/output/cache_read/cache_write_5m/cache_write_1h (USD per million
    tokens). A model in neither `pricing` nor `zero_cost_models` is never
    silently priced: its totals go into `unpriced` instead."""
    breakdown = CostBreakdown()
    for model, totals in totals_by_model.items():
        if model in zero_cost_models:
            breakdown.per_model[model] = 0.0
            continue
        price = pricing.get(model)
        if price is None:
            breakdown.unpriced[model] = totals
            continue
        cost = _model_cost(totals, price)
        breakdown.per_model[model] = cost
        breakdown.total += cost
    return breakdown
