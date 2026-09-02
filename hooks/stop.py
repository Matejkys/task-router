#!/usr/bin/env python3
"""Stop: pair the router's decision with what actually happened."""

from __future__ import annotations

import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import hookio, paths, telemetry, usage
from router.config import Pricing, load_pricing, load_settings
from router.state import SessionState, load_state, save_state


def _load_pricing() -> Pricing | None:
    """Pricing is a nice-to-have here: without it the row still records tokens.
    A broken or missing config must never cost us the telemetry row itself."""
    try:
        return load_pricing(paths.pricing_yaml())
    except Exception:
        return None


def _accumulate(state: SessionState, transcript: Path, cache_ttl: str) -> None:
    """Fold every subagent transcript's new bytes into state.sub_usage.

    Subagent files appear while the session runs, so the set grows between Stop
    calls; each file carries its own offset and is read only past it.
    """
    for path in usage.subagent_transcripts(transcript):
        key = str(path)
        records, new_offset = usage.read_usage(
            path, state.sub_offsets.get(key, 0), cache_ttl
        )
        usage.merge_into(state.sub_usage, records)
        state.sub_offsets[key] = new_offset


def _usage_block(state: SessionState, pricing: Pricing) -> dict:
    models = {name: asdict(p) for name, p in pricing.models.items()}
    main = usage.cost_usd(
        usage.totals_by_model_from_store(state.main_usage),
        models,
        pricing.zero_cost_models,
    )
    sub = usage.cost_usd(
        usage.totals_by_model_from_store(state.sub_usage),
        models,
        pricing.zero_cost_models,
    )
    return {
        "main": state.main_usage,
        "sub": state.sub_usage,
        "cost_usd": {
            "main": main.total,
            "sub": sub.total,
            "total": main.total + sub.total,
        },
        "unpriced_models": sorted(set(main.unpriced) | set(sub.unpriced)),
    }


def main(payload: dict) -> None:
    settings = load_settings(paths.SETTINGS_YAML)
    state_dir = paths.state_dir_override() or settings.state_dir
    state = load_state(state_dir, payload["session_id"])
    if state is None:
        return

    transcript = Path(payload["transcript_path"])
    pricing = _load_pricing()
    cache_ttl = pricing.default_cache_ttl if pricing else usage.FALLBACK_CACHE_TTL

    # One read of the main transcript per Stop, shared by budget accounting and
    # per-model usage. read_increment applies the same rules over the same
    # records, so out_tokens keeps counting sidechain records too.
    records, new_offset = usage.read_usage(
        transcript, state.transcript_offset, cache_ttl
    )
    state.out_tokens += sum(r.output_tokens for r in records)
    state.transcript_offset = new_offset

    usage_block: dict | None = None
    try:
        usage.merge_into(state.main_usage, [r for r in records if not r.is_sidechain])
        usage.merge_into(state.sub_usage, [r for r in records if r.is_sidechain])
        _accumulate(state, transcript, cache_ttl)
        if pricing is not None:
            usage_block = _usage_block(state, pricing)
    except Exception:
        # Cost accounting is an addition to this hook, not its purpose. Any
        # failure in it must leave the base row (what shipped before) intact.
        usage_block = None

    save_state(state_dir, state)

    outcome: dict = {
        "out_tokens": state.out_tokens,
        "exceeded": bool(state.budget_soft and state.out_tokens > state.budget_soft),
        "budget_notified": state.budget_notified,
        "main_loop_code_edits": state.main_loop_code_edits,
    }
    if usage_block is not None:
        outcome["usage"] = usage_block

    telemetry.append(
        paths.telemetry_override() or settings.telemetry_path,
        {
            "session_id": state.session_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "class": state.cls,
            "confidence": state.confidence,
            "source": state.source,
            "budget_soft": state.budget_soft,
            "user_override": state.user_override,
            "overrides": state.overrides,
            "outcome": outcome,
        },
    )


if __name__ == "__main__":
    hookio.run(main)
