from itertools import count
from pathlib import Path

from router.config import load_classes, load_settings
from router.report import propose, summarise

REPO = Path(__file__).resolve().parents[1]
CLASSES = load_classes(REPO / "config/classes.yaml", None)
SETTINGS = load_settings(REPO / "config/settings.yaml")

_seq = count()


def _row(
    cls: str, tokens: int, exceeded: bool = False, overrides=(), session=None
) -> dict:
    # Default each row to its own session so tests that don't care about
    # per-session dedup keep counting one sample per row.
    return {
        "session_id": session or f"auto-{next(_seq)}",
        "class": cls, "budget_soft": 100_000, "overrides": list(overrides),
        "outcome": {"out_tokens": tokens, "exceeded": exceeded},
    }


def test_summarise_groups_by_class():
    rows = [_row("triage", 100), _row("triage", 300), _row("pr_review", 50)]
    s = summarise(rows)
    assert s["triage"]["n"] == 2
    assert s["triage"]["median"] == 200
    assert s["pr_review"]["n"] == 1


def test_summarise_collapses_a_sessions_turns_to_one_sample():
    # One session appends a row per turn with growing out_tokens; it must count
    # once, at its fullest snapshot, not once per turn.
    rows = [
        _row("triage", 50, session="sess-A"),
        _row("triage", 180, session="sess-A"),
        _row("triage", 240, session="sess-A"),
    ]
    s = summarise(rows)
    assert s["triage"]["n"] == 1
    assert s["triage"]["median"] == 240


def test_dedup_keeps_the_fullest_snapshot_including_its_exceeded_flag():
    rows = [
        _row("triage", 50, exceeded=False, session="s1"),
        _row("triage", 250_000, exceeded=True, session="s1"),
    ]
    s = summarise(rows)
    assert s["triage"]["n"] == 1
    assert s["triage"]["exceeded"] == 1


def test_summarise_counts_exceeded_and_overrides():
    rows = [
        _row("triage", 500, exceeded=True, overrides=[{"precedence": "agent"}]),
        _row("triage", 10),
    ]
    s = summarise(rows)
    assert s["triage"]["exceeded"] == 1
    assert s["triage"]["overrides"] == 1


def _cost_row(cls, tokens, main, sub, session=None, ts=None) -> dict:
    row = _row(cls, tokens, session=session)
    row["outcome"]["usage"] = {
        "main": {}, "sub": {},
        "cost_usd": {"main": main, "sub": sub, "total": main + sub},
        "unpriced_models": [],
    }
    if ts is not None:
        row["ts"] = ts
    return row


def test_summarise_reports_cost_median_p75_and_main_share():
    rows = [
        _cost_row("triage", 10, main=1.0, sub=1.0),
        _cost_row("triage", 20, main=2.0, sub=6.0),
        _cost_row("triage", 30, main=3.0, sub=7.0),
    ]
    s = summarise(rows)
    assert s["triage"]["n_cost"] == 3
    assert s["triage"]["cost_median"] == 8.0
    assert s["triage"]["cost_p75"] == 10.0
    # 6 of 20 dollars went to the main loop.
    assert round(s["triage"]["main_share"], 1) == 30.0


def test_summarise_reports_no_cost_when_rows_lack_usage():
    s = summarise([_row("triage", 10), _row("triage", 20)])
    assert s["triage"]["n_cost"] == 0
    assert s["triage"]["cost_median"] is None
    assert s["triage"]["main_share"] is None


def test_cost_stats_ignore_sessions_without_usage():
    rows = [_row("triage", 10), _cost_row("triage", 20, main=1.0, sub=1.0)]
    s = summarise(rows)
    assert s["triage"]["n"] == 2
    assert s["triage"]["n_cost"] == 1
    assert s["triage"]["cost_median"] == 2.0


def test_dedup_prefers_the_latest_ts_over_the_highest_out_tokens():
    # After a compaction out_tokens can fall, but the later row holds the
    # cumulative usage block, so ts must win.
    rows = [
        _cost_row("triage", 900, main=9.0, sub=0.0, session="s1",
                  ts="2026-09-02T10:00:00+00:00"),
        _cost_row("triage", 100, main=1.0, sub=1.0, session="s1",
                  ts="2026-09-02T11:00:00+00:00"),
    ]
    s = summarise(rows)
    assert s["triage"]["n"] == 1
    assert s["triage"]["median"] == 100
    assert s["triage"]["cost_median"] == 2.0


def test_a_ts_bearing_row_beats_a_ts_less_one_for_the_same_session():
    rows = [
        _row("triage", 900, session="s1"),
        _cost_row("triage", 100, main=1.0, sub=1.0, session="s1",
                  ts="2026-09-02T11:00:00+00:00"),
    ]
    s = summarise(rows)
    assert s["triage"]["n"] == 1
    assert s["triage"]["n_cost"] == 1


def test_propose_flags_a_budget_that_is_too_low():
    rows = [_row("triage", 400_000, exceeded=True) for _ in range(5)]
    lines = propose(summarise(rows), CLASSES, SETTINGS)
    assert any("triage" in l and "100000" in l for l in lines)


def test_propose_stays_silent_on_thin_samples():
    thin = [_row("triage", 999_999, exceeded=True)] * (
        SETTINGS.min_samples_to_propose - 1
    )
    lines = propose(summarise(thin), CLASSES, SETTINGS)
    assert not any("triage" in l for l in lines)


def test_propose_flags_a_frequently_overridden_mandate():
    rows = [_row("triage", 10, overrides=[{"precedence": "agent"}]) for _ in range(5)]
    lines = propose(summarise(rows), CLASSES, SETTINGS)
    assert any("overridden" in l for l in lines)


def test_propose_never_returns_config_writes():
    rows = [_row("triage", 400_000, exceeded=True) for _ in range(5)]
    text = "\n".join(propose(summarise(rows), CLASSES, SETTINGS))
    assert "suggest" in text.lower() or "consider" in text.lower()
