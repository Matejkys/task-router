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
