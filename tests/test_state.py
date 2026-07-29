# tests/test_state.py
from pathlib import Path

from router.config import load_settings
from router.continuation import is_continuation
from router.state import SessionState, load_state, save_state

REPO = Path(__file__).resolve().parents[1]
SETTINGS = load_settings(REPO / "config/settings.yaml")


def _state(**kw) -> SessionState:
    base = dict(
        session_id="s1", cls="triage", confidence=0.9, source="rule:x",
        budget_soft=100_000, budget_notified=False, transcript_offset=0,
        out_tokens=0, user_override=False, overrides=[],
    )
    # ty cannot verify per-field types through a merged-dict ** unpack (the
    # merged dict's value type is a union of every field's type); the 8 call
    # sites below all pass values matching SessionState's real field types.
    return SessionState(**{**base, **kw})  # ty: ignore[invalid-argument-type]


def test_roundtrip(tmp_path):
    s = _state(out_tokens=1234, overrides=[{"from": "a", "to": "b"}])
    save_state(tmp_path, s)
    loaded = load_state(tmp_path, "s1")
    assert loaded is not None
    assert loaded.out_tokens == 1234
    assert loaded.cls == "triage"
    assert loaded.overrides == [{"from": "a", "to": "b"}]


def test_missing_state_is_none(tmp_path):
    assert load_state(tmp_path, "nope") is None


def test_corrupt_state_is_none_not_a_crash(tmp_path):
    (tmp_path / "s1.json").write_text("{not json")
    assert load_state(tmp_path, "s1") is None


def test_short_followup_in_a_progressed_session_is_continuation():
    st = _state(out_tokens=SETTINGS.continuation_min_out_tokens)
    assert is_continuation("that's fine, keep going", st, SETTINGS) is True


def test_no_prior_state_is_never_continuation():
    assert is_continuation("keep going", None, SETTINGS) is False


def test_barely_started_session_is_a_new_task():
    st = _state(out_tokens=SETTINGS.continuation_min_out_tokens - 1)
    assert is_continuation("keep going", st, SETTINGS) is False


def test_long_prompt_is_a_new_task():
    st = _state(out_tokens=999_999)
    assert is_continuation("x" * 500, st, SETTINGS) is False


def test_new_url_means_new_task_even_if_short():
    st = _state(out_tokens=999_999)
    assert is_continuation("fix https://example.com/a/b/issues/9", st, SETTINGS) is False
