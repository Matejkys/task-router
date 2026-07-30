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


def test_wellformed_session_id_keeps_its_own_name(tmp_path):
    save_state(tmp_path, _state(session_id="a1b2-c3d4_e5"))
    assert (tmp_path / "a1b2-c3d4_e5.json").exists()


def test_traversal_cannot_escape_the_state_dir(tmp_path):
    save_state(tmp_path, _state(session_id="../../../../etc/passwd"))
    written = list(tmp_path.iterdir())
    assert len(written) == 1
    assert written[0].parent.resolve() == tmp_path.resolve()


def test_ids_differing_only_in_stripped_characters_do_not_collide(tmp_path):
    save_state(tmp_path, _state(session_id="a/b", out_tokens=1))
    save_state(tmp_path, _state(session_id="a:b", out_tokens=2))
    first = load_state(tmp_path, "a/b")
    second = load_state(tmp_path, "a:b")
    assert first is not None and first.out_tokens == 1
    assert second is not None and second.out_tokens == 2


def test_empty_session_id_does_not_collide_with_another_stripped_id(tmp_path):
    save_state(tmp_path, _state(session_id="", out_tokens=7))
    save_state(tmp_path, _state(session_id="///", out_tokens=9))
    empty = load_state(tmp_path, "")
    slashes = load_state(tmp_path, "///")
    assert empty is not None and empty.out_tokens == 7
    assert slashes is not None and slashes.out_tokens == 9


def test_unreadable_state_degrades_to_none(tmp_path):
    (tmp_path / "s1.json").mkdir()  # read_text on a directory raises OSError
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
