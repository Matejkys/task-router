import io
import json

import pytest

from router import hookio


def test_emit_prints_the_envelope(capsys):
    hookio.emit("PreToolUse", additionalContext="hi")
    out = json.loads(capsys.readouterr().out)
    assert out == {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse", "additionalContext": "hi",
        }
    }


def test_emit_with_nothing_to_say_prints_nothing(capsys):
    hookio.emit("PreToolUse")
    assert capsys.readouterr().out == ""


def test_emit_drops_none_fields(capsys):
    hookio.emit("PreToolUse", updatedInput=None, additionalContext="x")
    out = json.loads(capsys.readouterr().out)
    assert "updatedInput" not in out["hookSpecificOutput"]


def test_run_exits_zero_on_success(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO('{"session_id": "s"}'))
    with pytest.raises(SystemExit) as exc:
        hookio.run(lambda payload: None)
    assert exc.value.code == 0


def test_run_passes_the_parsed_payload(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO('{"session_id": "abc"}'))
    seen: dict = {}
    with pytest.raises(SystemExit):
        hookio.run(lambda payload: seen.update(payload))
    assert seen["session_id"] == "abc"


def test_run_honours_a_returned_exit_code(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    with pytest.raises(SystemExit) as exc:
        hookio.run(lambda payload: 2)
    assert exc.value.code == 2


def test_run_fails_open_on_an_exception(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    def boom(payload):
        raise RuntimeError("kaboom")

    with pytest.raises(SystemExit) as exc:
        hookio.run(boom)
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == "", "must never emit partial JSON"
    assert "RuntimeError: kaboom" in captured.err


def test_run_fails_open_on_malformed_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("{not json"))
    with pytest.raises(SystemExit) as exc:
        hookio.run(lambda payload: None)
    assert exc.value.code == 0
    assert capsys.readouterr().out == ""
