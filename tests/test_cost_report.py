import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from analysis.cost_report import (
    build_session_row,
    count_dispatches,
    count_interrupts,
    count_user_turns,
    dominant_model,
    first_user_text,
    group_key,
    is_internal_session,
    parse_records,
    scan_sessions,
    session_span_minutes,
    session_telemetry_classes,
    summarize_group,
)

MARKER = "Classify the following work request by the KIND of operation it asks"


def _assistant(model, ts, side=False, out=10, tool_uses=None):
    content: list[dict] = [{"type": "text", "text": "hi"}]
    for name in tool_uses or []:
        content.append({"type": "tool_use", "name": name, "input": {}})
    return {
        "type": "assistant",
        "timestamp": ts,
        "isSidechain": side,
        "message": {
            "model": model,
            "usage": {"input_tokens": 1, "output_tokens": out},
            "content": content,
        },
    }


def _user(text, ts, side=False):
    return {
        "type": "user",
        "timestamp": ts,
        "isSidechain": side,
        "message": {"content": text},
    }


def _user_tool_result(ts):
    return {
        "type": "user",
        "timestamp": ts,
        "isSidechain": False,
        "message": {"content": [{"type": "tool_result", "content": "ok"}]},
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_parse_records_skips_bad_lines(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text("not json\n" + json.dumps({"type": "user"}) + "\n")
    records = parse_records(t)
    assert records == [{"type": "user"}]


def test_scan_sessions_filters_by_mtime(tmp_path):
    root = tmp_path / "projects"
    old = root / "proj1" / "old.jsonl"
    new = root / "proj1" / "new.jsonl"
    old.parent.mkdir(parents=True)
    old.write_text("{}\n")
    new.write_text("{}\n")
    old_time = time.time() - 200 * 86400
    os.utime(old, (old_time, old_time))
    found = scan_sessions(root, since_days=90)
    assert new in found
    assert old not in found


def test_dominant_model_counts_main_loop_turns_only():
    records = [
        _assistant("claude-sonnet-5", "2026-09-01T10:00:00Z"),
        _assistant("claude-sonnet-5", "2026-09-01T10:01:00Z"),
        _assistant("claude-opus-5", "2026-09-01T10:02:00Z", side=True),  # sidechain
    ]
    assert dominant_model(records) == "claude-sonnet-5"


def test_count_user_turns_excludes_tool_results_and_interrupts():
    records = [
        _user("hello", "t1"),
        _user_tool_result("t2"),
        _user("[Request interrupted by user]", "t3"),
        _user("follow up", "t4"),
    ]
    assert count_user_turns(records) == 2


def test_count_interrupts():
    records = [
        _user("hello", "t1"),
        _user("[Request interrupted by user]", "t2"),
        _user("[Request interrupted by user]", "t3"),
    ]
    assert count_interrupts(records) == 2


def test_count_dispatches_counts_agent_and_task_tool_uses():
    records = [
        _assistant("claude-sonnet-5", "t1", tool_uses=["Agent"]),
        _assistant("claude-sonnet-5", "t2", tool_uses=["Task", "Bash"]),
        _assistant("claude-sonnet-5", "t3", tool_uses=["Bash"]),
    ]
    assert count_dispatches(records, dispatch_tools=frozenset({"Agent", "Task"})) == 2


def test_first_user_text_skips_sidechain():
    records = [
        _user("side text", "t0", side=True),
        _user("real first prompt", "t1"),
    ]
    assert first_user_text(records) == "real first prompt"


def test_is_internal_session_detects_router_marker():
    assert is_internal_session(f"prefix {MARKER} suffix", MARKER) is True
    assert is_internal_session("a normal prompt", MARKER) is False


def test_session_span_minutes():
    records = [
        _assistant("m", "2026-09-01T10:00:00Z"),
        _assistant("m", "2026-09-01T10:30:00Z"),
    ]
    assert session_span_minutes(records) == 30.0


def test_session_telemetry_classes_uses_latest_row_per_session():
    rows = [
        {"session_id": "s1", "class": "triage", "outcome": {"out_tokens": 10}},
        {"session_id": "s1", "class": "triage", "outcome": {"out_tokens": 200}},
        {"session_id": "s2", "class": "pr_review", "outcome": {"out_tokens": 5}},
    ]
    classes = session_telemetry_classes(rows)
    assert classes == {"s1": "triage", "s2": "pr_review"}


def test_build_session_row_sums_main_and_sub_cost(tmp_path):
    root = tmp_path / "projects" / "proj1"
    transcript = root / "sess1.jsonl"
    _write_jsonl(
        transcript,
        [
            _user("do the thing", "2026-09-01T10:00:00Z"),
            _assistant("claude-sonnet-5", "2026-09-01T10:00:05Z", out=1000, tool_uses=["Agent"]),
        ],
    )
    sub = root / "sess1" / "subagents" / "agent-1.jsonl"
    _write_jsonl(sub, [_assistant("claude-haiku-4-5-20251001", "2026-09-01T10:00:10Z", out=2000)])

    pricing = {
        "claude-sonnet-5": {"input": 2, "output": 10, "cache_read": 0.2, "cache_write_5m": 2.5, "cache_write_1h": 4},
        "claude-haiku-4-5-20251001": {"input": 1, "output": 5, "cache_read": 0.1, "cache_write_5m": 1.25, "cache_write_1h": 2},
    }
    row, reason = build_session_row(
        transcript,
        pricing=pricing,
        zero_cost_models=frozenset(),
        dispatch_tools=frozenset({"Agent", "Task"}),
        telemetry_classes={},
        internal_marker="__no_match__",
    )
    assert reason == "ok"
    assert row is not None
    assert row["cost_main"] > 0
    assert row["cost_sub"] > 0
    assert row["cost_total"] == row["cost_main"] + row["cost_sub"]
    assert row["dispatches"] == 1
    assert row["user_turns"] == 1
    assert row["era"] == "claude-sonnet-5"


def test_build_session_row_excludes_internal_sessions(tmp_path):
    root = tmp_path / "projects" / "proj1"
    transcript = root / "sess-internal.jsonl"
    _write_jsonl(
        transcript,
        [
            _user(f"blah {MARKER} blah", "2026-09-01T10:00:00Z"),
            _assistant("claude-haiku-4-5-20251001", "2026-09-01T10:00:01Z"),
        ],
    )
    row, reason = build_session_row(
        transcript,
        pricing={},
        zero_cost_models=frozenset(),
        dispatch_tools=frozenset({"Agent", "Task"}),
        telemetry_classes={},
        internal_marker=MARKER,
    )
    assert row is None
    assert reason == "internal"


def test_build_session_row_skips_empty_transcript_no_assistant_anywhere(tmp_path):
    root = tmp_path / "projects" / "proj1"
    transcript = root / "sess-empty.jsonl"
    _write_jsonl(transcript, [_user("hello there", "2026-09-01T10:00:00Z")])
    row, reason = build_session_row(
        transcript,
        pricing={},
        zero_cost_models=frozenset(),
        dispatch_tools=frozenset({"Agent", "Task"}),
        telemetry_classes={},
        internal_marker="__no_match__",
    )
    assert row is None
    assert reason == "empty"


def test_build_session_row_not_empty_when_only_subagent_has_assistant_records(tmp_path):
    root = tmp_path / "projects" / "proj1"
    transcript = root / "sess-sub-only.jsonl"
    _write_jsonl(transcript, [_user("hello there", "2026-09-01T10:00:00Z")])
    sub = root / "sess-sub-only" / "subagents" / "agent-1.jsonl"
    _write_jsonl(sub, [_assistant("claude-haiku-4-5-20251001", "2026-09-01T10:00:05Z", out=5)])
    row, reason = build_session_row(
        transcript,
        pricing={},
        zero_cost_models=frozenset(),
        dispatch_tools=frozenset({"Agent", "Task"}),
        telemetry_classes={},
        internal_marker="__no_match__",
    )
    assert reason == "ok"
    assert row is not None
    assert row["era"] is None


def test_group_key_by_era_and_week_and_class():
    row = {"era": "claude-sonnet-5", "class": "triage", "start": "2026-09-01T10:00:00+00:00"}
    assert group_key(row, "era") == "claude-sonnet-5"
    assert group_key(row, "model") == "claude-sonnet-5"
    assert group_key(row, "class") == "triage"
    assert group_key(row, "week").startswith("2026-W")


def test_summarize_group_computes_medians_and_shares():
    rows = [
        {"cost_main": 1.0, "cost_sub": 1.0, "cost_total": 2.0, "user_turns": 2,
         "interrupts": 0, "dispatches": 1},
        {"cost_main": 3.0, "cost_sub": 1.0, "cost_total": 4.0, "user_turns": 4,
         "interrupts": 2, "dispatches": 3},
    ]
    s = summarize_group(rows)
    assert s["n"] == 2
    assert s["median_cost"] == 3.0
    assert s["main_share_pct"] == 100 * 4 / 6
    assert s["sub_share_pct"] == 100 * 2 / 6
    assert s["median_user_turns"] == 3.0
    assert s["median_interrupts"] == 1.0
    assert s["median_dispatches"] == 2.0


def test_summarize_group_shares_and_cost_per_turn_are_none_when_total_cost_zero():
    rows = [
        {"cost_main": 0.0, "cost_sub": 0.0, "cost_total": 0.0, "user_turns": 2,
         "interrupts": 0, "dispatches": 1},
        {"cost_main": 0.0, "cost_sub": 0.0, "cost_total": 0.0, "user_turns": 3,
         "interrupts": 0, "dispatches": 0},
    ]
    s = summarize_group(rows)
    assert s["main_share_pct"] is None
    assert s["sub_share_pct"] is None
    assert s["median_cost_per_user_turn"] is None
    # unaffected metrics still compute normally
    assert s["median_user_turns"] == 2.5


def test_cost_report_loads_pricing_through_the_shared_accessor(tmp_path, monkeypatch):
    """The report and the Stop hook must price the same tokens from the same
    table, so TASK_ROUTER_PRICING has to reach both."""
    import analysis.cost_report as cr

    alternate = tmp_path / "pricing.yaml"
    alternate.write_text("models: {}\n")
    monkeypatch.setenv("TASK_ROUTER_PRICING", str(alternate))

    seen = []

    class _Stop(Exception):
        pass

    def _spy(path):
        seen.append(path)
        raise _Stop  # nothing past pricing is under test here

    monkeypatch.setattr(cr, "load_pricing", _spy)
    try:
        cr.main(["--json"])
    except _Stop:
        pass
    assert seen == [alternate]


def test_pricing_accessor_defaults_to_the_repo_table(monkeypatch):
    from router import paths

    monkeypatch.delenv("TASK_ROUTER_PRICING", raising=False)
    assert paths.pricing_yaml() == paths.PRICING_YAML
