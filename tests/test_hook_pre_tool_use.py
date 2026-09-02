import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HOOK = REPO / "hooks/pre_tool_use.py"


def seed_state(state_dir: Path, **kw) -> None:
    state = {
        "session_id": "s1", "cls": "pr_review", "confidence": 0.9,
        "source": "rule:x", "budget_soft": 200_000, "budget_notified": False,
        "transcript_offset": 0, "out_tokens": 0, "user_override": False,
        "overrides": [],
    }
    state.update(kw)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "s1.json").write_text(json.dumps(state))


def run_hook(payload: dict, state_dir: Path, enforce: bool) -> dict:
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload), capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO),
             "TASK_ROUTER_STATE_DIR": str(state_dir),
             "TASK_ROUTER_ENFORCE": "1" if enforce else "0"},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == "", f"hook wrote a fail-open diagnostic: {proc.stderr}"
    return json.loads(proc.stdout) if proc.stdout.strip() else {}


def _agent_payload(tmp_path: Path) -> dict:
    return {
        "session_id": "s1", "hook_event_name": "PreToolUse",
        "tool_name": "Agent",
        "tool_input": {"model": "opus", "prompt": "review the diff"},
        "transcript_path": str(tmp_path / "t.jsonl"),
    }


def test_enforce_on_rewrites_the_model(tmp_path):
    seed_state(tmp_path)
    out = run_hook(_agent_payload(tmp_path), tmp_path, enforce=True)
    hso = out["hookSpecificOutput"]
    assert hso["updatedInput"] == {"model": "sonnet", "prompt": "review the diff"}
    assert "claude-sonnet-5" in hso["additionalContext"], "rewrite must be visible"


def test_shadow_mode_logs_but_does_not_rewrite(tmp_path):
    seed_state(tmp_path)
    out = run_hook(_agent_payload(tmp_path), tmp_path, enforce=False)
    hso = out.get("hookSpecificOutput", {})
    assert "updatedInput" not in hso
    assert json.loads((tmp_path / "s1.json").read_text())["overrides"], \
        "shadow mode must still record what it would have done"


def test_user_override_is_recorded(tmp_path):
    # The user named the model explicitly, diverging from the pr_review
    # mandate (claude-sonnet-5). The router must defer - no updatedInput -
    # but still record the override so `router report` can surface it.
    seed_state(tmp_path, user_override=True)
    out = run_hook(_agent_payload(tmp_path), tmp_path, enforce=True)
    assert "updatedInput" not in out.get("hookSpecificOutput", {})
    overrides = json.loads((tmp_path / "s1.json").read_text())["overrides"]
    assert overrides == [{
        "from": "claude-opus-5", "to": "claude-sonnet-5",
        "precedence": "user", "enforced": False,
    }]


def test_agent_override_is_recorded(tmp_path):
    # The dispatch carries a justified override marker with a reason. The
    # router must defer - no updatedInput - but still record the override.
    seed_state(tmp_path, user_override=False)
    payload = {
        "session_id": "s1", "hook_event_name": "PreToolUse",
        "tool_name": "Agent",
        "tool_input": {
            "model": "opus",
            "prompt": "review the diff\nmodel-override: needs cross-file reasoning",
        },
        "transcript_path": str(tmp_path / "t.jsonl"),
    }
    out = run_hook(payload, tmp_path, enforce=True)
    assert "updatedInput" not in out.get("hookSpecificOutput", {})
    overrides = json.loads((tmp_path / "s1.json").read_text())["overrides"]
    assert overrides == [{
        "from": "claude-opus-5", "to": "claude-sonnet-5",
        "precedence": "agent", "enforced": False,
    }]


def test_fill_emits_rewrite_but_no_override_and_no_notice(tmp_path):
    # No model dispatched at all - the router filling in the default, not a
    # conflict. updatedInput must still land (the mandate must land), but no
    # override is recorded and no ROUTER: notice is injected.
    seed_state(tmp_path)
    payload = {
        "session_id": "s1", "hook_event_name": "PreToolUse",
        "tool_name": "Agent",
        "tool_input": {"prompt": "review the diff"},
        "transcript_path": str(tmp_path / "t.jsonl"),
    }
    out = run_hook(payload, tmp_path, enforce=True)
    hso = out["hookSpecificOutput"]
    assert hso["updatedInput"] == {"model": "sonnet", "prompt": "review the diff"}
    assert "additionalContext" not in hso or "ROUTER:" not in hso.get(
        "additionalContext", ""
    )
    overrides = json.loads((tmp_path / "s1.json").read_text())["overrides"]
    assert overrides == []


def test_no_override_recorded_when_actor_matches_mandate(tmp_path):
    # The user named a model, but it happens to equal the mandate - nothing
    # was actually overridden, so no entry should be recorded.
    seed_state(tmp_path, user_override=True)
    payload = {
        "session_id": "s1", "hook_event_name": "PreToolUse",
        "tool_name": "Agent",
        "tool_input": {"model": "claude-sonnet-5", "prompt": "review the diff"},
        "transcript_path": str(tmp_path / "t.jsonl"),
    }
    run_hook(payload, tmp_path, enforce=True)
    overrides = json.loads((tmp_path / "s1.json").read_text())["overrides"]
    assert overrides == []


def test_spawn_task_denied_when_enforcing(tmp_path):
    seed_state(tmp_path)
    out = run_hook(
        {"session_id": "s1", "hook_event_name": "PreToolUse",
         "tool_name": "mcp__ccd_session__spawn_task", "tool_input": {},
         "transcript_path": str(tmp_path / "t.jsonl")},
        tmp_path, enforce=True,
    )
    hso = out["hookSpecificOutput"]
    assert hso["permissionDecision"] == "deny"
    assert "gh issue create" in hso["permissionDecisionReason"]


def test_spawn_task_allowed_in_shadow_mode(tmp_path):
    seed_state(tmp_path)
    out = run_hook(
        {"session_id": "s1", "hook_event_name": "PreToolUse",
         "tool_name": "mcp__ccd_session__spawn_task", "tool_input": {},
         "transcript_path": str(tmp_path / "t.jsonl")},
        tmp_path, enforce=False,
    )
    assert out.get("hookSpecificOutput", {}).get("permissionDecision") != "deny"


def test_budget_warning_fires_once(tmp_path):
    seed_state(tmp_path, budget_soft=100)
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(json.dumps({
        "type": "assistant",
        "message": {"model": "m", "usage": {"output_tokens": 500}},
    }) + "\n")
    payload = {"session_id": "s1", "hook_event_name": "PreToolUse",
               "tool_name": "Bash", "tool_input": {"command": "ls"},
               "transcript_path": str(transcript)}

    first = run_hook(payload, tmp_path, enforce=False)
    assert "BUDGET" in first["hookSpecificOutput"]["additionalContext"]

    second = run_hook(payload, tmp_path, enforce=False)
    assert "BUDGET" not in second.get("hookSpecificOutput", {}).get(
        "additionalContext", ""
    ), "the warning must not nag"


def test_unknown_session_is_a_no_op(tmp_path):
    out = run_hook(_agent_payload(tmp_path), tmp_path, enforce=True)
    assert out == {}


def _edit_payload(file_path: str, **extra) -> dict:
    payload = {
        "session_id": "s1", "hook_event_name": "PreToolUse",
        "tool_name": "Edit",
        "tool_input": {"file_path": file_path, "old_string": "a", "new_string": "b"},
        "transcript_path": "/dev/null",
    }
    payload.update(extra)
    return payload


def test_main_loop_edit_on_code_file_is_denied_when_enforcing(tmp_path):
    seed_state(tmp_path)
    payload = _edit_payload(str(tmp_path / "foo.py"))
    payload["transcript_path"] = str(tmp_path / "t.jsonl")
    out = run_hook(payload, tmp_path, enforce=True)
    hso = out["hookSpecificOutput"]
    assert hso["permissionDecision"] == "deny"
    assert "delegated" in hso["permissionDecisionReason"]
    persisted = json.loads((tmp_path / "s1.json").read_text())
    assert persisted["main_loop_code_edits"] == 1


def test_main_loop_edit_on_code_file_is_recorded_but_allowed_in_shadow_mode(tmp_path):
    seed_state(tmp_path)
    payload = _edit_payload(str(tmp_path / "foo.py"))
    payload["transcript_path"] = str(tmp_path / "t.jsonl")
    out = run_hook(payload, tmp_path, enforce=False)
    assert "permissionDecision" not in out.get("hookSpecificOutput", {})
    persisted = json.loads((tmp_path / "s1.json").read_text())
    assert persisted["main_loop_code_edits"] == 1


def test_subagent_edit_on_code_file_is_never_denied(tmp_path):
    seed_state(tmp_path)
    payload = _edit_payload(str(tmp_path / "foo.py"), agent_id="abc123")
    payload["transcript_path"] = str(tmp_path / "t.jsonl")
    out = run_hook(payload, tmp_path, enforce=True)
    assert out.get("hookSpecificOutput", {}).get("permissionDecision") != "deny"
    persisted = json.loads((tmp_path / "s1.json").read_text())
    assert persisted["main_loop_code_edits"] == 0


def test_main_loop_write_to_docs_is_not_denied(tmp_path):
    seed_state(tmp_path)
    payload = {
        "session_id": "s1", "hook_event_name": "PreToolUse",
        "tool_name": "Write",
        "tool_input": {"file_path": str(tmp_path / "notes.md"), "content": "x"},
        "transcript_path": str(tmp_path / "t.jsonl"),
    }
    out = run_hook(payload, tmp_path, enforce=True)
    assert out.get("hookSpecificOutput", {}).get("permissionDecision") != "deny"
    persisted = json.loads((tmp_path / "s1.json").read_text())
    assert persisted["main_loop_code_edits"] == 0


def test_main_loop_write_uppercase_extension_is_denied(tmp_path):
    seed_state(tmp_path)
    payload = {
        "session_id": "s1", "hook_event_name": "PreToolUse",
        "tool_name": "Write",
        "tool_input": {"file_path": str(tmp_path / "src/App.TS"), "content": "x"},
        "transcript_path": str(tmp_path / "t.jsonl"),
    }
    out = run_hook(payload, tmp_path, enforce=True)
    hso = out["hookSpecificOutput"]
    assert hso["permissionDecision"] == "deny"


def test_main_loop_edit_with_null_agent_id_is_denied_when_enforcing(tmp_path):
    seed_state(tmp_path)
    payload = _edit_payload(str(tmp_path / "foo.py"), agent_id=None)
    payload["transcript_path"] = str(tmp_path / "t.jsonl")
    out = run_hook(payload, tmp_path, enforce=True)
    hso = out["hookSpecificOutput"]
    assert hso["permissionDecision"] == "deny"
    persisted = json.loads((tmp_path / "s1.json").read_text())
    assert persisted["main_loop_code_edits"] == 1


def test_main_loop_edit_without_file_path_is_not_denied(tmp_path):
    seed_state(tmp_path)
    payload = {
        "session_id": "s1", "hook_event_name": "PreToolUse",
        "tool_name": "Edit",
        "tool_input": {"old_string": "a", "new_string": "b"},
        "transcript_path": str(tmp_path / "t.jsonl"),
    }
    out = run_hook(payload, tmp_path, enforce=True)
    assert out.get("hookSpecificOutput", {}).get("permissionDecision") != "deny"
    persisted = json.loads((tmp_path / "s1.json").read_text())
    assert persisted["main_loop_code_edits"] == 0
