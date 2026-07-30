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
    assert hso["updatedInput"] == {"model": "claude-sonnet-5", "effort": "medium"}
    assert "claude-sonnet-5" in hso["additionalContext"], "rewrite must be visible"


def test_shadow_mode_logs_but_does_not_rewrite(tmp_path):
    seed_state(tmp_path)
    out = run_hook(_agent_payload(tmp_path), tmp_path, enforce=False)
    hso = out.get("hookSpecificOutput", {})
    assert "updatedInput" not in hso
    assert json.loads((tmp_path / "s1.json").read_text())["overrides"], \
        "shadow mode must still record what it would have done"


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
