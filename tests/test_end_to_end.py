import json
import subprocess
import sys
from pathlib import Path

from router.telemetry import read_all

REPO = Path(__file__).resolve().parents[1]


def _run(hook: str, payload: dict, env: dict) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [sys.executable, str(REPO / "hooks" / hook)],
        input=json.dumps(payload), capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO), **env},
    )
    assert proc.returncode in (0, 2), f"{hook}: {proc.stderr}"
    # A hook that crashes still exits 0 (hookio.run fails open) with empty
    # stdout, so asserting on returncode/stdout alone cannot tell a broken
    # hook from a legitimate no-op. stderr is the tell: hookio.run prints the
    # exception there before swallowing it. See Task 8/9 for the same gap.
    assert proc.stderr == "", f"{hook} wrote a fail-open diagnostic: {proc.stderr}"
    return proc


def _stdout_json(proc: subprocess.CompletedProcess) -> dict:
    # hookio.emit stays silent (empty stdout) instead of printing an empty
    # envelope when a hook has nothing to say - e.g. pre_tool_use.py in
    # shadow mode when no budget notice is due yet. That is the documented
    # success path, not a failure, so it must not be handed to json.loads
    # unguarded (see tests/test_hook_pre_tool_use.py's run_hook for the same
    # guard against the same legitimate case).
    return json.loads(proc.stdout) if proc.stdout.strip() else {}


def test_full_cycle_in_shadow_mode(tmp_path):
    state_dir = tmp_path / "state"
    telemetry = tmp_path / "telemetry.jsonl"
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(json.dumps({
        "type": "assistant",
        "message": {"model": "m", "usage": {"output_tokens": 900}},
    }) + "\n")
    env = {
        "TASK_ROUTER_STATE_DIR": str(state_dir),
        "TASK_ROUTER_TELEMETRY": str(telemetry),
        "TASK_ROUTER_ENFORCE": "0",
    }
    sid = "e2e"

    submit = _run("user_prompt_submit.py", {
        "session_id": sid, "hook_event_name": "UserPromptSubmit",
        "prompt": "Please review https://example.com/o/r/pull/42",
        "transcript_path": str(transcript),
    }, env)
    assert "class=pr_review" in _stdout_json(submit)[
        "hookSpecificOutput"]["additionalContext"]

    pre = _run("pre_tool_use.py", {
        "session_id": sid, "hook_event_name": "PreToolUse",
        "tool_name": "Agent",
        "tool_input": {"model": "opus", "prompt": "review it"},
        "transcript_path": str(transcript),
    }, env)
    hso = _stdout_json(pre).get("hookSpecificOutput", {})
    assert "updatedInput" not in hso, "shadow mode must not rewrite"

    _run("stop.py", {
        "session_id": sid, "hook_event_name": "Stop",
        "transcript_path": str(transcript),
    }, env)

    rows = read_all(telemetry)
    assert len(rows) == 1
    assert rows[0]["class"] == "pr_review"
    assert rows[0]["overrides"][0]["enforced"] is False
    assert rows[0]["overrides"][0]["to"] == "claude-sonnet-5"


def test_enforcing_the_same_cycle_rewrites(tmp_path):
    state_dir = tmp_path / "state"
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("")
    env = {
        "TASK_ROUTER_STATE_DIR": str(state_dir),
        "TASK_ROUTER_TELEMETRY": str(tmp_path / "tel.jsonl"),
        "TASK_ROUTER_ENFORCE": "1",
    }
    _run("user_prompt_submit.py", {
        "session_id": "e2e2", "hook_event_name": "UserPromptSubmit",
        "prompt": "Please review https://example.com/o/r/pull/42",
        "transcript_path": str(transcript),
    }, env)
    pre = _run("pre_tool_use.py", {
        "session_id": "e2e2", "hook_event_name": "PreToolUse",
        "tool_name": "Agent",
        "tool_input": {"model": "opus", "prompt": "review it"},
        "transcript_path": str(transcript),
    }, env)
    hso = _stdout_json(pre)["hookSpecificOutput"]
    assert hso["updatedInput"]["model"] == "sonnet"
    assert "claude-sonnet-5" in hso["additionalContext"]
