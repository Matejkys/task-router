import json
import subprocess
import sys
from pathlib import Path

from router.config import load_classes

REPO = Path(__file__).resolve().parents[1]
HOOK = REPO / "hooks/user_prompt_submit.py"
CLASSES = load_classes(REPO / "config/classes.yaml", None)


def run_hook(payload: dict, state_dir: Path, expect_stderr: bool = False) -> dict:
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "TASK_ROUTER_STATE_DIR": str(state_dir),
             "PYTHONPATH": str(REPO)},
    )
    assert proc.returncode == 0, f"hook must never fail: {proc.stderr}"
    if expect_stderr:
        assert proc.stderr != "", "expected a fail-open diagnostic on stderr"
    else:
        assert proc.stderr == "", f"hook wrote a fail-open diagnostic: {proc.stderr}"
    return json.loads(proc.stdout) if proc.stdout.strip() else {}


def test_confident_prompt_gets_a_contract(tmp_path):
    out = run_hook(
        {"session_id": "s1", "hook_event_name": "UserPromptSubmit",
         "prompt": "Resolve issue https://example.com/o/r/issues/1088",
         "transcript_path": str(tmp_path / "t.jsonl")},
        tmp_path,
    )
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "class=resolve" in ctx
    assert (tmp_path / "s1.json").exists()


def test_unknown_prompt_gets_unclassified_contract(tmp_path):
    out = run_hook(
        {"session_id": "s2", "hook_event_name": "UserPromptSubmit",
         "prompt": "hmm", "transcript_path": str(tmp_path / "t.jsonl")},
        tmp_path,
    )
    assert "class=unclassified" in out["hookSpecificOutput"]["additionalContext"]


def test_followup_reuses_the_stored_class(tmp_path):
    first = {"session_id": "s3", "hook_event_name": "UserPromptSubmit",
             "prompt": "Check the service logs for repeating errors",
             "transcript_path": str(tmp_path / "t.jsonl")}
    run_hook(first, tmp_path)
    # Simulate a session that has already produced substantial output.
    state = json.loads((tmp_path / "s3.json").read_text())
    state["out_tokens"] = 500_000
    (tmp_path / "s3.json").write_text(json.dumps(state))

    out = run_hook({**first, "prompt": "yes, that's right"}, tmp_path)
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "class=triage" in ctx
    assert "source=continuation" in ctx


def test_new_task_starts_its_own_budget_at_zero_tokens(tmp_path):
    # Budgets are per-task/per-class (the contract advertises "BUDGET soft N
    # output tokens for class X"). A genuine new task (not a continuation)
    # must not inherit the prior task's cumulative out_tokens - that would
    # measure the new class's budget against lifetime session tokens and
    # trip the budget notice on the very first tool call. The transcript
    # offset, however, is not per-task: it must still carry forward so the
    # transcript is not re-counted from the start.
    state_dir = tmp_path
    (state_dir / "s5.json").write_text(json.dumps({
        "session_id": "s5", "cls": "recon", "confidence": 0.9,
        "source": "rule:find", "budget_soft": CLASSES["recon"].budget_soft,
        "budget_notified": False, "transcript_offset": 777,
        "out_tokens": 200_000, "user_override": False, "overrides": [],
    }))

    # A URL and an issue reference both mark this as new work, so it cannot
    # be mistaken for a continuation regardless of prior out_tokens/length.
    out = run_hook(
        {"session_id": "s5", "hook_event_name": "UserPromptSubmit",
         "prompt": "Resolve issue https://example.com/o/r/issues/2099",
         "transcript_path": str(tmp_path / "t.jsonl")},
        state_dir,
    )
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "class=resolve" in ctx
    assert "source=continuation" not in ctx

    persisted = json.loads((state_dir / "s5.json").read_text())
    assert persisted["cls"] == "resolve"
    assert persisted["out_tokens"] == 0
    assert persisted["budget_soft"] == CLASSES["resolve"].budget_soft
    assert persisted["transcript_offset"] == 777


def test_user_naming_a_model_is_recorded_as_override(tmp_path):
    run_hook(
        {"session_id": "s4", "hook_event_name": "UserPromptSubmit",
         "prompt": "Review https://example.com/o/r/pull/12 but use opus for it",
         "transcript_path": str(tmp_path / "t.jsonl")},
        tmp_path,
    )
    assert json.loads((tmp_path / "s4.json").read_text())["user_override"] is True


def test_garbage_stdin_exits_zero_and_emits_nothing(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(HOOK)], input="{not json",
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "TASK_ROUTER_STATE_DIR": str(tmp_path),
             "PYTHONPATH": str(REPO)},
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""
    assert proc.stderr != "", "expected a fail-open diagnostic on stderr"
