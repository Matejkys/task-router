import json
import subprocess
import sys
from pathlib import Path

from router.config import load_classes, load_settings

REPO = Path(__file__).resolve().parents[1]
HOOK = REPO / "hooks/user_prompt_submit.py"
CLASSES = load_classes(REPO / "config/classes.yaml", None)
SETTINGS = load_settings(REPO / "config/settings.yaml")


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


def test_haiku_prior_survives_a_low_confidence_followup(tmp_path):
    # Simulate refine_class.py (asyncRewake) having already landed a Haiku
    # refinement in state before this prompt runs.
    state_dir = tmp_path
    (state_dir / "s6.json").write_text(json.dumps({
        "session_id": "s6", "cls": "pr_review", "confidence": 0.8,
        "source": "haiku", "budget_soft": CLASSES["pr_review"].budget_soft,
        "budget_notified": False, "transcript_offset": 0, "out_tokens": 12_000,
        "user_override": False, "overrides": [],
    }))

    # Longer than continuation_max_chars so is_continuation does not claim
    # this first (out_tokens alone would otherwise make it a continuation),
    # but with no rule hit strong enough to reclassify confidently, and no
    # new_work_patterns marker.
    prompt = (
        "ok, thinking about it more, I'm not totally sure the naming on that "
        "helper reads well to someone new to the module, could you take another "
        "look at it with fresh eyes and see whether it still makes sense"
    )
    assert len(prompt) > 200
    out = run_hook(
        {"session_id": "s6", "hook_event_name": "UserPromptSubmit",
         "prompt": prompt, "transcript_path": str(tmp_path / "t.jsonl")},
        state_dir,
    )
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "class=pr_review" in ctx
    assert "source=sticky" in ctx

    persisted = json.loads((state_dir / "s6.json").read_text())
    assert persisted["cls"] == "pr_review"
    assert persisted["source"] == "haiku"  # persisted marker stays, only the
    # emitted Classification.source becomes "sticky"
    assert persisted["out_tokens"] == 12_000, "sticky keeps the task's budget count"
    assert persisted["budget_soft"] == CLASSES["pr_review"].budget_soft


def test_new_work_pattern_breaks_stickiness_and_resets_budget(tmp_path):
    state_dir = tmp_path
    (state_dir / "s7.json").write_text(json.dumps({
        "session_id": "s7", "cls": "pr_review", "confidence": 0.8,
        "source": "haiku", "budget_soft": CLASSES["pr_review"].budget_soft,
        "budget_notified": False, "transcript_offset": 0, "out_tokens": 12_000,
        "user_override": False, "overrides": [],
    }))

    out = run_hook(
        {"session_id": "s7", "hook_event_name": "UserPromptSubmit",
         "prompt": "Resolve issue https://example.com/o/r/issues/42",
         "transcript_path": str(tmp_path / "t.jsonl")},
        state_dir,
    )
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "class=resolve" in ctx
    assert "source=sticky" not in ctx

    persisted = json.loads((state_dir / "s7.json").read_text())
    assert persisted["cls"] == "resolve"
    assert persisted["out_tokens"] == 0, "new work resets the budget count"


def test_high_confidence_rules_hit_overrides_a_haiku_prior(tmp_path):
    state_dir = tmp_path
    (state_dir / "s8.json").write_text(json.dumps({
        "session_id": "s8", "cls": "triage", "confidence": 0.8,
        "source": "haiku", "budget_soft": CLASSES["triage"].budget_soft,
        "budget_notified": False, "transcript_offset": 0, "out_tokens": 12_000,
        "user_override": False, "overrides": [],
    }))

    # Matches the "feature" pattern confidently (confidence_single >=
    # confidence_threshold) - genuine new work, not a low-confidence read
    # that should defer to the prior. Long enough (>200 chars) to keep
    # is_continuation from claiming it first via the prior's out_tokens.
    prompt = (
        "Please implement and automate the export pipeline end to end, "
        "including the retry logic and the notification step once it "
        "finishes, and make sure it covers both the nightly and the "
        "on-demand trigger paths"
    )
    assert len(prompt) > 200
    out = run_hook(
        {"session_id": "s8", "hook_event_name": "UserPromptSubmit",
         "prompt": prompt, "transcript_path": str(tmp_path / "t.jsonl")},
        state_dir,
    )
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "class=feature" in ctx
    assert "source=sticky" not in ctx

    persisted = json.loads((state_dir / "s8.json").read_text())
    assert persisted["cls"] == "feature"
    assert persisted["out_tokens"] == 0


def test_full_contract_on_first_prompt_brief_when_class_unchanged(tmp_path):
    payload = {"session_id": "s9", "hook_event_name": "UserPromptSubmit",
               "prompt": "Resolve issue https://example.com/o/r/issues/7",
               "transcript_path": str(tmp_path / "t.jsonl")}
    first = run_hook(payload, tmp_path)
    first_ctx = first["hookSpecificOutput"]["additionalContext"]
    assert "SCOPE" in first_ctx, "first prompt in a session gets the full contract"

    second = run_hook(
        {**payload, "prompt": "also resolve issue https://example.com/o/r/issues/8"},
        tmp_path,
    )
    second_ctx = second["hookSpecificOutput"]["additionalContext"]
    assert "class=resolve" in second_ctx
    assert "SCOPE" not in second_ctx, "same class as prior -> brief contract"


def test_haiku_refined_class_gets_a_full_contract_exactly_once(tmp_path):
    # Pins the fix for: unclassified -> haiku-refined mid-session -> the next
    # prompt must render a FULL contract for the refined class (it has never
    # been shown one), and only that once - the prompt after must be brief.
    # Inferring "full vs brief" from "did the class change since prior.cls"
    # gets this wrong, because refine_class.py writes the refined class
    # straight into state (async, off the beaten path this hook takes), so
    # by the time this hook next runs, prior.cls already equals the new
    # class even though this session was never actually shown its contract.
    state_dir = tmp_path
    first = run_hook(
        {"session_id": "s10", "hook_event_name": "UserPromptSubmit",
         "prompt": "hmm, not sure what I want yet",
         "transcript_path": str(tmp_path / "t.jsonl")},
        state_dir,
    )
    first_ctx = first["hookSpecificOutput"]["additionalContext"]
    assert "class=unclassified" in first_ctx
    assert "SCOPE" in first_ctx, "first prompt in a session gets the full contract"
    assert (state_dir / "s10.json").read_text()
    persisted = json.loads((state_dir / "s10.json").read_text())
    assert persisted["contract_rendered_for"] == "unclassified"

    # Simulate refine_class.py's write: it only ever sets classification
    # fields (cls/confidence/source/budget_soft), never contract_rendered_for.
    persisted["cls"] = "recon"
    persisted["confidence"] = SETTINGS.refined_confidence
    persisted["source"] = "haiku"
    persisted["budget_soft"] = CLASSES["recon"].budget_soft
    (state_dir / "s10.json").write_text(json.dumps(persisted))

    # Long enough to skip is_continuation, weak enough (no rule hit) to stay
    # under confidence_threshold, and no new_work_patterns marker - the
    # sticky path, which is exactly the path a refined class takes on its
    # first post-refinement prompt.
    followup = (
        "yeah go ahead, take your time on it and let me know how it turns "
        "out once you've had a chance to look everything over, there's no "
        "rush at all so take whatever time you need on this before getting "
        "back to me"
    )
    assert len(followup) > 200

    second = run_hook(
        {"session_id": "s10", "hook_event_name": "UserPromptSubmit",
         "prompt": followup, "transcript_path": str(tmp_path / "t.jsonl")},
        state_dir,
    )
    second_ctx = second["hookSpecificOutput"]["additionalContext"]
    assert "class=recon" in second_ctx
    assert "source=sticky" in second_ctx
    assert "SCOPE" in second_ctx, "refined class must get its own full contract"
    assert "ESCALATION" in second_ctx

    persisted2 = json.loads((state_dir / "s10.json").read_text())
    assert persisted2["contract_rendered_for"] == "recon"

    third = run_hook(
        {"session_id": "s10", "hook_event_name": "UserPromptSubmit",
         "prompt": followup, "transcript_path": str(tmp_path / "t.jsonl")},
        state_dir,
    )
    third_ctx = third["hookSpecificOutput"]["additionalContext"]
    assert "class=recon" in third_ctx
    assert "SCOPE" not in third_ctx, "same class already shown -> brief"


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
