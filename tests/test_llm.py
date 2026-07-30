import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from router.config import load_classes, load_settings
from router.llm import build_prompt, parse_response, refine

REPO = Path(__file__).resolve().parents[1]
SETTINGS = load_settings(REPO / "config/settings.yaml")
CLASSES = load_classes(REPO / "config/classes.yaml", None)
NAMES = ["recon", "pr_review", "triage", "mechanical"]
HOOK = REPO / "hooks/refine_class.py"


def test_build_prompt_lists_the_classes_and_demands_one_word():
    p = build_prompt("check the logs", NAMES)
    for name in NAMES:
        assert name in p
    assert "check the logs" in p
    assert "unclassified" in p


def test_parse_response_accepts_a_bare_class_name():
    assert parse_response("triage", NAMES) == "triage"


def test_parse_response_tolerates_chatter():
    assert parse_response("This looks like: pr_review.\n", NAMES) == "pr_review"


def test_parse_response_rejects_an_invented_class():
    assert parse_response("obviously_a_database_thing", NAMES) is None


def test_parse_response_returns_none_when_multiple_classes_match():
    # "triage" and "pr_review" both appear; an ambiguous match teaches us
    # nothing usable, so this must not silently pick either one.
    assert parse_response("could be triage or maybe pr_review", NAMES) is None


def test_refine_uses_the_injected_runner():
    calls = []

    def runner(argv, timeout):
        calls.append((argv, timeout))
        return "mechanical"

    assert refine("bump the version", NAMES, SETTINGS, runner=runner) == "mechanical"
    argv, timeout = calls[0]
    assert SETTINGS.haiku_model in argv
    assert timeout == SETTINGS.haiku_timeout_s


def test_refine_returns_none_when_the_runner_fails():
    def runner(argv, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    assert refine("x", NAMES, SETTINGS, runner=runner) is None


def test_hook_exits_2_with_the_contract_on_stderr(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "s1.json").write_text(json.dumps({
        "session_id": "s1", "cls": "unclassified", "confidence": 0.0,
        "source": "none", "budget_soft": 0, "budget_notified": False,
        "transcript_offset": 0, "out_tokens": 0, "user_override": False,
        "overrides": [],
    }))
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps({"session_id": "s1", "prompt": "review the pull request",
                          "hook_event_name": "UserPromptSubmit"}),
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO),
             "TASK_ROUTER_STATE_DIR": str(state_dir),
             "TASK_ROUTER_FAKE_HAIKU": "pr_review"},
    )
    assert proc.returncode == 2, f"must exit 2 to rewake: {proc.stdout} {proc.stderr}"
    assert "class=pr_review" in proc.stderr
    assert json.loads((state_dir / "s1.json").read_text())["cls"] == "pr_review"


def test_hook_exit_2_persists_confidence_source_and_budget_soft(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "s1.json").write_text(json.dumps({
        "session_id": "s1", "cls": "unclassified", "confidence": 0.0,
        "source": "none", "budget_soft": 0, "budget_notified": False,
        "transcript_offset": 0, "out_tokens": 0, "user_override": False,
        "overrides": [],
    }))
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps({"session_id": "s1", "prompt": "review the pull request",
                          "hook_event_name": "UserPromptSubmit"}),
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO),
             "TASK_ROUTER_STATE_DIR": str(state_dir),
             "TASK_ROUTER_FAKE_HAIKU": "pr_review"},
    )
    assert proc.returncode == 2, f"must exit 2 to rewake: {proc.stdout} {proc.stderr}"
    persisted = json.loads((state_dir / "s1.json").read_text())
    assert persisted["confidence"] == SETTINGS.refined_confidence
    assert persisted["source"] == "haiku"
    assert persisted["budget_soft"] == CLASSES["pr_review"].budget_soft


def test_hook_exits_0_when_already_confident(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "s1.json").write_text(json.dumps({
        "session_id": "s1", "cls": "triage", "confidence": 0.9,
        "source": "rule:logs", "budget_soft": 100, "budget_notified": False,
        "transcript_offset": 0, "out_tokens": 0, "user_override": False,
        "overrides": [],
    }))
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps({"session_id": "s1", "prompt": "check the logs",
                          "hook_event_name": "UserPromptSubmit"}),
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO),
             "TASK_ROUTER_STATE_DIR": str(state_dir),
             "TASK_ROUTER_FAKE_HAIKU": "recon"},
    )
    assert proc.returncode == 0, "no rewake when the rules were already confident"


def test_hook_exits_0_when_haiku_confirms_the_current_class(tmp_path):
    # Confidence stays below the threshold, so this only exercises the
    # *agreement* branch (`cls == state.cls`), not the earlier confidence
    # early-return that test_hook_exits_0_when_already_confident covers.
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "s1.json").write_text(json.dumps({
        "session_id": "s1", "cls": "recon", "confidence": 0.5,
        "source": "rule:find", "budget_soft": 100, "budget_notified": False,
        "transcript_offset": 0, "out_tokens": 0, "user_override": False,
        "overrides": [],
    }))
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps({"session_id": "s1", "prompt": "find all the config files",
                          "hook_event_name": "UserPromptSubmit"}),
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO),
             "TASK_ROUTER_STATE_DIR": str(state_dir),
             "TASK_ROUTER_FAKE_HAIKU": "recon"},
    )
    assert proc.returncode == 0, f"agreement must not rewake: {proc.stdout} {proc.stderr}"
    assert proc.stderr == "", f"hook wrote to stderr on agreement: {proc.stderr}"


def _load_refine_hook_module():
    """Load hooks/refine_class.py as an importable module (name != "__main__",
    so the `hookio.run(main)` guard at the bottom does not fire) instead of
    shelling out to it. This lets a test monkeypatch the module's `refine`
    name directly, which is required to simulate a write landing *during*
    the hook's ~6.5s refinement window deterministically (see below)."""
    spec = importlib.util.spec_from_file_location("_refine_class_under_test", HOOK)
    assert spec is not None and spec.loader is not None, f"cannot load spec for {HOOK}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hook_preserves_a_concurrent_pre_tool_use_write_during_refinement(
    tmp_path, monkeypatch
):
    # refine_class.py is asyncRewake: true - it runs concurrently with the
    # live session for the ~6.5s refine() takes. A pre_tool_use.py hook
    # firing in that window does its own load -> modify -> save of
    # accounting fields (out_tokens, transcript_offset, overrides,
    # budget_notified). Seeding those fields into the state file *before*
    # the hook runs does not exercise the bug: with nothing else touching
    # the file mid-run, "mutate the loaded state and save" and "reload then
    # mutate and save" produce the same result. The write must land strictly
    # between refine_class's initial load_state and its eventual save_state,
    # so this monkeypatches `refine` (the slow step the real ~6.5s is spent
    # in) to perform that write itself as a side effect before returning,
    # then calls main() in-process. Fully deterministic - no real race,
    # thread, or sleep involved.
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state_path = state_dir / "s1.json"
    state_path.write_text(json.dumps({
        "session_id": "s1", "cls": "unclassified", "confidence": 0.0,
        "source": "none", "budget_soft": 0, "budget_notified": False,
        "transcript_offset": 0, "out_tokens": 0, "user_override": False,
        "overrides": [],
    }))

    overrides = [{"from": "opus", "to": "claude-sonnet-5",
                  "precedence": "contract", "enforced": False}]
    concurrent_write = {
        "session_id": "s1", "cls": "unclassified", "confidence": 0.0,
        "source": "none", "budget_soft": 0, "budget_notified": True,
        "transcript_offset": 1234, "out_tokens": 5000, "user_override": False,
        "overrides": overrides,
    }

    def fake_refine(prompt, names, settings):
        # Stand-in for the real ~6.5s call: pre_tool_use.py's own
        # load-modify-save lands here, strictly between refine_class's
        # initial load_state (already done by the time main() calls this)
        # and its eventual save_state (still to come).
        state_path.write_text(json.dumps(concurrent_write))
        return "pr_review"

    monkeypatch.delenv("TASK_ROUTER_FAKE_HAIKU", raising=False)
    monkeypatch.setenv("TASK_ROUTER_STATE_DIR", str(state_dir))

    hook = _load_refine_hook_module()
    monkeypatch.setattr(hook, "refine", fake_refine)

    code = hook.main({"session_id": "s1", "prompt": "review the pull request"})
    assert code == 2, "must exit 2 to rewake"

    persisted = json.loads(state_path.read_text())
    # The new classification must still land.
    assert persisted["cls"] == "pr_review"
    assert persisted["confidence"] == SETTINGS.refined_confidence
    assert persisted["source"] == "haiku"
    assert persisted["budget_soft"] == CLASSES["pr_review"].budget_soft
    # The concurrent pre_tool_use write must survive, not be clobbered back
    # to the pre-refinement snapshot (0 / 0 / [] / False).
    assert persisted["out_tokens"] == 5000
    assert persisted["transcript_offset"] == 1234
    assert persisted["overrides"] == overrides
    assert persisted["budget_notified"] is True


def test_hook_exits_0_when_the_fake_class_is_not_in_the_taxonomy(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "s1.json").write_text(json.dumps({
        "session_id": "s1", "cls": "unclassified", "confidence": 0.0,
        "source": "none", "budget_soft": 0, "budget_notified": False,
        "transcript_offset": 0, "out_tokens": 0, "user_override": False,
        "overrides": [],
    }))
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps({"session_id": "s1", "prompt": "do something",
                          "hook_event_name": "UserPromptSubmit"}),
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO),
             "TASK_ROUTER_STATE_DIR": str(state_dir),
             "TASK_ROUTER_FAKE_HAIKU": "not_a_real_class"},
    )
    assert proc.returncode == 0, f"unknown class must not rewake: {proc.stdout} {proc.stderr}"
    assert proc.stderr == "", f"hook wrote to stderr for an unknown class: {proc.stderr}"
