from pathlib import Path
from typing import Any

from router.config import ClassSpec, load_settings
from router.enforce import decide
from router.state import SessionState

REPO = Path(__file__).resolve().parents[1]
SETTINGS = load_settings(REPO / "config/settings.yaml")
SPEC = ClassSpec(
    name="pr_review", patterns=(), delegation="d",
    sub_model="claude-sonnet-5", sub_effort="medium",
    budget_soft=200_000, escalation="e",
)


def _state(**kw) -> SessionState:
    defaults: dict[str, Any] = {
        "session_id": "s",
        "cls": "pr_review",
        "confidence": 0.9,
        "source": "rule:x",
        "budget_soft": 200_000,
    }
    defaults.update(kw)
    return SessionState(**defaults)


def test_opus_call_is_rewritten_to_the_mandated_model():
    tool_input = {
        "description": "Implement X", "prompt": "review it",
        "subagent_type": "general-purpose", "model": "opus",
    }
    d = decide(tool_input, SPEC, _state(), SETTINGS)
    assert d.precedence == "contract"
    assert d.updated_input == {
        "description": "Implement X", "prompt": "review it",
        "subagent_type": "general-purpose", "model": "sonnet",
    }
    assert d.updated_input is not None and "effort" not in d.updated_input
    assert "pr_review" in d.reason
    assert d.divergence is True


def test_call_already_compliant_is_left_alone():
    d = decide(
        {"model": "claude-sonnet-5", "effort": "medium"}, SPEC, _state(), SETTINGS
    )
    assert d.updated_input is None
    assert d.precedence == "none"


def test_alias_compliant_call_is_left_alone():
    d = decide({"model": "sonnet", "effort": "medium"}, SPEC, _state(), SETTINGS)
    assert d.updated_input is None
    assert d.precedence == "none"
    assert d.divergence is False


def test_alias_lookup_is_case_insensitive():
    d = decide({"model": "Sonnet", "effort": "medium"}, SPEC, _state(), SETTINGS)
    assert d.updated_input is None
    assert d.precedence == "none"


def test_model_compliant_with_no_effort_key_is_fully_compliant():
    # Regression test: Agent has no `effort` parameter, so a dispatch that
    # never carries the key must not be treated as a fill needing a rewrite.
    tool_input = {
        "description": "Implement X", "prompt": "review it",
        "subagent_type": "general-purpose", "model": "sonnet",
    }
    d = decide(tool_input, SPEC, _state(), SETTINGS)
    assert d.updated_input is None
    assert d.precedence == "none"
    assert d.divergence is False


def test_no_model_dispatched_is_a_fill_not_a_divergence():
    tool_input = {"description": "Implement X", "prompt": "review it"}
    d = decide(tool_input, SPEC, _state(), SETTINGS)
    assert d.updated_input == {
        "description": "Implement X", "prompt": "review it", "model": "sonnet",
    }
    assert d.updated_input is not None and "effort" not in d.updated_input
    assert d.divergence is False


def test_wrong_model_is_a_divergence_recorded_canonically():
    tool_input = {"description": "Implement X", "prompt": "review it", "model": "opus"}
    d = decide(tool_input, SPEC, _state(), SETTINGS)
    assert d.updated_input == {
        "description": "Implement X", "prompt": "review it", "model": "sonnet",
    }
    assert d.updated_input is not None and "effort" not in d.updated_input
    assert d.divergence is True


def test_effort_key_present_and_diverging_still_rewrites():
    tool_input = {
        "description": "Implement X", "prompt": "review it",
        "model": "sonnet", "effort": "high",
    }
    d = decide(tool_input, SPEC, _state(), SETTINGS)
    assert d.updated_input == {
        "description": "Implement X", "prompt": "review it",
        "model": "sonnet", "effort": "medium",
    }
    assert d.divergence is True


def test_effort_key_present_and_matching_is_compliant():
    tool_input = {
        "description": "Implement X", "prompt": "review it",
        "model": "sonnet", "effort": "medium",
    }
    d = decide(tool_input, SPEC, _state(), SETTINGS)
    assert d.updated_input is None
    assert d.precedence == "none"


def test_unknown_model_string_passes_through_and_counts_as_divergence():
    tool_input = {"description": "Implement X", "prompt": "review it", "model": "gpt-9"}
    d = decide(tool_input, SPEC, _state(), SETTINGS)
    assert d.divergence is True
    assert d.updated_input == {
        "description": "Implement X", "prompt": "review it", "model": "sonnet",
    }
    assert d.updated_input is not None and "effort" not in d.updated_input


def test_user_override_wins_over_the_contract():
    d = decide({"model": "opus"}, SPEC, _state(user_override=True), SETTINGS)
    assert d.updated_input is None
    assert d.precedence == "user"


def test_agent_override_with_a_reason_is_respected():
    d = decide(
        {"model": "opus", "prompt": "model-override: needs cross-file reasoning"},
        SPEC, _state(), SETTINGS,
    )
    assert d.updated_input is None
    assert d.precedence == "agent"
    assert "cross-file reasoning" in d.reason


def test_agent_override_without_a_reason_is_not_respected():
    d = decide({"model": "opus", "prompt": "model-override:"}, SPEC, _state(), SETTINGS)
    assert d.precedence == "contract"
    assert d.updated_input is not None


def test_no_spec_means_no_mandate():
    d = decide({"model": "opus"}, None, _state(cls="unclassified"), SETTINGS)
    assert d.updated_input is None
    assert d.precedence == "none"


def test_class_without_a_sub_model_mandates_nothing():
    spec = ClassSpec("discussion", (), "no delegation", None, None, 50_000, "e")
    d = decide({"model": "opus"}, spec, _state(cls="discussion"), SETTINGS)
    assert d.updated_input is None
