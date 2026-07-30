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
    d = decide({"model": "opus", "prompt": "review it"}, SPEC, _state(), SETTINGS)
    assert d.precedence == "contract"
    assert d.updated_input == {"model": "claude-sonnet-5", "effort": "medium"}
    assert "pr_review" in d.reason


def test_call_already_compliant_is_left_alone():
    d = decide(
        {"model": "claude-sonnet-5", "effort": "medium"}, SPEC, _state(), SETTINGS
    )
    assert d.updated_input is None
    assert d.precedence == "none"


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
