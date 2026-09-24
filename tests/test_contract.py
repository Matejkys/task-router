from pathlib import Path

from router.classify import UNCLASSIFIED, Classification
from router.config import load_classes, load_scope_policy
from router.contract import render

REPO = Path(__file__).resolve().parents[1]
CLASSES = load_classes(REPO / "config/classes.yaml", None)
POLICY = load_scope_policy(REPO / "config/classes.yaml")


def test_classified_contract_names_class_and_delegation():
    c = Classification("triage", 0.9, "rule:logs")
    text = render(c, CLASSES["triage"], POLICY)
    assert "class=triage" in text
    assert "confidence=0.9" in text
    assert "DELEGATION" in text
    assert "claude-haiku-4-5-20251001" in text
    assert "gh issue create" in text


def test_contract_has_no_budget_line():
    # The budget is enforced (see budget_notice_template in settings.yaml),
    # but is deliberately not advertised up front: telling a model its token
    # ceiling causes premature wrap-up.
    c = Classification("triage", 0.9, "rule:logs")
    text = render(c, CLASSES["triage"], POLICY)
    assert "BUDGET" not in text


def test_unclassified_contract_has_no_mandate_but_keeps_scope_policy():
    c = Classification(UNCLASSIFIED, 0.0, "none")
    text = render(c, None, POLICY)
    assert "class=unclassified" in text
    assert "DELEGATION" not in text
    assert "BUDGET" not in text
    assert "gh issue create" in text


def test_discussion_class_mandates_no_subagent_model():
    c = Classification("discussion", 0.9, "rule:x")
    text = render(c, CLASSES["discussion"], POLICY)
    assert "no delegation" in text
    assert "claude-sonnet-5" not in text


def test_subagents_line_has_no_effort_fragment():
    c = Classification("triage", 0.9, "rule:logs")
    text = render(c, CLASSES["triage"], POLICY)
    assert "model=" in text
    assert "effort=" not in text


def test_contract_is_compact():
    c = Classification("feature", 0.9, "rule:implement")
    text = render(c, CLASSES["feature"], POLICY)
    # The contract is prepended to every prompt; keep it cheap.
    assert len(text) < 1600, f"contract is {len(text)} chars"


def test_brief_contract_has_only_identity_delegation_and_model():
    c = Classification("triage", 0.9, "rule:logs")
    text = render(c, CLASSES["triage"], POLICY, brief=True)
    assert "class=triage" in text
    assert "DELEGATION" in text
    assert "model=claude-haiku-4-5-20251001" in text
    assert "ESCALATION" not in text
    assert "SCOPE" not in text
    assert "BUDGET" not in text


def test_brief_contract_for_unclassified_has_no_delegation():
    c = Classification(UNCLASSIFIED, 0.0, "none")
    text = render(c, None, POLICY, brief=True)
    assert "class=unclassified" in text
    assert "DELEGATION" not in text
    assert "SCOPE" not in text
    assert "No delegation mandate" not in text


def test_brief_contract_is_much_shorter_than_full():
    c = Classification("feature", 0.9, "rule:implement")
    full = render(c, CLASSES["feature"], POLICY)
    brief = render(c, CLASSES["feature"], POLICY, brief=True)
    assert len(brief) < len(full) / 2
