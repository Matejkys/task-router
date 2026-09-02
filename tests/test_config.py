# tests/test_config.py
from pathlib import Path

import pytest

from router.config import ConfigError, load_classes, load_settings

REPO = Path(__file__).resolve().parents[1]


def test_load_classes_compiles_patterns():
    classes = load_classes(REPO / "config/classes.yaml", None)
    assert "triage" in classes
    spec = classes["triage"]
    assert spec.name == "triage"
    assert spec.sub_model == "claude-haiku-4-5-20251001"
    assert spec.sub_effort == "low"
    assert spec.budget_soft == 100_000
    assert any(p.search("check the logs for errors") for p in spec.patterns)


def test_missing_overlay_is_not_an_error():
    classes = load_classes(REPO / "config/classes.yaml", REPO / "config/nope.yaml")
    assert "triage" in classes


def test_overlay_merges_patterns_into_existing_class(tmp_path):
    overlay = tmp_path / "local.yaml"
    overlay.write_text(
        "version: 1\nclasses:\n  triage:\n    patterns:\n      - 'example-overlay-pattern'\n"
    )
    classes = load_classes(REPO / "config/classes.yaml", overlay)
    joined = [p.pattern for p in classes["triage"].patterns]
    assert "example-overlay-pattern" in joined
    assert len(joined) > 1, "overlay must add to base patterns, not replace them"


def test_missing_base_config_fails_loudly(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_classes(tmp_path / "absent.yaml", None)


def test_invalid_effort_is_rejected(tmp_path):
    bad = tmp_path / "classes.yaml"
    bad.write_text(
        "version: 1\nscope_policy: x\nclasses:\n  c:\n    patterns: ['a']\n"
        "    delegation: d\n    sub_model: claude-sonnet-5\n    sub_effort: turbo\n"
        "    budget_soft: 1\n    escalation: e\n"
    )
    with pytest.raises(ConfigError, match="effort"):
        load_classes(bad, None)


def test_settings_expands_user_paths():
    s = load_settings(REPO / "config/settings.yaml")
    assert s.enforce is False
    assert s.confidence_threshold == 0.7
    assert "~" not in str(s.state_dir)
    assert s.haiku_model == "claude-haiku-4-5-20251001"


def test_settings_carries_every_tunable():
    # The Global Constraints forbid constants in router/*.py, so anything the
    # code would otherwise hardcode has to arrive from here.
    s = load_settings(REPO / "config/settings.yaml")
    assert (s.confidence_single, s.confidence_multi, s.confidence_none) == (
        0.9, 0.6, 0.0,
    )
    assert s.refined_confidence == 0.8
    assert s.dispatch_tools == frozenset({"Agent", "Task"})
    assert s.chip_tool == "mcp__ccd_session__spawn_task"
    assert "gh issue create" in s.chip_reason
    assert s.min_samples_to_propose == 5
    assert s.exceeded_ratio_threshold == 0.5
    assert s.override_ratio_threshold == 0.5
    assert any(p.search("see https://example.com") for p in s.new_work_patterns)
    assert s.code_edit_tools == frozenset({"Edit", "Write", "NotebookEdit"})
    assert ".py" in s.code_file_extensions
    assert all(
        ext.startswith(".") and ext == ext.lower() for ext in s.code_file_extensions
    )
    assert "delegated" in s.code_edit_deny_reason
    assert s.model_aliases == {
        "opus": "claude-opus-5",
        "sonnet": "claude-sonnet-5",
        "haiku": "claude-haiku-4-5-20251001",
        "fable": "claude-fable-5-1",
    }
    assert all(alias == alias.lower() for alias in s.model_aliases)


def test_missing_required_setting_fails_loudly(tmp_path):
    partial = tmp_path / "settings.yaml"
    partial.write_text("version: 1\nenforce: false\n")
    with pytest.raises(ConfigError, match="confidence_threshold"):
        load_settings(partial)
