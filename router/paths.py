"""Canonical config locations, and the test-only environment overrides."""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLASSES_YAML = REPO / "config/classes.yaml"
CLASSES_LOCAL_YAML = REPO / "config/classes.local.yaml"
SETTINGS_YAML = REPO / "config/settings.yaml"

STATE_DIR_ENV = "TASK_ROUTER_STATE_DIR"


def state_dir_override() -> Path | None:
    raw = os.environ.get(STATE_DIR_ENV)
    return Path(raw) if raw else None


ENFORCE_ENV = "TASK_ROUTER_ENFORCE"


def enforce_override() -> bool | None:
    raw = os.environ.get(ENFORCE_ENV)
    if raw is None:
        return None
    return raw.strip() in {"1", "true", "yes"}


TELEMETRY_ENV = "TASK_ROUTER_TELEMETRY"


def telemetry_override() -> Path | None:
    raw = os.environ.get(TELEMETRY_ENV)
    return Path(raw) if raw else None


HAIKU_FAKE_ENV = "TASK_ROUTER_FAKE_HAIKU"


def haiku_fake_override() -> str | None:
    """Test-only: return a class name instead of spawning the Claude CLI."""
    return os.environ.get(HAIKU_FAKE_ENV) or None


INTERNAL_CALL_ENV = "TASK_ROUTER_INTERNAL_CALL"


def is_internal_call() -> bool:
    """True inside the `claude -p` subprocess router/llm.py spawns for Haiku
    refinement. That subprocess is a normal Claude Code invocation and would
    otherwise re-trigger these same hooks on the router's own classification
    prompt, polluting telemetry with the router talking to itself. Every hook
    checks this via hookio.run and no-ops immediately when it is set."""
    return os.environ.get(INTERNAL_CALL_ENV) == "1"
