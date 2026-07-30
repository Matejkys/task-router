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
