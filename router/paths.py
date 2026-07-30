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
