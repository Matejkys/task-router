#!/usr/bin/env python3
"""Stop: pair the router's decision with what actually happened."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import hookio, paths, telemetry
from router.budget import read_increment
from router.config import load_settings
from router.state import load_state, save_state


def main(payload: dict) -> None:
    settings = load_settings(paths.SETTINGS_YAML)
    state_dir = paths.state_dir_override() or settings.state_dir
    state = load_state(state_dir, payload["session_id"])
    if state is None:
        return

    tokens, new_offset = read_increment(
        Path(payload["transcript_path"]), state.transcript_offset
    )
    state.out_tokens += tokens
    state.transcript_offset = new_offset
    save_state(state_dir, state)

    telemetry.append(
        paths.telemetry_override() or settings.telemetry_path,
        {
            "session_id": state.session_id,
            "class": state.cls,
            "confidence": state.confidence,
            "source": state.source,
            "budget_soft": state.budget_soft,
            "user_override": state.user_override,
            "overrides": state.overrides,
            "outcome": {
                "out_tokens": state.out_tokens,
                "exceeded": bool(
                    state.budget_soft and state.out_tokens > state.budget_soft
                ),
                "budget_notified": state.budget_notified,
            },
        },
    )


if __name__ == "__main__":
    hookio.run(main)
