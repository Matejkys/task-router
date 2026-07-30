#!/usr/bin/env python3
"""UserPromptSubmit: classify the task and inject the contract.

Fast path only - rules, no network. Must stay under 100 ms and must never
block or break the session. On any failure it exits 0 with empty stdout, and
the session proceeds exactly as it would without the router.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import hookio, paths
from router.classify import UNCLASSIFIED, Classification, classify
from router.config import load_classes, load_scope_policy, load_settings
from router.continuation import is_continuation
from router.contract import render
from router.state import SessionState, load_state, save_state


def main(payload: dict) -> None:
    session_id = payload["session_id"]
    prompt = payload.get("prompt") or ""

    settings = load_settings(paths.SETTINGS_YAML)
    classes = load_classes(paths.CLASSES_YAML, paths.CLASSES_LOCAL_YAML)
    policy = load_scope_policy(paths.CLASSES_YAML)
    state_dir = paths.state_dir_override() or settings.state_dir

    prior = load_state(state_dir, session_id)

    if prior is not None and is_continuation(prompt, prior, settings):
        result = Classification(prior.cls, prior.confidence, "continuation")
        state = prior
    else:
        result = classify(prompt, classes, settings)
        spec = classes.get(result.cls)
        state = SessionState(
            session_id=session_id,
            cls=result.cls,
            confidence=result.confidence,
            source=result.source,
            budget_soft=spec.budget_soft if spec else 0,
            transcript_offset=prior.transcript_offset if prior else 0,
            out_tokens=prior.out_tokens if prior else 0,
        )

    state.user_override = any(
        p.search(prompt) for p in settings.user_override_patterns
    )
    save_state(state_dir, state)

    spec = classes.get(result.cls) if result.cls != UNCLASSIFIED else None
    hookio.emit(
        "UserPromptSubmit", additionalContext=render(result, spec, policy)
    )


if __name__ == "__main__":
    hookio.run(main)
