#!/usr/bin/env python3
"""UserPromptSubmit (asyncRewake): refine a low-confidence classification.

Runs in the background so its measured ~6.5 s never blocks a prompt. On a
successful refinement it exits 2 with the new contract on stderr, which the
harness delivers to the session as a system reminder. Any other outcome exits 0
and the session keeps the unclassified contract it already has.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import hookio, paths
from router.classify import Classification
from router.config import load_classes, load_scope_policy, load_settings
from router.contract import render
from router.llm import refine
from router.state import load_state, save_state


def main(payload: dict) -> int:
    settings = load_settings(paths.SETTINGS_YAML)
    state_dir = paths.state_dir_override() or settings.state_dir
    state = load_state(state_dir, payload["session_id"])
    if state is None or state.confidence >= settings.confidence_threshold:
        return 0

    classes = load_classes(paths.CLASSES_YAML, paths.CLASSES_LOCAL_YAML)
    names = list(classes)

    fake = paths.haiku_fake_override()
    if fake:
        cls = fake if fake in names else None
    else:
        cls = refine(payload.get("prompt") or "", names, settings)

    if cls is None or cls == state.cls:
        return 0

    spec = classes[cls]
    # This hook is asyncRewake: true, so it runs concurrently with the live
    # session for the ~6.5s refine() takes. A pre_tool_use.py hook firing in
    # that window does its own load->modify->save of accounting fields
    # (out_tokens, transcript_offset, overrides, budget_notified). Saving the
    # stale `state` we loaded at the top would silently clobber that write
    # (lost update). Re-loading immediately before saving shrinks the race
    # window from ~6.5s to microseconds and applies only the classification
    # fields on top of whatever is freshest on disk.
    fresh = load_state(state_dir, payload["session_id"]) or state
    fresh.cls = cls
    fresh.confidence = settings.refined_confidence
    fresh.source = "haiku"
    fresh.budget_soft = spec.budget_soft
    save_state(state_dir, fresh)

    contract = render(
        Classification(cls, settings.refined_confidence, "haiku"),
        spec,
        load_scope_policy(paths.CLASSES_YAML),
    )
    # stderr, not stdout: exit 2 makes the harness deliver this as a reminder.
    print(f"Router refined this task's classification:\n{contract}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    hookio.run(main)
