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
        # Computed at most once per prompt (this hook has a sub-100ms
        # budget): the continuation branch above never needs it, and the
        # sticky check below reuses this same read instead of classifying
        # the prompt twice.
        fresh = classify(prompt, classes, settings)
        if prior is not None and _refinement_sticks(prompt, fresh, prior, settings):
            # A Haiku refinement (asyncRewake, see refine_class.py) landed on
            # a previous prompt in this session. Without this branch, this
            # fresh rules read - when it's weak (below confidence_threshold)
            # and the prompt isn't obviously new work - would discard the
            # refinement and fall the class back toward unclassified, which
            # then fires refine_class.py AGAIN, landing a second,
            # contradicting contract mid-turn (observed live). Keep the
            # sticky classification instead; it is the same task, so its
            # budget count must not reset.
            result = Classification(prior.cls, prior.confidence, "sticky")
            state = prior
        else:
            result = fresh
            spec = classes.get(result.cls)
            state = SessionState(
                session_id=session_id,
                cls=result.cls,
                confidence=result.confidence,
                source=result.source,
                budget_soft=spec.budget_soft if spec else 0,
                transcript_offset=prior.transcript_offset if prior else 0,
                # Budgets are per-task/per-class (the contract advertises
                # "BUDGET soft N output tokens for class X"). A genuine new
                # task must start its own count at zero; carrying the prior
                # task's out_tokens forward would measure the new class's
                # budget against lifetime session tokens, tripping the budget
                # notice on the first tool call whenever a task switches to a
                # smaller-budget class.
                out_tokens=0,
                contract_rendered_for=prior.contract_rendered_for
                if prior
                else None,
            )

    state.user_override = any(
        p.search(prompt) for p in settings.user_override_patterns
    )

    spec = classes.get(result.cls) if result.cls != UNCLASSIFIED else None
    # Full contract (incl. SCOPE/ESCALATION) only when this session hasn't
    # already been shown one for this exact class; otherwise the model
    # already has the policy from earlier in the session, so a brief
    # identity/delegation line is enough. See contract.render for why.
    #
    # This is tracked explicitly via contract_rendered_for rather than
    # inferred from "class changed since last prompt": refine_class.py
    # writes a refined class to state directly (it only ever emits brief,
    # asynchronously, on stderr) without going through this comparison, so
    # comparing against prior.cls would see the refined class already
    # sitting in state and wrongly conclude a full contract had been shown
    # for it. contract_rendered_for is only ever set here, so it reflects
    # what was actually rendered, not merely what state.cls says.
    brief = state.contract_rendered_for == result.cls
    if not brief:
        state.contract_rendered_for = result.cls
    save_state(state_dir, state)

    hookio.emit(
        "UserPromptSubmit",
        additionalContext=render(result, spec, policy, brief=brief),
    )


def _refinement_sticks(
    prompt: str, fresh: Classification, prior: SessionState, settings
) -> bool:
    if prior.source != "haiku":
        return False
    if fresh.confidence >= settings.confidence_threshold:
        return False
    if any(p.search(prompt) for p in settings.new_work_patterns):
        return False
    return True


if __name__ == "__main__":
    hookio.run(main)
