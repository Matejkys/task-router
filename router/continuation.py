# router/continuation.py
"""Follow-up detection.

A short follow-up is not a small task. One 40-character follow-up in the corpus
arrived in a session that had already spent 283K output tokens; classifying it
afresh would have routed it as trivial.
"""

from __future__ import annotations

from router.config import Settings
from router.state import SessionState


def is_continuation(
    prompt: str, state: SessionState | None, settings: Settings
) -> bool:
    if state is None:
        return False
    if len(prompt) > settings.continuation_max_chars:
        return False
    if any(p.search(prompt) for p in settings.new_work_patterns):
        return False
    return state.out_tokens >= settings.continuation_min_out_tokens
