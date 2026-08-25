"""Shared hook entry-point plumbing.

The fail-open guarantee lives here, implemented and tested once, because a hook
that crashes or prints malformed JSON blocks the user's work.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable

from router import paths


def read_payload() -> dict:
    return json.load(sys.stdin)


def emit(hook_event_name: str, **fields) -> None:
    """Print a hookSpecificOutput envelope, or nothing when there is nothing
    to say. A hook with no output must stay silent rather than emit an empty
    envelope."""
    populated = {k: v for k, v in fields.items() if v is not None}
    if not populated:
        return
    print(json.dumps({
        "hookSpecificOutput": {"hookEventName": hook_event_name, **populated}
    }))


def run(main: Callable[[dict], int | None]) -> None:
    """Run a hook body, never letting it break the session.

    Exits with whatever `main` returns (`refine_class` returns 2 to rewake the
    session), or 0. Any exception is reported on stderr and swallowed with
    exit 0. sys.exit is called outside the try so its SystemExit is not caught.

    Short-circuits to a silent no-op inside the router's own internal
    `claude -p` refinement call (see `paths.is_internal_call`), so that call
    never re-enters the router on its own classification prompt.
    """
    if paths.is_internal_call():
        sys.exit(0)
    try:
        code = main(read_payload())
    except Exception as exc:  # noqa: BLE001 - failing open is mandatory
        print(f"task-router: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(0)
    sys.exit(code or 0)
