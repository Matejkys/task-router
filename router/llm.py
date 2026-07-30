# router/llm.py
"""Asynchronous class refinement via the Claude CLI.

Measured cost: 5.8-7.0 s per call, because each call starts a new CLI process.
Never call this on a blocking path.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence

from router.classify import UNCLASSIFIED
from router.config import Settings

Runner = Callable[[Sequence[str], int], str]


def build_prompt(prompt: str, class_names: list[str]) -> str:
    options = ", ".join([*class_names, UNCLASSIFIED])
    return (
        "Classify the following work request by the KIND of operation it asks "
        "for. Do not judge how hard it is.\n"
        f"Answer with exactly one of: {options}\n"
        "Answer with the single word and nothing else.\n\n"
        f"REQUEST:\n{prompt}"
    )


def parse_response(text: str, class_names: list[str]) -> str | None:
    lowered = (text or "").lower()
    hits = [n for n in class_names if n.lower() in lowered]
    # Exactly one recognised class, otherwise we learned nothing usable.
    return hits[0] if len(hits) == 1 else None


def _default_runner(argv: Sequence[str], timeout: int) -> str:
    proc = subprocess.run(
        list(argv), capture_output=True, text=True, timeout=timeout, check=True
    )
    return proc.stdout


def refine(
    prompt: str,
    class_names: list[str],
    settings: Settings,
    runner: Runner | None = None,
) -> str | None:
    run = runner or _default_runner
    argv = [
        "claude", "-p", "--model", settings.haiku_model,
        build_prompt(prompt, class_names),
    ]
    try:
        return parse_response(run(argv, settings.haiku_timeout_s), class_names)
    except Exception:  # noqa: BLE001 - refinement is best-effort by design
        return None
