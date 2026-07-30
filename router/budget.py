"""Incremental output-token accounting from a session transcript."""

from __future__ import annotations

import json
from pathlib import Path


def read_increment(transcript_path: Path, offset: int) -> tuple[int, int]:
    """Return (output tokens since `offset`, new offset).

    Only the bytes after `offset` are parsed, because this runs on every tool
    call. A file that shrank (compaction, rewrite) is re-read from the start.
    """
    try:
        size = transcript_path.stat().st_size
        start = 0 if offset > size else offset
        with transcript_path.open("r", errors="replace") as fh:
            fh.seek(start)
            data = fh.read()
    except OSError:
        # Missing, unreadable, or not a regular file. A hook that died here
        # would block the user's work.
        return 0, 0

    tokens = 0
    for line in data.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue  # a partially flushed final line; counted next time
        if rec.get("type") != "assistant":
            continue
        usage = (rec.get("message") or {}).get("usage") or {}
        tokens += usage.get("output_tokens") or 0

    return tokens, size
