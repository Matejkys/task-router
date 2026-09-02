"""Incremental output-token accounting from a session transcript."""

from __future__ import annotations

from pathlib import Path

from router.usage import read_usage


def read_increment(transcript_path: Path, offset: int) -> tuple[int, int]:
    """Return (output tokens since `offset`, new offset).

    A thin wrapper over router.usage.read_usage, which applies exactly the same
    rules: only the bytes after `offset` are parsed (this runs on every tool
    call), a file that shrank (compaction, rewrite) is re-read from the start,
    unparseable lines are skipped, and every `type == "assistant"` record counts
    -- sidechain (subagent) records included, as it always has. A missing or
    unreadable transcript yields (0, 0) rather than killing the hook.
    """
    records, new_offset = read_usage(transcript_path, offset)
    return sum(r.output_tokens for r in records), new_offset
