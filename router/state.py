# router/state.py
"""Per-session router state, persisted as one JSON file per session."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class SessionState:
    session_id: str
    cls: str
    confidence: float
    source: str
    budget_soft: int
    budget_notified: bool = False
    transcript_offset: int = 0
    out_tokens: int = 0
    user_override: bool = False
    overrides: list[dict] = field(default_factory=list)
    main_loop_code_edits: int = 0
    # Per-model token totals (usage.Totals as plain dicts, so state stays JSON).
    # main_usage: non-sidechain records of the main transcript. sub_usage:
    # sidechain records plus every subagent transcript. sub_offsets: byte
    # offset already consumed per subagent transcript path -- the set of those
    # files grows while the session runs, so each is tracked separately.
    main_usage: dict[str, dict] = field(default_factory=dict)
    sub_usage: dict[str, dict] = field(default_factory=dict)
    sub_offsets: dict[str, int] = field(default_factory=dict)


def _path(state_dir: Path, session_id: str) -> Path:
    # session_id comes from the harness, but never let it escape the directory.
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_")
    if safe != session_id:
        # Sanitisation dropped characters, so distinct ids could otherwise
        # collide on one file. Disambiguate with a digest of the raw id.
        digest = hashlib.sha256(session_id.encode()).hexdigest()[:16]
        safe = f"{safe}-{digest}" if safe else digest
    return state_dir / f"{safe}.json"


def load_state(state_dir: Path, session_id: str) -> SessionState | None:
    try:
        path = _path(state_dir, session_id)
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            return None
        # Forward-compatible: a state file written by a newer router may carry
        # keys this version does not know. Dropping them keeps the session's
        # accounting alive; treating them as an error would silently reset it.
        known = {f.name for f in fields(SessionState)}
        return SessionState(**{k: v for k, v in raw.items() if k in known})
    except (OSError, ValueError, TypeError):
        # Missing, unreadable or corrupt state degrades to "no state"; a hook
        # that died here would block the user's work. This covers a session_id
        # that cannot even be encoded into a filename.
        return None


# Deliberately asymmetric with load_state: a failed write must stay loud.
# hookio.run catches this, reports it on stderr and exits 0, so the session
# is unharmed while the failure stays visible. Swallowing it here would be
# worse than crashing: the router would carry on believing it had persisted
# state it had not. Do not wrap this in the same degrade-to-None handling.
def save_state(state_dir: Path, state: SessionState) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = _path(state_dir, state.session_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state)))
    tmp.replace(path)  # atomic, so a concurrent read never sees a partial file
