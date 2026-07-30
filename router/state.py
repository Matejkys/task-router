# router/state.py
"""Per-session router state, persisted as one JSON file per session."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
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
    path = _path(state_dir, session_id)
    try:
        return SessionState(**json.loads(path.read_text()))
    except (OSError, ValueError, TypeError):
        # Missing, unreadable or corrupt state degrades to "no state"; a hook
        # that died here would block the user's work.
        return None


def save_state(state_dir: Path, state: SessionState) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = _path(state_dir, state.session_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state)))
    tmp.replace(path)  # atomic, so a concurrent read never sees a partial file
