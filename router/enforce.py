"""Model/effort enforcement for subagent dispatches.

Deliberately dumb and table-driven: this is the only place with hard
enforcement, so it must be trivial to audit. Precedence is user > agent >
contract, because a router that cannot be reasoned with is worse than none.
"""

from __future__ import annotations

from dataclasses import dataclass

from router.config import ClassSpec, Settings
from router.state import SessionState


@dataclass(frozen=True)
class Decision:
    updated_input: dict | None
    reason: str
    precedence: str


def _agent_override_reason(tool_input: dict, marker: str) -> str | None:
    for field in ("prompt", "description", "label"):
        value = tool_input.get(field)
        if not isinstance(value, str) or marker not in value:
            continue
        reason = value.split(marker, 1)[1].strip()
        # A bare marker is not a justification.
        if reason:
            return reason.splitlines()[0].strip()
    return None


def decide(
    tool_input: dict,
    spec: ClassSpec | None,
    state: SessionState,
    settings: Settings,
) -> Decision:
    if state.user_override:
        return Decision(None, "user named the model explicitly", "user")

    reason = _agent_override_reason(tool_input, settings.agent_override_marker)
    if reason:
        return Decision(None, f"agent override: {reason}", "agent")

    if spec is None or not spec.sub_model:
        return Decision(None, "no delegation mandate for this class", "none")

    wanted = {"model": spec.sub_model, "effort": spec.sub_effort}
    current = {"model": tool_input.get("model"), "effort": tool_input.get("effort")}
    if current == wanted:
        return Decision(None, "already compliant", "none")

    return Decision(
        wanted,
        f"routed to {spec.sub_model}/{spec.sub_effort} per class {spec.name}",
        "contract",
    )
