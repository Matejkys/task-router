"""Model/effort enforcement for subagent dispatches.

Deliberately dumb and table-driven: this is the only place with hard
enforcement, so it must be trivial to audit. Precedence is user > agent >
contract, because a router that cannot be reasoned with is worse than none.

The Agent/Task tool's `model` parameter takes aliases (its schema enum is
`sonnet | opus | haiku | fable`); class mandates in config/classes.yaml use
canonical ids. Comparisons here are always done canonically, and any rewrite
this module proposes is emitted back in alias form (see `alias_for`) so the
tool's own enum accepts it.
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
    divergence: bool = False


def canonical(model: str | None, settings: Settings) -> str | None:
    """Resolve a dispatched model string to its canonical id.

    An alias (case-insensitive) maps to its canonical id; anything else
    (already canonical, or unrecognised) passes through unchanged.
    """
    if not model:
        return None
    return settings.model_aliases.get(model.strip().lower(), model)


def alias_for(canonical_id: str, settings: Settings) -> str:
    """Reverse lookup: canonical id -> alias the tool's enum accepts.

    Falls back to the canonical id itself when no alias is registered for it.
    """
    reverse = {value: key for key, value in settings.model_aliases.items()}
    return reverse.get(canonical_id, canonical_id)


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


def _field_status(dispatched: str | None, dispatched_canon: str | None, mandated) :
    """Classify one dispatched field against its mandate.

    'fill' when nothing was dispatched (the router filling in the default,
    not a conflict); 'match' when the canonicalized value already equals the
    mandate; 'divergence' when a real, different value was dispatched.
    """
    if not dispatched:
        return "fill"
    if dispatched_canon == mandated:
        return "match"
    return "divergence"


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

    dispatched_model = tool_input.get("model")
    dispatched_effort = tool_input.get("effort")
    model_canon = canonical(dispatched_model, settings)

    model_status = _field_status(dispatched_model, model_canon, spec.sub_model)
    effort_status = _field_status(dispatched_effort, dispatched_effort, spec.sub_effort)

    if model_status == "match" and effort_status == "match":
        return Decision(None, "already compliant", "none")

    updated_input: dict = {"model": alias_for(spec.sub_model, settings)}
    if spec.sub_effort is not None:
        updated_input["effort"] = spec.sub_effort

    divergence = model_status == "divergence" or effort_status == "divergence"

    mandate_desc = spec.sub_model
    if spec.sub_effort is not None:
        mandate_desc += f"/{spec.sub_effort}"
    reason_text = f"routed to {mandate_desc} per class {spec.name}"
    if divergence:
        asked = []
        if model_status == "divergence":
            asked.append(f"model={dispatched_model}")
        if effort_status == "divergence":
            asked.append(f"effort={dispatched_effort}")
        reason_text += f" (agent asked for {', '.join(asked)})"

    return Decision(updated_input, reason_text, "contract", divergence)
