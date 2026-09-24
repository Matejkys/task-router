"""Task contract rendering. This text is injected as additionalContext."""

from __future__ import annotations

from router.classify import UNCLASSIFIED, Classification
from router.config import ClassSpec


def render(
    classification: Classification,
    spec: ClassSpec | None,
    scope_policy: str,
    *,
    brief: bool = False,
) -> str:
    lines = [
        f"[ROUTER] class={classification.cls} "
        f"confidence={classification.confidence} "
        f"source={classification.source}"
    ]

    # brief=True is for a prompt where the class hasn't changed since the last
    # full contract: a current model retains a once-stated instruction, so
    # re-sending ESCALATION/SCOPE/the unclassified sentence on every turn is
    # pure token cost with nothing new to say. Only the identity + delegation
    # + model lines repeat, so the session can still see what class/model it
    # is under without paying for the whole policy again.
    if brief:
        if classification.cls != UNCLASSIFIED and spec is not None:
            lines.append(f"DELEGATION  {spec.delegation}")
            if spec.sub_model:
                lines.append(f"            subagents: model={spec.sub_model}")
        return "\n".join(lines)

    if classification.cls == UNCLASSIFIED or spec is None:
        lines.append(
            "No delegation mandate: the rules did not recognise this task kind. "
            "Choose your own approach, but honour the scope policy below."
        )
    else:
        lines.append(f"DELEGATION  {spec.delegation}")
        if spec.sub_model:
            lines.append(f"            subagents: model={spec.sub_model}")
        # No BUDGET line here deliberately: surfacing a token budget up front
        # causes premature wrap-up (the model treats it as a target to land
        # under rather than a ceiling). The budget is still enforced -- see
        # budget_notice_template in settings.yaml -- it only fires when the
        # soft budget is actually exceeded, not as an up-front expectation.
        lines.append(f"ESCALATION  {spec.escalation}")

    lines.append(f"SCOPE       {scope_policy}")
    return "\n".join(lines)
