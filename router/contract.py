"""Task contract rendering. This text is injected as additionalContext."""

from __future__ import annotations

from router.classify import UNCLASSIFIED, Classification
from router.config import ClassSpec


def render(
    classification: Classification, spec: ClassSpec | None, scope_policy: str
) -> str:
    lines = [
        f"[ROUTER] class={classification.cls} "
        f"confidence={classification.confidence} "
        f"source={classification.source}"
    ]

    if classification.cls == UNCLASSIFIED or spec is None:
        lines.append(
            "No delegation mandate: the rules did not recognise this task kind. "
            "Choose your own approach, but honour the scope policy below."
        )
    else:
        lines.append(f"DELEGATION  {spec.delegation}")
        if spec.sub_model:
            lines.append(f"            subagents: model={spec.sub_model}")
        lines.append(f"BUDGET      soft {spec.budget_soft} output tokens")
        lines.append(f"ESCALATION  {spec.escalation}")

    lines.append(f"SCOPE       {scope_policy}")
    return "\n".join(lines)
