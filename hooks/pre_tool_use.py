#!/usr/bin/env python3
"""PreToolUse: enforce delegation, deny chips, watch the budget.

Three concerns share one hook because they share one state read. Fails open on
any error via hookio.run: exit 0, no stdout, session unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import hookio, paths
from router.budget import read_increment
from router.config import load_classes, load_settings
from router.enforce import canonical, decide
from router.state import load_state, save_state


def main(payload: dict) -> None:
    settings = load_settings(paths.SETTINGS_YAML)
    state_dir = paths.state_dir_override() or settings.state_dir
    state = load_state(state_dir, payload["session_id"])
    if state is None:
        return  # no contract for this session; do nothing at all

    override = paths.enforce_override()
    enforce = settings.enforce if override is None else override
    tool_name = payload.get("tool_name", "")

    if tool_name == settings.chip_tool and enforce:
        hookio.emit(
            "PreToolUse",
            permissionDecision="deny",
            permissionDecisionReason=settings.chip_reason,
        )
        return

    if tool_name in settings.code_edit_tools and not payload.get("agent_id"):
        tool_input = payload.get("tool_input") or {}
        file_path = tool_input.get("file_path") or tool_input.get("notebook_path")
        if file_path:
            ext = Path(file_path).suffix.lower()
            if ext in settings.code_file_extensions:
                state.main_loop_code_edits += 1
                save_state(state_dir, state)
                if enforce:
                    hookio.emit(
                        "PreToolUse",
                        permissionDecision="deny",
                        permissionDecisionReason=settings.code_edit_deny_reason,
                    )
                    return

    notes: list[str] = []
    updated_input = None

    if tool_name in settings.dispatch_tools:
        classes = load_classes(paths.CLASSES_YAML, paths.CLASSES_LOCAL_YAML)
        tool_input = payload.get("tool_input") or {}
        spec = classes.get(state.cls)
        decision = decide(tool_input, spec, state, settings)
        dispatched_canon = canonical(tool_input.get("model"), settings)
        if decision.updated_input is not None:
            if decision.divergence and spec is not None:
                # A fill (agent named no model) is the router doing its
                # default job, not an override - only a real divergence
                # counts toward the override signal `router report` surfaces.
                state.overrides.append({
                    "from": dispatched_canon,
                    "to": spec.sub_model,
                    "precedence": decision.precedence,
                    "enforced": enforce,
                })
            if enforce:
                updated_input = decision.updated_input
                if decision.divergence:
                    notes.append(
                        settings.route_notice_template.format(reason=decision.reason)
                    )
        elif (
            decision.precedence in ("user", "agent")
            and spec is not None
            and spec.sub_model
            and dispatched_canon != spec.sub_model
        ):
            # A real mandate existed and diverged, but a higher-precedence
            # actor (user or agent) won, so the router deferred. Record it
            # anyway: this is the override-abuse signal `router report` is
            # built to surface, and it must fire regardless of enforce mode.
            state.overrides.append({
                "from": dispatched_canon,
                "to": spec.sub_model,
                "precedence": decision.precedence,
                "enforced": False,
            })

    tokens, new_offset = read_increment(
        Path(payload["transcript_path"]), state.transcript_offset
    )
    state.out_tokens += tokens
    state.transcript_offset = new_offset
    if (
        state.budget_soft
        and state.out_tokens > state.budget_soft
        and not state.budget_notified
    ):
        state.budget_notified = True
        notes.append(
            settings.budget_notice_template.format(
                used=state.out_tokens, budget=state.budget_soft, cls=state.cls
            )
        )

    save_state(state_dir, state)
    hookio.emit(
        "PreToolUse",
        updatedInput=updated_input,
        additionalContext="\n".join(notes) if notes else None,
    )


if __name__ == "__main__":
    hookio.run(main)
