# router/config.py
"""Configuration loading. No value in this package is hardcoded; everything
tunable comes from config/classes.yaml and config/settings.yaml."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

VALID_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


class ConfigError(Exception):
    """Configuration is missing or malformed. Never recovered from silently."""


@dataclass(frozen=True)
class ClassSpec:
    name: str
    patterns: tuple[re.Pattern[str], ...]
    delegation: str
    sub_model: str | None
    sub_effort: str | None
    budget_soft: int
    escalation: str


@dataclass(frozen=True)
class Settings:
    enforce: bool
    confidence_threshold: float
    confidence_single: float
    confidence_multi: float
    confidence_none: float
    refined_confidence: float
    continuation_max_chars: int
    continuation_min_out_tokens: int
    new_work_patterns: tuple[re.Pattern[str], ...]
    state_dir: Path
    telemetry_path: Path
    haiku_model: str
    haiku_timeout_s: int
    user_override_patterns: tuple[re.Pattern[str], ...]
    agent_override_marker: str
    dispatch_tools: frozenset[str]
    chip_tool: str
    chip_reason: str
    route_notice_template: str
    budget_notice_template: str
    min_samples_to_propose: int
    exceeded_ratio_threshold: float
    override_ratio_threshold: float
    code_edit_tools: frozenset[str]
    code_file_extensions: frozenset[str]
    code_edit_deny_reason: str


def _read_yaml(path: Path, required: bool) -> dict:
    if not path.exists():
        if required:
            raise ConfigError(f"config not found: {path}")
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"malformed YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def _require(mapping: dict, key: str, where: str):
    if key not in mapping:
        raise ConfigError(f"missing required key '{key}' in {where}")
    return mapping[key]


def load_classes(base: Path, overlay: Path | None) -> dict[str, ClassSpec]:
    """Load the taxonomy, merging a gitignored local overlay on top.

    The overlay adds patterns to existing classes and may define new ones; it
    never replaces a base class's pattern list. A missing overlay is normal.
    """
    raw = _read_yaml(base, required=True)
    merged: dict[str, dict] = {}
    for name, spec in (_require(raw, "classes", str(base)) or {}).items():
        merged[name] = dict(spec)

    if overlay is not None:
        extra = _read_yaml(overlay, required=False)
        for name, spec in (extra.get("classes") or {}).items():
            if name in merged:
                combined = list(merged[name].get("patterns") or [])
                combined += list(spec.get("patterns") or [])
                merged[name] = {**merged[name], **spec, "patterns": combined}
            else:
                merged[name] = dict(spec)

    out: dict[str, ClassSpec] = {}
    for name, spec in merged.items():
        where = f"class '{name}'"
        effort = spec.get("sub_effort")
        if effort is not None and effort not in VALID_EFFORTS:
            raise ConfigError(
                f"invalid effort '{effort}' in {where}; "
                f"valid: {sorted(VALID_EFFORTS)}"
            )
        patterns = _require(spec, "patterns", where)
        if not patterns:
            raise ConfigError(f"{where} has no patterns")
        try:
            compiled = tuple(re.compile(p) for p in patterns)
        except re.error as exc:
            raise ConfigError(f"bad regex in {where}: {exc}") from exc
        out[name] = ClassSpec(
            name=name,
            patterns=compiled,
            delegation=_require(spec, "delegation", where),
            sub_model=spec.get("sub_model"),
            sub_effort=effort,
            budget_soft=int(_require(spec, "budget_soft", where)),
            escalation=_require(spec, "escalation", where),
        )
    return out


def load_scope_policy(base: Path) -> str:
    return _require(_read_yaml(base, required=True), "scope_policy", str(base))


def _compile_list(raw: dict, key: str, where: str) -> tuple[re.Pattern[str], ...]:
    try:
        return tuple(re.compile(p) for p in _require(raw, key, where))
    except re.error as exc:
        raise ConfigError(f"bad regex in {key}: {exc}") from exc


def load_settings(path: Path) -> Settings:
    raw = _read_yaml(path, required=True)
    where = str(path)
    return Settings(
        enforce=bool(_require(raw, "enforce", where)),
        confidence_threshold=float(_require(raw, "confidence_threshold", where)),
        confidence_single=float(_require(raw, "confidence_single", where)),
        confidence_multi=float(_require(raw, "confidence_multi", where)),
        confidence_none=float(_require(raw, "confidence_none", where)),
        refined_confidence=float(_require(raw, "refined_confidence", where)),
        continuation_max_chars=int(_require(raw, "continuation_max_chars", where)),
        continuation_min_out_tokens=int(
            _require(raw, "continuation_min_out_tokens", where)
        ),
        new_work_patterns=_compile_list(raw, "new_work_patterns", where),
        state_dir=Path(_require(raw, "state_dir", where)).expanduser(),
        telemetry_path=Path(_require(raw, "telemetry_path", where)).expanduser(),
        haiku_model=_require(raw, "haiku_model", where),
        haiku_timeout_s=int(_require(raw, "haiku_timeout_s", where)),
        user_override_patterns=_compile_list(raw, "user_override_patterns", where),
        agent_override_marker=_require(raw, "agent_override_marker", where),
        dispatch_tools=frozenset(_require(raw, "dispatch_tools", where)),
        chip_tool=_require(raw, "chip_tool", where),
        chip_reason=_require(raw, "chip_reason", where),
        route_notice_template=_require(raw, "route_notice_template", where),
        budget_notice_template=_require(raw, "budget_notice_template", where),
        min_samples_to_propose=int(_require(raw, "min_samples_to_propose", where)),
        exceeded_ratio_threshold=float(
            _require(raw, "exceeded_ratio_threshold", where)
        ),
        override_ratio_threshold=float(
            _require(raw, "override_ratio_threshold", where)
        ),
        code_edit_tools=frozenset(_require(raw, "code_edit_tools", where)),
        code_file_extensions=frozenset(
            ext.lower() for ext in _require(raw, "code_file_extensions", where)
        ),
        code_edit_deny_reason=_require(raw, "code_edit_deny_reason", where),
    )
