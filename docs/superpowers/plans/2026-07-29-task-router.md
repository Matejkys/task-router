# Task Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify each incoming task by kind of operation, inject a task contract that mandates a delegation plan, and enforce subagent model/effort — so sessions take the direct path instead of keeping mechanical work on the main loop.

**Architecture:** Four thin hook entry points over a pure-Python library. A synchronous `UserPromptSubmit` hook classifies by regex rules (<100 ms) and injects the contract; a second `UserPromptSubmit` hook with `asyncRewake: true` refines low-confidence classifications in the background via Haiku. A `PreToolUse` hook rewrites `model`/`effort` on `Agent`/`Task` calls, denies `spawn_task`, and watches the output-token budget. A `Stop` hook closes telemetry. Everything tunable lives in `config/`; nothing session-derived is ever committed.

**Tech Stack:** Python 3.14, uv, pytest, PyYAML. No network dependency on the fast path. Haiku reached via `claude -p --model claude-haiku-4-5-20251001` (no `ANTHROPIC_API_KEY` on this machine — the CLI uses the existing login).

## Global Constraints

- **Zero hardcoded values.** Every threshold, model id, effort level, budget, path, tool name, and user-facing message lives in `config/classes.yaml` or `config/settings.yaml` and reaches the code through `Settings` or `ClassSpec`. A literal in `router/*.py` is a defect if changing it would change behaviour the user might want to tune. Two things are *not* tunables and stay in code: the sentinel `UNCLASSIFIED = "unclassified"` (an identifier other modules compare against, not a setting) and the `TASK_ROUTER_*` environment-variable names in `paths.py` (the test-override interface).
- **All four hooks use `router.hookio.run`.** The fail-open guarantee is implemented and tested once, not copied per hook.
- **No silent defaults.** A missing or malformed base config is a hard error with a clear message. A missing `config/classes.local.yaml` overlay is normal and must not error.
- **The repository is public.** Never commit anything derived from real sessions: `analysis/sessions.json`, `tests/fixtures/golden.local.jsonl`, `config/classes.local.yaml`, `state/`, `telemetry.jsonl`, `reports/`.
- **A hook must never emit malformed JSON and never crash the session.** Every hook entry point wraps its body in a catch-all that, on any exception, writes a diagnostic to stderr and exits 0 emitting no stdout. Failing open is mandatory.
- **The fast path must never block a prompt.** Rules classification budget is 100 ms, asserted by test.
- **Valid effort levels** are exactly `low`, `medium`, `high`, `xhigh`, `max`.
- **Model ids** are `claude-haiku-4-5-20251001`, `claude-sonnet-5`, `claude-opus-5`.
- **Shadow mode is the default.** `settings.yaml` ships `enforce: false`. Contract injection and budget notices are on; model rewrites and `spawn_task` denial are off.
- Python is invoked as `uv run python` inside `tools/task-router`; tests as `uv run pytest`.

---

### Task 1: Project scaffold and config loader

**Files:**
- Create: `pyproject.toml`
- Create: `config/classes.yaml`
- Create: `config/settings.yaml`
- Create: `router/__init__.py`
- Create: `router/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ClassSpec` (frozen dataclass: `name: str`, `patterns: tuple[re.Pattern, ...]`, `delegation: str`, `sub_model: str | None`, `sub_effort: str | None`, `budget_soft: int`, `escalation: str`); `Settings` (frozen dataclass — every tunable in the system, since the Global Constraints forbid constants in `router/*.py`: `enforce`, `confidence_threshold`, `confidence_single`, `confidence_multi`, `confidence_none`, `refined_confidence`, `continuation_max_chars`, `continuation_min_out_tokens`, `new_work_patterns`, `state_dir`, `telemetry_path`, `haiku_model`, `haiku_timeout_s`, `user_override_patterns`, `agent_override_marker`, `dispatch_tools`, `chip_tool`, `chip_reason`, `route_notice_template`, `budget_notice_template`, `min_samples_to_propose`, `exceeded_ratio_threshold`, `override_ratio_threshold` — types as written in the code below); `load_classes(base: Path, overlay: Path | None) -> dict[str, ClassSpec]`; `load_settings(path: Path) -> Settings`; `load_scope_policy(base: Path) -> str`; `ConfigError(Exception)`.

- [ ] **Step 1: Create the uv project**

```bash
cd /Users/matejkys/Development/tools/task-router
uv init --name task-router --python 3.14 --no-workspace --bare
uv add pyyaml
uv add --dev pytest
```

- [ ] **Step 2: Write `config/classes.yaml`**

Budgets are measured medians and p75 from the corpus; see the spec. Patterns here are generic on purpose — internal vocabulary goes in the gitignored overlay.

```yaml
# Task taxonomy. Internal vocabulary belongs in classes.local.yaml (gitignored).
version: 1

classes:
  recon:
    patterns:
      - '(?i)\b(find|locate|list all|search for)\b'
      - '(?i)\b(explore|inspect|describe)\b.*\b(backup|dump|export|archive)\b'
    delegation: >-
      all work in subagents: extraction on the cheap model, synthesis on the mid
      model; main loop dispatches and summarises only
    sub_model: claude-haiku-4-5-20251001
    sub_effort: low
    budget_soft: 150000
    escalation: >-
      if extraction is inconclusive, move synthesis to the mid model before
      widening the search

  pr_review:
    patterns:
      - '(?i)\breview\b.*\b(pr|pull[- ]request)\b'
      - '(?i)\bpull/\d+'
    delegation: >-
      mid-model reviewer subagent; main loop writes the final verdict only
    sub_model: claude-sonnet-5
    sub_effort: medium
    budget_soft: 200000
    escalation: >-
      escalate to the strong model only for a finding the reviewer marks
      architectural

  triage:
    patterns:
      - '(?i)\b(logs?|error|exception|stack ?trace|alert)\b.*\b(check|repeating|reported|investigate)\b'
      - '(?i)\b(check|why)\b.*\b(logs?|errors?|failing)\b'
    delegation: >-
      phase 1: cheap model reads the logs and answers exactly one question - is
      this a known error or a systemic problem? main loop does not read logs
    sub_model: claude-haiku-4-5-20251001
    sub_effort: low
    budget_soft: 100000
    escalation: >-
      open phase 2 (strong-model deep dive) ONLY with an evidenced systemic
      cause from phase 1

  mechanical:
    patterns:
      - '(?i)\b(bump|rename|remove|delete|drop|add)\b.*\b(column|field|config|configs|flag)\b'
      - '(?i)\b(release|redeploy|rollout|migrate)\b'
    delegation: >-
      mid model, parallel pipeline when there are N items; main loop verifies
      the first item then delegates the rest
    sub_model: claude-sonnet-5
    sub_effort: low
    budget_soft: 400000
    escalation: >-
      if an item fails in a way the others will share, stop the pipeline and
      report rather than retrying each

  resolve:
    patterns:
      - '(?i)\bresolve\b.*\bissue'
      - '(?i)\b(fix|finish)\b.*\bissues?/\d+'
      - '(?i)\bissues?/\d+'
    delegation: >-
      mid model implements, main loop acts as review gate before commit
    sub_model: claude-sonnet-5
    sub_effort: medium
    budget_soft: 500000
    escalation: >-
      if the issue turns out to need a design decision, stop and present
      options instead of choosing silently

  feature:
    patterns:
      - '(?i)\b(implement|build|automate)\b'
      - '(?i)\bphase [a-z0-9]+\b.*\b(backend|frontend)\b'
    delegation: >-
      main loop orchestrates and plans, mid model builds phase by phase; each
      phase ends with tests passing before the next starts
    sub_model: claude-sonnet-5
    sub_effort: high
    budget_soft: 2000000
    escalation: >-
      re-plan rather than widen scope; a new requirement discovered mid-build
      becomes a separate phase

  investigate:
    patterns:
      - '(?i)\b(why did|why does|figure out why|root cause)\b'
      - '(?i)\blooks like a bug\b'
    delegation: >-
      main loop reasons, subagents gather evidence in parallel
    sub_model: claude-sonnet-5
    sub_effort: high
    budget_soft: 300000
    escalation: >-
      at the soft budget STOP and present what is known and what you propose
      next - do not keep searching

  discussion:
    patterns:
      - "(?i)\\b(let's discuss|what do you think|should we|options for)\\b"
    delegation: "no delegation, no code changes"
    sub_model: null
    sub_effort: null
    budget_soft: 50000
    escalation: "if the discussion converges on work, reclassify before starting it"

scope_policy: >-
  A newly discovered problem that is RELATED to this task: fix it without asking
  for confirmation. A problem that is UNRELATED: create a GitHub issue with
  `gh issue create` and move on. Do not open a background-task chip and do not
  start a new session.
```

- [ ] **Step 3: Write `config/settings.yaml`**

```yaml
version: 1

# Shadow mode: contract injection and budget notices on, rewrites off.
enforce: false

# Below this confidence the async Haiku refinement fires.
confidence_threshold: 0.7

# Confidence the rules classifier assigns. One class matching is a clean read;
# several matching means the prompt is ambiguous and invites refinement.
confidence_single: 0.9
confidence_multi: 0.6
confidence_none: 0.0

# Confidence assigned when Haiku refines a classification.
refined_confidence: 0.8

# A prompt shorter than this, arriving in a session that has already produced at
# least continuation_min_out_tokens of output, with no new URL or issue
# reference, is treated as a follow-up rather than a new task.
continuation_max_chars: 200
continuation_min_out_tokens: 5000

# Markers that mean "this is new work", not a follow-up.
new_work_patterns:
  - 'https?://'
  - '\bissues?/\d+'
  - '\bpull/\d+'
  - '#\d{2,}'

state_dir: ~/.claude/task-router/state
telemetry_path: ~/.claude/task-router/telemetry.jsonl

haiku_model: claude-haiku-4-5-20251001
haiku_timeout_s: 30

# The user explicitly demanding a model outranks the contract.
user_override_patterns:
  - '(?i)\b(use|run|put (this|it) on)\s+(opus|sonnet|haiku)\b'
  - '(?i)\bmodel\s*[:=]\s*(opus|sonnet|haiku)'

# How a subagent dispatch declares a justified override.
agent_override_marker: 'model-override:'

# Tool names the router acts on.
dispatch_tools:
  - Agent
  - Task
chip_tool: mcp__ccd_session__spawn_task
chip_reason: >-
  Router: unrelated findings do not become background-task chips. File it with
  `gh issue create` and continue the current task.

# Nudges injected back into the session. Reword these without touching Python.
route_notice_template: 'ROUTER: {reason}'
budget_notice_template: >-
  BUDGET: {used} of {budget} output tokens for class {cls}. Stop and present
  what you have and what you propose next, rather than continuing.

# router report: how much evidence before proposing a change, and how often a
# budget must be blown or a mandate overridden before it counts as wrong.
min_samples_to_propose: 5
exceeded_ratio_threshold: 0.5
override_ratio_threshold: 0.5
```

- [ ] **Step 4: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path

import pytest

from router.config import ConfigError, load_classes, load_settings

REPO = Path(__file__).resolve().parents[1]


def test_load_classes_compiles_patterns():
    classes = load_classes(REPO / "config/classes.yaml", None)
    assert "triage" in classes
    spec = classes["triage"]
    assert spec.name == "triage"
    assert spec.sub_model == "claude-haiku-4-5-20251001"
    assert spec.sub_effort == "low"
    assert spec.budget_soft == 100_000
    assert any(p.search("check the logs for errors") for p in spec.patterns)


def test_missing_overlay_is_not_an_error():
    classes = load_classes(REPO / "config/classes.yaml", REPO / "config/nope.yaml")
    assert "triage" in classes


def test_overlay_merges_patterns_into_existing_class(tmp_path):
    overlay = tmp_path / "local.yaml"
    overlay.write_text(
        "version: 1\nclasses:\n  triage:\n    patterns:\n      - 'datadoghq'\n"
    )
    classes = load_classes(REPO / "config/classes.yaml", overlay)
    joined = [p.pattern for p in classes["triage"].patterns]
    assert "datadoghq" in joined
    assert len(joined) > 1, "overlay must add to base patterns, not replace them"


def test_missing_base_config_fails_loudly(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_classes(tmp_path / "absent.yaml", None)


def test_invalid_effort_is_rejected(tmp_path):
    bad = tmp_path / "classes.yaml"
    bad.write_text(
        "version: 1\nscope_policy: x\nclasses:\n  c:\n    patterns: ['a']\n"
        "    delegation: d\n    sub_model: claude-sonnet-5\n    sub_effort: turbo\n"
        "    budget_soft: 1\n    escalation: e\n"
    )
    with pytest.raises(ConfigError, match="effort"):
        load_classes(bad, None)


def test_settings_expands_user_paths():
    s = load_settings(REPO / "config/settings.yaml")
    assert s.enforce is False
    assert s.confidence_threshold == 0.7
    assert "~" not in str(s.state_dir)
    assert s.haiku_model == "claude-haiku-4-5-20251001"


def test_settings_carries_every_tunable():
    # The Global Constraints forbid constants in router/*.py, so anything the
    # code would otherwise hardcode has to arrive from here.
    s = load_settings(REPO / "config/settings.yaml")
    assert (s.confidence_single, s.confidence_multi, s.confidence_none) == (
        0.9, 0.6, 0.0,
    )
    assert s.refined_confidence == 0.8
    assert s.dispatch_tools == frozenset({"Agent", "Task"})
    assert s.chip_tool == "mcp__ccd_session__spawn_task"
    assert "gh issue create" in s.chip_reason
    assert s.min_samples_to_propose == 5
    assert s.exceeded_ratio_threshold == 0.5
    assert s.override_ratio_threshold == 0.5
    assert any(p.search("see https://example.com") for p in s.new_work_patterns)


def test_missing_required_setting_fails_loudly(tmp_path):
    partial = tmp_path / "settings.yaml"
    partial.write_text("version: 1\nenforce: false\n")
    with pytest.raises(ConfigError, match="confidence_threshold"):
        load_settings(partial)
```

- [ ] **Step 5: Run test to verify it fails**

Run: `cd /Users/matejkys/Development/tools/task-router && uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router'`

- [ ] **Step 6: Write `router/__init__.py` and `router/config.py`**

```python
# router/__init__.py
```

```python
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
    )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS, 8 passed

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock config/ router/ tests/test_config.py
git commit -m "feat: config loader with fail-fast validation and local overlay"
```

---

### Task 2: Rules classifier and golden set

**Files:**
- Create: `router/classify.py`
- Create: `analysis/build_golden.py`
- Create: `tests/fixtures/golden.example.jsonl`
- Test: `tests/test_classify.py`

**Interfaces:**
- Consumes: `ClassSpec`, `Settings`, `load_classes`, `load_settings` from Task 1.
- Produces: `Classification` (frozen dataclass: `cls: str`, `confidence: float`, `source: str`); `classify(prompt: str, classes: dict[str, ClassSpec], settings: Settings) -> Classification`; `UNCLASSIFIED: str = "unclassified"`.

Confidence is deliberately explainable: exactly one class matching scores `settings.confidence_single`; several matching scores `settings.confidence_multi` and the class with the most matching patterns wins (ties broken by config order); nothing matching scores `settings.confidence_none` and yields `UNCLASSIFIED`. At the shipped values (0.9 / 0.6 / 0.0 against `confidence_threshold: 0.7`) only a clean single match skips refinement. The scores are config, not constants — the Global Constraints forbid them in code.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_classify.py
import json
import time
from pathlib import Path

from router.classify import UNCLASSIFIED, classify
from router.config import load_classes, load_settings

REPO = Path(__file__).resolve().parents[1]
CLASSES = load_classes(REPO / "config/classes.yaml", None)
SETTINGS = load_settings(REPO / "config/settings.yaml")

# Floors differ because the two fixtures are different instruments: the example
# set deliberately includes prompts the shipped rules miss, the local set is the
# hand-labelled corpus whose measured baseline was 87.5%.
LOCAL_FLOOR = 0.85
EXAMPLE_FLOOR = 0.70


def test_single_match_is_confident():
    r = classify(
        "Resolve issue https://example.com/org/repo/issues/1088", CLASSES, SETTINGS
    )
    assert r.cls == "resolve"
    assert r.confidence == SETTINGS.confidence_single
    assert r.source.startswith("rule:")


def test_no_match_is_unclassified():
    r = classify("hmm", CLASSES, SETTINGS)
    assert r.cls == UNCLASSIFIED
    assert r.confidence == SETTINGS.confidence_none
    assert r.source == "none"


def test_ambiguous_prompt_scores_below_threshold():
    # Mentions both a release (mechanical) and implementing (feature).
    r = classify("Implement the release automation for the deploy", CLASSES, SETTINGS)
    assert r.confidence == SETTINGS.confidence_multi
    assert r.confidence < SETTINGS.confidence_threshold, "must invite refinement"
    assert r.cls in {"mechanical", "feature"}


def test_review_of_a_pull_request_is_pr_review():
    r = classify(
        "Please review https://example.com/org/repo/pull/1056", CLASSES, SETTINGS
    )
    assert r.cls == "pr_review"


def test_golden_set_coverage_does_not_regress():
    local = REPO / "tests/fixtures/golden.local.jsonl"
    example = REPO / "tests/fixtures/golden.example.jsonl"
    using_local = local.exists()
    path = local if using_local else example
    floor = LOCAL_FLOOR if using_local else EXAMPLE_FLOOR

    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert rows, f"empty fixture: {path}"
    misses = [
        (r["prompt"], r["class"], classify(r["prompt"], CLASSES, SETTINGS).cls)
        for r in rows
        if classify(r["prompt"], CLASSES, SETTINGS).cls != r["class"]
    ]
    coverage = (len(rows) - len(misses)) / len(rows)
    print(f"\ngolden fixture: {path.name}  coverage={coverage:.1%} "
          f"({len(rows) - len(misses)}/{len(rows)})")
    for prompt, expected, got in misses:
        print(f"  MISS want={expected:12} got={got:12} {prompt[:60]}")
    assert coverage >= floor, (
        f"coverage {coverage:.1%} below floor {floor:.0%} using {path.name}"
    )


def test_rules_path_is_under_100ms():
    prompt = "Resolve issue https://example.com/org/repo/issues/1088 " * 20
    start = time.perf_counter()
    for _ in range(100):
        classify(prompt, CLASSES, SETTINGS)
    per_call_ms = (time.perf_counter() - start) / 100 * 1000
    assert per_call_ms < 100, f"{per_call_ms:.1f} ms per call"
```

**The fixture is ground truth, not output.** If the coverage test fails, the fix is
never to edit a fixture label or to bolt a regex onto `classes.yaml` so that one
row passes. Report the misses and stop. A fixture tuned to the classifier tests
nothing, which is exactly the defect this test replaced.

- [ ] **Step 2: Write `tests/fixtures/golden.example.jsonl`**

Paraphrased, committed so the suite runs on a fresh clone. The real fixture is built in Step 5 and gitignored.

The last five rows are **deliberately hard**: hand-labelled by what the request
actually asks for, not by what the shipped regexes happen to return. Several of
them are expected to miss, which is why `EXAMPLE_FLOOR` is 0.70 rather than 1.0.
A fixture that only contains cases the rules already pass would assert nothing.

```jsonl
{"prompt": "Resolve issue https://example.com/org/repo/issues/1088", "class": "resolve"}
{"prompt": "Resolve these two issues by a single PR: https://example.com/org/repo/issues/1104", "class": "resolve"}
{"prompt": "Please review https://example.com/org/repo/pull/1056 before I publish", "class": "pr_review"}
{"prompt": "Describe this PR and then review it: https://example.com/org/repo/pull/1067", "class": "pr_review"}
{"prompt": "Check the service logs for repeating errors", "class": "triage"}
{"prompt": "Why does the deploy keep failing, check logs", "class": "triage"}
{"prompt": "Implement Phase A backend for the merge feature", "class": "feature"}
{"prompt": "I want to automate the redeploy after each release", "class": "mechanical"}
{"prompt": "Remove the unused columns from all the configs", "class": "mechanical"}
{"prompt": "Find the order forms in the backup archive", "class": "recon"}
{"prompt": "Why did twelve records get reassigned to one user last Friday", "class": "investigate"}
{"prompt": "Let's discuss how we should handle the API situation", "class": "discussion"}
{"prompt": "Take a look at pull/902 - it only touches fixtures", "class": "pr_review"}
{"prompt": "The nightly job silently stopped producing rows three days ago", "class": "investigate"}
{"prompt": "Bump the pinned dependency and make sure the suite still passes", "class": "mechanical"}
{"prompt": "We should probably split the orders module before adding more to it", "class": "discussion"}
{"prompt": "Find out why the export endpoint returns 500 for large accounts", "class": "investigate"}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_classify.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router.classify'`

- [ ] **Step 4: Write `router/classify.py`**

```python
# router/classify.py
"""Rule-based task classification.

Task difficulty is NOT predictable from prompt text (measured r=+0.24 for
length, 36-125x variance within a category). What is predictable is the kind of
operation, which is all this classifies.
"""

from __future__ import annotations

from dataclasses import dataclass

from router.config import ClassSpec, Settings

UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class Classification:
    cls: str
    confidence: float
    source: str


def classify(
    prompt: str, classes: dict[str, ClassSpec], settings: Settings
) -> Classification:
    matched: dict[str, list[str]] = {}
    for name, spec in classes.items():
        hits = [p.pattern for p in spec.patterns if p.search(prompt)]
        if hits:
            matched[name] = hits

    if not matched:
        return Classification(UNCLASSIFIED, settings.confidence_none, "none")

    # Most matching patterns wins; config order breaks ties because dicts
    # preserve insertion order and load_classes preserves YAML order.
    winner = max(matched, key=lambda n: len(matched[n]))
    confidence = (
        settings.confidence_single
        if len(matched) == 1
        else settings.confidence_multi
    )
    return Classification(winner, confidence, f"rule:{matched[winner][0]}")
```

- [ ] **Step 5: Write `analysis/build_golden.py`**

```python
# analysis/build_golden.py
"""Build the local golden fixture from the aggregated corpus.

Output is gitignored: it contains real prompt text. Run agg_sessions.py first.
"""

from __future__ import annotations

import json
from pathlib import Path

from router.classify import classify
from router.config import load_classes, load_settings

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "analysis/sessions.json"
DST = REPO / "tests/fixtures/golden.local.jsonl"


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} missing - run: uv run python analysis/agg_sessions.py 14")
    classes = load_classes(REPO / "config/classes.yaml", REPO / "config/classes.local.yaml")
    settings = load_settings(REPO / "config/settings.yaml")
    rows = json.loads(SRC.read_text())
    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("w") as fh:
        for row in rows:
            prompt = row["first_prompt"]
            guess = classify(prompt, classes, settings)
            fh.write(json.dumps({"prompt": prompt, "class": guess.cls}) + "\n")
    print(f"wrote {len(rows)} rows to {DST}")
    print("REVIEW EVERY LINE BY HAND: these are the classifier's own guesses, "
          "not labels. Correct the 'class' values before trusting the test.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_classify.py -v -s`
Expected: PASS, 6 passed. The output prints `golden fixture: golden.example.jsonl coverage=…` followed by a `MISS` line per hard case the rules do not catch. **Record the measured coverage and every MISS line in your report** — that number is the baseline everything later is compared against. Do not tune regexes or fixture labels to raise it.

- [ ] **Step 7: Commit**

```bash
git add router/classify.py analysis/build_golden.py tests/fixtures/golden.example.jsonl tests/test_classify.py
git commit -m "feat: rules classifier with explainable confidence and golden-set regression test"
```

---

### Task 3: Session state and continuation detection

**Files:**
- Create: `router/state.py`
- Create: `router/continuation.py`
- Test: `tests/test_state.py`

**Interfaces:**
- Consumes: `Settings` from Task 1.
- Produces: `SessionState` (mutable dataclass: `session_id: str`, `cls: str`, `confidence: float`, `source: str`, `budget_soft: int`, `budget_notified: bool`, `transcript_offset: int`, `out_tokens: int`, `user_override: bool`, `overrides: list[dict]`); `load_state(state_dir: Path, session_id: str) -> SessionState | None`; `save_state(state_dir: Path, state: SessionState) -> None`; `is_continuation(prompt: str, state: SessionState | None, settings: Settings) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_state.py
from pathlib import Path

from router.config import load_settings
from router.continuation import is_continuation
from router.state import SessionState, load_state, save_state

REPO = Path(__file__).resolve().parents[1]
SETTINGS = load_settings(REPO / "config/settings.yaml")


def _state(**kw) -> SessionState:
    base = dict(
        session_id="s1", cls="triage", confidence=0.9, source="rule:x",
        budget_soft=100_000, budget_notified=False, transcript_offset=0,
        out_tokens=0, user_override=False, overrides=[],
    )
    return SessionState(**{**base, **kw})


def test_roundtrip(tmp_path):
    s = _state(out_tokens=1234, overrides=[{"from": "a", "to": "b"}])
    save_state(tmp_path, s)
    loaded = load_state(tmp_path, "s1")
    assert loaded is not None
    assert loaded.out_tokens == 1234
    assert loaded.cls == "triage"
    assert loaded.overrides == [{"from": "a", "to": "b"}]


def test_missing_state_is_none(tmp_path):
    assert load_state(tmp_path, "nope") is None


def test_corrupt_state_is_none_not_a_crash(tmp_path):
    (tmp_path / "s1.json").write_text("{not json")
    assert load_state(tmp_path, "s1") is None


def test_wellformed_session_id_keeps_its_own_name(tmp_path):
    save_state(tmp_path, _state(session_id="a1b2-c3d4_e5"))
    assert (tmp_path / "a1b2-c3d4_e5.json").exists()


def test_traversal_cannot_escape_the_state_dir(tmp_path):
    save_state(tmp_path, _state(session_id="../../../../etc/passwd"))
    written = list(tmp_path.iterdir())
    assert len(written) == 1
    assert written[0].parent.resolve() == tmp_path.resolve()


def test_ids_differing_only_in_stripped_characters_do_not_collide(tmp_path):
    save_state(tmp_path, _state(session_id="a/b", out_tokens=1))
    save_state(tmp_path, _state(session_id="a:b", out_tokens=2))
    first = load_state(tmp_path, "a/b")
    second = load_state(tmp_path, "a:b")
    assert first is not None and first.out_tokens == 1
    assert second is not None and second.out_tokens == 2


def test_empty_session_id_does_not_collide_with_another_stripped_id(tmp_path):
    save_state(tmp_path, _state(session_id="", out_tokens=7))
    save_state(tmp_path, _state(session_id="///", out_tokens=9))
    empty = load_state(tmp_path, "")
    slashes = load_state(tmp_path, "///")
    assert empty is not None and empty.out_tokens == 7
    assert slashes is not None and slashes.out_tokens == 9


def test_unreadable_state_degrades_to_none(tmp_path):
    (tmp_path / "s1.json").mkdir()  # read_text on a directory raises OSError
    assert load_state(tmp_path, "s1") is None


def test_unencodable_session_id_degrades_to_none(tmp_path):
    # A lone surrogate cannot be UTF-8 encoded; _path must not crash the hook.
    assert load_state(tmp_path, "sess-\ud800-bad") is None


def test_short_followup_in_a_progressed_session_is_continuation():
    st = _state(out_tokens=SETTINGS.continuation_min_out_tokens)
    assert is_continuation("that's fine, keep going", st, SETTINGS) is True


def test_no_prior_state_is_never_continuation():
    assert is_continuation("keep going", None, SETTINGS) is False


def test_barely_started_session_is_a_new_task():
    st = _state(out_tokens=SETTINGS.continuation_min_out_tokens - 1)
    assert is_continuation("keep going", st, SETTINGS) is False


def test_long_prompt_is_a_new_task():
    st = _state(out_tokens=999_999)
    assert is_continuation("x" * 500, st, SETTINGS) is False


def test_new_url_means_new_task_even_if_short():
    st = _state(out_tokens=999_999)
    assert is_continuation("fix https://example.com/a/b/issues/9", st, SETTINGS) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router.state'`

- [ ] **Step 3: Write `router/state.py`**

```python
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
    try:
        path = _path(state_dir, session_id)
        return SessionState(**json.loads(path.read_text()))
    except (OSError, ValueError, TypeError):
        # Missing, unreadable or corrupt state degrades to "no state"; a hook
        # that died here would block the user's work. This covers a session_id
        # that cannot even be encoded into a filename.
        return None


def save_state(state_dir: Path, state: SessionState) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = _path(state_dir, state.session_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state)))
    tmp.replace(path)  # atomic, so a concurrent read never sees a partial file
```

- [ ] **Step 4: Write `router/continuation.py`**

```python
# router/continuation.py
"""Follow-up detection.

A short follow-up is not a small task. One 40-character follow-up in the corpus
arrived in a session that had already spent 283K output tokens; classifying it
afresh would have routed it as trivial.
"""

from __future__ import annotations

from router.config import Settings
from router.state import SessionState


def is_continuation(
    prompt: str, state: SessionState | None, settings: Settings
) -> bool:
    if state is None:
        return False
    if len(prompt) > settings.continuation_max_chars:
        return False
    if any(p.search(prompt) for p in settings.new_work_patterns):
        return False
    return state.out_tokens >= settings.continuation_min_out_tokens
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_state.py -v`
Expected: PASS, 14 passed

- [ ] **Step 6: Commit**

```bash
git add router/state.py router/continuation.py tests/test_state.py
git commit -m "feat: session state persistence and follow-up detection"
```

---

### Task 4: Contract rendering

**Files:**
- Create: `router/contract.py`
- Test: `tests/test_contract.py`

**Interfaces:**
- Consumes: `Classification` (Task 2), `ClassSpec`, `load_scope_policy` (Task 1).
- Produces: `render(classification: Classification, spec: ClassSpec | None, scope_policy: str) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contract.py
from pathlib import Path

from router.classify import UNCLASSIFIED, Classification
from router.config import load_classes, load_scope_policy
from router.contract import render

REPO = Path(__file__).resolve().parents[1]
CLASSES = load_classes(REPO / "config/classes.yaml", None)
POLICY = load_scope_policy(REPO / "config/classes.yaml")


def test_classified_contract_names_class_delegation_and_budget():
    c = Classification("triage", 0.9, "rule:logs")
    text = render(c, CLASSES["triage"], POLICY)
    assert "class=triage" in text
    assert "confidence=0.9" in text
    assert "DELEGATION" in text
    assert "claude-haiku-4-5-20251001" in text
    assert "100000" in text or "100K" in text
    assert "gh issue create" in text


def test_unclassified_contract_has_no_mandate_but_keeps_scope_policy():
    c = Classification(UNCLASSIFIED, 0.0, "none")
    text = render(c, None, POLICY)
    assert "class=unclassified" in text
    assert "DELEGATION" not in text
    assert "BUDGET" not in text
    assert "gh issue create" in text


def test_discussion_class_mandates_no_subagent_model():
    c = Classification("discussion", 0.9, "rule:x")
    text = render(c, CLASSES["discussion"], POLICY)
    assert "no delegation" in text
    assert "claude-sonnet-5" not in text


def test_contract_is_compact():
    c = Classification("feature", 0.9, "rule:implement")
    text = render(c, CLASSES["feature"], POLICY)
    # The contract is prepended to every prompt; keep it cheap.
    assert len(text) < 1600, f"contract is {len(text)} chars"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router.contract'`

- [ ] **Step 3: Write `router/contract.py`**

```python
# router/contract.py
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
            lines.append(
                f"            subagents: model={spec.sub_model} "
                f"effort={spec.sub_effort}"
            )
        lines.append(f"BUDGET      soft {spec.budget_soft} output tokens")
        lines.append(f"ESCALATION  {spec.escalation}")

    lines.append(f"SCOPE       {scope_policy}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_contract.py -v`
Expected: PASS, 4 passed

- [ ] **Step 5: Commit**

```bash
git add router/contract.py tests/test_contract.py
git commit -m "feat: task contract rendering"
```

---

### Task 5: UserPromptSubmit hook

**Files:**
- Create: `router/paths.py`
- Create: `router/hookio.py`
- Create: `hooks/user_prompt_submit.py`
- Test: `tests/test_hookio.py`
- Test: `tests/test_hook_prompt_submit.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces:
  - `router/paths.py` exposing `REPO: Path`, `CLASSES_YAML: Path`, `CLASSES_LOCAL_YAML: Path`, `SETTINGS_YAML: Path`, `state_dir_override() -> Path | None` (test-only env override, extended with `enforce_override()` in Task 8 and `telemetry_override()` in Task 9).
  - `router/hookio.py` exposing `read_payload() -> dict`, `emit(hook_event_name: str, **fields) -> None`, `run(main: Callable[[dict], int | None]) -> None`. **All four hooks in this plan use `hookio.run`** — the fail-open guarantee is implemented and tested once here instead of copied into each hook, because a hook that crashes or prints malformed JSON blocks the user's work.
  - `hooks/user_prompt_submit.py`, an executable hook writing `{"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": ...}}` on stdout.

Each hook keeps a three-line `sys.path` preamble so it runs as a standalone script; that preamble is deliberately not abstracted, since it must execute *before* `router` is importable.

Hook input fields used: `session_id` and `prompt` only. This hook deliberately does not read `transcript_path` — token accounting belongs to `pre_tool_use.py` and `stop.py`, and a new classification carries the offset forward from prior state (or 0 for a fresh session).

- [ ] **Step 1: Write the failing test for `hookio`**

```python
# tests/test_hookio.py
import io
import json

import pytest

from router import hookio


def test_emit_prints_the_envelope(capsys):
    hookio.emit("PreToolUse", additionalContext="hi")
    out = json.loads(capsys.readouterr().out)
    assert out == {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse", "additionalContext": "hi",
        }
    }


def test_emit_with_nothing_to_say_prints_nothing(capsys):
    hookio.emit("PreToolUse")
    assert capsys.readouterr().out == ""


def test_emit_drops_none_fields(capsys):
    hookio.emit("PreToolUse", updatedInput=None, additionalContext="x")
    out = json.loads(capsys.readouterr().out)
    assert "updatedInput" not in out["hookSpecificOutput"]


def test_run_exits_zero_on_success(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO('{"session_id": "s"}'))
    with pytest.raises(SystemExit) as exc:
        hookio.run(lambda payload: None)
    assert exc.value.code == 0


def test_run_passes_the_parsed_payload(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO('{"session_id": "abc"}'))
    seen: dict = {}
    with pytest.raises(SystemExit):
        hookio.run(lambda payload: seen.update(payload))
    assert seen["session_id"] == "abc"


def test_run_honours_a_returned_exit_code(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    with pytest.raises(SystemExit) as exc:
        hookio.run(lambda payload: 2)
    assert exc.value.code == 2


def test_run_fails_open_on_an_exception(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    def boom(payload):
        raise RuntimeError("kaboom")

    with pytest.raises(SystemExit) as exc:
        hookio.run(boom)
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == "", "must never emit partial JSON"
    assert "RuntimeError: kaboom" in captured.err


def test_run_fails_open_on_malformed_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("{not json"))
    with pytest.raises(SystemExit) as exc:
        hookio.run(lambda payload: None)
    assert exc.value.code == 0
    assert capsys.readouterr().out == ""
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_hookio.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router.hookio'`

- [ ] **Step 3: Write `router/paths.py` and `router/hookio.py`**

```python
# router/paths.py
"""Canonical config locations, and the test-only environment overrides."""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLASSES_YAML = REPO / "config/classes.yaml"
CLASSES_LOCAL_YAML = REPO / "config/classes.local.yaml"
SETTINGS_YAML = REPO / "config/settings.yaml"

STATE_DIR_ENV = "TASK_ROUTER_STATE_DIR"


def state_dir_override() -> Path | None:
    raw = os.environ.get(STATE_DIR_ENV)
    return Path(raw) if raw else None
```

```python
# router/hookio.py
"""Shared hook entry-point plumbing.

The fail-open guarantee lives here, implemented and tested once, because a hook
that crashes or prints malformed JSON blocks the user's work.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable


def read_payload() -> dict:
    return json.load(sys.stdin)


def emit(hook_event_name: str, **fields) -> None:
    """Print a hookSpecificOutput envelope, or nothing when there is nothing
    to say. A hook with no output must stay silent rather than emit an empty
    envelope."""
    populated = {k: v for k, v in fields.items() if v is not None}
    if not populated:
        return
    print(json.dumps({
        "hookSpecificOutput": {"hookEventName": hook_event_name, **populated}
    }))


def run(main: Callable[[dict], int | None]) -> None:
    """Run a hook body, never letting it break the session.

    Exits with whatever `main` returns (`refine_class` returns 2 to rewake the
    session), or 0. Any exception is reported on stderr and swallowed with
    exit 0. sys.exit is called outside the try so its SystemExit is not caught.
    """
    try:
        code = main(read_payload())
    except Exception as exc:  # noqa: BLE001 - failing open is mandatory
        print(f"task-router: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(0)
    sys.exit(code or 0)
```

- [ ] **Step 4: Run the `hookio` tests to verify they pass**

Run: `uv run pytest tests/test_hookio.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Write the failing hook test**

```python
# tests/test_hook_prompt_submit.py
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HOOK = REPO / "hooks/user_prompt_submit.py"


def run_hook(payload: dict, state_dir: Path, expect_stderr: bool = False) -> dict:
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "TASK_ROUTER_STATE_DIR": str(state_dir),
             "PYTHONPATH": str(REPO)},
    )
    assert proc.returncode == 0, f"hook must never fail: {proc.stderr}"
    if expect_stderr:
        assert proc.stderr != "", "expected a fail-open diagnostic on stderr"
    else:
        assert proc.stderr == "", f"hook wrote a fail-open diagnostic: {proc.stderr}"
    return json.loads(proc.stdout) if proc.stdout.strip() else {}


def test_confident_prompt_gets_a_contract(tmp_path):
    out = run_hook(
        {"session_id": "s1", "hook_event_name": "UserPromptSubmit",
         "prompt": "Resolve issue https://example.com/o/r/issues/1088",
         "transcript_path": str(tmp_path / "t.jsonl")},
        tmp_path,
    )
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "class=resolve" in ctx
    assert (tmp_path / "s1.json").exists()


def test_unknown_prompt_gets_unclassified_contract(tmp_path):
    out = run_hook(
        {"session_id": "s2", "hook_event_name": "UserPromptSubmit",
         "prompt": "hmm", "transcript_path": str(tmp_path / "t.jsonl")},
        tmp_path,
    )
    assert "class=unclassified" in out["hookSpecificOutput"]["additionalContext"]


def test_followup_reuses_the_stored_class(tmp_path):
    first = {"session_id": "s3", "hook_event_name": "UserPromptSubmit",
             "prompt": "Check the service logs for repeating errors",
             "transcript_path": str(tmp_path / "t.jsonl")}
    run_hook(first, tmp_path)
    # Simulate a session that has already produced substantial output.
    state = json.loads((tmp_path / "s3.json").read_text())
    state["out_tokens"] = 500_000
    (tmp_path / "s3.json").write_text(json.dumps(state))

    out = run_hook({**first, "prompt": "yes, that's right"}, tmp_path)
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "class=triage" in ctx
    assert "source=continuation" in ctx


def test_user_naming_a_model_is_recorded_as_override(tmp_path):
    run_hook(
        {"session_id": "s4", "hook_event_name": "UserPromptSubmit",
         "prompt": "Review https://example.com/o/r/pull/12 but use opus for it",
         "transcript_path": str(tmp_path / "t.jsonl")},
        tmp_path,
    )
    assert json.loads((tmp_path / "s4.json").read_text())["user_override"] is True


def test_garbage_stdin_exits_zero_and_emits_nothing(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(HOOK)], input="{not json",
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "TASK_ROUTER_STATE_DIR": str(tmp_path),
             "PYTHONPATH": str(REPO)},
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""
    assert proc.stderr != "", "expected a fail-open diagnostic on stderr"
```

- [ ] **Step 6: Run it to verify it fails**

Run: `uv run pytest tests/test_hook_prompt_submit.py -v`
Expected: FAIL — hook file does not exist

- [ ] **Step 7: Write `hooks/user_prompt_submit.py`**

```python
#!/usr/bin/env python3
"""UserPromptSubmit: classify the task and inject the contract.

Fast path only - rules, no network. Must stay under 100 ms and must never
block or break the session. On any failure it exits 0 with empty stdout, and
the session proceeds exactly as it would without the router.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import hookio, paths
from router.classify import UNCLASSIFIED, Classification, classify
from router.config import load_classes, load_scope_policy, load_settings
from router.continuation import is_continuation
from router.contract import render
from router.state import SessionState, load_state, save_state


def main(payload: dict) -> None:
    session_id = payload["session_id"]
    prompt = payload.get("prompt") or ""

    settings = load_settings(paths.SETTINGS_YAML)
    classes = load_classes(paths.CLASSES_YAML, paths.CLASSES_LOCAL_YAML)
    policy = load_scope_policy(paths.CLASSES_YAML)
    state_dir = paths.state_dir_override() or settings.state_dir

    prior = load_state(state_dir, session_id)

    if prior is not None and is_continuation(prompt, prior, settings):
        result = Classification(prior.cls, prior.confidence, "continuation")
        state = prior
    else:
        result = classify(prompt, classes, settings)
        spec = classes.get(result.cls)
        state = SessionState(
            session_id=session_id,
            cls=result.cls,
            confidence=result.confidence,
            source=result.source,
            budget_soft=spec.budget_soft if spec else 0,
            transcript_offset=prior.transcript_offset if prior else 0,
            out_tokens=prior.out_tokens if prior else 0,
        )

    state.user_override = any(
        p.search(prompt) for p in settings.user_override_patterns
    )
    save_state(state_dir, state)

    spec = classes.get(result.cls) if result.cls != UNCLASSIFIED else None
    hookio.emit(
        "UserPromptSubmit", additionalContext=render(result, spec, policy)
    )


if __name__ == "__main__":
    hookio.run(main)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/test_hookio.py tests/test_hook_prompt_submit.py -v`
Expected: PASS, 13 passed (8 hookio + 5 hook)

- [ ] **Step 9: Commit**

```bash
git add router/paths.py router/hookio.py hooks/user_prompt_submit.py tests/test_hookio.py tests/test_hook_prompt_submit.py
git commit -m "feat: UserPromptSubmit hook injecting the task contract"
```

---

### Task 6: Enforcement decision with precedence

**Files:**
- Create: `router/enforce.py`
- Test: `tests/test_enforce.py`

**Interfaces:**
- Consumes: `ClassSpec` (Task 1), `SessionState` (Task 3), `Settings` (Task 1).
- Produces: `Decision` (frozen dataclass: `updated_input: dict | None`, `reason: str`, `precedence: str`); `decide(tool_input: dict, spec: ClassSpec | None, state: SessionState, settings: Settings) -> Decision`. `precedence` is one of `"user"`, `"agent"`, `"contract"`, `"none"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_enforce.py
from pathlib import Path

from router.config import ClassSpec, load_settings
from router.enforce import decide
from router.state import SessionState

REPO = Path(__file__).resolve().parents[1]
SETTINGS = load_settings(REPO / "config/settings.yaml")
SPEC = ClassSpec(
    name="pr_review", patterns=(), delegation="d",
    sub_model="claude-sonnet-5", sub_effort="medium",
    budget_soft=200_000, escalation="e",
)


def _state(**kw) -> SessionState:
    return SessionState(**{
        "session_id": "s", "cls": "pr_review", "confidence": 0.9,
        "source": "rule:x", "budget_soft": 200_000, **kw,
    })


def test_opus_call_is_rewritten_to_the_mandated_model():
    d = decide({"model": "opus", "prompt": "review it"}, SPEC, _state(), SETTINGS)
    assert d.precedence == "contract"
    assert d.updated_input == {"model": "claude-sonnet-5", "effort": "medium"}
    assert "pr_review" in d.reason


def test_call_already_compliant_is_left_alone():
    d = decide(
        {"model": "claude-sonnet-5", "effort": "medium"}, SPEC, _state(), SETTINGS
    )
    assert d.updated_input is None
    assert d.precedence == "none"


def test_user_override_wins_over_the_contract():
    d = decide({"model": "opus"}, SPEC, _state(user_override=True), SETTINGS)
    assert d.updated_input is None
    assert d.precedence == "user"


def test_agent_override_with_a_reason_is_respected():
    d = decide(
        {"model": "opus", "prompt": "model-override: needs cross-file reasoning"},
        SPEC, _state(), SETTINGS,
    )
    assert d.updated_input is None
    assert d.precedence == "agent"
    assert "cross-file reasoning" in d.reason


def test_agent_override_without_a_reason_is_not_respected():
    d = decide({"model": "opus", "prompt": "model-override:"}, SPEC, _state(), SETTINGS)
    assert d.precedence == "contract"
    assert d.updated_input is not None


def test_no_spec_means_no_mandate():
    d = decide({"model": "opus"}, None, _state(cls="unclassified"), SETTINGS)
    assert d.updated_input is None
    assert d.precedence == "none"


def test_class_without_a_sub_model_mandates_nothing():
    spec = ClassSpec("discussion", (), "no delegation", None, None, 50_000, "e")
    d = decide({"model": "opus"}, spec, _state(cls="discussion"), SETTINGS)
    assert d.updated_input is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_enforce.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router.enforce'`

- [ ] **Step 3: Write `router/enforce.py`**

```python
# router/enforce.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_enforce.py -v`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add router/enforce.py tests/test_enforce.py
git commit -m "feat: enforcement decision with user/agent/contract precedence"
```

---

### Task 7: Incremental budget accounting

**Files:**
- Create: `router/budget.py`
- Test: `tests/test_budget.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `read_increment(transcript_path: Path, offset: int) -> tuple[int, int]` returning `(output_tokens_in_increment, new_offset)`.

Reading the whole transcript on every tool call would be expensive — the corpus median is 180 tool calls per session against files up to tens of megabytes. So the byte offset is remembered and only the tail is parsed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_budget.py
import json

from router.budget import read_increment


def _line(out_tokens: int) -> str:
    return json.dumps({
        "type": "assistant",
        "message": {"model": "m", "usage": {"output_tokens": out_tokens}},
    }) + "\n"


def test_counts_output_tokens_from_scratch(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text(_line(100) + _line(50))
    tokens, offset = read_increment(t, 0)
    assert tokens == 150
    assert offset == t.stat().st_size


def test_second_call_counts_only_the_increment(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text(_line(100))
    _, offset = read_increment(t, 0)
    with t.open("a") as fh:
        fh.write(_line(7))
    tokens, new_offset = read_increment(t, offset)
    assert tokens == 7
    assert new_offset > offset


def test_missing_file_is_zero_not_a_crash(tmp_path):
    assert read_increment(tmp_path / "absent.jsonl", 0) == (0, 0)


def test_non_assistant_lines_and_junk_are_ignored(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text('{"type":"user"}\n' + "not json\n" + _line(5))
    tokens, _ = read_increment(t, 0)
    assert tokens == 5


def test_truncated_file_resets_instead_of_reading_garbage(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text(_line(100) + _line(100))
    _, offset = read_increment(t, 0)
    t.write_text(_line(3))  # rewritten shorter, e.g. after compaction
    tokens, new_offset = read_increment(t, offset)
    assert tokens == 3
    assert new_offset == t.stat().st_size


def test_directory_in_place_of_transcript_is_zero_not_a_crash(tmp_path):
    path = tmp_path / "t.jsonl"
    path.mkdir()  # stat() succeeds; open() raises IsADirectoryError
    assert read_increment(path, 0) == (0, 0)


def test_null_message_is_skipped_not_a_crash(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text(json.dumps({"type": "assistant", "message": None}) + "\n" + _line(11))
    tokens, _ = read_increment(t, 0)
    assert tokens == 11
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_budget.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router.budget'`

- [ ] **Step 3: Write `router/budget.py`**

```python
# router/budget.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_budget.py -v`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add router/budget.py tests/test_budget.py
git commit -m "feat: incremental transcript token accounting"
```

---

### Task 8: PreToolUse hook

**Files:**
- Create: `hooks/pre_tool_use.py`
- Test: `tests/test_hook_pre_tool_use.py`

**Interfaces:**
- Consumes: Tasks 1, 3, 6, 7.
- Produces: an executable hook handling three concerns — enforce on `Agent`/`Task`, deny `spawn_task`, and warn once on budget — emitting `{"hookSpecificOutput": {"hookEventName": "PreToolUse", ...}}`.

Hook input fields used: `session_id`, `tool_name`, `tool_input`, `transcript_path`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hook_pre_tool_use.py
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HOOK = REPO / "hooks/pre_tool_use.py"


def seed_state(state_dir: Path, **kw) -> None:
    state = {
        "session_id": "s1", "cls": "pr_review", "confidence": 0.9,
        "source": "rule:x", "budget_soft": 200_000, "budget_notified": False,
        "transcript_offset": 0, "out_tokens": 0, "user_override": False,
        "overrides": [],
    }
    state.update(kw)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "s1.json").write_text(json.dumps(state))


def run_hook(payload: dict, state_dir: Path, enforce: bool) -> dict:
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload), capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO),
             "TASK_ROUTER_STATE_DIR": str(state_dir),
             "TASK_ROUTER_ENFORCE": "1" if enforce else "0"},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == "", f"hook wrote a fail-open diagnostic: {proc.stderr}"
    return json.loads(proc.stdout) if proc.stdout.strip() else {}


def _agent_payload(tmp_path: Path) -> dict:
    return {
        "session_id": "s1", "hook_event_name": "PreToolUse",
        "tool_name": "Agent",
        "tool_input": {"model": "opus", "prompt": "review the diff"},
        "transcript_path": str(tmp_path / "t.jsonl"),
    }


def test_enforce_on_rewrites_the_model(tmp_path):
    seed_state(tmp_path)
    out = run_hook(_agent_payload(tmp_path), tmp_path, enforce=True)
    hso = out["hookSpecificOutput"]
    assert hso["updatedInput"] == {"model": "claude-sonnet-5", "effort": "medium"}
    assert "claude-sonnet-5" in hso["additionalContext"], "rewrite must be visible"


def test_shadow_mode_logs_but_does_not_rewrite(tmp_path):
    seed_state(tmp_path)
    out = run_hook(_agent_payload(tmp_path), tmp_path, enforce=False)
    hso = out.get("hookSpecificOutput", {})
    assert "updatedInput" not in hso
    assert json.loads((tmp_path / "s1.json").read_text())["overrides"], \
        "shadow mode must still record what it would have done"


def test_spawn_task_denied_when_enforcing(tmp_path):
    seed_state(tmp_path)
    out = run_hook(
        {"session_id": "s1", "hook_event_name": "PreToolUse",
         "tool_name": "mcp__ccd_session__spawn_task", "tool_input": {},
         "transcript_path": str(tmp_path / "t.jsonl")},
        tmp_path, enforce=True,
    )
    hso = out["hookSpecificOutput"]
    assert hso["permissionDecision"] == "deny"
    assert "gh issue create" in hso["permissionDecisionReason"]


def test_spawn_task_allowed_in_shadow_mode(tmp_path):
    seed_state(tmp_path)
    out = run_hook(
        {"session_id": "s1", "hook_event_name": "PreToolUse",
         "tool_name": "mcp__ccd_session__spawn_task", "tool_input": {},
         "transcript_path": str(tmp_path / "t.jsonl")},
        tmp_path, enforce=False,
    )
    assert out.get("hookSpecificOutput", {}).get("permissionDecision") != "deny"


def test_budget_warning_fires_once(tmp_path):
    seed_state(tmp_path, budget_soft=100)
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(json.dumps({
        "type": "assistant",
        "message": {"model": "m", "usage": {"output_tokens": 500}},
    }) + "\n")
    payload = {"session_id": "s1", "hook_event_name": "PreToolUse",
               "tool_name": "Bash", "tool_input": {"command": "ls"},
               "transcript_path": str(transcript)}

    first = run_hook(payload, tmp_path, enforce=False)
    assert "BUDGET" in first["hookSpecificOutput"]["additionalContext"]

    second = run_hook(payload, tmp_path, enforce=False)
    assert "BUDGET" not in second.get("hookSpecificOutput", {}).get(
        "additionalContext", ""
    ), "the warning must not nag"


def test_unknown_session_is_a_no_op(tmp_path):
    out = run_hook(_agent_payload(tmp_path), tmp_path, enforce=True)
    assert out == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hook_pre_tool_use.py -v`
Expected: FAIL — hook file does not exist

- [ ] **Step 3: Add the enforce override to `router/paths.py`**

```python
# append to router/paths.py
ENFORCE_ENV = "TASK_ROUTER_ENFORCE"


def enforce_override() -> bool | None:
    raw = os.environ.get(ENFORCE_ENV)
    if raw is None:
        return None
    return raw.strip() in {"1", "true", "yes"}
```

- [ ] **Step 4: Write `hooks/pre_tool_use.py`**

```python
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
from router.enforce import decide
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

    notes: list[str] = []
    updated_input = None

    if tool_name in settings.dispatch_tools:
        classes = load_classes(paths.CLASSES_YAML, paths.CLASSES_LOCAL_YAML)
        tool_input = payload.get("tool_input") or {}
        decision = decide(tool_input, classes.get(state.cls), state, settings)
        if decision.updated_input is not None:
            state.overrides.append({
                "from": tool_input.get("model"),
                "to": decision.updated_input["model"],
                "precedence": decision.precedence,
                "enforced": enforce,
            })
            if enforce:
                updated_input = decision.updated_input
                notes.append(
                    settings.route_notice_template.format(reason=decision.reason)
                )

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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_hook_pre_tool_use.py -v`
Expected: PASS, 6 passed

- [ ] **Step 6: Commit**

```bash
git add router/paths.py hooks/pre_tool_use.py tests/test_hook_pre_tool_use.py
git commit -m "feat: PreToolUse hook enforcing delegation, denying chips, watching budget"
```

---

### Task 9: Telemetry and the Stop hook

**Files:**
- Create: `router/telemetry.py`
- Create: `hooks/stop.py`
- Test: `tests/test_telemetry.py`

**Interfaces:**
- Consumes: `SessionState` (Task 3), `Settings` (Task 1), `read_increment` (Task 7).
- Produces: `append(path: Path, record: dict) -> None`; `read_all(path: Path) -> list[dict]`; a `Stop` hook writing one record per session.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_telemetry.py
import json
import subprocess
import sys
from pathlib import Path

from router.telemetry import append, read_all

REPO = Path(__file__).resolve().parents[1]
HOOK = REPO / "hooks/stop.py"


def test_append_is_one_json_object_per_line(tmp_path):
    path = tmp_path / "t.jsonl"
    append(path, {"a": 1})
    append(path, {"a": 2})
    assert [r["a"] for r in read_all(path)] == [1, 2]


def test_read_all_skips_corrupt_lines(tmp_path):
    path = tmp_path / "t.jsonl"
    append(path, {"a": 1})
    with path.open("a") as fh:
        fh.write("not json\n")
    assert len(read_all(path)) == 1


def test_read_all_on_missing_file_is_empty(tmp_path):
    assert read_all(tmp_path / "absent.jsonl") == []


def test_stop_hook_writes_the_paired_outcome(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "s1.json").write_text(json.dumps({
        "session_id": "s1", "cls": "triage", "confidence": 0.9,
        "source": "rule:logs", "budget_soft": 100, "budget_notified": True,
        "transcript_offset": 0, "out_tokens": 250, "user_override": False,
        "overrides": [{"from": "opus", "to": "claude-sonnet-5",
                       "precedence": "contract", "enforced": False}],
    }))
    telemetry = tmp_path / "telemetry.jsonl"
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("")

    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps({"session_id": "s1", "hook_event_name": "Stop",
                          "transcript_path": str(transcript)}),
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO),
             "TASK_ROUTER_STATE_DIR": str(state_dir),
             "TASK_ROUTER_TELEMETRY": str(telemetry)},
    )
    assert proc.returncode == 0, proc.stderr

    rows = read_all(telemetry)
    assert len(rows) == 1
    row = rows[0]
    assert row["class"] == "triage"
    assert row["outcome"]["out_tokens"] == 250
    assert row["outcome"]["exceeded"] is True
    assert row["overrides"][0]["to"] == "claude-sonnet-5"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_telemetry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router.telemetry'`

- [ ] **Step 3: Write `router/telemetry.py`**

```python
# router/telemetry.py
"""Append-only decision log. Gitignored: it can quote prompts."""

from __future__ import annotations

import json
from pathlib import Path


def append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def read_all(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows
```

- [ ] **Step 4: Add the telemetry override to `router/paths.py`**

```python
# append to router/paths.py
TELEMETRY_ENV = "TASK_ROUTER_TELEMETRY"


def telemetry_override() -> Path | None:
    raw = os.environ.get(TELEMETRY_ENV)
    return Path(raw) if raw else None
```

- [ ] **Step 5: Write `hooks/stop.py`**

```python
#!/usr/bin/env python3
"""Stop: pair the router's decision with what actually happened."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import hookio, paths, telemetry
from router.budget import read_increment
from router.config import load_settings
from router.state import load_state, save_state


def main(payload: dict) -> None:
    settings = load_settings(paths.SETTINGS_YAML)
    state_dir = paths.state_dir_override() or settings.state_dir
    state = load_state(state_dir, payload["session_id"])
    if state is None:
        return

    tokens, new_offset = read_increment(
        Path(payload["transcript_path"]), state.transcript_offset
    )
    state.out_tokens += tokens
    state.transcript_offset = new_offset
    save_state(state_dir, state)

    telemetry.append(
        paths.telemetry_override() or settings.telemetry_path,
        {
            "session_id": state.session_id,
            "class": state.cls,
            "confidence": state.confidence,
            "source": state.source,
            "budget_soft": state.budget_soft,
            "user_override": state.user_override,
            "overrides": state.overrides,
            "outcome": {
                "out_tokens": state.out_tokens,
                "exceeded": bool(
                    state.budget_soft and state.out_tokens > state.budget_soft
                ),
                "budget_notified": state.budget_notified,
            },
        },
    )


if __name__ == "__main__":
    hookio.run(main)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_telemetry.py -v`
Expected: PASS, 4 passed

- [ ] **Step 7: Commit**

```bash
git add router/telemetry.py router/paths.py hooks/stop.py tests/test_telemetry.py
git commit -m "feat: telemetry log and Stop hook pairing decisions with outcomes"
```

---

### Task 10: `router report`

**Files:**
- Create: `router/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `read_all` (Task 9), `load_classes` (Task 1).
- Produces: `summarise(rows: list[dict]) -> dict[str, dict]` mapping class name to `{"n", "median", "p75", "exceeded", "overrides"}`; `propose(summary: dict, classes: dict[str, ClassSpec], settings: Settings) -> list[str]`; a `__main__` entry point.

`propose` returns human-readable suggestions and never writes config — a router that retunes itself silently detunes itself.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
from pathlib import Path

from router.config import load_classes, load_settings
from router.report import propose, summarise

REPO = Path(__file__).resolve().parents[1]
CLASSES = load_classes(REPO / "config/classes.yaml", None)
SETTINGS = load_settings(REPO / "config/settings.yaml")


def _row(cls: str, tokens: int, exceeded: bool = False, overrides=()) -> dict:
    return {
        "class": cls, "budget_soft": 100_000, "overrides": list(overrides),
        "outcome": {"out_tokens": tokens, "exceeded": exceeded},
    }


def test_summarise_groups_by_class():
    rows = [_row("triage", 100), _row("triage", 300), _row("pr_review", 50)]
    s = summarise(rows)
    assert s["triage"]["n"] == 2
    assert s["triage"]["median"] == 200
    assert s["pr_review"]["n"] == 1


def test_summarise_counts_exceeded_and_overrides():
    rows = [
        _row("triage", 500, exceeded=True, overrides=[{"precedence": "agent"}]),
        _row("triage", 10),
    ]
    s = summarise(rows)
    assert s["triage"]["exceeded"] == 1
    assert s["triage"]["overrides"] == 1


def test_propose_flags_a_budget_that_is_too_low():
    rows = [_row("triage", 400_000, exceeded=True) for _ in range(5)]
    lines = propose(summarise(rows), CLASSES, SETTINGS)
    assert any("triage" in l and "100000" in l for l in lines)


def test_propose_stays_silent_on_thin_samples():
    thin = [_row("triage", 999_999, exceeded=True)] * (
        SETTINGS.min_samples_to_propose - 1
    )
    lines = propose(summarise(thin), CLASSES, SETTINGS)
    assert not any("triage" in l for l in lines)


def test_propose_flags_a_frequently_overridden_mandate():
    rows = [_row("triage", 10, overrides=[{"precedence": "agent"}]) for _ in range(5)]
    lines = propose(summarise(rows), CLASSES, SETTINGS)
    assert any("overridden" in l for l in lines)


def test_propose_never_returns_config_writes():
    rows = [_row("triage", 400_000, exceeded=True) for _ in range(5)]
    text = "\n".join(propose(summarise(rows), CLASSES, SETTINGS))
    assert "suggest" in text.lower() or "consider" in text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router.report'`

- [ ] **Step 3: Write `router/report.py`**

```python
# router/report.py
"""Recompute per-class reality from telemetry and propose budget changes.

Proposes only. The config stays reviewable, because a router that retunes
itself silently detunes itself.
"""

from __future__ import annotations

import statistics as st
from collections import defaultdict
from pathlib import Path

from router import paths, telemetry
from router.config import ClassSpec, Settings, load_classes, load_settings


def summarise(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row.get("class", "unclassified")].append(row)

    out: dict[str, dict] = {}
    for cls, items in grouped.items():
        tokens = sorted(i.get("outcome", {}).get("out_tokens", 0) for i in items)
        idx = min(len(tokens) - 1, int(0.75 * len(tokens)))
        out[cls] = {
            "n": len(items),
            "median": int(st.median(tokens)),
            "p75": tokens[idx],
            "exceeded": sum(
                1 for i in items if i.get("outcome", {}).get("exceeded")
            ),
            "overrides": sum(1 for i in items if i.get("overrides")),
        }
    return out


def propose(
    summary: dict[str, dict], classes: dict[str, ClassSpec], settings: Settings
) -> list[str]:
    lines: list[str] = []
    for cls, stats in sorted(summary.items()):
        spec = classes.get(cls)
        if spec is None or stats["n"] < settings.min_samples_to_propose:
            continue
        if stats["exceeded"] / stats["n"] >= settings.exceeded_ratio_threshold:
            lines.append(
                f"{cls}: budget_soft {spec.budget_soft} exceeded in "
                f"{stats['exceeded']}/{stats['n']} sessions; observed p75 is "
                f"{stats['p75']}. Consider raising it to {stats['p75']}."
            )
        if stats["overrides"] / stats["n"] >= settings.override_ratio_threshold:
            lines.append(
                f"{cls}: overridden in {stats['overrides']}/{stats['n']} "
                f"sessions. Suggest revisiting sub_model={spec.sub_model} - "
                f"frequent overrides mean the mandate is wrong, not the agent."
            )
    return lines


def main() -> None:
    settings = load_settings(paths.SETTINGS_YAML)
    path = paths.telemetry_override() or settings.telemetry_path
    rows = telemetry.read_all(Path(path))
    summary = summarise(rows)
    print(f"{'class':14} {'n':>4} {'median':>10} {'p75':>10} {'over':>5} {'ovr':>5}")
    print("-" * 54)
    for cls, s in sorted(summary.items()):
        print(f"{cls:14} {s['n']:>4} {s['median']:>10,} {s['p75']:>10,} "
              f"{s['exceeded']:>5} {s['overrides']:>5}")
    classes = load_classes(paths.CLASSES_YAML, paths.CLASSES_LOCAL_YAML)
    suggestions = propose(summary, classes, settings)
    if suggestions:
        print("\nSuggestions (apply by hand):")
        for line in suggestions:
            print(f"  - {line}")
    else:
        print(
            f"\nNo suggestions: fewer than {settings.min_samples_to_propose} "
            f"sessions per class."
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_report.py -v`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
git add router/report.py tests/test_report.py
git commit -m "feat: router report proposing budget changes from telemetry"
```

---

### Task 11: Asynchronous Haiku refinement

**Files:**
- Create: `router/llm.py`
- Create: `hooks/refine_class.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `Settings` (Task 1), `Classification`/`classify` (Task 2), `SessionState` (Task 3), `render` (Task 4).
- Produces: `build_prompt(prompt: str, class_names: list[str]) -> str`; `parse_response(text: str, class_names: list[str]) -> str | None`; `refine(prompt: str, class_names: list[str], settings: Settings, runner=...) -> str | None`. `runner` is an injected callable `(list[str], int) -> str` so tests never spawn the CLI. Also extends `router/paths.py` with `HAIKU_FAKE_ENV` and `haiku_fake_override() -> str | None`.

Measured: `claude -p --model claude-haiku-4-5-20251001` takes 5.8–7.0 s per call on this machine. That is why this hook is `asyncRewake: true` — it must never sit in front of a prompt. It signals its result by exiting **2** with the refined contract on stderr, which the harness delivers as a system reminder.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm.py
import json
import subprocess
import sys
from pathlib import Path

from router.config import load_settings
from router.llm import build_prompt, parse_response, refine

REPO = Path(__file__).resolve().parents[1]
SETTINGS = load_settings(REPO / "config/settings.yaml")
NAMES = ["recon", "pr_review", "triage", "mechanical"]
HOOK = REPO / "hooks/refine_class.py"


def test_build_prompt_lists_the_classes_and_demands_one_word():
    p = build_prompt("check the logs", NAMES)
    for name in NAMES:
        assert name in p
    assert "check the logs" in p
    assert "unclassified" in p


def test_parse_response_accepts_a_bare_class_name():
    assert parse_response("triage", NAMES) == "triage"


def test_parse_response_tolerates_chatter():
    assert parse_response("This looks like: pr_review.\n", NAMES) == "pr_review"


def test_parse_response_rejects_an_invented_class():
    assert parse_response("obviously_a_database_thing", NAMES) is None


def test_refine_uses_the_injected_runner():
    calls = []

    def runner(argv, timeout):
        calls.append((argv, timeout))
        return "mechanical"

    assert refine("bump the version", NAMES, SETTINGS, runner=runner) == "mechanical"
    argv, timeout = calls[0]
    assert SETTINGS.haiku_model in argv
    assert timeout == SETTINGS.haiku_timeout_s


def test_refine_returns_none_when_the_runner_fails():
    def runner(argv, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    assert refine("x", NAMES, SETTINGS, runner=runner) is None


def test_hook_exits_2_with_the_contract_on_stderr(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "s1.json").write_text(json.dumps({
        "session_id": "s1", "cls": "unclassified", "confidence": 0.0,
        "source": "none", "budget_soft": 0, "budget_notified": False,
        "transcript_offset": 0, "out_tokens": 0, "user_override": False,
        "overrides": [],
    }))
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps({"session_id": "s1", "prompt": "review the pull request",
                          "hook_event_name": "UserPromptSubmit"}),
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO),
             "TASK_ROUTER_STATE_DIR": str(state_dir),
             "TASK_ROUTER_FAKE_HAIKU": "pr_review"},
    )
    assert proc.returncode == 2, f"must exit 2 to rewake: {proc.stdout} {proc.stderr}"
    assert "class=pr_review" in proc.stderr
    assert json.loads((state_dir / "s1.json").read_text())["cls"] == "pr_review"


def test_hook_exits_0_when_already_confident(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "s1.json").write_text(json.dumps({
        "session_id": "s1", "cls": "triage", "confidence": 0.9,
        "source": "rule:logs", "budget_soft": 100, "budget_notified": False,
        "transcript_offset": 0, "out_tokens": 0, "user_override": False,
        "overrides": [],
    }))
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps({"session_id": "s1", "prompt": "check the logs",
                          "hook_event_name": "UserPromptSubmit"}),
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO),
             "TASK_ROUTER_STATE_DIR": str(state_dir),
             "TASK_ROUTER_FAKE_HAIKU": "recon"},
    )
    assert proc.returncode == 0, "no rewake when the rules were already confident"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router.llm'`

- [ ] **Step 3: Write `router/llm.py`**

```python
# router/llm.py
"""Asynchronous class refinement via the Claude CLI.

Measured cost: 5.8-7.0 s per call, because each call starts a new CLI process.
Never call this on a blocking path.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence

from router.classify import UNCLASSIFIED
from router.config import Settings

Runner = Callable[[Sequence[str], int], str]


def build_prompt(prompt: str, class_names: list[str]) -> str:
    options = ", ".join([*class_names, UNCLASSIFIED])
    return (
        "Classify the following work request by the KIND of operation it asks "
        "for. Do not judge how hard it is.\n"
        f"Answer with exactly one of: {options}\n"
        "Answer with the single word and nothing else.\n\n"
        f"REQUEST:\n{prompt}"
    )


def parse_response(text: str, class_names: list[str]) -> str | None:
    lowered = (text or "").lower()
    hits = [n for n in class_names if n.lower() in lowered]
    # Exactly one recognised class, otherwise we learned nothing usable.
    return hits[0] if len(hits) == 1 else None


def _default_runner(argv: Sequence[str], timeout: int) -> str:
    proc = subprocess.run(
        list(argv), capture_output=True, text=True, timeout=timeout, check=True
    )
    return proc.stdout


def refine(
    prompt: str,
    class_names: list[str],
    settings: Settings,
    runner: Runner | None = None,
) -> str | None:
    run = runner or _default_runner
    argv = [
        "claude", "-p", "--model", settings.haiku_model,
        build_prompt(prompt, class_names),
    ]
    try:
        return parse_response(run(argv, settings.haiku_timeout_s), class_names)
    except Exception:  # noqa: BLE001 - refinement is best-effort by design
        return None
```

- [ ] **Step 4: Write `hooks/refine_class.py`**

```python
#!/usr/bin/env python3
"""UserPromptSubmit (asyncRewake): refine a low-confidence classification.

Runs in the background so its measured ~6.5 s never blocks a prompt. On a
successful refinement it exits 2 with the new contract on stderr, which the
harness delivers to the session as a system reminder. Any other outcome exits 0
and the session keeps the unclassified contract it already has.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import hookio, paths
from router.classify import Classification
from router.config import load_classes, load_scope_policy, load_settings
from router.contract import render
from router.llm import refine
from router.state import load_state, save_state


def main(payload: dict) -> int:
    settings = load_settings(paths.SETTINGS_YAML)
    state_dir = paths.state_dir_override() or settings.state_dir
    state = load_state(state_dir, payload["session_id"])
    if state is None or state.confidence >= settings.confidence_threshold:
        return 0

    classes = load_classes(paths.CLASSES_YAML, paths.CLASSES_LOCAL_YAML)
    names = list(classes)

    fake = paths.haiku_fake_override()
    if fake:
        cls = fake if fake in names else None
    else:
        cls = refine(payload.get("prompt") or "", names, settings)

    if cls is None or cls == state.cls:
        return 0

    spec = classes[cls]
    state.cls = cls
    state.confidence = settings.refined_confidence
    state.source = "haiku"
    state.budget_soft = spec.budget_soft
    save_state(state_dir, state)

    contract = render(
        Classification(cls, settings.refined_confidence, "haiku"),
        spec,
        load_scope_policy(paths.CLASSES_YAML),
    )
    # stderr, not stdout: exit 2 makes the harness deliver this as a reminder.
    print(f"Router refined this task's classification:\n{contract}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    hookio.run(main)
```

- [ ] **Step 5: Add the Haiku test override to `router/paths.py`**

```python
# append to router/paths.py
HAIKU_FAKE_ENV = "TASK_ROUTER_FAKE_HAIKU"


def haiku_fake_override() -> str | None:
    """Test-only: return a class name instead of spawning the Claude CLI."""
    return os.environ.get(HAIKU_FAKE_ENV) or None
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm.py -v`
Expected: PASS, 8 passed

- [ ] **Step 7: Verify the real CLI path once, by hand**

```bash
uv run python -c "
from router.config import load_settings
from router.llm import refine
from router.paths import SETTINGS_YAML
s = load_settings(SETTINGS_YAML)
print(refine('please review the pull request', ['recon','pr_review','triage'], s))
"
```
Expected: prints `pr_review` after roughly 6–7 seconds. If it prints `None`, run the `claude -p` command by hand to see whether the CLI is authenticated.

- [ ] **Step 8: Commit**

```bash
git add router/llm.py router/paths.py hooks/refine_class.py tests/test_llm.py
git commit -m "feat: async Haiku refinement for low-confidence classifications"
```

---

### Task 12: Wire the hooks in and smoke-test shadow mode

**Files:**
- Create: `README.md`
- Create: `tests/test_end_to_end.py`
- Modify: `/Users/matejkys/.claude/settings.json` (add three hook entries)

**Interfaces:**
- Consumes: all hooks from Tasks 5, 8, 9, 11.
- Produces: a live shadow-mode installation and a test proving the full prompt → dispatch → stop cycle writes correct telemetry.

- [ ] **Step 1: Write the failing end-to-end test**

```python
# tests/test_end_to_end.py
import json
import subprocess
import sys
from pathlib import Path

from router.telemetry import read_all

REPO = Path(__file__).resolve().parents[1]


def _run(hook: str, payload: dict, env: dict) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [sys.executable, str(REPO / "hooks" / hook)],
        input=json.dumps(payload), capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO), **env},
    )
    assert proc.returncode in (0, 2), f"{hook}: {proc.stderr}"
    return proc


def test_full_cycle_in_shadow_mode(tmp_path):
    state_dir = tmp_path / "state"
    telemetry = tmp_path / "telemetry.jsonl"
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(json.dumps({
        "type": "assistant",
        "message": {"model": "m", "usage": {"output_tokens": 900}},
    }) + "\n")
    env = {
        "TASK_ROUTER_STATE_DIR": str(state_dir),
        "TASK_ROUTER_TELEMETRY": str(telemetry),
        "TASK_ROUTER_ENFORCE": "0",
    }
    sid = "e2e"

    submit = _run("user_prompt_submit.py", {
        "session_id": sid, "hook_event_name": "UserPromptSubmit",
        "prompt": "Please review https://example.com/o/r/pull/42",
        "transcript_path": str(transcript),
    }, env)
    assert "class=pr_review" in json.loads(submit.stdout)[
        "hookSpecificOutput"]["additionalContext"]

    pre = _run("pre_tool_use.py", {
        "session_id": sid, "hook_event_name": "PreToolUse",
        "tool_name": "Agent",
        "tool_input": {"model": "opus", "prompt": "review it"},
        "transcript_path": str(transcript),
    }, env)
    hso = json.loads(pre.stdout).get("hookSpecificOutput", {})
    assert "updatedInput" not in hso, "shadow mode must not rewrite"

    _run("stop.py", {
        "session_id": sid, "hook_event_name": "Stop",
        "transcript_path": str(transcript),
    }, env)

    rows = read_all(telemetry)
    assert len(rows) == 1
    assert rows[0]["class"] == "pr_review"
    assert rows[0]["overrides"][0]["enforced"] is False
    assert rows[0]["overrides"][0]["to"] == "claude-sonnet-5"


def test_enforcing_the_same_cycle_rewrites(tmp_path):
    state_dir = tmp_path / "state"
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("")
    env = {
        "TASK_ROUTER_STATE_DIR": str(state_dir),
        "TASK_ROUTER_TELEMETRY": str(tmp_path / "tel.jsonl"),
        "TASK_ROUTER_ENFORCE": "1",
    }
    _run("user_prompt_submit.py", {
        "session_id": "e2e2", "hook_event_name": "UserPromptSubmit",
        "prompt": "Please review https://example.com/o/r/pull/42",
        "transcript_path": str(transcript),
    }, env)
    pre = _run("pre_tool_use.py", {
        "session_id": "e2e2", "hook_event_name": "PreToolUse",
        "tool_name": "Agent",
        "tool_input": {"model": "opus", "prompt": "review it"},
        "transcript_path": str(transcript),
    }, env)
    hso = json.loads(pre.stdout)["hookSpecificOutput"]
    assert hso["updatedInput"]["model"] == "claude-sonnet-5"
    assert "claude-sonnet-5" in hso["additionalContext"]
```

- [ ] **Step 2: Run it to verify it fails, then passes**

Run: `uv run pytest tests/test_end_to_end.py -v`
Expected: initially FAIL if any hook misbehaves; after Tasks 5–11 it should PASS, 2 passed. Fix the offending hook rather than the test.

- [ ] **Step 3: Run the whole suite and the latency check**

```bash
uv run pytest -v
```
Expected: all tests pass, including `test_rules_path_is_under_100ms`.

- [ ] **Step 4: Write `README.md`**

```markdown
# task-router

Routes agent work by *kind of operation* instead of by guesswork, and enforces
the delegation plan it decides on.

Built on a measured corpus of 41 real sessions. The headline finding: task
difficulty is **not** predictable from prompt text — correlation of prompt
length with output tokens is r=+0.24, and variance within a single task category
runs 36x to 125x. Two near-identical prompts against the same log dashboard cost
28K and 3,501K output tokens respectively. So this does not try to guess
difficulty. It classifies the kind of operation, which *is* readable, and
mandates how the work should be delegated.

The model of the main loop is deliberately out of scope: hooks cannot change it.
The lever is delegation, which `PreToolUse` can enforce.

## Install

Add to `~/.claude/settings.json` (see "Hooks" below), then:

```bash
uv sync
uv run pytest
```

Ships in **shadow mode** (`config/settings.yaml`, `enforce: false`): the contract
is injected and budgets are watched, but nothing is rewritten and no tool is
denied. Do not expect shadow mode to improve anything — advisory guidance
demonstrably does not change behaviour. It is there to calibrate the classifier
and the budgets before they get teeth.

## Hooks

Three entries in `~/.claude/settings.json`. `$ROUTER` below is
`/Users/matejkys/Development/tools/task-router`.

- `UserPromptSubmit` -> `hooks/user_prompt_submit.py` (synchronous, <100 ms)
- `UserPromptSubmit` -> `hooks/refine_class.py` (`asyncRewake: true`)
- `PreToolUse` -> `hooks/pre_tool_use.py` (matcher `Agent|Task|mcp__ccd_session__spawn_task|Bash|Edit|Write|Read`)
- `Stop` -> `hooks/stop.py`

## Tuning

```bash
uv run python -m router.report
```

Recomputes per-class medians and p75 from telemetry and *proposes* budget
changes. It never writes config: a router that retunes itself silently detunes
itself. Apply suggestions by hand to `config/classes.yaml`.

## Configuration

- `config/classes.yaml` — taxonomy, models, effort, budgets, generic patterns.
- `config/classes.local.yaml` — gitignored overlay for internal vocabulary.
  Adds patterns to base classes; never replaces them.
- `config/settings.yaml` — thresholds, paths, `enforce`.

## Privacy

This repository is public. Nothing derived from real sessions is committed:
`analysis/sessions.json`, `tests/fixtures/golden.local.jsonl`,
`config/classes.local.yaml`, `state/`, `telemetry.jsonl` and `reports/` are all
gitignored. `tests/fixtures/golden.example.jsonl` is paraphrased so the suite
runs on a fresh clone; the coverage test reports which fixture it used.
```

- [ ] **Step 5: Add the hooks to `~/.claude/settings.json`**

These keys go **inside** the existing top-level `"hooks"` object. `~/.claude/settings.json` already carries a `Notification` hook, three `PostToolUse` hooks (py_compile, tsc, `ty`) and a `Stop` hook running `notify-hook.sh` — so `Stop` must gain a second entry in its existing array, not replace it. **Back the file up first:**

```bash
cp ~/.claude/settings.json ~/.claude/settings.json.bak-$(date +%Y%m%d)
```

```json
{
  "UserPromptSubmit": [
    {
      "hooks": [
        {
          "type": "command",
          "command": "/Users/matejkys/Development/tools/task-router/.venv/bin/python /Users/matejkys/Development/tools/task-router/hooks/user_prompt_submit.py",
          "timeout": 10,
          "statusMessage": "routing task"
        },
        {
          "type": "command",
          "command": "/Users/matejkys/Development/tools/task-router/.venv/bin/python /Users/matejkys/Development/tools/task-router/hooks/refine_class.py",
          "asyncRewake": true,
          "timeout": 60
        }
      ]
    }
  ],
  "PreToolUse": [
    {
      "matcher": "Agent|Task|mcp__ccd_session__spawn_task|Bash|Edit|Write|Read",
      "hooks": [
        {
          "type": "command",
          "command": "/Users/matejkys/Development/tools/task-router/.venv/bin/python /Users/matejkys/Development/tools/task-router/hooks/pre_tool_use.py",
          "timeout": 10
        }
      ]
    }
  ],
  "Stop": [
    {
      "hooks": [
        {
          "type": "command",
          "command": "/Users/matejkys/Development/tools/task-router/.venv/bin/python /Users/matejkys/Development/tools/task-router/hooks/stop.py",
          "timeout": 10
        }
      ]
    }
  ]
}
```

- [ ] **Step 6: Verify the installation by hand**

```bash
echo '{"session_id":"install-check","hook_event_name":"UserPromptSubmit","prompt":"Please review https://example.com/o/r/pull/1","transcript_path":"/dev/null"}' \
  | /Users/matejkys/Development/tools/task-router/.venv/bin/python \
    /Users/matejkys/Development/tools/task-router/hooks/user_prompt_submit.py
```
Expected: a single line of JSON containing `class=pr_review`. Then start a fresh Claude Code session, send any prompt, and confirm the contract appears in the context and that `~/.claude/task-router/state/` gains a file.

- [ ] **Step 7: Commit**

```bash
git add README.md tests/test_end_to_end.py
git commit -m "feat: end-to-end shadow-mode test and installation docs"
```

---

## After the plan: the calibration step

The plan ends with a working router in shadow mode, which is deliberately not
the end of the work. Before setting `enforce: true`:

1. Run `uv run python analysis/agg_sessions.py 14`, then
   `uv run python analysis/build_golden.py`, then **correct every label by
   hand** — `build_golden.py` writes the classifier's own guesses, and a fixture
   of its own guesses tests nothing.
2. Let shadow mode run for a week, then `uv run python -m router.report`.
3. Compare the proposed budgets with `config/classes.yaml`. Several classes have
   n=3 or n=4 in the original corpus, so expect real movement.
4. Only then flip `enforce: true`, and watch the override rate: a class
   overridden in more than half its sessions has a wrong mandate, not a
   disobedient agent.
