# task-router

Routes agent work by *kind of operation* instead of by guesswork, and enforces
the delegation plan it decides on, via Claude Code hooks.

Built on a measured corpus of 41 real sessions. The headline finding: task
difficulty is **not** predictable from prompt text — correlation of prompt
length with output tokens is r=+0.24, and variance within a single task category
runs 36x to 125x. Two near-identical prompts against the same log dashboard cost
28K and 3,501K output tokens respectively. So this does not try to guess
difficulty. It classifies the kind of operation, which *is* readable, and
mandates how the work should be delegated.

The model of the main loop is deliberately out of scope: hooks cannot change it.
The lever is delegation, which `PreToolUse` can enforce — in two ways once
`enforce: true`:

- **Every subagent dispatch gets the class's mandate.** The `model`/`effort` of
  an `Agent`/`Task` call is rewritten to what the task class prescribes, and the
  rewrite is announced to the orchestrator so it can slice work accordingly.
- **The main loop never edits code.** Hooks fire inside subagents too, and the
  hook input carries `agent_id` only there — so an `Edit`/`Write`/`NotebookEdit`
  on a code-file extension with no `agent_id` is the orchestrator reaching for
  the editor itself. It is denied, with a reason telling it to dispatch a
  subagent with the mandated model and review the result. Docs and config
  (`.md`, `.yaml`, `.json`, …) stay editable in either context: writing specs,
  plans and configuration *is* the orchestrator's job.

In shadow mode both are only counted (`overrides`, `main_loop_code_edits` in the
telemetry row), so calibration also measures how often the orchestrator would
have been corrected.

## Install

```bash
uv sync
uv run pytest
```

Then add the four hook entries below to `~/.claude/settings.json` (see
"Hooks").

Ships in **shadow mode** (`config/settings.yaml`, `enforce: false`): the contract
is injected and budgets are watched, but nothing is rewritten and no tool is
denied. Do not expect shadow mode to improve anything — advisory guidance
demonstrably does not change behaviour. It is there to calibrate the classifier
and the budgets before they get teeth.

To turn enforcement on for **one installation** without changing the shipped
default, prefix the `pre_tool_use.py` hook command in `settings.json` with
`TASK_ROUTER_ENFORCE=1` — the environment overrides `enforce` in
`config/settings.yaml`. That is how the author's own install runs.

## Hooks

Four entries in `~/.claude/settings.json`:

- `UserPromptSubmit` -> `hooks/user_prompt_submit.py` (synchronous, <100 ms)
- `UserPromptSubmit` -> `hooks/refine_class.py` (`asyncRewake: true`)
- `PreToolUse` -> `hooks/pre_tool_use.py` (matcher
  `Agent|Task|mcp__ccd_session__spawn_task|Edit|Write|NotebookEdit` — the
  dispatch tools for mandate enforcement, the code-edit tools for the
  main-loop invariant. `Bash` and `Read` are deliberately left out: a Python
  spawn on every one of them costs ~150 ms per call, so token accounting is
  done per turn in `stop.py` instead of per tool call.)
- `Stop` -> `hooks/stop.py`

These keys go **inside** your existing top-level `"hooks"` object, merged with
whatever you already have — if you already have hooks on any of these same
events (`UserPromptSubmit`, `PreToolUse`, `Stop`), append to that event's array
rather than replacing it. Back your settings up first:

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
      "matcher": "Agent|Task|mcp__ccd_session__spawn_task|Edit|Write|NotebookEdit",
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

## Tuning

```bash
uv run python -m router.report
```

Recomputes per-class medians and p75 from telemetry and *proposes* budget
changes. It never writes config: a router that retunes itself silently detunes
itself. Apply suggestions by hand to `config/classes.yaml`.

Alongside the output-token columns it shows cost per class — `n$` (how many of
the class's sessions carry usage; older rows do not), `med$` and `p75$` of
`cost_usd.total`, and `main%`, the share of that cost spent in the main loop
rather than in subagents. A class with no usage rows prints `n/a` rather than a
misleading zero.

## Cost

```bash
uv run python -m analysis.cost_report            # last 90 days, grouped by era
uv run python -m analysis.cost_report --days 14
uv run python -m analysis.cost_report --since 2026-08-01
uv run python -m analysis.cost_report --by class --json
```

Reads the transcripts under `~/.claude/projects` retroactively: one row per
session with its era (the dominant main-loop model), main-loop vs subagent
cost, per-model breakdown, user turns, interrupts, subagent dispatches, span,
and the router class where telemetry knows the session. `--by` groups by
`era`, `model`, `class` or `week`; `--json` prints the whole thing for further
analysis. The router's own `claude -p` classifier sessions are excluded and
counted separately — they are the router talking to itself, not work.

The Stop hook records the same numbers live, incrementally, so no transcript is
re-parsed: each telemetry row carries a UTC `ts` and an `outcome.usage` block
with per-model token totals for `main` (the main loop) and `sub` (sidechain
records plus every subagent transcript), `cost_usd` split into
`main`/`sub`/`total`, and `unpriced_models`. Cost accounting is fail-open: if
pricing is unreadable the row is still written, just without `usage`.

## Configuration

- `config/classes.yaml` — taxonomy, models, effort, budgets, generic patterns.
- `config/classes.local.yaml` — gitignored overlay for internal vocabulary.
  Adds patterns to existing classes and may define new ones; never replaces a
  base class's pattern list.
- `config/settings.yaml` — thresholds, paths, `enforce`, and the code-edit
  invariant's tool list, code-extension list and deny reason.
- `config/pricing.yaml` — USD per million tokens per canonical model id
  (`input`, `output`, `cache_read`, `cache_write_5m`, `cache_write_1h`), plus
  `as_of`, `source`, `default_cache_ttl` (used when a record reports only a flat
  cache-creation total) and `zero_cost_models`. A model absent from the table is
  never silently priced: its tokens are reported as `unpriced` instead, which is
  the signal to add the new model. `TASK_ROUTER_PRICING` points both the hook
  and the report at another table.

  These are **list prices**, and the author is on a subscription — the dollars
  are a consistent relative measure of consumption, not an invoice. Cache reads
  dominate raw token volume, so compare priced cost, never token counts.

## Privacy

This repository is public. Nothing derived from real sessions is committed:
`analysis/sessions.json`, `tests/fixtures/golden.local.jsonl`,
`config/classes.local.yaml`, `state/`, `telemetry.jsonl` and `reports/` are all
gitignored. `tests/fixtures/golden.example.jsonl` is paraphrased so the suite
runs on a fresh clone; the coverage test reports which fixture it used.

## Before enforcing

Shadow mode is not the end state. Before setting `enforce: true` in
`config/settings.yaml`:

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
