# Task Router — Design

Date: 2026-07-29
Status: approved, ready for implementation planning

## Problem

Sessions take too long and solve tasks in a more complicated way than the task
warrants, and there is no reliable way to pick the right model up front.

Measured over 41 sessions (2026-06-30 .. 2026-07-29) from
`~/.claude/projects/**/*.jsonl`:

- The main loop runs on Opus in **90%** of assistant turns (Opus 4.8 46%,
  Opus 5 44%). Sonnet 6%, Fable 3%, Haiku **0%**.
- Subagents already route well: Sonnet 75%, Haiku 7.5%, Opus 17.5%.
- Recorded `effort` is only ever `high` (8250), `max` (3031) or `xhigh` (558).
  `low` and `medium` appear **zero** times.
- Median session: 60 active minutes, 264K output tokens, 180 tool calls.
  p75: 138 min / 747K. Maximum: 456 min and 3.5M output tokens.
- 20 user interrupts across 12 of 41 sessions, concentrated in log triage (5)
  and open-ended investigation (7).

### The finding that shapes this design

Task difficulty **cannot be predicted from the prompt text**. Correlation of
prompt length with output tokens is `r = +0.24`, and variance *within* a task
category is 36x to 125x.

The clearest evidence: two near-identical prompts against the same Datadog
dashboard of the same service.

| prompt | model | turns | wall clock | output |
|---|---|---|---|---|
| `Check <service> errors: <log url>` | Sonnet | 23 | 5 min | 28K |
| `Repeating error appears in the logs: <url>` | Opus | 1882 | 456 min | 3501K |

The difference was not in the request. It was in what the logs turned out to
contain. Any classifier of the form "prompt -> model" routes these two
identically and is therefore wrong about one of them.

What *is* reliably readable from the prompt is the **kind of operation**. That
is what this router classifies.

## Goals

Optimise for **time to result** and **directness of path**. Cost reduction is a
side effect, not the target. Investing up front in a well-formed task
specification is acceptable; wandering sideways afterwards is not.

Scope discipline is part of the goal:

- A newly discovered problem that is *related* to the task gets fixed without
  asking for confirmation.
- A problem that is *unrelated* becomes a GitHub issue — not a background-task
  chip, not a new session.

## Non-goal: the main-loop model

Verified against the hooks reference: a hook **cannot** change the model of the
main loop. `model` is a read-only input to `SessionStart` only, and there is no
`$CLAUDE_MODEL`. All 64 sessions in the corpus started from `claude-desktop`,
never from the CLI, so a launcher wrapper that execs `claude --model X` would
not intersect this workflow either.

This design therefore does not try to re-model the main loop. Opus in the main
loop is correct for the orchestrator role — planning, holding scope, deciding.
The defect is not that Opus is there; it is that Opus keeps the mechanical work
for itself. Five of five pure PR-review sessions ran the review on Opus, three
of them without dispatching a single subagent.

The lever is **delegation**, and delegation *is* enforceable: `PreToolUse` can
rewrite tool input via `updatedInput`, including the `model` and `effort`
parameters of an `Agent`/`Task` call.

## Architecture

Contextual decisions in hooks, static defaults in agent frontmatter as a
fallback, and permanent telemetry so the rules are tuned on data rather than
guesswork.

Classification engine is **hybrid**: deterministic rules decide the class; a
Haiku call is made only when no rule matches confidently, or when a scope fence
needs to be written in prose.

### Task classes

Budgets below are measured medians and p75 from the corpus, not estimates.
`n` is the number of sessions the baseline rules assigned to each class
(35 of 40 classified; 87.5% coverage).

| class | n | med | p75 | mandated delegation | sub effort | soft budget |
|---|---|---|---|---|---|---|
| `recon` | 4 | 70K | 809K | all work in subagents: Haiku extracts, Sonnet synthesises; main loop dispatches and summarises only | `low` | 150K |
| `pr_review` | 3 | 57K | 187K | Sonnet reviewer subagent; Opus writes the final verdict only | `medium` | 200K |
| `triage` | 4 | 240K | 3501K | two-phase, see below | `low` then `medium` | 100K for phase 1 |
| `mechanical` | 7 | 300K | 964K | Sonnet, parallel pipeline when there are N items | `low` | 400K |
| `resolve` | 9 | 263K | 458K | Sonnet implements, Opus acts as review gate | `medium` | 500K |
| `feature` | 4 | 1378K | 1947K | main loop orchestrates and plans, the strong model builds phase by phase | `high` | 2000K |
| `investigate` | 4 | 888K | 2715K | main loop reasons, subagents gather evidence on the strong model, with a mandatory checkpoint | `high` | 300K to checkpoint |
| `discussion` | — | — | — | no delegation, no code | n/a | 50K |
| `continuation` | — | — | — | inherits the running session's contract | inherited | inherited |

The effort column applies to *subagents dispatched by* the class, since main-loop
effort is not settable by a hook. For `discussion` there are no subagents, so no
effort is mandated.

**Addendum (2026-07-30): the sub-agent ceiling, decoupled from the main-loop
model.** The original wording above ("Opus orchestrates … Sonnet builds")
assumed the main loop itself runs on Opus, which matched the 90% baseline this
design was built from. In practice the main-loop model is a per-session user
choice — sessions that deliberately start on Sonnet still correctly dispatched
Opus subagents for the two hardest tasks in a later 21-day sample (a
~7-hour/2.7M-token session and a ~3.4-hour/1M-token one), but no class mandated
anything above Sonnet for a subagent. Once `enforce: true`, that same ad hoc
escalation would be silently rewritten back down to the class's mandate. So
`feature` and `investigate` — the two classes whose measured budgets are an
order of magnitude past the rest — now mandate `sub_model: claude-opus-5`
regardless of what model the main loop is running. The other six classes are
unchanged: their measured costs never showed a need past Sonnet.

Two classes carry nearly all of the loss and need explaining.

**`triage` is two-phase, not "some model".** The 125x spread came from differing
findings, not differing requests. So phase 1 is Haiku/Sonnet reading the logs to
answer exactly one question — *is this a known error or a systemic problem?* —
capped at 100K. Phase 2, an Opus deep dive, opens **only** when phase 1
evidences a systemic cause. The 3501K session would have earned phase 2
legitimately; the 28K session would never have opened it.

**`investigate` gets a checkpoint, not a ceiling.** It is the class with the
most interrupts (6 of 20), which is precisely the sideways wandering this
project exists to remove. At 300K the agent must stop and present what it knows
and what it proposes next, instead of searching for three hours until
interrupted.

**`continuation` exists because short follow-ups are not small tasks.** One
40-character follow-up — confirming a field-naming decision and pointing at
unresolved review comments — arrived in a session that had already spent 283K. A
naive router would classify it as trivial and route it wrongly.

### Task contract

What the router injects as `additionalContext`:

```
[ROUTER] class=triage  confidence=0.82  source=rule(datadog|logs\?query)
GOAL        <request normalised to one sentence>
SCOPE       in: … | out: …
DELEGATION  log reading -> Haiku/low · synthesis -> Sonnet/medium
            main loop: decisions and summary only
BUDGET      soft 100K -> stop and present state
ESCALATION  open phase 2 only with an evidenced systemic cause
SCOPE+      related problem -> fix it without asking
            unrelated -> gh issue create, NOT a new-session chip
```

The last line is enforced rather than merely advised: `PreToolUse` denies
`spawn_task` and returns the instruction to file a GitHub issue in
`permissionDecisionReason`.

### Precedence

Enforcement must not be impervious, or the result is a tool that cannot be
reasoned with.

1. Explicit user instruction in the prompt ("put this on Opus") — the router
   does not route, and logs `user_override`.
2. A justified override from the agent — respected, and recorded. `router
   report` shows how often and in which classes overrides happen, so abuse
   surfaces as data and is answered by fixing the rules, not by forbidding it.
3. The class contract — enforced.
4. None of the above — the default from the agent's frontmatter.

### Transparency of rewrites

A rewrite via `updatedInput` happens *after* the agent emitted the call, so the
agent cannot refuse it — and would not otherwise learn of it. It would believe
it dispatched Opus and receive Sonnet. That is actively harmful to the goal: an
orchestrator slices work according to the assumed strength of the subagent and
then gets a weaker result without knowing why, which produces exactly the
circling this project removes.

So every rewrite returns `additionalContext` alongside `updatedInput`: "this
call was routed to Sonnet per class `triage`". The agent can then cut the task
differently.

## Components

```
tools/task-router/
├── router/
│   ├── classify.py      rules -> (class, confidence)
│   ├── llm.py           async Haiku refinement when confidence < threshold
│   ├── continuation.py  follow-up detection from session state
│   ├── contract.py      contract rendering
│   ├── enforce.py       (class, agent_type) -> model + effort; honours the
│   │                    precedence order, including justified agent overrides
│   ├── budget.py        incremental token accounting from the transcript
│   └── telemetry.py     append-only JSONL
├── config/
│   ├── classes.yaml         taxonomy, models, effort, budgets, generic patterns
│   ├── classes.local.yaml   gitignored overlay: internal vocabulary
│   └── settings.yaml        thresholds, paths, feature flags
├── hooks/               four thin entry points: prompt submit, async refine,
│                        pre-tool-use, stop
├── analysis/            reproducible corpus aggregation
└── tests/
```

Everything tunable lives in `config/classes.yaml`; no constant is hardcoded in
code. If the config is missing, the router **fails with a clear error** rather
than inventing defaults.

### Public repository split

The repository is public, so nothing derived from real sessions may be
committed. Two things in this design would otherwise leak.

**Classification patterns.** Useful rules inevitably carry internal vocabulary —
Czech phrasings, repository names, internal feature names. So `classes.yaml`
ships the taxonomy, models, effort and budgets with generic
operation-kind patterns, and `classes.local.yaml` is a gitignored overlay merged
on top for internal vocabulary and per-repository overrides. A missing overlay is
normal and not an error; a missing base config is an error.

**The golden set.** The regression fixture is 40 real prompts and cannot be
committed. It splits the same way: `tests/fixtures/golden.local.jsonl` is
generated from the local corpus by `analysis/build_golden.py` and gitignored,
while `tests/fixtures/golden.example.jsonl` holds a small paraphrased set so the
suite runs for a colleague on a fresh clone. The coverage-regression test runs
against the local fixture when present and against the example set otherwise,
reporting which one it used so a passing run is never mistaken for the full
check.

`router report` output may quote prompts and therefore writes only to a
gitignored path.

## Data flow

**1. `UserPromptSubmit` — classify and contract.** First the continuation test:
if an active contract exists for the session and the prompt looks like a
follow-up (short, no new URL or issue reference, session turns above a
threshold), the existing contract is **re-issued** rather than reclassified.
Otherwise rules run, with Haiku as fallback. State goes to
`~/.claude/task-router/state/<session_id>.json`; the contract goes to stdout as
`additionalContext`.

Latency: the rules path is plain Python with precompiled regexes, targeting
under 100 ms, because this runs before every prompt. The router must never block
a prompt.

**The Haiku fallback is asynchronous, and this is measured, not assumed.**
`claude -p --model claude-haiku-4-5-*` takes 5.8–7.0 s on this machine (three
runs), because each call starts a new CLI process. A synchronous call would
therefore block every ambiguous prompt for about seven seconds. A native
`prompt`-type hook is not an alternative either: its result cannot become
`additionalContext`, only a permission decision.

So refinement runs as a second `UserPromptSubmit` hook with
`asyncRewake: true`. It does not block; when it finishes it exits with code 2 and
writes the refined contract to stderr, which reaches the session as a system
reminder. The cost of the 6.5 s is paid in the background, arriving a few seconds
into a task the agent is still reading. It fires only when the rules path is
unconfident — roughly 12.5% of prompts at measured coverage.

Until the refinement lands, the synchronous hook emits an `unclassified`
contract: no delegation mandate and no budget, only the scope policy. That
degraded state is deliberately useful rather than dead, because on a Haiku
failure or timeout it is also the final state.

**2. `PreToolUse` on `Agent|Task` — enforcement.** Loads the active contract,
looks up the mandated `model` and `effort` for the class and `subagent_type`,
and returns `updatedInput` when they differ. This is the only place in the
design with hard enforcement, which is why it is deliberately dumb and fully
table-driven.

**3. `PreToolUse` on `spawn_task`** — `permissionDecision: "deny"` with the
reason "unrelated finding -> `gh issue create`".

**4. `PreToolUse` on any tool — budget watch.** Reading the whole transcript on
every tool call would be expensive, so `budget.py` remembers a byte offset and
reads only the increment. On crossing the soft budget it injects, once (flagged
in state so it does not nag), `additionalContext`: "you are at 112K of 100K for
class `triage` — stop and present state".

**5. `Stop` — close telemetry.** Computes actual tokens and pairs them with the
router's decision. `Stop` fires once per assistant turn, so this appends one row
per turn under the session's id; `router report` collapses a session's rows into
one sample at read time (keeping the fullest snapshot). The append stays
unconditional because it is concurrency-safe across the many parallel sessions
this user runs, where a read-modify-write upsert would not be.

## Telemetry and tuning

One append-only JSONL, one row per turn, deduplicated to one sample per session
when read:

```json
{"class":"triage","confidence":0.82,"source":"rule","budget_soft":100000,
 "overrides":[{"from":"opus-5","to":"sonnet-5","agent_type":"general-purpose"}],
 "outcome":{"out_tokens":240000,"active_min":34,"interrupts":1,"exceeded":true}}
```

Plus `router report`, which recomputes per-class medians and p75 from telemetry
and **proposes** new budgets. Proposes — does not write. The config stays
reviewable; otherwise the router silently detunes itself.

## Rollout: shadow mode

`settings.yaml` carries `enforce: false`. `enforce` gates four behaviours, not
one, and they are deliberately not all off:

| behaviour | mechanism | shadow mode |
|---|---|---|
| contract injection | `additionalContext` | **on** — but it is only text |
| budget notice | `additionalContext` | **on** |
| model/effort rewrite for subagents | `updatedInput` | off |
| `spawn_task` denial | `permissionDecision: deny` | off |

In shadow mode the session can therefore ignore the contract completely and
effortlessly.

**Do not expect shadow mode to improve anything.** The corpus already shows
advisory does not work: one feature-implementation session had "subagent-driven"
explicitly in the prompt and still ran 1225 turns in the main loop. A session
that will not take direction from the user will not take it from a hook either.
If an improvement does appear, it is not evidence that the mechanism works.

Shadow mode measures the two things enforcement is not needed to measure:

1. **Classification accuracy** — does the class match what the session actually
   did?
2. **Budget realism** — the medians come from 40 sessions, and `pr_review` has
   n = 3. Enabling enforcement on those numbers would be enforcing noise.

It also serves as the control arm: advisory versus enforced on the same
taxonomy, so the difference is visible rather than only the absolute numbers.

Shadow mode is not free: the contract costs roughly 250 tokens per prompt, and
**if it classifies wrongly it causes sideways wandering by itself**. Hence teeth
come second. Disabling is one value in the config.

## Testing

- **Golden set from real data**: the 40 corpus prompts as a fixture with
  hand-checked classes, as a classifier regression test. Baseline is the
  measured 87.5% coverage; the test guards against a regex edit breaking
  another class. Local fixture, with a committed paraphrased fallback — see
  the public-repository split above.
- **Unit**: continuation detection, contract rendering, `enforce.py` (contract
  plus tool input -> expected `updatedInput`), incremental budget accounting.
- **Integration**: real hook payloads on stdin, asserting the shape of stdout
  JSON and the exit code. A hook that emits malformed JSON blocks the user's
  work, which makes this the most important test in the project.
- **Latency**: rules path under 100 ms, asserted as a test rather than a hope,
  since it runs before every prompt.

## Privacy

The repository is public. Everything derived from real sessions stays local:
`analysis/sessions.json`, `tests/fixtures/golden.local.jsonl`,
`config/classes.local.yaml`, `state/` and `telemetry.jsonl` are all gitignored.
Only reproducible code, the generic config and paraphrased fixtures are
committed. The spec itself quotes no real prompt text, internal identifier or
customer name.
