"""Aggregate Claude Code session transcripts into per-session metrics.

Reads ~/.claude/projects/**/*.jsonl modified in the last N days and emits one
JSON record per session with model usage, token counts, wall-clock span and the
first user prompt. Output is aggregate only -- raw transcript text never leaves
this script except for the truncated first prompt.
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from glob import glob

DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 14
ROOT = os.path.expanduser("~/.claude/projects")
CUTOFF = datetime.now(timezone.utc) - timedelta(days=DAYS)


def parse_ts(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def summarize(path):
    models = Counter()
    sidechain_models = Counter()
    efforts = Counter()
    tools = Counter()
    subagent_types = Counter()
    skills = Counter()
    out_tokens = 0
    cache_read = 0
    cache_create = 0
    assistant_turns = 0
    sidechain_turns = 0
    user_turns = 0
    interrupts = 0
    denials = 0
    first_prompt = None
    title = None
    ts_first = None
    ts_last = None
    ts_all = []
    git_branch = None
    cwd = None

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue

            rtype = rec.get("type")
            if rtype == "custom-title":
                title = rec.get("customTitle")
                continue

            ts = parse_ts(rec.get("timestamp"))
            if ts:
                ts_all.append(ts)
                if ts_first is None or ts < ts_first:
                    ts_first = ts
                if ts_last is None or ts > ts_last:
                    ts_last = ts

            cwd = cwd or rec.get("cwd")
            git_branch = git_branch or rec.get("gitBranch")
            side = bool(rec.get("isSidechain"))

            if rtype == "assistant":
                msg = rec.get("message", {})
                model = msg.get("model", "?")
                if side:
                    sidechain_turns += 1
                    sidechain_models[model] += 1
                else:
                    assistant_turns += 1
                    models[model] += 1
                if rec.get("effort"):
                    efforts[rec["effort"]] += 1
                usage = msg.get("usage", {}) or {}
                out_tokens += usage.get("output_tokens", 0) or 0
                cache_read += usage.get("cache_read_input_tokens", 0) or 0
                cache_create += usage.get("cache_creation_input_tokens", 0) or 0
                if rec.get("attributionSkill"):
                    skills[rec["attributionSkill"]] += 1
                for block in msg.get("content", []) or []:
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    name = block.get("name", "?")
                    tools[name] += 1
                    if name in ("Task", "Agent"):
                        inp = block.get("input", {}) or {}
                        subagent_types[inp.get("subagent_type", "default")] += 1

            elif rtype == "user":
                msg = rec.get("message", {})
                content = msg.get("content")
                is_tool_result = False
                if isinstance(content, list):
                    is_tool_result = any(
                        isinstance(b, dict) and b.get("type") == "tool_result"
                        for b in content
                    )
                if rec.get("toolDenialKind"):
                    denials += 1
                if is_tool_result or rec.get("isMeta") or side:
                    continue
                text = content
                if isinstance(content, list):
                    text = " ".join(
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                if not isinstance(text, str) or not text.strip():
                    continue
                stripped = text.strip()
                if "[Request interrupted" in stripped:
                    interrupts += 1
                    continue
                if stripped.startswith("<") or stripped.startswith("Caveat:"):
                    continue
                user_turns += 1
                if first_prompt is None:
                    first_prompt = stripped[:600]

    if ts_first is None or ts_last is None or first_prompt is None:
        return None
    if ts_last < CUTOFF:
        return None

    # Active time: sum gaps below 5 minutes, so idle overnight gaps do not count.
    ts_all.sort()
    active = 0.0
    for a, b in zip(ts_all, ts_all[1:]):
        gap = (b - a).total_seconds()
        if gap <= 300:
            active += gap

    return {
        "file": path,
        "project": os.path.basename(os.path.dirname(path)),
        "cwd": cwd,
        "branch": git_branch,
        "title": title,
        "start": ts_first.isoformat(),
        "span_min": round((ts_last - ts_first).total_seconds() / 60, 1),
        "active_min": round(active / 60, 1),
        "main_models": dict(models),
        "sidechain_models": dict(sidechain_models),
        "efforts": dict(efforts),
        "assistant_turns": assistant_turns,
        "sidechain_turns": sidechain_turns,
        "user_turns": user_turns,
        "interrupts": interrupts,
        "denials": denials,
        "out_tokens": out_tokens,
        "cache_read": cache_read,
        "cache_create": cache_create,
        "tool_calls": sum(tools.values()),
        "top_tools": dict(tools.most_common(8)),
        "subagents": dict(subagent_types),
        "skills": dict(skills),
        "first_prompt": first_prompt,
    }


def summarize_subagents(session_path):
    """Subagent transcripts live in <dir>/<sessionId>/subagents/agent-*.jsonl."""
    base = session_path[:-6]  # strip .jsonl
    paths = glob(os.path.join(base, "subagents", "*.jsonl"))
    models = Counter()
    efforts = Counter()
    out_tokens = 0
    turns = 0
    runs = []
    for path in paths:
        run_models = Counter()
        run_out = 0
        run_turns = 0
        ts = []
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                t = parse_ts(rec.get("timestamp"))
                if t:
                    ts.append(t)
                if rec.get("type") != "assistant":
                    continue
                msg = rec.get("message", {})
                model = msg.get("model", "?")
                models[model] += 1
                run_models[model] += 1
                if rec.get("effort"):
                    efforts[rec["effort"]] += 1
                usage = msg.get("usage", {}) or {}
                tok = usage.get("output_tokens", 0) or 0
                out_tokens += tok
                run_out += tok
                turns += 1
                run_turns += 1
        if run_turns:
            ts.sort()
            runs.append(
                {
                    "models": dict(run_models),
                    "turns": run_turns,
                    "out_tokens": run_out,
                    "span_min": round((ts[-1] - ts[0]).total_seconds() / 60, 1)
                    if len(ts) > 1
                    else 0.0,
                }
            )
    return {
        "sub_files": len(paths),
        "sub_models": dict(models),
        "sub_efforts": dict(efforts),
        "sub_out_tokens": out_tokens,
        "sub_turns": turns,
        "sub_runs": runs,
    }


rows = []
for path in glob(os.path.join(ROOT, "*", "*.jsonl")):
    try:
        if datetime.fromtimestamp(os.path.getmtime(path), timezone.utc) < CUTOFF:
            continue
        row = summarize(path)
        if row:
            row.update(summarize_subagents(path))
    except Exception as exc:  # noqa: BLE001 - keep going over a 400-file corpus
        print(f"ERR {path}: {exc}", file=sys.stderr)
        continue
    if row:
        rows.append(row)

rows.sort(key=lambda r: r["start"])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions.json")
with open(out, "w") as fh:
    json.dump(rows, fh, indent=1)
print(f"sessions={len(rows)} -> {out}")
