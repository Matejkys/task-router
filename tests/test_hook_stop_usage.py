"""End-to-end coverage of the Stop hook's per-model usage accounting.

Run as a subprocess like tests/test_end_to_end.py, because hookio.run fails
open: an exception leaves returncode 0 and empty stdout, and only stderr tells
a broken hook from a legitimate no-op.
"""

import json
import subprocess
import sys
from pathlib import Path

from router.state import SessionState, save_state
from router.telemetry import read_all

REPO = Path(__file__).resolve().parents[1]

MAIN_MODEL = "claude-fable-5-1"
SUB_MODEL = "claude-sonnet-5"


def _assistant(model: str, out_tokens: int, sidechain: bool = False) -> str:
    return json.dumps({
        "type": "assistant",
        "isSidechain": sidechain,
        "message": {
            "model": model,
            "usage": {
                "input_tokens": 10,
                "output_tokens": out_tokens,
                "cache_read_input_tokens": 1000,
                "cache_creation_input_tokens": 100,
            },
        },
    }) + "\n"


def _run_stop(session: str, transcript: Path, env: dict) -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO / "hooks" / "stop.py")],
        input=json.dumps({
            "session_id": session, "hook_event_name": "Stop",
            "transcript_path": str(transcript),
        }),
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO), **env},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == "", f"stop.py wrote a fail-open diagnostic: {proc.stderr}"


def _seed(state_dir: Path, session: str) -> None:
    save_state(state_dir, SessionState(
        session_id=session, cls="triage", confidence=0.9, source="rule:x",
        budget_soft=100_000,
    ))


def _fixture(tmp_path: Path, session: str) -> tuple[Path, Path, Path, dict]:
    state_dir = tmp_path / "state"
    telemetry = tmp_path / "telemetry.jsonl"
    transcript = tmp_path / f"{session}.jsonl"
    transcript.write_text(_assistant(MAIN_MODEL, 100) + _assistant(SUB_MODEL, 7, True))
    subagents = tmp_path / session / "subagents"
    subagents.mkdir(parents=True)
    (subagents / "agent-1.jsonl").write_text(_assistant(SUB_MODEL, 40))
    _seed(state_dir, session)
    env = {
        "TASK_ROUTER_STATE_DIR": str(state_dir),
        "TASK_ROUTER_TELEMETRY": str(telemetry),
        "TASK_ROUTER_ENFORCE": "0",
    }
    return state_dir, telemetry, transcript, env


def test_stop_records_main_and_sub_usage_with_a_timestamp(tmp_path):
    session = "usage1"
    _, telemetry, transcript, env = _fixture(tmp_path, session)

    _run_stop(session, transcript, env)

    row = read_all(telemetry)[-1]
    assert row["ts"].endswith("+00:00")
    block = row["outcome"]["usage"]
    assert block["main"][MAIN_MODEL]["output_tokens"] == 100
    # The main transcript's sidechain record and the subagent file both count
    # as subagent usage, under the same model.
    assert block["sub"][SUB_MODEL]["output_tokens"] == 47
    assert block["sub"][SUB_MODEL]["calls"] == 2
    assert block["unpriced_models"] == []
    assert block["cost_usd"]["total"] > 0
    assert (
        abs(block["cost_usd"]["main"] + block["cost_usd"]["sub"]
            - block["cost_usd"]["total"]) < 1e-9
    )
    # The base row survives unchanged: out_tokens still counts sidechain too.
    assert row["outcome"]["out_tokens"] == 107
    assert row["class"] == "triage"


def test_second_stop_adds_only_the_increment(tmp_path):
    session = "usage2"
    _, telemetry, transcript, env = _fixture(tmp_path, session)
    _run_stop(session, transcript, env)

    with transcript.open("a") as fh:
        fh.write(_assistant(MAIN_MODEL, 5))
    # A subagent dispatched after the first Stop, plus new bytes in the old one.
    subagents = tmp_path / session / "subagents"
    (subagents / "agent-2.jsonl").write_text(_assistant(SUB_MODEL, 3))
    with (subagents / "agent-1.jsonl").open("a") as fh:
        fh.write(_assistant(SUB_MODEL, 1))

    _run_stop(session, transcript, env)

    rows = read_all(telemetry)
    assert len(rows) == 2
    block = rows[-1]["outcome"]["usage"]
    assert block["main"][MAIN_MODEL]["output_tokens"] == 105  # 100 + 5
    assert block["sub"][SUB_MODEL]["output_tokens"] == 51  # 47 + 3 + 1
    assert rows[-1]["outcome"]["out_tokens"] == 112
    assert rows[-1]["ts"] >= rows[0]["ts"]


def test_broken_pricing_still_writes_the_base_row(tmp_path):
    session = "usage3"
    _, telemetry, transcript, env = _fixture(tmp_path, session)
    broken = tmp_path / "pricing.yaml"
    broken.write_text("models: [this is: not: a mapping\n")

    _run_stop(session, transcript, {**env, "TASK_ROUTER_PRICING": str(broken)})

    row = read_all(telemetry)[-1]
    assert "usage" not in row["outcome"]
    assert row["outcome"]["out_tokens"] == 107
    assert row["class"] == "triage"
    assert row["outcome"]["exceeded"] is False


def test_a_corrupt_usage_store_still_writes_the_base_row(tmp_path):
    # Pricing loads fine here; the failure is inside the usage computation
    # itself (a usage store that is not shaped like Totals). The base row must
    # survive that too, not just a broken pricing file.
    session = "usage5"
    state_dir = tmp_path / "state"
    telemetry = tmp_path / "telemetry.jsonl"
    transcript = tmp_path / f"{session}.jsonl"
    transcript.write_text(_assistant(MAIN_MODEL, 100))
    save_state(state_dir, SessionState(
        session_id=session, cls="triage", confidence=0.9, source="rule:x",
        budget_soft=100_000,
        main_usage={MAIN_MODEL: "not a totals dict"},  # ty: ignore[invalid-argument-type]
    ))

    _run_stop(session, transcript, {
        "TASK_ROUTER_STATE_DIR": str(state_dir),
        "TASK_ROUTER_TELEMETRY": str(telemetry),
    })

    row = read_all(telemetry)[-1]
    assert "usage" not in row["outcome"]
    assert row["outcome"]["out_tokens"] == 100
    assert row["class"] == "triage"


def test_unknown_model_is_reported_not_silently_priced(tmp_path):
    session = "usage4"
    state_dir = tmp_path / "state"
    telemetry = tmp_path / "telemetry.jsonl"
    transcript = tmp_path / f"{session}.jsonl"
    transcript.write_text(_assistant("claude-from-the-future", 10))
    _seed(state_dir, session)

    _run_stop(session, transcript, {
        "TASK_ROUTER_STATE_DIR": str(state_dir),
        "TASK_ROUTER_TELEMETRY": str(telemetry),
    })

    block = read_all(telemetry)[-1]["outcome"]["usage"]
    assert block["unpriced_models"] == ["claude-from-the-future"]
    assert block["cost_usd"]["total"] == 0.0
