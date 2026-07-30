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
    assert proc.stderr == "", f"hook wrote a fail-open diagnostic: {proc.stderr}"

    rows = read_all(telemetry)
    assert len(rows) == 1
    row = rows[0]
    assert row["class"] == "triage"
    assert row["outcome"]["out_tokens"] == 250
    assert row["outcome"]["exceeded"] is True
    assert row["overrides"][0]["to"] == "claude-sonnet-5"
