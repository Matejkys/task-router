import json
from pathlib import Path

from router.budget import read_increment


def _read_increment_pre_refactor(transcript_path: Path, offset: int) -> tuple[int, int]:
    """The implementation read_increment had before it became a wrapper over
    router.usage.read_usage. Kept verbatim so the equivalence test compares
    against the real prior behaviour rather than a paraphrase of it."""
    try:
        size = transcript_path.stat().st_size
        start = 0 if offset > size else offset
        with transcript_path.open("r", errors="replace") as fh:
            fh.seek(start)
            data = fh.read()
    except OSError:
        return 0, 0

    tokens = 0
    for line in data.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("type") != "assistant":
            continue
        usage = (rec.get("message") or {}).get("usage") or {}
        tokens += usage.get("output_tokens") or 0

    return tokens, size


def test_wrapper_matches_the_pre_refactor_reader_including_sidechains(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text(
        _line(100)
        + json.dumps({
            "type": "assistant", "isSidechain": True,
            "message": {"model": "sub", "usage": {"output_tokens": 33}},
        }) + "\n"
        + '{"type":"user","message":{"usage":{"output_tokens":999}}}\n'
        + "not json\n"
        + json.dumps({"type": "assistant", "message": None}) + "\n"
        + _line(4)
    )
    assert read_increment(t, 0) == _read_increment_pre_refactor(t, 0)
    # Sidechain records were counted before and must still be counted.
    assert read_increment(t, 0)[0] == 137

    _, offset = read_increment(t, 0)
    with t.open("a") as fh:
        fh.write(_line(9))
    assert read_increment(t, offset) == _read_increment_pre_refactor(t, offset)
    # And the degraded paths agree too.
    missing = tmp_path / "absent.jsonl"
    assert read_increment(missing, 0) == _read_increment_pre_refactor(missing, 0)


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
