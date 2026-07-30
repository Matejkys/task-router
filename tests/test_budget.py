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
