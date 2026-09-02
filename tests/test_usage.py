import json

from router.usage import (
    Totals,
    UsageRecord,
    aggregate,
    cost_usd,
    read_usage,
    subagent_transcripts,
)


def _assistant(
    model="claude-sonnet-5",
    ts="2026-09-01T10:00:00Z",
    side=False,
    input_tokens=0,
    output_tokens=0,
    cache_read=0,
    cache_creation=None,
    cache_creation_input_tokens=None,
):
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read,
    }
    if cache_creation is not None:
        usage["cache_creation"] = cache_creation
    if cache_creation_input_tokens is not None:
        usage["cache_creation_input_tokens"] = cache_creation_input_tokens
    rec = {
        "type": "assistant",
        "timestamp": ts,
        "isSidechain": side,
        "message": {"model": model, "usage": usage},
    }
    return json.dumps(rec) + "\n"


def test_read_usage_from_scratch(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text(_assistant(input_tokens=10, output_tokens=20))
    records, offset = read_usage(t, 0)
    assert len(records) == 1
    r = records[0]
    assert r.model == "claude-sonnet-5"
    assert r.input_tokens == 10
    assert r.output_tokens == 20
    assert r.is_sidechain is False
    assert offset == t.stat().st_size


def test_read_usage_offset_continuation(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text(_assistant(output_tokens=5))
    _, offset = read_usage(t, 0)
    with t.open("a") as fh:
        fh.write(_assistant(output_tokens=7))
    records, new_offset = read_usage(t, offset)
    assert len(records) == 1
    assert records[0].output_tokens == 7
    assert new_offset > offset


def test_missing_file_is_empty(tmp_path):
    records, offset = read_usage(tmp_path / "absent.jsonl", 0)
    assert records == []
    assert offset == 0


def test_bad_lines_skipped(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text('{"type":"user"}\n' + "not json\n" + _assistant(output_tokens=3))
    records, _ = read_usage(t, 0)
    assert len(records) == 1
    assert records[0].output_tokens == 3


def test_cache_creation_ttl_split(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text(
        _assistant(
            cache_creation={
                "ephemeral_5m_input_tokens": 100,
                "ephemeral_1h_input_tokens": 50,
            }
        )
    )
    records, _ = read_usage(t, 0)
    r = records[0]
    assert r.cache_write_5m == 100
    assert r.cache_write_1h == 50


def test_cache_creation_flat_falls_back_to_default_ttl(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text(_assistant(cache_creation_input_tokens=42))
    records, _ = read_usage(t, 0, default_cache_ttl="5m")
    r = records[0]
    assert r.cache_write_5m == 42
    assert r.cache_write_1h == 0


def test_cache_creation_flat_default_ttl_1h(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text(_assistant(cache_creation_input_tokens=42))
    records, _ = read_usage(t, 0, default_cache_ttl="1h")
    r = records[0]
    assert r.cache_write_5m == 0
    assert r.cache_write_1h == 42


def test_sidechain_flagged(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text(_assistant(side=True, output_tokens=9))
    records, _ = read_usage(t, 0)
    assert records[0].is_sidechain is True


def test_subagent_transcripts_discovery(tmp_path):
    transcript = tmp_path / "sess123.jsonl"
    transcript.write_text("{}\n")
    sub_dir = tmp_path / "sess123" / "subagents"
    sub_dir.mkdir(parents=True)
    (sub_dir / "agent-1.jsonl").write_text("{}\n")
    (sub_dir / "agent-2.jsonl").write_text("{}\n")
    found = subagent_transcripts(transcript)
    assert len(found) == 2
    assert all(p.suffix == ".jsonl" for p in found)


def test_subagent_transcripts_none(tmp_path):
    transcript = tmp_path / "sess999.jsonl"
    transcript.write_text("{}\n")
    assert subagent_transcripts(transcript) == []


def test_aggregate_sums_by_model():
    records = [
        UsageRecord(
            model="claude-sonnet-5",
            timestamp="2026-09-01T10:00:00Z",
            is_sidechain=False,
            input_tokens=10,
            cache_read=1,
            cache_write_5m=2,
            cache_write_1h=0,
            output_tokens=20,
        ),
        UsageRecord(
            model="claude-sonnet-5",
            timestamp="2026-09-01T10:01:00Z",
            is_sidechain=False,
            input_tokens=5,
            cache_read=0,
            cache_write_5m=0,
            cache_write_1h=0,
            output_tokens=8,
        ),
        UsageRecord(
            model="claude-opus-5",
            timestamp="2026-09-01T10:02:00Z",
            is_sidechain=False,
            input_tokens=1,
            cache_read=0,
            cache_write_5m=0,
            cache_write_1h=0,
            output_tokens=1,
        ),
    ]
    totals = aggregate(records)
    assert totals["claude-sonnet-5"] == Totals(
        input_tokens=15, cache_read=1, cache_write_5m=2, cache_write_1h=0,
        output_tokens=28, calls=2,
    )
    assert totals["claude-opus-5"].calls == 1


def test_cost_usd_arithmetic():
    totals = {
        "claude-sonnet-5": Totals(
            input_tokens=1_000_000, cache_read=0, cache_write_5m=0,
            cache_write_1h=0, output_tokens=1_000_000, calls=1,
        ),
    }
    pricing = {
        "claude-sonnet-5": {
            "input": 2, "output": 10, "cache_read": 0.2,
            "cache_write_5m": 2.5, "cache_write_1h": 4,
        },
    }
    breakdown = cost_usd(totals, pricing, zero_cost_models=frozenset())
    assert breakdown.per_model["claude-sonnet-5"] == 12.0
    assert breakdown.total == 12.0
    assert breakdown.unpriced == {}


def test_cost_usd_unknown_model_is_unpriced_not_priced():
    totals = {
        "some-new-model": Totals(
            input_tokens=1000, cache_read=0, cache_write_5m=0,
            cache_write_1h=0, output_tokens=1000, calls=1,
        ),
    }
    breakdown = cost_usd(totals, {}, zero_cost_models=frozenset())
    assert breakdown.total == 0.0
    assert breakdown.per_model == {}
    assert "some-new-model" in breakdown.unpriced
    assert breakdown.unpriced["some-new-model"].output_tokens == 1000


def test_cost_usd_zero_cost_model_not_unpriced():
    totals = {
        "<synthetic>": Totals(
            input_tokens=1000, cache_read=0, cache_write_5m=0,
            cache_write_1h=0, output_tokens=1000, calls=1,
        ),
    }
    breakdown = cost_usd(totals, {}, zero_cost_models=frozenset({"<synthetic>"}))
    assert breakdown.total == 0.0
    assert breakdown.unpriced == {}
    assert breakdown.per_model.get("<synthetic>", 0.0) == 0.0
