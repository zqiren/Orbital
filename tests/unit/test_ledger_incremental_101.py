# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 101 §4-C — incremental ledger reads.

The budget guard calls ``spend()`` at the top of every loop iteration and the
append hook calls it again after every LLM response. Each call parsed the
whole project-lifetime ``usage.jsonl`` on the event loop. The ledger is
append-only, so reads now parse only the bytes past the last complete line;
a shrink, same-size rewrite or inode swap forces a full re-read. The contract
these tests pin: the incremental answer is ALWAYS the full re-read's answer.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

import agent_os.agent.pricing as pricing_mod
from agent_os.budget import ledger as ledger_mod
from agent_os.budget.guard import evaluate_budget
from agent_os.budget.ledger import last_context_usage, ledger_path, spend


NOW = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _fresh_caches():
    for attr in ("_pricing_cache", "_full_providers_cache", "_override_cache",
                 "_override_mtime"):
        setattr(pricing_mod, attr, None)
    ledger_mod._clear_cache()
    yield
    ledger_mod._clear_cache()
    for attr in ("_pricing_cache", "_full_providers_cache", "_override_cache",
                 "_override_mtime"):
        setattr(pricing_mod, attr, None)


@pytest.fixture
def override_file(tmp_path):
    path = str(tmp_path / "ov.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"anthropic": {"_currency": "USD", "claude-x": {
            "input_per_1m": 3.0, "cached_input_per_1m": 0.3,
            "cache_write_per_1m": 3.75, "output_per_1m": 15.0}}}, f)
    return path


def _rec(i, source="management", minutes_ago=5, **tokens):
    return {
        "ts": (NOW - timedelta(minutes=minutes_ago)).isoformat(),
        "session_id": f"s{i % 3}",
        "source": source,
        "provider": "anthropic",
        "model": "claude-x",
        "uncached_input": tokens.get("uncached", 1000 + i),
        "cache_read": tokens.get("cache_read", 10 * i),
        "cache_write": 0,
        "output": tokens.get("output", 100 + i),
    }


def _write(path, text, mode="a"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        f.write(text)


def _lines(recs):
    return "".join(json.dumps(r) + "\n" for r in recs)


def _both(project, override_file, **kw):
    """(incremental answer, full re-read answer) for the same query."""
    kw.setdefault("now", NOW)
    inc = spend(project, "daily", override_path=override_file, **kw)
    ledger_mod._clear_cache()
    full = spend(project, "daily", override_path=override_file, **kw)
    return inc, full


@pytest.fixture
def parse_counter(monkeypatch):
    calls = []
    real = ledger_mod._parse_line

    def counting(raw):
        calls.append(raw)
        return real(raw)

    monkeypatch.setattr(ledger_mod, "_parse_line", counting)
    return calls


def test_append_parses_only_the_new_lines(tmp_path, override_file, parse_counter):
    project = str(tmp_path)
    path = ledger_path(project)
    _write(path, _lines(_rec(i) for i in range(50)))

    spend(project, "daily", now=NOW, override_path=override_file)
    assert len(parse_counter) == 50

    spend(project, "daily", now=NOW, override_path=override_file)
    assert len(parse_counter) == 50  # unchanged file: no parsing at all

    _write(path, _lines([_rec(50), _rec(51)]))
    spend(project, "daily", now=NOW, override_path=override_file,
          sources=["management"])
    assert len(parse_counter) == 52  # only the two appended lines


def test_incremental_equals_full_reread_across_appends(tmp_path, override_file):
    project = str(tmp_path)
    path = ledger_path(project)
    _write(path, _lines(_rec(i) for i in range(20)))
    spend(project, "daily", now=NOW, override_path=override_file)  # warm

    for batch in range(4):
        _write(path, _lines([
            _rec(100 + batch),
            _rec(200 + batch, source="subagent:codex"),
            _rec(300 + batch, minutes_ago=60 * 24 * 3),  # outside the window
        ]))
        for sources in (None, ["management"]):
            spend(project, "daily", now=NOW, override_path=override_file,
                  sources=sources)  # fold forward into the memo
        for sources in (None, ["management"]):
            inc, full = _both(project, override_file, sources=sources)
            assert inc == full
            spend(project, "daily", now=NOW, override_path=override_file,
                  sources=sources)  # re-warm after _both cleared the cache


def test_shrink_forces_a_full_reread(tmp_path, override_file, parse_counter):
    project = str(tmp_path)
    path = ledger_path(project)
    _write(path, _lines(_rec(i) for i in range(30)))
    before = spend(project, "daily", now=NOW, override_path=override_file)

    _write(path, _lines(_rec(i) for i in range(5)), mode="w")  # rewritten smaller
    parse_counter.clear()
    after = spend(project, "daily", now=NOW, override_path=override_file)

    assert len(parse_counter) == 5
    assert after["breakdown"][0]["tokens"]["uncached_input"] == sum(
        1000 + i for i in range(5))
    assert after != before
    inc, full = _both(project, override_file)
    assert inc == full


def test_same_size_rewrite_is_detected(tmp_path, override_file):
    project = str(tmp_path)
    path = ledger_path(project)
    _write(path, _lines([_rec(1, output=111)]))
    spend(project, "daily", now=NOW, override_path=override_file)

    _write(path, _lines([_rec(1, output=222)]), mode="w")  # same byte length
    os.utime(path, ns=(1, 1))  # filesystems can round two quick writes together
    result = spend(project, "daily", now=NOW, override_path=override_file)
    assert result["breakdown"][0]["tokens"]["output"] == 222


def test_replaced_file_is_detected(tmp_path, override_file):
    project = str(tmp_path)
    path = ledger_path(project)
    _write(path, _lines(_rec(i) for i in range(3)))
    spend(project, "daily", now=NOW, override_path=override_file)

    tmp = path + ".new"
    _write(tmp, _lines(_rec(i) for i in range(10)))
    os.replace(tmp, path)  # new inode, larger
    inc, full = _both(project, override_file)
    assert inc == full
    assert inc["breakdown"][0]["tokens"]["uncached_input"] == sum(
        1000 + i for i in range(10))


def test_unterminated_last_line_counts_once(tmp_path, override_file):
    """A line without its newline yet is counted (like the full read) but not
    committed: when the writer finishes it, it must not count twice."""
    project = str(tmp_path)
    path = ledger_path(project)
    _write(path, _lines([_rec(0)]) + json.dumps(_rec(1)))  # no trailing \n
    inc, full = _both(project, override_file)
    assert inc == full
    assert inc["breakdown"][0]["tokens"]["uncached_input"] == 1000 + 1001

    _write(path, "\n" + _lines([_rec(2)]))
    spend(project, "daily", now=NOW, override_path=override_file)
    inc, full = _both(project, override_file)
    assert inc == full
    assert inc["breakdown"][0]["tokens"]["uncached_input"] == 1000 + 1001 + 1002


def test_half_written_line_is_skipped_until_complete(tmp_path, override_file):
    project = str(tmp_path)
    path = ledger_path(project)
    line = json.dumps(_rec(1))
    _write(path, _lines([_rec(0)]) + line[:20])  # writer mid-line
    first = spend(project, "daily", now=NOW, override_path=override_file)
    assert first["breakdown"][0]["tokens"]["uncached_input"] == 1000

    _write(path, line[20:] + "\n")
    second = spend(project, "daily", now=NOW, override_path=override_file)
    assert second["breakdown"][0]["tokens"]["uncached_input"] == 1000 + 1001


def test_malformed_lines_skipped_and_warned_once(tmp_path, override_file, caplog):
    project = str(tmp_path)
    path = ledger_path(project)
    _write(path, _lines([_rec(0)]) + "not json\n" + _lines([_rec(1)]))
    with caplog.at_level("WARNING", logger="agent_os.budget.ledger"):
        spend(project, "daily", now=NOW, override_path=override_file)
        spend(project, "daily", now=NOW, override_path=override_file)
    warnings = [r for r in caplog.records if "malformed ledger" in r.getMessage()]
    assert len(warnings) == 1
    inc, full = _both(project, override_file)
    assert inc == full


def test_window_rollover_recomputes_from_cached_rows(tmp_path, override_file):
    project = str(tmp_path)
    path = ledger_path(project)
    _write(path, _lines([_rec(0, minutes_ago=60 * 13), _rec(1, minutes_ago=5)]))
    today = spend(project, "daily", now=NOW, override_path=override_file)
    tomorrow = spend(project, "daily", now=NOW + timedelta(days=1),
                     override_path=override_file)
    assert today["breakdown"]
    assert tomorrow["breakdown"] == []


def test_last_context_usage_sees_appends(tmp_path):
    project = str(tmp_path)
    path = ledger_path(project)
    _write(path, _lines([_rec(0)]))
    assert last_context_usage(project, "s0")["used"] == 1000
    _write(path, _lines([_rec(3, uncached=7, cache_read=5)]))  # session s0 again
    assert last_context_usage(project, "s0")["used"] == 12


def test_budget_guard_still_trips_at_the_limit(tmp_path, override_file):
    """Fixture ledger: 1M uncached management tokens at $3/1M = $3.00."""
    project = str(tmp_path)
    path = ledger_path(project)
    cfg = {"budget_limit_usd": 3.0, "budget_period": "daily",
           "budget_action": "pause"}

    def guard():
        return evaluate_budget(
            project, cfg, now=NOW,
            spend_fn=lambda *a, **k: spend(*a, override_path=override_file, **k))

    _write(path, _lines([_rec(0, uncached=500_000, cache_read=0, output=0)]))
    assert guard().tripped is False
    # Sub-agent usage is display-only: never counts toward the guard.
    _write(path, _lines([_rec(1, source="subagent:codex", uncached=900_000,
                              cache_read=0, output=0)]))
    assert guard().tripped is False
    _write(path, _lines([_rec(2, uncached=500_000, cache_read=0, output=0)]))
    decision = guard()
    assert decision.tripped is True
    assert decision.spend == pytest.approx(3.0)
