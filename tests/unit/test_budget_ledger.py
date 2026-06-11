# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the append-only token ledger (Budget Piece 1, Task 2).

Covers the writer in ``agent_os/budget/ledger.py``:

  - ledger path layout: ``{project_dir}/ledger/usage.jsonl``
  - event schema: disjoint token fields, ISO8601 UTC ts, source/provider/model
  - append-only (each call adds exactly one line; prior lines preserved)
  - directory auto-creation
  - resilience: an unwritable target must NOT raise (warn + continue)
"""

import json
import os
from datetime import datetime

import pytest

from agent_os.agent.project_paths import ProjectPaths
from agent_os.budget.ledger import (
    LedgerEvent,
    ledger_path,
    append_event,
)
from agent_os.budget.normalize import NormalizedUsage


def _read_lines(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestLedgerPath:
    def test_path_layout(self, tmp_path):
        """Ledger lives under the project state root:
        {workspace}/orbital/ledger/usage.jsonl."""
        p = ledger_path(str(tmp_path))
        assert p == os.path.join(
            str(tmp_path), "orbital", "ledger", "usage.jsonl",
        )

    def test_delegates_to_project_paths(self, tmp_path):
        """ProjectPaths.ledger_file is the single owner of the layout —
        ledger_path must agree with it byte-for-byte."""
        assert ledger_path(str(tmp_path)) == ProjectPaths(str(tmp_path)).ledger_file


class TestAppendEvent:
    def test_creates_dir_and_file(self, tmp_path):
        """First append materializes the ledger dir + file."""
        usage = NormalizedUsage(uncached_input=10, cache_read=2, cache_write=0, output=5)
        append_event(
            str(tmp_path),
            LedgerEvent(
                session_id="sess-1",
                source="management",
                provider="moonshot",
                model="kimi-k2.5",
                usage=usage,
            ),
        )
        p = ledger_path(str(tmp_path))
        assert os.path.isfile(p)
        lines = _read_lines(p)
        assert len(lines) == 1

    def test_event_schema_disjoint_fields(self, tmp_path):
        """The written event carries the disjoint token fields verbatim plus
        ts / session_id / source / provider / model."""
        usage = NormalizedUsage(uncached_input=100, cache_read=20, cache_write=5, output=50)
        append_event(
            str(tmp_path),
            LedgerEvent(
                session_id="sess-x",
                source="management",
                provider="anthropic",
                model="claude-sonnet",
                usage=usage,
            ),
        )
        ev = _read_lines(ledger_path(str(tmp_path)))[0]
        assert ev["session_id"] == "sess-x"
        assert ev["source"] == "management"
        assert ev["provider"] == "anthropic"
        assert ev["model"] == "claude-sonnet"
        assert ev["uncached_input"] == 100
        assert ev["cache_read"] == 20
        assert ev["cache_write"] == 5
        assert ev["output"] == 50
        # ts must be ISO8601 UTC parseable, and carry a timezone.
        parsed = datetime.fromisoformat(ev["ts"])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset().total_seconds() == 0

    def test_append_only_preserves_prior_lines(self, tmp_path):
        """Each append adds exactly one line; existing lines are untouched."""
        usage = NormalizedUsage(uncached_input=1, cache_read=0, cache_write=0, output=1)
        for i in range(3):
            append_event(
                str(tmp_path),
                LedgerEvent(
                    session_id=f"sess-{i}",
                    source="management",
                    provider="moonshot",
                    model="kimi-k2.5",
                    usage=usage,
                ),
            )
        lines = _read_lines(ledger_path(str(tmp_path)))
        assert len(lines) == 3
        assert [l["session_id"] for l in lines] == ["sess-0", "sess-1", "sess-2"]

    def test_unwritable_dir_does_not_raise(self, tmp_path):
        """A ledger-append failure must not propagate — warn and continue.

        Force failure by making project_dir a *file* so the ledger subdir
        cannot be created. append_event must swallow the error.
        """
        bogus = tmp_path / "not_a_dir"
        bogus.write_text("i am a file")
        usage = NormalizedUsage(uncached_input=1, cache_read=0, cache_write=0, output=1)
        # Must NOT raise.
        append_event(
            str(bogus),
            LedgerEvent(
                session_id="sess-fail",
                source="management",
                provider="moonshot",
                model="kimi-k2.5",
                usage=usage,
            ),
        )
        # No file was created (the parent is a regular file).
        assert not os.path.isdir(bogus)


class TestReportedCostSerialization:
    """P3-B: optional provider-reported cost fields round-trip through the
    on-disk record only when present (subscription/management rows omit them)."""

    def test_reported_cost_serialized_when_present(self, tmp_path):
        usage = NormalizedUsage(uncached_input=10, cache_read=2, cache_write=1, output=5)
        append_event(
            str(tmp_path),
            LedgerEvent(
                session_id="sess-rc",
                source="subagent:claude-code",
                provider="anthropic",
                model="claude-x",
                usage=usage,
                reported_cost=0.0123,
                reported_cost_currency="USD",
            ),
        )
        ev = _read_lines(ledger_path(str(tmp_path)))[0]
        assert ev["reported_cost"] == 0.0123
        assert ev["reported_cost_currency"] == "USD"
        # Tokens still serialize alongside the reported cost.
        assert ev["uncached_input"] == 10
        assert ev["output"] == 5

    def test_reported_cost_omitted_when_none(self, tmp_path):
        """A management row (no reported_cost) must NOT emit the keys at all —
        they are absent from the JSON, not present-but-null."""
        usage = NormalizedUsage(uncached_input=10, cache_read=0, cache_write=0, output=5)
        append_event(
            str(tmp_path),
            LedgerEvent(
                session_id="sess-mgmt",
                source="management",
                provider="anthropic",
                model="claude-x",
                usage=usage,
            ),
        )
        ev = _read_lines(ledger_path(str(tmp_path)))[0]
        assert "reported_cost" not in ev
        assert "reported_cost_currency" not in ev

    def test_default_fields_are_none(self):
        """The dataclass defaults keep management/loop call sites unchanged."""
        ev = LedgerEvent(
            session_id="s", source="management", provider="p", model="m",
            usage=NormalizedUsage(0, 0, 0, 0),
        )
        assert ev.reported_cost is None
        assert ev.reported_cost_currency is None
