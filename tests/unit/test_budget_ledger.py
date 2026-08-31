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
    last_context_usage,
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


class TestLastContextUsage:
    """Read path behind GET /sessions/{id}/context.

    The prompt size of the last management call IS the context in use: the
    four normalized fields are disjoint, so uncached_input + cache_read +
    cache_write reconstructs the whole prompt the provider billed for. No new
    bookkeeping is needed to show a context meter — the ledger already has it.
    """

    def _append(self, root, *, session_id, source="management",
                uncached=0, read=0, write=0, output=0,
                provider="anthropic", model="claude-sonnet"):
        append_event(
            root,
            LedgerEvent(
                session_id=session_id, source=source,
                provider=provider, model=model,
                usage=NormalizedUsage(
                    uncached_input=uncached, cache_read=read,
                    cache_write=write, output=output,
                ),
            ),
        )

    def test_none_when_no_ledger(self, tmp_path):
        """A session that has never made a call reports nothing, not zero."""
        assert last_context_usage(str(tmp_path), "sess-1") is None

    def test_sums_the_disjoint_input_fields(self, tmp_path):
        """used = uncached + cache_read + cache_write. Output is NOT context."""
        self._append(str(tmp_path), session_id="s1",
                     uncached=1000, read=300, write=50, output=900)
        got = last_context_usage(str(tmp_path), "s1")
        assert got["used"] == 1350
        assert got["provider"] == "anthropic"
        assert got["model"] == "claude-sonnet"

    def test_returns_the_latest_call_not_the_first(self, tmp_path):
        """Context is the CURRENT prompt size, so the last row wins."""
        self._append(str(tmp_path), session_id="s1", uncached=1000)
        self._append(str(tmp_path), session_id="s1", uncached=4000)
        assert last_context_usage(str(tmp_path), "s1")["used"] == 4000

    def test_ignores_other_sessions(self, tmp_path):
        """Sessions share one project ledger; the meter is per session."""
        self._append(str(tmp_path), session_id="s1", uncached=1000)
        self._append(str(tmp_path), session_id="s2", uncached=9000)
        assert last_context_usage(str(tmp_path), "s1")["used"] == 1000

    def test_ignores_sub_agent_rows(self, tmp_path):
        """Sub-agents append to the same ledger but carry their OWN context.

        Counting a worker's prompt as the management session's would make the
        meter jump on every dispatch.
        """
        self._append(str(tmp_path), session_id="s1", uncached=1000)
        self._append(str(tmp_path), session_id="s1",
                     source="subagent:claude-code", uncached=90_000)
        assert last_context_usage(str(tmp_path), "s1")["used"] == 1000

    def test_none_when_session_has_only_sub_agent_rows(self, tmp_path):
        self._append(str(tmp_path), session_id="s1",
                     source="subagent:codex", uncached=5000)
        assert last_context_usage(str(tmp_path), "s1") is None

    def test_survives_a_corrupt_line(self, tmp_path):
        """A torn write must not blank the meter."""
        self._append(str(tmp_path), session_id="s1", uncached=1000)
        with open(ledger_path(str(tmp_path)), "a", encoding="utf-8") as f:
            f.write("{not json\n")
        assert last_context_usage(str(tmp_path), "s1")["used"] == 1000

    def test_reports_the_model_that_actually_served(self, tmp_path):
        """Fallback rotation can change model mid-session; the window follows
        the model that served the LAST call, not the project's pinned one."""
        self._append(str(tmp_path), session_id="s1", uncached=1000,
                     provider="deepseek", model="deepseek-chat")
        self._append(str(tmp_path), session_id="s1", uncached=2000,
                     provider="moonshot", model="kimi-k2.5")
        got = last_context_usage(str(tmp_path), "s1")
        assert (got["provider"], got["model"]) == ("moonshot", "kimi-k2.5")
