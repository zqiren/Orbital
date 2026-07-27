# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: retry wrapper for session-end summarization LLM call (B1/RC-A).

These tests verify that ``run_session_end_routine`` retries the LLM dedup/merge
call up to 3 times on ``asyncio.TimeoutError``, with per-attempt timeouts of
30s, 60s, 90s, and no inter-attempt sleep. Non-timeout exceptions do not trigger
a retry.

Layer-1 memory redesign note: the session-end pass is now positioned as the
*cleanup*, not the state record. The LLM dedup/merge is best-effort — when it
times out (all attempts) or raises a non-timeout error, the routine NO LONGER
propagates the exception. Instead it logs and runs the DETERMINISTIC hard-cap
backstop (demote-to-archive for durable files), which never deletes durable
content. The retry/no-retry *decision* is the invariant under test here; the
old "exception propagates to the caller" contract is gone, replaced by "the
deterministic backstop still runs."
"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.testutils import streamable

from agent_os.agent import memory_entries as _mem
from agent_os.agent import workspace_files as wsf_module
from agent_os.agent.workspace_files import (
    WorkspaceFileManager,
    run_session_end_routine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_session(messages=None, session_id="sess_retry_test"):
    session = MagicMock()
    session.session_id = session_id
    session.session_uuid = session_id
    session.get_messages.return_value = messages or [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    return session


def _valid_llm_response(tag="x"):
    # New Layer-1 contract: keys are project_state / decisions / lessons / index.
    # (session_log_entry and "context" are gone.)
    return json.dumps({
        "project_state": f"# State\nstate-{tag}",
        "decisions": f"## 2026-06-18: Decision {tag}\n**Chose:** A\n\n",
        "lessons": f"1. **Lesson {tag}.** Problem p, fix f.\n",
        "index": f"# INDEX\n- PROJECT_STATE.md — current scratchpad ({tag}).\n",
    })


def _decisions(n: int, *, touched=None, body="x" * 120) -> str:
    """n stamped DECISIONS entries, oldest-first, with explicit touched dates."""
    out = []
    for i in range(1, n + 1):
        t = touched(i) if touched else f"2026-01-{i:02d}"
        out.append(
            f"## 2026-01-{i:02d}: Decision {i} "
            f"<!--mem id:d{i} created:2026-01-{i:02d} touched:{t}-->\n"
            f"**Chose:** option {i}\n**Reason:** {body}\n**Rejected:** alt {i}\n\n"
        )
    return "".join(out)


@pytest.fixture(autouse=True)
def _reset_completion_set():
    """Reset the module-level idempotency set between tests."""
    wsf_module._completed_session_ends.clear()
    yield
    wsf_module._completed_session_ends.clear()


# ---------------------------------------------------------------------------
# Test 1: Succeeds on first attempt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_succeeds_first_attempt(tmp_path, caplog):
    """LLM returns immediately; exactly 1 call, no retries logged."""
    ws = WorkspaceFileManager(str(tmp_path))
    session = _mock_session(session_id="s_first")

    provider = streamable(AsyncMock())
    resp = MagicMock()
    resp.text = _valid_llm_response("first")
    provider.complete.return_value = resp

    with caplog.at_level(logging.INFO, logger="agent_os.agent.workspace_files"):
        await run_session_end_routine(session, provider, ws, session_uuid="s_first")

    assert provider.complete.call_count == 1, (
        f"Expected exactly 1 LLM call, got {provider.complete.call_count}"
    )

    # No retry-related INFO messages
    retry_logs = [r for r in caplog.records if "timed out, retrying" in r.message]
    assert len(retry_logs) == 0, f"Unexpected retry logs: {[r.message for r in retry_logs]}"

    # Content is preserved (project_state overwrites STATE verbatim — not stamped)
    assert ws.read("state") == "# State\nstate-first"


# ---------------------------------------------------------------------------
# Test 2: Succeeds after one timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_succeeds_after_one_timeout(tmp_path, caplog):
    """LLM times out once, then returns; exactly 2 calls, INFO log for retry."""
    ws = WorkspaceFileManager(str(tmp_path))
    session = _mock_session(session_id="s_one_timeout")

    resp = MagicMock()
    resp.text = _valid_llm_response("one_timeout")

    provider = streamable(AsyncMock())
    provider.complete.side_effect = [
        asyncio.TimeoutError(),
        resp,
    ]

    with caplog.at_level(logging.INFO, logger="agent_os.agent.workspace_files"):
        await run_session_end_routine(session, provider, ws, session_uuid="s_one_timeout")

    assert provider.complete.call_count == 2, (
        f"Expected exactly 2 LLM calls, got {provider.complete.call_count}"
    )

    # INFO log about retry
    retry_logs = [r for r in caplog.records if "timed out, retrying" in r.message]
    assert len(retry_logs) >= 1, (
        f"Expected at least 1 INFO 'timed out, retrying' log, got: {[r.message for r in caplog.records]}"
    )
    assert all(r.levelno == logging.INFO for r in retry_logs), (
        "Retry logs must be at INFO level, not WARNING"
    )

    # Content preserved
    assert ws.read("state") == "# State\nstate-one_timeout"


# ---------------------------------------------------------------------------
# Test 3: Succeeds after two timeouts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_succeeds_after_two_timeouts(tmp_path, caplog):
    """LLM times out twice, returns on third; exactly 3 calls, content preserved."""
    ws = WorkspaceFileManager(str(tmp_path))
    session = _mock_session(session_id="s_two_timeouts")

    resp = MagicMock()
    resp.text = _valid_llm_response("two_timeouts")

    provider = streamable(AsyncMock())
    provider.complete.side_effect = [
        asyncio.TimeoutError(),
        asyncio.TimeoutError(),
        resp,
    ]

    with caplog.at_level(logging.INFO, logger="agent_os.agent.workspace_files"):
        await run_session_end_routine(session, provider, ws, session_uuid="s_two_timeouts")

    assert provider.complete.call_count == 3, (
        f"Expected exactly 3 LLM calls, got {provider.complete.call_count}"
    )

    retry_logs = [r for r in caplog.records if "timed out, retrying" in r.message]
    assert len(retry_logs) >= 2, (
        f"Expected at least 2 retry logs, got: {[r.message for r in caplog.records]}"
    )
    assert all(r.levelno == logging.INFO for r in retry_logs), (
        "Retry logs must be at INFO level, not WARNING"
    )

    # Content preserved
    assert ws.read("state") == "# State\nstate-two_timeouts"


# ---------------------------------------------------------------------------
# Test 4: All attempts timeout — does NOT raise; deterministic backstop runs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_attempts_timeout_runs_backstop(tmp_path, caplog, monkeypatch):
    """LLM times out three times.

    New behavior: ``run_session_end_routine`` does NOT propagate the TimeoutError.
    It logs an ERROR after the 3rd attempt (no 4th attempt) and then runs the
    DETERMINISTIC hard-cap backstop, which demotes over-budget durable content to
    the archive without deleting anything.
    """
    # Tiny budget so the deterministic hard cap is forced to demote — makes the
    # backstop observable.
    monkeypatch.setitem(_mem.FILE_BUDGETS, "decisions", {"soft": 120, "hard": 180})
    ws = WorkspaceFileManager(str(tmp_path))
    ws.ensure_dir()
    # Pre-seed an over-budget, stamped DECISIONS file. This also guarantees a
    # cleanup delta (no marker yet) so the LLM pass actually runs.
    ws.write("decisions", _decisions(12, body="x" * 20))
    session = _mock_session(session_id="s_all_timeout")

    provider = streamable(AsyncMock())
    provider.complete.side_effect = [
        asyncio.TimeoutError(),
        asyncio.TimeoutError(),
        asyncio.TimeoutError(),
    ]

    with caplog.at_level(logging.DEBUG, logger="agent_os.agent.workspace_files"):
        # Must NOT raise — the routine swallows the timeout and runs the backstop.
        await run_session_end_routine(
            session, provider, ws, session_uuid="s_all_timeout"
        )

    # Exactly 3 attempts, no 4th
    assert provider.complete.call_count == 3, (
        f"Expected exactly 3 LLM calls (no 4th), got {provider.complete.call_count}"
    )

    # ERROR log must be emitted for the final (all-attempts) timeout
    error_logs = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_logs) >= 1, (
        f"Expected at least 1 ERROR log on final failure, got: {[r.message for r in caplog.records]}"
    )

    # Deterministic backstop ran despite the timeout: live file capped, durable
    # content demoted to the archive (never deleted).
    kept = ws.read("decisions") or ""
    assert _mem.est_tokens(kept) <= _mem.FILE_BUDGETS["decisions"]["hard"], (
        "Hard cap did not trim the live DECISIONS file"
    )
    archive = ws.read("decisions_archive") or ""
    assert "Decision" in archive, (
        "Deterministic backstop did not demote any entry to the archive"
    )


# ---------------------------------------------------------------------------
# Test 5: Non-timeout exception does NOT retry (and does NOT propagate)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_timeout_exception_no_retry(tmp_path, caplog, monkeypatch):
    """ValueError on first attempt: no retry of the LLM call.

    The retry ladder is reserved for timeouts. A non-timeout error breaks out of
    the loop on the first attempt (so exactly 1 call, no retry). Under the new
    Layer-1 contract the error is NOT propagated — it is logged and the
    deterministic backstop still runs.
    """
    monkeypatch.setitem(_mem.FILE_BUDGETS, "decisions", {"soft": 120, "hard": 180})
    ws = WorkspaceFileManager(str(tmp_path))
    ws.ensure_dir()
    ws.write("decisions", _decisions(12, body="x" * 20))
    session = _mock_session(session_id="s_value_error")

    provider = streamable(AsyncMock())
    provider.complete.side_effect = ValueError("bad input")

    with caplog.at_level(logging.DEBUG, logger="agent_os.agent.workspace_files"):
        # Must NOT raise — non-timeout errors are swallowed, backstop runs.
        await run_session_end_routine(
            session, provider, ws, session_uuid="s_value_error"
        )

    # Only one attempt — non-timeout errors do NOT enter the retry ladder.
    assert provider.complete.call_count == 1, (
        f"Expected exactly 1 LLM call (no retry on ValueError), got {provider.complete.call_count}"
    )

    # The error was logged, not raised.
    failure_logs = [r for r in caplog.records if "bad input" in r.message]
    assert failure_logs, (
        f"Expected the ValueError to be logged, got: {[r.message for r in caplog.records]}"
    )

    # Deterministic backstop still ran.
    kept = ws.read("decisions") or ""
    assert _mem.est_tokens(kept) <= _mem.FILE_BUDGETS["decisions"]["hard"], (
        "Hard cap did not trim the live DECISIONS file"
    )
    assert "Decision" in (ws.read("decisions_archive") or ""), (
        "Deterministic backstop did not demote any entry to the archive"
    )
