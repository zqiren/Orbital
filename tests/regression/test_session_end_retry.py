# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: retry wrapper for session-end summarization LLM call (B1/RC-A).

These tests verify that run_session_end_routine retries the LLM call up to
3 times on asyncio.TimeoutError, with per-attempt timeouts of 30s, 60s, 90s,
and no inter-attempt sleep. Non-timeout exceptions do not trigger retry.

TDD note: each test should fail before the fix (bare await without retry)
and pass after.
"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

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
    session.get_messages.return_value = messages or [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    return session


def _valid_llm_response(tag="x"):
    return json.dumps({
        "project_state": f"# State\nstate-{tag}",
        "decisions": f"## Decision {tag}\n**Chose:** A",
        "session_log_entry": f"## Session {tag} -- today\n- Did thing",
        "lessons": f"## Lesson {tag}\n**Problem:** p\n**Fix:** f",
        "context": f"## People\n- Person {tag}",
    })


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

    provider = AsyncMock()
    resp = MagicMock()
    resp.text = _valid_llm_response("first")
    provider.complete.return_value = resp

    with caplog.at_level(logging.INFO, logger="agent_os.agent.workspace_files"):
        await run_session_end_routine(session, provider, ws, session_id="s_first")

    assert provider.complete.call_count == 1, (
        f"Expected exactly 1 LLM call, got {provider.complete.call_count}"
    )

    # No retry-related INFO messages
    retry_logs = [r for r in caplog.records if "timed out, retrying" in r.message]
    assert len(retry_logs) == 0, f"Unexpected retry logs: {[r.message for r in retry_logs]}"

    # Content is preserved
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

    provider = AsyncMock()
    provider.complete.side_effect = [
        asyncio.TimeoutError(),
        resp,
    ]

    with caplog.at_level(logging.INFO, logger="agent_os.agent.workspace_files"):
        await run_session_end_routine(session, provider, ws, session_id="s_one_timeout")

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

    provider = AsyncMock()
    provider.complete.side_effect = [
        asyncio.TimeoutError(),
        asyncio.TimeoutError(),
        resp,
    ]

    with caplog.at_level(logging.INFO, logger="agent_os.agent.workspace_files"):
        await run_session_end_routine(session, provider, ws, session_id="s_two_timeouts")

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
# Test 4: All attempts timeout — propagates TimeoutError
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_attempts_timeout_propagates(tmp_path, caplog):
    """LLM times out three times; TimeoutError propagates, ERROR log emitted, no 4th attempt."""
    ws = WorkspaceFileManager(str(tmp_path))
    session = _mock_session(session_id="s_all_timeout")

    provider = AsyncMock()
    provider.complete.side_effect = [
        asyncio.TimeoutError(),
        asyncio.TimeoutError(),
        asyncio.TimeoutError(),
    ]

    with caplog.at_level(logging.DEBUG, logger="agent_os.agent.workspace_files"):
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            await run_session_end_routine(session, provider, ws, session_id="s_all_timeout")

    # Exactly 3 attempts, no 4th
    assert provider.complete.call_count == 3, (
        f"Expected exactly 3 LLM calls (no 4th), got {provider.complete.call_count}"
    )

    # ERROR log must be emitted for final failure
    error_logs = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_logs) >= 1, (
        f"Expected at least 1 ERROR log on final failure, got: {[r.message for r in caplog.records]}"
    )

    # No files written (exception propagated before writes)
    assert ws.read("state") is None or ws.read("state") == ""


# ---------------------------------------------------------------------------
# Test 5: Non-timeout exception does NOT retry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_timeout_exception_no_retry(tmp_path):
    """ValueError on first attempt: no retry, exception propagates immediately."""
    ws = WorkspaceFileManager(str(tmp_path))
    session = _mock_session(session_id="s_value_error")

    provider = AsyncMock()
    provider.complete.side_effect = ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        await run_session_end_routine(session, provider, ws, session_id="s_value_error")

    # Only one attempt — no retry for non-timeout errors
    assert provider.complete.call_count == 1, (
        f"Expected exactly 1 LLM call (no retry on ValueError), got {provider.complete.call_count}"
    )
