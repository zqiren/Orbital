# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: run_session_end_routine is idempotent per session_id.

Two callers can fire the routine for the same session boundary:
  - loop.py fire-and-forget after loop exits (via _on_session_end callback)
  - agent_manager.new_session() synchronous pre-archival

Without a guard, this produces duplicate entries in SESSION_LOG / DECISIONS /
CONTEXT (which use append()). The guard lives inside run_session_end_routine
keyed by session_id; the second call short-circuits with an INFO log.

The guard is set AFTER successful writes only — a failed run must allow a
retry from the second caller.
"""

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent import workspace_files as wsf_module
from agent_os.agent.workspace_files import (
    WorkspaceFileManager,
    run_session_end_routine,
)


# ---------------------------------------------------------------------------
# Helpers (mirrors tests/unit/test_workspace_files.py patterns)
# ---------------------------------------------------------------------------

def _mock_session(messages=None, session_id="sess_test"):
    session = MagicMock()
    session.session_id = session_id
    session.get_messages.return_value = messages or [
        {"role": "user", "content": "Hello"},
    ]
    return session


def _mock_provider(response_text):
    provider = AsyncMock()
    resp = MagicMock()
    resp.text = response_text
    provider.complete.return_value = resp
    return provider


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
# Test 1: Duplicate session_id short-circuits (sequential)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_duplicate_session_id_short_circuits(tmp_path, caplog):
    """Calling twice with same session_id runs the LLM exactly once."""
    ws = WorkspaceFileManager(str(tmp_path))
    session = _mock_session(session_id="s1")
    provider = _mock_provider(_valid_llm_response("s1"))

    await run_session_end_routine(session, provider, ws, session_id="s1")
    assert provider.complete.call_count == 1

    with caplog.at_level(logging.INFO, logger="agent_os.agent.workspace_files"):
        await run_session_end_routine(session, provider, ws, session_id="s1")

    # Second call must NOT have hit the LLM
    assert provider.complete.call_count == 1
    # Short-circuit must log at INFO
    assert any(
        "already completed" in rec.message and rec.levelno == logging.INFO
        for rec in caplog.records
    ), f"Expected 'already completed' INFO log, got: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Test 2: Different session_ids both run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_different_session_ids_both_run(tmp_path):
    """Different session_ids do not interfere — both runs complete."""
    ws = WorkspaceFileManager(str(tmp_path))

    session1 = _mock_session(session_id="s1")
    provider1 = _mock_provider(_valid_llm_response("s1"))
    await run_session_end_routine(session1, provider1, ws, session_id="s1")

    session2 = _mock_session(session_id="s2")
    provider2 = _mock_provider(_valid_llm_response("s2"))
    await run_session_end_routine(session2, provider2, ws, session_id="s2")

    assert provider1.complete.call_count == 1
    assert provider2.complete.call_count == 1

    # Both session_ids are in the completion set
    assert "s1" in wsf_module._completed_session_ends
    assert "s2" in wsf_module._completed_session_ends


# ---------------------------------------------------------------------------
# Test 3: Concurrent calls for the same session_id — exactly one runs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_calls_same_session_id(tmp_path):
    """Two asyncio.gather tasks for the same session_id — LLM called once."""
    ws = WorkspaceFileManager(str(tmp_path))
    session = _mock_session(session_id="s_conc")

    # Shared provider so we can count total calls across both tasks.
    provider = _mock_provider(_valid_llm_response("s_conc"))

    results = await asyncio.gather(
        run_session_end_routine(session, provider, ws, session_id="s_conc"),
        run_session_end_routine(session, provider, ws, session_id="s_conc"),
        return_exceptions=True,
    )

    # Neither task raised
    for r in results:
        assert not isinstance(r, Exception), f"unexpected exception: {r!r}"

    # LLM was called exactly once — second task short-circuited on the guard.
    assert provider.complete.call_count == 1, (
        f"expected LLM call count=1, got {provider.complete.call_count}. "
        f"The idempotency guard did not prevent the duplicate run."
    )

    # No file corruption: each append-style file contains exactly ONE entry.
    session_log = ws.read("session_log") or ""
    decisions = ws.read("decisions") or ""
    context = ws.read("context") or ""

    assert session_log.count("## Session s_conc") == 1, (
        f"SESSION_LOG has duplicate entries:\n{session_log}"
    )
    assert decisions.count("## Decision s_conc") == 1, (
        f"DECISIONS has duplicate entries:\n{decisions}"
    )
    assert context.count("## People") == 1, (
        f"CONTEXT has duplicate entries:\n{context}"
    )

    # All files parseable as text (no truncation / torn writes).
    assert ws.read("state") == "# State\nstate-s_conc"
    assert ws.read("lessons") == "## Lesson s_conc\n**Problem:** p\n**Fix:** f"


# ---------------------------------------------------------------------------
# Test 4: Failed run does NOT mark complete — retry is allowed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_failed_run_allows_retry_with_same_session_id(tmp_path):
    """If the first call raises during the LLM phase, a second call with the
    same session_id must still run (no silent data loss)."""
    ws = WorkspaceFileManager(str(tmp_path))
    session = _mock_session(session_id="s_retry")

    # First provider raises on complete()
    failing_provider = AsyncMock()
    failing_provider.complete.side_effect = RuntimeError("LLM exploded")

    with pytest.raises(RuntimeError, match="LLM exploded"):
        await run_session_end_routine(
            session, failing_provider, ws, session_id="s_retry",
        )

    # The session_id must NOT be in the completion set after a failure.
    assert "s_retry" not in wsf_module._completed_session_ends

    # Second call with a working provider should run the routine fully.
    working_provider = _mock_provider(_valid_llm_response("retry_ok"))
    await run_session_end_routine(
        session, working_provider, ws, session_id="s_retry",
    )

    assert failing_provider.complete.call_count == 1
    assert working_provider.complete.call_count == 1

    # Files reflect the second (successful) call's output.
    assert ws.read("state") == "# State\nstate-retry_ok"
    session_log = ws.read("session_log") or ""
    assert "## Session retry_ok" in session_log
    assert session_log.count("## Session") == 1

    # Now marked complete.
    assert "s_retry" in wsf_module._completed_session_ends


# ---------------------------------------------------------------------------
# Test 5: session_id keyword-only and required
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_id_is_required_keyword_only(tmp_path):
    """run_session_end_routine must raise TypeError if session_id is missing
    (enforces deliberate caller updates; no silent regressions)."""
    ws = WorkspaceFileManager(str(tmp_path))
    session = _mock_session(session_id="s_kw")
    provider = _mock_provider(_valid_llm_response("s_kw"))

    with pytest.raises(TypeError):
        # Missing session_id kwarg
        await run_session_end_routine(session, provider, ws)
