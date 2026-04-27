# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: bypass_idempotency=False (default) preserves existing /new behavior.

The /new command calls run_session_end_routine with no bypass_idempotency kwarg
(i.e. bypass_idempotency=False by default). This must still:
1. Short-circuit on a duplicate session_id (guard fires as before)
2. Mark the session complete in _completed_session_ends

The bypass_idempotency=True path (periodic refresh) must NOT mark complete,
so the /new guard is not corrupted by a mid-session refresh.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent import workspace_files as wsf_module
from agent_os.agent.workspace_files import WorkspaceFileManager, run_session_end_routine


def _mock_session(session_id="sess_new"):
    session = MagicMock()
    session.session_id = session_id
    session.get_messages.return_value = [{"role": "user", "content": "Hello"}]
    return session


def _mock_provider(tag="ok"):
    provider = AsyncMock()
    resp = MagicMock()
    resp.text = json.dumps({
        "project_state": f"# State {tag}",
        "decisions": f"## Dec {tag}\n**Chose:** X",
        "session_log_entry": f"## Session {tag} -- today\n- did it",
        "lessons": f"1. Lesson {tag}",
        "context": f"- ctx {tag}",
    })
    provider.complete.return_value = resp
    return provider


@pytest.fixture(autouse=True)
def _reset_idempotency_set():
    wsf_module._completed_session_ends.clear()
    yield
    wsf_module._completed_session_ends.clear()


@pytest.mark.asyncio
async def test_default_bypass_false_short_circuits_on_duplicate(tmp_path):
    """Default (bypass_idempotency=False) still short-circuits on duplicate session_id."""
    ws = WorkspaceFileManager(str(tmp_path))
    session = _mock_session("s_new")
    provider = _mock_provider("first")

    # First call — runs normally
    await run_session_end_routine(session, provider, ws, session_id="s_new")
    assert provider.complete.call_count == 1
    assert "s_new" in wsf_module._completed_session_ends

    # Second call with same session_id — must short-circuit (bypass_idempotency defaults to False)
    provider2 = _mock_provider("second")
    await run_session_end_routine(session, provider2, ws, session_id="s_new")
    assert provider2.complete.call_count == 0, (
        "Second call with same session_id must short-circuit when bypass_idempotency=False"
    )


@pytest.mark.asyncio
async def test_bypass_true_does_not_add_to_completion_set(tmp_path):
    """bypass_idempotency=True must NOT mark session as complete."""
    ws = WorkspaceFileManager(str(tmp_path))
    session = _mock_session("s_bypass")
    provider = _mock_provider("bypass")

    await run_session_end_routine(
        session, provider, ws,
        session_id="s_bypass",
        bypass_idempotency=True,
    )

    # The session must NOT be in the set — periodic refresh should not block /new
    assert "s_bypass" not in wsf_module._completed_session_ends, (
        "bypass_idempotency=True must not mark session complete; "
        "/new must still be able to run the routine"
    )
    assert provider.complete.call_count == 1, "LLM was called once"


@pytest.mark.asyncio
async def test_bypass_true_runs_even_after_default_call(tmp_path):
    """bypass_idempotency=True fires even if the default-path already ran for this session."""
    ws = WorkspaceFileManager(str(tmp_path))
    session = _mock_session("s_both")

    # First: /new runs with default (bypass=False)
    provider_new = _mock_provider("new")
    await run_session_end_routine(session, provider_new, ws, session_id="s_both")
    assert "s_both" in wsf_module._completed_session_ends

    # Now a periodic refresh tries to run (bypass=True) — must succeed despite guard
    provider_refresh = _mock_provider("refresh")
    await run_session_end_routine(
        session, provider_refresh, ws,
        session_id="s_both",
        bypass_idempotency=True,
    )
    assert provider_refresh.complete.call_count == 1, (
        "bypass_idempotency=True must run even after default-path completed for same session_id"
    )


@pytest.mark.asyncio
async def test_default_path_marks_complete_after_bypass(tmp_path):
    """After a bypass refresh, the default path can still mark complete (one-way gate)."""
    ws = WorkspaceFileManager(str(tmp_path))
    session = _mock_session("s_order")

    # Periodic refresh (bypass=True) runs first
    provider_refresh = _mock_provider("refresh")
    await run_session_end_routine(
        session, provider_refresh, ws,
        session_id="s_order",
        bypass_idempotency=True,
    )
    assert "s_order" not in wsf_module._completed_session_ends

    # /new runs second (bypass=False default) — should mark complete
    provider_new = _mock_provider("new")
    await run_session_end_routine(session, provider_new, ws, session_id="s_order")
    assert "s_order" in wsf_module._completed_session_ends
    assert provider_new.complete.call_count == 1
