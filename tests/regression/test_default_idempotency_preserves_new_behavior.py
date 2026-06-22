# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: bypass_idempotency=False (default) preserves existing /new behavior.

The /new command calls run_session_end_routine with no bypass_idempotency kwarg
(i.e. bypass_idempotency=False by default). This must still:
1. Short-circuit on a duplicate session_id (idempotency guard fires as before)
2. Mark the session complete in _completed_session_ends

The bypass_idempotency=True path (periodic refresh) must NOT mark complete,
so the /new guard is not corrupted by a mid-session refresh.

Layer-1 redesign added a SECOND gate in front of the LLM call: a persisted
NO-DELTA gate (``_has_cleanup_delta`` / the ``orbital/.memory_cleanup.json``
marker). It fires AFTER the idempotency guard but BEFORE the LLM, and a
successful routine writes the marker on exit. So a second identical call would
be skipped by the no-delta gate even when the idempotency guard would have let
it through. These tests therefore force a Layer-1 delta between calls when the
intent is to prove the *idempotency* behavior in isolation, and add a dedicated
test for the gate-interplay itself.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent import workspace_files as wsf_module
from agent_os.agent.workspace_files import WorkspaceFileManager, run_session_end_routine


def _mock_session(session_id="sess_new"):
    session = MagicMock()
    session.session_id = session_id
    session.session_uuid = session_id
    session.get_messages.return_value = [{"role": "user", "content": "Hello"}]
    return session


def _mock_provider(tag="ok"):
    provider = AsyncMock()
    resp = MagicMock()
    # New Layer-1 session-end contract: project_state / decisions / lessons / index.
    # (session_log_entry and "context" are gone — SESSION_LOG removed, CONTEXT→INDEX.)
    resp.text = json.dumps({
        "project_state": f"# State {tag}",
        "decisions": f"## 2026-06-18: Dec {tag}\n**Chose:** X\n",
        "lessons": f"1. **Lesson {tag}.** learn it.\n",
        "index": f"- ctx {tag}",
    })
    provider.complete.return_value = resp
    return provider


def _force_layer1_delta(ws, marker="poke"):
    """Make the no-delta gate see a change so the next routine actually runs.

    A successful run_session_end_routine writes the cleanup marker on exit, so
    a subsequent identical call would be a no-op via the no-delta gate. Writing
    a Layer-1 file bumps its mtime past the marker => delta present again. This
    mirrors real activity (the agent edited STATE/DECISIONS during the session).
    """
    ws.write("state", f"# scratch {marker}")
    assert wsf_module._has_cleanup_delta(ws) is True


@pytest.fixture(autouse=True)
def _reset_idempotency_set():
    wsf_module._completed_session_ends.clear()
    yield
    wsf_module._completed_session_ends.clear()


@pytest.mark.asyncio
async def test_default_bypass_false_short_circuits_on_duplicate(tmp_path):
    """Default (bypass_idempotency=False) still short-circuits on duplicate session_id.

    We force a Layer-1 delta before the second call so the short-circuit can ONLY
    be attributed to the idempotency guard — not the no-delta gate, which would
    otherwise skip the LLM independently.
    """
    ws = WorkspaceFileManager(str(tmp_path))
    session = _mock_session("s_new")
    provider = _mock_provider("first")

    # First call — fresh workspace (no marker) => delta present => runs normally.
    await run_session_end_routine(session, provider, ws, session_uuid="s_new")
    assert provider.complete.call_count == 1
    assert "s_new" in wsf_module._completed_session_ends

    # Second call with same session_id — must short-circuit on the IDEMPOTENCY
    # guard, even though a real delta now exists (gate would otherwise allow it).
    _force_layer1_delta(ws)
    provider2 = _mock_provider("second")
    await run_session_end_routine(session, provider2, ws, session_uuid="s_new")
    assert provider2.complete.call_count == 0, (
        "Second call with same session_id must short-circuit on the idempotency "
        "guard when bypass_idempotency=False, even with a Layer-1 delta present"
    )


@pytest.mark.asyncio
async def test_bypass_true_does_not_add_to_completion_set(tmp_path):
    """bypass_idempotency=True must NOT mark session as complete."""
    ws = WorkspaceFileManager(str(tmp_path))
    session = _mock_session("s_bypass")
    provider = _mock_provider("bypass")

    await run_session_end_routine(
        session, provider, ws,
        session_uuid="s_bypass",
        bypass_idempotency=True,
    )

    # The session must NOT be in the set — periodic refresh should not block /new.
    assert "s_bypass" not in wsf_module._completed_session_ends, (
        "bypass_idempotency=True must not mark session complete; "
        "/new must still be able to run the routine"
    )
    assert provider.complete.call_count == 1, "LLM was called once"


@pytest.mark.asyncio
async def test_bypass_true_runs_even_after_default_call(tmp_path):
    """bypass_idempotency=True fires even if the default-path already ran for this session.

    A Layer-1 delta is introduced before the refresh so the no-delta gate does
    not mask the idempotency-bypass behavior under test.
    """
    ws = WorkspaceFileManager(str(tmp_path))
    session = _mock_session("s_both")

    # First: /new runs with default (bypass=False).
    provider_new = _mock_provider("new")
    await run_session_end_routine(session, provider_new, ws, session_uuid="s_both")
    assert "s_both" in wsf_module._completed_session_ends

    # A subsequent session touched a Layer-1 file => the no-delta gate would
    # permit the routine; the only thing that could block it is the idempotency
    # guard, which bypass_idempotency=True must override.
    _force_layer1_delta(ws)
    provider_refresh = _mock_provider("refresh")
    await run_session_end_routine(
        session, provider_refresh, ws,
        session_uuid="s_both",
        bypass_idempotency=True,
    )
    assert provider_refresh.complete.call_count == 1, (
        "bypass_idempotency=True must run even after default-path completed for "
        "same session_id (idempotency guard overridden, delta present)"
    )


@pytest.mark.asyncio
async def test_default_path_marks_complete_after_bypass(tmp_path):
    """After a bypass refresh, the default path can still mark complete (one-way gate)."""
    ws = WorkspaceFileManager(str(tmp_path))
    session = _mock_session("s_order")

    # Periodic refresh (bypass=True) runs first — fresh workspace => delta present.
    provider_refresh = _mock_provider("refresh")
    await run_session_end_routine(
        session, provider_refresh, ws,
        session_uuid="s_order",
        bypass_idempotency=True,
    )
    assert "s_order" not in wsf_module._completed_session_ends

    # /new runs second (bypass=False default). The bypass run wrote the cleanup
    # marker, so force a fresh delta to prove the default path runs AND records
    # completion (not merely skipped by the no-delta gate).
    _force_layer1_delta(ws)
    provider_new = _mock_provider("new")
    await run_session_end_routine(session, provider_new, ws, session_uuid="s_order")
    assert "s_order" in wsf_module._completed_session_ends
    assert provider_new.complete.call_count == 1


@pytest.mark.asyncio
async def test_no_delta_gate_skips_llm_even_when_idempotency_would_allow(tmp_path):
    """Idempotency + no-delta interplay: the no-delta gate skips the LLM on its own.

    bypass_idempotency=True clears the idempotency guard, so the ONLY remaining
    gate is the no-delta gate. With the marker freshly written and no Layer-1
    change since, a bypass refresh must be a no-op (no LLM call). This is the
    second-gate behavior the Layer-1 redesign added in front of the LLM.
    """
    ws = WorkspaceFileManager(str(tmp_path))
    session = _mock_session("s_nodelta")

    # First bypass run consolidates and writes the cleanup marker.
    provider_first = _mock_provider("first")
    await run_session_end_routine(
        session, provider_first, ws,
        session_uuid="s_nodelta",
        bypass_idempotency=True,
    )
    assert provider_first.complete.call_count == 1
    # Marker now matches current mtimes => no un-consolidated delta.
    assert wsf_module._has_cleanup_delta(ws) is False

    # Second bypass run with NOTHING changed: idempotency guard is bypassed, but
    # the no-delta gate must short-circuit before any LLM call.
    provider_second = _mock_provider("second")
    await run_session_end_routine(
        session, provider_second, ws,
        session_uuid="s_nodelta",
        bypass_idempotency=True,
    )
    assert provider_second.complete.call_count == 0, (
        "no-delta gate must skip the LLM even when bypass_idempotency=True "
        "clears the idempotency guard"
    )

    # And once a Layer-1 file changes again, the bypass refresh runs the LLM.
    _force_layer1_delta(ws)
    provider_third = _mock_provider("third")
    await run_session_end_routine(
        session, provider_third, ws,
        session_uuid="s_nodelta",
        bypass_idempotency=True,
    )
    assert provider_third.complete.call_count == 1, (
        "a fresh Layer-1 delta must re-arm the routine after a no-delta skip"
    )
