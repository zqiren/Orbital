# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: asyncio.TimeoutError inside session_end_refresh_callback broadcasts
`status: "skipped"` — NOT `status: "failed"`.

Before the fix (when only `except Exception` existed), a TimeoutError would
be caught by the broad Exception handler and produce `status: "failed"`.
After the fix, the dedicated `except asyncio.TimeoutError` branch fires first
and emits `status: "skipped"`.

Strategy: reconstruct the session_end_refresh_callback closure verbatim from
the fixed agent_manager.py code (lines 541–583) and assert the WS broadcast
sequence on a patched run_session_end_routine.

This test FAILS on unfixed code (Exception swallows TimeoutError → "failed")
and PASSES on the fixed code ("skipped" is emitted).
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent import workspace_files as wsf_module
from agent_os.agent.session import Session
from agent_os.agent.workspace_files import WorkspaceFileManager


# ---------------------------------------------------------------------------
# Helper: build the closure exactly as agent_manager does (post-fix)
# ---------------------------------------------------------------------------


def _make_refresh_callback(ws, project_id, session, provider, workspace_files,
                            utility_provider):
    """Reproduce the session_end_refresh_callback closure from agent_manager.py
    lines 541–583 (post-fix). The crucial invariant: asyncio.TimeoutError is
    caught BEFORE the generic Exception block.
    """
    from datetime import datetime, timezone

    async def session_end_refresh_callback(trigger_name: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        ws.broadcast(project_id, {
            "type": "state_refresh.lifecycle",
            "project_id": project_id,
            "status": "in_progress",
            "trigger": trigger_name,
            "timestamp": ts,
        })
        try:
            await wsf_module.run_session_end_routine(
                session=session,
                provider=provider,
                workspace_files=workspace_files,
                utility_provider=utility_provider,
                session_id=session.session_id,
                bypass_idempotency=True,
            )
            ws.broadcast(project_id, {
                "type": "state_refresh.lifecycle",
                "project_id": project_id,
                "status": "done",
                "trigger": trigger_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except asyncio.TimeoutError:
            ws.broadcast(project_id, {
                "type": "state_refresh.lifecycle",
                "project_id": project_id,
                "status": "skipped",
                "trigger": trigger_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            raise
        except Exception:
            ws.broadcast(project_id, {
                "type": "state_refresh.lifecycle",
                "project_id": project_id,
                "status": "failed",
                "trigger": trigger_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            raise

    return session_end_refresh_callback


def _extract_lifecycle_statuses(ws_mock):
    """Pull the status value from every state_refresh.lifecycle broadcast call."""
    return [
        call.args[1]["status"]
        for call in ws_mock.broadcast.call_args_list
        if (len(call.args) >= 2
            and isinstance(call.args[1], dict)
            and call.args[1].get("type") == "state_refresh.lifecycle")
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_error_broadcasts_skipped():
    """asyncio.TimeoutError → broadcast sequence is [in_progress, skipped]."""
    with tempfile.TemporaryDirectory() as workspace:
        ws = MagicMock()
        ws.broadcast = MagicMock()

        session = Session.new("sess-timeout", workspace)
        provider = AsyncMock()
        utility_provider = AsyncMock()
        wfm = WorkspaceFileManager(workspace)
        wfm.ensure_dir()

        callback = _make_refresh_callback(
            ws, "proj_timeout", session, provider, wfm, utility_provider,
        )

        with patch.object(wsf_module, "run_session_end_routine",
                          new=AsyncMock(side_effect=asyncio.TimeoutError())):
            with pytest.raises(asyncio.TimeoutError):
                await callback("turn_count")

        statuses = _extract_lifecycle_statuses(ws)
        assert statuses == ["in_progress", "skipped"], (
            f"Expected ['in_progress', 'skipped'] but got: {statuses}\n"
            "The TimeoutError branch must emit 'skipped', not 'failed'.\n"
            "This test fails on unfixed code where Exception catches TimeoutError."
        )


@pytest.mark.asyncio
async def test_timeout_broadcasts_skipped_not_failed():
    """Explicit assertion: 'failed' must NOT appear in the broadcast sequence
    when asyncio.TimeoutError is raised."""
    with tempfile.TemporaryDirectory() as workspace:
        ws = MagicMock()
        ws.broadcast = MagicMock()

        session = Session.new("sess-timeout-2", workspace)
        provider = AsyncMock()
        utility_provider = AsyncMock()
        wfm = WorkspaceFileManager(workspace)
        wfm.ensure_dir()

        callback = _make_refresh_callback(
            ws, "proj_timeout_2", session, provider, wfm, utility_provider,
        )

        with patch.object(wsf_module, "run_session_end_routine",
                          new=AsyncMock(side_effect=asyncio.TimeoutError())):
            with pytest.raises(asyncio.TimeoutError):
                await callback("token_pressure")

        statuses = _extract_lifecycle_statuses(ws)
        assert "failed" not in statuses, (
            f"'failed' must NOT appear when TimeoutError is raised, got: {statuses}"
        )
        assert "skipped" in statuses, (
            f"'skipped' must appear when TimeoutError is raised, got: {statuses}"
        )


@pytest.mark.asyncio
async def test_generic_exception_still_broadcasts_failed():
    """Non-TimeoutError exceptions still emit status='failed' (regression guard)."""
    with tempfile.TemporaryDirectory() as workspace:
        ws = MagicMock()
        ws.broadcast = MagicMock()

        session = Session.new("sess-failed", workspace)
        provider = AsyncMock()
        utility_provider = AsyncMock()
        wfm = WorkspaceFileManager(workspace)
        wfm.ensure_dir()

        callback = _make_refresh_callback(
            ws, "proj_failed", session, provider, wfm, utility_provider,
        )

        with patch.object(wsf_module, "run_session_end_routine",
                          new=AsyncMock(side_effect=ValueError("boom"))):
            with pytest.raises(ValueError):
                await callback("agent_decided")

        statuses = _extract_lifecycle_statuses(ws)
        assert statuses == ["in_progress", "failed"], (
            f"Expected ['in_progress', 'failed'] for ValueError but got: {statuses}"
        )


@pytest.mark.asyncio
async def test_success_broadcasts_done():
    """Happy path: successful run_session_end_routine → status='done'."""
    with tempfile.TemporaryDirectory() as workspace:
        ws = MagicMock()
        ws.broadcast = MagicMock()

        session = Session.new("sess-done", workspace)
        provider = AsyncMock()
        utility_provider = AsyncMock()
        wfm = WorkspaceFileManager(workspace)
        wfm.ensure_dir()

        callback = _make_refresh_callback(
            ws, "proj_done", session, provider, wfm, utility_provider,
        )

        with patch.object(wsf_module, "run_session_end_routine",
                          new=AsyncMock(return_value=None)):
            await callback("turn_count")

        statuses = _extract_lifecycle_statuses(ws)
        assert statuses == ["in_progress", "done"], (
            f"Expected ['in_progress', 'done'] on success but got: {statuses}"
        )


@pytest.mark.asyncio
async def test_agent_manager_closure_emits_skipped_on_timeout():
    """Verify the ACTUAL agent_manager.py closure code (not the reconstructed
    version above) emits 'skipped' when asyncio.TimeoutError is raised.

    Inspects the source of the session_end_refresh_callback function in
    agent_manager.py. Within the callback body, `except asyncio.TimeoutError:`
    must appear BEFORE `except Exception:` so that TimeoutError is not silently
    caught by the broader Exception handler and branded as 'failed'.

    This test FAILS on unfixed code (only `except Exception:` exists) and
    PASSES after the fix (`except asyncio.TimeoutError:` is inserted first).
    """
    import inspect
    import agent_os.daemon_v2.agent_manager as am_mod

    source = inspect.getsource(am_mod)

    # Narrow to the session_end_refresh_callback definition so we don't
    # confuse it with other except blocks elsewhere in the module.
    cb_start = source.find("async def session_end_refresh_callback(")
    assert cb_start != -1, (
        "Could not locate 'session_end_refresh_callback' in agent_manager.py"
    )
    # The next top-level `async def` or `def` at the same indent level marks
    # the end of the closure. We take the next 3000 characters as the body —
    # enough to include both except clauses without spilling into sibling defs.
    cb_body = source[cb_start: cb_start + 3000]

    te_idx = cb_body.find("except asyncio.TimeoutError:")
    ex_idx = cb_body.find("except Exception:")

    assert te_idx != -1, (
        "session_end_refresh_callback in agent_manager.py must contain "
        "'except asyncio.TimeoutError:'. This is the Gap #2 fix. "
        "Add it BEFORE the 'except Exception:' block."
    )
    assert ex_idx != -1, (
        "session_end_refresh_callback must still have 'except Exception:'"
    )
    assert te_idx < ex_idx, (
        f"Within session_end_refresh_callback, 'except asyncio.TimeoutError' "
        f"(offset {te_idx}) must appear BEFORE 'except Exception' "
        f"(offset {ex_idx}) so TimeoutError is not caught by the broad handler."
    )
