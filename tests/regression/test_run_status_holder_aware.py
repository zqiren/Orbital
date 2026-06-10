# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase 1 (seam 3): get_run_status with no session_id must be holder-aware.

Root cause (REPORT-streaming-status-frontend.md, issue #2): the run-status
endpoint calls get_run_status(project_id) with no session_id, which resolved to
the ``"default"`` handle and returned ``idle`` even while a turn ran under a
non-default session (e.g. ``sess_…`` / a uuid). The REST poll then reverted the
WS-driven ``running``.

Fix: a no-session get_run_status resolves to the active-loop-slot holder, and
returns ``idle`` only when there is no holder. An explicit session_id still
targets that session (unchanged).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager


@pytest.fixture
def manager():
    ws = MagicMock()
    ws.broadcast = MagicMock()
    sub_agent_manager = MagicMock()
    sub_agent_manager.list_active = MagicMock(return_value=[])
    mgr = AgentManager(
        project_store=MagicMock(),
        ws_manager=ws,
        sub_agent_manager=sub_agent_manager,
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )
    return mgr


def _running_handle():
    h = MagicMock()
    h.session.is_stopped.return_value = False
    h.session._paused_for_approval = False
    h.task = MagicMock()
    h.task.done.return_value = False
    return h


def test_no_session_resolves_to_holder_running_under_uuid(manager):
    """Turn running under a uuid-keyed handle (NOT 'default'): no-session
    get_run_status reports running, not idle."""
    manager._handles[("proj_1", "proj_aaaa1111")] = _running_handle()

    assert manager.get_run_status("proj_1") == "running"


def test_no_session_returns_idle_when_no_holder(manager):
    """No running session anywhere -> idle."""
    assert manager.get_run_status("proj_1") == "idle"


def test_explicit_session_id_still_targets_that_session(manager):
    """An explicit session_id is unchanged: it reports that session's status,
    not the holder's."""
    manager._handles[("proj_1", "proj_aaaa1111")] = _running_handle()
    # A different, non-existent session id reports idle (no handle for it),
    # even though a different session is the holder.
    assert manager.get_run_status("proj_1", session_id="proj_other") == "idle"
