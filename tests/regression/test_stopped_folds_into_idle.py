# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Step 5 — `stopped` folds into `idle`.

A session that was stopped is, from the runtime-status perspective, just an
idle session: not currently holding the slot, nothing running. The
externally-visible status (what the frontend reads via /status and
list_sessions) must be `idle`, never `stopped`. The historical
`last_terminal_event` breadcrumb may still record "stopped"; the live status
does not.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle


def _make_manager(tmp_path):
    mgr = AgentManager(
        project_store=MagicMock(),
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
        platform_provider=None,
        registry=MagicMock(),
        setup_engine=MagicMock(),
        settings_store=None,
        credential_store=None,
    )
    mgr._state_file = tmp_path / "daemon-state.json"
    return mgr


def _stopped_handle(workspace="/tmp/ws"):
    session = MagicMock()
    session.is_stopped.return_value = True
    session._paused_for_approval = False
    session._messages = []
    session.session_uuid = "proj_dead0000"
    task = MagicMock()
    task.done.return_value = True
    return ProjectHandle(
        session=session, loop=MagicMock(), provider=MagicMock(),
        registry=MagicMock(), context_manager=MagicMock(), interceptor=MagicMock(),
        task=task, config_snapshot={"workspace": workspace},
        started_at="2026-01-01T00:00:00+00:00",
    )


def test_get_run_status_stopped_session_reports_idle(tmp_path):
    mgr = _make_manager(tmp_path)
    mgr._handles[("proj", "default")] = _stopped_handle()
    assert mgr.get_run_status("proj") == "idle"


def test_list_sessions_stopped_session_reports_idle(tmp_path):
    mgr = _make_manager(tmp_path)
    # No project_store workspace → disk merge is a no-op; only the in-memory
    # handle's status matters here.
    mgr._project_store.get_project.return_value = None
    mgr._handles[("proj", "default")] = _stopped_handle()
    out = mgr.list_sessions("proj")
    statuses = {s["status"] for s in out}
    assert statuses == {"idle"}, out
    assert "stopped" not in statuses
