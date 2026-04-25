# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration test for TASK-cancel-arch-06: POST /agents/{pid}/new-session
stops sub-agent adapters after session-end summarization.

Uses the inline TestClient pattern from test_new_session_retry.py.  Drives the
real HTTP path so the route → AgentManager.new_session → stop_all chain is
exercised end-to-end at the request level.

Test scenario: project with a mock sub-agent adapter registered in
_adapters[project_id]. POST /new-session; assert:
  - HTTP 200 within 30s
  - _adapters[project_id] is empty after rotation (or stop_all was called)
  - New session has no inherited sub-agent state
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.routes import agents_v2
from agent_os.daemon_v2.project_store import project_dir_name as _project_dir_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent_manager(stop_all_mock=None):
    """Build a minimal AgentManager with a controllable stop_all mock."""
    from agent_os.daemon_v2.agent_manager import AgentManager

    project_store = MagicMock()
    ws = MagicMock()
    ws.broadcast = MagicMock()

    sub_agent_mgr = MagicMock()
    sub_agent_mgr.list_active = MagicMock(return_value=[])
    sub_agent_mgr.stop = AsyncMock()
    sub_agent_mgr.stop_all = stop_all_mock if stop_all_mock is not None else AsyncMock()
    # Initialize _adapters so we can populate it in tests
    sub_agent_mgr._adapters = {}

    activity_translator = MagicMock()
    process_manager = MagicMock()
    process_manager.set_session = MagicMock()

    mgr = AgentManager(
        project_store=project_store,
        ws_manager=ws,
        sub_agent_manager=sub_agent_mgr,
        activity_translator=activity_translator,
        process_manager=process_manager,
    )
    return mgr, ws, project_store, sub_agent_mgr


def _build_test_client(mgr, project_store):
    """Wire the real agents_v2 router into a TestClient.

    Returns (client, saved_globals). Caller must restore saved_globals via
    _restore_agents_v2().
    """
    saved = {
        "_project_store": agents_v2._project_store,
        "_agent_manager": agents_v2._agent_manager,
        "_ws_manager": agents_v2._ws_manager,
        "_sub_agent_manager": agents_v2._sub_agent_manager,
    }
    agents_v2.configure(
        project_store=project_store,
        agent_manager=mgr,
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
    )
    app = FastAPI()
    app.include_router(agents_v2.router)
    client = TestClient(app)
    return client, saved


def _restore_agents_v2(saved: dict) -> None:
    agents_v2._project_store = saved["_project_store"]
    agents_v2._agent_manager = saved["_agent_manager"]
    agents_v2._ws_manager = saved["_ws_manager"]
    agents_v2._sub_agent_manager = saved["_sub_agent_manager"]


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

def test_new_session_stops_subagents_via_http():
    """POST /api/v2/agents/{pid}/new-session returns 200 and stop_all is
    invoked (including the post-session-end T06 call) within 30s.

    The mock sub-agent adapter registered in _adapters must be stopped;
    new session has no inherited sub-agent state.
    """
    from agent_os.daemon_v2.agent_manager import ProjectHandle
    from agent_os.agent.session import Session

    project_id = "proj_subagent_lifecycle_test"
    project_name = "SubAgent Lifecycle Test"

    stop_all_calls: list[str] = []

    stop_all_mock = AsyncMock()

    async def _recording_stop_all(pid: str) -> None:
        stop_all_calls.append(pid)

    stop_all_mock.side_effect = _recording_stop_all

    mgr, _ws, project_store, sub_agent_mgr = _make_agent_manager(stop_all_mock=stop_all_mock)

    with tempfile.TemporaryDirectory() as workspace:
        dir_name = _project_dir_name(project_name, project_id)

        # Register a mock sub-agent adapter to simulate an active sub-agent
        mock_adapter = MagicMock()
        mock_adapter.is_alive.return_value = True
        mock_adapter.is_idle.return_value = False
        sub_agent_mgr._adapters[project_id] = {"helper-1": mock_adapter}

        # Build a session handle (task=None = idle agent)
        session = Session.new("old_subagent_session", workspace, project_dir_name=dir_name)
        handle = ProjectHandle(
            session=session,
            loop=MagicMock(),
            provider=MagicMock(),
            registry=MagicMock(),
            context_manager=MagicMock(),
            interceptor=MagicMock(),
            task=None,
            config_snapshot={"workspace": workspace},
            project_dir_name=dir_name,
        )
        mgr._handles[project_id] = handle

        project_store.get_project.return_value = {
            "project_id": project_id,
            "name": project_name,
            "workspace": workspace,
        }

        client, saved = _build_test_client(mgr, project_store)
        try:
            import time

            with patch(
                "agent_os.daemon_v2.agent_manager.run_session_end_routine",
                new=AsyncMock(),
            ):
                start = time.monotonic()
                response = client.post(f"/api/v2/agents/{project_id}/new-session")
                elapsed = time.monotonic() - start

            # ---- HTTP-level assertions ----
            assert response.status_code == 200, (
                f"Expected HTTP 200, got {response.status_code}: {response.text}"
            )
            assert elapsed < 30.0, (
                f"new-session must return within 30s, took {elapsed:.2f}s"
            )

            body = response.json()
            assert body.get("status") == "ok", f"Expected status 'ok': {body}"

            # ---- stop_all was called EXACTLY ONCE (the T06 post-session-end call) ----
            assert len(stop_all_calls) == 1, (
                f"Expected stop_all called exactly once (T06); got {stop_all_calls}"
            )
            assert stop_all_calls[0] == project_id, (
                f"stop_all must be called with project_id={project_id}: {stop_all_calls}"
            )

            # ---- New session is distinct from old ----
            new_handle = mgr._handles[project_id]
            assert new_handle.session.session_id != "old_subagent_session", (
                "New session must be distinct from old session"
            )
            new_session_id = new_handle.session.session_id
            assert new_session_id, "New session must have a session_id"

        finally:
            client.close()
            _restore_agents_v2(saved)


def test_new_session_subagent_stop_all_timeout_does_not_block():
    """POST /api/v2/agents/{pid}/new-session completes within 30s even when
    the post-session-end stop_all hangs past its 10s budget.

    Verifies the wait_for timeout guard is effective at the HTTP level.
    """
    from agent_os.daemon_v2.agent_manager import ProjectHandle
    from agent_os.agent.session import Session

    project_id = "proj_subagent_timeout_test"
    project_name = "SubAgent Timeout Test"

    # T06: stop_all is called exactly once. Make it hang indefinitely.
    async def _hanging_stop_all(pid: str) -> None:
        # Simulate indefinite hang — the wait_for budget will cancel us
        await asyncio.sleep(9999)

    stop_all_mock = AsyncMock(side_effect=_hanging_stop_all)

    mgr, _ws, project_store, sub_agent_mgr = _make_agent_manager(stop_all_mock=stop_all_mock)

    with tempfile.TemporaryDirectory() as workspace:
        dir_name = _project_dir_name(project_name, project_id)
        session = Session.new("old_timeout_session", workspace, project_dir_name=dir_name)
        handle = ProjectHandle(
            session=session,
            loop=MagicMock(),
            provider=MagicMock(),
            registry=MagicMock(),
            context_manager=MagicMock(),
            interceptor=MagicMock(),
            task=None,
            config_snapshot={"workspace": workspace},
            project_dir_name=dir_name,
        )
        mgr._handles[project_id] = handle

        project_store.get_project.return_value = {
            "project_id": project_id,
            "name": project_name,
            "workspace": workspace,
        }

        client, saved = _build_test_client(mgr, project_store)
        try:
            import time

            with patch(
                "agent_os.daemon_v2.agent_manager.run_session_end_routine",
                new=AsyncMock(),
            ):
                start = time.monotonic()
                response = client.post(f"/api/v2/agents/{project_id}/new-session")
                elapsed = time.monotonic() - start

            # Must complete within 30s (10s wait_for + overhead)
            assert elapsed < 30.0, (
                f"new-session must complete within 30s even on stop_all timeout; "
                f"took {elapsed:.2f}s"
            )
            assert response.status_code == 200, (
                f"Expected HTTP 200, got {response.status_code}: {response.text}"
            )

            body = response.json()
            assert body.get("status") == "ok", f"Expected status 'ok': {body}"

            # New session must still be built despite the timeout
            new_handle = mgr._handles[project_id]
            assert new_handle.session.session_id != "old_timeout_session", (
                "New session must be built even when stop_all times out"
            )

        finally:
            client.close()
            _restore_agents_v2(saved)
