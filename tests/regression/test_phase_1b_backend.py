# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: Phase 1B backend extensions for multi-session UI.

Tests the five backend additions introduced in feat/v1-ui-phase-1b:

1. ``GET /projects/{pid}/sessions`` includes ``last_activity_at``.
2. ``GET /agents/{pid}/chat?session_id=X`` filters to a single session
   (unfiltered path unchanged).
3. ``GET /agents/{pid}/run-status`` includes ``current_holder_session_id``.
4. ``GET /blocked`` returns aggregate blocked-session count and list.
5. WS ``blocked-count-changed`` fires on enter/leave of ``pending_approval``.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager():
    ws = MagicMock()
    ws.broadcast = MagicMock()
    ws.broadcast_global = MagicMock()
    project_store = MagicMock()
    sub_agent_mgr = MagicMock()
    sub_agent_mgr.list_active = MagicMock(return_value=[])
    sub_agent_mgr.stop_all = AsyncMock()
    activity_translator = MagicMock()
    process_manager = MagicMock()
    mgr = AgentManager(
        project_store=project_store,
        ws_manager=ws,
        sub_agent_manager=sub_agent_mgr,
        activity_translator=activity_translator,
        process_manager=process_manager,
    )
    return mgr, ws


def _make_idle_handle(mgr, project_id, session_id="default", *, messages=None):
    """Plant an idle handle. ``messages`` populates session._messages."""
    session = MagicMock()
    session.is_stopped = MagicMock(return_value=False)
    session._paused_for_approval = False
    session.session_id = session_id              # F1 (user-facing)
    session.session_uuid = f"{session_id}-f2-stem"  # F2 (JSONL stem)
    session._messages = messages or []

    async def _done_immediately():
        return None

    task = asyncio.get_event_loop().create_task(_done_immediately())

    handle = ProjectHandle(
        session=session,
        loop=MagicMock(),
        provider=MagicMock(),
        registry=MagicMock(),
        context_manager=MagicMock(),
        interceptor=MagicMock(),
        task=task,
    )
    mgr._handles[(project_id, session_id)] = handle
    return handle, task


def _make_paused_handle(mgr, project_id, session_id="default"):
    """Plant a handle in pending_approval state."""
    session = MagicMock()
    session.is_stopped = MagicMock(return_value=False)
    session._paused_for_approval = True
    session.session_id = session_id              # F1 (user-facing)
    session.session_uuid = f"{session_id}-f2-stem"  # F2 (JSONL stem)
    session._messages = []
    session.pop_queued_messages = MagicMock(return_value=[])
    session.pop_deferred_messages = MagicMock(return_value=[])
    session.has_result_for = MagicMock(return_value=False)
    session.append_tool_result = MagicMock()
    session.append = MagicMock()
    session.resume = MagicMock()

    interceptor = MagicMock()
    interceptor._pending_approvals = {
        "tc_1": {
            "tool_name": "write_file",
            "tool_args": {"path": "x.txt"},
            "what": "write_file(x.txt)",
            "tool_call_id": "tc_1",
            "recent_activity": [],
        }
    }
    interceptor.get_pending = MagicMock(
        side_effect=lambda tc_id: interceptor._pending_approvals.get(tc_id)
    )
    interceptor.remove_pending = MagicMock(
        side_effect=lambda tc_id: interceptor._pending_approvals.pop(tc_id, None)
    )

    task_done = MagicMock()
    task_done.done.return_value = True
    task_done.exception.return_value = None

    handle = ProjectHandle(
        session=session,
        loop=MagicMock(),
        provider=MagicMock(),
        registry=MagicMock(),
        context_manager=MagicMock(),
        interceptor=interceptor,
        task=task_done,
    )
    mgr._handles[(project_id, session_id)] = handle
    return handle


# ===========================================================================
# 1. last_activity_at in list_sessions
# ===========================================================================

class TestLastActivityAt:
    """list_sessions() must include last_activity_at on every entry."""

    @pytest.mark.asyncio
    async def test_no_messages_returns_null(self):
        mgr, _ = _make_manager()
        _make_idle_handle(mgr, "proj_laa_null", messages=[])
        sessions = mgr.list_sessions("proj_laa_null")
        assert len(sessions) == 1
        entry = sessions[0]
        assert "last_activity_at" in entry, "last_activity_at must be present"
        assert entry["last_activity_at"] is None

    @pytest.mark.asyncio
    async def test_with_messages_returns_last_timestamp(self):
        msgs = [
            {"role": "user", "content": "hi", "timestamp": "2026-05-01T00:00:00+00:00"},
            {"role": "assistant", "content": "hello", "timestamp": "2026-05-01T00:01:00+00:00"},
        ]
        mgr, _ = _make_manager()
        _make_idle_handle(mgr, "proj_laa_ts", messages=msgs)
        sessions = mgr.list_sessions("proj_laa_ts")
        assert sessions[0]["last_activity_at"] == "2026-05-01T00:01:00+00:00"

    @pytest.mark.asyncio
    async def test_field_updates_when_messages_appended(self):
        """Adding a new message to the in-memory list is reflected immediately."""
        msgs = [{"role": "user", "content": "start", "timestamp": "2026-05-01T10:00:00+00:00"}]
        mgr, _ = _make_manager()
        handle, _ = _make_idle_handle(mgr, "proj_laa_update", messages=msgs)

        # Simulate appending a new message (in real code session.append() does this)
        handle.session._messages.append(
            {"role": "assistant", "content": "response", "timestamp": "2026-05-02T00:00:00+00:00"}
        )
        sessions = mgr.list_sessions("proj_laa_update")
        assert sessions[0]["last_activity_at"] == "2026-05-02T00:00:00+00:00"


# ===========================================================================
# 2. /chat?session_id filter
# ===========================================================================

class TestChatSessionIdFilter:
    """GET /agents/{pid}/chat?session_id=X returns only that session's messages."""

    def _make_api_app(self, tmp_path):
        """Build a minimal FastAPI app wired to a real AgentManager + project."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from agent_os.api.ws import WebSocketManager
        from agent_os.api.routes import agents_v2
        from agent_os.daemon_v2.project_store import ProjectStore
        from agent_os.daemon_v2.activity_translator import ActivityTranslator
        from agent_os.daemon_v2.process_manager import ProcessManager

        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir)
        workspace = str(tmp_path / "workspace")
        os.makedirs(workspace)

        project_store = ProjectStore(data_dir=data_dir)
        ws_manager = WebSocketManager()
        activity_translator = ActivityTranslator(ws_manager)
        process_manager = ProcessManager(ws_manager, activity_translator)

        mock_settings = MagicMock()
        mock_settings.get.return_value = MagicMock(
            llm=MagicMock(provider="anthropic", model="claude-sonnet-4-20250514", api_key="", base_url="")
        )
        mock_credential = MagicMock()
        mock_credential.get_api_key.return_value = "sk-test"

        mock_platform = MagicMock()
        mock_platform.get_capabilities.return_value = MagicMock(
            platform="windows", setup_complete=True
        )

        from agent_os.daemon_v2.agent_manager import AgentManager
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.daemon_v2.browser_manager import BrowserManager
        from agent_os.daemon_v2.process_manager import ProcessManager as PM

        mock_registry = MagicMock()
        mock_registry.list_manifests.return_value = []
        mock_setup = MagicMock()
        mock_provider_reg = MagicMock()
        mock_trigger = MagicMock()

        browser_manager = BrowserManager(
            profile_dir=os.path.join(data_dir, "browser"),
            headless=True,
        )

        sub_agent_manager = SubAgentManager(
            process_manager=process_manager,
            registry=mock_registry,
            setup_engine=mock_setup,
            platform_provider=mock_platform,
            project_store=project_store,
        )

        agent_manager = AgentManager(
            project_store=project_store,
            settings_store=mock_settings,
            credential_store=mock_credential,
            ws_manager=ws_manager,
            activity_translator=activity_translator,
            process_manager=process_manager,
            platform_provider=mock_platform,
            sub_agent_manager=sub_agent_manager,
            browser_manager=browser_manager,
            provider_registry=mock_provider_reg,
        )

        agents_v2.configure(
            project_store=project_store,
            agent_manager=agent_manager,
            ws_manager=ws_manager,
            sub_agent_manager=sub_agent_manager,
            setup_engine=mock_setup,
            settings_store=mock_settings,
            credential_store=mock_credential,
            trigger_manager=mock_trigger,
            provider_registry=mock_provider_reg,
        )

        app = FastAPI()
        app.include_router(agents_v2.router)

        return TestClient(app), project_store, agent_manager, workspace

    def test_session_filter_returns_only_that_session(self, tmp_path):
        """Fast path: a live handle resolves F1 session_id → F2 session_uuid,
        and the filter returns only that session's messages.

        Crucially, F1 != F2 here (real Sessions distinguish the two), and the
        JSONL files are named by F2 (the stem). The records intentionally do
        NOT carry a ``session_id`` field, so the disk-scan fallback (which
        matches on record ``session_id``) CANNOT resolve them — only the
        handle's F1→F2 mapping can. This proves the fast path works and is not
        masked by the fallback.
        """
        client, project_store, agent_manager, workspace = self._make_api_app(tmp_path)

        # Create a project directly (project_store.create_project takes a config dict
        # and auto-generates the project_id; we inject it manually so we control the id)
        project_store._projects["proj_filter"] = {
            "project_id": "proj_filter",
            "name": "Filter Test",
            "workspace": workspace,
            "model": "claude-sonnet-4-20250514",
            "api_key": "sk-test",
            "provider": "anthropic",
        }

        from agent_os.agent.project_paths import ProjectPaths
        sessions_dir = ProjectPaths(workspace).sessions_dir
        os.makedirs(sessions_dir, exist_ok=True)

        # F1 (user-facing) and F2 (JSONL stem) are DISTINCT, mirroring a real
        # Session. JSONL files are named by F2.
        sess_a_f1 = "session_a"
        sess_a_f2 = "proj_session_a_abc123"
        sess_b_f1 = "session_b"
        sess_b_f2 = "proj_session_b_def456"

        # Records DO NOT include a session_id field — only the handle mapping
        # can resolve F1 → F2, so the fallback (content scan on session_id)
        # would find nothing if the fast path were broken.
        with open(os.path.join(sessions_dir, f"{sess_a_f2}.jsonl"), "w") as f:
            f.write(json.dumps({"role": "user", "content": "hello from A",
                                "timestamp": "2026-01-01T00:00:00+00:00"}) + "\n")
            f.write(json.dumps({"role": "assistant", "content": "reply from A",
                                "timestamp": "2026-01-01T00:01:00+00:00"}) + "\n")

        with open(os.path.join(sessions_dir, f"{sess_b_f2}.jsonl"), "w") as f:
            f.write(json.dumps({"role": "user", "content": "hello from B",
                                "timestamp": "2026-01-02T00:00:00+00:00"}) + "\n")

        # Plant live handles so list_sessions can resolve F1 → F2 for both.
        from agent_os.daemon_v2.agent_manager import ProjectHandle

        def _plant(f1, f2):
            sess = MagicMock()
            sess.is_stopped.return_value = False
            sess._paused_for_approval = False
            sess.session_id = f1     # F1 (user-facing)
            sess.session_uuid = f2   # F2 (JSONL stem)
            sess._messages = []
            agent_manager._handles[("proj_filter", f1)] = ProjectHandle(
                session=sess,
                loop=MagicMock(), provider=MagicMock(), registry=MagicMock(),
                context_manager=MagicMock(), interceptor=MagicMock(), task=None,
            )

        _plant(sess_a_f1, sess_a_f2)
        _plant(sess_b_f1, sess_b_f2)

        # Guard: ensure the disk-scan fallback is NOT used. If the fast path
        # F1→F2 mapping fails to resolve, the route would fall through to
        # _find_session_uuid_on_disk; patch THAT to raise so any such
        # fall-through fails loudly. (We target the fallback function itself
        # rather than os.listdir, because list_sessions is now disk-backed and
        # legitimately calls os.listdir on the fast path.)
        import agent_os.api.routes.agents_v2 as av2

        def _no_fallback(sessions_dir, session_id):
            raise AssertionError(
                "disk-scan fallback was reached; fast path F1→F2 mapping failed"
            )

        # Filtered: session_a only — must resolve via the live handle (fast path).
        orig_fallback = av2._find_session_uuid_on_disk
        av2._find_session_uuid_on_disk = _no_fallback
        try:
            resp = client.get("/api/v2/agents/proj_filter/chat?session_id=session_a")
        finally:
            av2._find_session_uuid_on_disk = orig_fallback
        assert resp.status_code == 200, resp.text
        msgs = resp.json()
        assert len(msgs) == 2, f"Expected 2 messages from session_a, got {len(msgs)}"
        assert [m["content"] for m in msgs] == ["hello from A", "reply from A"]

        # Unfiltered: all messages returned (3 total across both sessions).
        resp_all = client.get("/api/v2/agents/proj_filter/chat")
        assert resp_all.status_code == 200
        all_msgs = resp_all.json()
        assert len(all_msgs) == 3, f"Expected 3 total messages, got {len(all_msgs)}"

    def test_session_filter_disk_fallback(self, tmp_path):
        """When the handle is not live, the disk-scan fallback finds the JSONL."""
        client, project_store, agent_manager, workspace = self._make_api_app(tmp_path)

        project_store._projects["proj_fallback"] = {
            "project_id": "proj_fallback",
            "name": "Fallback Test",
            "workspace": workspace,
            "model": "claude-sonnet-4-20250514",
            "api_key": "sk-test",
            "provider": "anthropic",
        }

        from agent_os.agent.project_paths import ProjectPaths
        sessions_dir = ProjectPaths(workspace).sessions_dir
        os.makedirs(sessions_dir, exist_ok=True)

        sess_uuid = "old_sess_uuid"
        with open(os.path.join(sessions_dir, f"{sess_uuid}.jsonl"), "w") as f:
            f.write(json.dumps({"role": "user", "content": "old message",
                                "session_id": "old_session", "session_uuid": sess_uuid,
                                "timestamp": "2026-01-01T00:00:00+00:00"}) + "\n")

        # No live handle for old_session — fallback must scan disk
        resp = client.get("/api/v2/agents/proj_fallback/chat?session_id=old_session")
        assert resp.status_code == 200
        msgs = resp.json()
        assert len(msgs) == 1
        assert msgs[0]["content"] == "old message"

    def test_unknown_session_id_returns_empty(self, tmp_path):
        """An unknown session_id returns empty list, not 404."""
        client, project_store, _am, workspace = self._make_api_app(tmp_path)

        project_store._projects["proj_unknown_sess"] = {
            "project_id": "proj_unknown_sess",
            "name": "Unknown Sess",
            "workspace": workspace,
            "model": "claude-sonnet-4-20250514",
            "api_key": "sk-test",
            "provider": "anthropic",
        }

        resp = client.get("/api/v2/agents/proj_unknown_sess/chat?session_id=does_not_exist")
        assert resp.status_code == 200
        assert resp.json() == []
        assert resp.headers["x-total-count"] == "0"


# ===========================================================================
# 3. run-status includes current_holder_session_id
# ===========================================================================

class TestRunStatusHolder:
    """GET /agents/{pid}/run-status must include current_holder_session_id."""

    def _make_app_and_mgr(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from agent_os.api.ws import WebSocketManager
        from agent_os.api.routes import agents_v2
        from agent_os.daemon_v2.project_store import ProjectStore
        from agent_os.daemon_v2.activity_translator import ActivityTranslator
        from agent_os.daemon_v2.process_manager import ProcessManager
        from agent_os.daemon_v2.agent_manager import AgentManager
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.daemon_v2.browser_manager import BrowserManager

        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir)
        workspace = str(tmp_path / "ws")
        os.makedirs(workspace)

        project_store = ProjectStore(data_dir=data_dir)
        ws_manager = WebSocketManager()
        activity_translator = ActivityTranslator(ws_manager)
        process_manager = ProcessManager(ws_manager, activity_translator)

        mock_settings = MagicMock()
        mock_settings.get.return_value = MagicMock(
            llm=MagicMock(provider="anthropic", model="claude-sonnet-4-20250514", api_key="", base_url="")
        )
        mock_cred = MagicMock()
        mock_cred.get_api_key.return_value = "sk-test"
        mock_platform = MagicMock()
        mock_platform.get_capabilities.return_value = MagicMock(platform="windows", setup_complete=True)
        mock_registry = MagicMock()
        mock_registry.list_manifests.return_value = []
        mock_setup = MagicMock()
        mock_pr = MagicMock()
        mock_trigger = MagicMock()

        browser_manager = BrowserManager(
            profile_dir=os.path.join(data_dir, "browser"), headless=True
        )

        sub_agent_manager = SubAgentManager(
            process_manager=process_manager,
            registry=mock_registry,
            setup_engine=mock_setup,
            platform_provider=mock_platform,
            project_store=project_store,
        )

        agent_manager = AgentManager(
            project_store=project_store,
            settings_store=mock_settings,
            credential_store=mock_cred,
            ws_manager=ws_manager,
            activity_translator=activity_translator,
            process_manager=process_manager,
            platform_provider=mock_platform,
            sub_agent_manager=sub_agent_manager,
            browser_manager=browser_manager,
            provider_registry=mock_pr,
        )

        agents_v2.configure(
            project_store=project_store,
            agent_manager=agent_manager,
            ws_manager=ws_manager,
            sub_agent_manager=sub_agent_manager,
            setup_engine=mock_setup,
            settings_store=mock_settings,
            credential_store=mock_cred,
            trigger_manager=mock_trigger,
            provider_registry=mock_pr,
        )

        app = FastAPI()
        app.include_router(agents_v2.router)
        return TestClient(app), project_store, agent_manager, workspace

    def test_no_active_session_holder_is_null(self, tmp_path):
        client, project_store, _am, workspace = self._make_app_and_mgr(tmp_path)
        project_store._projects["proj_holder_null"] = {
            "project_id": "proj_holder_null",
            "name": "Holder Null",
            "workspace": workspace,
            "model": "claude-sonnet-4-20250514",
            "api_key": "sk-test",
            "provider": "anthropic",
        }
        resp = client.get("/api/v2/agents/proj_holder_null/run-status")
        assert resp.status_code == 200
        body = resp.json()
        assert "current_holder_session_id" in body
        assert body["current_holder_session_id"] is None

    def test_paused_for_approval_holder_is_session_id(self, tmp_path):
        client, project_store, agent_manager, workspace = self._make_app_and_mgr(tmp_path)
        project_store._projects["proj_holder_paused"] = {
            "project_id": "proj_holder_paused",
            "name": "Holder Paused",
            "workspace": workspace,
            "model": "claude-sonnet-4-20250514",
            "api_key": "sk-test",
            "provider": "anthropic",
        }
        # Plant a paused handle — it holds the slot
        session = MagicMock()
        session.is_stopped.return_value = False
        session._paused_for_approval = True
        session._messages = []

        task_done = MagicMock()
        task_done.done.return_value = True
        task_done.exception.return_value = None

        from agent_os.daemon_v2.agent_manager import ProjectHandle
        agent_manager._handles[("proj_holder_paused", "sess_x")] = ProjectHandle(
            session=session, loop=MagicMock(), provider=MagicMock(),
            registry=MagicMock(), context_manager=MagicMock(),
            interceptor=MagicMock(), task=task_done,
        )
        resp = client.get("/api/v2/agents/proj_holder_paused/run-status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["current_holder_session_id"] == "sess_x"


# ===========================================================================
# 4. GET /blocked — global blocked summary
# ===========================================================================

class TestBlockedEndpoint:
    """GET /api/v2/blocked returns aggregate pending_approval count."""

    def test_list_blocked_sessions_empty(self):
        mgr, _ = _make_manager()
        result = mgr.list_blocked_sessions()
        assert result == []

    def test_list_blocked_sessions_one_paused(self):
        mgr, _ = _make_manager()
        _make_paused_handle(mgr, "proj_blocked", "sess_blocked")
        result = mgr.list_blocked_sessions()
        assert len(result) == 1
        assert result[0]["project_id"] == "proj_blocked"
        assert result[0]["session_id"] == "sess_blocked"

    def test_list_blocked_sessions_multiple_projects(self):
        mgr, _ = _make_manager()
        _make_paused_handle(mgr, "proj_a", "sess_1")
        _make_paused_handle(mgr, "proj_b", "sess_2")
        # Not-blocked: a handle without _paused_for_approval
        not_paused = MagicMock()
        not_paused.is_stopped.return_value = False
        not_paused._paused_for_approval = False
        not_paused.session_id = "sess_c"            # F1
        not_paused.session_uuid = "sess_c-f2-stem"  # F2
        not_paused._messages = []
        handle_c = ProjectHandle(
            session=not_paused,
            loop=MagicMock(), provider=MagicMock(), registry=MagicMock(),
            context_manager=MagicMock(), interceptor=MagicMock(), task=None,
        )
        mgr._handles[("proj_c", "default")] = handle_c
        result = mgr.list_blocked_sessions()
        assert len(result) == 2
        pids = {r["project_id"] for r in result}
        assert pids == {"proj_a", "proj_b"}

    def test_blocked_endpoint_returns_correct_shape(self, tmp_path):
        """HTTP test: GET /api/v2/blocked returns expected shape."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from agent_os.api.ws import WebSocketManager
        from agent_os.api.routes import agents_v2
        from agent_os.daemon_v2.project_store import ProjectStore
        from agent_os.daemon_v2.activity_translator import ActivityTranslator
        from agent_os.daemon_v2.process_manager import ProcessManager
        from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.daemon_v2.browser_manager import BrowserManager

        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir)
        workspace = str(tmp_path / "ws")
        os.makedirs(workspace)

        project_store = ProjectStore(data_dir=data_dir)
        ws_manager = WebSocketManager()
        activity_translator = ActivityTranslator(ws_manager)
        process_manager = ProcessManager(ws_manager, activity_translator)

        mock_settings = MagicMock()
        mock_settings.get.return_value = MagicMock(
            llm=MagicMock(provider="anthropic", model="claude-sonnet-4-20250514", api_key="", base_url="")
        )
        mock_cred = MagicMock()
        mock_cred.get_api_key.return_value = "sk-test"
        mock_platform = MagicMock()
        mock_platform.get_capabilities.return_value = MagicMock(platform="windows", setup_complete=True)
        mock_reg = MagicMock()
        mock_reg.list_manifests.return_value = []
        mock_setup = MagicMock()
        mock_pr = MagicMock()
        mock_trigger = MagicMock()

        browser_manager = BrowserManager(
            profile_dir=os.path.join(data_dir, "browser"), headless=True
        )
        sub_agent_manager = SubAgentManager(
            process_manager=process_manager, registry=mock_reg,
            setup_engine=mock_setup, platform_provider=mock_platform,
            project_store=project_store,
        )
        agent_manager = AgentManager(
            project_store=project_store, settings_store=mock_settings,
            credential_store=mock_cred, ws_manager=ws_manager,
            activity_translator=activity_translator, process_manager=process_manager,
            platform_provider=mock_platform, sub_agent_manager=sub_agent_manager,
            browser_manager=browser_manager, provider_registry=mock_pr,
        )
        agents_v2.configure(
            project_store=project_store, agent_manager=agent_manager,
            ws_manager=ws_manager, sub_agent_manager=sub_agent_manager,
            setup_engine=mock_setup, settings_store=mock_settings,
            credential_store=mock_cred, trigger_manager=mock_trigger,
            provider_registry=mock_pr,
        )
        app = FastAPI()
        app.include_router(agents_v2.router)
        client = TestClient(app)

        # No blocked sessions initially
        resp = client.get("/api/v2/blocked")
        assert resp.status_code == 200
        body = resp.json()
        assert body["blocked_count"] == 0
        assert body["blocked_sessions"] == []

        # Plant a paused handle
        sess = MagicMock()
        sess.is_stopped.return_value = False
        sess._paused_for_approval = True
        sess._messages = []
        task_done = MagicMock()
        task_done.done.return_value = True
        task_done.exception.return_value = None
        agent_manager._handles[("proj_bp", "sess_bp")] = ProjectHandle(
            session=sess, loop=MagicMock(), provider=MagicMock(),
            registry=MagicMock(), context_manager=MagicMock(),
            interceptor=MagicMock(), task=task_done,
        )

        resp2 = client.get("/api/v2/blocked")
        assert resp2.status_code == 200
        body2 = resp2.json()
        assert body2["blocked_count"] == 1
        assert body2["blocked_sessions"][0]["project_id"] == "proj_bp"
        assert body2["blocked_sessions"][0]["session_id"] == "sess_bp"


# ===========================================================================
# 4b. Budget-paused projects in the Blocked surface (P3-G)
# ===========================================================================

class TestBlockedBudgetProjects:
    """list_blocked_budget_projects() + GET /api/v2/blocked budget_paused_projects.

    Codes-only addition to the existing cross-project blocked endpoint: lists
    projects whose persisted queue state is PAUSED with pause_reason="budget".
    Reads the persisted queue.json so it works without a live dispatcher.
    """

    @staticmethod
    def _store_with_projects(tmp_path):
        from agent_os.daemon_v2.project_store import ProjectStore
        from agent_os.queue.store import QueueStore
        from agent_os.queue.models import QueueRunState
        from agent_os.agent.project_paths import ProjectPaths

        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir, exist_ok=True)
        ps = ProjectStore(data_dir=data_dir)

        def make(name, run_state, reason=None):
            ws = str(tmp_path / name)
            os.makedirs(ws, exist_ok=True)
            pid = ps.create_project({"name": name, "workspace": ws})
            store = QueueStore(ProjectPaths(ws).queue_file)
            store.set_queue_state(run_state, reason=reason)
            return pid

        return ps, make

    def _manager_with_store(self, project_store):
        ws = MagicMock()
        ws.broadcast = MagicMock()
        ws.broadcast_global = MagicMock()
        sub_agent_mgr = MagicMock()
        sub_agent_mgr.list_active = MagicMock(return_value=[])
        sub_agent_mgr.stop_all = AsyncMock()
        return AgentManager(
            project_store=project_store,
            ws_manager=ws,
            sub_agent_manager=sub_agent_mgr,
            activity_translator=MagicMock(),
            process_manager=MagicMock(),
        ), ws

    def test_empty_when_no_budget_pauses(self, tmp_path):
        from agent_os.queue.models import QueueRunState
        ps, make = self._store_with_projects(tmp_path)
        make("running_proj", QueueRunState.RUNNING)
        make("user_paused", QueueRunState.PAUSED, reason="user")
        mgr, _ = self._manager_with_store(ps)
        assert mgr.list_blocked_budget_projects() == []

    def test_lists_only_budget_paused_projects(self, tmp_path):
        from agent_os.queue.models import QueueRunState
        ps, make = self._store_with_projects(tmp_path)
        pid_budget = make("budget_paused", QueueRunState.PAUSED, reason="budget")
        make("user_paused", QueueRunState.PAUSED, reason="user")
        make("running_proj", QueueRunState.RUNNING)
        mgr, _ = self._manager_with_store(ps)
        result = mgr.list_blocked_budget_projects()
        assert [r["project_id"] for r in result] == [pid_budget]

    def test_multiple_budget_paused(self, tmp_path):
        from agent_os.queue.models import QueueRunState
        ps, make = self._store_with_projects(tmp_path)
        a = make("budget_a", QueueRunState.PAUSED, reason="budget")
        b = make("budget_b", QueueRunState.PAUSED, reason="budget")
        mgr, _ = self._manager_with_store(ps)
        pids = {r["project_id"] for r in mgr.list_blocked_budget_projects()}
        assert pids == {a, b}

    def test_endpoint_includes_budget_paused_projects(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from agent_os.api.routes import agents_v2
        from agent_os.queue.models import QueueRunState

        ps, make = self._store_with_projects(tmp_path)
        pid_budget = make("budget_paused", QueueRunState.PAUSED, reason="budget")
        make("running_proj", QueueRunState.RUNNING)
        mgr, ws = self._manager_with_store(ps)

        agents_v2.configure(
            project_store=ps, agent_manager=mgr, ws_manager=ws,
            sub_agent_manager=MagicMock(), setup_engine=MagicMock(),
            settings_store=MagicMock(), credential_store=MagicMock(),
            trigger_manager=MagicMock(), provider_registry=MagicMock(),
        )
        app = FastAPI()
        app.include_router(agents_v2.router)
        client = TestClient(app)

        resp = client.get("/api/v2/blocked")
        assert resp.status_code == 200
        body = resp.json()
        # Approval count is untouched (no pending_approval handles here).
        assert body["blocked_count"] == 0
        assert body["blocked_sessions"] == []
        # Budget-paused project surfaces under the new codes-only field.
        assert body["budget_paused_projects"] == [{"project_id": pid_budget}]


# ===========================================================================
# 5. WS blocked-count-changed event
# ===========================================================================

class TestBlockedCountWsEvent:
    """WS broadcast_global fires blocked-count-changed on enter/leave."""

    def test_blocked_count_broadcast_on_enter(self):
        """_on_loop_done fires blocked-count-changed when session enters pending_approval."""
        mgr, ws = _make_manager()

        # Build a handle whose task exits normally with _paused_for_approval = True
        session = MagicMock()
        session.is_stopped.return_value = False
        session._paused_for_approval = True
        session.pop_deferred_messages.return_value = []
        session.pop_queued_messages.return_value = []
        session._messages = []

        task_mock = MagicMock()
        task_mock.exception.return_value = None
        task_mock.done.return_value = True

        from agent_os.daemon_v2.agent_manager import ProjectHandle
        mgr._handles[("proj_enter", "sess_e")] = ProjectHandle(
            session=session, loop=MagicMock(), provider=MagicMock(),
            registry=MagicMock(), context_manager=MagicMock(),
            interceptor=MagicMock(), task=task_mock,
        )

        cb = mgr._on_loop_done("proj_enter", session_id="sess_e")
        cb(task_mock)

        # broadcast_global must have been called with blocked-count-changed
        ws.broadcast_global.assert_called()
        global_calls = [
            call for call in ws.broadcast_global.call_args_list
            if len(call.args) >= 1
            and isinstance(call.args[0], dict)
            and call.args[0].get("type") == "blocked-count-changed"
        ]
        assert len(global_calls) >= 1, (
            f"Expected blocked-count-changed broadcast on enter; "
            f"got calls: {ws.broadcast_global.call_args_list}"
        )
        payload = global_calls[0].args[0]
        assert payload["blocked_count"] >= 1
        assert any(
            s["project_id"] == "proj_enter" and s["session_id"] == "sess_e"
            for s in payload["blocked_sessions"]
        )
        # P3-G: the broadcast must also carry the budget-paused project list so
        # the sidebar Blocked surface stays live without a separate fetch. The
        # mocked project_store yields no projects here, so the field is present
        # and empty — pinning the payload contract, not the contents.
        assert payload["budget_paused_projects"] == []

    def test_no_blocked_count_broadcast_on_normal_exit(self):
        """_on_loop_done for a normal (non-approval) idle exit must NOT emit
        blocked-count-changed. Locks the guard so the event only fires on a
        real pending_approval transition, not every loop completion."""
        mgr, ws = _make_manager()

        # Handle whose task exits cleanly, NOT paused for approval, with no
        # queued messages and no busy sub-agents → plain idle exit.
        session = MagicMock()
        session.is_stopped.return_value = False
        session._paused_for_approval = False
        session.pop_deferred_messages.return_value = []
        session.pop_queued_messages.return_value = []
        session.session_id = "sess_n"
        session.session_uuid = "sess_n-f2-stem"
        session._messages = []

        task_mock = MagicMock()
        task_mock.exception.return_value = None
        task_mock.done.return_value = True

        from agent_os.daemon_v2.agent_manager import ProjectHandle
        mgr._handles[("proj_normal", "sess_n")] = ProjectHandle(
            session=session, loop=MagicMock(), provider=MagicMock(),
            registry=MagicMock(), context_manager=MagicMock(),
            interceptor=MagicMock(), task=task_mock,
        )

        cb = mgr._on_loop_done("proj_normal", session_id="sess_n")
        cb(task_mock)

        # No blocked-count-changed event on a normal exit.
        blocked_calls = [
            call for call in ws.broadcast_global.call_args_list
            if len(call.args) >= 1
            and isinstance(call.args[0], dict)
            and call.args[0].get("type") == "blocked-count-changed"
        ]
        assert blocked_calls == [], (
            f"Normal idle exit must NOT emit blocked-count-changed; "
            f"got: {blocked_calls}"
        )

    @pytest.mark.asyncio
    async def test_blocked_count_broadcast_on_cancel(self):
        """cancel_message fires blocked-count-changed when session leaves pending_approval."""
        mgr, ws = _make_manager()
        mgr._record_approval_decision = MagicMock()

        handle = _make_paused_handle(mgr, "proj_cancel_bc", "sess_cancel")

        result = await mgr.cancel_message("proj_cancel_bc", session_id="sess_cancel")
        assert result == {"status": "cancelled"}

        # After cancel, _paused_for_approval must be False
        assert handle.session._paused_for_approval is False

        # broadcast_global must have fired with blocked-count-changed (count=0)
        ws.broadcast_global.assert_called()
        global_calls = [
            call for call in ws.broadcast_global.call_args_list
            if len(call.args) >= 1
            and isinstance(call.args[0], dict)
            and call.args[0].get("type") == "blocked-count-changed"
        ]
        assert len(global_calls) >= 1, (
            f"Expected blocked-count-changed on cancel-leave; "
            f"got: {ws.broadcast_global.call_args_list}"
        )
        payload = global_calls[-1].args[0]
        assert payload["blocked_count"] == 0

    @pytest.mark.asyncio
    async def test_blocked_count_broadcast_on_deny(self):
        """deny() fires blocked-count-changed when session leaves pending_approval."""
        mgr, ws = _make_manager()
        mgr._record_approval_decision = MagicMock()
        mgr._start_loop = AsyncMock()

        handle = _make_paused_handle(mgr, "proj_deny_bc", "sess_deny")

        await mgr.deny("proj_deny_bc", "tc_1", "not allowed", session_id="sess_deny")

        ws.broadcast_global.assert_called()
        global_calls = [
            call for call in ws.broadcast_global.call_args_list
            if len(call.args) >= 1
            and isinstance(call.args[0], dict)
            and call.args[0].get("type") == "blocked-count-changed"
        ]
        assert len(global_calls) >= 1, (
            f"Expected blocked-count-changed on deny-leave; "
            f"got: {ws.broadcast_global.call_args_list}"
        )
        payload = global_calls[-1].args[0]
        assert payload["blocked_count"] == 0

    @pytest.mark.asyncio
    async def test_blocked_count_broadcast_on_approve(self):
        """approve() fires blocked-count-changed when session leaves pending_approval."""
        mgr, ws = _make_manager()
        mgr._record_approval_decision = MagicMock()
        mgr._start_loop = AsyncMock()

        handle = _make_paused_handle(mgr, "proj_approve_bc", "sess_approve")
        # Make registry.execute work for approve path
        handle.registry.is_async.return_value = False
        handle.registry.execute.return_value = MagicMock(content="ok", meta=None)

        await mgr.approve("proj_approve_bc", "tc_1", session_id="sess_approve")

        ws.broadcast_global.assert_called()
        global_calls = [
            call for call in ws.broadcast_global.call_args_list
            if len(call.args) >= 1
            and isinstance(call.args[0], dict)
            and call.args[0].get("type") == "blocked-count-changed"
        ]
        assert len(global_calls) >= 1, (
            f"Expected blocked-count-changed on approve-leave; "
            f"got: {ws.broadcast_global.call_args_list}"
        )
        payload = global_calls[-1].args[0]
        assert payload["blocked_count"] == 0
