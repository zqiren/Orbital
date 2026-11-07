# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Folder grants never run on the daemon's event loop.

Feedback row 4 (2026-11-07, Windows 11, v0.20.0): ``start_agent`` called
``grant_folder_access`` synchronously inside an ``async def``. On Windows that
is ``icacls`` over the whole workspace; on a large imported D:\\ tree it held
the event loop for 1016 s, so chat, the UI and every other session froze and
the user restarted twice thinking the scan had hung. The hang log caught the
loop thread inside that call three times (cold-start-scan and
start-onboarding).

Each async caller now hands the grant to a worker thread. The tests record
which thread the provider was called on.
"""

from __future__ import annotations

import asyncio
import os
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.platform.types import PermissionResult

SID = "proj_1_sess0001"


def _recording_provider(seen: list[int]) -> MagicMock:
    provider = MagicMock()

    def grant(path, mode, *, scope):
        seen.append(threading.get_ident())
        return PermissionResult(success=True, path=path)

    provider.grant_folder_access.side_effect = grant
    provider.get_capabilities.return_value = MagicMock(
        isolation_method="sandbox_user", setup_complete=True)
    return provider


def _manager(provider):
    from agent_os.daemon_v2.agent_manager import AgentManager
    from tests.card_doubles import FakeCardStore

    sub_agent_mgr = MagicMock()
    sub_agent_mgr.list_active = MagicMock(return_value=[])
    sub_agent_mgr.stop = AsyncMock()
    sub_agent_mgr.stop_all = AsyncMock()
    return AgentManager(
        project_store=MagicMock(),
        ws_manager=MagicMock(),
        sub_agent_manager=sub_agent_mgr,
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
        platform_provider=provider,
        settings_store=FakeCardStore.with_default(
            provider="anthropic", model="claude-x", key="sk-test",
        ),
    )


@pytest.mark.asyncio
async def test_start_agent_grants_the_workspace_off_the_event_loop(tmp_path):
    from agent_os.daemon_v2.models import AgentConfig

    seen: list[int] = []
    provider = _recording_provider(seen)
    mgr = _manager(provider)
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    config = AgentConfig(workspace=workspace, model="gpt-4", api_key="sk-test")

    with patch("agent_os.daemon_v2.agent_manager.LLMProvider"), \
         patch("agent_os.daemon_v2.agent_manager.ToolRegistry") as MockReg, \
         patch("agent_os.daemon_v2.agent_manager.PromptBuilder"), \
         patch("agent_os.daemon_v2.agent_manager.Session") as MockSession, \
         patch("agent_os.daemon_v2.agent_manager.ContextManager"), \
         patch("agent_os.daemon_v2.agent_manager.AgentLoop") as MockLoop, \
         patch("agent_os.daemon_v2.agent_manager.AutonomyInterceptor"):
        MockSession.new.return_value = MagicMock()
        reg = MagicMock()
        reg.tool_names.return_value = ["read"]
        MockReg.return_value = reg
        loop = MagicMock()
        loop.run = AsyncMock()
        MockLoop.return_value = loop

        await mgr.start_agent("proj_1", config, session_id=SID)

    assert seen, "the workspace grant still runs"
    assert threading.get_ident() not in seen, "grant ran on the event loop"
    provider.grant_folder_access.assert_any_call(
        workspace, "read_write", scope=os.path.realpath(workspace))


@pytest.mark.asyncio
async def test_request_access_approval_grants_off_the_event_loop(tmp_path):
    seen: list[int] = []
    provider = _recording_provider(seen)
    mgr = _manager(provider)
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)

    pending = {
        "tool_name": "request_access",
        "tool_args": {"path": "/some/target", "reason": "r", "access_type": "write"},
    }
    session = MagicMock()
    session.has_result_for.return_value = False
    interceptor = MagicMock()
    interceptor.get_pending.return_value = pending
    task = asyncio.get_running_loop().create_future()
    task.set_result(None)
    mgr._handles[("proj_1", SID)] = MagicMock(
        session=session, task=task, interceptor=interceptor,
        config_snapshot={"workspace": workspace},
    )

    with patch.object(mgr, "_start_loop", new_callable=AsyncMock):
        await mgr.approve("proj_1", "tc_1", session_id=SID)

    assert seen and threading.get_ident() not in seen


@pytest.mark.asyncio
async def test_folder_access_route_grants_off_the_event_loop(tmp_path, monkeypatch):
    from agent_os.api.routes import platform as platform_routes

    seen: list[int] = []
    provider = _recording_provider(seen)
    monkeypatch.setattr(platform_routes, "_platform_provider", provider)
    monkeypatch.setattr(platform_routes, "_resolve_grant_scope",
                        lambda project_id: str(tmp_path))
    req = platform_routes.FolderGrantRequest(path=str(tmp_path), mode="read_write",
                                             project_id="proj_1")
    out = await platform_routes.platform_folders_grant(req)

    assert out["status"] == "ok"
    assert seen and threading.get_ident() not in seen


@pytest.mark.asyncio
async def test_session_scope_route_syncs_portals_off_the_event_loop(monkeypatch):
    """Quick Tasks scope "all" grants a read portal on every project."""
    from agent_os.api.routes import agents_v2

    seen: list[int] = []
    agent_manager = MagicMock()
    agent_manager.set_session_scope.side_effect = (
        lambda *a, **k: seen.append(threading.get_ident()) or {"mode": "all"})
    store = MagicMock()
    store.get_project.return_value = {"is_scratch": True}
    monkeypatch.setattr(agents_v2, "_agent_manager", agent_manager)
    monkeypatch.setattr(agents_v2, "_project_store", store)

    body = agents_v2.SessionScopeBody(mode="all", selected_project_ids=[])
    await agents_v2.put_session_scope("scratch_1", "sess1", body)

    assert seen and threading.get_ident() not in seen
