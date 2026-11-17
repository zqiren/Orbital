# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Agent-held connectors: each agent reports its own MCP servers over its
protocol while it runs; nothing scans its config and nothing is spawned to
ask.

- Claude Code: every turn's ``init`` system message (``mcp_servers``); when
  init still said ``pending`` (claude.ai servers connect after init), one
  ``get_mcp_status`` after that turn's result.
- Codex: the spec-184 ``mcpServerStatus/list`` answer is reused; a worker
  without Orbital's mount gets one status list in the background.
- Failures are swallowed; a spawn is never delayed by the report.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.transports.codex_transport import CodexRpcError, CodexTransport
from agent_os.agent.transports.sdk_transport import SDKTransport


def _init(servers, sid="sdk-1"):
    from claude_agent_sdk.types import SystemMessage
    data = {"session_id": sid}
    if servers is not None:
        data["mcp_servers"] = servers
    return SystemMessage(subtype="init", data=data)


def _result():
    from claude_agent_sdk.types import ResultMessage
    return ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1,
                         is_error=False, num_turns=1, session_id="sdk-1",
                         total_cost_usd=0.0, usage={}, result=None)


def _sdk():
    t = SDKTransport()
    got = []
    t.set_mcp_servers_sink(got.append)
    return t, got


INIT = [{"name": "playwright", "status": "connected", "source": "user"},
        {"name": "claude.ai Gmail", "status": "pending", "source": "claudeai"},
        {"name": "orbital", "status": "connected", "source": "dynamic"}]


# ---------------------------------------------------------------- Claude Code

def test_claude_init_reports_names_and_status():
    t, got = _sdk()
    t._message_to_events(_init(INIT))
    assert got == [[{"name": "playwright", "status": "connected"},
                    {"name": "claude.ai Gmail", "status": "pending"},
                    {"name": "orbital", "status": "connected"}]]


def test_claude_reports_on_every_turns_init():
    t, got = _sdk()
    t._message_to_events(_init(INIT))
    t._message_to_events(_init(INIT))          # same session, next turn
    assert len(got) == 2


@pytest.mark.parametrize("servers", [None, "x", 5])
def test_claude_init_without_a_server_list_reports_nothing(servers):
    t, got = _sdk()
    t._message_to_events(_init(servers))
    assert got == []


def test_a_sink_failure_never_breaks_the_turn():
    t = SDKTransport()

    def boom(_):
        raise RuntimeError("x")
    t.set_mcp_servers_sink(boom)
    events = t._message_to_events(_init(INIT))
    assert [e.event_type for e in events] == ["thread_started"]


def test_no_sink_is_fine():
    SDKTransport()._message_to_events(_init(INIT))


async def test_pending_at_init_asks_get_mcp_status_once_after_the_result():
    t, got = _sdk()
    t._client = MagicMock()
    t._client.get_mcp_status = AsyncMock(return_value={"mcpServers": [
        {"name": "playwright", "status": "connected",
         "config": {"type": "stdio", "command": "npx"}},
        {"name": "claude.ai Gmail", "status": "connected",
         "config": {"type": "claudeai-proxy", "url": "https://x"}}]})
    await t._handle_message(_init(INIT))
    await t._handle_message(_result())
    await asyncio.wait_for(t._mcp_status_task, 2)
    assert got[-1] == [{"name": "playwright", "status": "connected"},
                       {"name": "claude.ai Gmail", "status": "connected"}]
    # Once per process: the next turn's pending init does not ask again.
    await t._handle_message(_init(INIT))
    await t._handle_message(_result())
    await asyncio.sleep(0)
    assert t._client.get_mcp_status.await_count == 1


async def test_nothing_pending_means_no_extra_request():
    t, got = _sdk()
    t._client = MagicMock()
    t._client.get_mcp_status = AsyncMock()
    await t._handle_message(_init([{"name": "playwright", "status": "connected"}]))
    await t._handle_message(_result())
    await asyncio.sleep(0)
    t._client.get_mcp_status.assert_not_awaited()
    assert len(got) == 1


async def test_a_failed_status_request_is_swallowed():
    t, got = _sdk()
    t._client = MagicMock()
    t._client.get_mcp_status = AsyncMock(side_effect=RuntimeError("old CLI"))
    await t._handle_message(_init(INIT))
    await t._handle_message(_result())
    await asyncio.wait_for(t._mcp_status_task, 2)
    assert len(got) == 1                     # only the init report


# ---------------------------------------------------------------------- Codex

def _codex(**kw):
    t = CodexTransport(**kw)
    t._thread_id = "T1"
    got = []
    t.set_mcp_servers_sink(got.append)
    return t, got


STATUS = {"data": [
    {"name": "orbital", "tools": {"dispatch": {}}, "authStatus": "bearerToken"},
    {"name": "node_repl", "tools": {"js": {}}, "authStatus": "unsupported",
     "serverInfo": {"name": "rmcp"}, "httpOrigin": None},
    {"name": "chatcut", "tools": {}, "authStatus": "notLoggedIn",
     "toolsError": "OAuth required", "httpOrigin": "https://api.example"},
]}


async def test_codex_mount_check_answer_is_reported():
    t, got = _codex(dispatch_url="http://127.0.0.1:8541/mcp")

    async def fake_request(method, params=None, timeout=30.0):
        return STATUS
    t._request = fake_request
    await t._check_dispatch_mount()
    assert t.dispatch_mounted is True
    assert got == [[{"name": "orbital", "status": "connected"},
                    {"name": "node_repl", "status": "connected"},
                    {"name": "chatcut", "status": "needs-auth"}]]


async def test_codex_without_the_mount_asks_once_in_the_background():
    t, got = _codex()
    calls = []

    async def fake_request(method, params=None, timeout=30.0):
        calls.append((method, params))
        return STATUS
    t._request = fake_request
    await t._publish_mcp_status()
    assert calls == [("mcpServerStatus/list",
                      {"threadId": "T1", "detail": "toolsAndAuthOnly"})]
    assert [s["name"] for s in got[0]] == ["orbital", "node_repl", "chatcut"]


@pytest.mark.parametrize("exc", [asyncio.TimeoutError(),
                                 CodexRpcError("mcpServerStatus/list",
                                               {"code": -32601, "message": "no"})])
async def test_codex_status_failure_reports_nothing(exc):
    t, got = _codex()

    async def fake_request(method, params=None, timeout=30.0):
        raise exc
    t._request = fake_request
    await t._publish_mcp_status()
    assert got == []


async def test_codex_start_without_mount_schedules_the_status_list(monkeypatch):
    from agent_os.agent.transports import codex_transport as ct
    from tests.unit.test_codex_transport import _FakeStdin
    t = CodexTransport()
    t.set_mcp_servers_sink(lambda servers: None)
    order = []

    class _P:
        pid = os.getpid()
        stdin = _FakeStdin()
        returncode = None

    async def fake_exec(*a, **kw):
        return _P()

    async def nothing(*a, **kw):
        return None

    async def fake_request(method, params=None, timeout=30.0):
        return {}

    async def open_thread(workspace):
        order.append("open")

    async def status():
        order.append("status")

    monkeypatch.setattr(ct.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(t, "_read_loop", nothing)
    monkeypatch.setattr(t, "_drain_stderr", nothing)
    monkeypatch.setattr(t, "_publish_rate_limits", nothing)
    monkeypatch.setattr(t, "_publish_skills", nothing)
    monkeypatch.setattr(t, "_request", fake_request)
    monkeypatch.setattr(t, "_notify", nothing)
    monkeypatch.setattr(t, "_open_thread", open_thread)
    monkeypatch.setattr(t, "_publish_mcp_status", status)
    monkeypatch.setattr("agent_os.utils.windows_job.assign_worker", lambda pid: None)
    await t.start("/fake/codex", [], "/ws", {})
    await asyncio.wait_for(t._mcp_status_task, 2)
    assert order == ["open", "status"]


async def test_codex_with_the_mount_does_not_ask_twice(monkeypatch):
    t = CodexTransport(dispatch_url="http://127.0.0.1:8541/mcp")
    t._thread_id = "T1"
    asked = []

    async def status():
        asked.append(1)
    monkeypatch.setattr(t, "_publish_mcp_status", status)
    t._schedule_mcp_status()
    assert t._mcp_status_task is None and asked == []


# --------------------------------------------------- wiring into the registry

async def test_a_spawned_worker_reports_into_the_registry(tmp_path):
    from agent_os.daemon_v2 import agent_connectors as ac
    from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
    ac.configure(str(tmp_path))
    try:
        t, _ = _sdk()
        SubAgentManager._attach_commands_sink(t, "claude-code", str(tmp_path))
        t._message_to_events(_init(INIT))
        for _ in range(50):
            if ac.held("claude-code"):
                break
            await asyncio.sleep(0.02)
        assert [c["name"] for c in ac.held("claude-code")] == ["Gmail", "playwright"]
    finally:
        ac.configure(None)
