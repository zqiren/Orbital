# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Connectors shared with workers, served on the daemon's one MCP endpoint.

The bearer token a worker already mounts (spec 161) resolves a context that
now also carries the connectors that worker may use. The endpoint lists those
connectors' tools under names every worker's MCP client accepts, runs a read,
asks the user before a write (the sub-agent approval card), and returns a
denial as a tool error. A worker the manager dispatched has a token for its
connectors but no ``dispatch`` tool. Access taken away mid-session stops the
calls at once, even though the worker's tool list is fixed until it rebuilds.
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

from agent_os.connectors import load_catalog
from agent_os.connectors.manager import ConnectorManager
from agent_os.daemon_v2.connector_sharing import exposed_tool_name
from agent_os.daemon_v2.dispatch_tools import DispatchContext, attach_dispatch_server

from tests.unit._mock_mcp_server import build_mock_server, in_memory_opener

HEADERS = {"Accept": "application/json, text/event-stream",
           "Content-Type": "application/json"}
# Anthropic / OpenAI function-name rule; Claude Code prefixes mcp__orbital__ (14).
VALID = re.compile(r"^[A-Za-z0-9_-]{1,50}$")


async def _cm(tmp_path):
    mgr = ConnectorManager(catalog=load_catalog(), credential_store=None,
                           data_dir=str(tmp_path),
                           session_opener=in_memory_opener(build_mock_server()))
    await mgr.add_custom("Mock", "https://mock/mcp", "none")
    return mgr


def _sam(cm):
    sam = MagicMock()
    sam.send = AsyncMock(return_value="Message sent to codex. Transcript: t.jsonl")
    tokens = {
        # A pinned seat-holder with the connector.
        "tok-top": DispatchContext("p1", "s1", "claude-code",
                                   agents=("claude-code", "codex"),
                                   connectors=("custom-mock",)),
        # A worker the manager dispatched: connectors only.
        "tok-low": DispatchContext("p1", "s1", "codex", can_dispatch=False,
                                   connectors=("custom-mock",)),
        # Today's seat-holder without connectors.
        "tok-plain": DispatchContext("p1", "s1", "claude-code",
                                     agents=("claude-code", "codex")),
    }
    sam.dispatch_context = lambda token: tokens.get(token)
    sam._dispatch_url = None
    sam._connector_manager = cm
    sam.live = {"custom-mock"}
    sam.live_connector_ids = lambda pid, handle: tuple(sorted(sam.live))
    sam.request_connector_approval = AsyncMock(return_value=True)
    # Writes ask under check-in (test_connector_sharing_autonomy covers all three).
    from agent_os.agent.prompt_builder import Autonomy
    sam.project_autonomy = lambda pid: Autonomy.CHECK_IN
    return sam


async def _rpc(client, method, params, token):
    headers = dict(HEADERS)
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    resp = await client.post("/mcp", headers=headers, json={
        "jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.fixture
async def served(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OS_PORT", "8541")
    cm = await _cm(tmp_path)
    app = FastAPI()
    sam = _sam(cm)
    server = attach_dispatch_server(app, sam, connector_manager=cm)
    await server.start()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://127.0.0.1:8541") as client:
        yield client, sam
    await server.stop()


async def _names(client, token):
    body = await _rpc(client, "tools/list", {}, token)
    return [t["name"] for t in body["result"]["tools"]]


def test_exposed_names_are_valid_for_every_client_and_stable():
    assert exposed_tool_name("gmail", "send_message") == "gmail__send_message"
    assert exposed_tool_name("google-calendar", "list_events") == "google-calendar__list_events"
    odd = exposed_tool_name("custom-my-server", "do.thing/now")
    assert odd == "custom-my-server__do_thing_now"
    long = exposed_tool_name("custom-" + "x" * 40, "a_really_long_tool_name_here")
    assert VALID.match(long) and len(long) == 50
    assert long == exposed_tool_name("custom-" + "x" * 40, "a_really_long_tool_name_here")
    assert long != exposed_tool_name("custom-" + "x" * 40, "a_really_long_tool_name_hers")
    for name in (odd, long, "gmail__send_message"):
        assert VALID.match(name)


async def test_attach_hands_the_connector_manager_to_the_sub_agent_manager(served):
    _, sam = served
    assert sam._connector_manager is not None


async def test_a_seat_holder_with_connectors_lists_dispatch_and_the_connector_tools(served):
    client, _ = served
    names = await _names(client, "tok-top")
    assert names == ["dispatch", "custom-mock__echo_read", "custom-mock__echo_write"]
    assert all(VALID.match(n) for n in names)


async def test_a_dispatched_worker_lists_connector_tools_but_not_dispatch(served):
    client, _ = served
    assert await _names(client, "tok-low") == [
        "custom-mock__echo_read", "custom-mock__echo_write"]


async def test_a_dispatched_worker_cannot_call_dispatch(served):
    client, sam = served
    body = await _rpc(client, "tools/call", {"name": "dispatch", "arguments": {
        "agent": "claude-code", "task": "x"}}, "tok-low")
    assert body["result"]["isError"] is True
    sam.send.assert_not_awaited()


async def test_a_seat_holder_without_connectors_lists_exactly_todays_tool(served):
    client, _ = served
    body = await _rpc(client, "tools/list", {}, "tok-plain")
    tools = body["result"]["tools"]
    assert [t["name"] for t in tools] == ["dispatch"]


async def test_the_tool_carries_the_connector_name_and_schema(served):
    client, _ = served
    body = await _rpc(client, "tools/list", {}, "tok-low")
    read = next(t for t in body["result"]["tools"] if t["name"].endswith("echo_read"))
    assert read["description"].startswith("Mock: ")
    assert read["inputSchema"]["properties"]["text"]["type"] == "string"
    assert read["annotations"]["readOnlyHint"] is True


async def test_a_read_runs_without_asking(served):
    client, sam = served
    body = await _rpc(client, "tools/call", {"name": "custom-mock__echo_read",
                                             "arguments": {"text": "hi"}}, "tok-low")
    result = body["result"]
    assert result.get("isError") is not True
    assert result["content"][0]["text"] == "read:hi"
    sam.request_connector_approval.assert_not_awaited()


async def test_a_write_asks_first_and_runs_when_approved(served):
    client, sam = served
    body = await _rpc(client, "tools/call", {"name": "custom-mock__echo_write",
                                             "arguments": {"text": "hi"}}, "tok-low")
    assert body["result"]["content"][0]["text"] == "write:hi"
    sam.request_connector_approval.assert_awaited_once()
    ctx, label, args = sam.request_connector_approval.await_args.args
    assert (ctx.handle, label, args) == ("codex", "custom-mock.echo_write", {"text": "hi"})


async def test_a_denied_write_never_runs_and_says_so(served, monkeypatch):
    client, sam = served
    sam.request_connector_approval.return_value = False
    calls = []
    real = sam._connector_manager.call_tool

    async def spy(*a, **kw):
        calls.append(a)
        return await real(*a, **kw)
    monkeypatch.setattr(sam._connector_manager, "call_tool", spy)
    body = await _rpc(client, "tools/call", {"name": "custom-mock__echo_write",
                                             "arguments": {"text": "hi"}}, "tok-top")
    result = body["result"]
    assert result["isError"] is True
    assert "declined" in result["content"][0]["text"]
    assert calls == []


async def test_access_taken_away_mid_session_stops_calls_and_lists(served):
    client, sam = served
    sam.live = set()
    body = await _rpc(client, "tools/call", {"name": "custom-mock__echo_read",
                                             "arguments": {"text": "hi"}}, "tok-low")
    assert body["result"]["isError"] is True
    assert "no longer" in body["result"]["content"][0]["text"]
    assert await _names(client, "tok-top") == ["dispatch"]


async def test_unknown_token_gets_no_connector_tools(served):
    client, _ = served
    assert await _names(client, "tok-nope") == ["dispatch"]
    body = await _rpc(client, "tools/call", {"name": "custom-mock__echo_read",
                                             "arguments": {"text": "hi"}}, "tok-nope")
    assert body["result"]["isError"] is True


async def test_a_connector_that_cannot_list_costs_only_its_own_tools(served, monkeypatch):
    client, sam = served
    await sam._connector_manager.add_custom("Broken", "https://broken/mcp", "none")
    sam.live = {"custom-mock", "custom-broken"}
    tokens = {"tok-two": DispatchContext("p1", "s1", "codex", can_dispatch=False,
                                         connectors=("custom-broken", "custom-mock"))}
    old = sam.dispatch_context
    sam.dispatch_context = lambda t: tokens.get(t) or old(t)
    real = sam._connector_manager.list_tools

    async def flaky(cid):
        if cid == "custom-broken":
            raise RuntimeError("down")
        return await real(cid)
    monkeypatch.setattr(sam._connector_manager, "list_tools", flaky)
    assert await _names(client, "tok-two") == [
        "custom-mock__echo_read", "custom-mock__echo_write"]


# Live 2026-11-17: Claude Code gave up on a write after 60 s while the card
# was up; the daemon kept waiting and ran the write once the user approved,
# with no one left to read the result. A write whose caller has gone is
# never run, and its card comes down.
async def test_a_write_the_worker_gave_up_on_never_runs(tmp_path, monkeypatch):
    import asyncio
    from agent_os.daemon_v2 import connector_sharing
    monkeypatch.setattr(connector_sharing, "DISCONNECT_POLL_S", 0.01)
    cm = await _cm(tmp_path)
    sam = _sam(cm)
    asked = asyncio.Event()
    cancelled = []

    async def never_answered(ctx, label, args):
        asked.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.append(label)
            raise
    sam.request_connector_approval = never_answered
    calls = []
    real = cm.call_tool

    async def spy(*a, **kw):
        calls.append(a)
        return await real(*a, **kw)
    monkeypatch.setattr(cm, "call_tool", spy)

    class _Req:
        gone = False

        async def is_disconnected(self):
            return self.gone
    req = _Req()
    tools = connector_sharing.WorkerConnectorTools(sam)
    ctx = sam.dispatch_context("tok-low")
    task = asyncio.create_task(tools.call(ctx, "custom-mock__echo_write", {"text": "x"},
                                          request=req))
    await asyncio.wait_for(asked.wait(), 5)
    req.gone = True
    result = await asyncio.wait_for(task, 5)
    assert result.isError is True and "not run" in result.content[0].text
    assert cancelled == ["custom-mock.echo_write"]
    assert calls == []
