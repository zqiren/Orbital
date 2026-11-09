# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 161 §5.1: one streamable-HTTP MCP endpoint at ``/mcp`` on the daemon.

The bearer token in the ``Authorization`` header resolves the seat-holder's
``DispatchContext`` (minted per top-of-session spawn); the one tool calls
``dispatch`` with it. An unknown token is tool error text, never a stack
trace. Without ``AGENT_OS_PORT`` nothing is attached and one warning says so.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

from agent_os.daemon_v2.dispatch_tools import DispatchContext, attach_dispatch_server

HEADERS = {"Accept": "application/json, text/event-stream",
           "Content-Type": "application/json"}


def _sam():
    sam = MagicMock()
    sam.send = AsyncMock(return_value="Message sent to codex. Transcript: t.jsonl")
    tokens = {"tok-good": DispatchContext("p1", "s1", "claude-code",
                                          agents=("claude-code", "codex", "pi"))}
    sam.dispatch_context = lambda token: tokens.get(token)
    sam._dispatch_url = None
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
async def served(monkeypatch):
    monkeypatch.setenv("AGENT_OS_PORT", "8541")
    app = FastAPI()
    sam = _sam()
    server = attach_dispatch_server(app, sam)
    assert server is not None
    await server.start()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://127.0.0.1:8541") as client:
        yield client, sam
    await server.stop()


@pytest.mark.asyncio
async def test_attach_records_the_loopback_url(served):
    _, sam = served
    assert sam._dispatch_url == "http://127.0.0.1:8541/mcp"


@pytest.mark.asyncio
async def test_bearer_token_resolves_the_context_and_the_tool_calls_dispatch(served):
    client, sam = served
    body = await _rpc(client, "tools/call",
                      {"name": "dispatch", "arguments": {"agent": "codex", "task": "write hello.txt"}},
                      "tok-good")
    result = body["result"]
    assert result.get("isError") is not True
    text = result["content"][0]["text"]
    assert text.startswith("Dispatched to codex. Message sent to codex.")
    # Plain text for the model, not a structured {"result": ...} wrapper.
    assert "structuredContent" not in result or result["structuredContent"] is None
    sam.send.assert_awaited_once_with(
        "p1", "codex", "write hello.txt", session_id="s1", depth=1,
        fresh=False, initiator="worker:claude-code")


@pytest.mark.asyncio
@pytest.mark.parametrize("token", ["tok-unknown", None])
async def test_unknown_or_missing_token_is_tool_error_text(served, token):
    client, sam = served
    body = await _rpc(client, "tools/call",
                      {"name": "dispatch", "arguments": {"agent": "codex", "task": "x"}},
                      token)
    text = body["result"]["content"][0]["text"]
    assert text.startswith("Error:")
    assert "Traceback" not in text
    sam.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_raising_is_error_text_not_a_stack_trace(served):
    client, sam = served
    sam.send.side_effect = RuntimeError("boom")
    body = await _rpc(client, "tools/call",
                      {"name": "dispatch", "arguments": {"agent": "codex", "task": "x"}},
                      "tok-good")
    text = body["result"]["content"][0]["text"]
    assert text.startswith("Error:") and "Traceback" not in text


@pytest.mark.asyncio
async def test_tool_list_names_the_other_enabled_agents(served):
    client, _ = served
    body = await _rpc(client, "tools/list", {}, "tok-good")
    [tool] = body["result"]["tools"]
    assert tool["name"] == "dispatch"
    assert "codex, pi" in tool["description"]
    assert "claude-code" not in tool["description"]
    props = tool["inputSchema"]["properties"]
    assert set(props) == {"agent", "task", "fresh"}
    assert set(tool["inputSchema"].get("required", [])) == {"agent", "task"}


@pytest.mark.asyncio
async def test_tool_says_when_to_use_it_instead_of_a_shell_command(served):
    """Live 2026-11-09 (v0.21.0, Quick Tasks): pinned Codex listed the tool,
    then ran ``claude -p`` through its shell instead, and the reply never
    reached the user. The description must name the trigger and rule out
    the shell route."""
    client, _ = served
    body = await _rpc(client, "tools/list", {}, "tok-good")
    [tool] = body["result"]["tools"]
    desc = tool["description"].lower()
    assert "asks for another agent" in desc
    assert "shell" in desc


@pytest.mark.asyncio
async def test_every_parameter_is_described(served):
    client, _ = served
    body = await _rpc(client, "tools/list", {}, "tok-good")
    [tool] = body["result"]["tools"]
    props = tool["inputSchema"]["properties"]
    for name in ("agent", "task", "fresh"):
        assert props[name].get("description"), name
    assert props["fresh"].get("default") is False


def test_no_port_means_no_attach_and_one_warning(monkeypatch, caplog):
    monkeypatch.delenv("AGENT_OS_PORT", raising=False)
    app = FastAPI()
    sam = _sam()
    before = len(app.router.routes)
    with caplog.at_level(logging.WARNING):
        assert attach_dispatch_server(app, sam) is None
    assert len(app.router.routes) == before
    assert sam._dispatch_url is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "AGENT_OS_PORT" in warnings[0].getMessage()
