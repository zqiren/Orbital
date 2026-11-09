# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 161 for ACP workers: the seat-holder gets Orbital's dispatch endpoint.

ACP lets the client hand MCP servers to the agent on every session open
(``session/new``, ``session/load``, ``session/resume``). When the transport
is given ``dispatch_mcp`` and the agent advertises HTTP MCP
(``agentCapabilities.mcpCapabilities.http``, which codebuddy, cursor, dsh,
kimi-code and minimax-code all do, probe 2026-11-09), it passes the one
``orbital`` HTTP server with the bearer header. Without HTTP support it
mounts nothing and says so (``dispatch_mounted`` is False).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from acp.schema import (
    AgentCapabilities,
    HttpHeader,
    HttpMcpServer,
    InitializeResponse,
    LoadSessionResponse,
    McpCapabilities,
    NewSessionResponse,
    ResumeSessionResponse,
    SessionCapabilities,
    SessionResumeCapabilities,
)

from agent_os.agent.transports.acp_sdk_transport import ACPSDKTransport

DISPATCH = {"url": "http://127.0.0.1:8541/mcp", "token": "tok-123"}
EXPECTED = [HttpMcpServer(type="http", name="orbital", url=DISPATCH["url"],
                          headers=[HttpHeader(name="Authorization",
                                              value="Bearer tok-123")])]


def _transport(tmp_path, *, http=True, resume=None, load=False, resume_cap=False,
               dispatch=DISPATCH):
    t = ACPSDKTransport(resume_record={"session_id": resume} if resume else None,
                        dispatch_mcp=dispatch)
    t._workspace = str(tmp_path)
    t._connection = AsyncMock()
    t._connection.initialize.return_value = InitializeResponse(
        protocol_version=1,
        agent_capabilities=AgentCapabilities(
            load_session=load,
            mcp_capabilities=McpCapabilities(http=http),
            session_capabilities=SessionCapabilities(
                resume=SessionResumeCapabilities() if resume_cap else None),
        ),
    )
    t._connection.new_session.return_value = NewSessionResponse(
        session_id="fresh", config_options=[])
    t._connection.load_session.return_value = LoadSessionResponse(config_options=[])
    t._connection.resume_session.return_value = ResumeSessionResponse(config_options=[])
    t._wait_for_session_updates = AsyncMock()
    return t


@pytest.mark.asyncio
async def test_new_session_carries_the_orbital_http_server(tmp_path):
    t = _transport(tmp_path)
    await t._initialize_and_open_session()
    assert t._connection.new_session.await_args.kwargs["mcp_servers"] == EXPECTED
    assert t.dispatch_mounted is True


@pytest.mark.asyncio
async def test_load_session_carries_it_too(tmp_path):
    t = _transport(tmp_path, resume="old", load=True)
    await t._initialize_and_open_session()
    assert t._connection.load_session.await_args.kwargs["mcp_servers"] == EXPECTED


@pytest.mark.asyncio
async def test_resume_session_carries_it_too(tmp_path):
    t = _transport(tmp_path, resume="old", resume_cap=True)
    await t._initialize_and_open_session()
    assert t._connection.resume_session.await_args.kwargs["mcp_servers"] == EXPECTED


@pytest.mark.asyncio
async def test_no_http_support_mounts_nothing(tmp_path):
    t = _transport(tmp_path, http=False)
    await t._initialize_and_open_session()
    assert not t._connection.new_session.await_args.kwargs.get("mcp_servers")
    assert t.dispatch_mounted is False


@pytest.mark.asyncio
async def test_a_worker_below_the_seat_gets_no_server(tmp_path):
    t = _transport(tmp_path, dispatch=None)
    await t._initialize_and_open_session()
    assert not t._connection.new_session.await_args.kwargs.get("mcp_servers")
    assert t.dispatch_mounted is False
