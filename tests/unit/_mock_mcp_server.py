# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""In-repo mock MCP server + helpers for connector tests (Task B3 fixture).

A real ``mcp`` SDK ``FastMCP`` server exposing two tools:
  - ``echo_read``  — carries ``readOnlyHint=True`` (classified read)
  - ``echo_write`` — no annotation (fails closed to write)

``in_memory_opener`` adapts it to the ``ConnectorManager`` session-opener
contract ``(server_url, headers) -> AsyncContextManager[ClientSession]`` using
the SDK's in-memory transport, so tests never open a socket. This same server
is the live smoke target: a Tier-0 ``add_custom`` pointed at a running instance
should surface both tools in a scratch registry.
"""

from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import ToolAnnotations

# A fixed name for the leading emoji-free tool namespace in reflection tests.
MOCK_CONNECTOR_ID = "custom-mock"


def build_mock_server(name: str = "mock") -> FastMCP:
    """Return a FastMCP server with one read tool and one write tool."""
    srv = FastMCP(name)

    @srv.tool(
        name="echo_read",
        description="Echo text back (read-only).",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    def echo_read(text: str) -> str:
        return f"read:{text}"

    @srv.tool(
        name="echo_write",
        description="Echo text back (unclassified -> write).",
    )
    def echo_write(text: str) -> str:
        return f"write:{text}"

    return srv


def in_memory_opener(server: FastMCP):
    """Build a session-opener bound to an in-memory connection to ``server``.

    Matches ``ConnectorManager``'s injectable ``session_opener`` signature:
    ``(server_url, headers) -> AsyncContextManager[ClientSession]``. The url and
    headers are ignored (there is no socket); the manager still calls
    ``initialize()`` on the yielded session, which the in-memory helper tolerates.
    """
    @asynccontextmanager
    async def _opener(server_url, headers=None):
        async with create_connected_server_and_client_session(
            server._mcp_server
        ) as session:
            yield session

    return _opener
