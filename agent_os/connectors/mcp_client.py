# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Production MCP session opener over streamable HTTP (Spec 011 §2; Task B2).

``ConnectorManager`` takes an injectable ``session_opener`` with the signature
``(server_url, headers) -> AsyncContextManager[ClientSession]``. This module
supplies the real one (the ``mcp`` SDK's streamable-HTTP transport); tests
inject an in-memory opener instead so nothing leaves the machine.
"""

from contextlib import asynccontextmanager

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


@asynccontextmanager
async def streamable_http_opener(server_url: str, headers: dict[str, str] | None = None):
    """Open a ClientSession against a remote streamable-HTTP MCP server.

    The session is yielded UN-initialized; ``ConnectorManager`` calls
    ``initialize()`` (the single init point across both real and in-memory
    openers). The manager keeps the yielded session open (cached) and closes
    the context on disconnect / drop."""
    async with streamablehttp_client(server_url, headers=headers) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            yield session
