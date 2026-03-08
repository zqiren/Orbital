# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""E2E ACP transport validation through real daemon wiring."""
import os
import pytest
from agent_os.agent.adapters.base import AdapterConfig
from agent_os.agent.adapters.cli_adapter import CLIAdapter
from agent_os.agent.transports.acp_transport import ACPTransport


class TestACPTransportE2E:
    @pytest.mark.asyncio
    async def test_acp_roundtrip_with_dummy_agent(self):
        script = os.path.join(os.path.dirname(__file__), "fixtures", "dummy_acp_agent.py")
        transport = ACPTransport()
        adapter = CLIAdapter(handle="dummy-acp", display_name="Dummy ACP", transport=transport)
        config = AdapterConfig(command="python", workspace=".", approval_patterns=[], args=[script])
        await adapter.start(config)
        assert transport.session_id is not None
        assert transport.session_id == "dummy-sess-001"
        await adapter.send("hello ACP")
        response = adapter._last_response
        assert response is not None
        assert "Echo: hello ACP" in response
        await adapter.stop()

    @pytest.mark.asyncio
    async def test_acp_session_persists(self):
        script = os.path.join(os.path.dirname(__file__), "fixtures", "dummy_acp_agent.py")
        transport = ACPTransport()
        adapter = CLIAdapter(handle="dummy-acp", display_name="Dummy ACP", transport=transport)
        config = AdapterConfig(command="python", workspace=".", approval_patterns=[], args=[script])
        await adapter.start(config)
        sid = transport.session_id
        await adapter.send("msg1")
        assert transport.session_id == sid  # same session
        await adapter.send("msg2")
        assert transport.session_id == sid
        await adapter.stop()

    @pytest.mark.asyncio
    async def test_acp_permission_request(self):
        script = os.path.join(os.path.dirname(__file__), "fixtures", "dummy_acp_agent_with_permissions.py")
        transport = ACPTransport()
        adapter = CLIAdapter(handle="dummy-acp-perms", display_name="Dummy ACP Perms", transport=transport)
        config = AdapterConfig(command="python", workspace=".", approval_patterns=[], args=[script])
        await adapter.start(config)
        await adapter.send("do something")
        response = adapter._last_response
        assert response is not None
        assert "Executed: do something" in response
        await adapter.stop()

    @pytest.mark.asyncio
    async def test_acp_stop_sends_shutdown(self):
        script = os.path.join(os.path.dirname(__file__), "fixtures", "dummy_acp_agent.py")
        transport = ACPTransport()
        adapter = CLIAdapter(handle="dummy-acp", display_name="Dummy ACP", transport=transport)
        config = AdapterConfig(command="python", workspace=".", approval_patterns=[], args=[script])
        await adapter.start(config)
        assert transport.is_alive()
        await adapter.stop()
        assert not transport.is_alive()

    @pytest.mark.asyncio
    async def test_acp_transport_standalone(self):
        """Test ACPTransport directly without CLIAdapter."""
        script = os.path.join(os.path.dirname(__file__), "fixtures", "dummy_acp_agent.py")
        transport = ACPTransport()
        await transport.start("python", [script], ".")
        assert transport.session_id == "dummy-sess-001"
        response = await transport.send("direct ACP test")
        assert "Echo: direct ACP test" in response
        await transport.stop()
        assert not transport.is_alive()
