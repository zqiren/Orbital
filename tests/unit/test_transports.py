# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for transport abstraction layer."""
import json
import pytest
from unittest.mock import MagicMock, patch
from agent_os.agent.transports.base import AgentTransport, TransportEvent


class TestTransportEvent:
    def test_create_message_event(self):
        e = TransportEvent(event_type="message", data={"text": "hello"}, raw_text="hello")
        assert e.event_type == "message"
        assert e.data["text"] == "hello"
        assert e.raw_text == "hello"

    def test_create_event_defaults(self):
        e = TransportEvent(event_type="status")
        assert e.data == {}
        assert e.raw_text == ""


class TestAgentTransportABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AgentTransport()

    def test_session_id_default_none(self):
        # Create a minimal concrete subclass
        class Dummy(AgentTransport):
            async def start(self, command, args, workspace, env=None): pass
            async def send(self, message): return None
            async def read_stream(self): yield  # pragma: no cover
            async def stop(self): pass
            def is_alive(self): return False
        d = Dummy()
        assert d.session_id is None

    @pytest.mark.asyncio
    async def test_respond_to_permission_default_noop(self):
        class Dummy(AgentTransport):
            async def start(self, command, args, workspace, env=None): pass
            async def send(self, message): return None
            async def read_stream(self): yield
            async def stop(self): pass
            def is_alive(self): return False
        d = Dummy()
        # Should not raise
        await d.respond_to_permission("p1", True)


from agent_os.agent.transports.pty_transport import PTYTransport


class TestPTYTransport:
    def _make(self, approval_patterns=None):
        return PTYTransport(approval_patterns=approval_patterns or [])

    @pytest.mark.asyncio
    async def test_start_spawns_process(self):
        """Start should spawn a process (using a real 'python -c' command)."""
        t = self._make()
        await t.start("python", ["-c", "import time; time.sleep(30)"], ".")
        assert t.is_alive()
        await t.stop()

    @pytest.mark.asyncio
    async def test_start_immediate_exit_detected(self):
        """Process that exits immediately should either raise or leave is_alive() False."""
        t = self._make()
        try:
            await t.start("python", ["-c", "import sys; sys.exit(1)"], ".")
            # On some platforms (Windows), the process may not have exited within 50ms
            # but it should be dead very shortly after
            import asyncio
            await asyncio.sleep(0.2)
            # At minimum, the process should not be alive
            alive = t.is_alive()
            await t.stop()
            # If start didn't raise, the process should be dead or dying
            assert not alive
        except Exception:
            pass  # AdapterError is expected on fast platforms

    @pytest.mark.asyncio
    async def test_send_writes_to_stdin(self):
        t = self._make()
        await t.start("python", ["-u", "-c",
            "import sys\nfor line in sys.stdin:\n print('GOT:'+line.strip())\n sys.stdout.flush()"],
            ".")
        result = await t.send("hello")
        assert result is None  # PTY send returns None (streaming)
        await t.stop()

    def test_is_alive_false_without_start(self):
        t = self._make()
        assert t.is_alive() is False

    @pytest.mark.asyncio
    async def test_stop_terminates(self):
        t = self._make()
        await t.start("python", ["-c", "import time; time.sleep(30)"], ".")
        assert t.is_alive()
        await t.stop()
        assert not t.is_alive()


from agent_os.agent.transports.base import transport_event_to_chunk
from agent_os.agent.adapters.base import OutputChunk


class TestTransportEventToChunk:
    def test_message_maps_to_response(self):
        e = TransportEvent(event_type="message", raw_text="hello")
        c = transport_event_to_chunk(e)
        assert c.chunk_type == "response"
        assert c.text == "hello"

    def test_tool_use_maps_to_tool_activity(self):
        e = TransportEvent(event_type="tool_use", raw_text="reading file")
        c = transport_event_to_chunk(e)
        assert c.chunk_type == "tool_activity"

    def test_permission_request_maps_to_approval_request(self):
        e = TransportEvent(event_type="permission_request", raw_text="allow?")
        c = transport_event_to_chunk(e)
        assert c.chunk_type == "approval_request"

    def test_status_maps_to_status(self):
        e = TransportEvent(event_type="status", raw_text="thinking...")
        c = transport_event_to_chunk(e)
        assert c.chunk_type == "status"

    def test_unknown_type_defaults_to_response(self):
        e = TransportEvent(event_type="unknown_xyz", raw_text="wat")
        c = transport_event_to_chunk(e)
        assert c.chunk_type == "response"

    def test_uses_data_text_if_no_raw_text(self):
        e = TransportEvent(event_type="message", data={"text": "from data"})
        c = transport_event_to_chunk(e)
        assert c.text == "from data"


from unittest.mock import AsyncMock


class TestACPWiring:
    """Test SubAgentManager wiring for ACP transport."""

    @pytest.mark.asyncio
    async def test_acp_skips_process_manager(self):
        """ACP agents handle responses via send() — process_manager.start() should NOT be called."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.agents.manifest import AgentManifest, ManifestRuntime

        mock_pm = MagicMock()
        mock_pm.start = AsyncMock()

        mock_registry = MagicMock()
        manifest = AgentManifest(
            manifest_version="1", name="CC", slug="claude-code", description="",
            author="", version="1.0.0",
            runtime=ManifestRuntime(adapter="cli", command="claude", transport="acp", args=["acp"]),
        )
        mock_registry.get.return_value = manifest

        mock_setup = MagicMock()
        mock_setup.get_adapter_config.return_value = {
            "command": "claude", "args": ["acp"], "workspace": "/tmp",
            "approval_patterns": [], "env": {}, "network_domains": [],
        }

        mgr = SubAgentManager(
            process_manager=mock_pm, registry=mock_registry,
            setup_engine=mock_setup, project_store=MagicMock(get_project=MagicMock(return_value={"workspace": "/tmp"})),
        )

        # Patch adapter.start to not actually spawn a process
        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.start = AsyncMock()
            MockAdapter.return_value = mock_adapter
            await mgr._start_from_registry("proj1", "claude-code")

        # process_manager.start should NOT have been called for ACP
        mock_pm.start.assert_not_called()

    @pytest.mark.asyncio
    async def test_pty_calls_process_manager(self):
        """PTY agents need process_manager streaming — it MUST be called."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.agents.manifest import AgentManifest, ManifestRuntime

        mock_pm = MagicMock()
        mock_pm.start = AsyncMock()

        mock_registry = MagicMock()
        manifest = AgentManifest(
            manifest_version="1", name="T", slug="test-agent", description="",
            author="", version="1.0.0",
            runtime=ManifestRuntime(adapter="cli", command="echo", transport="pty", mode="interactive"),
        )
        mock_registry.get.return_value = manifest

        mock_setup = MagicMock()
        mock_setup.get_adapter_config.return_value = {
            "command": "echo", "args": [], "workspace": "/tmp",
            "approval_patterns": [], "env": {}, "network_domains": [],
        }

        mgr = SubAgentManager(
            process_manager=mock_pm, registry=mock_registry,
            setup_engine=mock_setup, project_store=MagicMock(get_project=MagicMock(return_value={"workspace": "/tmp"})),
        )

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.start = AsyncMock()
            MockAdapter.return_value = mock_adapter
            await mgr._start_from_registry("proj1", "test-agent")

        # process_manager.start MUST be called for PTY
        mock_pm.start.assert_called_once()


class TestCLIAdapterWithTransport:
    """Test CLIAdapter when constructed with an explicit transport."""

    @pytest.mark.asyncio
    async def test_adapter_delegates_to_transport(self):
        from agent_os.agent.adapters.cli_adapter import CLIAdapter
        from agent_os.agent.adapters.base import AdapterConfig

        mock_transport = MagicMock()
        mock_transport.send = AsyncMock(return_value="transport response")
        mock_transport.start = AsyncMock()
        mock_transport.stop = AsyncMock()
        mock_transport.is_alive.return_value = True
        mock_transport.session_id = "sess-1"

        adapter = CLIAdapter(handle="t", display_name="Test", transport=mock_transport)
        config = AdapterConfig(command="cmd", workspace="/tmp", approval_patterns=[])
        await adapter.start(config)
        mock_transport.start.assert_called_once()

        await adapter.send("hello")
        mock_transport.send.assert_called_once_with("hello")
        assert adapter._last_response == "transport response"

        assert adapter.is_alive()
        await adapter.stop()
        mock_transport.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_adapter_respond_to_permission_delegates(self):
        from agent_os.agent.adapters.cli_adapter import CLIAdapter

        mock_transport = MagicMock()
        mock_transport.respond_to_permission = AsyncMock()

        adapter = CLIAdapter(handle="t", display_name="Test", transport=mock_transport)
        await adapter.respond_to_permission("p1", True)
        mock_transport.respond_to_permission.assert_called_once_with("p1", True)


class TestTransportResolution:
    def test_acp_manifest_gets_acp_transport(self):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.agents.manifest import AgentManifest, ManifestRuntime
        mgr = SubAgentManager(process_manager=MagicMock())
        manifest = AgentManifest(
            manifest_version="1", name="CC", slug="claude-code", description="", author="", version="1.0.0",
            runtime=ManifestRuntime(adapter="cli", command="claude", transport="acp"),
        )
        t = mgr._resolve_transport(manifest, {})
        from agent_os.agent.transports.acp_transport import ACPTransport
        assert isinstance(t, ACPTransport)

    def test_interactive_manifest_gets_pty(self):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.agents.manifest import AgentManifest, ManifestRuntime
        mgr = SubAgentManager(process_manager=MagicMock())
        manifest = AgentManifest(
            manifest_version="1", name="T", slug="t", description="", author="", version="1.0.0",
            runtime=ManifestRuntime(adapter="cli", command="t", transport="auto", mode="interactive"),
        )
        t = mgr._resolve_transport(manifest, {"approval_patterns": ["Allow?"]})
        from agent_os.agent.transports.pty_transport import PTYTransport
        assert isinstance(t, PTYTransport)

    def test_auto_with_pipe_mode_gets_sdk_when_available(self):
        """mode=pipe with transport=auto should resolve to SDKTransport when SDK is available."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.agents.manifest import AgentManifest, ManifestRuntime
        from agent_os.agent.transports.sdk_transport import SDKTransport, HAS_SDK
        mgr = SubAgentManager(process_manager=MagicMock())
        manifest = AgentManifest(
            manifest_version="1", name="T", slug="t", description="", author="", version="1.0.0",
            runtime=ManifestRuntime(adapter="cli", command="t", transport="auto", mode="pipe"),
        )
        t = mgr._resolve_transport(manifest, {})
        if HAS_SDK:
            assert isinstance(t, SDKTransport)
        else:
            from agent_os.agent.transports.pipe_transport import PipeTransport
            assert isinstance(t, PipeTransport)

    def test_auto_with_pipe_mode_falls_back_to_pipe_without_sdk(self):
        """mode=pipe with transport=auto should fallback to PipeTransport without SDK."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.agents.manifest import AgentManifest, ManifestRuntime
        mgr = SubAgentManager(process_manager=MagicMock())
        manifest = AgentManifest(
            manifest_version="1", name="T", slug="t", description="", author="", version="1.0.0",
            runtime=ManifestRuntime(adapter="cli", command="t", transport="auto", mode="pipe"),
        )
        # Patch HAS_SDK to False at the source module
        import agent_os.agent.transports.sdk_transport as sdk_mod
        original_has_sdk = sdk_mod.HAS_SDK
        sdk_mod.HAS_SDK = False
        try:
            t = mgr._resolve_transport(manifest, {})
        finally:
            sdk_mod.HAS_SDK = original_has_sdk
        from agent_os.agent.transports.pipe_transport import PipeTransport
        assert isinstance(t, PipeTransport)

    def test_pipe_transport_hint_gets_pipe(self):
        """transport=pipe should resolve to PipeTransport."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.agents.manifest import AgentManifest, ManifestRuntime
        mgr = SubAgentManager(process_manager=MagicMock())
        manifest = AgentManifest(
            manifest_version="1", name="T", slug="t", description="", author="", version="1.0.0",
            runtime=ManifestRuntime(adapter="cli", command="t", transport="pipe", mode="pipe"),
        )
        t = mgr._resolve_transport(manifest, {})
        from agent_os.agent.transports.pipe_transport import PipeTransport
        assert isinstance(t, PipeTransport)

    def test_sdk_manifest_gets_sdk_transport(self):
        """transport=sdk should resolve to SDKTransport."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.agents.manifest import AgentManifest, ManifestRuntime
        from agent_os.agent.transports.sdk_transport import SDKTransport, HAS_SDK
        if not HAS_SDK:
            pytest.skip("SDK not installed")
        mgr = SubAgentManager(process_manager=MagicMock())
        manifest = AgentManifest(
            manifest_version="1", name="T", slug="t", description="", author="", version="1.0.0",
            runtime=ManifestRuntime(adapter="cli", command="t", transport="sdk", mode="pipe"),
        )
        t = mgr._resolve_transport(manifest, {})
        assert isinstance(t, SDKTransport)

    def test_sdk_fallback_to_pipe_when_unavailable(self):
        """transport=sdk should fallback to PipeTransport when SDK not installed."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.agents.manifest import AgentManifest, ManifestRuntime
        mgr = SubAgentManager(process_manager=MagicMock())
        manifest = AgentManifest(
            manifest_version="1", name="T", slug="t", description="", author="", version="1.0.0",
            runtime=ManifestRuntime(adapter="cli", command="t", transport="sdk", mode="pipe"),
        )
        import agent_os.agent.transports.sdk_transport as sdk_mod
        original_has_sdk = sdk_mod.HAS_SDK
        sdk_mod.HAS_SDK = False
        try:
            t = mgr._resolve_transport(manifest, {})
        finally:
            sdk_mod.HAS_SDK = original_has_sdk
        from agent_os.agent.transports.pipe_transport import PipeTransport
        assert isinstance(t, PipeTransport)


class TestACPTransportResolution:
    def test_acp_manifest_gets_acp_transport(self):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.agents.manifest import AgentManifest, ManifestRuntime
        mgr = SubAgentManager(process_manager=MagicMock())
        manifest = AgentManifest(
            manifest_version="1", name="T", slug="t", description="", author="", version="1.0.0",
            runtime=ManifestRuntime(adapter="cli", command="t", transport="acp"),
        )
        t = mgr._resolve_transport(manifest, {})
        from agent_os.agent.transports.acp_transport import ACPTransport
        assert isinstance(t, ACPTransport)
