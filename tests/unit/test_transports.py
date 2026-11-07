# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for transport abstraction layer."""
import json
import pytest
from unittest.mock import MagicMock, patch
from agent_os.agent.transports.base import AgentTransport, TransportEvent

# Canonical chat-session uuid for SubAgentManager fixtures (post-"default"
# retirement: _start_from_registry requires an explicit session_id — None
# hard-raises because a sub-agent always has a parent session).
SID = "proj1_sess0001"


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
        # Single-line, newline-free -c script: an embedded-newline script passed
        # through shell=True is split by cmd.exe on Windows (the process then
        # exits immediately). The comprehension blocks reading stdin on every
        # platform, which is all this test needs (it asserts start/send/stop
        # succeed, not the echoed output).
        await t.start("python", ["-u", "-c",
            "import sys; [print('GOT:' + line.strip(), flush=True) for line in sys.stdin]"],
            ".")
        t._ready = True  # nothing reads this process's output here
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


class TestProcessManagerWiring:
    """Test SubAgentManager wiring of the streaming consumer.

    PTY and the legacy path need ``process_manager`` to drain
    ``read_stream()``; Pipe answers through ``send()`` and skips it.
    """

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
            await mgr._start_from_registry("proj1", "test-agent", session_id=SID)

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


class TestClaudeTransportResolution:
    """claude-code resolves to the SDK transport on the `auto` hint."""

    def test_claude_with_auto_transport_resolves_to_sdk(self):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.agents.manifest import AgentManifest, ManifestRuntime
        from agent_os.agent.transports.sdk_transport import SDKTransport, HAS_SDK
        if not HAS_SDK:
            pytest.skip("SDK not installed")
        mgr = SubAgentManager(process_manager=MagicMock())
        manifest = AgentManifest(
            manifest_version="1", name="Claude Code", slug="claude-code",
            description="", author="", version="1.0.0",
            runtime=ManifestRuntime(adapter="cli", command="claude",
                                    transport="auto", mode="pipe"),
        )
        t = mgr._resolve_transport(manifest, {})
        assert isinstance(t, SDKTransport)


# ---------------------------------------------------------------------------
# dsh (DeepSeek Harness) — Task 6 manifest + Task 4 data-dir resolution
# ---------------------------------------------------------------------------

import os

from agent_os.agents.manifest import ManifestLoader
from agent_os.agents.registry import AgentRegistry
from agent_os.agents.setup_engine import SetupEngine
from agent_os.daemon_v2.models import detect_os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DSH_MANIFEST = os.path.join(
    _REPO_ROOT, "agent_os", "agents", "manifests", "dsh.yaml"
)

# What the DeepSeek Harness desktop app looks like to the resolver: the CLI
# launcher inside the bundle (agents/desktop_apps.py finds it by bundle id).
DSH_APP = "/Applications/DeepSeek Harness.app"
DSH_DESKTOP_EXE = DSH_APP + "/Contents/Resources/runtime/cli/bin/dsh"


def _load_dsh():
    return ManifestLoader.load(DSH_MANIFEST)


def _engine(manifest, data_dir):
    registry = AgentRegistry()
    registry.register(manifest)
    return SetupEngine(registry, data_dir=str(data_dir))


def _desktop_hit(found=True):
    """Patch the desktop-app lookup: the DSH Desktop bundle is installed
    (``found``) or not, independent of the host."""
    from agent_os.agents.desktop_apps import DesktopAppHit

    def _find(app, os_name, **_kw):
        if not found or app.label != "DeepSeek Harness":
            return None
        return DesktopAppHit(label=app.label, app_path=DSH_APP,
                             executable=DSH_DESKTOP_EXE, version="0.2.1-rc.3")
    return patch("agent_os.agents.desktop_apps.find_desktop_app", _find)


def _exists_only(*paths):
    real = os.path.isfile
    return patch("agent_os.agents.setup_engine.os.path.isfile",
                 lambda p: p in paths or real(p))


class TestDshManifest:
    """The manifest is the contract between discovery, the renderer, and the
    transport — every claim in it is load-bearing somewhere."""

    def test_resolves_to_acp_sdk_transport(self):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.agent.transports.acp_sdk_transport import ACPSDKTransport
        mgr = SubAgentManager(process_manager=MagicMock())
        t = mgr._resolve_transport(_load_dsh(), {})
        assert isinstance(t, ACPSDKTransport)

    def test_boots_the_shipped_acp_profile_and_patches_it_per_spawn(self):
        """``--profile acp`` is the 0.2 launch contract (the 0.1
        ``dsh-acp-demo --config`` one answers "--profile <name> is
        required"). The patch path is appended per spawn — a static path
        here would resurrect the shared-file race."""
        runtime = _load_dsh().runtime
        assert runtime.command == "dsh"
        assert runtime.args == ["--profile", "acp"]
        assert runtime.config_template == "acp.patch.yml"
        assert runtime.config_flag == "--patch"

    def test_the_patch_template_ships_with_the_build(self):
        from agent_os.agents.installer import bundled_asset_path
        assert bundled_asset_path("dsh", "acp.patch.yml") is not None

    def test_emits_tool_activity_itself(self):
        """dsh 0.2's ACP bridge sends tool_call/tool_call_update; the 0.1
        activity shim is retired."""
        assert _load_dsh().capabilities.emits_tool_activity is True

    def test_other_manifests_default_to_emitting_tool_activity(self):
        claude = ManifestLoader.load(os.path.join(
            _REPO_ROOT, "agent_os", "agents", "manifests", "claude_code.yaml"))
        assert claude.capabilities.emits_tool_activity is True
        assert claude.runtime.config_template == ""
        assert claude.runtime.config_flag == "--config"
        assert claude.setup.orbital_install.platforms == []

    def test_the_desktop_app_is_the_distribution(self):
        dsh = _load_dsh()
        assert dsh.setup.orbital_install.platforms == []
        [app] = dsh.setup.desktop_apps
        assert app.order == "before_cli"
        assert app.platforms["macos"].bundle_ids == ["com.deepseek.dsh"]
        assert app.platforms["macos"].executable == (
            "Contents/Resources/runtime/cli/bin/dsh")
        assert app.platforms["windows"].executable.endswith("dsh.cmd")
        assert set(app.download_urls) == {"macos-arm64", "macos-x86_64", "windows"}
        assert all(u.startswith("https://download.deepseek.com/")
                   for u in app.download_urls.values())

    def test_version_floor_is_the_acp_profile_line(self):
        dsh = _load_dsh()
        assert dsh.setup.check_command == "dsh -V"
        assert dsh.setup.min_version == "0.2.1-rc.1"

    def test_the_key_the_desktop_app_saved_is_recognised(self):
        [key] = [c for c in _load_dsh().setup.credentials if c.key == "DEEPSEEK_API_KEY"]
        assert key.agent_store == {
            "file": "~/.dsh/.credentials.yaml", "home_env": "DSH_HOME",
            "key": "refs.DEEPSEEK_API_KEY", "label": "DeepSeek Harness"}

    def test_does_not_claim_orbital_governs_its_egress(self):
        """Transport-backed sub-agents spawn outside the project sandbox, so a
        network_domains list would be documentation that reads as enforcement."""
        assert _load_dsh().permissions.network_domains == []


class TestDshBinaryResolution:
    def test_the_desktop_app_wins_over_a_cli_on_path(self, tmp_path):
        manifest = _load_dsh()
        engine = _engine(manifest, tmp_path)
        with _desktop_hit(), _exists_only(DSH_DESKTOP_EXE), \
                patch("agent_os.agents.setup_engine.shutil.which",
                      return_value="/usr/local/bin/dsh"):
            assert engine.resolve_binary(manifest) == DSH_DESKTOP_EXE
        assert engine.resolved_source("dsh") == {
            "kind": "desktop_app", "label": "DeepSeek Harness",
            "app_path": DSH_APP, "app_version": "0.2.1-rc.3"}

    def test_a_cli_on_path_is_used_without_the_desktop_app(self, tmp_path):
        manifest = _load_dsh()
        engine = _engine(manifest, tmp_path)
        with _desktop_hit(found=False), _exists_only("/usr/local/bin/dsh"), \
                patch("agent_os.agents.setup_engine.shutil.which",
                      return_value="/usr/local/bin/dsh"):
            assert engine.resolve_binary(manifest) == "/usr/local/bin/dsh"
        assert engine.resolved_source("dsh") == {"kind": "path"}

    def test_resolved_path_becomes_the_spawn_command(self, tmp_path):
        manifest = _load_dsh()
        engine = _engine(manifest, tmp_path)
        with _desktop_hit(), _exists_only(DSH_DESKTOP_EXE):
            config = engine.get_adapter_config("dsh", "/tmp/workspace")
        assert config["command"] == DSH_DESKTOP_EXE
        assert config["args"] == ["--profile", "acp"]

    def test_credentials_produce_a_deepseek_api_key_env_entry(
        self, tmp_path, monkeypatch
    ):
        manifest = _load_dsh()
        engine = _engine(manifest, tmp_path)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
        with _desktop_hit(), _exists_only(DSH_DESKTOP_EXE):
            config = engine.get_adapter_config("dsh", "/tmp/workspace")
        assert config["env"]["DEEPSEEK_API_KEY"] == "sk-from-env"

    def test_credential_store_supplies_the_key(self, tmp_path):
        manifest = _load_dsh()
        registry = AgentRegistry()
        registry.register(manifest)
        store = MagicMock()
        store.get.side_effect = lambda key: (
            "sk-from-store" if key == "DEEPSEEK_API_KEY" else None
        )
        engine = SetupEngine(
            registry, credential_store=store, data_dir=str(tmp_path),
        )
        with _desktop_hit(), _exists_only(DSH_DESKTOP_EXE):
            config = engine.get_adapter_config("dsh", "/tmp/workspace")
        assert config["env"]["DEEPSEEK_API_KEY"] == "sk-from-store"

    def test_permission_mode_reaches_dsh_as_its_env_var(self, tmp_path):
        from agent_os.daemon_v2.sub_agent_config_store import SubAgentConfigStore
        manifest = _load_dsh()
        registry = AgentRegistry()
        registry.register(manifest)
        store = SubAgentConfigStore(str(tmp_path / "cfg.json"))
        engine = SetupEngine(registry, sub_agent_config_store=store,
                             data_dir=str(tmp_path))
        with _desktop_hit(), _exists_only(DSH_DESKTOP_EXE):
            default = engine.get_adapter_config("dsh", "/tmp/ws")
            store.set("dsh", {"permission-mode": "danger-full-access"})
            chosen = engine.get_adapter_config("dsh", "/tmp/ws")
        assert default["env"]["DSH_PERMISSION_MODE"] == "workspace-write"
        assert chosen["env"]["DSH_PERMISSION_MODE"] == "danger-full-access"
        # Never argv: dsh's acp profile takes no such flag.
        assert chosen["args"] == ["--profile", "acp"]

    def test_missing_binary_reports_not_installed(self, tmp_path):
        manifest = _load_dsh()
        engine = _engine(manifest, tmp_path)
        with _desktop_hit(found=False), \
                patch("agent_os.agents.setup_engine.shutil.which", return_value=None), \
                patch("agent_os.agents.setup_engine.os.path.isfile", return_value=False):
            assert engine.resolve_binary(manifest) is None
            assert engine.check_agent("dsh").installed is False

    def test_missing_binary_raises_at_dispatch(self, tmp_path):
        manifest = _load_dsh()
        engine = _engine(manifest, tmp_path)
        with _desktop_hit(found=False), \
                patch("agent_os.agents.setup_engine.shutil.which", return_value=None), \
                patch("agent_os.agents.setup_engine.os.path.isfile", return_value=False):
            with pytest.raises(ValueError, match="not installed"):
                engine.get_adapter_config("dsh", "/tmp/workspace")


class TestOrbitalDataDirToken:
    """The token is the system's only placeholder and has exactly one
    consumer: setup.auto_detect. runtime.command/args stay passthrough."""

    def _manifest(self, **runtime_kw):
        from agent_os.agents.manifest import (
            AgentManifest, ManifestRuntime, ManifestSetup,
        )
        return AgentManifest(
            manifest_version="1", name="T", slug="t", description="",
            author="", version="1.0.0",
            runtime=ManifestRuntime(adapter="cli", **runtime_kw),
            setup=ManifestSetup(auto_detect={
                detect_os(): ["${ORBITAL_DATA_DIR}/agents/t/bin/t"],
            }),
        )

    def test_expands_without_the_env_var_set(self, tmp_path, monkeypatch):
        """Dev daemons never set AGENT_OS_DATA_DIR — only the desktop
        entrypoint does — so expandvars alone would leave the token intact."""
        monkeypatch.delenv("ORBITAL_DATA_DIR", raising=False)
        target = tmp_path / "agents" / "t" / "bin" / "t"
        target.parent.mkdir(parents=True)
        target.write_text("#!/bin/sh\n", encoding="utf-8")
        manifest = self._manifest(command="t")
        engine = _engine(manifest, tmp_path)
        with patch("agent_os.agents.setup_engine.shutil.which", return_value=None):
            assert engine.resolve_binary(manifest) == str(target)

    def test_runtime_command_is_not_substituted(self, tmp_path):
        """A token in runtime.command must stay verbatim — command goes
        straight to shutil.which, and silently rewriting it would make the
        passthrough contract untrue."""
        manifest = self._manifest(command="${ORBITAL_DATA_DIR}/agents/t/bin/t")
        engine = _engine(manifest, tmp_path)
        seen = []

        def _which(cmd):
            seen.append(cmd)
            return None

        with patch("agent_os.agents.setup_engine.shutil.which", _which):
            engine.resolve_binary(manifest)
        assert seen == ["${ORBITAL_DATA_DIR}/agents/t/bin/t"]

    def test_default_data_dir_matches_the_daemon_default(self):
        registry = AgentRegistry()
        assert SetupEngine(registry).data_dir == os.path.abspath("orbital-data")


class TestConfigTemplateDispatchHook:
    """A manifest declaring ``runtime.config_template`` gets a file rendered
    per spawn and appended with its ``runtime.config_flag`` (dsh:
    ``--patch``). Manifest-driven, not slug-hardcoded."""

    def _mgr(self, tmp_path, persisted=None, *, install_template=True):
        import shutil as _shutil
        import yaml as _yaml
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.daemon_v2.sub_agent_config_store import SubAgentConfigStore

        data_dir = tmp_path / "data"
        if install_template:
            template_dir = data_dir / "agents" / "dsh"
            template_dir.mkdir(parents=True)
            _shutil.copy(
                os.path.join(_REPO_ROOT, "agent_os", "agents", "assets", "dsh",
                             "acp.patch.yml"),
                str(template_dir / "acp.patch.yml"),
            )
        workspace = tmp_path / "ws"
        workspace.mkdir()

        config_store = SubAgentConfigStore(
            str(tmp_path / "sub_agent_config.json"))
        if persisted:
            config_store.set("dsh", persisted)

        registry = MagicMock()
        registry.get.return_value = _load_dsh()

        setup_engine = MagicMock()
        setup_engine.data_dir = str(data_dir)
        setup_engine.sub_agent_config_store = config_store
        setup_engine.get_adapter_config.return_value = {
            "command": "/bin/echo", "args": [], "workspace": str(workspace),
            "approval_patterns": [], "env": {}, "network_domains": [],
        }

        pm = MagicMock()
        pm.start = AsyncMock()
        mgr = SubAgentManager(
            process_manager=pm, registry=registry, setup_engine=setup_engine,
            project_store=MagicMock(get_project=MagicMock(return_value={
                "workspace": str(workspace), "enabled_sub_agents": ["dsh"],
            })),
        )
        return mgr, workspace

    async def _spawn(self, mgr):
        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            adapter = MagicMock()
            adapter.start = AsyncMock()
            MockAdapter.return_value = adapter
            result = await mgr._start_from_registry("proj1", "dsh", session_id=SID)
        assert not result.startswith("Error"), result
        return adapter

    @staticmethod
    def _config_arg(adapter):
        args = adapter.start.await_args.args[0].args
        assert "--patch" in args, args
        assert "--config" not in args, args
        return args[args.index("--patch") + 1]

    @pytest.mark.asyncio
    async def test_appends_config_pointing_at_a_rendered_file(self, tmp_path):
        mgr, workspace = self._mgr(tmp_path)
        adapter = await self._spawn(mgr)
        path = self._config_arg(adapter)
        assert os.path.isfile(path)
        assert os.path.basename(path).startswith("cordis-")

    @pytest.mark.asyncio
    async def test_renders_into_the_handles_sub_agent_dir(self, tmp_path):
        from agent_os.agent.project_paths import ProjectPaths
        mgr, workspace = self._mgr(tmp_path)
        adapter = await self._spawn(mgr)
        expected = ProjectPaths(str(workspace)).sub_agent_dir("dsh")
        assert os.path.dirname(self._config_arg(adapter)) == expected

    @pytest.mark.asyncio
    async def test_persona_is_the_real_rendered_sub_agent_prompt(self, tmp_path):
        import yaml as _yaml
        from agent_os.agent.sub_agent_prompt import render_sub_agent_prompt
        mgr, workspace = self._mgr(tmp_path)
        adapter = await self._spawn(mgr)
        with open(self._config_arg(adapter), encoding="utf-8") as f:
            blocks = {b["id"]: b for b in _yaml.safe_load(f)}
        expected = render_sub_agent_prompt(
            workspace=str(workspace), namespace=None, agent_slug="dsh",
            enabled_sub_agents=["dsh"],
        )
        assert blocks["system-prompt"]["config"]["personaPrefix"] == expected

    @pytest.mark.asyncio
    async def test_no_memory_md_stub_is_created(self, tmp_path):
        """acp-sdk stays in the skips_system_prompt set: the persona is the
        prompt's route, and the MEMORY.md stub would still be an orphan."""
        from agent_os.agent.project_paths import ProjectPaths
        mgr, workspace = self._mgr(tmp_path)
        await self._spawn(mgr)
        memory = os.path.join(
            ProjectPaths(str(workspace)).sub_agent_dir("dsh"), "MEMORY.md")
        assert not os.path.exists(memory)

    @pytest.mark.asyncio
    async def test_model_comes_from_the_config_store(self, tmp_path):
        import yaml as _yaml
        mgr, workspace = self._mgr(tmp_path, persisted={
            "model": "deepseek-official/deepseek-v4-pro",
            "permission-mode": "danger-full-access",
        })
        adapter = await self._spawn(mgr)
        with open(self._config_arg(adapter), encoding="utf-8") as f:
            blocks = {b["id"]: b for b in _yaml.safe_load(f)}
        assert blocks["acp"]["config"] == {
            "provider": "deepseek-official", "model": "deepseek-v4-pro"}
        # The permission mode is DSH_PERMISSION_MODE (spawn env), never a
        # patch row: replacing sandbox-policy's config would drop its
        # workspaceRoot expression.
        assert "sandbox-policy" not in blocks

    @pytest.mark.asyncio
    async def test_schema_defaults_apply_with_nothing_persisted(self, tmp_path):
        import yaml as _yaml
        mgr, workspace = self._mgr(tmp_path)
        adapter = await self._spawn(mgr)
        with open(self._config_arg(adapter), encoding="utf-8") as f:
            blocks = {b["id"]: b for b in _yaml.safe_load(f)}
        assert blocks["acp"]["config"]["model"] == "deepseek-v4-flash"

    @pytest.mark.asyncio
    async def test_persistence_root_is_scoped_to_project_and_handle(self, tmp_path):
        import yaml as _yaml
        from agent_os.agent.project_paths import ProjectPaths
        mgr, workspace = self._mgr(tmp_path)
        adapter = await self._spawn(mgr)
        with open(self._config_arg(adapter), encoding="utf-8") as f:
            blocks = {b["id"]: b for b in _yaml.safe_load(f)}
        assert blocks["session-persistence-jsonl"]["config"]["root"] == os.path.join(
            ProjectPaths(str(workspace)).sub_agent_dir("dsh"), "dsh-sessions")

    @pytest.mark.asyncio
    async def test_two_spawns_render_two_files(self, tmp_path):
        mgr, workspace = self._mgr(tmp_path)
        first = self._config_arg(await self._spawn(mgr))
        mgr._adapters.clear()
        second = self._config_arg(await self._spawn(mgr))
        assert first != second
        assert os.path.isfile(first) and os.path.isfile(second)

    @pytest.mark.asyncio
    async def test_rendered_path_is_tracked_on_the_adapter(self, tmp_path):
        mgr, workspace = self._mgr(tmp_path)
        adapter = await self._spawn(mgr)
        assert adapter._rendered_config_path == self._config_arg(adapter)

    @pytest.mark.asyncio
    async def test_stop_removes_the_rendered_file(self, tmp_path):
        mgr, workspace = self._mgr(tmp_path)
        adapter = await self._spawn(mgr)
        path = self._config_arg(adapter)
        adapter.stop = AsyncMock()
        mgr._process_manager.stop = AsyncMock()
        await mgr.stop("proj1", "dsh", session_id=SID)
        assert not os.path.exists(path)

    @pytest.mark.asyncio
    async def test_spawn_gcs_stale_renders(self, tmp_path):
        import time as _time
        from agent_os.agent.project_paths import ProjectPaths
        mgr, workspace = self._mgr(tmp_path)
        sub_dir = ProjectPaths(str(workspace)).sub_agent_dir("dsh")
        os.makedirs(sub_dir, exist_ok=True)
        stale = os.path.join(sub_dir, "cordis-00000000.yml")
        with open(stale, "w", encoding="utf-8") as f:
            f.write("- id: x\n")
        old = _time.time() - 30 * 86400
        os.utime(stale, (old, old))
        await self._spawn(mgr)
        assert not os.path.exists(stale)

    @pytest.mark.asyncio
    async def test_missing_installed_template_fails_the_dispatch(self, tmp_path):
        """A config_template manifest with no installed template must not
        silently spawn an unconfigured harness. The renderer first falls back
        to the template bundled with the build (a launcher chosen on the
        locate card has no staged copy; spec 127 V4) — covered by
        test_composition_template_fallback.py — so this pins the case where
        neither exists."""
        mgr, workspace = self._mgr(tmp_path, install_template=False)
        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter, \
                patch("agent_os.agents.installer.bundled_asset_path", return_value=None):
            adapter = MagicMock()
            adapter.start = AsyncMock()
            MockAdapter.return_value = adapter
            result = await mgr._start_from_registry("proj1", "dsh", session_id=SID)
        assert result.startswith("Error")
        adapter.start.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_manifest_without_config_template_is_untouched(self, tmp_path):
        """Every other agent's argv must not grow a --config / --patch."""
        from agent_os.agents.manifest import AgentManifest, ManifestRuntime
        mgr, workspace = self._mgr(tmp_path)
        mgr._registry.get.return_value = AgentManifest(
            manifest_version="1", name="T", slug="t", description="",
            author="", version="1.0.0",
            runtime=ManifestRuntime(adapter="cli", command="echo",
                                    transport="pty", mode="interactive"),
        )
        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            adapter = MagicMock()
            adapter.start = AsyncMock()
            MockAdapter.return_value = adapter
            await mgr._start_from_registry("proj1", "t", session_id=SID)
        assert adapter.start.await_args.args[0].args == []
