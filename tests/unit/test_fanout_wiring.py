# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for Task 6: fanout startup wiring + hardening (spec 009).

- App factory constructs a ``FanoutRegistry`` and wires it into both
  ``LifecycleObserver`` and ``SubAgentManager`` (including the worker-deps
  factory built by Task 3, ``AgentManager.build_worker_deps``).
- ``configure_network`` guard: native ``worker:`` handles must never trigger
  the CLI sub-agent start path's per-start network-allowlist rewrite.
- ``ProjectPaths.sub_agent_dir`` sanitizes ":" out of the handle for
  filesystem use (NTFS ADS hazard) while leaving CLI slugs (no colons)
  byte-identical, and the write/read transcript paths stay in parity.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.project_paths import ProjectPaths
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

SID = "proj_sess0001"


# ---------------------------------------------------------------------------
# 1. App factory wiring
# ---------------------------------------------------------------------------


class TestAppFactoryFanoutWiring:
    """create_app() must construct one FanoutRegistry and wire it into both
    LifecycleObserver and SubAgentManager, plus the worker-deps factory."""

    def test_fanout_registry_wired(self, monkeypatch, tmp_path):
        # Headless keychain hang hazard (CLAUDE.md): create_app() touches
        # credential storage during construction.
        monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "in-memory")
        monkeypatch.setenv("AGENT_OS_API_KEY", "test")

        from agent_os.api.app import create_app
        from agent_os.api.routes import agents_v2
        from agent_os.daemon_v2.fanout import FanoutRegistry

        create_app(data_dir=str(tmp_path))

        lifecycle_observer = agents_v2._lifecycle_observer
        sub_agent_manager = agents_v2._sub_agent_manager
        agent_manager = agents_v2._agent_manager
        ws_manager = agents_v2._ws_manager

        assert lifecycle_observer is not None
        assert sub_agent_manager is not None

        # Exactly one registry, shared by both wiring points.
        assert lifecycle_observer.fanout_registry is not None
        assert isinstance(lifecycle_observer.fanout_registry, FanoutRegistry)
        assert sub_agent_manager._fanout_registry is lifecycle_observer.fanout_registry

        # Worker-deps factory (Task 3) is wired — dispatch_fanout can run.
        assert sub_agent_manager._worker_deps_factory is not None
        assert sub_agent_manager._worker_deps_factory.__func__.__name__ == "build_worker_deps"
        assert sub_agent_manager._worker_deps_factory.__self__ is agent_manager

        # Callable shapes match Task 2's documented wiring contract.
        registry = lifecycle_observer.fanout_registry
        assert registry._inject.__func__.__name__ == "inject_system_message"
        assert registry._inject.__self__ is agent_manager
        assert registry._broadcast == ws_manager.broadcast
        assert registry._stop_worker.__func__.__name__ == "stop"
        assert registry._stop_worker.__self__ is sub_agent_manager


# ---------------------------------------------------------------------------
# 2. configure_network guard for native worker handles
# ---------------------------------------------------------------------------


def _make_manager(**kwargs):
    pm = MagicMock()
    pm.start = AsyncMock()
    pm.stop = AsyncMock()
    return SubAgentManager(process_manager=pm, **kwargs)


def _make_mock_adapter():
    adapter = AsyncMock()
    adapter.is_alive = MagicMock(return_value=True)
    adapter.is_idle = MagicMock(return_value=False)
    adapter.stop = AsyncMock()
    adapter.start = AsyncMock()
    return adapter


class TestConfigureNetworkWorkerGuard:
    """Native worker handles (``worker:<fanout_id>-<i>``) must never trigger
    the legacy/registry CLI start paths' per-start configure_network rewrite.

    dispatch_fanout constructs NativeWorkerAdapter directly and never calls
    start()/_start_from_registry, so this is belt-and-braces: verified here
    directly against the legacy adapter_configs path, which is the one place
    a handle string reaches configure_network."""

    @pytest.mark.asyncio
    async def test_worker_handle_skips_configure_network(self):
        mgr = _make_manager()
        platform_provider = MagicMock()
        platform_provider.configure_network = MagicMock()
        mgr._platform_provider = platform_provider

        config = MagicMock()
        config.workspace = "/tmp"
        config.env = {}
        config.approval_patterns = []
        config.args = None
        config.command = "echo"
        mgr._adapter_configs["worker:abc123-0"] = config

        adapter = _make_mock_adapter()
        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter", return_value=adapter):
            await mgr.start("proj", "worker:abc123-0", session_id=SID)

        platform_provider.configure_network.assert_not_called()

    @pytest.mark.asyncio
    async def test_normal_handle_still_configures_network(self):
        """Contrast case: a genuine CLI handle (no ``worker:`` prefix) must
        still hit configure_network — the guard is handle-specific, not a
        global regression."""
        mgr = _make_manager()
        platform_provider = MagicMock()
        platform_provider.configure_network = MagicMock()
        mgr._platform_provider = platform_provider

        config = MagicMock()
        config.workspace = "/tmp"
        config.env = {}
        config.approval_patterns = []
        config.args = None
        config.command = "echo"
        mgr._adapter_configs["claude-code"] = config

        adapter = _make_mock_adapter()
        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter", return_value=adapter):
            await mgr.start("proj", "claude-code", session_id=SID)

        platform_provider.configure_network.assert_called_once()


# ---------------------------------------------------------------------------
# 3. ProjectPaths.sub_agent_dir sanitizes ":" (Windows ADS hazard)
# ---------------------------------------------------------------------------


class TestSubAgentDirSanitization:
    def test_worker_handle_colon_stripped_from_final_path_component(self, tmp_path):
        pp = ProjectPaths(str(tmp_path))
        path = pp.sub_agent_dir("worker:abc123-0")
        final_component = os.path.basename(os.path.normpath(path))
        assert ":" not in final_component
        # The whole path must not contain a colon past the drive letter (if
        # any) — check the sub_agents-relative tail specifically.
        tail = os.path.relpath(path, pp.sub_agents_dir)
        assert ":" not in tail

    def test_cli_slug_path_unchanged(self, tmp_path):
        """No-op for existing CLI slugs (no colons) — locks parity with the
        pre-existing test_project_paths.py::test_sub_agent_dir assertion."""
        pp = ProjectPaths(str(tmp_path))
        expected = os.path.normpath(
            os.path.join(str(tmp_path), "orbital", "sub_agents", "claude-code")
        )
        assert os.path.normpath(pp.sub_agent_dir("claude-code")) == expected

    def test_write_then_read_parity_for_worker_handle(self, tmp_path):
        """A transcript written via SubAgentTranscript.open_for_handle with a
        ``worker:`` handle must be found by SubAgentManager.read_transcript_entries's
        disk fallback — both go through the same ProjectPaths.sub_agent_dir
        join point, so sanitizing it once keeps read/write in parity."""
        from agent_os.daemon_v2.sub_agent_transcript import SubAgentTranscript

        workspace = str(tmp_path)
        handle = "worker:def456-1"

        transcript = SubAgentTranscript.open_for_handle(workspace, handle, fresh=True)
        transcript.append({"content": "hello", "chunk_type": "response", "source": "worker"})

        project_store = MagicMock()
        project_store.get_project = MagicMock(
            return_value={"workspace": workspace}
        )
        pm = MagicMock()
        mgr = SubAgentManager(process_manager=pm, project_store=project_store)

        entries = mgr.read_transcript_entries("proj", handle, session_id=None)
        assert entries is not None
        assert any(e.get("content") == "hello" for e in entries)
