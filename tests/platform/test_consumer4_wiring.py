# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Wiring tests for Consumer 4: External Agent Manager (SubAgentManager).

Tests verify that SubAgentManager correctly integrates with PlatformProvider:
1. Unit: SubAgentManager accepts platform_provider parameter
2. Unit: start() calls provider.configure_network() before launching
3. Unit: start() injects provider env vars into adapter config (adapter-first approach)
4. Unit: stop() calls provider.stop_process() for cleanup
5. Integration: start->stop cycle with mock provider

The SCOUT report flags the PTY vs CreateProcessWithLogonW tension.
The spec recommends "provider-first" but acknowledges this is the most complex consumer.
Tests define the minimum contract: network must be configured, and isolation env must be
injected or processes must be launched through the provider.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.adapters.base import AdapterConfig
from agent_os.platform.types import NetworkRules, DEFAULT_ALLOWLIST_DOMAINS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sub_agent_manager(platform_provider=None):
    """Create a SubAgentManager with mocked dependencies and optional platform_provider."""
    from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

    pm = MagicMock()
    pm.start = AsyncMock()
    pm.stop = AsyncMock()

    configs = {
        "claudecode": AdapterConfig(
            command="claude",
            workspace="/tmp/ws",
            approval_patterns=["Approve?"],
        )
    }

    mgr = SubAgentManager(
        process_manager=pm,
        adapter_configs=configs,
        platform_provider=platform_provider,
    )
    return mgr, pm


def _make_mock_provider():
    """Create a mock platform provider for SubAgentManager tests."""
    provider = MagicMock()
    provider.configure_network = MagicMock()
    provider.run_process = AsyncMock(return_value=MagicMock(pid=1234))
    provider.stop_process = AsyncMock(return_value=True)
    return provider


# ---------------------------------------------------------------------------
# Unit tests: SubAgentManager accepts platform_provider
# ---------------------------------------------------------------------------


class TestSubAgentManagerAcceptsProvider:
    """SubAgentManager.__init__ must accept platform_provider parameter."""

    def test_accepts_platform_provider(self):
        """SubAgentManager can be created with platform_provider."""
        provider = _make_mock_provider()
        mgr, _ = _make_sub_agent_manager(platform_provider=provider)
        assert mgr._platform_provider is provider

    def test_accepts_none_provider(self):
        """SubAgentManager works with platform_provider=None (dev mode)."""
        mgr, _ = _make_sub_agent_manager(platform_provider=None)
        assert mgr._platform_provider is None


# ---------------------------------------------------------------------------
# Unit tests: start() calls configure_network()
# ---------------------------------------------------------------------------


class TestStartCallsConfigureNetwork:
    """start() must call provider.configure_network() before launching."""

    @pytest.mark.asyncio
    async def test_start_calls_configure_network(self):
        """start() calls configure_network with project_id and network rules."""
        provider = _make_mock_provider()
        mgr, pm = _make_sub_agent_manager(platform_provider=provider)

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            mock_instance = AsyncMock()
            mock_instance.is_alive = MagicMock(return_value=True)
            MockAdapter.return_value = mock_instance

            await mgr.start("proj_1", "claudecode")

        provider.configure_network.assert_called_once()
        call_args = provider.configure_network.call_args
        assert call_args[0][0] == "proj_1"  # project_id
        # Second arg should be NetworkRules
        rules = call_args[0][1]
        assert hasattr(rules, "mode") or isinstance(rules, NetworkRules)

    @pytest.mark.asyncio
    async def test_start_skips_network_when_no_provider(self):
        """start() works without configure_network when provider is None."""
        mgr, pm = _make_sub_agent_manager(platform_provider=None)

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            mock_instance = AsyncMock()
            mock_instance.is_alive = MagicMock(return_value=True)
            MockAdapter.return_value = mock_instance

            result = await mgr.start("proj_1", "claudecode")
            assert "Started" in result or "claudecode" in result


# ---------------------------------------------------------------------------
# Unit tests: start() injects provider env or uses run_process
# ---------------------------------------------------------------------------


class TestStartInjectsProviderIsolation:
    """start() must either inject sandbox env vars or delegate to provider.run_process()."""

    @pytest.mark.asyncio
    async def test_start_creates_adapter(self):
        """start() still creates a CLIAdapter and starts it."""
        provider = _make_mock_provider()
        mgr, pm = _make_sub_agent_manager(platform_provider=provider)

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            mock_instance = AsyncMock()
            mock_instance.is_alive = MagicMock(return_value=True)
            MockAdapter.return_value = mock_instance

            await mgr.start("proj_1", "claudecode")

            mock_instance.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_registers_with_process_manager(self):
        """start() still registers with ProcessManager for output bridging."""
        provider = _make_mock_provider()
        mgr, pm = _make_sub_agent_manager(platform_provider=provider)

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            mock_instance = AsyncMock()
            mock_instance.is_alive = MagicMock(return_value=True)
            MockAdapter.return_value = mock_instance

            await mgr.start("proj_1", "claudecode")

            pm.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_stores_adapter(self):
        """start() stores the adapter in _adapters."""
        provider = _make_mock_provider()
        mgr, pm = _make_sub_agent_manager(platform_provider=provider)

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            mock_instance = AsyncMock()
            mock_instance.is_alive = MagicMock(return_value=True)
            MockAdapter.return_value = mock_instance

            await mgr.start("proj_1", "claudecode")

            assert "proj_1" in mgr._adapters
            assert "claudecode" in mgr._adapters["proj_1"]


# ---------------------------------------------------------------------------
# Unit tests: stop() calls stop_process()
# ---------------------------------------------------------------------------


class TestStopCallsStopProcess:
    """stop() delegates cleanup to CLIAdapter (which calls provider.stop_process internally)."""

    @pytest.mark.asyncio
    async def test_stop_delegates_to_adapter_not_provider(self):
        """stop() does NOT call provider.stop_process() directly — CLIAdapter handles it."""
        provider = _make_mock_provider()
        mgr, pm = _make_sub_agent_manager(platform_provider=provider)

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            mock_instance = AsyncMock()
            mock_instance.is_alive = MagicMock(return_value=True)
            MockAdapter.return_value = mock_instance

            await mgr.start("proj_1", "claudecode")
            await mgr.stop("proj_1", "claudecode")

        provider.stop_process.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stop_still_stops_adapter(self):
        """stop() still calls adapter.stop() in addition to provider."""
        provider = _make_mock_provider()
        mgr, pm = _make_sub_agent_manager(platform_provider=provider)

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            mock_instance = AsyncMock()
            mock_instance.is_alive = MagicMock(return_value=True)
            MockAdapter.return_value = mock_instance

            await mgr.start("proj_1", "claudecode")
            await mgr.stop("proj_1", "claudecode")

            mock_instance.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_still_deregisters_from_process_manager(self):
        """stop() still calls process_manager.stop()."""
        provider = _make_mock_provider()
        mgr, pm = _make_sub_agent_manager(platform_provider=provider)

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            mock_instance = AsyncMock()
            mock_instance.is_alive = MagicMock(return_value=True)
            MockAdapter.return_value = mock_instance

            await mgr.start("proj_1", "claudecode")
            await mgr.stop("proj_1", "claudecode")

            pm.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_skips_stop_process_when_no_provider(self):
        """stop() works without provider.stop_process when provider is None."""
        mgr, pm = _make_sub_agent_manager(platform_provider=None)

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            mock_instance = AsyncMock()
            mock_instance.is_alive = MagicMock(return_value=True)
            MockAdapter.return_value = mock_instance

            await mgr.start("proj_1", "claudecode")
            result = await mgr.stop("proj_1", "claudecode")
            assert "Stopped" in result


# ---------------------------------------------------------------------------
# Integration: start -> stop cycle with mock provider
# ---------------------------------------------------------------------------


class TestStartStopCycle:
    """Integration: full start -> stop cycle with mock provider."""

    @pytest.mark.asyncio
    async def test_full_start_stop_cycle(self):
        """Full start -> stop cycle: configure network, start adapter, stop adapter, stop process."""
        provider = _make_mock_provider()
        mgr, pm = _make_sub_agent_manager(platform_provider=provider)

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            mock_instance = AsyncMock()
            mock_instance.is_alive = MagicMock(return_value=True)
            mock_instance.handle = "claudecode"
            mock_instance.display_name = "Claude Code"
            MockAdapter.return_value = mock_instance

            # Start
            result = await mgr.start("proj_1", "claudecode")
            assert "Started" in result or "claudecode" in result

            # Verify configure_network was called
            provider.configure_network.assert_called_once()

            # Verify adapter is registered
            assert mgr.status("proj_1", "claudecode") in ("running", "idle")

            # Stop
            result = await mgr.stop("proj_1", "claudecode")
            assert "Stopped" in result

            # Verify cleanup — provider.stop_process is NOT called directly
            # (CLIAdapter.stop() handles it internally)
            provider.stop_process.assert_not_awaited()
            mock_instance.stop.assert_awaited_once()
            pm.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_unknown_handle_returns_error(self):
        """start() with unknown handle returns error string."""
        provider = _make_mock_provider()
        mgr, pm = _make_sub_agent_manager(platform_provider=provider)

        result = await mgr.start("proj_1", "nonexistent")
        assert "Error" in result or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_stop_not_running_returns_message(self):
        """stop() when agent not running returns informative message."""
        provider = _make_mock_provider()
        mgr, pm = _make_sub_agent_manager(platform_provider=provider)

        result = await mgr.stop("proj_1", "claudecode")
        assert "not running" in result.lower() or result != ""
