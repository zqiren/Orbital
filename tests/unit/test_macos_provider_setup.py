# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for MacOSPlatformProvider setup logic — all mocked, no macOS needed."""

import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.platform.macos.provider import MacOSPlatformProvider
from agent_os.platform.types import PlatformCapabilities, SetupResult


class TestMacOSProviderSetup:
    """Tests for setup() and is_setup_complete()."""

    @pytest.mark.asyncio
    @patch("agent_os.platform.macos.provider.shutil.which", return_value="/usr/bin/sandbox-exec")
    async def test_setup_success_when_sandbox_exec_exists(self, mock_which):
        """Mock shutil.which("sandbox-exec") returns path -> SetupResult(success=True)."""
        provider = MacOSPlatformProvider()
        result = await provider.setup()
        assert isinstance(result, SetupResult)
        assert result.success is True

    @pytest.mark.asyncio
    @patch("agent_os.platform.macos.provider.shutil.which", return_value=None)
    async def test_setup_fails_when_sandbox_exec_missing(self, mock_which):
        """Mock shutil.which -> None -> SetupResult(success=False)."""
        provider = MacOSPlatformProvider()
        result = await provider.setup()
        assert isinstance(result, SetupResult)
        assert result.success is False

    @pytest.mark.asyncio
    @patch("agent_os.platform.macos.provider.shutil.which", return_value="/usr/bin/sandbox-exec")
    async def test_capabilities_platform_macos(self, mock_which):
        """get_capabilities() returns platform='macos', isolation_method='seatbelt'."""
        provider = MacOSPlatformProvider()
        caps = provider.get_capabilities()
        assert isinstance(caps, PlatformCapabilities)
        assert caps.platform == "macos"
        assert caps.isolation_method == "seatbelt"

    @pytest.mark.asyncio
    @patch("agent_os.platform.macos.provider.shutil.which", return_value=None)
    async def test_capabilities_setup_incomplete(self, mock_which):
        """When sandbox-exec missing -> setup_complete=False, setup_issues non-empty."""
        provider = MacOSPlatformProvider()
        caps = provider.get_capabilities()
        assert caps.setup_complete is False
        assert len(caps.setup_issues) > 0
        assert any("sandbox-exec" in issue for issue in caps.setup_issues)

    @pytest.mark.asyncio
    async def test_teardown_stops_all_proxies_and_processes(self):
        """Mock proxies/processes -> teardown calls stop."""
        provider = MacOSPlatformProvider()

        # Mock proxy
        mock_proxy = AsyncMock()
        provider._proxies["proj1"] = mock_proxy

        # Mock process
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()

        mock_handle = MagicMock()
        mock_handle.stdin = None
        mock_handle.stdout = None
        mock_handle.stderr = None

        provider._processes["proj1"] = (mock_handle, mock_proc)

        result = await provider.teardown()
        assert result.success is True
        mock_proxy.stop.assert_awaited_once()
        mock_proc.terminate.assert_called_once()
        assert len(provider._proxies) == 0
        assert len(provider._processes) == 0

    @patch("agent_os.platform.macos.provider.subprocess.Popen")
    def test_prevent_sleep_launches_caffeinate(self, mock_popen):
        """Mock subprocess.Popen -> verify called with ['caffeinate', '-di']."""
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc

        provider = MacOSPlatformProvider()
        handle = provider.prevent_sleep("running agent task")

        mock_popen.assert_called_once_with(["caffeinate", "-di"])
        assert handle is mock_proc

    @patch("agent_os.platform.macos.provider.subprocess.Popen")
    def test_allow_sleep_terminates_caffeinate(self, mock_popen):
        """Mock process -> verify .terminate() called."""
        mock_proc = MagicMock()
        mock_proc.terminate = MagicMock()
        mock_popen.return_value = mock_proc

        provider = MacOSPlatformProvider()
        handle = provider.prevent_sleep("running agent task")
        provider.allow_sleep(handle)

        mock_proc.terminate.assert_called_once()
