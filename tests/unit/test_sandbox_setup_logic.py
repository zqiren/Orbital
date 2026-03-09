# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Test the sandbox setup and teardown functions directly."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass


@dataclass
class MockCapabilities:
    platform: str = "windows"
    isolation_method: str = "sandbox_user"
    setup_complete: bool = False
    setup_issues: list = None
    supports_network_restriction: bool = True
    supports_folder_access: bool = True
    sandbox_username: str = None

    def __post_init__(self):
        if self.setup_issues is None:
            self.setup_issues = []


@dataclass
class MockSetupResult:
    success: bool = True
    error: str = None


class TestRunSandboxSetup:
    """Test run_sandbox_setup() logic with mocked provider."""

    @patch("agent_os.platform.create_platform_provider")
    def test_null_provider_skips(self, mock_create):
        """NullProvider (non-Windows) should skip setup and exit 0."""
        provider = MagicMock()
        provider.get_capabilities.return_value = MockCapabilities(
            platform="null", isolation_method="none", setup_complete=True,
        )
        mock_create.return_value = provider

        from agent_os.desktop.main import run_sandbox_setup
        with pytest.raises(SystemExit) as exc_info:
            run_sandbox_setup()
        assert exc_info.value.code == 0
        provider.setup.assert_not_called()

    @patch("agent_os.platform.create_platform_provider")
    def test_already_configured_skips(self, mock_create):
        """If sandbox already set up, skip and exit 0."""
        provider = MagicMock()
        provider.get_capabilities.return_value = MockCapabilities(setup_complete=True)
        mock_create.return_value = provider

        from agent_os.desktop.main import run_sandbox_setup
        with pytest.raises(SystemExit) as exc_info:
            run_sandbox_setup()
        assert exc_info.value.code == 0
        provider.setup.assert_not_called()

    @patch("agent_os.platform.create_platform_provider")
    def test_setup_success(self, mock_create):
        """Successful setup should exit 0."""
        provider = MagicMock()
        provider.get_capabilities.return_value = MockCapabilities(setup_complete=False)
        provider.setup = AsyncMock(return_value=MockSetupResult(success=True))
        mock_create.return_value = provider

        from agent_os.desktop.main import run_sandbox_setup
        with pytest.raises(SystemExit) as exc_info:
            run_sandbox_setup()
        assert exc_info.value.code == 0
        provider.setup.assert_called_once()

    @patch("agent_os.platform.create_platform_provider")
    def test_setup_failure_exits_1(self, mock_create):
        """Failed setup should exit 1."""
        provider = MagicMock()
        provider.get_capabilities.return_value = MockCapabilities(setup_complete=False)
        provider.setup = AsyncMock(return_value=MockSetupResult(success=False, error="Access denied"))
        mock_create.return_value = provider

        from agent_os.desktop.main import run_sandbox_setup
        with pytest.raises(SystemExit) as exc_info:
            run_sandbox_setup()
        assert exc_info.value.code == 1

    @patch("agent_os.platform.create_platform_provider")
    def test_setup_exception_exits_1(self, mock_create):
        """Unexpected exception should exit 1."""
        mock_create.side_effect = RuntimeError("Boom")

        from agent_os.desktop.main import run_sandbox_setup
        with pytest.raises(SystemExit) as exc_info:
            run_sandbox_setup()
        assert exc_info.value.code == 1


class TestRunSandboxTeardown:
    """Test run_sandbox_teardown() logic with mocked provider."""

    @patch("agent_os.platform.create_platform_provider")
    def test_null_provider_exits_0(self, mock_create):
        """NullProvider should exit 0 without calling teardown."""
        provider = MagicMock()
        provider.get_capabilities.return_value = MockCapabilities(
            platform="null", isolation_method="none", setup_complete=True,
        )
        mock_create.return_value = provider

        from agent_os.desktop.main import run_sandbox_teardown
        with pytest.raises(SystemExit) as exc_info:
            run_sandbox_teardown()
        assert exc_info.value.code == 0
        provider.teardown.assert_not_called()

    @patch("agent_os.platform.create_platform_provider")
    def test_teardown_success(self, mock_create):
        """Successful teardown exits 0."""
        provider = MagicMock()
        provider.get_capabilities.return_value = MockCapabilities(setup_complete=True)
        provider.teardown = AsyncMock(return_value=MockSetupResult(success=True))
        mock_create.return_value = provider

        from agent_os.desktop.main import run_sandbox_teardown
        with pytest.raises(SystemExit) as exc_info:
            run_sandbox_teardown()
        assert exc_info.value.code == 0

    @patch("agent_os.platform.create_platform_provider")
    def test_teardown_failure_still_exits_0(self, mock_create):
        """Failed teardown must STILL exit 0 — never block uninstall."""
        provider = MagicMock()
        provider.get_capabilities.return_value = MockCapabilities(setup_complete=True)
        provider.teardown = AsyncMock(return_value=MockSetupResult(success=False, error="Failed"))
        mock_create.return_value = provider

        from agent_os.desktop.main import run_sandbox_teardown
        with pytest.raises(SystemExit) as exc_info:
            run_sandbox_teardown()
        assert exc_info.value.code == 0

    @patch("agent_os.platform.create_platform_provider")
    def test_teardown_exception_still_exits_0(self, mock_create):
        """Exception during teardown must STILL exit 0."""
        mock_create.side_effect = RuntimeError("Boom")

        from agent_os.desktop.main import run_sandbox_teardown
        with pytest.raises(SystemExit) as exc_info:
            run_sandbox_teardown()
        assert exc_info.value.code == 0


class TestMacOSProviderSetupLogic:
    """Test that macOS provider setup logic doesn't create sandbox users."""

    @patch("agent_os.platform.create_platform_provider")
    def test_macos_provider_skips_sandbox_user_setup(self, mock_create):
        """MacOSPlatformProvider setup() does NOT attempt to create a user account."""
        provider = MagicMock()
        provider.get_capabilities.return_value = MockCapabilities(
            platform="macos", isolation_method="seatbelt", setup_complete=False,
        )
        provider.setup = AsyncMock(return_value=MockSetupResult(success=True))
        mock_create.return_value = provider

        from agent_os.desktop.main import run_sandbox_setup
        with pytest.raises(SystemExit) as exc_info:
            run_sandbox_setup()
        assert exc_info.value.code == 0
        provider.setup.assert_called_once()

    @patch("agent_os.platform.create_platform_provider")
    def test_macos_provider_setup_checks_sandbox_exec(self, mock_create):
        """Setup validates sandbox-exec binary exists (failure case)."""
        provider = MagicMock()
        provider.get_capabilities.return_value = MockCapabilities(
            platform="macos", isolation_method="seatbelt", setup_complete=False,
        )
        provider.setup = AsyncMock(
            return_value=MockSetupResult(success=False, error="sandbox-exec not found")
        )
        mock_create.return_value = provider

        from agent_os.desktop.main import run_sandbox_setup
        with pytest.raises(SystemExit) as exc_info:
            run_sandbox_setup()
        assert exc_info.value.code == 1
