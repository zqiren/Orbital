# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Cross-platform contract tests for any PlatformProvider returned by the factory.

These should pass on any platform (Windows, macOS, Linux/NullProvider).
"""

import sys

import pytest
import pytest_asyncio

from agent_os.platform import create_platform_provider
from agent_os.platform.types import (
    FolderInfo,
    NetworkRules,
    PermissionResult,
    PlatformCapabilities,
    SetupResult,
)


@pytest_asyncio.fixture
async def provider():
    """Create the platform provider via factory and run setup/teardown."""
    p = create_platform_provider()
    await p.setup()
    yield p
    await p.teardown()


class TestProviderContract:
    """Contract tests that every PlatformProvider must satisfy."""

    def test_capabilities_returns_valid_structure(self, provider):
        """get_capabilities() returns PlatformCapabilities with all fields."""
        caps = provider.get_capabilities()
        assert isinstance(caps, PlatformCapabilities)
        assert isinstance(caps.platform, str)
        assert isinstance(caps.isolation_method, str)
        assert isinstance(caps.setup_complete, bool)
        assert isinstance(caps.setup_issues, list)
        assert isinstance(caps.supports_network_restriction, bool)
        assert isinstance(caps.supports_folder_access, bool)

    def test_capabilities_platform_matches_os(self, provider):
        """platform field matches current OS."""
        caps = provider.get_capabilities()
        if sys.platform == "win32":
            assert caps.platform == "windows"
        elif sys.platform == "darwin":
            assert caps.platform == "macos"
        else:
            # NullProvider on Linux or unsupported platform
            assert caps.platform in ("null", "linux")

    @pytest.mark.asyncio
    async def test_setup_returns_setup_result(self):
        """setup() returns SetupResult."""
        p = create_platform_provider()
        result = await p.setup()
        assert isinstance(result, SetupResult)
        assert isinstance(result.success, bool)
        await p.teardown()

    def test_is_setup_complete_returns_bool(self, provider):
        """is_setup_complete() returns bool."""
        result = provider.is_setup_complete()
        assert isinstance(result, bool)

    def test_get_available_folders_returns_list(self, provider):
        """get_available_folders() returns list (may be empty for NullProvider)."""
        folders = provider.get_available_folders()
        assert isinstance(folders, list)

    def test_folder_info_has_required_fields(self, provider):
        """Each FolderInfo has path, display_name, accessible."""
        folders = provider.get_available_folders()
        if not folders:
            pytest.skip("Provider returned no folders (e.g. NullProvider)")
        for fi in folders:
            assert isinstance(fi, FolderInfo)
            assert isinstance(fi.path, str)
            assert isinstance(fi.display_name, str)
            assert isinstance(fi.accessible, bool)

    def test_configure_network_accepts_rules(self, provider):
        """configure_network() doesn't crash with valid NetworkRules."""
        rules = NetworkRules(mode="allowlist", domains=["example.com"])
        # Should not raise
        provider.configure_network("contract_test", rules)

    @pytest.mark.asyncio
    async def test_stop_nonexistent_process_returns_false(self, provider):
        """stop_process('nonexistent') returns False."""
        result = await provider.stop_process("nonexistent_process_id_12345")
        assert result is False

    def test_grant_folder_access_returns_permission_result(self, provider):
        """grant_folder_access() returns PermissionResult."""
        result = provider.grant_folder_access("/tmp/contract_test", "read_only")
        assert isinstance(result, PermissionResult)
        assert isinstance(result.success, bool)
        assert isinstance(result.path, str)

    def test_revoke_folder_access_returns_permission_result(self, provider):
        """revoke_folder_access() returns PermissionResult."""
        result = provider.revoke_folder_access("/tmp/contract_test")
        assert isinstance(result, PermissionResult)
        assert isinstance(result.success, bool)
        assert isinstance(result.path, str)
