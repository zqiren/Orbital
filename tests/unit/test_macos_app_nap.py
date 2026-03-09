# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for macOS App Nap prevention."""

from unittest.mock import MagicMock, patch

import pytest

from agent_os.platform.macos.provider import MacOSPlatformProvider


def _make_provider() -> MacOSPlatformProvider:
    """Create a bare MacOSPlatformProvider without calling __init__."""
    provider = MacOSPlatformProvider.__new__(MacOSPlatformProvider)
    provider._app_nap_activity = None
    return provider


def _mock_appkit():
    """Return a mock AppKit module with NSProcessInfo."""
    mock_process_info = MagicMock()
    mock_activity = MagicMock(name="activity_handle")
    mock_process_info.processInfo.return_value.beginActivityWithOptions_reason_.return_value = mock_activity

    mock_module = MagicMock()
    mock_module.NSProcessInfo = mock_process_info
    return mock_module, mock_process_info, mock_activity


# --- _disable_app_nap ---


def test_disable_app_nap_calls_begin_activity():
    """beginActivityWithOptions_reason_ is called with correct options and reason."""
    provider = _make_provider()
    mock_appkit, mock_npi, _ = _mock_appkit()

    with patch.dict("sys.modules", {"AppKit": mock_appkit}):
        provider._disable_app_nap()

    mock_npi.processInfo.return_value.beginActivityWithOptions_reason_.assert_called_once_with(
        0x00FFFFFF,
        "Orbital daemon: maintaining agent connections and background tasks",
    )


def test_disable_app_nap_stores_activity_handle():
    """After _disable_app_nap(), _app_nap_activity is set."""
    provider = _make_provider()
    mock_appkit, _, mock_activity = _mock_appkit()

    with patch.dict("sys.modules", {"AppKit": mock_appkit}):
        provider._disable_app_nap()

    assert provider._app_nap_activity is mock_activity


def test_disable_app_nap_idempotent():
    """Calling _disable_app_nap() twice does not create a second assertion."""
    provider = _make_provider()
    mock_appkit, mock_npi, mock_activity = _mock_appkit()

    with patch.dict("sys.modules", {"AppKit": mock_appkit}):
        provider._disable_app_nap()
        provider._disable_app_nap()  # Second call — should be a no-op

    mock_npi.processInfo.return_value.beginActivityWithOptions_reason_.assert_called_once()


def test_disable_app_nap_handles_import_error():
    """When AppKit is not available, logs warning and continues."""
    provider = _make_provider()

    with patch.dict("sys.modules", {"AppKit": None}):
        # Should not raise
        provider._disable_app_nap()

    assert provider._app_nap_activity is None


# --- _enable_app_nap ---


def test_enable_app_nap_calls_end_activity():
    """endActivity_ is called with the stored handle."""
    provider = _make_provider()
    mock_appkit, mock_npi, mock_activity = _mock_appkit()
    provider._app_nap_activity = mock_activity

    with patch.dict("sys.modules", {"AppKit": mock_appkit}):
        provider._enable_app_nap()

    mock_npi.processInfo.return_value.endActivity_.assert_called_once_with(mock_activity)


def test_enable_app_nap_clears_handle():
    """After _enable_app_nap(), _app_nap_activity is None."""
    provider = _make_provider()
    mock_appkit, _, mock_activity = _mock_appkit()
    provider._app_nap_activity = mock_activity

    with patch.dict("sys.modules", {"AppKit": mock_appkit}):
        provider._enable_app_nap()

    assert provider._app_nap_activity is None


def test_enable_app_nap_noop_when_not_disabled():
    """_enable_app_nap() with no active assertion does nothing."""
    provider = _make_provider()
    # Should not raise — _app_nap_activity is already None
    provider._enable_app_nap()
    assert provider._app_nap_activity is None


# --- setup / teardown integration ---


def test_setup_disables_app_nap():
    """setup() calls _disable_app_nap()."""
    import asyncio

    provider = _make_provider()
    provider._workspace_base = "/tmp/orbital-test-workspace"

    with patch.object(provider, "_disable_app_nap") as mock_disable, \
         patch("shutil.which", return_value="/usr/bin/sandbox-exec"):
        asyncio.run(provider.setup())

    mock_disable.assert_called_once()


def test_teardown_enables_app_nap():
    """teardown() calls _enable_app_nap()."""
    import asyncio

    provider = _make_provider()
    provider._proxies = {}
    provider._processes = {}
    provider._pending_rules = {}

    with patch.object(provider, "_enable_app_nap") as mock_enable:
        asyncio.run(provider.teardown())

    mock_enable.assert_called_once()
