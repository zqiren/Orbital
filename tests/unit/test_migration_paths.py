# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for platform-aware DATA_DIR in migration.py."""

import os
from unittest.mock import patch

import pytest

from agent_os.desktop.migration import _get_data_dir


class TestDataDirPlatform:
    """Verify _get_data_dir() returns correct paths per platform."""

    @patch("agent_os.desktop.migration.sys")
    def test_data_dir_macos(self, mock_sys):
        """When sys.platform == 'darwin' -> path contains 'Library/Application Support/Orbital'."""
        mock_sys.platform = "darwin"
        result = _get_data_dir()
        assert "Library" in result
        assert "Application Support" in result
        assert "Orbital" in result

    @patch("agent_os.desktop.migration.sys")
    def test_data_dir_windows(self, mock_sys):
        """When sys.platform == 'win32' -> uses APPDATA."""
        mock_sys.platform = "win32"
        fake_appdata = os.path.join("C:", "Users", "test", "AppData", "Roaming")
        with patch.dict(os.environ, {"APPDATA": fake_appdata}):
            result = _get_data_dir()
        assert fake_appdata in result
        assert "Orbital" in result

    @patch("agent_os.desktop.migration.sys")
    def test_data_dir_linux(self, mock_sys):
        """When sys.platform == 'linux' -> uses ~/.orbital."""
        mock_sys.platform = "linux"
        result = _get_data_dir()
        assert result.endswith(".orbital")
