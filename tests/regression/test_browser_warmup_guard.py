# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: launch_warmup() must raise RuntimeError when patchright is missing."""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

import agent_os.daemon_v2.browser_manager as mod


def test_launch_warmup_raises_when_patchright_missing(tmp_path):
    """launch_warmup() must raise RuntimeError, not TypeError, when patchright is unavailable."""
    manager = mod.BrowserManager.__new__(mod.BrowserManager)
    manager._profile_dir = Path(tmp_path / "profile")
    with patch.object(mod, "async_playwright", None):
        with pytest.raises(RuntimeError, match="patchright is not installed"):
            asyncio.run(manager.launch_warmup("https://example.com"))
