# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys

from agent_os.platform.base import PlatformProvider


def create_platform_provider(**kwargs) -> PlatformProvider:
    """Create the appropriate platform provider for the current OS."""
    if os.environ.get("AGENT_OS_NO_SANDBOX"):
        from agent_os.platform.null import NullProvider
        return NullProvider()
    if sys.platform == "win32":
        from agent_os.platform.windows.provider import WindowsPlatformProvider
        return WindowsPlatformProvider(**kwargs)
    elif sys.platform == "darwin":
        from agent_os.platform.macos.provider import MacOSPlatformProvider
        return MacOSPlatformProvider(**kwargs)
    else:
        from agent_os.platform.null import NullProvider
        return NullProvider()
