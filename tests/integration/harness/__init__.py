# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Live-daemon integration test harness.

Exposes three primitives used by integration tests:

- :class:`DaemonProcess` — spawns a real uvicorn subprocess running
  ``agent_os.api.app:create_app`` and manages its lifecycle + log capture.
- :class:`ApiClient` — thin async HTTP + WebSocket client pointed at a
  DaemonProcess. Knows the v2 API endpoints used by the harness.
- ``process_tools`` — cross-platform ``psutil`` helpers for inspecting and
  reaping process trees.

These are infrastructure only; they do not depend on any production code
path inside the daemon.
"""

from .daemon import DaemonProcess
from .api_client import ApiClient
from . import process_tools

__all__ = ["DaemonProcess", "ApiClient", "process_tools"]
