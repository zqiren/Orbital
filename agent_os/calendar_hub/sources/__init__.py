# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Calendar sources feeding the CalendarHub (Task G1)."""

from .eventkit import EventKitSource
from .mcp_calendar import McpCalendarSource

__all__ = ["EventKitSource", "McpCalendarSource"]
