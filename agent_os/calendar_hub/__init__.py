# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Calendar hub (Spec 011 §0.4/§2; Task G1).

Normalizes calendar sources (macOS EventKit, a connected Google-Calendar MCP
connector, future CalDAV/Outlook) into one event feed with a project-linkage
layer. ``CalendarHub`` is a global singleton wired beside ``AgentManager`` in
the app factory and driven by ``agent_os/api/routes/calendar.py``.
"""

from .hub import CalendarHub
from .linkage import Linkage
from .models import NormalizedEvent

__all__ = ["CalendarHub", "Linkage", "NormalizedEvent"]
