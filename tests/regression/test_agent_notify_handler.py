# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: agent.notify WS event handler.

Root cause: NotifyTool emitted 'agent.notify' WS events but no frontend
handler existed, causing notifications to be silently dropped.

Fix: Added AgentNotifyEvent interface to types.ts, handler in ChatView.tsx
that renders notifications as info cards and triggers browser notifications.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from agent_os.agent.tools.notify import NotifyTool


class TestNotifyToolEmitsEvent:
    """Verify NotifyTool broadcasts correct WS event shape."""

    def test_notify_broadcasts_correct_type(self):
        """execute() broadcasts event with type='agent.notify'."""
        ws = MagicMock()
        tool = NotifyTool(ws, "proj_1")
        tool.execute(title="Build done", body="All tests pass")

        ws.broadcast.assert_called_once()
        args = ws.broadcast.call_args
        assert args[0][0] == "proj_1"
        payload = args[0][1]
        assert payload["type"] == "agent.notify"
        assert payload["project_id"] == "proj_1"
        assert payload["title"] == "Build done"
        assert payload["body"] == "All tests pass"
        assert payload["urgency"] == "normal"
        assert "timestamp" in payload

    def test_notify_with_high_urgency(self):
        """execute() respects urgency parameter."""
        ws = MagicMock()
        tool = NotifyTool(ws, "proj_2")
        tool.execute(title="Error", body="Build failed", urgency="high")

        payload = ws.broadcast.call_args[0][1]
        assert payload["urgency"] == "high"

    def test_notify_with_low_urgency(self):
        """Low urgency returns different message."""
        ws = MagicMock()
        tool = NotifyTool(ws, "proj_3")
        result = tool.execute(title="FYI", body="Updated", urgency="low")
        assert "queued" in result.content.lower()

    def test_notify_requires_title_and_body(self):
        """execute() with empty title/body returns error."""
        ws = MagicMock()
        tool = NotifyTool(ws, "proj_4")
        result = tool.execute(title="", body="")
        assert "error" in result.content.lower()
        ws.broadcast.assert_not_called()
