# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for agent_os.agent.tools.notify — NotifyTool."""

import pytest

from agent_os.agent.tools.notify import NotifyTool


class FakeWSManager:
    """Captures broadcast calls."""

    def __init__(self):
        self.broadcasts: list[tuple[str, dict]] = []

    def broadcast(self, project_id: str, payload: dict):
        self.broadcasts.append((project_id, payload))


class TestNotifyTool:
    def setup_method(self):
        self.ws = FakeWSManager()
        self.tool = NotifyTool(ws_manager=self.ws, project_id="proj_abc")

    def test_emits_agent_notify_event(self):
        result = self.tool.execute(title="Done", body="Task finished")
        assert result.content == "Notification sent: Done"
        assert len(self.ws.broadcasts) == 1
        pid, payload = self.ws.broadcasts[0]
        assert pid == "proj_abc"
        assert payload["type"] == "agent.notify"
        assert payload["project_id"] == "proj_abc"
        assert payload["title"] == "Done"
        assert payload["body"] == "Task finished"
        assert payload["urgency"] == "normal"
        assert "timestamp" in payload

    def test_low_urgency_emits_event_with_low_flag(self):
        result = self.tool.execute(title="FYI", body="Minor update", urgency="low")
        assert result.content == "Notification queued (low urgency)"
        assert len(self.ws.broadcasts) == 1
        _, payload = self.ws.broadcasts[0]
        assert payload["urgency"] == "low"

    def test_high_urgency(self):
        result = self.tool.execute(title="Alert", body="Critical issue", urgency="high")
        assert result.content == "Notification sent: Alert"
        _, payload = self.ws.broadcasts[0]
        assert payload["urgency"] == "high"

    def test_missing_title_returns_error(self):
        result = self.tool.execute(title="", body="some body")
        assert "Error" in result.content
        assert len(self.ws.broadcasts) == 0

    def test_missing_body_returns_error(self):
        result = self.tool.execute(title="Title", body="")
        assert "Error" in result.content
        assert len(self.ws.broadcasts) == 0

    def test_default_urgency_is_normal(self):
        self.tool.execute(title="Test", body="Body")
        _, payload = self.ws.broadcasts[0]
        assert payload["urgency"] == "normal"

    def test_schema_has_required_fields(self):
        schema = self.tool.schema()
        func = schema["function"]
        assert func["name"] == "notify"
        assert "title" in func["parameters"]["properties"]
        assert "body" in func["parameters"]["properties"]
        assert func["parameters"]["required"] == ["title", "body"]
