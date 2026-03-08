# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for reasoning field in approval.request payload."""
import pytest
from unittest.mock import MagicMock

from agent_os.agent.prompt_builder import Autonomy
from agent_os.daemon_v2.autonomy import AutonomyInterceptor


class MockWS:
    def __init__(self):
        self.last_payload = None

    def broadcast(self, project_id, payload):
        self.last_payload = payload


def _make_interceptor():
    ws = MockWS()
    return AutonomyInterceptor(Autonomy.SUPERVISED, ws, "test_proj"), ws


def test_approval_payload_includes_reasoning_when_present():
    interceptor, ws = _make_interceptor()
    interceptor.on_intercept(
        {"id": "tc1", "name": "shell", "arguments": {"command": "ls"}},
        [],
        reasoning="I need to list the files to find the config",
    )
    assert ws.last_payload["reasoning"] == "I need to list the files to find the config"


def test_approval_payload_omits_reasoning_when_absent():
    interceptor, ws = _make_interceptor()
    interceptor.on_intercept(
        {"id": "tc1", "name": "shell", "arguments": {"command": "ls"}},
        [],
        reasoning=None,
    )
    assert "reasoning" not in ws.last_payload


def test_approval_payload_omits_reasoning_for_whitespace():
    interceptor, ws = _make_interceptor()
    interceptor.on_intercept(
        {"id": "tc1", "name": "shell", "arguments": {"command": "ls"}},
        [],
        reasoning="   \n  ",
    )
    assert "reasoning" not in ws.last_payload


def test_approval_flow_works_without_reasoning():
    """Existing approve/deny flow unbroken when reasoning absent."""
    interceptor, ws = _make_interceptor()
    interceptor.on_intercept(
        {"id": "tc1", "name": "write", "arguments": {"path": "x.py", "content": "hi"}},
        [],
    )
    # Verify essential fields still present
    assert ws.last_payload["type"] == "approval.request"
    assert ws.last_payload["what"] == "Writing x.py"
    assert ws.last_payload["tool_name"] == "write"
    assert ws.last_payload["tool_call_id"] == "tc1"
    assert "reasoning" not in ws.last_payload
