# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for approval card readable context — autonomy.py + session.py changes."""

from unittest.mock import MagicMock

from agent_os.agent.prompt_builder import Autonomy
from agent_os.daemon_v2.autonomy import AutonomyInterceptor


# ── autonomy.py: _describe_tool integration ──────────────────────────


def _make_interceptor():
    ws = MagicMock()
    interceptor = AutonomyInterceptor(
        preset=Autonomy.SUPERVISED,
        ws_manager=ws,
        project_id="test-proj",
    )
    return interceptor, ws


def test_approval_what_uses_describe_tool():
    """Shell tool → 'Running: npm install express'."""
    interceptor, ws = _make_interceptor()
    tool_call = {
        "id": "tc_1",
        "name": "shell",
        "arguments": {"command": "npm install express"},
    }
    interceptor.on_intercept(tool_call, [])
    payload = ws.broadcast.call_args[0][1]
    assert payload["what"] == "Running: npm install express"


def test_approval_what_for_write_tool():
    """Write tool → 'Writing src/app.py'."""
    interceptor, ws = _make_interceptor()
    tool_call = {
        "id": "tc_2",
        "name": "write",
        "arguments": {"path": "src/app.py", "content": "hello"},
    }
    interceptor.on_intercept(tool_call, [])
    payload = ws.broadcast.call_args[0][1]
    assert payload["what"] == "Writing src/app.py"


# ── session.py: recent_activity filtering ────────────────────────────

from agent_os.agent.session import Session


def _make_session_with_messages(messages):
    session = Session.__new__(Session)
    session._messages = list(messages)
    return session


def test_recent_activity_filters_noise():
    """Only user and assistant messages with content are returned."""
    messages = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Hello"},
        {"role": "tool", "content": "tool output here"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    session = _make_session_with_messages(messages)
    result = session.recent_activity()
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"


def test_recent_activity_excludes_empty_content():
    """Assistant messages with content=None are excluded."""
    messages = [
        {"role": "user", "content": "Do something"},
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": "Done!"},
    ]
    session = _make_session_with_messages(messages)
    result = session.recent_activity()
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[1]["content"] == "Done!"
