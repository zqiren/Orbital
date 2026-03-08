# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for tool description persistence via _activity_descriptions.

Covers:
- ActivityTranslator attaches _activity_descriptions to assistant messages with tool_calls
- on_append fires BEFORE JSONL write so descriptions are persisted
- Crashing on_append does not prevent JSONL write
- Descriptions survive session reload from disk
"""

import json
from unittest.mock import MagicMock

from agent_os.agent.session import Session
from agent_os.daemon_v2.activity_translator import ActivityTranslator


def test_activity_descriptions_persisted_to_jsonl(tmp_path):
    """Tool descriptions should be persisted in JSONL via _activity_descriptions."""
    filepath = str(tmp_path / "test_session.jsonl")
    session = Session(filepath)
    # Create empty file
    with open(filepath, "w") as f:
        pass

    # Wire up activity translator
    ws = MagicMock()
    translator = ActivityTranslator(ws)

    def on_append(msg):
        translator.on_message(msg, "test_project")

    session.on_append = on_append

    # Append assistant message with browser navigate tool call
    msg = {
        "role": "assistant",
        "content": None,
        "source": "management",
        "tool_calls": [
            {
                "id": "tc_001",
                "type": "function",
                "function": {
                    "name": "browser",
                    "arguments": json.dumps({"action": "navigate", "url": "https://example.com"})
                }
            }
        ]
    }
    session.append(msg)

    # Read back from JSONL
    with open(filepath, "r", encoding="utf-8") as f:
        persisted = json.loads(f.readline())

    assert "_activity_descriptions" in persisted
    assert persisted["_activity_descriptions"]["tc_001"] == "Navigating to https://example.com"


def test_activity_descriptions_multiple_tool_calls(tmp_path):
    """Multiple tool calls in one message should each get a description."""
    filepath = str(tmp_path / "test_session.jsonl")
    session = Session(filepath)
    with open(filepath, "w") as f:
        pass

    ws = MagicMock()
    translator = ActivityTranslator(ws)
    session.on_append = lambda msg: translator.on_message(msg, "test_project")

    msg = {
        "role": "assistant",
        "content": None,
        "source": "management",
        "tool_calls": [
            {
                "id": "tc_001",
                "type": "function",
                "function": {
                    "name": "browser",
                    "arguments": json.dumps({"action": "search", "query": "AI agents"})
                }
            },
            {
                "id": "tc_002",
                "type": "function",
                "function": {
                    "name": "shell",
                    "arguments": json.dumps({"command": "ls -la"})
                }
            },
        ]
    }
    session.append(msg)

    with open(filepath, "r", encoding="utf-8") as f:
        persisted = json.loads(f.readline())

    descs = persisted["_activity_descriptions"]
    assert len(descs) == 2
    assert descs["tc_001"] == "Searching web for 'AI agents'"
    assert descs["tc_002"] == "Running: ls -la"


def test_activity_descriptions_absent_for_non_tool_messages(tmp_path):
    """Plain assistant messages (no tool_calls) should NOT have _activity_descriptions."""
    filepath = str(tmp_path / "test_session.jsonl")
    session = Session(filepath)
    with open(filepath, "w") as f:
        pass

    ws = MagicMock()
    translator = ActivityTranslator(ws)
    session.on_append = lambda msg: translator.on_message(msg, "test_project")

    msg = {
        "role": "assistant",
        "content": "Hello, how can I help?",
        "source": "management",
    }
    session.append(msg)

    with open(filepath, "r", encoding="utf-8") as f:
        persisted = json.loads(f.readline())

    assert "_activity_descriptions" not in persisted


def test_on_append_failure_does_not_prevent_jsonl_write(tmp_path):
    """A crashing on_append callback should NOT prevent the message from being written to JSONL."""
    filepath = str(tmp_path / "test_session.jsonl")
    session = Session(filepath)
    with open(filepath, "w") as f:
        pass

    def bad_callback(msg):
        raise RuntimeError("Callback explosion!")

    session.on_append = bad_callback

    msg = {
        "role": "user",
        "content": "test message",
        "source": "user",
    }
    session.append(msg)

    # Message should still be written despite callback failure
    with open(filepath, "r", encoding="utf-8") as f:
        persisted = json.loads(f.readline())

    assert persisted["content"] == "test message"
    assert persisted["role"] == "user"


def test_descriptions_survive_session_reload(tmp_path):
    """Descriptions persisted to JSONL should be present when session is reloaded from disk."""
    filepath = str(tmp_path / "test_session.jsonl")
    session = Session(filepath)
    with open(filepath, "w") as f:
        pass

    ws = MagicMock()
    translator = ActivityTranslator(ws)
    session.on_append = lambda msg: translator.on_message(msg, "test_project")

    msg = {
        "role": "assistant",
        "content": None,
        "source": "management",
        "tool_calls": [
            {
                "id": "tc_reload_1",
                "type": "function",
                "function": {
                    "name": "browser",
                    "arguments": json.dumps({"action": "screenshot"})
                }
            }
        ]
    }
    session.append(msg)

    # Also add a tool result so we don't trigger orphan healing
    session.append({
        "role": "tool",
        "content": "screenshot taken",
        "tool_call_id": "tc_reload_1",
        "source": "management",
    })

    # Reload from disk
    reloaded = Session.load(filepath)
    messages = reloaded.get_messages()
    assistant_msg = [m for m in messages if m.get("role") == "assistant"][0]

    assert "_activity_descriptions" in assistant_msg
    assert assistant_msg["_activity_descriptions"]["tc_reload_1"] == "Taking screenshot"
