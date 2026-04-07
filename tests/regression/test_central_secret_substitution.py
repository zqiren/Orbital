# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: shell tool received literal `<secret:KEY>` strings.

`substitute_secrets()` from browser_safety.py worked, but only BrowserTool
called it. ShellTool and the approval execution path passed raw arguments
through, so the literal token was forwarded to the shell. The fix moves
substitution into ToolRegistry.execute()/execute_async() so every tool
benefits without needing tool-specific code.

These tests fail without the fix and pass with it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_os.agent.tools.base import Tool, ToolResult
from agent_os.agent.tools.registry import ToolRegistry


class _RecorderTool(Tool):
    """Sync tool that records the arguments it received."""

    name = "recorder"
    description = "test tool"
    parameters = {"type": "object", "properties": {}}

    def __init__(self):
        self.received_args: dict | None = None

    def execute(self, **arguments) -> ToolResult:
        self.received_args = arguments
        return ToolResult(content="ok")


class _AsyncRecorderTool(Tool):
    """Async tool that records the arguments it received."""

    name = "async_recorder"
    description = "test tool"
    parameters = {"type": "object", "properties": {}}
    is_async = True

    def __init__(self):
        self.received_args: dict | None = None

    async def execute(self, **arguments) -> ToolResult:
        self.received_args = arguments
        return ToolResult(content="ok")


def _make_store(values: dict[tuple[str, str], str | None]):
    store = MagicMock()
    store.get_value.side_effect = lambda name, field: values.get((name, field))
    return store


def test_sync_tool_receives_substituted_value():
    """Registry.execute() must substitute <secret:...> tokens before dispatching."""
    store = _make_store({("ical", "url"): "https://calendar.example.com/feed.ics"})
    registry = ToolRegistry(user_credential_store=store)
    tool = _RecorderTool()
    registry.register(tool)

    result = registry.execute(
        "recorder",
        {"command": "curl '<secret:ical.url>'"},
    )

    assert result.content == "ok"
    assert tool.received_args == {
        "command": "curl 'https://calendar.example.com/feed.ics'"
    }
    store.get_value.assert_called_once_with("ical", "url")


@pytest.mark.asyncio
async def test_async_tool_receives_substituted_value():
    """Registry.execute_async() must substitute <secret:...> tokens before dispatching."""
    store = _make_store({("ical", "url"): "https://calendar.example.com/feed.ics"})
    registry = ToolRegistry(user_credential_store=store)
    tool = _AsyncRecorderTool()
    registry.register(tool)

    result = await registry.execute_async(
        "async_recorder",
        {"command": "curl '<secret:ical.url>'"},
    )

    assert result.content == "ok"
    assert tool.received_args == {
        "command": "curl 'https://calendar.example.com/feed.ics'"
    }
    store.get_value.assert_called_once_with("ical", "url")


def test_nested_dict_arguments_are_substituted():
    """Substitution must walk nested dicts/lists, not only the top level."""
    store = _make_store({
        ("ical", "url"): "https://calendar.example.com/feed.ics",
        ("api", "key"): "sk-real-value",
    })
    registry = ToolRegistry(user_credential_store=store)
    tool = _RecorderTool()
    registry.register(tool)

    registry.execute(
        "recorder",
        {
            "config": {
                "url": "<secret:ical.url>",
                "headers": ["Authorization: Bearer <secret:api.key>"],
            },
        },
    )

    assert tool.received_args == {
        "config": {
            "url": "https://calendar.example.com/feed.ics",
            "headers": ["Authorization: Bearer sk-real-value"],
        },
    }


def test_no_credential_store_is_noop():
    """When no credential store is configured, args pass through unchanged."""
    registry = ToolRegistry(user_credential_store=None)
    tool = _RecorderTool()
    registry.register(tool)

    registry.execute(
        "recorder",
        {"command": "curl '<secret:ical.url>'"},
    )

    assert tool.received_args == {"command": "curl '<secret:ical.url>'"}


def test_missing_credential_returns_tool_error():
    """An unknown secret key must surface as a ToolResult error, not an unhandled raise."""
    store = _make_store({})  # empty — no keys resolve
    registry = ToolRegistry(user_credential_store=store)
    tool = _RecorderTool()
    registry.register(tool)

    result = registry.execute(
        "recorder",
        {"command": "curl '<secret:ical.url>'"},
    )

    # Tool was never invoked because substitution failed first.
    assert tool.received_args is None
    assert "Error" in result.content
    assert "ical.url" in result.content
