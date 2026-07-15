# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for TASK-V5-03: lifecycle observer."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver


class TestLifecycleObserver:
    @pytest.mark.asyncio
    async def test_on_completed_injects_system_message(self):
        am = MagicMock()
        am.inject_system_message = AsyncMock(return_value="delivered")
        ws = MagicMock()
        ws.broadcast = MagicMock()

        observer = LifecycleObserver(am, ws)
        await observer.on_completed("proj1", "claude-code", "Refactored auth", "/path/transcript.jsonl")

        am.inject_system_message.assert_called_once()
        content = am.inject_system_message.call_args[0][1]
        assert "[Sub-agent] claude-code completed" in content
        assert "Refactored auth" in content
        assert "/path/transcript.jsonl" in content

    @pytest.mark.asyncio
    async def test_on_message_routed_user_mention(self):
        am = MagicMock()
        am.inject_system_message = AsyncMock(return_value="delivered")
        ws = MagicMock()

        observer = LifecycleObserver(am, ws)
        await observer.on_message_routed("proj1", "claude-code", "user_mention",
                                          "refactor the auth module", "/path/t.jsonl")

        content = am.inject_system_message.call_args[0][1]
        assert "User sent @claude-code" in content
        assert "refactor the auth module" in content
        assert "/path/t.jsonl" in content

    @pytest.mark.asyncio
    async def test_on_message_routed_management_agent(self):
        am = MagicMock()
        am.inject_system_message = AsyncMock(return_value="delivered")
        ws = MagicMock()

        observer = LifecycleObserver(am, ws)
        await observer.on_message_routed("proj1", "claude-code", "management_agent",
                                          "refactor auth", "/path/t.jsonl")

        content = am.inject_system_message.call_args[0][1]
        assert "Message sent to claude-code" in content

    @pytest.mark.asyncio
    async def test_on_message_routed_stamps_meta_with_dispatch_id(self):
        """TASK-dispatch-id-pairing: when a dispatch_id is given, it is
        stamped onto the injected message's ``_meta`` alongside handle and
        transcript_path — the join key the chat renderer uses instead of
        positional pairing."""
        am = MagicMock()
        am.inject_system_message = AsyncMock(return_value="delivered")
        ws = MagicMock()

        observer = LifecycleObserver(am, ws)
        await observer.on_message_routed(
            "proj1", "claude-code", "management_agent",
            "refactor auth", "/path/t.jsonl",
            dispatch_id="sess_X:aaaa1111",
        )

        am.inject_system_message.assert_awaited_once()
        kwargs = am.inject_system_message.await_args.kwargs
        assert kwargs.get("meta") == {
            "dispatch_id": "sess_X:aaaa1111",
            "handle": "claude-code",
            "transcript_path": "/path/t.jsonl",
        }
        # Human-readable prose is untouched — the migration script and old
        # UIs still rely on its exact shape.
        content = am.inject_system_message.call_args[0][1]
        assert "Message sent to claude-code" in content

    @pytest.mark.asyncio
    async def test_on_message_routed_without_dispatch_id_omits_meta(self):
        """Back-compat: a caller that doesn't pass dispatch_id gets the old
        behavior — no ``meta`` kwarg at all (never a bare ``None`` id)."""
        am = MagicMock()
        am.inject_system_message = AsyncMock(return_value="delivered")
        ws = MagicMock()

        observer = LifecycleObserver(am, ws)
        await observer.on_message_routed(
            "proj1", "claude-code", "management_agent",
            "refactor auth", "/path/t.jsonl",
        )

        kwargs = am.inject_system_message.await_args.kwargs
        assert "meta" not in kwargs

    @pytest.mark.asyncio
    async def test_on_error_injects_system_message(self):
        am = MagicMock()
        am.inject_system_message = AsyncMock(return_value="delivered")
        ws = MagicMock()
        ws.broadcast = MagicMock()

        observer = LifecycleObserver(am, ws)
        await observer.on_error("proj1", "claude-code", "context window exceeded", "/path/t.jsonl")

        content = am.inject_system_message.call_args[0][1]
        assert "stopped with error" in content
        assert "context window exceeded" in content
        assert "/path/t.jsonl" in content

    @pytest.mark.asyncio
    async def test_on_started_injects_and_broadcasts(self):
        am = MagicMock()
        am.inject_system_message = AsyncMock(return_value="delivered")
        ws = MagicMock()
        ws.broadcast = MagicMock()

        observer = LifecycleObserver(am, ws)
        await observer.on_started("proj1", "claude-code", "user_mention",
                                   transcript_path="/path/t.jsonl")

        content = am.inject_system_message.call_args[0][1]
        assert "[Sub-agent] claude-code started" in content
        assert "user_mention" in content
        assert "/path/t.jsonl" in content

        ws.broadcast.assert_called_once()
        event = ws.broadcast.call_args[0][1]
        assert event["type"] == "sub_agent.started"

    @pytest.mark.asyncio
    async def test_completed_broadcasts_websocket_event(self):
        am = MagicMock()
        am.inject_system_message = AsyncMock(return_value="delivered")
        ws = MagicMock()
        ws.broadcast = MagicMock()

        observer = LifecycleObserver(am, ws)
        await observer.on_completed("proj1", "claude-code", "Done", "/path/t.jsonl")

        ws.broadcast.assert_called_once()
        event = ws.broadcast.call_args[0][1]
        assert event["type"] == "sub_agent.completed"
        assert event["handle"] == "claude-code"

    @pytest.mark.asyncio
    async def test_error_broadcasts_websocket_event(self):
        am = MagicMock()
        am.inject_system_message = AsyncMock(return_value="delivered")
        ws = MagicMock()
        ws.broadcast = MagicMock()

        observer = LifecycleObserver(am, ws)
        await observer.on_error("proj1", "claude-code", "timeout", "/path/t.jsonl")

        ws.broadcast.assert_called_once()
        event = ws.broadcast.call_args[0][1]
        assert event["type"] == "sub_agent.error"
        assert event["handle"] == "claude-code"
        assert event["error"] == "timeout"

    @pytest.mark.asyncio
    async def test_inject_tolerates_none_agent_manager(self):
        """If agent_manager is None, _inject should silently return."""
        ws = MagicMock()
        observer = LifecycleObserver(None, ws)
        # Should not raise
        await observer.on_completed("proj1", "claude-code", "Done", "/path/t.jsonl")

    @pytest.mark.asyncio
    async def test_inject_tolerates_agent_manager_exception(self):
        """If inject_system_message raises, _inject should log and continue."""
        am = MagicMock()
        am.inject_system_message = AsyncMock(side_effect=RuntimeError("session gone"))
        ws = MagicMock()
        ws.broadcast = MagicMock()

        observer = LifecycleObserver(am, ws)
        # Should not raise
        await observer.on_started("proj1", "claude-code", "management_agent")
