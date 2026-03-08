# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: user messages broadcast via WebSocket for cross-device sync.

Bug: When a user sends a message from phone (via relay), the desktop
webapp never sees it because ActivityTranslator.on_message() didn't
handle role:"user" messages. Desktop required a manual refresh.

Fix: Broadcast chat.user_message event on user message append, with a
nonce for deduplication so the originating client doesn't render twice.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.daemon_v2.activity_translator import ActivityTranslator


class TestUserMessageBroadcast:
    """ActivityTranslator broadcasts chat.user_message for user messages."""

    def test_user_message_broadcasts_event(self):
        """A role:'user' message produces a chat.user_message WS event."""
        ws = MagicMock()
        translator = ActivityTranslator(ws)

        translator.on_message(
            {"role": "user", "content": "hello from phone", "timestamp": "2026-01-01T00:00:00Z"},
            "proj_abc",
        )

        ws.broadcast.assert_called_once()
        args = ws.broadcast.call_args
        assert args[0][0] == "proj_abc"
        payload = args[0][1]
        assert payload["type"] == "chat.user_message"
        assert payload["project_id"] == "proj_abc"
        assert payload["content"] == "hello from phone"
        assert payload["timestamp"] == "2026-01-01T00:00:00Z"

    def test_user_message_includes_nonce(self):
        """Nonce from the message dict is forwarded in the WS event."""
        ws = MagicMock()
        translator = ActivityTranslator(ws)

        translator.on_message(
            {
                "role": "user",
                "content": "test",
                "nonce": "abc-123",
                "timestamp": "2026-01-01T00:00:00Z",
            },
            "proj_abc",
        )

        payload = ws.broadcast.call_args[0][1]
        assert payload["nonce"] == "abc-123"

    def test_user_message_missing_nonce_sends_empty(self):
        """When no nonce is provided, the event includes an empty nonce."""
        ws = MagicMock()
        translator = ActivityTranslator(ws)

        translator.on_message(
            {"role": "user", "content": "no nonce"},
            "proj_abc",
        )

        payload = ws.broadcast.call_args[0][1]
        assert payload["nonce"] == ""

    def test_assistant_message_still_works(self):
        """Existing assistant+tool_calls broadcast is not broken."""
        ws = MagicMock()
        translator = ActivityTranslator(ws)

        translator.on_message(
            {
                "role": "assistant",
                "tool_calls": [{"id": "tc_1", "function": {"name": "shell", "arguments": "{}"}}],
            },
            "proj_abc",
        )

        payload = ws.broadcast.call_args[0][1]
        assert payload["type"] == "agent.activity"
        assert payload["category"] == "command_exec"

    def test_tool_message_still_works(self):
        """Existing tool result broadcast is not broken."""
        ws = MagicMock()
        translator = ActivityTranslator(ws)

        translator.on_message(
            {"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
            "proj_abc",
        )

        payload = ws.broadcast.call_args[0][1]
        assert payload["type"] == "agent.activity"
        assert payload["category"] == "tool_result"


class TestInjectNoncePropagation:
    """Nonce flows from inject endpoint through to session and WS event."""

    @pytest.mark.asyncio
    async def test_nonce_in_session_append(self):
        """inject_message includes nonce in the appended user message."""
        from agent_os.daemon_v2.agent_manager import AgentManager

        mgr = AgentManager(
            project_store=MagicMock(),
            ws_manager=MagicMock(),
            sub_agent_manager=MagicMock(),
            activity_translator=MagicMock(),
            process_manager=MagicMock(),
        )

        # Create a mock handle with an idle (done) task and a real-ish session
        mock_session = MagicMock()
        mock_session._paused_for_approval = False
        mock_session.pop_queued_messages = MagicMock(return_value=[])
        appended_messages = []
        mock_session.append = MagicMock(side_effect=lambda m: appended_messages.append(m))

        mock_task = MagicMock()
        mock_task.done.return_value = True
        mock_task.exception.return_value = None

        mock_handle = MagicMock()
        mock_handle.task = mock_task
        mock_handle.session = mock_session

        mgr._handles["proj_test"] = mock_handle

        # Patch _start_loop to avoid actually starting an agent loop
        with patch.object(mgr, "_start_loop", new_callable=AsyncMock):
            result = await mgr.inject_message("proj_test", "hello", nonce="nonce-xyz")

        assert result == "delivered"
        assert len(appended_messages) == 1
        assert appended_messages[0]["role"] == "user"
        assert appended_messages[0]["content"] == "hello"
        assert appended_messages[0]["nonce"] == "nonce-xyz"

    @pytest.mark.asyncio
    async def test_no_nonce_omits_field(self):
        """When nonce is None, the field is not added to the message."""
        from agent_os.daemon_v2.agent_manager import AgentManager

        mgr = AgentManager(
            project_store=MagicMock(),
            ws_manager=MagicMock(),
            sub_agent_manager=MagicMock(),
            activity_translator=MagicMock(),
            process_manager=MagicMock(),
        )

        mock_session = MagicMock()
        mock_session._paused_for_approval = False
        mock_session.pop_queued_messages = MagicMock(return_value=[])
        appended = []
        mock_session.append = MagicMock(side_effect=lambda m: appended.append(m))

        mock_task = MagicMock()
        mock_task.done.return_value = True
        mock_task.exception.return_value = None

        mock_handle = MagicMock()
        mock_handle.task = mock_task
        mock_handle.session = mock_session

        mgr._handles["proj_test"] = mock_handle

        with patch.object(mgr, "_start_loop", new_callable=AsyncMock):
            await mgr.inject_message("proj_test", "hi")

        assert "nonce" not in appended[0]


class TestFullRoundTrip:
    """End-to-end: phone injects → WS broadcasts → desktop dedup logic."""

    def test_phone_inject_broadcasts_to_desktop(self):
        """Simulates: phone injects message with nonce, WS event is broadcast,
        desktop receives it. Since nonce doesn't match local set, message renders."""
        ws = MagicMock()
        translator = ActivityTranslator(ws)

        # Phone inject: message appended to session → on_append fires
        translator.on_message(
            {
                "role": "user",
                "content": "hello from phone",
                "nonce": "phone-nonce-1",
                "timestamp": "2026-01-01T00:00:00Z",
            },
            "proj_abc",
        )

        payload = ws.broadcast.call_args[0][1]

        # Desktop receives this event. Its local nonce set is empty,
        # so nonce "phone-nonce-1" is NOT in the set → message should render.
        desktop_local_nonces: set[str] = set()
        should_render = payload["nonce"] not in desktop_local_nonces
        assert should_render is True
        assert payload["content"] == "hello from phone"

    def test_desktop_own_send_deduplicates(self):
        """Simulates: desktop sends with nonce, adds to local set,
        then receives WS echo. Nonce matches → skip rendering."""
        ws = MagicMock()
        translator = ActivityTranslator(ws)

        # Desktop generates nonce and adds to local set BEFORE sending
        desktop_local_nonces: set[str] = set()
        nonce = "desktop-nonce-1"
        desktop_local_nonces.add(nonce)

        # Message goes to backend → session.append → on_append → broadcast
        translator.on_message(
            {
                "role": "user",
                "content": "hello from desktop",
                "nonce": nonce,
                "timestamp": "2026-01-01T00:00:00Z",
            },
            "proj_abc",
        )

        payload = ws.broadcast.call_args[0][1]

        # Desktop receives echo. Nonce IS in local set → should NOT render.
        should_render = payload["nonce"] not in desktop_local_nonces
        assert should_render is False

        # After dedup, nonce is cleaned up
        desktop_local_nonces.discard(nonce)
        assert nonce not in desktop_local_nonces

    def test_no_nonce_always_renders(self):
        """Messages without a nonce (e.g. queued messages) always render."""
        ws = MagicMock()
        translator = ActivityTranslator(ws)

        translator.on_message(
            {"role": "user", "content": "queued msg"},
            "proj_abc",
        )

        payload = ws.broadcast.call_args[0][1]

        desktop_local_nonces: set[str] = set()
        # Empty nonce is never in the local set
        should_render = payload["nonce"] not in desktop_local_nonces
        assert should_render is True

    def test_multiple_devices_independent_nonces(self):
        """Two devices send simultaneously — each deduplicates only its own."""
        ws = MagicMock()
        translator = ActivityTranslator(ws)

        phone_nonces: set[str] = set()
        desktop_nonces: set[str] = set()

        phone_nonce = "phone-n1"
        desktop_nonce = "desktop-n1"
        phone_nonces.add(phone_nonce)
        desktop_nonces.add(desktop_nonce)

        # Phone message broadcast
        translator.on_message(
            {"role": "user", "content": "from phone", "nonce": phone_nonce},
            "proj_abc",
        )
        phone_payload = ws.broadcast.call_args[0][1]

        # Desktop receives phone's message: not in desktop's nonce set → render
        assert phone_payload["nonce"] not in desktop_nonces

        # Phone receives its own echo: IS in phone's nonce set → skip
        assert phone_payload["nonce"] in phone_nonces

        ws.broadcast.reset_mock()

        # Desktop message broadcast
        translator.on_message(
            {"role": "user", "content": "from desktop", "nonce": desktop_nonce},
            "proj_abc",
        )
        desktop_payload = ws.broadcast.call_args[0][1]

        # Phone receives desktop's message: not in phone's nonce set → render
        assert desktop_payload["nonce"] not in phone_nonces

        # Desktop receives its own echo: IS in desktop's nonce set → skip
        assert desktop_payload["nonce"] in desktop_nonces
