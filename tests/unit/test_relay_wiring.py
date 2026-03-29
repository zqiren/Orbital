# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for relay startup integration and event bus wiring."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.api.ws import WebSocketManager


class TestBroadcastHook:
    """WebSocketManager.broadcast() fires registered hooks."""

    @pytest.mark.asyncio
    async def test_hook_called_on_broadcast(self):
        """Registered hook receives (project_id, payload) on broadcast."""
        ws_manager = WebSocketManager()
        received = []

        async def hook(project_id, payload):
            received.append((project_id, payload))

        ws_manager.add_broadcast_hook(hook)

        # Connect a fake client and subscribe
        fake_ws = MagicMock()
        fake_ws.send_json = AsyncMock()
        ws_manager.connect(fake_ws)
        ws_manager.subscribe(fake_ws, ["proj_1"])

        # Broadcast
        ws_manager.broadcast("proj_1", {"type": "test", "data": 42})

        # Give the drain loop time to process
        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0] == ("proj_1", {"type": "test", "data": 42})

    @pytest.mark.asyncio
    async def test_hook_error_does_not_break_broadcast(self):
        """A failing hook does not prevent message delivery."""
        ws_manager = WebSocketManager()

        async def bad_hook(project_id, payload):
            raise RuntimeError("hook exploded")

        ws_manager.add_broadcast_hook(bad_hook)

        fake_ws = MagicMock()
        fake_ws.send_json = AsyncMock()
        ws_manager.connect(fake_ws)
        ws_manager.subscribe(fake_ws, ["proj_1"])

        ws_manager.broadcast("proj_1", {"type": "test"})
        await asyncio.sleep(0.1)

        # Message should still have been delivered to ws client
        fake_ws.send_json.assert_called_once_with({"type": "test"})

    @pytest.mark.asyncio
    async def test_multiple_hooks(self):
        """Multiple hooks all get called."""
        ws_manager = WebSocketManager()
        calls = {"a": 0, "b": 0}

        async def hook_a(pid, p):
            calls["a"] += 1

        async def hook_b(pid, p):
            calls["b"] += 1

        ws_manager.add_broadcast_hook(hook_a)
        ws_manager.add_broadcast_hook(hook_b)

        fake_ws = MagicMock()
        fake_ws.send_json = AsyncMock()
        ws_manager.connect(fake_ws)
        ws_manager.subscribe(fake_ws, ["p"])

        ws_manager.broadcast("p", {"type": "x"})
        await asyncio.sleep(0.1)

        assert calls["a"] == 1
        assert calls["b"] == 1


class TestRelayOptIn:
    """Relay is only activated when AGENT_OS_RELAY_URL is set."""

    def test_no_relay_without_env(self):
        """Without AGENT_OS_RELAY_URL, app.state.relay_client is None."""
        env = os.environ.copy()
        env.pop("AGENT_OS_RELAY_URL", None)
        with patch.dict(os.environ, env, clear=True):
            from agent_os.api.app import create_app
            app = create_app(data_dir="orbital-test-data")
            assert app.state.relay_client is None

    def test_relay_initialized_with_env(self, tmp_path, monkeypatch):
        """With AGENT_OS_RELAY_URL set, relay_client is created."""
        monkeypatch.setenv("AGENT_OS_RELAY_URL", "https://relay.example.com")
        # Use tmp_path for device identity to avoid touching real ~/orbital
        monkeypatch.setattr(
            "agent_os.relay.device.get_or_create_device_identity",
            lambda: {"device_id": "dev_test", "device_secret": "secret"},
        )

        from agent_os.api.app import create_app
        app = create_app(data_dir="orbital-test-data")

        assert app.state.relay_client is not None
        assert app.state.relay_client.relay_url == "https://relay.example.com"
        assert app.state.relay_client.device_id == "dev_test"
