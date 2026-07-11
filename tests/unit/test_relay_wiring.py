# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for relay startup integration and event bus wiring."""

import asyncio
import os
import threading
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


class TestBroadcastCrossThread:
    """WebSocketManager.broadcast() must be safe to call from a thread
    running its own asyncio event loop (e.g. NetworkProxy's dedicated
    "orbital-network-loop" thread), not just from the loop that owns the
    drain task.
    """

    def test_broadcast_from_foreign_loop_marshals_via_call_soon_threadsafe(self):
        """A broadcast issued while a *different* loop is running on the
        calling thread must not touch the queue directly (unsafe — the
        queue/drain task belong to the owning loop's thread). It must
        hand the item to the owning loop via ``call_soon_threadsafe``,
        the only thread-safe scheduling primitive asyncio provides.

        Pre-fix, ``broadcast()`` calls ``self._queue.put_nowait(...)``
        unconditionally once a queue exists, regardless of which loop is
        currently running on the calling thread — so this assertion
        fails against the foreign thread's identity. Post-fix, the put
        is marshaled onto the owning loop and never executed directly
        from the foreign thread.
        """
        ws_manager = WebSocketManager()
        fake_ws = MagicMock()
        fake_ws.send_json = AsyncMock()
        ws_manager.connect(fake_ws)
        ws_manager.subscribe(fake_ws, ["proj_x"])

        # Establish an "owning" loop for the drain task/queue, the way a
        # real broadcast from the main FastAPI loop would.
        owning_loop = asyncio.new_event_loop()
        try:
            owning_loop.run_until_complete(asyncio.sleep(0))  # give it a tick
            owning_loop.run_until_complete(_call_ensure_drain(ws_manager))
            assert ws_manager._loop is owning_loop

            # Spy on the mechanisms without letting either actually run,
            # so this test is deterministic (no reliance on the owning
            # loop being pumped concurrently).
            put_nowait_threads = []
            real_put_nowait = ws_manager._queue.put_nowait

            def spy_put_nowait(*args, **kwargs):
                put_nowait_threads.append(threading.get_ident())
                return real_put_nowait(*args, **kwargs)

            ws_manager._queue.put_nowait = spy_put_nowait
            owning_loop.call_soon_threadsafe = MagicMock(name="call_soon_threadsafe")

            foreign_thread_ident = {}

            def foreign_thread_body():
                foreign_loop = asyncio.new_event_loop()

                async def run():
                    foreign_thread_ident["id"] = threading.get_ident()
                    ws_manager.broadcast("proj_x", {"type": "cross_thread"})

                try:
                    foreign_loop.run_until_complete(run())
                finally:
                    foreign_loop.close()

            t = threading.Thread(target=foreign_thread_body, daemon=True)
            t.start()
            t.join(timeout=2)
            assert not t.is_alive(), "foreign thread broadcast() call did not return"

            # The item must never be enqueued directly from the foreign
            # thread — only via the owning loop's call_soon_threadsafe.
            assert foreign_thread_ident["id"] not in put_nowait_threads, (
                "broadcast() called Queue.put_nowait directly from a "
                "foreign thread/loop instead of marshaling via "
                "call_soon_threadsafe onto the owning loop"
            )
            owning_loop.call_soon_threadsafe.assert_called_once()
            scheduled_callback, scheduled_args = owning_loop.call_soon_threadsafe.call_args[0]
            assert scheduled_callback == ws_manager._queue.put_nowait
            assert scheduled_args == ("proj_x", {"type": "cross_thread"})
        finally:
            owning_loop.close()

    @pytest.mark.asyncio
    async def test_broadcast_from_foreign_loop_delivers_to_subscriber(self):
        """End-to-end: a broadcast from a thread running its own loop is
        eventually delivered to a subscriber on the owning loop's drain
        task, exercised the way NetworkProxy._fire_blocked really calls
        it (real queue, real drain task, real threads).
        """
        ws_manager = WebSocketManager()
        fake_ws = MagicMock()
        fake_ws.send_json = AsyncMock()
        ws_manager.connect(fake_ws)
        ws_manager.subscribe(fake_ws, ["proj_x"])

        # Establish this test coroutine's loop as the owning loop.
        ws_manager._ensure_drain()
        assert ws_manager._loop is asyncio.get_running_loop()

        def foreign_thread_body():
            foreign_loop = asyncio.new_event_loop()

            async def run():
                ws_manager.broadcast("proj_x", {"type": "cross_thread"})

            try:
                foreign_loop.run_until_complete(run())
            finally:
                foreign_loop.close()

        t = threading.Thread(target=foreign_thread_body, daemon=True)
        t.start()
        t.join(timeout=2)
        assert not t.is_alive()

        for _ in range(40):
            if fake_ws.send_json.await_count:
                break
            await asyncio.sleep(0.05)

        fake_ws.send_json.assert_awaited_once_with({"type": "cross_thread"})


async def _call_ensure_drain(ws_manager):
    ws_manager._ensure_drain()


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
