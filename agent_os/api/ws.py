# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""WebSocket manager for real-time event broadcasting.

Supports subscribe/broadcast pattern. Clients subscribe to project IDs
and receive events only for subscribed projects.

broadcast() is synchronous (called from session callbacks) but queues
messages for async delivery via a background drain task.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and subscriptions."""

    def __init__(self):
        self._clients: set = set()
        self._subscriptions: dict[object, set[str]] = {}  # ws -> set of project_ids
        self._queue: asyncio.Queue | None = None
        self._drain_task: asyncio.Task | None = None
        self._on_broadcast_hooks: list = []  # list of async callables(project_id, payload)

    def add_broadcast_hook(self, hook) -> None:
        """Register an async callback invoked on every broadcast.

        The hook signature is: async def hook(project_id: str, payload: dict).
        """
        self._on_broadcast_hooks.append(hook)

    def _ensure_drain(self) -> None:
        """Start the background drain loop if not running."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = loop.create_task(self._drain_loop())

    async def _drain_loop(self) -> None:
        """Background task: pull (project_id, payload) from queue and send."""
        while True:
            try:
                project_id, payload = await self._queue.get()
            except Exception:
                break
            disconnected = []
            for ws in list(self._clients):
                subs = self._subscriptions.get(ws, set())
                if project_id in subs:
                    try:
                        await ws.send_json(payload)
                    except Exception:
                        disconnected.append(ws)
            for ws in disconnected:
                self.disconnect(ws)
            # Fire broadcast hooks (e.g. relay event forwarding)
            for hook in self._on_broadcast_hooks:
                try:
                    await hook(project_id, payload)
                except Exception:
                    logger.debug("Broadcast hook error", exc_info=True)

    def connect(self, websocket) -> None:
        self._clients.add(websocket)
        self._subscriptions[websocket] = set()

    def disconnect(self, websocket) -> None:
        self._clients.discard(websocket)
        self._subscriptions.pop(websocket, None)

    def subscribe(self, websocket, project_ids: list[str]) -> None:
        self._subscriptions[websocket] = set(project_ids)

    def broadcast(self, project_id: str, payload: dict) -> None:
        """Send payload to all clients subscribed to project_id.

        Safe to call from sync code. Enqueues for async delivery.
        Falls back to direct (sync) send_json for non-async WS objects
        (e.g. MagicMock in unit tests).
        """
        self._ensure_drain()
        if self._queue is not None:
            try:
                self._queue.put_nowait((project_id, payload))
                return
            except Exception:
                pass
        # Fallback: direct sync send (works with MagicMock in unit tests)
        disconnected = []
        for ws in list(self._clients):
            subs = self._subscriptions.get(ws, set())
            if project_id in subs:
                try:
                    ws.send_json(payload)
                except Exception:
                    disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)

    def broadcast_global(self, payload: dict) -> None:
        """Send payload to ALL connected clients regardless of subscription.

        Used for daemon-level events that aren't tied to a single project
        (e.g. sub-agent login progress). Same sync/async semantics as
        ``broadcast()``.
        """
        # Use a sentinel project_id understood by the drain loop's filter
        # below: subscription set membership treats this id as if every
        # client subscribed. Easiest implementation: skip the queue and
        # send to every client directly.
        disconnected = []
        for ws in list(self._clients):
            try:
                # Async path will land in the drain queue if we put a
                # broadcast there with a wildcard, but to keep the
                # implementation simple we pump directly.
                send_fn = getattr(ws, "send_json", None)
                if send_fn is None:
                    continue
                result = send_fn(payload)
                # If the send_json is awaitable (real WebSocket), schedule it.
                if asyncio.iscoroutine(result):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(result)
                    except RuntimeError:
                        pass
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)
