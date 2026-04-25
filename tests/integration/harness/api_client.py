# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Async HTTP + WebSocket client pointed at a :class:`DaemonProcess`.

Endpoints are verified against ``agent_os/api/routes/agents_v2.py`` and
``agent_os/api/app.py`` rather than the task spec's (non-existent)
``API-reference.md``. Notable paths used here:

- ``POST   /api/v2/projects``                  — create
- ``GET    /api/v2/projects``                  — list
- ``GET    /api/v2/projects/{id}``             — get single
- ``DELETE /api/v2/projects/{id}``             — delete
- ``POST   /api/v2/agents/start``              — start agent (dispatch)
- ``POST   /api/v2/agents/{id}/inject``        — inject message
- ``POST   /api/v2/agents/{id}/stop``          — stop agent
- ``GET    /api/v2/agents/{id}/run-status``    — runtime status
- ``WS     /ws``                               — single daemon-wide ws
  with subscribe/pong protocol (see ``agent_os/api/app.py``)
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from typing import Any, AsyncIterator

import httpx

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError as e:  # pragma: no cover - surfaced if dev deps missing
    raise ImportError(
        "The `websockets` package is required for the integration harness"
    ) from e

logger = logging.getLogger(__name__)


class ApiClient:
    """Thin async wrapper around the v2 REST + WebSocket surface."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ApiClient":
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url, timeout=self._timeout
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------ #
    # Generic HTTP helpers
    # ------------------------------------------------------------------ #

    async def get(self, path: str, **kwargs) -> httpx.Response:
        client = await self._ensure_client()
        return await client.get(path, **kwargs)

    async def post(self, path: str, **kwargs) -> httpx.Response:
        client = await self._ensure_client()
        return await client.post(path, **kwargs)

    async def delete(self, path: str, **kwargs) -> httpx.Response:
        client = await self._ensure_client()
        return await client.delete(path, **kwargs)

    # ------------------------------------------------------------------ #
    # Project CRUD
    # ------------------------------------------------------------------ #

    async def create_project(
        self,
        name: str,
        workspace: str,
        *,
        model: str = "",
        api_key: str = "",
        provider: str = "custom",
        sdk: str = "openai",
        autonomy: str = "hands_off",
        is_scratch: bool = False,
        **extra: Any,
    ) -> dict:
        """POST /api/v2/projects.

        ``workspace`` must exist on disk (the route enforces this).
        """
        payload = {
            "name": name,
            "workspace": workspace,
            "model": model,
            "api_key": api_key,
            "provider": provider,
            "sdk": sdk,
            "autonomy": autonomy,
            "is_scratch": is_scratch,
        }
        payload.update(extra)
        resp = await self.post("/api/v2/projects", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def list_projects(self) -> list[dict]:
        resp = await self.get("/api/v2/projects")
        resp.raise_for_status()
        return resp.json()

    async def get_project(self, project_id: str) -> dict:
        resp = await self.get(f"/api/v2/projects/{project_id}")
        resp.raise_for_status()
        return resp.json()

    async def delete_project(self, project_id: str, clear_output: bool = False) -> None:
        resp = await self.delete(
            f"/api/v2/projects/{project_id}",
            params={"clear_output": str(clear_output).lower()},
        )
        # 403 = scratch project (can't delete); let caller decide.
        resp.raise_for_status()

    # ------------------------------------------------------------------ #
    # Agent lifecycle
    # ------------------------------------------------------------------ #

    async def start_agent(self, project_id: str, initial_message: str | None = None) -> dict:
        """POST /api/v2/agents/start — boots the management agent."""
        body: dict = {"project_id": project_id}
        if initial_message is not None:
            body["initial_message"] = initial_message
        resp = await self.post("/api/v2/agents/start", json=body)
        resp.raise_for_status()
        return resp.json()

    async def inject(
        self,
        project_id: str,
        content: str,
        *,
        target: str | None = None,
        nonce: str | None = None,
    ) -> dict:
        """POST /api/v2/agents/{id}/inject — send a user message."""
        body: dict = {"content": content}
        if target is not None:
            body["target"] = target
        if nonce is not None:
            body["nonce"] = nonce
        resp = await self.post(f"/api/v2/agents/{project_id}/inject", json=body)
        resp.raise_for_status()
        return resp.json()

    # Alias keeping the spec's naming.
    dispatch = inject

    async def stop(self, project_id: str) -> dict:
        """POST /api/v2/agents/{id}/stop."""
        resp = await self.post(f"/api/v2/agents/{project_id}/stop")
        # 404 when no active session — let caller inspect.
        if resp.status_code == 404:
            return {"status": "not-running"}
        resp.raise_for_status()
        return resp.json()

    async def get_status(self, project_id: str) -> dict:
        """GET /api/v2/agents/{id}/run-status."""
        resp = await self.get(f"/api/v2/agents/{project_id}/run-status")
        resp.raise_for_status()
        return resp.json()

    async def poll_until_idle(
        self,
        project_id: str,
        *,
        timeout: float = 60.0,
        interval: float = 0.5,
    ) -> dict:
        """Spin on run-status until the reported state is idle-ish."""
        deadline = time.monotonic() + timeout
        last: dict = {}
        idle_states = {"idle", "stopped", "not_running", "done", "completed"}
        while time.monotonic() < deadline:
            last = await self.get_status(project_id)
            status = str(last.get("status", "")).lower()
            if status in idle_states or status == "":
                return last
            import asyncio as _asyncio
            await _asyncio.sleep(interval)
        return last

    async def get_settings(self) -> dict:
        resp = await self.get("/api/v2/settings")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ #
    # WebSocket
    # ------------------------------------------------------------------ #

    def websocket(self, project_ids: list[str] | None = None) -> "_WebSocketContext":
        """Return an async context manager yielding a :class:`WebSocketHandle`.

        The daemon exposes a single multiplexed WebSocket at ``/ws``.
        Clients send ``{"type": "subscribe", "project_ids": [...]}`` to
        receive events for specific projects. Passing ``project_ids``
        here subscribes automatically on connect.
        """
        ws_url = self._base_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        ) + "/ws"
        return _WebSocketContext(ws_url, project_ids or [])


class WebSocketHandle:
    """Thin helper around a live websocket connection."""

    def __init__(self, ws: "websockets.WebSocketClientProtocol") -> None:
        self._ws = ws

    async def send_json(self, payload: dict) -> None:
        await self._ws.send(json.dumps(payload))

    async def receive(self, timeout: float = 10.0) -> dict:
        """Receive one JSON frame within ``timeout`` seconds."""
        import asyncio

        raw = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return json.loads(raw)

    async def receive_until(
        self,
        predicate,
        timeout: float = 10.0,
    ) -> dict:
        """Receive frames until ``predicate(frame)`` is truthy or timeout."""
        import asyncio

        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise TimeoutError("receive_until timed out")
            frame = await self.receive(timeout=remaining)
            if predicate(frame):
                return frame

    async def drain(self, timeout: float = 0.2) -> list[dict]:
        """Pull every frame available within ``timeout`` and return them."""
        import asyncio

        collected: list[dict] = []
        try:
            while True:
                frame = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
                if isinstance(frame, bytes):
                    frame = frame.decode("utf-8", errors="replace")
                try:
                    collected.append(json.loads(frame))
                except json.JSONDecodeError:
                    continue
        except (asyncio.TimeoutError, ConnectionClosed):
            pass
        return collected


class _WebSocketContext:
    """Async context manager yielding a :class:`WebSocketHandle`."""

    def __init__(self, url: str, subscribe_to: list[str]) -> None:
        self._url = url
        self._subs = subscribe_to
        self._cm = None
        self._ws = None

    async def __aenter__(self) -> WebSocketHandle:
        self._cm = websockets.connect(self._url, open_timeout=10, close_timeout=2)
        self._ws = await self._cm.__aenter__()
        handle = WebSocketHandle(self._ws)
        if self._subs:
            await handle.send_json({"type": "subscribe", "project_ids": self._subs})
            # Drain the "subscribed" ack so callers don't see it first.
            import asyncio
            try:
                await asyncio.wait_for(handle.receive(timeout=2.0), timeout=2.0)
            except (asyncio.TimeoutError, TimeoutError):
                pass
        return handle

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._cm is not None:
            with contextlib.suppress(Exception):
                await self._cm.__aexit__(exc_type, exc, tb)
