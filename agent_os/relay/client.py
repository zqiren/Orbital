# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""RelayClient — WebSocket tunnel client for cloud relay.

Maintains a persistent WSS connection to the relay server, proxying
REST requests to the local daemon and forwarding real-time events
(with optional push notifications) to paired mobile clients.
"""

import asyncio
import base64
import json
import logging
import time
from urllib.parse import urlencode

import httpx
import websockets

logger = logging.getLogger(__name__)

# Reconnect back-off parameters
_BACKOFF_BASE = 1.0
_BACKOFF_MAX = 30.0
_HEARTBEAT_INTERVAL = 30  # seconds


class RelayClient:
    """Connects to a cloud relay via WSS and tunnels daemon traffic."""

    def __init__(
        self,
        relay_url: str,
        device_id: str,
        device_secret: str,
        daemon_base_url: str = "http://localhost:8000",
        project_store=None,
    ):
        self.relay_url = relay_url.rstrip("/")
        self.device_id = device_id
        self.device_secret = device_secret
        self.daemon_base_url = daemon_base_url.rstrip("/")
        self._ws = None
        self._running = False
        self._backoff = _BACKOFF_BASE
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._project_store = project_store
        self._push_counts: dict[str, list[float]] = {}
        self._tunnel_semaphore = asyncio.Semaphore(20)
        self._tunnel_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Connect to relay and run the message loop (with auto-reconnect)."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_run()
            except Exception as exc:
                logger.warning("Relay connection lost: %s", exc)
                if self._running:
                    await self._backoff_sleep()

    async def stop(self):
        """Gracefully disconnect from the relay."""
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def _connect_and_run(self):
        """Single connection lifecycle: register, connect, authenticate, loop."""
        # Re-register device before each connection attempt.
        # The relay stores device registrations in memory, so after a relay
        # restart the device entry is gone and WSS auth will fail.
        try:
            from agent_os.relay.device import register_device
            await register_device(self.relay_url, self.device_id, self.device_secret)
        except Exception as exc:
            logger.warning("Device re-registration failed: %s", exc)

        params = urlencode({
            "device_id": self.device_id,
            "secret": self.device_secret,
        })
        # Convert http(s) scheme to ws(s)
        ws_url = self.relay_url.replace("https://", "wss://").replace("http://", "ws://")
        url = f"{ws_url}/relay/tunnel?{params}"

        logger.info("Connecting to relay: %s", url)
        async with websockets.connect(url) as ws:
            self._ws = ws
            self._backoff = _BACKOFF_BASE  # reset on successful connect
            logger.info("Relay tunnel connected")

            # Run heartbeat and message loop concurrently
            heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws))
            try:
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.warning("Non-JSON message from relay: %s", raw[:200])
                        continue
                    task = asyncio.create_task(self._handle_tunnel_message_bounded(msg))
                    self._tunnel_tasks.add(task)
                    task.add_done_callback(self._tunnel_tasks.discard)
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
                for t in self._tunnel_tasks:
                    t.cancel()
                self._tunnel_tasks.clear()
                self._ws = None

    async def _heartbeat_loop(self, ws):
        """Send periodic heartbeats to keep the tunnel alive."""
        while True:
            try:
                await ws.send(json.dumps({"type": "device.status", "status": "online"}))
            except Exception:
                break
            await asyncio.sleep(_HEARTBEAT_INTERVAL)

    async def _backoff_sleep(self):
        """Exponential back-off between reconnect attempts."""
        logger.info("Reconnecting in %.1fs ...", self._backoff)
        await asyncio.sleep(self._backoff)
        self._backoff = min(self._backoff * 2, _BACKOFF_MAX)

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def _handle_tunnel_message_bounded(self, msg: dict):
        """Process a tunnel message with concurrency limit."""
        async with self._tunnel_semaphore:
            await self._handle_tunnel_message(msg)

    async def _handle_tunnel_message(self, msg: dict):
        """Dispatch incoming relay messages by type."""
        msg_type = msg.get("type", "")

        if msg_type == "rest.request":
            await self._tunnel_rest_request(msg)
        elif msg_type == "pairing.code":
            self._resolve_pending(msg.get("request_id"), msg)
        elif msg_type == "pairing.complete":
            # Notification (not request/response) — persist the paired device
            phone_id = msg.get("phone_id")
            if phone_id:
                from agent_os.api.routes.pairing import add_paired_device
                add_paired_device(phone_id)
                logger.info("Paired phone registered: %s", phone_id)
        elif msg_type == "error":
            logger.error("Relay error: %s", msg.get("message", ""))
        else:
            logger.debug("Unhandled relay message type: %s", msg_type)

    # ------------------------------------------------------------------
    # REST tunnelling
    # ------------------------------------------------------------------

    async def _tunnel_rest_request(self, msg: dict):
        """Proxy a REST request from the relay to the local daemon."""
        request_id = msg.get("request_id", "")
        method = msg.get("method", "GET").upper()
        path = msg.get("path", "/")
        headers = msg.get("headers", {})
        body = msg.get("body")
        multipart = msg.get("multipart")

        # Mark request as coming through relay
        headers["X-Via-Relay"] = "true"

        # Strip stale content-type/content-length for multipart
        if multipart:
            headers.pop("content-type", None)
            headers.pop("content-length", None)

        url = f"{self.daemon_base_url}{path}"

        try:
            async with httpx.AsyncClient() as client:
                if multipart:
                    file_info = multipart["file"]
                    file_data = base64.b64decode(file_info["data"])
                    files = {
                        file_info["fieldname"]: (
                            file_info["filename"],
                            file_data,
                            file_info["mimetype"],
                        )
                    }
                    fields = multipart.get("fields", {})
                    resp = await client.request(
                        method, url, headers=headers, files=files, data=fields, timeout=30,
                    )
                else:
                    resp = await client.request(
                        method, url, headers=headers, json=body, timeout=30,
                    )
                response_body = resp.text
                try:
                    response_body = resp.json()
                except Exception:
                    pass

                # Filter out hop-by-hop and content-length headers:
                # the relay re-serialises the body so the original
                # Content-Length would be wrong.
                skip = {"content-length", "transfer-encoding", "connection"}
                fwd_headers = {
                    k: v for k, v in resp.headers.items()
                    if k.lower() not in skip
                }

                await self._send({
                    "type": "rest.response",
                    "request_id": request_id,
                    "status": resp.status_code,
                    "headers": fwd_headers,
                    "body": response_body,
                })
        except Exception as exc:
            logger.error("Tunnel REST error: %s", exc)
            await self._send({
                "type": "rest.response",
                "request_id": request_id,
                "status": 502,
                "headers": {},
                "body": {"detail": str(exc)},
            })

    # ------------------------------------------------------------------
    # Event forwarding
    # ------------------------------------------------------------------

    async def forward_event(self, project_id: str, payload: dict):
        """Forward a WebSocket event through the relay tunnel.

        Called by the WebSocketManager integration hook.
        Wraps the event in an event.forward envelope and optionally
        attaches push notification metadata.
        """
        if not self._ws:
            logger.warning(
                "relay_forward_drop_no_ws type=%s project=%s",
                payload.get("type"),
                project_id,
            )
            return

        event_forward = {
            "type": "event.forward",
            "event": {"project_id": project_id, **payload},
        }
        push = self._should_push(payload, project_id)
        if push:
            event_forward["push"] = push

        await self._send(event_forward)

    def _should_push(self, event: dict, project_id: str) -> dict | None:
        """Return {title, body} for push-worthy events, else None.

        Three tiers:
        - Tier 1 (always push): approval.request, budget.exhausted
        - Tier 2 (check prefs): agent.status, agent.notify
        - Tier 3 (never push): everything else
        """
        event_type = event.get("type", "")
        is_scheduled = event.get("trigger_source") == "schedule"

        # Tier 1: always push, never rate-limited
        if event_type == "approval.request":
            title = "(scheduled) Agent needs approval" if is_scheduled else "Agent needs approval"
            return {
                "title": title,
                "body": self._approval_body(event),
            }
        if event_type == "budget.exhausted":
            return {
                "title": "Budget exhausted",
                "body": "Agent paused — budget limit reached",
            }

        # Tier 3: never push
        if event_type not in ("agent.status", "agent.notify"):
            return None

        # Tier 2: check project notification_prefs
        prefs = self._get_notification_prefs(project_id)

        if event_type == "agent.status":
            status = event.get("status", "")

            # Trigger started: cron-triggered run began
            if status == "running" and is_scheduled and prefs.get("trigger_started", False):
                return {
                    "title": "(scheduled) Run started",
                    "body": "A scheduled task has begun",
                }

            if status == "error" and prefs.get("errors", True):
                title = "(scheduled) Agent error" if is_scheduled else "Agent error"
                return {
                    "title": title,
                    "body": (event.get("reason") or "")[:100],
                }
            if status in ("completed", "idle") and event.get("had_activity") and prefs.get("task_completed", True):
                title = "(scheduled) Task completed" if is_scheduled else "Task completed"
                return {
                    "title": title,
                    "body": (event.get("summary") or "Agent finished")[:100],
                }
            return None

        if event_type == "agent.notify":
            urgency = event.get("urgency", "normal")
            if urgency == "low":
                return None
            if urgency == "normal" and not prefs.get("agent_messages", True):
                return None
            # urgency == "high" bypasses pref check
            if self._is_rate_limited(project_id):
                return None
            title = event.get("title", "Agent notification")
            if is_scheduled:
                title = f"(scheduled) {title}"
            return {
                "title": title,
                "body": (event.get("body") or "")[:200],
            }

        return None

    def _get_notification_prefs(self, project_id: str) -> dict:
        """Read notification_prefs from project_store, falling back to defaults."""
        from agent_os.daemon_v2.project_store import DEFAULT_NOTIFICATION_PREFS
        if self._project_store is None:
            return dict(DEFAULT_NOTIFICATION_PREFS)
        project = self._project_store.get_project(project_id)
        if project is None:
            return dict(DEFAULT_NOTIFICATION_PREFS)
        return project.get("notification_prefs", dict(DEFAULT_NOTIFICATION_PREFS))

    def _is_rate_limited(self, project_id: str) -> bool:
        """Sliding-window rate limit: max 5 pushes per hour per project."""
        now = time.time()
        window = 3600.0  # 1 hour
        max_pushes = 5

        timestamps = self._push_counts.get(project_id, [])
        # Prune old entries
        timestamps = [t for t in timestamps if now - t < window]
        self._push_counts[project_id] = timestamps

        if len(timestamps) >= max_pushes:
            return True

        timestamps.append(now)
        return False

    def _approval_body(self, event: dict) -> str:
        """Extract a readable body from an approval.request event."""
        what = event.get("what")
        if what:
            return str(what)[:100]
        return str(event.get("tool_args", ""))[:100]

    # ------------------------------------------------------------------
    # Pairing helpers
    # ------------------------------------------------------------------

    async def send_pairing_create(self, timeout: float = 30.0) -> dict:
        """Send pairing.create and wait for the relay to return a code."""
        future = asyncio.get_running_loop().create_future()
        request_id = f"pair_{int(time.time() * 1000)}"
        self._pending_requests[request_id] = future

        await self._send({
            "type": "pairing.create",
            "request_id": request_id,
        })

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending_requests.pop(request_id, None)

    async def send_pairing_revoke(self, phone_id: str):
        """Revoke a paired phone through the relay tunnel."""
        await self._send({
            "type": "pairing.revoke",
            "phone_id": phone_id,
        })

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_pending(self, request_id: str | None, msg: dict):
        """Resolve a pending future by request_id."""
        if request_id and request_id in self._pending_requests:
            future = self._pending_requests.pop(request_id)
            if not future.done():
                future.set_result(msg)

    async def _send(self, payload: dict):
        """Send a JSON message through the tunnel WebSocket.

        For event.forward messages, retries up to 3 times with exponential
        backoff on failure.  Other message types (rest.response, pairing)
        are sent once — they are request-scoped and stale retries would
        confuse the receiver.
        """
        if not self._ws:
            logger.warning(
                "relay_send_drop_no_ws type=%s",
                payload.get("type", "unknown"),
            )
            return

        msg_type = payload.get("type", "")
        event = payload.get("event", {})
        event_type = event.get("type", "") if isinstance(event, dict) else ""
        seq = event.get("seq", "") if isinstance(event, dict) else ""

        retryable = msg_type == "event.forward"
        max_attempts = 3 if retryable else 1
        delays = (0.5, 1.0, 2.0)

        for attempt in range(max_attempts):
            try:
                await self._ws.send(json.dumps(payload))
                if retryable:
                    logger.info(
                        "relay_send type=%s seq=%s", event_type, seq,
                    )
                return
            except Exception as exc:
                if retryable:
                    logger.warning(
                        "relay_send_fail type=%s seq=%s attempt=%d err=%s",
                        event_type, seq, attempt + 1, exc,
                    )
                else:
                    logger.warning("Failed to send relay message: %s", exc)
                    return  # non-retryable — give up immediately

                if attempt < max_attempts - 1:
                    await asyncio.sleep(delays[attempt])

        logger.error(
            "relay_send_drop type=%s seq=%s — exhausted %d retries",
            event_type, seq, max_attempts,
        )
