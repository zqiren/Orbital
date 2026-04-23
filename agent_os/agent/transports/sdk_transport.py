# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""SDK transport: uses claude-agent-sdk to communicate with Claude Code."""
import asyncio
import logging
import uuid
from typing import AsyncIterator

from agent_os.agent.transports.base import AgentTransport, TransportEvent
from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.transports.tool_risk import should_auto_approve

try:
    from claude_agent_sdk import (
        ClaudeSDKClient,
        ClaudeAgentOptions,
        PermissionResultAllow,
        PermissionResultDeny,
    )
    from claude_agent_sdk.types import (
        AssistantMessage,
        ResultMessage,
        TextBlock,
        ToolUseBlock,
    )
    HAS_SDK = True
except ImportError:
    HAS_SDK = False

logger = logging.getLogger(__name__)


class SDKTransport(AgentTransport):
    """Transport using the Claude Agent SDK (claude-agent-sdk).

    The SDK spawns claude.exe as a subprocess internally. We configure it
    with a can_use_tool callback that routes permission requests through
    AgentOS's approval system.
    """

    def __init__(self, autonomy: "Autonomy | None" = None):
        if not HAS_SDK:
            raise ImportError("claude-agent-sdk is not installed")
        self._client: "ClaudeSDKClient | None" = None
        self._session_id: str | None = None
        self._alive: bool = False
        self._workspace: str = ""
        # Pending permission requests: request_id -> asyncio.Future
        self._pending_approvals: dict[str, asyncio.Future] = {}
        # Metadata for pending approvals (for REST recovery):
        # request_id -> {"request_id", "tool_name", "tool_input"}
        self._pending_approval_data: dict[str, dict] = {}
        # Event queue for streaming events (tool use, permission requests)
        self._event_queue: asyncio.Queue[TransportEvent] = asyncio.Queue()
        # Flush flag: set when receive_response() ends without a ResultMessage
        # (e.g. due to unknown message type crash). Next send() drains stale
        # messages before issuing a new query.
        self._needs_flush: bool = False
        # Autonomy preset for filtering permission requests.
        # When set, low-risk tools are auto-approved without surfacing to the user.
        self._autonomy: Autonomy | None = autonomy
        # Strongly-referenced handle to the background response-consumption
        # task so the event loop cannot GC it mid-stream. Also used by
        # stop() to cancel in-flight iteration before disconnect().
        self._bg_task: asyncio.Task | None = None

    async def start(self, command: str, args: list[str], workspace: str, env: dict | None = None) -> None:
        self._workspace = workspace
        # Build env for SDK subprocess: override CLAUDECODE to prevent nested session detection
        # The SDK merges options.env on top of os.environ, so we must explicitly blank it
        sdk_env = dict(env) if env else {}
        sdk_env.pop("CLAUDECODE", None)
        sdk_env["CLAUDECODE"] = ""  # Override os.environ value in subprocess
        options = ClaudeAgentOptions(
            cwd=workspace,
            permission_mode="default",
            can_use_tool=self._handle_permission,
            cli_path=command or None,
            env=sdk_env,
        )
        self._client = ClaudeSDKClient(options=options)
        await self._client.connect()
        self._alive = True

    async def send(self, message: str) -> str | None:
        if self._client is None:
            return "Error: SDK client not initialized"

        # If previous receive_response() ended abnormally (without ResultMessage),
        # drain any stale messages left in the SDK buffer before querying again.
        if self._needs_flush:
            await self._flush_stale_messages()

        try:
            await self._client.query(message)
        except Exception as e:
            return f"Error: SDK query failed: {e}"

        response_parts = []
        got_result = False
        try:
            async for msg in self._client.receive_response():
                if isinstance(msg, ResultMessage):
                    got_result = True
                events = self._message_to_events(msg)
                for event in events:
                    # Queue ALL events for read_stream consumers (enables real-time streaming)
                    await self._event_queue.put(event)
                    # Also collect message text for send() return value
                    if event.event_type == "message":
                        response_parts.append(event.raw_text)
        except Exception as e:
            # SDK may raise on unknown message types (e.g. rate_limit_event).
            # If we already collected some response text, return it rather than failing.
            if response_parts:
                logger.warning("SDKTransport.send: partial response due to: %s", e)
            else:
                return f"Error: SDK receive failed: {e}"

        # If iteration ended without a ResultMessage, the SDK's internal buffer
        # may still contain the real ResultMessage. Flag for flush on next send().
        if not got_result:
            self._needs_flush = True
            logger.warning("SDKTransport.send: response stream ended without ResultMessage; will flush on next send")
        else:
            self._needs_flush = False

        if not response_parts:
            return "(no response)"

        return "\n".join(response_parts)

    async def dispatch(self, message: str) -> None:
        """Send query without waiting for response. Events flow via read_stream()."""
        if self._client is None:
            raise RuntimeError("SDK client not initialized — call start() first")

        if self._needs_flush:
            await self._flush_stale_messages()

        await self._client.query(message)
        # Strongly reference the background task on self so it cannot be
        # garbage-collected mid-stream (which produced "Task was destroyed
        # but it is pending" warnings). The done-callback routes any
        # non-cancellation exception to the logger instead of being lost.
        self._bg_task = asyncio.create_task(
            self._consume_response_background(),
            name=f"sdk-consume-{id(self)}",
        )
        self._bg_task.add_done_callback(self._on_bg_task_done)

    def _on_bg_task_done(self, task: "asyncio.Task") -> None:
        """Completion hook for the background consumer task.

        Responsibilities:
        - swallow CancelledError (expected on stop())
        - log any other exception at ERROR with traceback so it does not
          vanish into the void the way an orphan task's exception would.
        """
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        if isinstance(exc, asyncio.CancelledError):
            return
        logger.error(
            "SDKTransport: background response task raised",
            exc_info=exc,
        )

    async def _consume_response_background(self) -> None:
        """Background task: iterate receive_response(), feed events to queue."""
        got_result = False
        try:
            try:
                async for msg in self._client.receive_response():
                    if isinstance(msg, ResultMessage):
                        got_result = True
                    events = self._message_to_events(msg)
                    for event in events:
                        await self._event_queue.put(event)
            except Exception as e:
                logger.warning("SDKTransport: background response consumption failed: %s", e)
                await self._event_queue.put(TransportEvent(
                    event_type="error",
                    data={"error": str(e)},
                    raw_text=f"Error: {e}",
                ))

            if not got_result:
                self._needs_flush = True
                logger.warning("SDKTransport: background response ended without ResultMessage; will flush on next send")
            else:
                self._needs_flush = False
        finally:
            # Always signal turn completion so adapter transitions to idle
            await self._event_queue.put(TransportEvent(event_type="turn_complete"))

    async def _flush_stale_messages(self) -> None:
        """Drain leftover messages from the SDK buffer after a prior crash.

        When receive_response() raises mid-stream (e.g. on an unknown message
        type like rate_limit_event), the ResultMessage may still be sitting in
        the SDK's internal channel. We consume it here so the next query()
        starts with a clean slate.
        """
        if self._client is None:
            self._needs_flush = False
            return

        logger.info("SDKTransport: flushing stale messages from previous response")
        try:
            async for msg in self._client.receive_response():
                if isinstance(msg, ResultMessage):
                    self._session_id = getattr(msg, 'session_id', None)
                    logger.info("SDKTransport: flushed stale ResultMessage (session=%s)", self._session_id)
                    break
        except Exception as e:
            logger.warning("SDKTransport: flush encountered error (ignored): %s", e)

        self._needs_flush = False

    async def read_stream(self) -> AsyncIterator[TransportEvent]:
        """Yield events from the queue. Waits for events while alive."""
        while self._alive:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.5)
                yield event
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> None:
        # Deny any pending approvals
        for request_id, future in list(self._pending_approvals.items()):
            if not future.done():
                future.set_result(False)
        self._pending_approvals.clear()
        self._pending_approval_data.clear()

        # Cancel the background response-consumption task BEFORE
        # disconnecting the client. If the order were reversed, the task
        # would observe a None/closed client mid-iteration and surface a
        # spurious error.
        bg_task = self._bg_task
        if bg_task is not None and not bg_task.done():
            bg_task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(bg_task, return_exceptions=True),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "SDKTransport: background task did not cancel within 5s; "
                    "proceeding with disconnect"
                )
        self._bg_task = None

        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._client = None

        self._alive = False
        self._session_id = None

    def is_alive(self) -> bool:
        return self._alive

    def update_autonomy(self, preset: Autonomy) -> None:
        """Update the autonomy preset for permission filtering (live settings update)."""
        self._autonomy = preset

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def respond_to_permission(self, permission_id: str, approved: bool) -> None:
        """Resolve a pending permission request."""
        future = self._pending_approvals.pop(permission_id, None)
        self._pending_approval_data.pop(permission_id, None)
        if future is not None and not future.done():
            future.set_result(approved)

    async def _handle_permission(self, tool_name: str, tool_input: dict, context) -> "PermissionResultAllow | PermissionResultDeny":
        """Called by the SDK when a tool needs permission.

        Emits a permission_request TransportEvent and waits for approval
        via respond_to_permission().
        """
        # Auto-approve if autonomy preset allows this tool
        if self._autonomy is not None and should_auto_approve(tool_name, self._autonomy):
            logger.debug("SDKTransport: auto-approved %s (autonomy=%s)", tool_name, self._autonomy.value)
            return PermissionResultAllow()

        request_id = str(uuid.uuid4())

        # Create a future for the approval response
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_approvals[request_id] = future
        self._pending_approval_data[request_id] = {
            "request_id": request_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
        }

        # Emit permission request event
        event = TransportEvent(
            event_type="permission_request",
            data={
                "request_id": request_id,
                "tool_name": tool_name,
                "tool_input": tool_input,
            },
            raw_text=f"Permission requested: {tool_name}",
        )
        await self._event_queue.put(event)

        # Wait for approval/denial
        try:
            approved = await future
        except asyncio.CancelledError:
            self._pending_approvals.pop(request_id, None)
            self._pending_approval_data.pop(request_id, None)
            return PermissionResultDeny(message="Request cancelled")

        if approved:
            return PermissionResultAllow()
        else:
            return PermissionResultDeny(message="Denied by user")

    def _message_to_events(self, msg) -> list[TransportEvent]:
        """Convert SDK message types to TransportEvents."""
        events = []

        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    events.append(TransportEvent(
                        event_type="message",
                        data={"text": block.text},
                        raw_text=block.text,
                    ))
                elif isinstance(block, ToolUseBlock):
                    events.append(TransportEvent(
                        event_type="tool_use",
                        data={
                            "tool_name": block.name,
                            "tool_id": block.id,
                            "tool_input": block.input,
                        },
                        raw_text=f"[Using tool: {block.name}]",
                    ))

        elif isinstance(msg, ResultMessage):
            self._session_id = getattr(msg, 'session_id', None)
            if msg.is_error and msg.result:
                events.append(TransportEvent(
                    event_type="error",
                    data={"error": msg.result},
                    raw_text=f"Error: {msg.result}",
                ))

        return events
