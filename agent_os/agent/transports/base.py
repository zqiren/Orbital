# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Abstract transport interface and shared event type."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class TransportEvent:
    """Structured event from a transport."""
    event_type: str  # "message" | "tool_use" | "tool_result" | "thinking" | "permission_request" | "status" | "error" | "session_created" | "turn_complete"
    data: dict = field(default_factory=dict)
    raw_text: str = ""


class AgentTransport(ABC):
    """How to talk to an agent process. Separate from what the agent is."""

    @abstractmethod
    async def start(self, command: str, args: list[str], workspace: str, env: dict | None = None) -> None:
        """Spawn or prepare the agent process."""

    @abstractmethod
    async def send(self, message: str) -> str | None:
        """Send a message. Returns response text for request-response transports (pipe), None for streaming."""

    @abstractmethod
    async def read_stream(self) -> AsyncIterator[TransportEvent]:
        """Yield structured events."""

    @abstractmethod
    async def stop(self) -> None:
        """Terminate the agent process."""

    @abstractmethod
    def is_alive(self) -> bool:
        """Check if agent process is running."""

    @property
    def session_id(self) -> str | None:
        """Current session ID, if transport tracks sessions."""
        return None

    async def respond_to_permission(self, permission_id: str, approved: bool) -> None:
        """Respond to a permission request. No-op for transports that don't support it."""
        pass

    # Files the user attached by local path (desktop app) are a read grant
    # for every agent. SubAgentManager hands each transport a callable that
    # returns the project's granted realpaths; a transport whose CLI asks
    # before reading outside its folder answers "allow" for exactly those
    # files, through the CLI's own permission mechanism.
    _read_grants = None

    def set_read_grants(self, grants) -> None:
        """``grants()`` → the realpaths this worker may read (attached files)."""
        self._read_grants = grants

    def granted_read_paths(self) -> frozenset:
        fn = self._read_grants
        if fn is None:
            return frozenset()
        try:
            return frozenset(fn() or ())
        except Exception:
            return frozenset()

    # Spec 166: the worker's own command list (Claude's initialize commands,
    # Codex's skills/list, an ACP available_commands_update), handed to the
    # daemon's worker-commands store whenever the worker reports it.
    # SubAgentManager sets the sink right after constructing the transport.
    _commands_sink = None

    def set_commands_sink(self, sink) -> None:
        """``sink(commands)`` receives each report in the normal form."""
        self._commands_sink = sink

    def _publish_commands(self, commands) -> None:
        """Hand a parsed report on; None (malformed) is dropped. Never raises."""
        sink = self._commands_sink
        if sink is None or commands is None:
            return
        try:
            sink(commands)
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "worker commands sink failed", exc_info=True)

    # Batch 1110 item 13: the native thread id as soon as the worker has one
    # (Claude's init frame, Codex's thread/start), so a daemon restart during
    # the FIRST turn can still resume it. The completion-time record keeps
    # adding model and process identity on top.
    _thread_sink = None

    def set_thread_sink(self, sink) -> None:
        """``sink(native_id, rollout_path=None)`` receives each new id."""
        self._thread_sink = sink

    def _publish_thread(self, native_id, rollout_path=None) -> None:
        """Hand a new native thread id on. Never raises."""
        sink = self._thread_sink
        if sink is None or not native_id:
            return
        try:
            sink(native_id, rollout_path=rollout_path)
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "thread sink failed", exc_info=True)

    def read_is_granted(self, path) -> bool:
        """True when ``path`` (absolute) resolves to a granted attached file."""
        import os
        if not isinstance(path, str) or not path or not os.path.isabs(path):
            return False
        grants = self.granted_read_paths()
        return bool(grants) and os.path.realpath(path) in grants


def transport_event_to_chunk(event: TransportEvent) -> "OutputChunk":
    """Convert a TransportEvent to an OutputChunk for backward compatibility."""
    from agent_os.agent.adapters.base import OutputChunk
    type_map = {
        "message": "response",
        "tool_use": "tool_activity",
        # Spec 100: a worker's tool results and non-empty thinking keep their
        # own chunk types. Falling through to the "response" default would make
        # them the turn's summary text and a chat.sub_agent_message bubble.
        "tool_result": "tool_result",
        "thinking": "thinking",
        "permission_request": "approval_request",
        # Provider reverse requests (questions/plans) block the current turn
        # and must reach ProcessManager as control-plane state, not fall
        # through to an ordinary response bubble.
        "interaction_required": "interaction_required",
        "status": "status",
        "turn_complete": "status",
        # Errors must keep their identity: without this mapping they fell
        # through the "response" default, so the error text became the
        # last-response "summary" and a failed turn was reported as
        # completed (TASK-honest-subagent-completion-reporting, fix 1).
        "error": "error",
        # BACKLOG 005 §4a: an eager resume-identity control event. Must keep
        # its own chunk_type so ProcessManager records the resume id and does
        # NOT treat it as response content (the "response" default would write
        # it to the transcript and broadcast it as a sub-agent message).
        "thread_started": "thread_started",
        # ACP SDK exposes the freshly-created session before the provider's
        # first prompt. Treat it as the same identity control-plane event,
        # never as an empty response that opens a phantom turn.
        "session_created": "thread_started",
    }
    return OutputChunk(
        text=event.raw_text or event.data.get("text", ""),
        chunk_type=type_map.get(event.event_type, "response"),
        timestamp=event.data.get("timestamp", ""),
        metadata=dict(event.data),
    )
