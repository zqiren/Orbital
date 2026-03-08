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
    event_type: str  # "message" | "tool_use" | "permission_request" | "status" | "error" | "session_created"
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


def transport_event_to_chunk(event: TransportEvent) -> "OutputChunk":
    """Convert a TransportEvent to an OutputChunk for backward compatibility."""
    from agent_os.agent.adapters.base import OutputChunk
    type_map = {
        "message": "response",
        "tool_use": "tool_activity",
        "permission_request": "approval_request",
        "status": "status",
    }
    return OutputChunk(
        text=event.raw_text or event.data.get("text", ""),
        chunk_type=type_map.get(event.event_type, "response"),
        timestamp=event.data.get("timestamp", ""),
        metadata=dict(event.data),
    )
