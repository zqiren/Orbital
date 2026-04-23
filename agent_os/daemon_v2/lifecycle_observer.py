# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lifecycle observer — injects system messages on sub-agent state transitions.

Every sub-agent lifecycle event (start, message route, completion, error)
injects a system message into the management agent's session with [Sub-agent]
prefix and transcript path.
"""

import logging

logger = logging.getLogger(__name__)


class LifecycleObserver:
    """Observes sub-agent state transitions and injects trace messages
    into the management agent's session."""

    def __init__(self, agent_manager, ws_manager):
        self._agent_manager = agent_manager
        self._ws = ws_manager

    async def on_started(self, project_id: str, handle: str, initiator: str,
                         transcript_path: str = "unknown") -> None:
        """Sub-agent process spawned."""
        content = f"[Sub-agent] {handle} started (initiated by: {initiator}). Transcript: {transcript_path}"
        await self._inject(project_id, content)
        self._ws.broadcast(project_id, {
            "type": "sub_agent.started",
            "project_id": project_id,
            "handle": handle,
            "initiator": initiator,
        })

    async def on_message_routed(self, project_id: str, handle: str, initiator: str,
                                message_preview: str, transcript_path: str) -> None:
        """A message was routed to a sub-agent."""
        preview = message_preview[:100]
        if initiator == "user_mention":
            content = f'[Sub-agent] User sent @{handle}: "{preview}". Transcript: {transcript_path}'
        else:
            content = f'[Sub-agent] Message sent to {handle}: "{preview}". Transcript: {transcript_path}'
        await self._inject(project_id, content)

    async def on_completed(self, project_id: str, handle: str, summary: str,
                           transcript_path: str) -> None:
        """Sub-agent finished its current task."""
        summary_text = summary[:500] if summary else "(no output)"
        content = f"[Sub-agent] {handle} completed. Summary: {summary_text}. Transcript: {transcript_path}"
        await self._inject(project_id, content)
        self._ws.broadcast(project_id, {
            "type": "sub_agent.completed",
            "project_id": project_id,
            "handle": handle,
            "summary": summary_text,
        })

    async def on_error(self, project_id: str, handle: str, error: str,
                       transcript_path: str) -> None:
        """Sub-agent encountered an error."""
        content = f"[Sub-agent] {handle} stopped with error: {error}. Transcript: {transcript_path}"
        await self._inject(project_id, content)
        self._ws.broadcast(project_id, {
            "type": "sub_agent.error",
            "project_id": project_id,
            "handle": handle,
            "error": error,
        })

    def on_failed(self, project_id: str, handle: str, reason: str) -> None:
        """Sub-agent adapter transitioned into broken state (e.g. background_send exception).

        Synchronous by design — safe to call from exception handlers without
        requiring an event loop context. Broadcasts a failure event so the
        frontend can release the idle indicator and surface a loud failure
        state. No transcript injection here because _inject is async and the
        caller path is a narrow exception branch; the error log plus the
        WebSocket event are the authoritative signals.
        """
        self._ws.broadcast(project_id, {
            "type": "sub_agent.failed",
            "project_id": project_id,
            "handle": handle,
            "reason": reason,
        })

    async def _inject(self, project_id: str, content: str) -> None:
        """Inject a system message into the management agent's session."""
        if self._agent_manager is None:
            return
        try:
            await self._agent_manager.inject_system_message(project_id, content)
        except Exception as e:
            logger.warning("Failed to inject lifecycle message for %s: %s", project_id, e)
