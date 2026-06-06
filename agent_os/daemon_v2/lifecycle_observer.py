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
                         transcript_path: str = "unknown",
                         *, session_id: str | None = None,
                         inject: bool = True) -> None:
        """Sub-agent process spawned.

        ``inject=False`` suppresses the session injection (the WS broadcast
        always fires — frontend status chips need it). Used by spawn-on-demand
        ``send`` (TASK-collapse-dispatch-to-send H2): one action, one session
        message — the dispatch ack carries the spawn/resume status, so a
        standalone "started" line would double-announce.
        """
        if inject:
            content = f"[Sub-agent] {handle} started (initiated by: {initiator}). Transcript: {transcript_path}"
            await self._inject(project_id, content, session_id=session_id)
        self._ws.broadcast(project_id, {
            "type": "sub_agent.started",
            "project_id": project_id,
            "handle": handle,
            "initiator": initiator,
            "session_id": session_id,
        })

    async def on_message_routed(self, project_id: str, handle: str, initiator: str,
                                message_preview: str, transcript_path: str,
                                *, session_id: str | None = None) -> None:
        """A message was routed to a sub-agent."""
        preview = message_preview[:100]
        if initiator == "user_mention":
            content = f'[Sub-agent] User sent @{handle}: "{preview}". Transcript: {transcript_path}'
        else:
            content = f'[Sub-agent] Message sent to {handle}: "{preview}". Transcript: {transcript_path}'
        await self._inject(project_id, content, session_id=session_id)

    async def on_completed(self, project_id: str, handle: str, summary: str,
                           transcript_path: str,
                           *, session_id: str | None = None) -> None:
        """Sub-agent finished its current task."""
        summary_text = summary[:500] if summary else "(no output)"
        content = f"[Sub-agent] {handle} completed. Summary: {summary_text}. Transcript: {transcript_path}"
        await self._inject(project_id, content, session_id=session_id)
        self._ws.broadcast(project_id, {
            "type": "sub_agent.completed",
            "project_id": project_id,
            "handle": handle,
            "summary": summary_text,
            "session_id": session_id,
        })

    async def on_error(self, project_id: str, handle: str, error: str,
                       transcript_path: str,
                       *, session_id: str | None = None) -> None:
        """Sub-agent encountered an error."""
        content = f"[Sub-agent] {handle} stopped with error: {error}. Transcript: {transcript_path}"
        await self._inject(project_id, content, session_id=session_id)
        self._ws.broadcast(project_id, {
            "type": "sub_agent.error",
            "project_id": project_id,
            "handle": handle,
            "error": error,
            "session_id": session_id,
        })

    async def on_failed(self, project_id: str, handle: str, reason: str,
                        *, session_id: str | None = None) -> None:
        """Sub-agent adapter transitioned into broken state (e.g. background_send exception).

        Injects a system message into the management session AND broadcasts
        the failure event. WebSocket-only (the old behavior) meant the
        management agent was never told its dispatch failed and would wait
        on a sub-agent that was already dead
        (TASK-honest-subagent-completion-reporting, path e).
        """
        content = (
            f"[Sub-agent] {handle} failed: {reason}. "
            f"The dispatched task did not complete."
        )
        await self._inject(project_id, content, session_id=session_id)
        self._ws.broadcast(project_id, {
            "type": "sub_agent.failed",
            "project_id": project_id,
            "handle": handle,
            "reason": reason,
            "session_id": session_id,
        })

    async def on_user_stopped(self, project_id: str, handle: str, *,
                              terminated: list[str] | None = None,
                              session_id: str | None = None) -> None:
        """User pressed the sub-agent stop button (Piece 3 Part D).

        Injects an honest record into the management session — including
        which tracked background work was terminated — and broadcasts so
        every client updates its badge. Loud by design: a kill that
        destroys background work must never be silent.
        """
        content = f"[Sub-agent] {handle} stopped by user."
        if terminated:
            cmds = "; ".join(c[:80] for c in terminated)
            content += (
                f" Terminated {len(terminated)} background process(es): "
                f"{cmds}. This background work did NOT complete."
            )
        await self._inject(project_id, content, session_id=session_id)
        self._ws.broadcast(project_id, {
            "type": "sub_agent.stopped",
            "project_id": project_id,
            "handle": handle,
            "background_terminated": terminated or [],
            "session_id": session_id,
        })

    async def on_background_work_lost(self, project_id: str, handle: str, *,
                                      commands: list[str],
                                      session_id: str | None = None) -> None:
        """A teardown (eviction / project stop / shutdown) terminated live
        tracked background work (Piece 3 Part E). LOUD by contract — silent
        loss recreates the 'completed ≠ done' corruption."""
        cmds = "; ".join(c[:80] for c in commands)
        content = (
            f"[Sub-agent] {handle} teardown terminated "
            f"{len(commands)} live background process(es): {cmds}. "
            f"This background work did NOT complete."
        )
        await self._inject(project_id, content, session_id=session_id)
        self._ws.broadcast(project_id, {
            "type": "sub_agent.background_lost",
            "project_id": project_id,
            "handle": handle,
            "commands": commands,
            "session_id": session_id,
        })

    async def on_thread_update(self, project_id: str, handle: str,
                               *, claude_session_id: str,
                               model: str | None = None,
                               session_id: str | None = None) -> None:
        """A sub-agent turn completed carrying its resume identity.

        Routes to AgentManager, which persists the ``(SessionKey, handle)``
        record into the management session's meta rows
        (TASK-resume-persistence). Not a user-facing event — no injection,
        no broadcast.
        """
        if self._agent_manager is None:
            return
        try:
            self._agent_manager.record_sub_agent_thread(
                project_id, handle,
                claude_session_id=claude_session_id, model=model,
                session_id=session_id,
            )
        except Exception:
            logger.exception(
                "record_sub_agent_thread failed for %s/%s", project_id, handle,
            )

    async def _inject(self, project_id: str, content: str,
                      *, session_id: str | None = None) -> None:
        """Inject a system message into the management agent's session.

        ``session_id`` carries the management session that owns this sub-agent
        so the push lands under the correct SessionKey (the handle that ran
        the dispatch). Without it, inject_system_message defaults to the
        single-loop sentinel and can miss a non-default management session.
        """
        if self._agent_manager is None:
            return
        try:
            await self._agent_manager.inject_system_message(
                project_id, content, session_id=session_id,
            )
        except Exception as e:
            logger.warning("Failed to inject lifecycle message for %s: %s", project_id, e)
