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
        # Fanout join core (spec 009, Task 2/6): None until app startup wires
        # it post-construction (mirrors SubAgentManager._fanout_registry).
        # When set, every terminal-event method below routes the event
        # through it FIRST — a fanout-member handle's per-worker session
        # injection is absorbed into the group's single join summary instead
        # of firing individually.
        self.fanout_registry = None

    def _absorb_terminal(self, project_id: str, handle: str,
                         session_id: str | None, *, kind: str,
                         summary: str, transcript_path: str) -> bool:
        """Route a terminal event through ``fanout_registry`` if wired.
        Returns True when the caller must skip its own ``_inject`` (the WS
        broadcast still fires unconditionally — the progress card needs it
        regardless of fanout membership)."""
        if self.fanout_registry is None:
            return False
        try:
            return self.fanout_registry.absorb_terminal(
                project_id, handle, session_id, kind=kind, summary=summary,
                transcript_path=transcript_path,
            )
        except Exception:
            logger.exception(
                "fanout_registry.absorb_terminal raised for %s/%s "
                "(kind=%s) — falling back to per-worker injection",
                project_id, handle, kind,
            )
            return False

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
        if summary and summary.strip():
            # The sub-agent's full final message is already shown to the user as
            # its own chat bubble (subagent-last-message-display). Re-summarizing
            # it here would duplicate what the user already sees, so steer the
            # management agent toward verifying / advancing instead of echoing.
            guidance = (
                " The user can already see this summary in chat as the sub-agent's "
                "own message — do NOT repeat or re-summarize it. Verify the work if "
                "needed, then continue or reply only with what's new."
            )
        else:
            # No final message → no bubble was shown to the user, so the agent
            # should report the outcome itself rather than stay silent.
            guidance = (
                " The sub-agent produced no final message, so nothing was shown to "
                "the user — briefly tell the user the outcome yourself."
            )
        content = (
            f"[Sub-agent] {handle} completed. Summary: {summary_text}. "
            f"Transcript: {transcript_path}.{guidance}"
        )
        absorbed = self._absorb_terminal(
            project_id, handle, session_id, kind="completed",
            summary=summary_text, transcript_path=transcript_path,
        )
        if not absorbed:
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
        absorbed = self._absorb_terminal(
            project_id, handle, session_id, kind="error",
            summary=error, transcript_path=transcript_path,
        )
        if not absorbed:
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
        # No transcript_path parameter on this event — "" defers to the
        # task's dispatch-time path already recorded in the fanout group
        # (FanoutRegistry.absorb_terminal keeps the existing value when
        # passed a falsy transcript_path).
        absorbed = self._absorb_terminal(
            project_id, handle, session_id, kind="failed",
            summary=reason, transcript_path="",
        )
        if not absorbed:
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
        # No transcript_path parameter on this event either — "" defers to
        # the task's already-recorded transcript path (see on_failed above).
        absorbed = self._absorb_terminal(
            project_id, handle, session_id, kind="stopped",
            summary=content, transcript_path="",
        )
        if not absorbed:
            await self._inject(project_id, content, session_id=session_id)
        self._ws.broadcast(project_id, {
            "type": "sub_agent.stopped",
            "project_id": project_id,
            "handle": handle,
            "background_terminated": terminated or [],
            "session_id": session_id,
        })

    async def on_turn_interrupted(self, project_id: str, handle: str,
                                  transcript_path: str = "unknown",
                                  *, session_id: str | None = None) -> None:
        """A sub-agent turn ended `interrupted` while the agent stays alive
        (Codex `cancel` approval decision — no teardown ran). The management
        session may be AWAITING the dispatch result; silence here is the
        Piece-3 Part-C silent-hang class. Honest framing: stopped before
        completing — NOT a completion (no result), NOT an error."""
        content = (
            f"[Sub-agent] {handle} was stopped before completing its current "
            f"task (turn interrupted — e.g. an approval denied with stop). "
            f"No result was produced. The agent remains available. "
            f"Transcript: {transcript_path}"
        )
        absorbed = self._absorb_terminal(
            project_id, handle, session_id, kind="interrupted",
            summary="turn interrupted before completion",
            transcript_path=transcript_path,
        )
        if not absorbed:
            await self._inject(project_id, content, session_id=session_id)
        self._ws.broadcast(project_id, {
            "type": "sub_agent.turn_interrupted",
            "project_id": project_id,
            "handle": handle,
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
                               session_id: str | None = None,
                               proc_pid: int | None = None,
                               proc_create_time: float | None = None,
                               rollout_path: str | None = None) -> None:
        """A sub-agent turn completed carrying its resume identity.

        Routes to AgentManager, which persists the ``(SessionKey, handle)``
        record into the management session's meta rows
        (TASK-resume-persistence). ``proc_pid``/``proc_create_time`` (Piece 3
        Part F) anchor the live process so a later resume can detect a
        still-live attachment. Not a user-facing event — no injection,
        no broadcast.
        """
        if self._agent_manager is None:
            return
        try:
            self._agent_manager.record_sub_agent_thread(
                project_id, handle,
                claude_session_id=claude_session_id, model=model,
                session_id=session_id,
                proc_pid=proc_pid, proc_create_time=proc_create_time,
                rollout_path=rollout_path,
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
