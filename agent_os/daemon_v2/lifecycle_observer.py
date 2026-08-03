# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lifecycle observer — injects system messages on sub-agent state transitions.

Every sub-agent lifecycle event (start, message route, completion, error)
injects a system message into the management agent's session with [Sub-agent]
prefix and transcript path.
"""

import logging
import json

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
                                *, session_id: str | None = None,
                                dispatch_id: str | None = None) -> None:
        """A message was routed to a sub-agent.

        ``dispatch_id`` (TASK-dispatch-id-pairing), when given, is stamped
        onto this marker's ``_meta`` alongside ``handle``/``transcript_path``
        — the identity the chat renderer joins against the transcript's
        matching ``turn_complete`` boundary row, instead of pairing markers
        to turns by position (which drifted once a transcript outlived the
        chat session that started it). The human-readable ``content`` prose
        is unchanged either way — old UIs and the legacy-data migration
        script both still rely on its exact shape. Omitted only by callers
        that predate the id; such a marker renders no bubble on read (its
        one-line text still shows).

        ``initiator == "user_mention"`` used to be a NO-OP (backlog #24 D3):
        the @mention API route fired this notification itself, in ADDITION
        to the one ``SubAgentManager.send()`` already fired internally (via
        ``_dispatch_prompt_locked``) for the very same ``dispatch_id`` — one
        physical dispatch, two markers, so the route's copy was dropped.

        Backlog #23 D3 supersedes that: ``send()`` now threads the caller's
        ``initiator`` all the way through ``_QueuedPrompt`` to this, its one
        internal notification call (immediate dispatch AND a later-drained
        queued one both go through ``_dispatch_prompt_locked`` the same
        way) — the @mention route no longer fires a call of its own at all.
        Since this is now the ONLY marker a mention dispatch ever gets, it
        cannot go back to being silent: the LLM-facing ``content`` carries
        one extra guidance line telling the management agent the user
        addressed the sub-agent directly (do not answer on its behalf;
        supervise/relay instead) — a weak model otherwise papers over the
        silence by answering for the sub-agent. The rendered timeline row
        stays on the same clean "Message sent to …" text a
        "management_agent"-initiated dispatch gets, via the same
        ``_meta.display_content`` split ``on_completed`` uses (backlog #24
        D2) — the guidance is agent-facing only and must never render.
        """
        preview = message_preview[:100]
        display_content = f'[Sub-agent] Message sent to {handle}: "{preview}". Transcript: {transcript_path}'
        if initiator == "user_mention":
            content = display_content + (
                " The user addressed this sub-agent directly — do not "
                "answer on its behalf; supervise or relay the sub-agent's "
                "response instead."
            )
        else:
            content = display_content
        meta: dict = {}
        if dispatch_id:
            meta["dispatch_id"] = dispatch_id
            meta["handle"] = handle
            meta["transcript_path"] = transcript_path
        if content != display_content:
            meta["display_content"] = display_content
        await self._inject(project_id, content, session_id=session_id,
                           meta=meta or None)

    async def on_interaction_required(
        self, project_id: str, handle: str, *, interaction_id: str,
        kind: str, prompt: str, options=None, plan=None,
        session_id: str | None = None,
    ) -> None:
        """A sub-agent is blocked on an answer from its management agent.

        This is intentionally provider-neutral.  The transport retains the
        original in-flight request; Orbital only wakes its owning management
        session and gives it the opaque interaction id needed by
        ``agent_message(action="respond")``.
        """
        detail = prompt or "(no prompt supplied)"
        if options:
            detail += f" Options: {json.dumps(options, ensure_ascii=False)}"
        if plan:
            detail += f" Plan: {json.dumps(plan, ensure_ascii=False)}"
        if plan or kind.endswith("create_plan"):
            response_guidance = (
                "For a plan, pass selection=\"accept\", "
                "selection=\"reject\" (with message only as an optional "
                "rejection reason), or selection=\"cancel\"."
            )
        elif options:
            response_guidance = (
                "Choose the offered option id with selection=\"option-id\"; "
                "for multiple questions use selection={\"question-id\": "
                "[\"option-id\"]}. Use message only when the interaction "
                "explicitly requests free-form text."
            )
        else:
            response_guidance = (
                "For a free-form question, put the answer in message."
            )
        content = (
            f"[Sub-agent] {handle} requires input ({kind}): {detail} "
            f"Respond on the same in-flight request with "
            f"agent_message(action=\"respond\", agent=\"{handle}\", "
            f"interaction_id=\"{interaction_id}\", ...). "
            f"{response_guidance} Do not send a new task."
        )
        meta = {
            "event": "interaction_required",
            "handle": handle,
            "interaction_id": interaction_id,
            "kind": kind,
        }
        await self._inject(
            project_id, content, session_id=session_id, meta=meta)
        self._ws.broadcast(project_id, {
            "type": "sub_agent.interaction_required",
            "project_id": project_id,
            "session_id": session_id,
            "handle": handle,
            "interaction_id": interaction_id,
            "kind": kind,
            "prompt": prompt,
            "options": options,
            "plan": plan,
        })

    async def on_completed(self, project_id: str, handle: str, summary: str,
                           transcript_path: str,
                           *, session_id: str | None = None) -> None:
        """Sub-agent finished its current task."""
        summary_text = summary[:500] if summary else "(no output)"
        # Display split (backend _meta contract, mirrors commit 967237d's
        # fanout join-summary split): display_content is the clean,
        # user-visible marker; the steering guidance below is agent-facing
        # only and must never render in the chat timeline.
        display_content = (
            f"[Sub-agent] {handle} completed. Summary: {summary_text}. "
            f"Transcript: {transcript_path}."
        )
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
        content = display_content + guidance
        absorbed = self._absorb_terminal(
            project_id, handle, session_id, kind="completed",
            summary=summary_text, transcript_path=transcript_path,
        )
        if not absorbed:
            # event/kind ride ALONGSIDE the display split (backlog #28 a): a
            # completion landing mid-turn was appended silently, so the
            # management agent only noticed its sub-agent had finished the
            # next time the user typed. Same wake tag as the negative
            # terminals below — display_content stays untouched.
            await self._inject(
                project_id, content, session_id=session_id,
                meta={"event": "sub_agent_terminal", "kind": "completed",
                      "display_content": display_content},
            )
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
            # meta.kind (backlog #23 D1): lets _on_loop_done recognize a
            # negative-terminal event in the deferred-message drain and wake
            # the management loop instead of silently appending it.
            await self._inject(
                project_id, content, session_id=session_id,
                meta={"event": "sub_agent_terminal", "kind": "error"},
            )
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
            # meta.kind (backlog #23 D1) — see on_error above.
            await self._inject(
                project_id, content, session_id=session_id,
                meta={"event": "sub_agent_terminal", "kind": "failed"},
            )
        self._ws.broadcast(project_id, {
            "type": "sub_agent.failed",
            "project_id": project_id,
            "handle": handle,
            "reason": reason,
            "session_id": session_id,
        })

    async def on_queue_dropped(self, project_id: str, handle: str, *,
                               why: str,
                               session_id: str | None = None) -> None:
        """A queued prompt was discarded before it ever reached the sub-agent.

        Backlog #35a. These rows used to be written through ``on_failed``,
        which was honest in the reason text and wrong in everything around
        it: red failure chrome for what is usually the user's own stop, a
        hardcoded "The dispatched task did not complete." tail claiming a
        dispatch that never happened, and a second colon from nesting one
        reason inside another. Nothing was dispatched and nothing
        malfunctioned — the message simply never left the queue, which is a
        warning, not a failure.

        The reason half is composed by ``SubAgentManager``'s drop sites; the
        parity guard discovers them from source and pins the text this
        produces (``test_every_queue_drop_site_is_marked_and_pinned``).

        Routes through ``_absorb_terminal`` like the other negative terminals
        so a fanout worker's dropped queue still lands in the group's join
        summary rather than a per-worker session it no longer owns; kind maps
        to the same task status ``failed`` did, so the join is unchanged.
        """
        content = f"[Sub-agent] {handle} queued message dropped: {why}."
        absorbed = self._absorb_terminal(
            project_id, handle, session_id, kind="queue_dropped",
            summary=f"queued message dropped: {why}", transcript_path="",
        )
        if not absorbed:
            # Same wake tag the terminal above carries (backlog #23 D1): the
            # management agent may be awaiting this very message, and a drop
            # it never hears about is the silent-hang class again. No WS
            # broadcast — the drop always rides along with an event that
            # already broadcast the handle's new state (stop, send failure,
            # transport end), and the handle's live status is unchanged by
            # it either way.
            await self._inject(
                project_id, content, session_id=session_id,
                meta={"event": "sub_agent_terminal", "kind": "queue_dropped"},
            )

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
            # meta.kind (backlog #23 D1) — see on_error above.
            await self._inject(
                project_id, content, session_id=session_id,
                meta={"event": "sub_agent_terminal", "kind": "stopped"},
            )
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
        (Codex `cancel` approval decision — no teardown ran). Also produced
        by a NativeWorkerAdapter deliberately torn down mid-turn (stop_all /
        user stop — round-3 review IMPORTANT 3): `_background_send` routes
        here instead of `on_error` when the adapter's `_stop_requested` flag
        is set, so a fanout worker stopped on purpose is never reported as
        having failed. The management session may be AWAITING the dispatch
        result; silence here is the Piece-3 Part-C silent-hang class.
        Honest framing: stopped before completing — NOT a completion (no
        result), NOT an error."""
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
            # Wake tag (backlog #28 b): the docstring above names the silence
            # here the Piece-3 Part-C silent-hang class, and a mid-turn
            # interruption deferred into the management session was exactly
            # that — appended, never processed. kind matches the
            # `_absorb_terminal` call above.
            await self._inject(
                project_id, content, session_id=session_id,
                meta={"event": "sub_agent_terminal", "kind": "interrupted"},
            )
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
                      *, session_id: str | None = None,
                      meta: dict | None = None) -> None:
        """Inject a system message into the management agent's session.

        ``session_id`` carries the management session that owns this sub-agent
        so the push lands under the correct SessionKey (the handle that ran
        the dispatch). Without it, inject_system_message defaults to the
        single-loop sentinel and can miss a non-default management session.

        ``meta``, when given, is forwarded to ``inject_system_message`` as
        structured ``_meta`` on the appended record (e.g. the dispatch_id
        join key — TASK-dispatch-id-pairing). Omitted entirely (not passed
        as an explicit ``None``) when absent, so callers with a narrower
        ``inject_system_message`` signature are unaffected.
        """
        if self._agent_manager is None:
            return
        try:
            kwargs = {"session_id": session_id}
            if meta:
                kwargs["meta"] = meta
            await self._agent_manager.inject_system_message(project_id, content, **kwargs)
        except Exception as e:
            logger.warning("Failed to inject lifecycle message for %s: %s", project_id, e)
