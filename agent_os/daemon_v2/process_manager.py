# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sub-agent output -> transcript bridge.

Consumes adapter output streams and writes to per-agent transcript files.
v5: ProcessManager never writes role=agent to the management session.
"""

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProcessManager:
    """Consumes adapter output streams. Writes to sub-agent transcripts (v5)."""

    def __init__(self, ws_manager, activity_translator, lifecycle_observer=None):
        self._ws = ws_manager
        self._activity_translator = activity_translator
        self._lifecycle = lifecycle_observer
        # Piece 3 Part A: provenance-based background-work classification.
        # Fed below from run_in_background tool_use chunks; read by
        # SubAgentManager status and the eviction gate. See
        # agent_os/daemon_v2/background_work.py for the classification rules.
        from agent_os.daemon_v2.background_work import BackgroundWorkRegistry
        self.background_work = BackgroundWorkRegistry()
        # Piece 3 Part E (keying fix): consumer slots are scoped by SESSION,
        # not just (project, handle) — two chat sessions running the same
        # sub-agent handle must not share/clobber each other's consumer task
        # or turn-open flag. Key: "{project_id}:{session_id}:{handle}".
        self._tasks: dict[str, asyncio.Task] = {}
        # same key -> True while a turn has produced activity
        # that no turn-closure has accounted for yet. A stream that ends with
        # an open turn died abnormally (→ on_error); a closed or never-opened
        # turn ending is a clean teardown (→ emit nothing). Closures: a
        # turn_complete chunk, or note_turn_closed() from the blocking-send
        # path (Pipe/ACP, which never emit turn_complete).
        self._turn_open: dict[str, bool] = {}
        # same key -> the SubAgentTranscript for the live consumer, so the
        # blocking-send turn-closer (note_turn_closed) can append the per-turn
        # boundary the consume() turn_complete branch appends for streaming
        # transports. Without it, blocking transports (Pipe/ACP/PTY) leave a
        # boundaryless transcript and the multi-turn split aliases.
        self._transcripts: dict[str, object] = {}
        # same key -> a FIFO queue of dispatch_ids enqueued but not yet
        # consumed by a boundary (TASK-dispatch-id-pairing). set_active_
        # dispatch() appends; the boundary write (consume()'s turn_complete
        # branch, or note_turn_closed for blocking transports) pops the
        # OLDEST entry, so the chat renderer can join a dispatch marker to
        # ITS OWN turn by id instead of by position.
        #
        # A per-key FIFO (not a single last-writer-wins slot) matters
        # because a second dispatch to the same handle can be enqueued
        # before the first turn's boundary has been written (rapid
        # re-dispatch — no busy-guard on the real dispatch path; the
        # transport cancels the prior turn and emits ITS boundary before the
        # next turn's query() runs). Boundaries therefore close in dispatch
        # order, so FIFO restores 1:1 pairing by construction — whichever id
        # was enqueued first is popped by whichever boundary closes first. A
        # single overwritable slot would let the second enqueue clobber the
        # first id before it's consumed, misattributing dispatch 2's id
        # onto dispatch 1's (possibly partial-text) aborted turn and losing
        # dispatch 2's own id entirely (Important review finding).
        self._active_dispatch_id: dict[str, "deque[str]"] = {}
        # Optional orchestration hooks. SubAgentManager wires these at
        # construction time; keeping them optional preserves standalone
        # ProcessManager use in tests and legacy embedders.
        self.on_turn_closed = None
        self.on_permission_request = None

    def set_turn_closed_callback(self, callback) -> None:
        self.on_turn_closed = callback

    def set_permission_request_callback(self, callback) -> None:
        self.on_permission_request = callback

    @staticmethod
    def _key(project_id: str, session_id: "str | None", handle: str) -> str:
        return f"{project_id}:{session_id or ''}:{handle}"

    @staticmethod
    def _append_turn_boundary(transcript, handle: str,
                              dispatch_id: "str | None" = None) -> None:
        """Append the per-turn delimiter the read path splits on.

        Empty content → never renders (chatTransform gates empty agent rows;
        the unfiltered chat merge drops turn_complete rows outright).

        ``dispatch_id``, when given, is stamped onto the boundary row —
        ``read_sub_agent_summary`` surfaces it per-turn so the chat renderer
        can identity-join a dispatch marker to THIS turn (TASK-dispatch-id-
        pairing), rather than pairing markers to turns by position (which
        drifted once a transcript outlived the chat session that started
        it). Omitted (no key at all) when no active dispatch was recorded —
        e.g. legacy data, or a boundary that fires without a fresh dispatch.
        """
        if transcript is not None:
            row = {
                "source": handle,
                "content": "",
                "timestamp": _now(),
                "chunk_type": "turn_complete",
            }
            if dispatch_id:
                row["dispatch_id"] = dispatch_id
            transcript.append(row)

    def set_active_dispatch(self, project_id: str, handle: str,
                            dispatch_id: "str | None",
                            *, session_id: "str | None" = None) -> None:
        """Enqueue the dispatch_id of the turn about to start on this handle.

        Called by ``SubAgentManager.send()`` right before it dispatches, so
        the boundary row this turn closes with can be stamped with the same
        id the message_routed marker carries in its ``_meta``
        (TASK-dispatch-id-pairing). APPENDS to a per-key FIFO rather than
        overwriting a single slot — see the ``_active_dispatch_id`` field
        comment for why a single slot is unsafe under rapid re-dispatch. A
        falsy ``dispatch_id`` is a no-op (nothing to enqueue); use
        ``clear_dispatch`` to remove a specific id that was enqueued but
        will never be consumed.
        """
        if not dispatch_id:
            return
        key = self._key(project_id, session_id, handle)
        self._active_dispatch_id.setdefault(key, deque()).append(dispatch_id)

    def clear_dispatch(self, project_id: str, handle: str, dispatch_id: str,
                       *, session_id: "str | None" = None) -> None:
        """Remove ONE specific, not-yet-consumed dispatch_id from this
        handle's queue.

        Guard against FIFO desync (Important review finding): if a dispatch
        fails after ``set_active_dispatch`` enqueued its id but BEFORE the
        transport ever owned the turn (``transport.dispatch()`` raised, or
        the blocking-send background task failed before producing a
        response), no boundary will EVER pop that id — left in the queue it
        would be handed to the NEXT dispatch's boundary instead of its own,
        permanently misaligning every later pairing on this handle. Removes
        by VALUE (not position), so an older still-pending id ahead of it
        in the queue is untouched. A no-op if the id isn't present (already
        consumed, or never enqueued).
        """
        key = self._key(project_id, session_id, handle)
        dq = self._active_dispatch_id.get(key)
        if dq is None:
            return
        try:
            dq.remove(dispatch_id)
        except ValueError:
            return
        if not dq:
            self._active_dispatch_id.pop(key, None)

    def _pop_active_dispatch(self, key: str) -> "str | None":
        """Pop and return the OLDEST enqueued dispatch_id for ``key``, or
        ``None`` if none is pending. Called at every boundary write so each
        closing turn consumes exactly the id that was enqueued for it, in
        FIFO order (TASK-dispatch-id-pairing)."""
        dq = self._active_dispatch_id.get(key)
        if not dq:
            return None
        dispatch_id = dq.popleft()
        if not dq:
            self._active_dispatch_id.pop(key, None)
        return dispatch_id

    def _matching_keys(self, project_id: str, handle: str) -> list[str]:
        """All consumer keys for (project, handle) across sessions — used by
        legacy session-less callers."""
        prefix = f"{project_id}:"
        suffix = f":{handle}"
        return [k for k in self._tasks
                if k.startswith(prefix) and k.endswith(suffix)]

    async def start(self, project_id: str, handle: str, adapter, transcript=None,
                    *, session_id: str | None = None) -> None:
        """Start background task consuming adapter.read_stream().

        ``session_id`` is the management session that owns this sub-agent. It is
        forwarded to every lifecycle event the consumer fires so the push lands
        under the correct SessionKey. Without it, ``on_completed`` /
        ``on_error`` would default to the single-loop sentinel and silently
        drop on non-default sessions (see investigation
        ``INVESTIGATION-2026-05-28-backend-still-broken.md`` — this is the SDK
        transport completion path the prior fix missed).
        """
        key = self._key(project_id, session_id, handle)
        # Piece 3 Part E: never silently overwrite a still-live consumer's
        # slot (the respawn-over-leak clobber). Cancel-and-await the old one
        # so exactly one consumer owns the key.
        prior = self._tasks.get(key)
        if prior is not None and not prior.done():
            logger.warning(
                "ProcessManager.start: live consumer already registered for "
                "%s — cancelling it before starting a new one", key,
            )
            prior.cancel()
            try:
                await prior
            except (asyncio.CancelledError, Exception):
                pass
            # TASK-dispatch-id-pairing, guard (b): a respawn discards the OLD
            # transport/adapter entirely — any dispatch_id(s) still queued
            # for it will never be popped by a real boundary (that turn's
            # consumer is gone). Fail closed: clear them here rather than
            # risk a stale id leaking into the NEW incarnation's pairing.
            self._active_dispatch_id.pop(key, None)

        async def consume():
            last_response_text = ""
            last_error_text = ""
            try:
                async for chunk in adapter.read_stream():
                    if chunk.chunk_type == "turn_complete":
                        # Route by the transport's honest cause. Only a
                        # verified success may be reported as completed
                        # (TASK-honest-subagent-completion-reporting):
                        #   success → on_completed
                        #   stopped → deliberate teardown, emit nothing
                        #   error / missing → on_error
                        meta = getattr(chunk, "metadata", None) or {}
                        cause = meta.get("cause")
                        # Resume persistence: a completed turn (success OR
                        # error — a failed turn still has a live, resumable
                        # session) carries the thread's resume identity.
                        # Record it before the lifecycle event so the id is
                        # on disk even if injection fails. cause=stopped is
                        # skipped: the id was recorded at the last completed
                        # turn (TASK-resume-persistence).
                        # interrupted: the turn was cancelled but the thread
                        # is alive and resumable.
                        if (self._lifecycle and meta.get("session_id")
                                and cause in ("success", "error", "interrupted")):
                            # Piece 3 Part F: persist the process identity
                            # (pid + create_time) alongside the resume
                            # record, so a later spawn can detect a
                            # still-live attachment and never double-attach.
                            proc = getattr(
                                getattr(adapter, "_transport", None),
                                "_proc", None)
                            try:
                                proc_pid = proc.pid
                                proc_create_time = proc.create_time()
                            except Exception:
                                proc_pid, proc_create_time = None, None
                            await self._lifecycle.on_thread_update(
                                project_id, handle,
                                claude_session_id=meta["session_id"],
                                model=meta.get("model"),
                                session_id=session_id,
                                proc_pid=proc_pid,
                                proc_create_time=proc_create_time,
                                rollout_path=meta.get("rollout_path"),
                            )
                        if self._lifecycle and transcript is not None:
                            if cause == "success":
                                await self._lifecycle.on_completed(
                                    project_id, handle,
                                    summary=last_response_text or "(no output)",
                                    transcript_path=transcript.filepath,
                                    session_id=session_id,
                                )
                            elif cause == "stopped":
                                pass  # clean reap — a kill is not a completion
                            elif cause == "interrupted":
                                # Turn cancelled while the agent lives on
                                # (Codex cancel decision; no teardown event
                                # will speak). An awaiting management session
                                # must be woken honestly — silence here is
                                # the Part-C hang class.
                                await self._lifecycle.on_turn_interrupted(
                                    project_id, handle,
                                    transcript_path=transcript.filepath,
                                    session_id=session_id,
                                )
                            else:
                                await self._lifecycle.on_error(
                                    project_id, handle,
                                    last_error_text
                                    or "turn ended without a completion signal",
                                    transcript.filepath,
                                    session_id=session_id,
                                )
                        # Append a turn boundary so read_sub_agent_summary can
                        # split the transcript per-turn (multi-turn display).
                        # Appended for EVERY terminal cause — including "stopped"
                        # (an SDK per-turn cancel fired when a new dispatch
                        # arrives mid-stream): the stopped turn still has a
                        # preceding message_routed marker, so omitting its
                        # boundary would drift the i-th marker ↔ i-th slice
                        # pairing and alias the next turn's text into this
                        # dispatch's bubble (TASK-subagent-last-message-display).
                        # Consume (pop) the OLDEST queued dispatch_id so it
                        # stamps ONLY this boundary, in FIFO order — a later
                        # turn with no fresh send() call must never inherit
                        # a stale id (TASK-dispatch-id-pairing).
                        self._append_turn_boundary(
                            transcript, handle,
                            self._pop_active_dispatch(key))
                        last_response_text = ""  # reset for next turn
                        last_error_text = ""
                        self._turn_open[key] = False
                        if self.on_turn_closed is not None:
                            await self.on_turn_closed(
                                project_id, handle, session_id=session_id,
                                cause=cause,
                            )
                        continue
                    if chunk.chunk_type == "thread_started":
                        # BACKLOG 005 §4a — eager resume-record persistence.
                        # The transport surfaced the upstream session id at
                        # turn START (SDK init message). Persist it NOW, before
                        # the turn completes, so a concurrent @-mention/dispatch
                        # in the same chat session re-attaches at the provider
                        # level instead of starting cold. The terminal
                        # turn_complete still re-records (refreshing model/pid),
                        # so this is purely additive. A control event — never
                        # written to the transcript, never broadcast.
                        meta = getattr(chunk, "metadata", None) or {}
                        sid = meta.get("session_id")
                        if self._lifecycle and sid:
                            proc = getattr(
                                getattr(adapter, "_transport", None),
                                "_proc", None)
                            try:
                                proc_pid = proc.pid
                                proc_create_time = proc.create_time()
                            except Exception:
                                proc_pid, proc_create_time = None, None
                            await self._lifecycle.on_thread_update(
                                project_id, handle,
                                claude_session_id=sid,
                                model=meta.get("model"),
                                session_id=session_id,
                                proc_pid=proc_pid,
                                proc_create_time=proc_create_time,
                                rollout_path=meta.get("rollout_path"),
                            )
                        continue
                    if chunk.chunk_type == "interaction_required":
                        # A reverse request blocks the current provider turn;
                        # it is control-plane state, not transcript content.
                        # Wake the owning management session so it can answer
                        # through agent_message(respond) on the same request.
                        self._turn_open[key] = True
                        metadata = getattr(chunk, "metadata", None) or {}
                        if self._lifecycle is not None:
                            await self._lifecycle.on_interaction_required(
                                project_id, handle,
                                interaction_id=str(
                                    metadata.get("interaction_id", "")),
                                kind=str(metadata.get("kind", "question")),
                                prompt=str(
                                    metadata.get("prompt")
                                    or metadata.get("question") or ""),
                                options=metadata.get("options"),
                                plan=metadata.get("plan"),
                                session_id=session_id,
                            )
                        continue
                    entry = {
                        "source": handle,
                        "content": chunk.text,
                        "timestamp": _now(),
                        "chunk_type": chunk.chunk_type,
                    }
                    # Write to sub-agent transcript (v5: never to management session)
                    if transcript is not None:
                        transcript.append(entry)

                    # Any non-closure chunk means a turn is in flight; a
                    # stream that ends while one is open died abnormally.
                    self._turn_open[key] = True

                    # Piece 3 Part A — provenance capture: a Bash tool_use
                    # with run_in_background=true IS the registration of real
                    # background work (the only trusted classifier; see
                    # background_work.py). tool_input survives into
                    # chunk.metadata via transport_event_to_chunk
                    # (transports/base.py).
                    if chunk.chunk_type == "tool_activity":
                        meta = getattr(chunk, "metadata", None) or {}
                        tool_input = meta.get("tool_input") or {}
                        if (meta.get("tool_name") == "Bash"
                                and tool_input.get("run_in_background")):
                            root_proc = getattr(
                                getattr(adapter, "_transport", None),
                                "_proc", None,
                            )
                            self.background_work.register(
                                project_id, session_id or "", handle,
                                command=str(tool_input.get("command", "")),
                                root_proc=root_proc,
                            )

                    if chunk.chunk_type == "error":
                        # Keep error text out of last_response_text — it is
                        # not a "summary"; it feeds on_error at turn close.
                        last_error_text = chunk.text

                    # Track last response text for completion summary
                    if chunk.chunk_type in ("response", "message") or chunk.chunk_type is None:
                        last_response_text = chunk.text
                        self._ws.broadcast(project_id, {
                            "type": "chat.sub_agent_message",
                            "project_id": project_id,
                            "session_id": session_id,
                            "content": chunk.text,
                            "source": handle,
                            "timestamp": entry["timestamp"],
                        })

                    if chunk.chunk_type == "approval_request":
                        metadata = getattr(chunk, 'metadata', {}) or {}
                        request_id = str(
                            metadata.get("request_id")
                            or metadata.get("permission_id") or "")
                        handled = False
                        if self.on_permission_request is not None and request_id:
                            handled = await self.on_permission_request(
                                project_id, handle, request_id,
                                session_id=session_id, metadata=metadata,
                            )
                        if not handled:
                            raw_tool_args = metadata.get(
                                "tool_input", metadata.get("tool_args", {}))
                            tool_args = (
                                dict(raw_tool_args)
                                if isinstance(raw_tool_args, dict)
                                else {"input": raw_tool_args}
                            )
                            if metadata.get("options"):
                                tool_args["permission_options"] = metadata["options"]
                            self._ws.broadcast(project_id, {
                            "type": "approval.request",
                            "project_id": project_id,
                            "session_id": session_id,
                            "what": f"Sub-agent {handle} requests approval",
                            "tool_name": metadata.get("tool_name", ""),
                            "tool_call_id": request_id,
                            "tool_args": tool_args,
                            "source": handle,
                            "recent_activity": [],
                            })

                    self._activity_translator.on_message(
                        {"role": "agent", "source": handle, "content": chunk.text, "timestamp": entry["timestamp"]},
                        project_id,
                        session_id=session_id,
                    )

                # Stream ended. With an open turn this is an abnormal death
                # (process died without any completion signal — path c); a
                # closed or never-opened turn is a clean teardown and must
                # emit NOTHING. The old unconditional on_completed here is
                # what stamped "completed" on killed sub-agents.
                if self._turn_open.get(key):
                    # FIFO desync guard, closure requested after re-review
                    # (TASK-dispatch-id-pairing): no turn_complete was ever
                    # emitted for this turn, so no boundary will EVER pop
                    # its queued id. Discard it (no boundary row is written
                    # here either, unchanged) — left queued it would leak
                    # into whatever dispatch runs next on this key.
                    self._pop_active_dispatch(key)
                    if self.on_turn_closed is not None:
                        await self.on_turn_closed(
                            project_id, handle, session_id=session_id,
                            cause="stream_ended",
                        )
                    if self._lifecycle and transcript is not None:
                        await self._lifecycle.on_error(
                            project_id, handle,
                            last_error_text
                            or "stream ended without a completion signal",
                            transcript.filepath,
                            session_id=session_id,
                        )
            except asyncio.CancelledError:
                pass
            except Exception as e:
                # Same FIFO guard: an unexpected exception inside the
                # consumer loop itself means this turn's boundary will
                # never be written either — discard its queued id.
                self._pop_active_dispatch(key)
                if self.on_turn_closed is not None:
                    await self.on_turn_closed(
                        project_id, handle, session_id=session_id,
                        cause="consumer_exception",
                    )
                if self._lifecycle and transcript is not None:
                    await self._lifecycle.on_error(
                        project_id, handle, str(e), transcript.filepath,
                        session_id=session_id,
                    )

        task = asyncio.create_task(consume())
        self._tasks[key] = task
        self._transcripts[key] = transcript

    def note_turn_closed(self, project_id: str, handle: str,
                         session_id: "str | None" = None) -> None:
        """Mark the current turn as accounted for by a lifecycle event.

        Called by the blocking-send path (Pipe/ACP — transports that never
        emit ``turn_complete``) after it fires on_completed/on_error itself,
        so a later clean teardown's stream-end is not misread as an abnormal
        mid-turn death.

        This is the blocking-transport analog of the consume() turn_complete
        branch, so it also appends the per-turn boundary, popping the OLDEST
        queued dispatch_id (TASK-dispatch-id-pairing, FIFO): without a
        boundary a blocking transcript has ZERO of them and the multi-turn
        split collapses to one slice, aliasing the last turn's text onto the
        first dispatch.

        (Minor review finding, resolved: this used to special-case
        ``session_id=None`` as "clear the turn_open flag in every matching
        (project, handle) slot, no boundary" — a legacy fallback from before
        per-session keying (Piece 3 Part E). Confirmed dead: every real
        caller (``SubAgentManager._dispatch_async``'s blocking-send path)
        always passes a concrete ``session_id``, and the fallback's
        "clear every session's flag but write no boundary, pop no id"
        behavior doesn't even have a well-defined dispatch_id/transcript to
        act on when scanning multiple sessions at once. Removed rather than
        patched to "honor the invariant": ``session_id`` (including a bare
        ``None``) now always resolves to its own single key via ``_key()``,
        the same as every other call — one code path, no silent partial
        cleanup.)
        """
        key = self._key(project_id, session_id, handle)
        self._turn_open[key] = False
        self._append_turn_boundary(
            self._transcripts.get(key), handle,
            self._pop_active_dispatch(key))

    async def stop(self, project_id: str, handle: str, *,
                   session_id: "str | None" = None) -> None:
        """Cancel consumer task(s). Exact session slot when ``session_id`` is
        given; all of (project, handle)'s slots otherwise (legacy callers)."""
        if session_id is not None:
            keys = [self._key(project_id, session_id, handle)]
        else:
            keys = self._matching_keys(project_id, handle)
        for key in keys:
            self._turn_open.pop(key, None)
            self._transcripts.pop(key, None)
            self._active_dispatch_id.pop(key, None)
            task = self._tasks.pop(key, None)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
