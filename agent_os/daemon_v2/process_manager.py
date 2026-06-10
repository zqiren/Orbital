# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sub-agent output -> transcript bridge.

Consumes adapter output streams and writes to per-agent transcript files.
v5: ProcessManager never writes role=agent to the management session.
"""

import asyncio
import logging
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

    @staticmethod
    def _key(project_id: str, session_id: "str | None", handle: str) -> str:
        return f"{project_id}:{session_id or ''}:{handle}"

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
                        last_response_text = ""  # reset for next turn
                        last_error_text = ""
                        self._turn_open[key] = False
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
                        self._ws.broadcast(project_id, {
                            "type": "approval.request",
                            "project_id": project_id,
                            "session_id": session_id,
                            "what": f"Sub-agent {handle} requests approval",
                            "tool_name": metadata.get("tool_name", ""),
                            "tool_call_id": metadata.get("request_id", ""),
                            "tool_args": metadata.get("tool_input", {}),
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
                if self._lifecycle and transcript is not None:
                    await self._lifecycle.on_error(
                        project_id, handle, str(e), transcript.filepath,
                        session_id=session_id,
                    )

        task = asyncio.create_task(consume())
        self._tasks[key] = task

    def note_turn_closed(self, project_id: str, handle: str,
                         session_id: "str | None" = None) -> None:
        """Mark the current turn as accounted for by a lifecycle event.

        Called by the blocking-send path (Pipe/ACP — transports that never
        emit ``turn_complete``) after it fires on_completed/on_error itself,
        so a later clean teardown's stream-end is not misread as an abnormal
        mid-turn death. ``session_id=None`` closes the turn flag in every
        session-scoped slot for (project, handle) — legacy back-compat.
        """
        if session_id is not None:
            self._turn_open[self._key(project_id, session_id, handle)] = False
            return
        for k in list(self._turn_open):
            if k.startswith(f"{project_id}:") and k.endswith(f":{handle}"):
                self._turn_open[k] = False

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
            task = self._tasks.pop(key, None)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
