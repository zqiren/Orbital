# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Conversation session with JSONL persistence.

Owned by Component A. The session is an append-only log of all messages,
persisted to a JSONL file. Provides sliding window over recent messages.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone

from agent_os.agent.token_utils import estimate_message_tokens
from agent_os.utils.file_lock import session_lock

logger = logging.getLogger(__name__)


def _now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


_NAME_MAX_LEN = 50

# Machine prefixes stamped into first user messages, which must never become
# the display name. Shapes mirror their producers — keep in sync:
#   - <attached_files> block: agent_os/api/routes/_attachment_formatter.py
#   - queue wrapper: agent_os/queue/dispatcher.py (bracket header + the
#     HEADER_CONTRACT paragraph; the START/END markers below are that
#     contract's first and last sentences)
#   - the frontend classifier reads the same shapes: web/src/lib/sessionLabel.ts
# The trigger prefix ("[Triggered by …") is deliberately NOT stripped — the
# frontend extracts the trigger's own name from it.
_ATTACHED_BLOCK_RE = re.compile(r"^<attached_files>\n([\s\S]*?)\n</attached_files>\n\n?")
_QUEUE_BRACKET_RE = re.compile(r"^\[QUEUE ITEM \|[^\]\n]*\]\n")
_QUEUE_CONTRACT_START = "You are working on a queue item."
_QUEUE_CONTRACT_END = "Do not end with plain text.\n"
# Chat-composer uploads are timestamp-prefixed (web/src/lib/attachment-upload.ts).
_UPLOAD_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{6}-")


def _strip_machine_prefixes(text: str) -> tuple[str, str | None]:
    """Strip leading machine blocks from a first user message.

    Returns ``(remainder, attachment_fallback)`` where the fallback is the
    first attached file's basename (upload timestamp stripped) — used as the
    name when nothing human-typed remains. Dispatcher order for queue items
    with staged files is attach block first, then the queue wrapper.
    """
    fallback = None
    m = _ATTACHED_BLOCK_RE.match(text)
    if m:
        for line in m.group(1).split("\n"):
            line = line.strip()
            if not line.startswith("- "):
                continue
            path = line[2:]
            paren = path.rfind(" (")
            if paren > 0:
                path = path[:paren]
            base = _UPLOAD_TS_RE.sub("", path.rsplit("/", 1)[-1]).strip()
            if base:
                fallback = base
            break
        text = text[m.end():]
    b = _QUEUE_BRACKET_RE.match(text)
    if b:
        text = text[b.end():]
        # The header contract follows the bracket line. If its wording ever
        # drifts and the markers stop matching, we degrade to naming from the
        # contract head (same visibility as before this fix), never crash.
        if text.startswith(_QUEUE_CONTRACT_START):
            end = text.find(_QUEUE_CONTRACT_END)
            if end >= 0:
                text = text[end + len(_QUEUE_CONTRACT_END):]
    return text, fallback


def is_machine_derived_name(name) -> bool:
    """True when a stored display name is machine markup, not user text.

    Pre-fix sessions auto-named from the RAW first message stamped names like
    ``"<attached_files>\\n-…"`` or ``"[QUEUE ITEM | id=…"`` into their meta.
    Readers ignore such stored names and re-derive from the first user message
    instead (display-only; the file is not rewritten). A user rename could
    theoretically collide with these prefixes and be overridden — accepted.
    """
    if not isinstance(name, str):
        return False
    return name.startswith("<attached_files>") or name.startswith("[QUEUE ITEM")


def _derive_name(content) -> str | None:
    """Derive a session display name from a user message's content.

    Machine prefixes (attachment block, queue wrapper) are stripped first so
    the name is the user's actual text; an attachment-only message falls back
    to the file's basename. Truncates to ``_NAME_MAX_LEN`` characters,
    word-boundary-aware, appending an ellipsis (``…``) when truncated.
    Returns ``None`` for content that is not a non-empty string (e.g.
    multimodal list content with no usable text).
    """
    if not isinstance(content, str):
        return None
    text, attachment_fallback = _strip_machine_prefixes(content)
    text = text.strip() or (attachment_fallback or "")
    if not text:
        return None
    if len(text) <= _NAME_MAX_LEN:
        return text
    # Truncate to the budget, then back up to the last word boundary so we
    # don't cut mid-word. Fall back to a hard cut if the first word is longer
    # than the budget.
    head = text[:_NAME_MAX_LEN]
    cut = head.rfind(" ")
    if cut > 0:
        head = head[:cut]
    head = head.rstrip()
    return head + "…"


def _splice_pending_cancellations(messages: list, pending_ids: set, content: str, now: str):
    """Place CANCELLED tool-result rows contiguously after each owning assistant.

    For every assistant ``tool_calls`` message owning one or more ``pending_ids``,
    insert a CANCELLED tool-result row for those ids immediately after the
    assistant's existing contiguous tool results — i.e. *before* any interleaved
    non-tool row (a lifecycle ``system`` echo, a ``meta`` resume marker, …).

    Operates on the FULL message list (meta rows included) so nothing is dropped.
    Pure reorder/insert: existing rows keep their order, cancellations are added
    in tool_calls *declaration* order (deterministic), and only ids that have an
    owning assistant here are inserted. Returns ``(new_messages, inserted)`` where
    ``inserted`` is the list of newly created cancellation dicts.
    """
    orphaned = set(pending_ids)
    if not orphaned:
        return messages, []
    inserted: list[dict] = []
    out: list[dict] = []
    n = len(messages)
    i = 0
    while i < n:
        msg = messages[i]
        out.append(msg)
        tcs = msg.get("tool_calls") if msg.get("role") == "assistant" else None
        if tcs:
            tc_ids = {tc.get("id", "") for tc in tcs if tc.get("id")}
            here = orphaned & tc_ids
            if here:
                # Carry over already-present sibling tool results, contiguous.
                j = i + 1
                while (j < n and messages[j].get("role") == "tool"
                       and messages[j].get("tool_call_id") in tc_ids):
                    out.append(messages[j])
                    j += 1
                for tc in tcs:
                    tid = tc.get("id", "")
                    if tid in here:
                        row = {
                            "role": "tool",
                            "content": content,
                            "tool_call_id": tid,
                            "source": "management",
                            "timestamp": now,
                        }
                        out.append(row)
                        inserted.append(row)
                orphaned -= here
                i = j
                continue
        i += 1
    return out, inserted


class Session:
    """Append-only conversation log backed by a JSONL file.

    Identity carries two values (see ``TASK/ACTIVE-session-and-queue-model.md``
    and ``TASK/INVESTIGATION-session-id-canonical-audit.md``):

    - ``session_uuid``: Format 2. The JSONL filename stem
      (``{sanitized_project_name}_{uuid4().hex[:8]}``). Used for file paths,
      ``tool-results/`` directories, and idempotency keys for the session-end
      routine. Always distinct across sessions, never user-facing.

    - ``session_id``: Format 1. The user-facing chat-thread identity (e.g.
      ``"default"``, ``"sess_research_openhuman"``). Stamped onto every message
      via ``append()``. Equality of this value is what the frontend uses to
      group messages into a chat thread and to draw session-boundary
      separators (``web/src/utils/chatTransform.ts``).

    For legacy JSONLs loaded via ``Session.load()`` there is no F1 in the
    file — the ``session_id`` field falls back to the filename stem (F2). This
    is the Option B compat layer per the F7 audit's Q5. Old messages already
    carry their F2 in their ``session_id`` field, and the frontend treats that
    field as an opaque equality key regardless of format.
    """

    def __init__(self, filepath: str):
        self._filepath = filepath
        self._messages: list[dict] = []
        self._lock = threading.Lock()
        # Strategy is selected by AGENT_OS_USE_SENTINEL_LOCK; default is
        # DirectFileLock on the JSONL data file itself (no `.lock` sentinel
        # artifact on disk). See agent_os/utils/file_lock.py.
        self._file_lock = session_lock(filepath)

        # Session identity. ``session_uuid`` is the JSONL filename stem (F2);
        # ``session_id`` is the user-facing chat id (F1) and is set explicitly
        # by ``Session.new``. For legacy sessions loaded via ``Session.load()``
        # there is no F1 in the JSONL, so we default ``session_id`` to the F2
        # stem as a compat fallback. The frontend treats it as an opaque
        # equality key either way.
        self.session_uuid: str = os.path.splitext(os.path.basename(filepath))[0]
        self.session_id: str = self.session_uuid
        # Seam 3 / Phase 4: how this session was born — "chat" (user-initiated)
        # or "queue" (minted by the QueueDispatcher per attempt). Set by new()
        # and recovered by load() from the session_start meta; legacy logs with
        # no origin meta default to "chat".
        self.origin: str = "chat"

        # Pending tool call tracking
        self.pending_tool_calls: set[str] = set()

        # Lifecycle flags
        self._paused: bool = False
        self._stopped: bool = False
        self._paused_for_approval: bool = False

        # Message injection queue — each item is (content, nonce)
        self._queue: list[tuple[str, str | None]] = []

        # Deferred messages (lifecycle notifications buffered during tool execution)
        self._deferred_messages: list[dict] = []

        # Observer callbacks (sync only)
        self.on_append = None
        self.on_stream = None
        # Fired by the loop's mid-run queue drain with the list of non-empty
        # nonces just appended, so the manager can clear the FE "Waiting…"
        # lines (mirrors on_append/on_stream; wired at agent_manager alongside
        # them). Fault-isolated at the call site; None → no-op.
        self.on_queue_drained = None

        # Deferred session_start meta record. A session is a file on disk; the
        # file is not created until the first message. ``Session.new`` stashes
        # the session_start meta here, and the first physical write flushes it
        # as the first line (see ``_write_line``). None for loaded sessions.
        self._pending_meta: dict | None = None

        # Human-readable display label (TASK-session-naming-and-deletion.md).
        # Auto-derived from the first ``role: user`` message (truncated) and
        # user-editable via ``set_name``. Stored on the ``session_start`` meta
        # record so it lives in the single JSONL with no sidecar. ``None`` until
        # a name exists (e.g. headless sessions with no user message — DATA-1).
        # Display-only: never an identifier, never used for routing/lookup.
        self.name: str | None = None

        # Pinned-to-top flag (BACKLOG spec 067). Rides the SAME session_start
        # meta record as ``name`` — per-session, no sidecar, and carried
        # verbatim through compaction/stub-truncation by
        # ``_collect_meta_lines``. Display/ordering only: never an identifier,
        # never used for routing or lookup. Absent on every pre-067 log, which
        # reads as False — so there is no migration.
        self.pinned: bool = False

        # Pinned sub-agent target (BACKLOG spec 074). The slug of the worker
        # this chat session is pinned to (composer "Talking to" dropdown), or
        # None when the session talks to the management agent. Same
        # session_start-meta persistence as ``name``/``pinned``; absent on
        # every pre-074 log, which reads as unpinned — no migration.
        self.pinned_target: str | None = None

        # Sub-agent thread registry (TASK-resume-persistence, piece 2).
        # handle -> {"session_id", "model", "last_used_at"}: the resume
        # identity of each sub-agent thread this session owns. The composite
        # key is (SessionKey, handle): this JSONL is SessionKey-scoped, the
        # map is per-handle. Persisted as ``event: sub_agent_thread`` meta
        # rows (last row wins on load) — same single-JSONL, no-sidecar rule
        # as ``name``. Governance settings are deliberately NOT here; they
        # resolve live at dispatch.
        self.sub_agent_threads: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def new(
        cls,
        session_uuid: str,
        workspace: str,
        project_id: str = "",
        *,
        session_id: str | None = None,
        provider: str = "unknown",
        model: str = "unknown",
        sdk: str = "unknown",
        fallback_models: list[str] | None = None,
        origin: str = "chat",
    ) -> Session:
        """Create a fresh session.

        File at ``{workspace}/orbital/sessions/{session_uuid}.jsonl``.

        ``session_uuid`` is the Format-2 JSONL stem
        (``{sanitized_project_name}_{uuid4().hex[:8]}``). ``session_id`` is the
        Format-1 user-facing chat id; if omitted it defaults to ``session_uuid``
        (mirrors legacy behaviour where the two values were one and the same).

        Records identity (provider/model/sdk/fallback_models) as a
        ``role: meta`` line written as the very first line of the JSONL so
        post-mortem investigations can fingerprint which adapter served the
        session without inferring from error strings.
        """
        from agent_os.agent.project_paths import ProjectPaths
        sessions_dir = ProjectPaths(workspace).sessions_dir
        os.makedirs(sessions_dir, exist_ok=True)
        filepath = os.path.join(sessions_dir, f"{session_uuid}.jsonl")
        session = cls(filepath)
        session.session_uuid = session_uuid
        # Honour explicit F1 from caller; otherwise default to the F2 stem so
        # existing callers (and legacy tests) keep working unchanged. The
        # frontend treats the field as an opaque equality key in either form.
        session.session_id = session_id if session_id is not None else session_uuid
        session.origin = origin
        meta_record = {
            "role": "meta",
            "event": "session_start",
            "session_id": session.session_id,
            "session_uuid": session.session_uuid,
            "origin": origin,
            "provider": provider,
            "model": model,
            "sdk": sdk,
            "fallback_models": list(fallback_models) if fallback_models else [],
            "timestamp": _now(),
        }
        # Deferred creation: do NOT write the file here. A session is a file on
        # disk only once it has a first message — minting a Session object that
        # is never messaged must leave no on-disk trace. The session_start meta
        # is stashed and flushed as the first physical line by the first write
        # (see ``_write_line``). The meta record is identity metadata, not
        # conversation, so it is never added to self._messages and never
        # reaches the LLM via get_messages()/get_recent().
        session._pending_meta = meta_record
        return session

    @classmethod
    def load(cls, filepath: str) -> Session:
        """Read existing JSONL, rebuild in-memory state."""
        session = cls(filepath)
        with session._file_lock:
            skipped = 0
            stored_name: str | None = None
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        logger.warning("Skipped corrupted JSONL line: %s", line[:100])
                        continue
                    # Meta records (session_start, model_swap, …) are identity
                    # metadata, not conversation — keep them out of in-memory
                    # message state so the sliding window never surfaces them
                    # to the LLM.
                    if msg.get("role") == "meta":
                        # The session_start meta carries the display name when
                        # it was set (auto or via rename). Capture it for the
                        # name backfill below.
                        if msg.get("event") == "session_start":
                            if msg.get("name") is not None:
                                stored_name = msg["name"]
                            # Origin recovered from meta; legacy logs predating
                            # the field keep the "chat" default set in __init__.
                            if msg.get("origin"):
                                session.origin = msg["origin"]
                            # Pin state (spec 067). Absent on every pre-067
                            # log → the False default in __init__ stands, so
                            # unpinned is the no-migration default.
                            if msg.get("pinned") is not None:
                                session.pinned = bool(msg["pinned"])
                            # Pinned worker target (spec 074). Same
                            # no-migration default: absent/null reads as None.
                            if msg.get("pinned_target"):
                                session.pinned_target = msg["pinned_target"]
                        elif msg.get("event") == "sub_agent_thread":
                            # Resume identity rows: append-only, last row per
                            # handle wins (TASK-resume-persistence).
                            if msg.get("handle") and isinstance(
                                    msg.get("thread"), dict):
                                session.sub_agent_threads[msg["handle"]] = (
                                    dict(msg["thread"])
                                )
                        continue
                    session._messages.append(msg)

            if skipped > 0:
                logger.warning("Skipped %d corrupted lines during session load", skipped)

            # Name resolution (display label):
            #   1. Stored name on the session_start meta wins (explicit rename
            #      or a previously persisted auto-name) — UNLESS it is machine
            #      markup stamped by the pre-strip auto-namer ("[QUEUE ITEM…",
            #      "<attached_files>…"), which is ignored and re-derived so
            #      legacy sessions heal without a migration.
            #   2. Otherwise derive lazily from the first user message — legacy
            #      sessions and meta-records-without-name. This is IN-MEMORY
            #      ONLY: we do NOT rewrite the JSONL on load (avoid touching
            #      every legacy file). It persists on the next append/rename.
            #   3. Else None (headless / no user message — DATA-1).
            if stored_name is not None and not is_machine_derived_name(stored_name):
                session.name = stored_name
            else:
                for m in session._messages:
                    if m.get("role") == "user":
                        derived = _derive_name(m.get("content"))
                        if derived is not None:
                            session.name = derived
                        break

            # Rebuild pending_tool_calls
            all_tool_call_ids: set[str] = set()
            resolved_ids: set[str] = set()
            for msg in session._messages:
                if msg.get("role") == "assistant" and "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        tc_id = tc.get("id", "")
                        if tc_id:
                            all_tool_call_ids.add(tc_id)
                elif msg.get("role") == "tool" and "tool_call_id" in msg:
                    resolved_ids.add(msg["tool_call_id"])
            session.pending_tool_calls = all_tool_call_ids - resolved_ids

            # Startup recovery: heal orphaned tool calls from interrupted sessions
            if session.pending_tool_calls:
                session._heal_orphaned_tool_calls()

        return session

    # ------------------------------------------------------------------
    # Message operations
    # ------------------------------------------------------------------

    def append(self, message: dict) -> None:
        """Add message to session. Writes to JSONL, tracks tool_calls, fires callback."""
        if "timestamp" not in message:
            message["timestamp"] = _now()

        if "session_id" not in message:
            # F1 user-facing chat id — read by the frontend at
            # ``web/src/utils/chatTransform.ts`` as an opaque equality key.
            message["session_id"] = self.session_id
        if "session_uuid" not in message:
            # F2 storage stem — debug-only field, never read by the frontend
            # for UI logic. Stamped for forensic clarity on the JSONL.
            message["session_uuid"] = self.session_uuid

        self._messages.append(message)

        # Auto-name on the FIRST user message. The name is a display label
        # derived by truncation (see ``_derive_name``); it is stored on the
        # session_start meta record so it persists with the JSONL. Only the
        # first user message sets it — later messages and renames don't.
        if self.name is None and message.get("role") == "user":
            derived = _derive_name(message.get("content"))
            if derived is not None:
                self.name = derived
                self._apply_meta_fields(name=derived)

        # Track tool_call IDs from assistant messages
        if message.get("role") == "assistant" and "tool_calls" in message:
            for tc in message["tool_calls"]:
                tc_id = tc.get("id", "")
                if tc_id:
                    self.pending_tool_calls.add(tc_id)

        # Fire observer BEFORE write so it can enrich the message (e.g., attach descriptions)
        if self.on_append is not None:
            try:
                self.on_append(message)
            except Exception:
                logger.exception("on_append callback failed")

        # Thread-safe + cross-process JSONL write (now includes any fields added by observer)
        line_bytes = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        self._write_line(line_bytes)

    def _write_line(self, line_bytes: bytes) -> None:
        """Append one physical JSONL line under both locks.

        Deferred creation: if a session_start meta record is still pending
        (set by ``Session.new`` and not yet flushed), write it as the very
        first line of the file before this line. The file is created here by
        ``O_CREAT`` — it does not exist until the first write.
        """
        with self._lock:
            with self._file_lock:
                fd = os.open(self._filepath, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
                try:
                    if self._pending_meta is not None:
                        meta_bytes = (
                            json.dumps(self._pending_meta, ensure_ascii=False) + "\n"
                        ).encode("utf-8")
                        os.write(fd, meta_bytes)
                        self._pending_meta = None
                    os.write(fd, line_bytes)
                finally:
                    os.close(fd)

    def append_tool_result(
        self,
        tool_call_id: str,
        result: str | list,
        meta: dict | None = None,
        context_limit: int | None = None,
    ) -> None:
        """Build and append a tool result message. Removes from pending set.

        If context_limit is provided and result is a string, caps the result
        at 30% of the context window. List content (multimodal) is not capped.
        """
        if context_limit is not None and isinstance(result, str):
            result = self._cap_tool_result(result, context_limit)
        msg: dict = {
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call_id,
            "source": "management",
            "timestamp": _now(),
        }
        if meta is not None:
            msg["_meta"] = meta
        self.pending_tool_calls.discard(tool_call_id)
        self.append(msg)

    @staticmethod
    def _cap_tool_result(content: str, context_limit: int) -> str:
        """Truncate tool result if it exceeds 30% of the context window.

        - cap_chars: 30% of tokens * ~4 chars/token
        - hard_max: 400,000 chars absolute ceiling
        - min_preserve: 200 chars minimum (never truncate below this)
        - Truncates at last newline boundary within the cap
        """
        cap_chars = int(context_limit * 0.30 * 4)
        hard_max = 400_000
        min_preserve = 200

        cap = min(cap_chars, hard_max)
        cap = max(cap, min_preserve)

        if len(content) <= cap:
            return content

        # Find last newline within cap to avoid cutting mid-line
        truncation_point = content.rfind("\n", 0, cap)
        if truncation_point < min_preserve:
            truncation_point = cap

        omitted = len(content) - truncation_point
        return content[:truncation_point] + f"\n[truncated — {omitted} chars omitted]"

    def append_system(self, content: str) -> None:
        """Append a system message."""
        msg = {
            "role": "system",
            "content": content,
            "source": "management",
            "timestamp": _now(),
        }
        self.append(msg)

    def append_meta(self, event: str, **fields) -> None:
        """Append a ``role: meta`` lifecycle record to the JSONL.

        Used for events that describe the session's identity rather than its
        conversation — currently ``session_start`` (written by ``new()``) and
        ``model_swap`` (written by the agent loop on fallback rotation).

        Meta records are intentionally NOT added to ``self._messages`` so they
        never appear in ``get_messages()``/``get_recent()`` and therefore never
        reach the LLM. They are forensic-only: ``jq '.role == "meta"'`` over
        the JSONL is the canonical reader.
        """
        record = {
            "role": "meta",
            "event": event,
            "timestamp": _now(),
            **fields,
        }
        line_bytes = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        self._write_line(line_bytes)

    # ------------------------------------------------------------------
    # Sub-agent thread registry (resume persistence)
    # ------------------------------------------------------------------

    def set_sub_agent_thread(self, handle: str, *, session_id: str,
                             model: str | None = None,
                             background_loss: bool = False,
                             proc_pid: int | None = None,
                             proc_create_time: float | None = None,
                             rollout_path: str | None = None) -> None:
        """Record (or refresh) the resume identity of a sub-agent thread.

        Appends an ``event: sub_agent_thread`` meta row (the persistence) and
        updates the in-memory map. Called on each sub-agent completion that
        carries a thread id, so ``last_used_at`` stays current. The record
        carries thread-identity only — resume ``session_id`` and
        ``model`` (``model`` is display/debug metadata: which model last
        served the thread — NEVER consulted to drive resume; model resolves
        live from config at dispatch; AMENDS piece 2). Governance settings
        resolve live at dispatch and are never snapshotted here.

        ``background_loss`` (Piece 3 Part E): a teardown terminated this
        thread's live background work — the next resume's honesty clause
        briefs the orchestrator about the loss. The flag clears naturally on
        the next completed turn's record overwrite.
        """
        record = {
            "session_id": session_id,
            "model": model,
            "last_used_at": _now(),
        }
        if background_loss:
            record["background_loss"] = True
        if proc_pid is not None:
            # Piece 3 Part F: live-attachment anchor (pid + create_time is
            # PID-reuse-safe) — the resume backstop checks it before
            # attaching to this claude session id.
            record["proc_pid"] = proc_pid
            record["proc_create_time"] = proc_create_time
        if rollout_path is not None:
            # Codex thread (TASK-codex-appserver-transport): exact rollout
            # JSONL path from thread/start|resume — pre-checked before resume.
            record["rollout_path"] = rollout_path
        self.sub_agent_threads[handle] = record
        self.append_meta("sub_agent_thread", handle=handle, thread=record)

    def get_sub_agent_thread(self, handle: str) -> dict | None:
        """Return the persisted resume record for ``handle``, or None."""
        return self.sub_agent_threads.get(handle)

    # ------------------------------------------------------------------
    # Session naming (display label)
    # ------------------------------------------------------------------

    def set_name(self, name: str) -> None:
        """Set the session's display name (user rename) and persist it.

        Updates ``self.name`` in memory and writes the value onto the
        ``session_start`` meta record — the pending meta if the file does not
        exist yet, otherwise an in-place rewrite of the first meta line. The
        name is display-only and never affects routing or hydration.
        """
        self.name = name
        self._apply_meta_fields(name=name)

    def set_pinned(self, pinned: bool) -> None:
        """Pin/unpin this session to the top of the sidebar and persist it.

        Same mechanism as ``set_name`` — in-memory field plus a stamp on the
        ``session_start`` meta — because a pin is exactly the same KIND of
        per-session display state as the display label. Ordering only; it never
        affects routing or hydration.
        """
        self.pinned = bool(pinned)
        self._apply_meta_fields(pinned=self.pinned)

    def set_pinned_target(self, target: str | None) -> None:
        """Pin this chat session to a sub-agent (spec 074) and persist it.

        ``target`` is the worker's registry slug; ``None`` clears the pin
        (back to the management agent). Same session_start-meta mechanism as
        ``set_name``/``set_pinned`` — an explicit ``null`` on the meta line is
        how a cleared pin persists, and it loads back as None.
        """
        self.pinned_target = target if target else None
        self._apply_meta_fields(pinned_target=self.pinned_target)

    def _apply_meta_fields(self, **fields) -> None:
        """Stamp arbitrary fields onto the session_start meta.

        If the file is not yet materialized, the meta is still pending — set the
        fields there so they land when the first write flushes it. Otherwise the
        meta is already the first physical line: rewrite it in place.

        Field-agnostic on purpose: ``name`` and ``pinned`` are two writers on
        ONE line, so giving each its own read-modify-write would let a rename
        and a pin issued close together clobber each other. Routing both through
        here keeps them behind the same ``_lock``/``_file_lock`` pair.
        """
        if self._pending_meta is not None:
            self._pending_meta.update(fields)
            return
        self._rewrite_meta_fields(**fields)

    def _rewrite_meta_fields(self, **fields) -> None:
        """Read-modify-write the session_start meta line to carry ``fields``.

        The JSONL is otherwise append-only; this is the one mutation of an
        existing line. Atomic via tmp-file + ``os.replace`` (same pattern as
        ``_compact``). No-op if the file has no session_start meta line.
        """
        with self._lock:
            with self._file_lock:
                try:
                    with open(self._filepath, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                except OSError:
                    return

                changed = False
                for idx, raw in enumerate(lines):
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    try:
                        rec = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("role") == "meta" and rec.get("event") == "session_start":
                        rec.update(fields)
                        lines[idx] = json.dumps(rec, ensure_ascii=False) + "\n"
                        changed = True
                        break
                if not changed:
                    return

                tmp_path = self._filepath + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self._filepath)

    def get_messages(self) -> list[dict]:
        """Return full in-memory message list."""
        return list(self._messages)

    def get_recent(self, max_tokens: int) -> list[dict]:
        """Return recent messages fitting within token budget (newest first priority).

        Iterates backward, estimates tokens as len(str(msg))/4.
        """
        result = []
        used_tokens = 0
        for msg in reversed(self._messages):
            est = estimate_message_tokens(msg)
            if used_tokens + est > max_tokens:
                break
            result.append(msg)
            used_tokens += est
        result.reverse()
        return result

    def recent_activity(self) -> list[dict]:
        """Return last ~10 messages for approval context display."""
        filtered = [
            msg for msg in self._messages
            if msg.get("role") in ("user", "assistant") and msg.get("content")
        ]
        return filtered[-10:]

    # ------------------------------------------------------------------
    # Queue (user injection while loop runs)
    # ------------------------------------------------------------------

    def queue_message(self, content: str, *, nonce: str | None = None) -> None:
        """Append to internal queue list as (content, nonce) tuple."""
        self._queue.append((content, nonce))

    def pop_queued_messages(self) -> list[tuple[str, str | None]]:
        """Return and clear queue. Each item is (content, nonce)."""
        queued = self._queue[:]
        self._queue.clear()
        return queued

    def remove_queued_message(self, nonce: str) -> str | None:
        """Remove the queued ``(content, nonce)`` tuple matching ``nonce``.

        Returns the removed content, or ``None`` if no entry matched. Supports
        the ↑-dequeue / recall of a same-session queued message (spec 006
        §11d): the FE recalls a still-queued message into the composer and the
        server authoritatively reports whether it removed one.
        """
        for idx, (content, q_nonce) in enumerate(self._queue):
            if q_nonce == nonce:
                del self._queue[idx]
                return content
        return None

    def list_queued(self) -> list[tuple[str, str | None]]:
        """Non-destructive copy of the queue (GET /pending recovery, §12 R3).

        Unlike ``pop_queued_messages`` this does not clear the queue — it is a
        read-only snapshot so a reconnecting client can rebuild the
        same-session waiting line without consuming the pending messages.
        """
        return list(self._queue)

    # ------------------------------------------------------------------
    # Deferred messages (lifecycle notifications during tool execution)
    # ------------------------------------------------------------------

    def defer_message(self, content: str, role: str = "system",
                      source: str = "daemon", meta: dict | None = None) -> None:
        """Queue a message for insertion after the current tool batch completes.

        Used for lifecycle notifications that must not interrupt tool execution."""
        self._deferred_messages.append({
            "role": role,
            "content": content,
            "source": source,
            "timestamp": _now(),
            **({"_meta": meta} if meta else {}),
        })

    def pop_deferred_messages(self) -> list[dict]:
        """Pop all deferred messages. Called by the loop after tool batch completes."""
        msgs = list(self._deferred_messages)
        self._deferred_messages.clear()
        return msgs

    # ------------------------------------------------------------------
    # Lifecycle flags
    # ------------------------------------------------------------------

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False
        # Note: _paused_for_approval is reset by loop.run() after the guard check

    def stop(self) -> None:
        self._stopped = True

    def is_paused(self) -> bool:
        return self._paused

    def is_stopped(self) -> bool:
        return self._stopped

    # ------------------------------------------------------------------
    # Tool result guard
    # ------------------------------------------------------------------

    def has_result_for(self, tool_call_id: str) -> bool:
        """Return True if a tool result already exists for this tool_call_id."""
        return tool_call_id not in self.pending_tool_calls

    def _persist_cancellations_adjacent(self, content: str) -> list[dict]:
        """Insert CANCELLED results for all pending tool calls adjacent to their
        originating assistant's tool block, persisted to disk with meta rows
        preserved. Returns the inserted message dicts ([] if nothing was placed).

        Reads the FULL on-disk content (not ``self._messages``) so meta rows
        (session_start, sub_agent_thread) are never dropped — a meta-stripped
        rewrite was the latent bug here. The caller MUST hold ``self._file_lock``
        (and ``self._lock`` for a mid-session write); ``load()`` already holds the
        file lock when it heals.
        """
        pending = set(self.pending_tool_calls)
        if not pending or not os.path.exists(self._filepath):
            return []
        full: list[dict] = []
        with open(self._filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    full.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipped corrupted JSONL line during cancel-splice")
        # Cancel exactly the ids genuinely unresolved ON DISK (declared by some
        # assistant, no result yet) that we also intend to cancel. Reading truth
        # from `full` rather than trusting self.pending_tool_calls makes this
        # idempotent: it can never double-insert a result for an already-resolved
        # id, nor act on an id with no owning assistant (memory/disk desync).
        declared: set = set()
        resolved: set = set()
        for m in full:
            role = m.get("role")
            if role == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    if tc.get("id"):
                        declared.add(tc["id"])
            elif role == "tool" and m.get("tool_call_id"):
                resolved.add(m["tool_call_id"])
        effective = pending & (declared - resolved)
        if not effective:
            return []
        new_full, inserted = _splice_pending_cancellations(full, effective, content, _now())
        if not inserted:
            return []
        # Atomic JSONL rewrite (same pattern as compaction / heal).
        tmp_path = self._filepath + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for msg in new_full:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self._filepath)
        self._messages = [m for m in new_full if m.get("role") != "meta"]
        self.pending_tool_calls -= {m["tool_call_id"] for m in inserted}
        return inserted

    def resolve_pending_tool_calls(self) -> None:
        """Inject CANCELLED for all orphaned tool_call IDs, placed contiguously
        after their assistant's tool block (so strict providers don't 400 on a
        tool result that doesn't follow its tool_calls) and persisted to disk
        with meta rows preserved.
        """
        if not self.pending_tool_calls:
            return
        with self._lock:
            with self._file_lock:
                inserted = self._persist_cancellations_adjacent(
                    "CANCELLED: This tool call was not executed."
                )
        # WS-activity parity: notify observers of the inserted rows outside the
        # locks (on_append may broadcast). Persisted order is already correct.
        for msg in inserted:
            if self.on_append is not None:
                try:
                    self.on_append(msg)
                except Exception:
                    logger.exception("on_append callback failed")

    def _heal_orphaned_tool_calls(self) -> None:
        """Insert CANCELLED results adjacent to originating assistant messages.

        Called during load() (under the file lock) when pending_tool_calls is
        non-empty (crash/kill scenario). Delegates placement to the shared
        ``_persist_cancellations_adjacent``, which preserves meta rows
        (session_start, sub_agent_thread) — a meta-stripped rewrite was the prior
        bug — and keeps the distinct crash-recovery wording.
        """
        if not self.pending_tool_calls:
            return
        inserted = self._persist_cancellations_adjacent(
            "CANCELLED: This tool call was not executed (session interrupted)."
        )
        if inserted:
            logger.warning(
                "Session %s: healed %d orphaned tool calls from interrupted session",
                self.session_uuid, len(inserted),
            )

    # ------------------------------------------------------------------
    # Streaming observer
    # ------------------------------------------------------------------

    def notify_stream(self, chunk) -> None:
        """If on_stream is set, call it with chunk."""
        if self.on_stream is not None:
            self.on_stream(chunk)

    # ------------------------------------------------------------------
    # Meta record preservation (JSONL rewrite helper)
    # ------------------------------------------------------------------

    def _collect_meta_lines(self) -> list[str]:
        """Raw ``role: meta`` JSONL lines currently on disk (file order),
        prefixed by a still-pending session_start if one exists.

        Rewrite paths (stub truncation, compaction) regenerate the file from
        ``self._messages``, which deliberately excludes meta records — without
        this carry-over the first rewrite erases session identity
        (``session_start``) and the worker tag (``session_kind``), which is
        what leaked fanout worker sessions into the sidebar. Caller must hold
        ``self._lock`` + ``self._file_lock``.
        """
        lines: list[str] = []
        if self._pending_meta is not None:
            lines.append(json.dumps(self._pending_meta, ensure_ascii=False) + "\n")
        try:
            with open(self._filepath, "r", encoding="utf-8") as fh:
                for raw in fh:
                    raw_s = raw.strip()
                    if not raw_s:
                        continue
                    try:
                        rec = json.loads(raw_s)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("role") == "meta":
                        lines.append(raw_s + "\n")
        except OSError:
            pass
        return lines

    # ------------------------------------------------------------------
    # Tool result lifecycle
    # ------------------------------------------------------------------

    def replace_tool_results_with_stubs(self, stubs: dict[str, str]) -> None:
        """Replace tool result content for given call_ids with stub text.

        Marks replaced messages with _stubbed=True so they are not
        re-processed on subsequent iterations.
        Atomic file replacement: write tmp, fsync, os.replace.
        """
        if not stubs:
            return

        changed = False
        for msg in self._messages:
            if msg.get("role") == "tool" and msg.get("tool_call_id") in stubs:
                msg["content"] = stubs[msg["tool_call_id"]]
                msg["_stubbed"] = True
                changed = True

        if not changed:
            return

        # Atomic JSONL rewrite (same pattern as _compact)
        tmp_path = self._filepath + ".tmp"
        with self._lock:
            with self._file_lock:
                meta_lines = self._collect_meta_lines()
                with open(tmp_path, "w", encoding="utf-8") as f:
                    for line in meta_lines:
                        f.write(line)
                    for msg in self._messages:
                        f.write(json.dumps(msg, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self._filepath)
                self._pending_meta = None

    # ------------------------------------------------------------------
    # Compaction support (PRIVATE)
    # ------------------------------------------------------------------

    def _compact(self, summary_message: dict, split_index: int) -> None:
        """Replace messages[0:split_index] with summary_message.

        Atomic file replacement: write tmp, fsync, os.replace.
        """
        new_messages = [summary_message] + self._messages[split_index:]
        tmp_path = self._filepath + ".tmp"
        with self._lock:
            with self._file_lock:
                meta_lines = self._collect_meta_lines()
                with open(tmp_path, "w", encoding="utf-8") as f:
                    for line in meta_lines:
                        f.write(line)
                    for msg in new_messages:
                        f.write(json.dumps(msg, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self._filepath)
                self._pending_meta = None
        self._messages = new_messages


def persist_user_row(session: Session, content: str,
                     nonce: str | None = None) -> dict:
    """THE single writer of an injected user message (bug #59).

    Whoever injects a user message persists it — right there, synchronously,
    before anything that can fail. ``AgentLoop.run()`` deliberately does NOT
    append an initial message: it resumes from a session that already contains
    one. Two writers coordinated by a flag is how double-append ships, so
    there is exactly one, and it is this function.

    Callers today: the auto-start branches of
    ``AgentManager.inject_message`` / ``start_agent``
    (``agent_os/daemon_v2/agent_manager.py``) and
    ``NativeWorkerAdapter.send`` (``agent_os/daemon_v2/native_worker.py``).

    ``session.append`` writes the JSONL line immediately (creating the file
    and flushing the deferred ``session_start`` meta on the first write), so
    the message is durable the moment this returns — not once a loop task
    gets its first execution slice.

    ``nonce`` must ride the row: it is the dedup key the frontend matches its
    optimistic bubble against, and downstream consumers read it off the
    persisted message (``agent/providers/openai_compat.py``). Falsy nonces are
    omitted rather than stored as empty strings, matching the previous
    ``AgentLoop.run()`` behaviour byte for byte.

    Returns the appended message dict (already enriched by ``append`` with
    timestamp / session ids / any ``on_append`` observer fields).
    """
    message: dict = {
        "role": "user",
        "content": content,
        "source": "user",
    }
    if nonce:
        message["nonce"] = nonce
    session.append(message)
    return message
