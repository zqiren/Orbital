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
import threading
from datetime import datetime, timezone

from agent_os.agent.token_utils import estimate_message_tokens
from agent_os.utils.file_lock import FileLock

logger = logging.getLogger(__name__)


def _now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class Session:
    """Append-only conversation log backed by a JSONL file."""

    def __init__(self, filepath: str):
        self._filepath = filepath
        self._messages: list[dict] = []
        self._lock = threading.Lock()
        self._file_lock = FileLock(filepath + ".lock")

        # Session identity (set by new(); derived from filename for load())
        self.session_id: str = os.path.splitext(os.path.basename(filepath))[0]

        # Pending tool call tracking
        self.pending_tool_calls: set[str] = set()

        # Lifecycle flags
        self._paused: bool = False
        self._stopped: bool = False
        self._paused_for_approval: bool = False

        # Message injection queue — each item is (content, nonce)
        self._queue: list[tuple[str, str | None]] = []

        # Observer callbacks (sync only)
        self.on_append = None
        self.on_stream = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def new(cls, session_id: str, workspace: str, project_id: str = "",
            project_dir_name: str = "") -> Session:
        """Create a fresh session.

        File at {workspace}/.agent-os/{project_dir_name}/sessions/{session_id}.jsonl
        when *project_dir_name* is provided (preferred), otherwise falls back to
        *project_id* for backward compat.
        """
        ns = project_dir_name or project_id
        if ns:
            sessions_dir = os.path.join(workspace, ".agent-os", ns, "sessions")
        else:
            sessions_dir = os.path.join(workspace, ".agent-os", "sessions")
        os.makedirs(sessions_dir, exist_ok=True)
        filepath = os.path.join(sessions_dir, f"{session_id}.jsonl")
        session = cls(filepath)
        session.session_id = session_id
        # Create empty file under file lock
        with session._file_lock:
            with open(filepath, "w", encoding="utf-8") as f:
                pass
        return session

    @classmethod
    def load(cls, filepath: str) -> Session:
        """Read existing JSONL, rebuild in-memory state."""
        session = cls(filepath)
        with session._file_lock:
            skipped = 0
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        session._messages.append(msg)
                    except json.JSONDecodeError:
                        skipped += 1
                        logger.warning("Skipped corrupted JSONL line: %s", line[:100])

            if skipped > 0:
                logger.warning("Skipped %d corrupted lines during session load", skipped)

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
            message["session_id"] = self.session_id

        self._messages.append(message)

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
        with self._lock:
            with self._file_lock:
                fd = os.open(self._filepath, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
                try:
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

    def resolve_pending_tool_calls(self) -> None:
        """Inject CANCELLED for all orphaned tool_call IDs."""
        pending = list(self.pending_tool_calls)
        for tc_id in pending:
            self.append_tool_result(tc_id, "CANCELLED: This tool call was not executed.")

    def _heal_orphaned_tool_calls(self) -> None:
        """Insert CANCELLED results adjacent to originating assistant messages.

        Called during load() when pending_tool_calls is non-empty (crash/kill
        scenario). Inserts results immediately after the assistant message and
        any existing sibling tool results, then rewrites the JSONL atomically.
        """
        orphaned = set(self.pending_tool_calls)
        if not orphaned:
            return

        healed_count = 0
        new_messages: list[dict] = []

        i = 0
        while i < len(self._messages):
            msg = self._messages[i]
            new_messages.append(msg)

            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                # Collect tool_call IDs from this assistant message
                tc_ids_in_msg = {
                    tc.get("id", "") for tc in msg["tool_calls"] if tc.get("id")
                }
                orphans_here = orphaned & tc_ids_in_msg

                if orphans_here:
                    # Advance past any existing sibling tool results
                    j = i + 1
                    while j < len(self._messages):
                        next_msg = self._messages[j]
                        if next_msg.get("role") == "tool" and next_msg.get("tool_call_id") in tc_ids_in_msg:
                            new_messages.append(next_msg)
                            j += 1
                        else:
                            break

                    # Insert CANCELLED for orphaned IDs
                    for tc_id in orphans_here:
                        cancel_msg = {
                            "role": "tool",
                            "content": "CANCELLED: This tool call was not executed (session interrupted).",
                            "tool_call_id": tc_id,
                            "source": "management",
                            "timestamp": _now(),
                        }
                        new_messages.append(cancel_msg)
                        healed_count += 1

                    orphaned -= orphans_here
                    i = j
                    continue

            i += 1

        if healed_count == 0:
            return

        # Atomic JSONL rewrite (same pattern as compaction)
        tmp_path = self._filepath + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for msg in new_messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self._filepath)
        self._messages = new_messages
        self.pending_tool_calls.clear()

        logger.warning(
            "Session %s: healed %d orphaned tool calls from interrupted session",
            self.session_id, healed_count,
        )

    # ------------------------------------------------------------------
    # Streaming observer
    # ------------------------------------------------------------------

    def notify_stream(self, chunk) -> None:
        """If on_stream is set, call it with chunk."""
        if self.on_stream is not None:
            self.on_stream(chunk)

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
                with open(tmp_path, "w", encoding="utf-8") as f:
                    for msg in self._messages:
                        f.write(json.dumps(msg, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self._filepath)

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
                with open(tmp_path, "w", encoding="utf-8") as f:
                    for msg in new_messages:
                        f.write(json.dumps(msg, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self._filepath)
        self._messages = new_messages
