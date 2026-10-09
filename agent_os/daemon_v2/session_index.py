# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rebuildable per-project index of the session list (spec 066 phase 2).

The session JSONL files stay the file of record. This index only remembers,
for each file, the row the session list derives from it, keyed by the file's
mtime and size, so drawing the list does not open and parse every session log
again after each daemon restart (the in-memory cache it complements is cold on
every start). It lives next to the logs:

    {workspace}/orbital/sessions/.session-index.sqlite

and is disposable by design: delete it, or let it get corrupted, and the next
start rebuilds it from the files and produces the identical list. A row whose
file changed is re-derived from the file and written through. Nothing about a
session is stored ONLY here.

``derive_disk_entry`` is the single derivation of a list row from a log file;
the pre-index scan and the index build both call it, which is what makes the
two paths produce byte-identical lists.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading

logger = logging.getLogger(__name__)

INDEX_FILENAME = ".session-index.sqlite"

# Bump whenever ``derive_disk_entry``'s output changes (a new field, a new
# naming rule): an index written under another version is emptied and rebuilt
# from the files, so it can never serve rows in an old shape.
#   "2": ``last_user_at`` / ``last_reply_at`` (spec 108).
INDEX_VERSION = "2"

# ``_meta.event`` values of the system rows that answer the user (spec 108):
# a worker's terminal marker (completed / error / failed / stopped /
# interrupted / queue_dropped, lifecycle_observer) and a worker asking a
# question (``interaction_required`` — the user has to act). The dispatch ack
# ("Message sent to …") and "… started" rows carry no ``event`` and are not
# replies. Keep in step with ``lifecycle_observer.py``.
REPLY_META_EVENTS = frozenset({"sub_agent_terminal", "interaction_required"})


def index_path(sessions_dir: str) -> str:
    return os.path.join(sessions_dir, INDEX_FILENAME)


def read_session_f1(path: str) -> str | None:
    """Head-only read of a session log's original F1 ``session_id``.

    The ``session_start`` meta carries it and every non-meta record stamps it
    too, so the first record with a ``session_id`` field IS the F1 — reading
    past it never changes the answer. Stops there; never parses the rest of
    the file (spec 107: the chat route's fallback used to, which made landing
    on a freshly minted id cost O(total session bytes) — 4 s on a 225 MB
    project). Torn lines before the first match are skipped. ``None`` when
    the file cannot be read or carries no ``session_id`` at all.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                sid = rec.get("session_id")
                if sid:
                    return sid
    except OSError:
        return None
    return None


def resolve_session_uuid(sessions_dir: str, identifier: str) -> str | None:
    """Resolve an F1 or F2 identifier to its JSONL stem (the F2 uuid), or None.

    The one resolver behind the chat read path (``_find_session_uuid_on_disk``)
    and hydrate-on-inject (``_load_session_from_disk``), so both answer the
    same question the same way:

    * ``{identifier}.jsonl`` exists → ``identifier`` (uuid addressing — how the
      sidebar names disk-only sessions; the common case, one ``stat``).
    * otherwise ``identifier`` is a legacy F1 chat id (``sess_*``/``default``)
      → scan every log by its HEAD only (``read_session_f1``) and return the
      stem of the newest-mtime match. F1 is not unique across rotated legacy
      logs, so "newest wins" is the tie-break; the first-in-listdir-order
      answer the chat route used to give was arbitrary on APFS.

    Cost is O(files), never O(bytes): a freshly minted id (no file yet) reads
    one record per log. Blocking disk I/O — call off the event loop.
    """
    if not os.path.isdir(sessions_dir):
        return None
    if os.path.isfile(os.path.join(sessions_dir, f"{identifier}.jsonl")):
        return identifier
    best: str | None = None
    best_mtime = -1.0
    for fname in os.listdir(sessions_dir):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(sessions_dir, fname)
        if read_session_f1(fpath) != identifier:
            continue
        try:
            mtime = os.path.getmtime(fpath)
        except OSError:
            continue
        if mtime > best_mtime:
            best_mtime = mtime
            best = fname[:-6]  # strip .jsonl
    return best


def is_reply_row(rec: dict) -> bool:
    """A row that answers the user: the agent's final text, a worker's
    terminal marker, a worker question, a queue signal, or an LLM error.
    Never the user's own row, a mid-turn tool step, or a dispatch ack.

    The session list exposes the timestamp of the last such row as
    ``last_reply_at`` (spec 108): "the agent finished / needs you" after the
    user's last message. A row shape this does not know is not a reply —
    a missed badge, never a false one.
    """
    role = rec.get("role")
    if role == "assistant":
        return not rec.get("tool_calls")  # the final answer, not a tool step
    if role == "system":
        meta = rec.get("_meta") or {}
        if isinstance(meta, dict) and meta.get("event") in REPLY_META_EVENTS:
            return True
        # Task signal ("Task completed" / "Task blocked") or a management
        # notice that ends the turn (LLM error, cancelled).
        return rec.get("source") in ("queue_signal", "management")
    return False


def derive_disk_entry(path: str, uuid: str) -> dict | None:
    """The idle session-list row for one on-disk session log, or None.

    None means "not a sidebar session": an empty / meta-only log (a session
    becomes real with its first message), or a fanout worker thread
    (``session_kind: worker`` meta, spec 009 §3a). ``scope`` is not part of
    the row — it is live in-memory state the caller adds.

    Raises ``OSError`` when the file cannot be read.
    """
    from agent_os.agent.session import (
        _derive_name, is_machine_derived_name, trigger_type_of,
    )

    stored_name = None  # name on the session_start meta, if present
    stored_pinned = False  # `pinned` on the same meta (spec 067)
    stored_pinned_target = None  # `pinned_target` (spec 074)
    stored_trigger_type = None  # `trigger_type` (spec 066 4b)
    origin = "chat"  # session_start meta origin; legacy logs → chat
    first_user_content = None  # for name backfill
    is_worker = False  # session_kind:"worker" meta (spec 009 fanout)
    first_real = None  # first non-meta (conversation) record
    last_real = None
    last_user_at = None  # timestamp of the last user row (spec 108)
    last_reply_at = None  # timestamp of the last reply-class row (spec 108)
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue  # valid JSON but not a record — skipped like a torn line
            # Skip identity metadata (session_start, model_swap, …) — a log
            # with only meta records is not a real session.
            if rec.get("role") == "meta":
                if rec.get("event") == "session_start":
                    if rec.get("name") is not None:
                        stored_name = rec["name"]
                    if rec.get("origin"):
                        origin = rec["origin"]
                    if rec.get("pinned") is not None:
                        stored_pinned = bool(rec["pinned"])
                    if rec.get("pinned_target"):
                        stored_pinned_target = rec["pinned_target"]
                    if rec.get("trigger_type"):
                        stored_trigger_type = rec["trigger_type"]
                elif rec.get("event") == "session_kind" and rec.get("kind") == "worker":
                    # Fanout native-worker session (spec 009 §3a): a one-shot,
                    # anonymous sub-task thread tagged immediately at
                    # construction. Never a sidebar entry — reachable only via
                    # transcript links in the fanout join summary / drill-in.
                    is_worker = True
                continue
            if first_real is None:
                first_real = rec
            if rec.get("role") == "user":
                if first_user_content is None:
                    first_user_content = rec.get("content")
                last_user_at = rec.get("timestamp")
            elif is_reply_row(rec):
                last_reply_at = rec.get("timestamp")
            last_real = rec
    if first_real is None or is_worker:
        return None
    last_activity_at = last_real.get("timestamp") if last_real is not None else None
    # Name: stored meta name wins — unless it is machine markup from the
    # pre-strip auto-namer ("[QUEUE ITEM…", "<attached_files>…"), which is
    # ignored so legacy sessions heal (mirrors Session.load); else derive from
    # the first user message (lazy backfill, in-memory only — no file
    # rewrite); else None.
    if stored_name is not None and not is_machine_derived_name(stored_name):
        name = stored_name
    else:
        name = _derive_name(first_user_content)
    # Address a disk-only session by its unique session_uuid: the F1
    # session_id on disk is often "default" and not unique across many prior
    # logs, which would shadow the active session and break the chat
    # endpoint's F1→F2 fast-path mapping. The uuid is unique and is what
    # callers use to load/act on (hydrate) the disk-only session.
    return {
        "session_id": uuid,
        "status": "idle",
        "session_uuid": uuid,
        "origin": origin,
        # Automation kind (spec 066 4b): stamped on the meta by new logs,
        # derived from the first user message (the trigger header) for old ones.
        "trigger_type": stored_trigger_type or trigger_type_of(first_user_content),
        "name": name,
        "pinned": stored_pinned,
        "pinned_target": stored_pinned_target,
        "last_terminal_event": None,
        "last_activity_at": last_activity_at,
        # Spec 108: who spoke last. The client badges a row whose reply is
        # newer than the user's last message and newer than the reply it
        # last showed; both are daemon-clock ISO strings.
        "last_user_at": last_user_at,
        "last_reply_at": last_reply_at,
    }


class SessionIndex:
    """The index of one sessions directory.

    Rows live in an in-memory mirror (the hot path) backed by the SQLite
    file (what survives a restart). Every SQLite failure degrades to the
    mirror alone — the list stays correct, it just is not persisted — and a
    file that cannot be opened as a valid index is discarded and recreated.
    Thread-safe: the list is served from worker threads while a build may run
    on another.
    """

    def __init__(self, sessions_dir: str):
        self.sessions_dir = sessions_dir
        self.path = index_path(sessions_dir)
        self._lock = threading.RLock()
        # fname -> (mtime_ns, size, entry-or-None)
        self._rows: dict[str, tuple[int, int, dict | None]] = {}
        self.ready = False
        self.building = False

    # -- SQLite plumbing -------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None,
                               check_same_thread=False)
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _open_checked(self) -> sqlite3.Connection:
        """Open the index, creating it if missing; raise if it is not a sound
        index of this version's schema."""
        conn = self._connect()
        try:
            (verdict,) = conn.execute("PRAGMA quick_check").fetchone()
            if verdict != "ok":
                raise sqlite3.DatabaseError(f"quick_check: {verdict}")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS index_meta ("
                " key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS sessions ("
                " fname TEXT PRIMARY KEY,"
                " mtime_ns INTEGER NOT NULL,"
                " size INTEGER NOT NULL,"
                " session_uuid TEXT,"
                " origin TEXT,"
                " trigger_type TEXT,"
                " name TEXT,"
                " last_activity_at TEXT,"
                " entry TEXT)"  # the list row as JSON; NULL = not a sidebar session
            )
            row = conn.execute(
                "SELECT value FROM index_meta WHERE key = 'version'"
            ).fetchone()
            if row is None or row[0] != INDEX_VERSION:
                conn.execute("DELETE FROM sessions")
                conn.execute(
                    "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('version', ?)",
                    (INDEX_VERSION,),
                )
            return conn
        except BaseException:
            conn.close()
            raise

    def _discard_file(self) -> None:
        for suffix in ("", "-journal", "-wal", "-shm"):
            try:
                os.remove(self.path + suffix)
            except FileNotFoundError:
                pass

    def load_rows(self) -> dict[str, tuple[int, int, dict | None]]:
        """Read every row from disk into the mirror (rebuilding a bad index).

        Returns the mirror. A missing file yields an empty index; a corrupt or
        unreadable one is deleted and recreated empty — the build that follows
        re-derives every row from the logs.
        """
        rows: dict[str, tuple[int, int, dict | None]] = {}
        try:
            try:
                conn = self._open_checked()
            except sqlite3.DatabaseError:
                logger.info("Session index %s is unreadable; rebuilding it", self.path)
                self._discard_file()
                conn = self._open_checked()
            try:
                for fname, mtime_ns, size, entry in conn.execute(
                    "SELECT fname, mtime_ns, size, entry FROM sessions"
                ):
                    try:
                        parsed = json.loads(entry) if entry is not None else None
                    except (TypeError, ValueError):
                        continue  # re-derived by the build
                    if parsed is not None and not isinstance(parsed, dict):
                        continue
                    rows[fname] = (int(mtime_ns), int(size), parsed)
            finally:
                conn.close()
        except (sqlite3.Error, OSError):
            logger.warning("Session index %s unavailable; serving the list from memory",
                           self.path, exc_info=True)
        with self._lock:
            self._rows = rows
        return rows

    def _write(self, upserts: list[tuple], deletes: list[str]) -> None:
        if not upserts and not deletes:
            return
        try:
            # Checked open, not a bare connect: if the file was deleted under a
            # running daemon this recreates a sound (empty) index instead of
            # failing on a missing table; the next start re-derives the rest.
            conn = self._open_checked()
            try:
                conn.execute("BEGIN")
                if upserts:
                    conn.executemany(
                        "INSERT OR REPLACE INTO sessions (fname, mtime_ns, size, session_uuid,"
                        " origin, trigger_type, name, last_activity_at, entry)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        upserts,
                    )
                if deletes:
                    conn.executemany("DELETE FROM sessions WHERE fname = ?",
                                     [(f,) for f in deletes])
                conn.execute("COMMIT")
            finally:
                conn.close()
        except (sqlite3.Error, OSError):
            logger.warning("Could not write the session index %s", self.path, exc_info=True)

    @staticmethod
    def _upsert_row(fname: str, st: os.stat_result, entry: dict | None) -> tuple:
        e = entry or {}
        return (
            fname, st.st_mtime_ns, st.st_size, e.get("session_uuid"), e.get("origin"),
            e.get("trigger_type"), e.get("name"), e.get("last_activity_at"),
            json.dumps(entry, ensure_ascii=False) if entry is not None else None,
        )

    # -- the API the list uses -------------------------------------------

    def get(self, fname: str, st: os.stat_result) -> tuple[bool, dict | None]:
        """``(True, row)`` when the indexed row is current for this stat,
        else ``(False, None)``. ``row`` may be None: "not a sidebar session"."""
        with self._lock:
            cached = self._rows.get(fname)
        if cached is None or cached[0] != st.st_mtime_ns or cached[1] != st.st_size:
            return False, None
        return True, cached[2]

    def put(self, fname: str, st: os.stat_result, entry: dict | None) -> None:
        self.put_many([(fname, st, entry)])

    def put_many(self, items: list[tuple[str, os.stat_result, dict | None]]) -> None:
        if not items:
            return
        with self._lock:
            for fname, st, entry in items:
                self._rows[fname] = (st.st_mtime_ns, st.st_size, entry)
            self._write([self._upsert_row(f, st, e) for f, st, e in items], [])

    def prune(self, present: set[str]) -> None:
        """Drop rows whose log file is gone."""
        with self._lock:
            gone = [f for f in self._rows if f not in present]
            if not gone:
                return
            for f in gone:
                del self._rows[f]
            self._write([], gone)
