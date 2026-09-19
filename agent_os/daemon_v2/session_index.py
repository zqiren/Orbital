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
INDEX_VERSION = "1"


def index_path(sessions_dir: str) -> str:
    return os.path.join(sessions_dir, INDEX_FILENAME)


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
            if first_user_content is None and rec.get("role") == "user":
                first_user_content = rec.get("content")
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
