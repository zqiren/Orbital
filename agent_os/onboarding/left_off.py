# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Where the user left off in one project (spec 138 "Left-off cards").

Sources: the project's own Orbital sessions (their JSONL) and its native
sessions — the scanned history whose folder is the workspace, plus the
``imported_sessions`` members a group project stores. A native session that
was opened in Orbital (materialised) but not continued there keeps the stop
cause its native transcript recorded; once the user wrote in it, Orbital's
own rows decide. Blocking I/O — call off the event loop.
"""

from __future__ import annotations

import logging
import os
import re
import threading

from agent_os.onboarding.history import (
    HistoryScanner, LeftOffEntry, git_facts, is_noise, resets_at_if_future, select_left_off,
)
from agent_os.onboarding.history_types import HistorySession
from agent_os.onboarding.sources._common import TurnState, iter_json_lines, norm_ts

logger = logging.getLogger(__name__)

_orbital_cache: dict[tuple, "OrbitalSession | None"] = {}
_orbital_lock = threading.Lock()
_ORBITAL_CACHE_MAX = 2048


# Daemon bookkeeping rows (spec 112 A6, 124, 146): the end-of-run record turn,
# a tidy note, an ask card. None of them is a turn of the conversation, so
# they never decide how the last turn ended or what it said (spec 154 §5.2).
_BOOKKEEPING_KINDS = frozenset({"memory_record", "memory_tidy", "ask_card"})
# Worker terminal kinds whose ``_meta.reply`` carries the worker's own words
# (lifecycle_observer: completed / continued); every other kind carries none.
_REPLY_TERMINAL_KINDS = frozenset({"completed", "continued"})
_STATUS_RE = re.compile(r"\[STATUS:[^\]]*\]", re.IGNORECASE)
_LINE_MARKERS_RE = re.compile(r"^\s*(?:#{1,6}\s+|>\s*|[-*+]\s+|\d+[.)]\s+)+")
_INLINE_MARKERS_RE = re.compile(r"(\*\*|__|`+|~~)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
RESULT_LINE_MAX = 140


def reply_line(text: str | None) -> str | None:
    """The first non-empty line of a reply, as a card's muted line: markdown
    markers stripped, ``[STATUS: …]`` removed, at most 140 characters."""
    if not isinstance(text, str):
        return None
    for raw in _STATUS_RE.sub("", text).splitlines():
        line = _LINE_MARKERS_RE.sub("", raw)
        line = _LINK_RE.sub(r"\1", line)
        line = _INLINE_MARKERS_RE.sub("", line)
        line = " ".join(line.split())
        if not line or set(line) <= set("-*_=|:"):
            continue  # blank, or a rule / table separator
        if len(line) > RESULT_LINE_MAX:
            line = line[:RESULT_LINE_MAX - 1].rstrip() + "…"
        return line
    return None


class OrbitalSession:
    __slots__ = ("session_id", "agent", "entry", "own_rows", "skip", "reply_text")

    def __init__(self, session_id: str, agent: str | None, entry: LeftOffEntry,
                 own_rows: bool, skip: bool, reply_text: str | None = None):
        self.session_id = session_id
        self.agent = agent
        self.entry = entry
        self.own_rows = own_rows   # any conversation row written in Orbital (not imported)
        self.skip = skip           # memory / automation / fan-out child: never a card
        # Spec 154 §5.2: the last reply's text (the agent's final answer, or a
        # completed / continued worker's ``_meta.reply``); None after any
        # other ending.
        self.reply_text = reply_text

    @property
    def reply_line(self) -> str | None:
        return reply_line(self.reply_text)


def _row_imported(row: dict) -> bool:
    meta = row.get("_meta")
    return isinstance(meta, dict) and bool(meta.get("imported"))


def read_orbital_session(path: str) -> OrbitalSession | None:
    """One Orbital session JSONL → its left-off facts. None when unreadable
    or it holds no conversation."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_mtime_ns, st.st_size)
    with _orbital_lock:
        if key in _orbital_cache:
            return _orbital_cache[key]
    result = _read_orbital(path)
    with _orbital_lock:
        if len(_orbital_cache) >= _ORBITAL_CACHE_MAX:
            _orbital_cache.clear()
        _orbital_cache[key] = result
    return result


def _read_orbital(path: str) -> OrbitalSession | None:
    stem = os.path.basename(path)[:-len(".jsonl")]
    probe = HistorySession(agent="orbital", native_id=stem, path=path)
    turn = TurnState(probe)
    session_id = None
    agent = None
    skip = False
    own_rows = False
    any_row = False
    reply_text: str | None = None
    try:
        for row in iter_json_lines(path):
            if session_id is None and isinstance(row.get("session_id"), str):
                session_id = row["session_id"]
            role = row.get("role")
            ts = norm_ts(row.get("timestamp"))
            meta = row.get("_meta") if isinstance(row.get("_meta"), dict) else {}
            if role == "meta":
                event = row.get("event")
                if event == "session_start":
                    if row.get("origin") == "memory" or row.get("trigger_type"):
                        skip = True
                elif event == "session_kind":
                    skip = True
                continue
            if row.get("_kind") in _BOOKKEEPING_KINDS:
                continue
            imported = _row_imported(row)
            if role == "user":
                if row.get("source") == "daemon":
                    continue
                reply_text = None
                any_row = True
                own_rows = own_rows or not imported
                target = row.get("target")
                agent = target if isinstance(target, str) and target else None
                probe.ended_at = ts or probe.ended_at
                turn.user_text(row.get("content"), ts)
            elif role == "assistant":
                any_row = True
                own_rows = own_rows or not imported
                probe.ended_at = ts or probe.ended_at
                if row.get("tool_calls"):
                    turn.working()
                elif isinstance(row.get("content"), str) and row["content"].strip():
                    turn.answered()
                    reply_text = row["content"]
            elif role == "tool":
                any_row = True
                turn.working()
            elif role == "system":
                content = row.get("content") if isinstance(row.get("content"), str) else ""
                event = meta.get("event")
                if event == "sub_agent_terminal":
                    any_row = True
                    own_rows = own_rows or not imported
                    probe.ended_at = ts or probe.ended_at
                    kind = meta.get("kind")
                    worker_reply = meta.get("reply")
                    reply_text = (worker_reply if kind in _REPLY_TERMINAL_KINDS
                                  and isinstance(worker_reply, str) else None)
                    if kind == "error":
                        turn.error(content, ts)
                    elif kind == "stopped":
                        turn.interrupted("stopped")
                    else:
                        turn.answered()
                elif row.get("source") == "management":
                    if content.startswith("LLM error") or "API key was rejected" in content:
                        probe.ended_at = ts or probe.ended_at
                        turn.error(content, ts)
                        reply_text = None
                    elif content.startswith("[cancelled by user]"):
                        probe.ended_at = ts or probe.ended_at
                        turn.interrupted("cancelled")
                elif content.startswith("[Sub-agent]") and "stopped with error" in content:
                    probe.ended_at = ts or probe.ended_at
                    turn.error(content, ts)
    except OSError:
        return None
    if not any_row:
        return None
    turn.finish()
    entry = LeftOffEntry(
        session_id=session_id or stem,
        agent=agent,
        user_text=probe.last_user_text,
        at=probe.ended_at or probe.last_user_at,
        stop=probe.stop,
    )
    return OrbitalSession(entry.session_id, entry.agent, entry, own_rows,
                          skip or probe.automation, reply_text)


def _orbital_entries(workspace: str) -> list[OrbitalSession]:
    from agent_os.agent.project_paths import ProjectPaths
    sessions_dir = ProjectPaths(workspace).sessions_dir
    try:
        names = os.listdir(sessions_dir)
    except OSError:
        return []
    out = []
    for name in names:
        if not name.endswith(".jsonl"):
            continue
        parsed = read_orbital_session(os.path.join(sessions_dir, name))
        if parsed is not None and not parsed.skip:
            out.append(parsed)
    return out


def _native_entry(s: HistorySession) -> LeftOffEntry:
    from agent_os.onboarding.session_import import imported_session_id
    return LeftOffEntry(
        session_id=imported_session_id(s.agent, s.native_id),
        agent=s.agent,
        user_text=s.last_user_text,
        at=s.ended_at or s.last_user_at,
        stop=s.stop,
        branch=s.branch,
    )


def project_native_sessions(project: dict, sessions: list[HistorySession],
                            env) -> list[HistorySession]:
    """The scanned sessions of a project: same folder, or a stored member."""
    from agent_os.onboarding.import_scanner import path_key
    workspace = project.get("workspace") or ""
    want = path_key(workspace) if workspace else None
    members = {(m.get("agent"), m.get("native_id"))
               for m in project.get("imported_sessions") or [] if isinstance(m, dict)}
    out = []
    for s in sessions:
        if is_noise(s, env):
            continue
        if (s.agent, s.native_id) in members or (want and s.cwd and path_key(s.cwd) == want):
            out.append(s)
    return out


def _entry_json(e: LeftOffEntry) -> dict:
    return {"session_id": e.session_id, "agent": e.agent, "user_text": e.user_text, "at": e.at}


def left_off_for_project(project: dict, scanner: HistoryScanner | None, *,
                         now=None) -> dict:
    workspace = project.get("workspace") or ""
    entries: dict[str, LeftOffEntry] = {}
    natives: dict[str, LeftOffEntry] = {}
    native_ids: dict[str, str] = {}
    if scanner is not None and not project.get("is_scratch"):
        try:
            sessions = scanner.collect()
            for s in project_native_sessions(project, sessions, scanner.env()):
                e = _native_entry(s)
                natives[e.session_id] = e
                native_ids[e.session_id] = s.native_id
        except Exception:
            logger.warning("left-off: native history unavailable", exc_info=True)
    entries.update(natives)
    if workspace and os.path.isdir(workspace):
        orbital = _orbital_entries(workspace)
        # A native session one of Orbital's own sessions ran as its worker
        # thread is that Orbital session, not the user's history (Kimi Code,
        # MiniMax Code and pi leave no marker of who started them).
        from agent_os.onboarding.session_import import orbital_thread_ids
        own_ids = {o.session_id for o in orbital}
        driven = orbital_thread_ids(workspace)
        for sid in natives:
            if native_ids[sid] in driven and sid not in own_ids:
                entries.pop(sid, None)
        for o in orbital:
            # An imported session the user never wrote in keeps the native
            # transcript's ending (materialising one rewrites every reply as
            # a completed row).
            if o.session_id in natives and not o.own_rows:
                continue
            entries[o.session_id] = o.entry
    picked = select_left_off(list(entries.values()), now=now)
    last = picked["last_time"]
    last_json = None
    if last is not None:
        last_json = _entry_json(last)
        last_json["git"] = git_facts(workspace, last.at, last.branch) if workspace else None
    unfinished = []
    for e in picked["unfinished"]:
        item = _entry_json(e)
        item["cause"] = e.stop.cause
        item["resets_at"] = resets_at_if_future(e.stop, now)
        unfinished.append(item)
    return {"last_time": last_json, "unfinished": unfinished}
