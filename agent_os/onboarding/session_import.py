# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A project's native agent conversations, as Orbital sessions.

A folder a user brings to Orbital often already holds conversations they had
with Claude Code, Codex, pi, Kimi Code or MiniMax Code in it. This module
finds them and turns one into Orbital's own row shapes:

* :func:`list_native_sessions` — the folder's native sessions, metadata only
  (a sqlite row or a bounded head + tail read of a Claude transcript for the
  title). Cached in memory by ``(path, mtime, size)``, so the session list can
  call it on every refresh.
* :func:`convert_transcript` — one native transcript as the rows the live
  pinned-worker path writes: user rows, and each worker reply as the
  ``sub_agent_terminal`` completed system row, so the chat renders the old
  conversation with no new row kind and the recap builder reads it unchanged.
  Ends with the ``sub_agent_thread`` resume record that makes the next send
  resume the native thread through the existing resume path.

User text is read through :func:`text_clean.clean_user_text` (spec 138):
Orbital's own injected turns are dropped with their replies, an automation
turn keeps only its task, a recap keeps only the words after it.

A project made from a group of conversations that ran elsewhere names them
(``imported_sessions`` on the project); :func:`list_native_sessions` lists
those ``members`` beside the folder's own.

pi / Kimi Code / MiniMax Code are parsed by their spec 138 history readers
(``agent_os.onboarding.sources``), so a format is parsed in one place.

Strictly read-only against every native store: nothing here opens a native
file for writing. Sessions Orbital itself started (its Codex app-server
threads, its Claude SDK sessions, and the kimi / mcode ACP sessions its own
session records name) are skipped — they already are Orbital sessions.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from agent_os.onboarding.import_scanner import claude_project_dirs_for, path_key
from agent_os.onboarding.sources import kimi_code as _kimi_src
from agent_os.onboarding.sources import minimax_code as _minimax_src
from agent_os.onboarding.sources import pi as _pi_src
from agent_os.onboarding.sources._native import cleaned_turns, history_home
from agent_os.onboarding.text_clean import KIND_INJECTED, clean_user_text

logger = logging.getLogger(__name__)

AGENT_CLAUDE = "claude-code"
AGENT_CODEX = "codex"
AGENT_PI = "pi"
AGENT_KIMI = "kimi-code"
AGENT_MINIMAX = "minimax-code"

AGENT_DISPLAY_NAMES = {AGENT_CLAUDE: "Claude Code", AGENT_CODEX: "Codex", AGENT_PI: "Pi",
                       AGENT_KIMI: "Kimi Code", AGENT_MINIMAX: "MiniMax Code"}

# Spec 138 readers by agent: they parse these agents' stores for both the
# history scan and this module.
_HISTORY_SOURCES = {AGENT_PI: _pi_src.SOURCE, AGENT_KIMI: _kimi_src.SOURCE,
                    AGENT_MINIMAX: _minimax_src.SOURCE}

# Agents whose thread record carries the native file the transport reopens.
_RESUME_PATH_AGENTS = frozenset({AGENT_CODEX, AGENT_PI})

IMPORTED_SESSION_PREFIX = "imp_"

# Bounded metadata read of a Claude transcript (contract §6).
_HEAD_BYTES = 256 * 1024
_TAIL_BYTES = 64 * 1024

# Conversion bounds (contract §6): the last 400 turns, at most 2 MB of reply text.
MAX_TURNS = 400
MAX_REPLY_BYTES = 2 * 1024 * 1024

# Characters of the reply carried in the terminal row's one-line content.
_SUMMARY_CHARS = 200

# Claude entrypoints of programmatic SDK sessions. Orbital's own claude-code
# worker runs through the Python SDK ("sdk-py"); those sessions are already
# Orbital sessions and must not be listed a second time. `claude -p` is
# "sdk-cli" and the interactive CLI "cli" — both are the user's own.
_CLAUDE_SKIP_ENTRYPOINTS = frozenset({"sdk-py", "sdk-ts"})

# Codex rollout ``originator`` of threads Orbital's app-server transport opened
# (clientInfo name, codex_transport.py).
_CODEX_SKIP_ORIGINATORS = frozenset({"orbital"})
_CODEX_SKIP_THREAD_SOURCES = frozenset({"subagent", "guardian_review"})

# Claude user rows that are not the user's words: slash-command plumbing and
# the interrupt marker.
_CLAUDE_MACHINE_PREFIXES = (
    "<command-name>", "<command-message>", "<command-args>",
    "<local-command-stdout>", "<local-command-stderr>", "<local-command-caveat>",
    "[Request interrupted by user",
)

# Codex injects context into user messages as tagged blocks
# (<environment_context>, <user_instructions>, <image …>, <heartbeat>, …) and the
# AGENTS.md preamble. Neither is the user's words.
_CODEX_TAG_BLOCK_RE = re.compile(r"^</?[A-Za-z_][\w-]*(\s[^>]*)?>")
_CODEX_AGENTS_MD_PREFIX = "# AGENTS.md instructions for"


# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------


def claude_home(claude_dir: str | None = None) -> str:
    """``$CLAUDE_CONFIG_DIR`` or ``~/.claude`` (an explicit arg wins). Resolved
    by ``history_paths`` so the session list and the history scan always read
    the same folder (``ORBITAL_HISTORY_HOME`` included)."""
    if claude_dir:
        return claude_dir
    from agent_os.onboarding import history_paths
    return history_paths.claude_dir()


def codex_home(codex_dir: str | None = None) -> str:
    """``$ORBITAL_CODEX_HOME`` > ``$CODEX_HOME`` > ``~/.codex``.

    ``ORBITAL_CODEX_HOME`` points only the IMPORTER at a staging dir, without
    moving codex's own auth/config (smokes)."""
    if codex_dir:
        return codex_dir
    from agent_os.onboarding import history_paths
    return history_paths.codex_dir()


def imported_session_id(agent: str, native_id: str) -> str:
    """Stable Orbital session id of a native session: ``imp_<agent>_<16 chars>``.

    The first 16 alphanumerics of the native id (UUID / UUIDv7 → 64 bits of
    hex), so the id never changes across daemon restarts and doubles as the
    JSONL stem once the session is materialized."""
    compact = re.sub(r"[^0-9A-Za-z]", "", native_id or "")[:16]
    return f"{IMPORTED_SESSION_PREFIX}{agent}_{compact}"


# Full shape, not a prefix: a project named "Imp" mints ``imp_<hex8>`` chat ids.
_IMPORTED_SESSION_ID_RE = re.compile(
    rf"{IMPORTED_SESSION_PREFIX}(?:{'|'.join(map(re.escape, AGENT_DISPLAY_NAMES))})"
    r"_[0-9A-Za-z]{1,16}")


def is_imported_session_id(session_id: str | None) -> bool:
    return (isinstance(session_id, str)
            and _IMPORTED_SESSION_ID_RE.fullmatch(session_id) is not None)


# --------------------------------------------------------------------------
# Listing
# --------------------------------------------------------------------------


@dataclass
class NativeSession:
    agent: str               # claude-code | codex | pi | kimi-code | minimax-code
    native_id: str           # the id the agent resumes by
    path: str                # the transcript the reader parses
    title: str | None
    last_activity: str | None  # ISO-8601 UTC
    size_bytes: int
    rollout_path: str | None = None  # the file resume reopens: Codex rollout, pi session

    @property
    def session_id(self) -> str:
        return imported_session_id(self.agent, self.native_id)


_cache_lock = threading.Lock()
_cache: dict[tuple, object] = {}
_CACHE_MAX = 4096


def _cache_get(key: tuple):
    with _cache_lock:
        return _cache.get(key)


def _cache_put(key: tuple, value) -> None:
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            _cache.clear()
        _cache[key] = value


def clear_cache() -> None:
    """Drop every memoized read (tests)."""
    with _cache_lock:
        _cache.clear()


# How the first text Orbital sends a worker begins (dispatch_tools.seat_note,
# skills.worker_skills_note, job_launcher's launcher note). A native thread
# titled by one of them is a thread Orbital ran, or another app's copy of one:
# the Codex desktop app imports Claude Code sessions into its own store under
# a new id and its own originator, so neither the folder's thread records nor
# the originator can tell (seen 2026-11-26: 33 such threads in 38 folders).
_ORBITAL_OPENERS = (
    "Other agents in this project:",
    "Skills you can use (from Orbital):",
    "Long-running commands: never wait on a command",
)


def _titled_by_orbital(title: str | None) -> bool:
    return isinstance(title, str) and title.lstrip().startswith(_ORBITAL_OPENERS)


def list_native_sessions(workspace: str, *, members=None, claude_dir: str | None = None,
                         codex_dir: str | None = None,
                         history_dir: str | None = None) -> list[NativeSession]:
    """Every native session that ran in ``workspace``, plus the ``members``
    (``[{"agent", "native_id"}]``, a group project's ``imported_sessions``)
    wherever they ran — newest first, each once.

    Metadata only. Sessions with no user message (sidechains, empty) and
    sessions Orbital itself started are skipped. Never raises: a source that is
    missing or unreadable contributes nothing. ``history_dir`` is the home the
    pi / Kimi Code / MiniMax Code stores sit in (``$ORBITAL_HISTORY_HOME``,
    else ``~``).
    """
    if not workspace or not os.path.isdir(workspace):
        return []
    homes = {AGENT_CLAUDE: claude_home(claude_dir), AGENT_CODEX: codex_home(codex_dir),
             "history": history_home(history_dir)}
    out: list[NativeSession] = []
    for fn, home in ((_list_claude, homes[AGENT_CLAUDE]), (_list_codex, homes[AGENT_CODEX]),
                     (_list_pi, homes["history"]), (_list_kimi, homes["history"]),
                     (_list_minimax, homes["history"])):
        try:
            out.extend(fn(workspace, home))
        except Exception:
            logger.warning("session import: %s listing failed for %s",
                           fn.__name__, workspace, exc_info=True)
    if out:
        # Orbital's own worker threads, known by the ids the folder's session
        # records hold. The only mark ACP agents leave, and the backstop for
        # Codex / Claude threads whose originator or entrypoint did not say so.
        own = orbital_thread_ids(workspace)
        out = [s for s in out
               if s.native_id not in own and not _titled_by_orbital(s.title)]
    seen = {(s.agent, s.native_id) for s in out}
    for agent, native_id in _member_pairs(members):
        if (agent, native_id) in seen:
            continue
        seen.add((agent, native_id))
        try:
            native = _find_member(agent, native_id, homes)
        except Exception:
            logger.warning("session import: member %s %s lookup failed",
                           agent, native_id, exc_info=True)
            native = None
        if native is not None:
            out.append(native)
    out.sort(key=lambda s: s.last_activity or "", reverse=True)
    return out


def find_native_session(workspace: str, session_id: str, **kwargs) -> NativeSession | None:
    """The native session behind an ``imp_…`` id, or None."""
    for native in list_native_sessions(workspace, **kwargs):
        if native.session_id == session_id:
            return native
    return None


def _member_pairs(members) -> list[tuple[str, str]]:
    pairs = []
    for m in members or []:
        if not isinstance(m, dict):
            continue
        agent, native_id = m.get("agent"), m.get("native_id")
        if (agent in AGENT_DISPLAY_NAMES and isinstance(native_id, str) and native_id
                and (agent, native_id) not in pairs):
            pairs.append((agent, native_id))
    return pairs


def _find_member(agent: str, native_id: str, homes: dict) -> NativeSession | None:
    if agent == AGENT_CLAUDE:
        return _claude_member(native_id, homes[AGENT_CLAUDE])
    if agent == AGENT_CODEX:
        return _codex_member(native_id, homes[AGENT_CODEX])
    if agent == AGENT_PI:
        home = homes["history"]
        suffix = f"_{native_id}.jsonl"
        for path in _pi_src.SOURCE.session_files(home):
            if path.endswith(suffix) or (_pi_header(path) or {}).get("id") == native_id:
                return _history_native(AGENT_PI, path)
        return None
    if agent == AGENT_KIMI:
        for sdir in _kimi_src.SOURCE.session_dirs(homes["history"]):
            if os.path.basename(sdir) == native_id:
                return _history_native(AGENT_KIMI,
                                       os.path.join(sdir, "agents", "main", "wire.jsonl"))
        return None
    if agent == AGENT_MINIMAX:
        v2 = _minimax_src.SOURCE.root(homes["history"])
        row = _minimax_src.SOURCE.runtime_rows(v2).get(native_id)
        path = _minimax_messages_path(v2, row) if row else None
        if path is None:
            for sdir in _minimax_src.SOURCE.session_dirs(homes["history"]):
                if _minimax_src.SOURCE.session_id_of(sdir) == native_id:
                    path = os.path.join(sdir, "messages.jsonl")
                    break
        return _history_native(AGENT_MINIMAX, path) if path else None
    return None


def orbital_thread_ids(workspace: str) -> set[str]:
    """Native ids the project's own Orbital sessions hold as worker threads
    (``sub_agent_thread`` records) — the native sessions Orbital ran.
    Each session file is read once per ``(mtime, size)``. One cache entry per
    folder, not per file: the session list runs this for every project in the
    Workbench feed, and per-file keys would fill the shared cache and clear it
    on every pass.

    On a miss (always, right after a daemon start) the folder's session index
    answers first: a row whose (mtime_ns, size) matches the file and that
    knows its ids saves reading the file. A file read for a row with the
    matching stamp but no ids writes them back, so the next start skips it."""
    from agent_os.agent.project_paths import ProjectPaths
    from agent_os.daemon_v2.session_index import read_thread_ids, store_thread_ids
    sessions_dir = ProjectPaths(workspace).sessions_dir
    ids: set[str] = set()
    try:
        names = os.listdir(sessions_dir)
    except OSError:
        return ids
    key = ("orbital-threads", sessions_dir)
    before = _cache_get(key) or {}
    now: dict[str, tuple] = {}
    indexed = None  # the index's rows, read on the first miss
    learned: list[tuple] = []
    for name in names:
        if not name.endswith(".jsonl"):
            continue
        fpath = os.path.join(sessions_dir, name)
        try:
            st = os.stat(fpath)
        except OSError:
            continue
        stamp = (st.st_mtime_ns, st.st_size)
        hit = before.get(name)
        if hit is not None and hit[0] == stamp:
            found = hit[1]
        else:
            if indexed is None:
                indexed = read_thread_ids(sessions_dir)
            row = indexed.get(name)
            current = row is not None and (row[0], row[1]) == stamp
            if current and row[2] is not None:
                found = row[2]
            else:
                found = frozenset(_thread_ids_in(fpath))
                if current:
                    learned.append((name, stamp[0], stamp[1], found))
        now[name] = (stamp, found)
        ids |= found
    store_thread_ids(sessions_dir, learned)
    if now != before:
        _cache_put(key, now)
    return ids


def _thread_ids_in(path: str) -> set[str]:
    found: set[str] = set()
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if '"sub_agent_thread"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except (ValueError, TypeError):
                    continue
                thread = rec.get("thread") if isinstance(rec, dict) else None
                if isinstance(thread, dict) and isinstance(thread.get("session_id"), str):
                    found.add(thread["session_id"])
    except OSError:
        pass
    return found


# ---- Claude Code -----------------------------------------------------------


def _claude_dirs(workspace: str, projects_dir: str) -> list[str]:
    from agent_os.agent.transports.sdk_transport import SDKTransport
    slug_dir = os.path.join(projects_dir, SDKTransport.claude_project_slug(workspace))
    if os.path.isdir(slug_dir):
        return [slug_dir]
    # Slug drift: probe each project dir's recorded cwd. Memoized on the
    # projects dir's mtime (a new project dir bumps it).
    try:
        st = os.stat(projects_dir)
    except OSError:
        return []
    key = ("claude-probe", projects_dir, st.st_mtime_ns, path_key(workspace))
    cached = _cache_get(key)
    if cached is None:
        cached = claude_project_dirs_for(projects_dir, workspace)
        _cache_put(key, cached)
    return list(cached)


def _list_claude(workspace: str, home: str) -> list[NativeSession]:
    projects_dir = os.path.join(home, "projects")
    want = path_key(workspace)
    out: list[NativeSession] = []
    for d in _claude_dirs(workspace, projects_dir):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            if not name.endswith(".jsonl"):
                continue
            native = _claude_native(os.path.join(d, name), want)
            if native is not None:
                out.append(native)
    return out


def _claude_native(fpath: str, want: str | None) -> NativeSession | None:
    """One Claude transcript as a listing entry, or None when it has no user
    message, is Orbital's own, or ran in another folder than ``want`` (None:
    any folder — a named group member)."""
    try:
        st = os.stat(fpath)
    except OSError:
        return None
    key = ("claude-meta", fpath, st.st_mtime_ns, st.st_size)
    meta = _cache_get(key)
    if meta is None:
        meta = _claude_metadata(fpath, st)
        _cache_put(key, meta)
    if not meta or not meta.get("has_user"):
        return None
    if meta.get("entrypoint") in _CLAUDE_SKIP_ENTRYPOINTS:
        return None
    cwd = meta.get("cwd")
    if want is not None and cwd and path_key(cwd) != want:
        return None  # lossy slug: another folder that encodes the same
    native_id = meta.get("session_id") or os.path.basename(fpath)[:-len(".jsonl")]
    return NativeSession(
        agent=AGENT_CLAUDE, native_id=native_id, path=fpath,
        title=meta.get("title"), last_activity=meta.get("last_activity"),
        size_bytes=st.st_size,
    )


def _claude_member(native_id: str, home: str) -> NativeSession | None:
    projects_dir = os.path.join(home, "projects")
    try:
        dirs = os.listdir(projects_dir)
    except OSError:
        return None
    for d in dirs:
        fpath = os.path.join(projects_dir, d, f"{native_id}.jsonl")
        if os.path.isfile(fpath):
            native = _claude_native(fpath, None)
            if native is not None:
                return native
    return None


def _read_bounded(path: str, size: int) -> tuple[list[str], list[str]]:
    """``(head_lines, tail_lines)`` of a file: whole lines within the first
    256 KB and the last 64 KB (the tail empty when the head covered it)."""
    with open(path, "rb") as fh:
        head = fh.read(_HEAD_BYTES)
        tail = b""
        if size > _HEAD_BYTES:
            fh.seek(max(_HEAD_BYTES, size - _TAIL_BYTES))
            tail = fh.read()
    head_lines = head.decode("utf-8", errors="ignore").split("\n")
    if size > _HEAD_BYTES:
        head_lines = head_lines[:-1]  # a line cut by the bound
    tail_lines = tail.decode("utf-8", errors="ignore").split("\n")
    if tail_lines and size - _TAIL_BYTES > _HEAD_BYTES:
        tail_lines = tail_lines[1:]  # starts mid-line
    return head_lines, tail_lines


def _json_rows(lines):
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(rec, dict):
            yield rec


def _claude_metadata(path: str, st: os.stat_result) -> dict:
    """Title, cwd, entrypoint, session id, last activity, has-user — from a
    bounded head + tail read, never the whole file."""
    try:
        head, tail = _read_bounded(path, st.st_size)
    except OSError:
        return {}
    meta: dict = {"has_user": False}
    first_user_text = None
    title = None
    custom_title = None
    last_ts = None
    for rec in _json_rows(head + tail):
        rtype = rec.get("type")
        if rtype == "ai-title" and isinstance(rec.get("aiTitle"), str):
            title = rec["aiTitle"]
        elif rtype == "custom-title" and isinstance(rec.get("customTitle"), str):
            custom_title = rec["customTitle"]
        if isinstance(rec.get("sessionId"), str) and "session_id" not in meta:
            meta["session_id"] = rec["sessionId"]
        if rec.get("isSidechain") is True:
            continue
        if isinstance(rec.get("cwd"), str) and "cwd" not in meta:
            meta["cwd"] = rec["cwd"]
        if isinstance(rec.get("entrypoint"), str) and "entrypoint" not in meta:
            meta["entrypoint"] = rec["entrypoint"]
        if rtype in ("user", "assistant") and rec.get("timestamp"):
            last_ts = rec["timestamp"]
        if rtype == "user" and first_user_text is None:
            text = _claude_user_text(rec)
            if text:
                # Title from the user's words, not a recap / trigger header.
                first_user_text = _cleaned(text) or text
                meta["has_user"] = True
    from agent_os.agent.session import _derive_name
    meta["title"] = (custom_title or title or _derive_name(first_user_text))
    meta["last_activity"] = _norm_ts(last_ts) or _iso_epoch(st.st_mtime)
    return meta


def _claude_user_text(rec: dict) -> str | None:
    """The user's own words on a Claude ``user`` row, else None (tool results,
    meta/compact rows, slash-command plumbing, interrupt markers)."""
    if rec.get("isMeta") or rec.get("isCompactSummary") or rec.get("isSidechain"):
        return None
    msg = rec.get("message")
    if not isinstance(msg, dict):
        return None
    content = msg.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = [b.get("text") for b in content
                 if isinstance(b, dict) and b.get("type") == "text"
                 and isinstance(b.get("text"), str)]
        text = "\n\n".join(parts)
    else:
        return None
    text = text.strip()
    if not text or text.startswith(_CLAUDE_MACHINE_PREFIXES):
        return None
    return text


# ---- Codex -----------------------------------------------------------------


def _list_codex(workspace: str, home: str) -> list[NativeSession]:
    index = os.path.join(home, "state_5.sqlite")
    if os.path.isfile(index):
        rows = _codex_threads(index)
        if rows is not None:
            # Memoized per (index snapshot, folder): the filter realpaths
            # every thread's cwd, which the list must not redo on each read.
            key = ("codex-sessions", id(rows), path_key(workspace))
            cached = _cache_get(key)
            if cached is None or cached[0] is not rows:
                cached = (rows, _codex_sessions_from_rows(rows, workspace))
                _cache_put(key, cached)
            return list(cached[1])
    return _codex_sessions_from_rollouts(os.path.join(home, "sessions"), workspace)


def _codex_threads(index: str) -> list[dict] | None:
    """Non-archived rows of the threads index, memoized on the db + WAL stat.
    None when the index is not the expected schema (caller falls back)."""
    # Resolve a symlinked index first: its ``-wal`` sits beside the real file,
    # and an update that is still only in the WAL must change the cache key.
    index = os.path.realpath(index)
    try:
        st = os.stat(index)
        wal = os.stat(index + "-wal") if os.path.exists(index + "-wal") else None
    except OSError:
        return None
    key = ("codex-index", index, st.st_mtime_ns, st.st_size,
           wal.st_mtime_ns if wal else 0, wal.st_size if wal else 0)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        con = sqlite3.connect(f"file:{index}?mode=ro", uri=True, timeout=2.0)
    except sqlite3.Error:
        return None
    try:
        try:
            cols = {r[1] for r in con.execute("PRAGMA table_info(threads)")}
        except sqlite3.Error:
            return None
        if not {"id", "rollout_path", "cwd"} <= cols:
            return None
        wanted = [c for c in ("id", "rollout_path", "cwd", "title", "first_user_message",
                              "preview", "updated_at", "updated_at_ms", "source",
                              "thread_source", "originator", "archived")
                  if c in cols]
        sql = f"SELECT {', '.join(wanted)} FROM threads"
        if "archived" in cols:
            sql += " WHERE archived = 0"
        try:
            rows = [dict(zip(wanted, r)) for r in con.execute(sql).fetchall()]
        except sqlite3.Error:
            return None
    finally:
        con.close()
    _cache_put(key, rows)
    return rows


def _codex_row_is_own(row: dict) -> bool:
    """Threads Orbital opened, or codex's own sub-agent / reviewer threads."""
    if (row.get("originator") or "") in _CODEX_SKIP_ORIGINATORS:
        return True
    if (row.get("thread_source") or "") in _CODEX_SKIP_THREAD_SOURCES:
        return True
    source = row.get("source") or ""
    return isinstance(source, str) and source.startswith("{") and "subagent" in source


def _codex_sessions_from_rows(rows: list[dict], workspace: str) -> list[NativeSession]:
    want = path_key(workspace)
    out: list[NativeSession] = []
    for row in rows:
        cwd = row.get("cwd")
        if not isinstance(cwd, str) or not cwd or path_key(cwd) != want:
            continue
        native = _codex_native_from_row(row)
        if native is not None:
            out.append(native)
    return out


def _codex_native_from_row(row: dict) -> NativeSession | None:
    from agent_os.agent.session import _derive_name
    if _codex_row_is_own(row):
        return None
    first = (row.get("first_user_message") or "").strip()
    if not first:
        return None  # no user message
    rollout = row.get("rollout_path")
    if not isinstance(rollout, str) or not rollout:
        return None
    if not os.path.exists(rollout) and os.path.exists(rollout + ".zst"):
        rollout += ".zst"  # Codex compressed it; the index keeps the old name
    try:
        size = os.path.getsize(rollout)
    except OSError:
        return None  # the transcript is gone: nothing to open
    ms = row.get("updated_at_ms")
    last = (_iso_epoch(ms / 1000.0) if isinstance(ms, (int, float)) and ms
            else _iso_epoch(row.get("updated_at")))
    title = (row.get("title") or "").strip() or first
    return NativeSession(
        agent=AGENT_CODEX, native_id=str(row["id"]), path=rollout,
        title=_derive_name(title), last_activity=last, size_bytes=size,
        rollout_path=rollout,
    )


def _codex_member(native_id: str, home: str) -> NativeSession | None:
    index = os.path.join(home, "state_5.sqlite")
    rows = _codex_threads(index) if os.path.isfile(index) else None
    if rows is not None:
        row = next((r for r in rows if str(r.get("id")) == native_id), None)
        return _codex_native_from_row(row) if row else None
    from agent_os.agent.session import _derive_name
    for root, _dirs, files in os.walk(os.path.join(home, "sessions")):
        for fname in files:
            if not (fname.startswith("rollout-") and fname.endswith(f"{native_id}.jsonl")):
                continue
            fpath = os.path.join(root, fname)
            try:
                st = os.stat(fpath)
            except OSError:
                continue
            meta = _codex_rollout_metadata(fpath, st)
            if meta.get("id") == native_id and not meta.get("own") and meta.get("first_user"):
                return NativeSession(
                    agent=AGENT_CODEX, native_id=native_id, path=fpath,
                    title=_derive_name(meta["first_user"]),
                    last_activity=meta.get("last_activity") or _iso_epoch(st.st_mtime),
                    size_bytes=st.st_size, rollout_path=fpath)
    return None


# ---- pi / Kimi Code / MiniMax Code (spec 138 readers) -----------------------


def _history_native(agent: str, path: str | None, want: str | None = None) -> NativeSession | None:
    """A session file read by the agent's spec 138 reader as a listing entry,
    memoized by ``(path, mtime, size)``. None when nobody said anything in
    it, or when ``want`` is a folder key and the session ran elsewhere."""
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = ("history-meta", agent, path, st.st_mtime_ns, st.st_size)
    meta = _cache_get(key)
    if meta is None:
        meta = _history_metadata(agent, path, st)
        _cache_put(key, meta)
    if not meta:
        return None
    if want is not None and (not meta.get("cwd") or path_key(meta["cwd"]) != want):
        return None
    return NativeSession(
        agent=agent, native_id=meta["native_id"], path=path, title=meta.get("title"),
        last_activity=meta.get("last_activity"), size_bytes=st.st_size,
        rollout_path=path if agent == AGENT_PI else None,
    )


def _history_metadata(agent: str, path: str, st: os.stat_result) -> dict:
    from agent_os.agent.session import _derive_name
    hs = _HISTORY_SOURCES[agent].read(path)
    if hs is None:
        return {}
    return {"native_id": hs.native_id, "cwd": hs.cwd,
            "title": hs.title or _derive_name(hs.first_user_text or hs.last_user_text),
            "last_activity": hs.ended_at or _iso_epoch(st.st_mtime)}


def _pi_header(path: str) -> dict | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = ("pi-head", path, st.st_mtime_ns, st.st_size)
    head = _cache_get(key)
    if head is None:
        head = _pi_src.SOURCE.header(path) or {}
        _cache_put(key, head)
    return head


def _list_pi(workspace: str, home: str) -> list[NativeSession]:
    want = path_key(workspace)
    out: list[NativeSession] = []
    for path in _pi_src.SOURCE.session_files(home):
        cwd = (_pi_header(path) or {}).get("cwd")
        if not isinstance(cwd, str) or path_key(cwd) != want:
            continue  # the one-line header decides before a full read
        native = _history_native(AGENT_PI, path, want)
        if native is not None:
            out.append(native)
    return out


def _list_kimi(workspace: str, home: str) -> list[NativeSession]:
    want = path_key(workspace)
    out: list[NativeSession] = []
    for sdir in _kimi_src.SOURCE.session_dirs(home):
        state_path = os.path.join(sdir, "state.json")
        try:
            st = os.stat(state_path)
        except OSError:
            continue
        key = ("kimi-state", state_path, st.st_mtime_ns, st.st_size)
        state = _cache_get(key)
        if state is None:
            state = _kimi_src.SOURCE.state(sdir)
            _cache_put(key, state)
        cwd = state.get("cwd")
        if not isinstance(cwd, str) or path_key(cwd) != want:
            continue
        native = _history_native(
            AGENT_KIMI, os.path.join(sdir, "agents", "main", "wire.jsonl"), want)
        if native is not None:
            out.append(native)
    return out


def _minimax_messages_path(v2: str, row: dict) -> str | None:
    rel = row.get("history_relative_dir")
    if not isinstance(rel, str) or not rel or rel.startswith(("/", "\\")) or ".." in rel:
        return None
    path = os.path.join(v2, "sessions", *rel.split("/"), "messages.jsonl")
    return path if os.path.isfile(path) else None


def _list_minimax(workspace: str, home: str) -> list[NativeSession]:
    want = path_key(workspace)
    v2 = _minimax_src.SOURCE.root(home)
    out: list[NativeSession] = []
    for row in _minimax_src.SOURCE.runtime_rows(v2).values():
        cwd = row.get("workspace_dir")
        if not isinstance(cwd, str) or not cwd or path_key(cwd) != want:
            continue
        if not _minimax_src.SOURCE.is_listed(row):
            continue
        native = _history_native(AGENT_MINIMAX, _minimax_messages_path(v2, row), want)
        if native is not None:
            out.append(native)
    return out


def _codex_sessions_from_rollouts(sessions_dir: str, workspace: str) -> list[NativeSession]:
    """Fallback without the index: each rollout's ``session_meta`` head."""
    from agent_os.agent.session import _derive_name
    if not os.path.isdir(sessions_dir):
        return []
    want = path_key(workspace)
    out: list[NativeSession] = []
    for root, _dirs, files in os.walk(sessions_dir):
        for fname in files:
            if not (fname.startswith("rollout-") and fname.endswith(".jsonl")):
                continue
            fpath = os.path.join(root, fname)
            try:
                st = os.stat(fpath)
            except OSError:
                continue
            key = ("codex-rollout-meta", fpath, st.st_mtime_ns, st.st_size)
            meta = _cache_get(key)
            if meta is None:
                meta = _codex_rollout_metadata(fpath, st)
                _cache_put(key, meta)
            if not meta or not meta.get("cwd") or path_key(meta["cwd"]) != want:
                continue
            if meta.get("own") or not meta.get("first_user"):
                continue
            out.append(NativeSession(
                agent=AGENT_CODEX, native_id=meta["id"], path=fpath,
                title=_derive_name(meta["first_user"]),
                last_activity=meta.get("last_activity") or _iso_epoch(st.st_mtime),
                size_bytes=st.st_size, rollout_path=fpath,
            ))
    return out


def _codex_rollout_metadata(path: str, st: os.stat_result) -> dict:
    try:
        head, tail = _read_bounded(path, st.st_size)
    except OSError:
        return {}
    meta: dict = {}
    last_ts = None
    for rec in _json_rows(head + tail):
        payload = rec.get("payload") if isinstance(rec.get("payload"), dict) else {}
        if rec.get("type") == "session_meta" and "id" not in meta:
            if not payload.get("id"):
                continue
            meta["id"] = str(payload["id"])
            meta["cwd"] = payload.get("cwd")
            source = payload.get("source")
            meta["own"] = _codex_row_is_own({
                "originator": payload.get("originator"),
                "source": json.dumps(source) if isinstance(source, dict) else source,
            })
        elif (rec.get("type") == "response_item" and payload.get("type") == "message"
              and payload.get("role") == "user" and "first_user" not in meta):
            text = _codex_user_text(payload)
            if text:
                meta["first_user"] = text
        if rec.get("timestamp"):
            last_ts = rec["timestamp"]
    if "id" not in meta:
        return {}
    meta["last_activity"] = _norm_ts(last_ts)
    return meta


def _codex_user_text(payload: dict) -> str:
    parts = []
    for block in payload.get("content") or []:
        if not isinstance(block, dict) or block.get("type") not in ("input_text", "text"):
            continue
        text = block.get("text")
        if not isinstance(text, str):
            continue
        stripped = text.strip()
        if not stripped or _CODEX_TAG_BLOCK_RE.match(stripped) \
                or stripped.startswith(_CODEX_AGENTS_MD_PREFIX):
            continue
        parts.append(stripped)
    return "\n\n".join(parts).strip()


def _codex_assistant_text(payload: dict) -> str:
    parts = []
    for block in payload.get("content") or []:
        if isinstance(block, dict) and block.get("type") in ("output_text", "text"):
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------


@dataclass
class _Turn:
    user: str | None = None
    user_ts: str | None = None
    texts: list[str] = field(default_factory=list)
    tools: "OrderedDict[str, int]" = field(default_factory=OrderedDict)
    reply_ts: str | None = None
    # Orbital's own injected turn: never pushed, so it and its reply drop out.
    dropped: bool = False

    def has_output(self) -> bool:
        return bool(self.texts or self.tools)

    def reply(self) -> str:
        reply = "\n\n".join(self.texts)
        if self.tools:
            line = "(tools: " + ", ".join(f"{n} ×{c}" for n, c in self.tools.items()) + ")"
            reply = f"{reply}\n\n{line}" if reply else line
        return reply


@dataclass
class ConvertedTranscript:
    rows: list[dict]           # header + conversation rows, in order
    thread_record: dict        # the sub_agent_thread meta row (last row)
    truncated: bool
    last_activity: str | None

    def all_rows(self) -> list[dict]:
        return [*self.rows, self.thread_record]


def convert_transcript(agent: str, path: str, *, handle: str | None = None,
                       now: str | None = None, native_id: str | None = None,
                       rollout_path: str | None = None) -> ConvertedTranscript:
    """Convert one native transcript into Orbital rows (contract §6).

    ``handle`` is the worker slug the rows are attributed to (defaults to
    ``agent``); ``native_id`` the resume id (defaults to the id the transcript
    records); ``rollout_path`` the file resume reopens (Codex rollout, pi
    session; defaults to ``path``). Reads the whole file, streaming; keeps only
    the last ``MAX_TURNS`` turns and ``MAX_REPLY_BYTES`` of reply text.
    """
    handle = handle or agent
    now = now or datetime.now(timezone.utc).isoformat()
    turns: deque[_Turn] = deque()
    dropped = False

    def push(turn: _Turn) -> None:
        nonlocal dropped
        turns.append(turn)
        if len(turns) > MAX_TURNS + 1:  # +1: the open turn may still be empty
            old = turns.popleft()
            if old.user or old.has_output():
                dropped = True

    if agent == AGENT_CLAUDE:
        found_id = _claude_turns(path, push)
        native_id = native_id or found_id or os.path.basename(path)[:-len(".jsonl")]
    elif agent == AGENT_CODEX:
        found_id = _codex_turns(path, push)
        native_id = native_id or found_id
        rollout_path = rollout_path or path
    elif agent in _HISTORY_SOURCES:
        found_id = _history_turns(agent, path, push)
        native_id = native_id or found_id
        if agent == AGENT_PI:
            rollout_path = rollout_path or path
    else:
        raise ValueError(f"unknown agent {agent!r}")

    kept = [t for t in turns if t.user or t.has_output()]
    if len(kept) > MAX_TURNS:
        kept = kept[-MAX_TURNS:]
        dropped = True
    # Reply budget, newest first.
    budget = MAX_REPLY_BYTES
    start = len(kept)
    for i in range(len(kept) - 1, -1, -1):
        size = len(kept[i].reply().encode("utf-8"))
        if size > budget:
            break
        budget -= size
        start = i
    if start > 0:
        kept = kept[start:]
        dropped = True

    conversation: list[dict] = []
    for turn in kept:
        if turn.user:
            conversation.append({
                "role": "user", "source": "user", "content": turn.user,
                # ``target`` is what the live pinned send stamps on a user row
                # addressed to a worker; the recap builder reads it to skip
                # rows the resumed native thread already holds.
                "target": handle,
                "timestamp": _norm_ts(turn.user_ts) or now,
                "_meta": {"imported": True},
            })
        if turn.has_output():
            reply = turn.reply()
            content = f"[Sub-agent] {handle} completed. Summary: {reply[:_SUMMARY_CHARS]}"
            conversation.append({
                "role": "system", "source": "daemon",
                "timestamp": (_norm_ts(turn.reply_ts) or _norm_ts(turn.user_ts) or now),
                "content": content,
                "_meta": {
                    "event": "sub_agent_terminal", "kind": "completed",
                    "handle": handle, "reply": reply,
                    "display_content": content, "suppress_wake": True,
                    "imported": True,
                },
            })

    last_activity = conversation[-1]["timestamp"] if conversation else None
    first_ts = conversation[0]["timestamp"] if conversation else now
    agent_name = AGENT_DISPLAY_NAMES.get(agent, agent)
    when = (last_activity or now)[:10]
    header_content = f"[Imported] Conversation imported from {agent_name} (last active {when})"
    if dropped:
        header_content += " — earlier messages not imported"
    header = {
        "role": "system", "source": "daemon", "content": header_content,
        "timestamp": first_ts,
        "_meta": {
            "event": "imported_transcript", "agent": agent, "native_id": native_id,
            "last_activity": last_activity, "truncated": dropped,
            "suppress_wake": True,
        },
    }
    thread = {"session_id": native_id, "model": None, "last_used_at": now}
    if agent in _RESUME_PATH_AGENTS and rollout_path:
        thread["rollout_path"] = rollout_path
    thread_record = {
        "role": "meta", "event": "sub_agent_thread", "timestamp": now,
        "handle": handle, "thread": thread,
    }
    return ConvertedTranscript(
        rows=[header, *conversation], thread_record=thread_record,
        truncated=dropped, last_activity=last_activity,
    )


def _iter_file_rows(path: str):
    from agent_os.onboarding.sources._common import open_text

    with open_text(path) as fh:
        yield from _json_rows(fh)


def _claude_turns(path: str, push) -> str | None:
    """Stream a Claude transcript into turns; returns its sessionId."""
    session_id = None
    current: _Turn | None = None
    for rec in _iter_file_rows(path):
        if session_id is None and isinstance(rec.get("sessionId"), str):
            session_id = rec["sessionId"]
        if rec.get("isSidechain") is True:
            continue
        rtype = rec.get("type")
        if rtype == "user":
            text = _claude_user_text(rec)
            if text is None:
                continue  # tool results, meta rows, command plumbing
            text = _cleaned(text)
            if text is None:
                current = _Turn(dropped=True)  # Orbital's own turn
                continue
            current = _Turn(user=text, user_ts=rec.get("timestamp"))
            push(current)
        elif rtype == "assistant":
            msg = rec.get("message")
            if not isinstance(msg, dict) or msg.get("model") == "<synthetic>":
                continue
            if current is None:
                current = _Turn()
                push(current)
            content = msg.get("content")
            blocks = ([{"type": "text", "text": content}] if isinstance(content, str)
                      else content if isinstance(content, list) else [])
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text = (block.get("text") or "").strip()
                    if text:
                        current.texts.append(text)
                        current.reply_ts = rec.get("timestamp") or current.reply_ts
                elif block.get("type") == "tool_use":
                    name = block.get("name") or "tool"
                    current.tools[name] = current.tools.get(name, 0) + 1
                    current.reply_ts = rec.get("timestamp") or current.reply_ts
                # thinking / redacted_thinking / anything else: dropped
    return session_id


def _codex_turns(path: str, push) -> str | None:
    """Stream a Codex rollout into turns; returns the session_meta id."""
    thread_id = None
    current: _Turn | None = None
    for rec in _iter_file_rows(path):
        rtype = rec.get("type")
        payload = rec.get("payload") if isinstance(rec.get("payload"), dict) else {}
        if rtype == "session_meta":
            if thread_id is None and payload.get("id"):
                thread_id = str(payload["id"])
            continue
        if rtype != "response_item":
            continue  # event_msg, turn_context, world_state, token counts: dropped
        ptype = payload.get("type")
        ts = rec.get("timestamp")
        if ptype == "message":
            role = payload.get("role")
            if role == "user":
                text = _codex_user_text(payload)
                if text:
                    text = _cleaned(text)
                    if text is None:
                        current = _Turn(dropped=True)  # Orbital's own turn
                    elif (current is not None and not current.dropped
                          and current.user is None and not current.has_output()):
                        current.user, current.user_ts = text, ts
                    elif current is not None and not current.dropped and not current.has_output():
                        current.user = f"{current.user}\n\n{text}"
                    else:
                        current = _Turn(user=text, user_ts=ts)
                        push(current)
                elif current is None or current.has_output() or current.dropped:
                    # Injected context only (environment, AGENTS.md, an
                    # automation heartbeat): a turn boundary with no user row.
                    current = _Turn(user_ts=ts)
                    push(current)
            elif role == "assistant":
                text = _codex_assistant_text(payload)
                if text:
                    if current is None:
                        current = _Turn()
                        push(current)
                    current.texts.append(text)
                    current.reply_ts = ts or current.reply_ts
            # developer / system: dropped
        elif isinstance(ptype, str) and ptype.endswith("_call"):
            if current is None:
                current = _Turn()
                push(current)
            name = payload.get("name") or ptype[:-len("_call")]
            current.tools[name] = current.tools.get(name, 0) + 1
            current.reply_ts = ts or current.reply_ts
        # *_call_output, reasoning, anything else: dropped
    return thread_id


def _history_turns(agent: str, path: str, push) -> str | None:
    """A spec 138 reader's turns, cleaned; returns the native id."""
    transcript = _HISTORY_SOURCES[agent].transcript(path)
    if transcript is None:
        return None
    for turn, kind, text in cleaned_turns(transcript):
        push(_Turn(user=text if kind is not None else None, user_ts=turn.user_ts,
                   texts=list(turn.texts), tools=OrderedDict(turn.tools),
                   reply_ts=turn.reply_ts))
    return transcript.native_id


def _cleaned(text: str) -> str | None:
    """The user's words in a native user turn, or None when the whole turn is
    Orbital's (an automation turn keeps its task)."""
    cleaned = clean_user_text(text)
    if cleaned.kind == KIND_INJECTED:
        return None
    return cleaned.text


# --------------------------------------------------------------------------
# Materialization
# --------------------------------------------------------------------------


def materialize_session(workspace: str, native: NativeSession, *,
                        project_id: str = "", now: str | None = None) -> str:
    """Write ``native`` as an Orbital session JSONL; returns its path.

    The file is ``{workspace}/orbital/sessions/{imp id}.jsonl``: the
    session_start meta first (``pinned_target`` = the agent, ``origin``
    ``imported``, ``imported_from``, ``name``), then the converted rows, then
    the ``sub_agent_thread`` resume record written by
    ``Session.set_sub_agent_thread`` itself, so the shape is the real one.

    Written to a temp file in the same directory and renamed into place: a
    crash mid-write never leaves a half transcript that a later open would
    take for the finished one. The caller serializes concurrent calls and
    skips when the file already exists.
    """
    from agent_os.agent.session import Session
    from agent_os.utils.file_lock import session_lock

    sid = native.session_id
    conv = convert_transcript(native.agent, native.path, handle=native.agent,
                              now=now, native_id=native.native_id,
                              rollout_path=native.rollout_path)
    session = Session.new(sid, workspace, project_id, session_id=sid, origin="imported")
    final_path = session._filepath
    tmp_path = os.path.join(os.path.dirname(final_path), f".{sid}.importing")
    try:
        os.remove(tmp_path)  # a leftover from a crashed attempt
    except FileNotFoundError:
        pass
    session._filepath = tmp_path
    session._file_lock = session_lock(tmp_path)
    session.pinned_target = native.agent
    if native.title:
        session.name = native.title
    session._apply_meta_fields(
        pinned_target=native.agent,
        imported_from={"agent": native.agent, "native_id": native.native_id},
        **({"name": native.title} if native.title else {}),
    )
    try:
        for row in conv.rows:
            session.append(dict(row))
        session.set_sub_agent_thread(
            native.agent, session_id=native.native_id,
            # Codex: the rollout. pi: the native session file the pi
            # transport reopens with ``--session <file>`` (contract §5).
            rollout_path=native.rollout_path if native.agent in _RESUME_PATH_AGENTS else None,
        )
        os.replace(tmp_path, final_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    finally:
        try:
            os.remove(tmp_path + ".lock")  # Windows sentinel lock, if any
        except OSError:
            pass
    return final_path


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _iso_epoch(epoch) -> str | None:
    if not isinstance(epoch, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _norm_ts(value) -> str | None:
    """A native timestamp as Orbital writes them (UTC ``isoformat()``), so
    imported rows sort with Orbital's own by plain string comparison."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()
