# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The user's agent history, read once and kept (spec 138).

* :class:`HistoryScanner` — reads every session of every loaded
  :mod:`~agent_os.onboarding.sources` reader in a worker thread, with an
  on-disk cache keyed by ``(path, mtime, size)`` so a re-read only parses what
  changed. :meth:`HistoryScanner.status` is what ``GET /onboarding/history``
  returns and never blocks.
* :func:`analyze` — the three grouping passes (folder, files touched, what
  leftovers share) over the scanned sessions. No model.
* :func:`select_left_off` — the "Last time" card and the "Didn't finish"
  list of one project.

Everything here is local and read-only against the agents' own files.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
import subprocess
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from agent_os.onboarding.history_paths import HistoryEnv, has_project_instructions, history_home
from agent_os.onboarding.history_types import HistorySession, PrivateNote, StopInfo
from agent_os.utils.subprocess_flags import win_no_window_flags

logger = logging.getLogger(__name__)

CACHE_VERSION = 1
CACHE_DIRNAME = "history"
CACHE_FILENAME = "sessions-cache.json"

# Pass 1: a folder with this many sessions is a project even without git or
# an AGENTS.md / CLAUDE.md.
MIN_FOLDER_SESSIONS = 3

# Pass 2 weights and thresholds (spec "Grouping" 2).
_WEIGHT_EDIT, _WEIGHT_READ, _WEIGHT_MENTION = 3, 2, 1
_PER_PATH_CAP = 5
_ATTACH_SHARE = 0.6
_ATTACH_MIN_SCORE = 4

# Pass 3 opening-request key.
_OPENER_CHARS = 24
_OPENER_MIN = 12

LEFT_OFF_UNFINISHED_MAX = 3
LEFT_OFF_WINDOW = timedelta(days=30)


# --------------------------------------------------------------------------
# Session (de)serialisation for the cache
# --------------------------------------------------------------------------


def session_to_dict(s: HistorySession) -> dict:
    return dataclasses.asdict(s)


def session_from_dict(d: dict) -> HistorySession | None:
    try:
        d = dict(d)
        stop = d.pop("stop", None)
        known = {f.name for f in dataclasses.fields(HistorySession)}
        s = HistorySession(**{k: v for k, v in d.items() if k in known})
        s.stop = StopInfo(**stop) if isinstance(stop, dict) else None
        return s
    except (TypeError, ValueError):
        return None


def session_key(s: HistorySession) -> tuple[str, str]:
    return (s.agent, s.native_id)


# --------------------------------------------------------------------------
# Scanner
# --------------------------------------------------------------------------


class HistoryScanner:
    """Reads the history in the background and keeps it fresh.

    ``status`` is ``scanning`` until the first full read and analysis are
    done, ``ready`` afterwards (a later refresh keeps serving the last
    result while it runs)."""

    def __init__(self, data_dir: str | None, *, home: str | None = None,
                 sources=None, env: HistoryEnv | None = None, on_ready=None):
        self._data_dir = data_dir
        self._home = home
        self._sources = sources
        self._env = env
        self._on_ready = on_ready
        self._lock = threading.Lock()          # guards the fields below
        self._collect_lock = threading.Lock()  # one read pass at a time
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._cache: dict[str, dict] = {}      # path -> {mtime_ns, size, session}
        self._cache_loaded = False
        self._sessions: list[HistorySession] = []
        self._analysis: Analysis | None = None
        self._by_agent: dict[str, int] = {}
        # Spec 153: the agents' private notes, read once per scan.
        self._notes: list[PrivateNote] = []
        self._notes_read = False
        self._scan_seconds: float | None = None

    # -- configuration -------------------------------------------------------

    def home(self) -> str:
        return self._home or history_home()

    def env(self) -> HistoryEnv:
        return self._env or HistoryEnv.for_home(self.home())

    def sources(self) -> list:
        if self._sources is None:
            from agent_os.onboarding.sources import load_sources
            self._sources = load_sources()
        return self._sources

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Start the first scan if none ran yet (idempotent, non-blocking)."""
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name="history-scan", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        started = time.monotonic()
        try:
            sessions = self.collect()
            self.collect_notes()
            analysis = analyze(sessions, self.env())
            with self._lock:
                self._analysis = analysis
        except Exception:
            logger.exception("history scan failed")
            with self._lock:
                self._analysis = self._analysis or Analysis()
        finally:
            self._scan_seconds = time.monotonic() - started
            self._ready.set()
            logger.info("history scan: %d sessions in %.2fs", len(self._sessions),
                        self._scan_seconds)
            if self._on_ready is not None:
                try:
                    self._on_ready(self)
                except Exception:
                    logger.warning("history scan: on_ready hook failed", exc_info=True)

    def refresh(self) -> bool:
        """Re-read what changed since the last pass and re-analyze, once the
        first scan is done (before that, the running scan is the fresh read).
        Incremental: every session file is stat'ed, only new or changed ones
        are parsed (~0.1 s on 900 sessions), then the notes (~0.2 s) and the
        analysis (~10 ms). Blocking — call off the event loop. Returns
        whether it ran; a failure keeps serving the last result."""
        if not self._ready.is_set():
            return False
        try:
            sessions = self.collect()
            self.collect_notes()
            analysis = analyze(sessions, self.env())
        except Exception:
            logger.warning("history refresh failed", exc_info=True)
            return False
        with self._lock:
            self._analysis = analysis
        return True

    def wait(self, timeout: float | None = None) -> bool:
        self.start()
        return self._ready.wait(timeout)

    def is_ready(self) -> bool:
        return self._ready.is_set()

    # -- reads ---------------------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            by_agent = dict(self._by_agent)
        if not by_agent:
            # Every loaded reader has a key, 0 until its sessions are counted.
            by_agent = {getattr(src, "agent", "?"): 0 for src in self.sources()}
        notes_by_agent, notes_by_folder = self._notes_summary()
        return {
            "status": "ready" if self._ready.is_set() else "scanning",
            "sessions_total": sum(by_agent.values()),
            "by_agent": by_agent,
            "notes_by_agent": notes_by_agent,
            "notes_by_folder": notes_by_folder,
        }

    # -- private notes (spec 153) --------------------------------------------

    def collect_notes(self) -> list[PrivateNote]:
        """Read every loaded agent's private notes (only Claude Code has a
        reader). Blocking; the scan calls it once, right after the sessions."""
        from agent_os.onboarding.sources import claude_code

        found: list[PrivateNote] = []
        read = False
        if any(getattr(src, "agent", None) == claude_code.AGENT for src in self.sources()):
            read = True
            try:
                found = claude_code.notes(self.home())
            except Exception:
                logger.warning("history: Claude notes read failed", exc_info=True)
        with self._lock:
            self._notes = found
            self._notes_read = read
        return found

    def notes(self) -> list[PrivateNote]:
        with self._lock:
            return list(self._notes)

    def notes_for(self, workspace: str) -> list[PrivateNote]:
        """The notes whose recorded folder is ``workspace`` (same identity key
        as the import's already-imported match)."""
        from agent_os.onboarding.import_scanner import path_key

        want = path_key(workspace)
        return [n for n in self.notes() if n.folder and path_key(n.folder) == want]

    def _notes_summary(self) -> tuple[dict[str, int], list[dict]]:
        with self._lock:
            notes, read = list(self._notes), self._notes_read
        by_agent: dict[str, int] = {}
        if read:
            from agent_os.onboarding.sources import claude_code
            by_agent[claude_code.AGENT] = 0
        by_folder: dict[tuple, dict] = {}
        for n in notes:
            by_agent[n.agent] = by_agent.get(n.agent, 0) + 1
            key = (n.agent, n.folder, n.encoded)
            row = by_folder.setdefault(key, {"agent": n.agent, "folder": n.folder,
                                             "encoded": n.encoded, "n": 0})
            row["n"] += 1
        rows = sorted(by_folder.values(),
                      key=lambda r: (-r["n"], r["folder"] is None, r["folder"] or r["encoded"]))
        return by_agent, rows

    def sessions(self) -> list[HistorySession]:
        with self._lock:
            return list(self._sessions)

    def analysis(self) -> "Analysis | None":
        with self._lock:
            return self._analysis if self._ready.is_set() else None

    @property
    def scan_seconds(self) -> float | None:
        return self._scan_seconds

    # -- the read pass -------------------------------------------------------

    def collect(self) -> list[HistorySession]:
        """Stat every session file, parse the ones whose ``(mtime, size)``
        changed, persist the cache. Blocking — call off the event loop."""
        with self._collect_lock:
            if not self._cache_loaded:
                self._cache = self._load_cache()
                self._cache_loaded = True
            home = self.home()
            fresh: dict[str, dict] = {}
            sessions: list[HistorySession] = []
            by_agent: dict[str, int] = {}
            changed = False
            for src in self.sources():
                agent = getattr(src, "agent", "?")
                by_agent.setdefault(agent, 0)
                try:
                    files = src.session_files(home)
                except Exception:
                    logger.warning("history: %s listing failed", agent, exc_info=True)
                    continue
                for path in files:
                    try:
                        st = os.stat(path)
                    except OSError:
                        continue
                    entry = self._cache.get(path)
                    if (entry is None or entry.get("mtime_ns") != st.st_mtime_ns
                            or entry.get("size") != st.st_size or entry.get("agent") != agent):
                        try:
                            parsed = src.read(path)
                        except Exception:
                            logger.warning("history: %s read failed for %s", agent, path,
                                           exc_info=True)
                            parsed = None
                        entry = {"mtime_ns": st.st_mtime_ns, "size": st.st_size, "agent": agent,
                                 "session": session_to_dict(parsed) if parsed else None}
                        changed = True
                    fresh[path] = entry
                    if entry.get("session"):
                        s = session_from_dict(entry["session"])
                        if s is not None:
                            sessions.append(s)
                            by_agent[agent] += 1
            if set(fresh) != set(self._cache):
                changed = True
            self._cache = fresh
            if changed:
                self._save_cache()
            with self._lock:
                self._sessions = sessions
                self._by_agent = by_agent
            return sessions

    def _cache_path(self) -> str | None:
        if not self._data_dir:
            return None
        return os.path.join(self._data_dir, CACHE_DIRNAME, CACHE_FILENAME)

    def _load_cache(self) -> dict:
        path = self._cache_path()
        if not path:
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
            return {}
        if data.get("home") != self.home():
            return {}
        entries = data.get("entries")
        return entries if isinstance(entries, dict) else {}

    def _save_cache(self) -> None:
        path = self._cache_path()
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"version": CACHE_VERSION, "home": self.home(),
                           "entries": self._cache}, fh, ensure_ascii=False)
            os.replace(tmp, path)
        except OSError:
            logger.warning("history: could not write the cache", exc_info=True)


# --------------------------------------------------------------------------
# Git
# --------------------------------------------------------------------------

_git_cache: dict[str, str | None] = {}
_git_lock = threading.Lock()


def git_main_root(path: str) -> str | None:
    """The main checkout root of the repo holding ``path`` (a worktree folds
    onto its main repo), or None outside a repo. Cached per path."""
    with _git_lock:
        if path in _git_cache:
            return _git_cache[path]
    root = None
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "--path-format=absolute",
             "--show-toplevel", "--git-common-dir"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}, creationflags=win_no_window_flags())
        if out.returncode == 0:
            lines = out.stdout.strip().splitlines()
            if len(lines) >= 2:
                top, common = lines[0], lines[1]
                root = os.path.dirname(common) if os.path.basename(common) == ".git" else top
                root = os.path.normpath(root)
    except (OSError, subprocess.SubprocessError):
        root = None
    with _git_lock:
        _git_cache[path] = root
    return root


def clear_git_cache() -> None:
    with _git_lock:
        _git_cache.clear()


def _git(root: str, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "--no-optional-locks", "-C", root, *args],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=10, creationflags=win_no_window_flags())
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def git_facts(workspace: str, since_iso: str | None, session_branch: str | None) -> dict | None:
    """``{branch, session_branch, uncommitted, commits_since}`` of the repo at
    ``workspace``, or None when it is not a git repo."""
    if not workspace or git_main_root(workspace) is None:
        return None
    branch = (_git(workspace, "rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None
    status = _git(workspace, "status", "--porcelain")
    uncommitted = len([ln for ln in (status or "").splitlines() if ln.strip()])
    commits = 0
    if since_iso:
        out = _git(workspace, "rev-list", "--count", f"--since={since_iso}", "HEAD")
        try:
            commits = int((out or "0").strip() or 0)
        except ValueError:
            commits = 0
    return {"branch": branch, "session_branch": session_branch,
            "uncommitted": uncommitted, "commits_since": commits}


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------

_PROBE_RE = re.compile(
    r"^(?:please\s+)?(?:reply|respond|answer|say|print|output|return)\b.{0,80}?"
    r"\b(?:exactly|only|single word|one word|just)\b", re.IGNORECASE | re.DOTALL)


def is_probe_text(text: str | None) -> bool:
    if not text:
        return True
    t = text.strip()
    if len(t.split()) <= 1:
        return True
    return len(t) <= 160 and bool(_PROBE_RE.match(t))


def is_noise(s: HistorySession, env: HistoryEnv) -> bool:
    """Sessions never shown or grouped: Orbital's own worker runs, automation
    runs, sessions under temp roots, and one-word probes ("PING", "Reply with
    exactly …")."""
    if s.orbital_started or s.automation:
        return True
    if s.cwd and env.is_temp(s.cwd):
        return True
    if s.user_turns == 0:
        return True
    if s.user_turns <= 2 and is_probe_text(s.first_user_text) and is_probe_text(s.last_user_text):
        return True
    return False


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------


@dataclass
class FolderProject:
    root: str
    is_git: bool
    has_agent_md: bool
    minor: bool
    sessions: list[HistorySession] = field(default_factory=list)


@dataclass
class Group:
    id: str
    members: list[HistorySession]
    reasons: list[dict]
    suggested_folder: str | None

    def to_dict(self) -> dict:
        by_agent: Counter = Counter(s.agent for s in self.members)
        return {
            "id": self.id,
            "session_count": len(self.members),
            "last_activity": max((s.ended_at or "" for s in self.members), default="") or None,
            "by_agent": dict(by_agent),
            "reasons": self.reasons,
            "suggested_folder": self.suggested_folder,
            "members": [{"agent": s.agent, "native_id": s.native_id} for s in self.members],
        }


@dataclass
class Analysis:
    projects: dict[str, FolderProject] = field(default_factory=dict)
    groups: list[Group] = field(default_factory=list)
    # session key -> project root it was placed in (pass 1 or 2)
    placed: dict[tuple[str, str], str] = field(default_factory=dict)

    def group(self, group_id: str) -> Group | None:
        return next((g for g in self.groups if g.id == group_id), None)

    def members_for(self, root: str) -> list[HistorySession]:
        """Sessions placed in ``root`` (by folder, worktree or files)."""
        project = self.projects.get(root)
        return list(project.sessions) if project else []


def opener_key(text: str | None) -> str | None:
    t = re.sub(r"[\W_]+", "", (text or "").lower())
    return t[:_OPENER_CHARS] if len(t) >= _OPENER_MIN else None


def _folder_of(s: HistorySession, env: HistoryEnv, isdir) -> str | None:
    """Pass 1 home: the git main root, else the folder itself; None for a
    session without a live folder."""
    if not s.cwd:
        return None
    cwd = os.path.normpath(s.cwd)
    if not isdir(cwd):
        return None
    root = git_main_root(cwd)
    if root and not env.is_container(root) and not env.is_temp(root):
        return root
    return cwd


def analyze(sessions: list[HistorySession], env: HistoryEnv, *, isdir=os.path.isdir) -> Analysis:
    live = [s for s in sessions if not is_noise(s, env)]
    result = Analysis()

    # Pass 1 — by folder.
    by_folder: dict[str, list[HistorySession]] = defaultdict(list)
    for s in live:
        home = _folder_of(s, env, isdir)
        if home is not None:
            by_folder[home].append(s)
    for folder, members in by_folder.items():
        if env.is_container(folder) or env.is_temp(folder) or not isdir(folder):
            continue
        is_git = git_main_root(folder) is not None
        if env.is_chat_folder(folder) and not is_git:
            continue  # a Codex app chat: its sessions go on to passes 2-3
        has_md = has_project_instructions(folder)
        minor = not (is_git or has_md or len(members) >= MIN_FOLDER_SESSIONS)
        result.projects[folder] = FolderProject(folder, is_git, has_md, minor, list(members))
        for s in members:
            result.placed[session_key(s)] = folder

    # Pass 2 — by files touched, onto real (non-minor) projects only.
    roots = sorted((r for r, p in result.projects.items() if not p.minor), key=len, reverse=True)

    def project_of(path: str) -> str | None:
        p = os.path.normpath(path)
        for r in roots:
            if p == r or p.startswith(r.rstrip(os.sep) + os.sep):
                return r
        return None

    leftovers: list[HistorySession] = []
    for s in live:
        if session_key(s) in result.placed:
            continue
        score: Counter = Counter()
        for weight, bags in ((_WEIGHT_EDIT, (s.edits, s.creates)), (_WEIGHT_READ, (s.reads,)),
                             (_WEIGHT_MENTION, (s.mentions,))):
            for bag in bags:
                for path, n in bag.items():
                    r = project_of(path)
                    if r:
                        score[r] += weight * min(int(n), _PER_PATH_CAP)
        if score:
            top, value = score.most_common(1)[0]
            if value >= _ATTACH_MIN_SCORE and value / sum(score.values()) >= _ATTACH_SHARE:
                result.projects[top].sessions.append(s)
                result.placed[session_key(s)] = top
                continue
        leftovers.append(s)

    # Pass 3 — what leftovers share.
    result.groups = _link_leftovers(leftovers, env, isdir)
    return result


def _written(s: HistorySession, env: HistoryEnv) -> list[str]:
    return [p for p in {**s.edits, **s.creates} if not env.is_temp(p) and not env.is_private(p)]


def _link_leftovers(leftovers: list[HistorySession], env: HistoryEnv, isdir) -> list[Group]:
    parent = list(range(len(leftovers)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_key: dict[tuple, list[int]] = defaultdict(list)
    for i, s in enumerate(leftovers):
        keys = {("file", p) for p in _written(s, env)}
        keys |= {("skill", k) for k in s.skills}
        op = opener_key(s.first_user_text)
        if op:
            keys.add(("opener", op))
        if s.cwd:
            cwd = os.path.normpath(s.cwd)
            if not env.is_container(cwd) and not env.is_temp(cwd):
                keys.add(("cwd", cwd))
        for k in keys:
            by_key[k].append(i)
    for idxs in by_key.values():
        for j in idxs[1:]:
            parent[find(j)] = find(idxs[0])

    clusters: dict[int, list[HistorySession]] = defaultdict(list)
    for i, s in enumerate(leftovers):
        clusters[find(i)].append(s)

    groups: list[Group] = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda s: s.ended_at or "", reverse=True)
        ident = "\n".join(sorted(f"{s.agent}:{s.native_id}" for s in members))
        gid = "grp_" + hashlib.sha1(ident.encode("utf-8")).hexdigest()[:8]
        groups.append(Group(gid, members, _reasons(members, env),
                            _suggest_folder(members, env, isdir)))
    groups.sort(key=lambda g: (len(g.members), max(s.ended_at or "" for s in g.members)),
                reverse=True)
    return groups


def _reasons(members: list[HistorySession], env: HistoryEnv) -> list[dict]:
    reasons: list[dict] = []
    files: Counter = Counter(p for s in members for p in set(_written(s, env)))
    skills: Counter = Counter(k for s in members for k in set(s.skills))
    openers: Counter = Counter(op for s in members for op in [opener_key(s.first_user_text)] if op)
    folders: Counter = Counter(
        os.path.normpath(s.cwd) for s in members
        if s.cwd and not env.is_container(os.path.normpath(s.cwd))
        and not env.is_temp(os.path.normpath(s.cwd)))
    if skills and skills.most_common(1)[0][1] >= 2:
        name, n = skills.most_common(1)[0]
        reasons.append({"kind": "same_skill", "value": name, "count": n})
    if files and files.most_common(1)[0][1] >= 2:
        path, n = files.most_common(1)[0]
        reasons.append({"kind": "same_files", "value": os.path.basename(path), "count": n})
    if openers and openers.most_common(1)[0][1] >= 2:
        key, n = openers.most_common(1)[0]
        first = min((s for s in members if opener_key(s.first_user_text) == key),
                    key=lambda s: s.started_at or "")
        text = " ".join((first.first_user_text or "").split())
        reasons.append({"kind": "same_request", "value": text[:40].rstrip(), "count": n})
    if folders and folders.most_common(1)[0][1] >= 2:
        folder, n = folders.most_common(1)[0]
        reasons.append({"kind": "same_folder", "value": folder, "count": n})
    reasons.sort(key=lambda r: r["count"], reverse=True)
    return reasons


def _suggest_folder(members: list[HistorySession], env: HistoryEnv, isdir) -> str | None:
    """The folder holding most of the group's written files: every written
    file counts toward each of its ancestors; containers, temp and the
    agents' own dirs never qualify; the deepest folder wins a tie."""
    counts: Counter = Counter()
    from agent_os.onboarding.history_paths import _key
    home = _key(env.home).rstrip(os.sep) + os.sep
    for s in members:
        for path in set(_written(s, env)):
            d = os.path.dirname(os.path.normpath(path))
            while d and d != os.path.dirname(d):
                above_home = home.startswith(_key(d).rstrip(os.sep) + os.sep)
                if not (above_home or env.is_container(d) or env.is_temp(d)
                        or env.is_private(d)):
                    counts[d] += 1
                d = os.path.dirname(d)
    best = None
    for folder, n in counts.items():
        if not isdir(folder):
            continue
        if best is None or (n, len(folder)) > (counts[best], len(best)):
            best = folder
    return best


# --------------------------------------------------------------------------
# Left-off cards
# --------------------------------------------------------------------------


@dataclass
class LeftOffEntry:
    session_id: str
    agent: str | None
    user_text: str | None
    at: str | None
    stop: StopInfo | None = None
    branch: str | None = None


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def select_left_off(entries: list[LeftOffEntry], *, now: datetime | None = None) -> dict:
    """``{"last_time": entry|None, "unfinished": [entry…]}`` (contract §3).

    Last time = the newest finished session with the user's words. Didn't
    finish = abnormal stops of the last 30 days, newest first, at most 3,
    never repeating the Last time session."""
    now = now or datetime.now(timezone.utc)
    usable = [e for e in entries if e.user_text and _parse_iso(e.at)]
    usable.sort(key=lambda e: _parse_iso(e.at), reverse=True)
    last = next((e for e in usable if e.stop is None), None)
    cutoff = now - LEFT_OFF_WINDOW
    unfinished = [e for e in usable
                  if e.stop is not None and _parse_iso(e.at) >= cutoff
                  and (last is None or e.session_id != last.session_id)]
    return {"last_time": last, "unfinished": unfinished[:LEFT_OFF_UNFINISHED_MAX]}


def resets_at_if_future(stop: StopInfo | None, now: datetime | None = None) -> str | None:
    if stop is None or not stop.resets_at:
        return None
    at = _parse_iso(stop.resets_at)
    now = now or datetime.now(timezone.utc)
    return stop.resets_at if at and at > now else None
