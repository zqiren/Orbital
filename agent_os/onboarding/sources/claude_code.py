# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Claude Code history: ``~/.claude/projects/<encoded cwd>/<session>.jsonl``.

Also the agent's private notes (spec 153): ``<encoded cwd>/memory/*.md``
plus a ``MEMORY.md`` index of one ``- [Title](file.md) — hook`` line per
note, read by :func:`notes`.

One transcript per session. Records this reader uses: ``user`` (the user's
words, tool results, interrupt markers), ``assistant`` (text / tool_use
blocks; ``isApiErrorMessage`` rows carry the error the turn died on), the
title rows, and the per-record ``cwd`` / ``gitBranch`` / ``entrypoint``.
Sidechain (Task sub-agent) records are skipped. Read-only.
"""

from __future__ import annotations

import logging
import os
import re

import yaml

from agent_os.onboarding.history_paths import claude_dir
from agent_os.onboarding.history_types import (
    CAUSE_INTERRUPTED, CAUSE_NETWORK_AUTH, CAUSE_USAGE_LIMIT, HistorySession, PrivateNote,
)
from agent_os.onboarding.sources._common import (
    TurnState, add_skill, bump, iter_json_lines, norm_ts, note_mentions, touch,
)

logger = logging.getLogger(__name__)

AGENT = "claude-code"

# Programmatic SDK sessions are Orbital's own claude-code worker runs.
_ORBITAL_ENTRYPOINTS = frozenset({"sdk-py", "sdk-ts"})

# User rows that are slash-command plumbing, not the user's words.
_MACHINE_PREFIXES = (
    "<command-name>", "<command-message>", "<command-args>",
    "<local-command-stdout>", "<local-command-stderr>", "<local-command-caveat>",
    "Caveat: The messages below were generated",
)
_INTERRUPT_PREFIX = "[Request interrupted by user"

_EDIT_TOOLS = frozenset({"Edit", "MultiEdit", "NotebookEdit"})
_CREATE_TOOLS = frozenset({"Write"})
_READ_TOOLS = frozenset({"Read", "Grep", "Glob", "LS"})

# ``error`` on an API-error row, when its text names no cause.
_ERROR_FIELD_CAUSES = {
    "rate_limit": CAUSE_USAGE_LIMIT,
    "billing_error": CAUSE_USAGE_LIMIT,
    "authentication_failed": CAUSE_NETWORK_AUTH,
}


class ClaudeCodeSource:
    agent = AGENT

    def session_files(self, home: str) -> list[str]:
        projects = os.path.join(claude_dir(home), "projects")
        out: list[str] = []
        try:
            dirs = list(os.scandir(projects))
        except OSError:
            return out
        for d in dirs:
            try:
                if not d.is_dir():
                    continue
                for f in os.scandir(d.path):
                    if f.name.endswith(".jsonl") and f.is_file():
                        out.append(f.path)
            except OSError:
                continue
        return out

    def read(self, path: str) -> HistorySession | None:
        try:
            return _read(path)
        except Exception:
            logger.warning("history: could not read Claude transcript %s", path, exc_info=True)
            return None


def _user_text(content) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text") for b in content
                 if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)]
        return "\n\n".join(parts) if parts else None
    return None


def _assistant_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text") or "" for b in content
                         if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _read(path: str) -> HistorySession | None:
    stem = os.path.basename(path)[:-len(".jsonl")] if path.endswith(".jsonl") else os.path.basename(path)
    s = HistorySession(agent=AGENT, native_id=stem, path=path, resume_path=path)
    turn = TurnState(s)
    title = custom_title = None
    seen_id = False
    any_message = False
    for rec in iter_json_lines(path):
        rtype = rec.get("type")
        if rtype == "custom-title" and isinstance(rec.get("customTitle"), str):
            custom_title = rec["customTitle"]
        elif rtype == "ai-title" and isinstance(rec.get("aiTitle"), str):
            title = rec["aiTitle"]
        elif rtype == "summary" and isinstance(rec.get("summary"), str) and title is None:
            title = rec["summary"]
        if not seen_id and isinstance(rec.get("sessionId"), str):
            s.native_id, seen_id = rec["sessionId"], True
        if rec.get("isSidechain") is True:
            continue
        if s.cwd is None and isinstance(rec.get("cwd"), str) and rec["cwd"]:
            s.cwd = rec["cwd"]
        branch = rec.get("gitBranch")
        if isinstance(branch, str) and branch and branch != "HEAD":
            s.branch = branch
        if rec.get("entrypoint") in _ORBITAL_ENTRYPOINTS:
            s.orbital_started = True
        if rtype not in ("user", "assistant"):
            continue
        msg = rec.get("message")
        if not isinstance(msg, dict):
            continue
        ts = norm_ts(rec.get("timestamp"))
        content = msg.get("content")
        if rtype == "user":
            if rec.get("isMeta") or rec.get("isCompactSummary"):
                continue
            any_message = True
            touch(s, ts)
            text = _user_text(content)
            if text is not None:
                stripped = text.strip()
                if stripped.startswith(_INTERRUPT_PREFIX):
                    turn.interrupted("interrupted by user")
                elif stripped and not stripped.startswith(_MACHINE_PREFIXES):
                    turn.user_text(stripped, ts)
                continue
            if isinstance(content, list) and any(
                    isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                turn.working()
            continue
        # assistant
        if rec.get("isApiErrorMessage"):
            any_message = True
            touch(s, ts)
            fallback = _ERROR_FIELD_CAUSES.get(rec.get("error"), CAUSE_INTERRUPTED)
            turn.error(_assistant_text(content), ts, fallback=fallback)
            continue
        if msg.get("model") == "<synthetic>":
            continue
        any_message = True
        touch(s, ts)
        blocks = content if isinstance(content, list) else (
            [{"type": "text", "text": content}] if isinstance(content, str) else [])
        has_tool = has_text = False
        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text" and (block.get("text") or "").strip():
                has_text = True
            elif btype == "tool_use":
                has_tool = True
                _note_tool(s, block.get("name"), block.get("input"))
        if has_tool:
            turn.working()
        elif has_text:
            turn.answered()
    if not any_message:
        return None
    turn.finish()
    s.title = custom_title or title
    return s


def _note_tool(s: HistorySession, name, inp) -> None:
    if not isinstance(inp, dict):
        return
    fp = inp.get("file_path") or inp.get("notebook_path")
    if name in _EDIT_TOOLS:
        bump(s.edits, fp)
    elif name in _CREATE_TOOLS:
        bump(s.creates, fp)
    elif name in _READ_TOOLS:
        bump(s.reads, fp or inp.get("path"))
    elif name == "Skill":
        skill = inp.get("skill") or inp.get("command")
        if isinstance(skill, str):
            add_skill(s, skill.split(":")[-1].strip() or None)
    note_mentions(s, inp)


# --------------------------------------------------------------------------
# Private notes (spec 153)
# --------------------------------------------------------------------------

MEMORY_DIRNAME = "memory"
MEMORY_INDEX = "MEMORY.md"
_FRONTMATTER_MAX_LINES = 60
# ``- [Title](target) — hook``; the dash may be an em dash, en dash or hyphen.
_INDEX_LINE_RE = re.compile(r"^\s*[-*]\s+\[(?P<title>[^\]]*)\]\((?P<target>[^)\s]+)\)"
                            r"(?:\s*[\u2014\u2013-]+\s*(?P<hook>.*))?$")


def notes(home: str) -> list[PrivateNote]:
    """Every note under ``<home>/.claude/projects/*/memory/``. Read-only;
    never raises. A file whose frontmatter does not parse still counts."""
    projects = os.path.join(claude_dir(home), "projects")
    try:
        dirs = [d.path for d in os.scandir(projects) if d.is_dir()]
    except OSError:
        return []
    return notes_in_dirs(dirs)


def notes_in_dirs(dirs: list[str], *, folder: str | None = None) -> list[PrivateNote]:
    """The notes of these Claude project dirs. ``folder`` stands in for the
    transcript probe when the caller already knows the dirs' cwd."""
    out: list[PrivateNote] = []
    for d in sorted(dirs):
        try:
            out.extend(_dir_notes(d, folder))
        except Exception:
            logger.warning("history: could not read Claude notes in %s", d, exc_info=True)
    return out


def _dir_notes(project_dir: str, folder: str | None) -> list[PrivateNote]:
    mem = os.path.join(project_dir, MEMORY_DIRNAME)
    try:
        files = sorted(e.name for e in os.scandir(mem)
                       if e.is_file() and e.name.lower().endswith(".md")
                       and e.name != MEMORY_INDEX)
    except OSError:
        return []
    if not files:
        return []
    if folder is None:
        folder = _dir_cwd(project_dir)
    index = _index_lines(os.path.join(mem, MEMORY_INDEX))
    encoded = os.path.basename(project_dir.rstrip("/\\"))
    out = []
    for name in files:
        meta = _frontmatter(os.path.join(mem, name))
        line = index.get(name)
        title = _one_line(meta.get("name")) or os.path.splitext(name)[0]
        hook = _one_line(meta.get("description")) or (line[1] if line else None)
        kind = meta.get("type")
        if kind is None and isinstance(meta.get("metadata"), dict):
            kind = meta["metadata"].get("type")
        out.append(PrivateNote(
            agent=AGENT, folder=folder, encoded=encoded, path=os.path.join(mem, name),
            title=title, hook=hook, kind=kind if isinstance(kind, str) and kind else None,
            index_line=line[0] if line else None))
    return out


def _dir_cwd(project_dir: str) -> str | None:
    """The cwd a transcript in this dir recorded (newest first), or None."""
    from agent_os.onboarding.import_scanner import _claude_session_cwd

    try:
        sessions = [e.path for e in os.scandir(project_dir)
                    if e.is_file() and e.name.endswith(".jsonl")]
    except OSError:
        return None

    def mtime(p: str) -> float:
        try:
            return os.stat(p).st_mtime
        except OSError:
            return 0.0

    for path in sorted(sessions, key=mtime, reverse=True)[:5]:
        cwd = _claude_session_cwd(path)
        if cwd:
            return os.path.normpath(cwd)
    return None


def _index_lines(path: str) -> dict[str, tuple[str, str | None]]:
    """``{note file name: (index line, hook)}`` from MEMORY.md."""
    out: dict[str, tuple[str, str | None]] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return out
    for raw in lines:
        m = _INDEX_LINE_RE.match(raw)
        if not m:
            continue
        name = os.path.basename(m.group("target").split("#", 1)[0].replace("\\", "/"))
        if name and name not in out:
            out[name] = (raw.strip(), _one_line(m.group("hook")))
    return out


def _frontmatter(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = []
            for i, line in enumerate(fh):
                if i >= _FRONTMATTER_MAX_LINES:
                    break
                lines.append(line.rstrip("\r\n"))
    except OSError:
        return {}
    if not lines or lines[0].strip() != "---":
        return {}
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return {}
    try:
        meta = yaml.safe_load("\n".join(lines[1:end]))
    except Exception:
        # Not only YAMLError: well-formed YAML with an impossible date raises
        # ValueError from the constructor. The note still counts (spec 153).
        return {}
    return meta if isinstance(meta, dict) else {}


def _one_line(value) -> str | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    text = " ".join(str(value).split())
    return text or None


SOURCE = ClaudeCodeSource()
