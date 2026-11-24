# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Helpers shared by the history readers: the stop-state machine every
reader drives, path / skill extraction from tool inputs, timestamps.

Not a reader itself (``load_sources`` only imports the modules it names).
"""

from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime, timezone

from agent_os.onboarding.history_types import (
    CAUSE_INTERRUPTED, CAUSE_NO_REPLY, CAUSE_USAGE_LIMIT, HistorySession, StopInfo,
    classify_error_text,
)
from agent_os.onboarding.text_clean import KIND_AUTOMATION, KIND_USER, clean_user_text

# Absolute paths named in a tool input. Only user-space roots: a mention is a
# hint about which project a session worked on, never a system path.
# Windows: a drive path (``C:\Users\…`` or ``C:/…``), with either separator;
# the POSIX form stops at a backslash, which there is a JSON escape.
ABS_PATH_RE = re.compile(
    r"(?<![\w.~/:-])(/(?:Users|home|Volumes|private|tmp|var|opt|srv|mnt|workspace|data)"
    r"/[^\s\"'\\`)(,;:<>|*?\[\]{}]+)"
    r"|(?<![\w.~/\\-])([A-Za-z]:[\\/][^\s\"'`)(,;:<>|*?\[\]{}]+)")
# ~/.claude/skills/<name>/…, ~/.codex/skills/…, ~/.agents/skills/… (and the
# same under a project); dot-dirs are not skills. Either separator (Windows).
SKILL_RE = re.compile(r"[\\/]\.(?:claude|codex|agents)[\\/]+skills[\\/]+([^\\/.\s][^\\/\s]*)[\\/]")
# Codex apply_patch headers.
PATCH_RE = re.compile(r"\*\*\* (Add|Update|Delete) File: ([^\\\"\n]+)")

# Distinct paths kept per bag: a session that greps the whole disk must not
# blow up the cache.
_MAX_PATHS = 400

_TEXT_CAP = 2000


def bump(bag: dict, path: str | None) -> None:
    if not path or not isinstance(path, str) or not os.path.isabs(path):
        return
    path = os.path.normpath(path)
    if path in bag:
        bag[path] += 1
    elif len(bag) < _MAX_PATHS:
        bag[path] = 1


def note_mentions(s: HistorySession, blob: object) -> None:
    if not isinstance(blob, str):
        try:
            blob = json.dumps(blob, ensure_ascii=False)
        except (TypeError, ValueError):
            return
    for posix, drive in ABS_PATH_RE.findall(blob):
        p = (posix or drive).rstrip(".")
        bump(s.mentions, p)
        m = SKILL_RE.search(p + "/")
        if m:
            add_skill(s, m.group(1))


def add_skill(s: HistorySession, name: str | None) -> None:
    if isinstance(name, str) and name and not name.startswith(".") and name not in s.skills:
        s.skills.append(name)


def norm_ts(value) -> str | None:
    """A native timestamp as UTC ``isoformat()`` (sortable as a string)."""
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


class TurnState:
    """How the session's last turn ended, driven record by record.

    ``pending``  the user (or an automation) spoke and nothing answered yet;
    ``working``  the agent was mid-tool (a tool call or tool result last);
    ``answered`` a real reply closed the turn (a question is still a reply);
    ``stopped``  a stop record (interrupt / abort / error) closed it.
    """

    def __init__(self, s: HistorySession):
        self.s = s
        self.state: str | None = None
        self.stop: StopInfo | None = None
        self._automation_turns = 0

    def user_text(self, raw: object, ts: str | None) -> None:
        cleaned = clean_user_text(raw)
        if cleaned.kind == KIND_USER:
            text = cleaned.text[:_TEXT_CAP]
            self.s.user_turns += 1
            if self.s.first_user_text is None:
                self.s.first_user_text = text
            self.s.last_user_text = text
            self.s.last_user_at = ts or self.s.last_user_at
        elif cleaned.kind == KIND_AUTOMATION:
            self.automation_text(cleaned.text, ts)
            return
        else:
            return  # Orbital's own prompt: the turn state is untouched
        self.state, self.stop = "pending", None

    def automation_text(self, text: str | None, ts: str | None) -> None:
        """A turn an automation started (Orbital's triggers, or one the agent
        app itself ran): a session of only these is an automation run."""
        self._automation_turns += 1
        if self.s.first_user_text is None and text:
            self.s.first_user_text = text[:_TEXT_CAP]
        self.state, self.stop = "pending", None

    def working(self) -> None:
        self.state, self.stop = "working", None

    def answered(self) -> None:
        self.state, self.stop = "answered", None

    def interrupted(self, detail: str | None = None) -> None:
        self.state, self.stop = "stopped", StopInfo(CAUSE_INTERRUPTED, detail)

    def error(self, text: str | None, ts: str | None, *, fallback: str = CAUSE_INTERRUPTED,
              resets_at: str | None = None) -> None:
        """An error ended the turn. Text that names no cause still counts as
        "didn't finish": ``fallback`` (``interrupted``, the same choice every
        reader makes). ``resets_at`` (spec 197): the reset Orbital already
        worked out for its own usage-limit row, which wins over the text."""
        stop = classify_error_text(text, at=ts)
        if stop is not None and resets_at and stop.cause == CAUSE_USAGE_LIMIT:
            stop = StopInfo(stop.cause, stop.detail, resets_at)
        if stop is None:
            detail = " ".join((text or "").split())[:200] or None
            stop = StopInfo(fallback, detail)
        self.state, self.stop = "stopped", stop

    def finish(self) -> None:
        s = self.s
        s.automation = self._automation_turns > 0 and s.user_turns == 0
        if self.state == "pending":
            s.stop = StopInfo(CAUSE_NO_REPLY)
        elif self.state == "working":
            s.stop = StopInfo(CAUSE_INTERRUPTED)
        elif self.state == "stopped":
            s.stop = self.stop
        else:
            s.stop = None


def touch(s: HistorySession, ts: str | None) -> None:
    if not ts:
        return
    if s.started_at is None or ts < s.started_at:
        s.started_at = ts
    if s.ended_at is None or ts > s.ended_at:
        s.ended_at = ts


def open_text(path: str):
    """A history file as text. Codex compresses older rollouts to
    ``rollout-*.jsonl.zst`` (418 of 487 on a dev Mac), so a reader that only
    opened ``.jsonl`` missed most of a long-time user's Codex history."""
    if path.endswith(".zst"):
        import zstandard

        raw = open(path, "rb")
        try:
            stream = zstandard.ZstdDecompressor().stream_reader(raw, closefd=True)
        except Exception:
            raw.close()
            raise
        return io.TextIOWrapper(stream, encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def iter_json_lines(path: str):
    with open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(rec, dict):
                yield rec
