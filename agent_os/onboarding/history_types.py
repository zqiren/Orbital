# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared shapes for reading the user's agent history (spec 138).

One :class:`HistorySession` per native session of a supported agent, read
from that agent's own files on this machine. Each agent contributes a
:class:`HistorySource` module under ``agent_os.onboarding.sources``; the scan
(``agent_os.onboarding.history``) caches records by ``(path, mtime, size)``,
groups them into projects and picks the left-off cards from them.

Text fields hold the user's words AFTER :func:`text_clean.clean_user_text`.
Nothing here ever leaves the machine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Why a session's last turn did not end in a real answer. Order = display
# priority among equally recent rows is NOT implied; the cards sort by time.
CAUSE_USAGE_LIMIT = "usage_limit"
CAUSE_MODEL_UNAVAILABLE = "model_unavailable"
CAUSE_NETWORK_AUTH = "network_auth"
CAUSE_INTERRUPTED = "interrupted"
CAUSE_NO_REPLY = "no_reply"
CAUSES = (CAUSE_USAGE_LIMIT, CAUSE_MODEL_UNAVAILABLE, CAUSE_NETWORK_AUTH,
          CAUSE_INTERRUPTED, CAUSE_NO_REPLY)


@dataclass(frozen=True)
class StopInfo:
    cause: str                       # one of CAUSES
    detail: str | None = None        # short error text, local display only
    resets_at: str | None = None     # ISO-8601 UTC, usage limits only


@dataclass
class HistorySession:
    agent: str                       # sub-agent slug: claude-code, codex, pi, kimi-code, minimax-code
    native_id: str                   # the id the agent resumes by
    path: str                        # the file whose mtime/size changes with the session
    cwd: str | None = None           # folder the session ran in, as recorded
    title: str | None = None
    started_at: str | None = None    # ISO-8601 UTC
    ended_at: str | None = None      # ISO-8601 UTC, last user/assistant record
    first_user_text: str | None = None   # cleaned
    last_user_text: str | None = None    # cleaned, verbatim
    last_user_at: str | None = None
    user_turns: int = 0              # turns with the user's own words
    automation: bool = False         # every user turn was an Orbital trigger
    orbital_started: bool = False    # Orbital itself drove this session (its own worker run)
    edits: dict[str, int] = field(default_factory=dict)     # abs path -> times written
    creates: dict[str, int] = field(default_factory=dict)   # abs path -> times created
    reads: dict[str, int] = field(default_factory=dict)     # abs path -> times read
    mentions: dict[str, int] = field(default_factory=dict)  # abs path -> times named
    skills: list[str] = field(default_factory=list)         # skill names used
    stop: StopInfo | None = None     # None = the last turn ended in a real answer
    resume_path: str | None = None   # file the agent's resume needs (Codex rollout, pi session file)
    branch: str | None = None        # git branch the session ran on, when recorded


@dataclass(frozen=True)
class PrivateNote:
    """One note an agent keeps for itself that no other agent reads (spec
    153). The file is the fact; every other field is decoration read from
    its frontmatter or the agent's own index, and may be missing.

    ``folder`` is the cwd recorded by a transcript next to the notes, or None
    when the agent's dir holds notes but no transcript (never decoded from
    the dir name, which is lossy). ``encoded`` is that dir's name.
    ``index_line`` is the agent's own index line for this note, if any."""

    agent: str
    folder: str | None
    encoded: str
    path: str
    title: str
    hook: str | None = None
    kind: str | None = None
    index_line: str | None = None


class HistorySource(Protocol):
    """One agent's reader. Module ``agent_os.onboarding.sources.<name>``
    exposes ``SOURCE`` implementing this."""

    agent: str

    def session_files(self, home: str) -> list[str]:
        """Every session file under ``home`` (cheap: a directory listing)."""

    def read(self, path: str) -> HistorySession | None:
        """Parse one session file. None for an empty or unreadable one.
        Never raises; read-only."""


# --------------------------------------------------------------------------
# Error text -> StopInfo (shared by every source)
# --------------------------------------------------------------------------

_USAGE_LIMIT_RE = re.compile(
    r"hit your (?:session|usage|weekly|daily) limit|out of (?:usage )?credits"
    r"|usage limit (?:reached|exceeded)|rate limit reached for your plan"
    r"|credit balance is too low|requires more credits|insufficient (?:credits|balance|quota)"
    r"|exceeded your current quota",
    re.IGNORECASE)
_MODEL_RE = re.compile(r"model (?:is )?(?:not (?:supported|available|found)|unavailable)|only available hosted in|unsupported model"
                       r"|does not have access to (?:the )?model"
                       # MiniMax Code: 'Model "minimax/X" is not available for the … route'
                       r"|model \"[^\"\n]*\" is not (?:supported|available)", re.IGNORECASE)
_NETWORK_AUTH_RE = re.compile(
    r"ECONNRESET|ETIMEDOUT|ENOTFOUND|socket hang up|network error|connection (?:reset|refused|error)"
    r"|\b40[13]\b|failed to authenticate|authentication (?:failed|error)|invalid api key"
    r"|unauthori[sz]ed|forbidden|please (?:run )?/?login"
    # Spec 167: a signed-out worker's could-not-start row.
    r"|authentication required|not signed in|not logged in", re.IGNORECASE)

# "resets 10pm (Asia/Shanghai)", "resets at 9:30 am (UTC)", "resets 22:00"
_RESET_RE = re.compile(
    r"resets?(?: at)?\s+(\d{1,2})(?::(\d{2}))?\s*([ap]m)?(?:\s*\(([A-Za-z_]+(?:/[A-Za-z_+\-]+)*)\))?",
    re.IGNORECASE)


def classify_error_text(text: str | None, *, at: str | None = None) -> StopInfo | None:
    """The stop cause an agent's error text names, or None if it names none.

    ``at`` (ISO-8601) is when the error happened; a reset time in the text is
    resolved to its next occurrence after ``at`` in the named time zone (UTC
    when none is named) and returned as ISO-8601 UTC.
    """
    if not text:
        return None
    detail = " ".join(text.split())[:200]
    if _USAGE_LIMIT_RE.search(text):
        return StopInfo(CAUSE_USAGE_LIMIT, detail, _parse_reset(text, at))
    if _MODEL_RE.search(text):
        return StopInfo(CAUSE_MODEL_UNAVAILABLE, detail)
    if _NETWORK_AUTH_RE.search(text):
        return StopInfo(CAUSE_NETWORK_AUTH, detail)
    return None


def _parse_reset(text: str, at: str | None) -> str | None:
    m = _RESET_RE.search(text)
    if not m:
        return None
    hour, minute, ampm, tzname = int(m.group(1)), int(m.group(2) or 0), m.group(3), m.group(4)
    if ampm:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if ampm.lower() == "pm" else 0)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    try:
        tz = ZoneInfo(tzname) if tzname else timezone.utc
    except (ZoneInfoNotFoundError, ValueError):
        tz = timezone.utc
    base = _parse_iso(at) or datetime.now(timezone.utc)
    local = base.astimezone(tz)
    reset = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if reset <= local:
        reset += timedelta(days=1)
    return reset.astimezone(timezone.utc).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
