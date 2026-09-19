# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The asks list: ``orbital/ASKS.md``, an append-only event log (spec 089 §3).

An *ask* is something waiting on the user that must outlive the conversation it
came up in: a dated commitment, something the user said they will do
themselves, or a decision the agent parked so it could move on. A question the
user can answer in the same conversation stays in chat.

Asks used to be ``[user]``-flagged bullets inside PROJECT_STATE. That prose is
reworded constantly, so every consumer had to reconstruct an item's identity
with a fuzzy title matcher, a hidden ``resolved:`` stamp and a separate
retractions file — and in practice nothing ever recorded "done". Here identity
is an Orbital-stamped id on a line nobody may change, and an ask's state is
simply its last event.

Grammar — one event per line, oldest first::

    - open <id> <YYYY-MM-DD> [due:YYYY-MM-DD[THH:MM]] <text>
    - done <id> <YYYY-MM-DD> by:<user|agent|editor> <note>
    - dropped <id> <YYYY-MM-DD> by:<user|agent|editor> <note>
    - reopen <id> <YYYY-MM-DD> by:<user|agent> <note>

- ids are 6 lowercase hex chars assigned by Orbital. An agent writes
  ``- open <text>`` (optionally ``due:``) with no id; the write path stamps the
  id and today's date.
- a ``done``/``dropped`` by anyone but the user carries the user's own words,
  quoted, as its note. Those closes are undoable from the Workbench for
  ``RECENTLY_CLOSED_DAYS``.
- open asks = asks whose last event is ``open`` or ``reopen``.

Append-only is enforced on the agent's write path (``process_write``, reached
through ``memory_entries.process_on_write``): removed or altered lines are put
back in their original order, new lines are validated, stamped and appended.
Writers that bypass the tools (external agents, a human in an editor) are
tolerated by the fold: an id-less ``open`` gets a deterministic text-derived
id, and the next Orbital append stamps exactly that id onto the line — the
file is never rewritten on a read.

Public API
----------
- ``read_asks(orbital_dir)`` / ``fold(content)`` — the current asks.
- ``open_ask(orbital_dir, text, *, due=None)`` — mint one ask.
- ``append_event(orbital_dir, kind, ask_id, note="", by="user")`` — close,
  drop or reopen an ask. Everything that closes an ask outside the agent's own
  file writes goes through here (the Workbench today; the memory editor's
  quote-backed closes, ``by="editor"``, at integration).
- ``extract_legacy_flags(state_content)`` — pure ``[user]`` → ask conversion.
- ``convert_legacy_on_write(orbital_dir, state_content)`` — the PROJECT_STATE
  write-path hook built on it.
- ``render_runtime_block(orbital_dir)`` — the per-call "Open asks" block.
- ``migrate_project(workspace)`` / ``migrate_all(project_store)`` — upgrade.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import sys
import threading
import time
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone

from agent_os.agent import user_flags

logger = logging.getLogger(__name__)

ASKS_FILENAME = "ASKS.md"
# One-time migration marker + backups, both under orbital/.
MIGRATION_MARKER = ".asks-migrated.json"
BACKUPS_DIRNAME = "backups"
_BACKUP_PREFIX = "asks-migration-"

CLOSE_KINDS = ("done", "dropped")
EVENT_KINDS = ("open", "reopen") + CLOSE_KINDS
WRITERS = ("user", "agent", "editor")

# Runtime block bounds (spec §3.5).
OPEN_LIMIT = 10
DROPPED_DAYS = 60
DROPPED_LIMIT = 15
# Workbench "Recently closed" window (spec §3.6).
RECENTLY_CLOSED_DAYS = 7
# One ask can be a paragraph; the per-call block must stay bounded.
_RUNTIME_LINE_CHARS = 240
# The per-turn cue when nothing is open (live test (b): with no asks the block
# was omitted, so a project got no per-turn reminder before its first ask).
EMPTY_CUE = "Open asks: none (orbital/ASKS.md)"

# The file's contract, on line 1 like every other Layer-1 file — so any agent
# that opens it, including external ones, learns the grammar from the file.
FORMAT_HEADER = (
    "<!--format ASKS is an append-only log of things waiting on the user, one "
    "line per event; Orbital restores any line that is changed or deleted. "
    "Open an ask only for something that must outlive the conversation: "
    "something with a date, something the user said they will do themselves, "
    "or a decision the user put off for later (\"I'll decide after …\") while "
    "you carry on. A TBD note in a file or in chat is not an ask — only "
    "ASKS.md reaches the user's Workbench. To open one, add a line "
    "`- open <text>` or `- open due:YYYY-MM-DD <text>` — Orbital stamps the "
    "id and date. When the user answers or completes one, add "
    "`- done <id> \"<the user's own words>\"`; when the user declines it, "
    "`- dropped <id> \"<their words>\"`. Close only with the user's words; "
    "never re-propose a dropped ask. An ask's state is its last line.-->"
)

_ID_RE = r"[0-9a-fA-F]{6}"
_DATE_RE = r"\d{4}-\d{2}-\d{2}"
_DUE_VALUE_RE = r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2})?"

_EVENT_LINE_RE = re.compile(
    r"^\s*[-*]\s+(?P<kind>open|done|dropped|reopen)(?=\s|$)(?P<rest>.*)$"
)
# Canonical open: id AND date. An id alone is not trusted as an id on an
# open line — "- open facade redesign" must not read "facade" as one.
_OPEN_HEAD_RE = re.compile(rf"^\s*\[?(?P<id>{_ID_RE})\]?\s+(?P<date>{_DATE_RE})(?=\s|$)")
_CLOSE_ID_RE = re.compile(rf"^\s*\[?(?P<id>{_ID_RE})\]?(?=\s|$)")
_LEADING_DATE_RE = re.compile(rf"^\s*(?P<date>{_DATE_RE})(?=\s|$)")
_LEADING_BY_RE = re.compile(r"^\s*by:(?P<by>\S+)")
_DUE_TOKEN_RE = re.compile(rf"(?<!\S)\[?due:(?P<due>{_DUE_VALUE_RE})\]?(?!\S)")
_ASK_ID_RE = re.compile(r"^[0-9a-f]{6}$")
_ISO_DATE_RE = re.compile(rf"^{_DATE_RE}$")

# A legacy flagged bullet: "- [user ...] text" / "3. [user] text".
_LEGACY_TAG_RE = re.compile(
    rf"^(?P<prefix>{user_flags.LIST_MARKER})\[(?P<tag>[^\]]*)\]\s*(?P<text>.*)$"
)
_LIST_MARKER_PREFIX_RE = re.compile(rf"^\s*(?:{user_flags.LIST_MARKER}|[*+]\s+)")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    kind: str                  # open | done | dropped | reopen
    id: str | None             # None only for an id-less close (unresolvable)
    date: str | None
    by: str | None             # closes/reopens only
    text: str                  # open: the ask; otherwise: the note
    due: str | None = None     # open only
    synthetic: bool = False    # id derived from text (an id-less open line)
    line: int = -1             # 0-based line index in the file, -1 if none


@dataclass(frozen=True)
class Ask:
    id: str
    text: str
    due: str | None
    opened: str | None         # date of the first open event
    state: str                 # "open" | "done" | "dropped"
    updated: str | None        # date of the last event
    closed_by: str | None      # who closed it (done/dropped), else None
    note: str                  # note of the last close/reopen event
    seq: int                   # file position of the last event

    @property
    def is_open(self) -> bool:
        return self.state == "open"


def _one_line(text: str | None) -> str:
    return " ".join((text or "").split())


def _norm(text: str | None) -> str:
    """Identity key for "is this the same ask" — whitespace/case-insensitive."""
    return _one_line(text).casefold()


def _today() -> str:
    return date.today().isoformat()


def _valid_date(value: str | None) -> str | None:
    if value and _ISO_DATE_RE.match(value):
        try:
            date.fromisoformat(value)
            return value
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------

def parse_line(line: str, line_no: int = -1) -> Event | None:
    """Parse one event line, leniently. ``None`` for anything that isn't one."""
    m = _EVENT_LINE_RE.match(line)
    if not m:
        return None
    kind = m.group("kind")
    rest = m.group("rest")
    if kind == "open":
        ask_id = event_date = None
        head = _OPEN_HEAD_RE.match(rest)
        if head:
            ask_id = head.group("id").lower()
            event_date = _valid_date(head.group("date"))
            rest = rest[head.end():]
        due = None
        dm = _DUE_TOKEN_RE.search(rest)
        if dm:
            due = dm.group("due")
            rest = rest[:dm.start()] + " " + rest[dm.end():]
        return Event("open", ask_id, event_date, None, _one_line(rest), due, line=line_no)

    idm = _CLOSE_ID_RE.match(rest)
    if not idm:
        return Event(kind, None, None, None, _one_line(rest), line=line_no)
    ask_id = idm.group("id").lower()
    rest = rest[idm.end():]
    event_date = by = None
    while True:
        dm = _LEADING_DATE_RE.match(rest)
        if dm and event_date is None:
            event_date = _valid_date(dm.group("date"))
            rest = rest[dm.end():]
            continue
        bm = _LEADING_BY_RE.match(rest)
        if bm and by is None:
            by = bm.group("by").lower()
            rest = rest[bm.end():]
            continue
        break
    return Event(kind, ask_id, event_date, by, _one_line(rest), line=line_no)


def render_event(e: Event) -> str:
    """The canonical one-line form of ``e``."""
    if e.kind == "open":
        parts = ["- open", e.id or "", e.date or _today()]
        if e.due:
            parts.append(f"due:{e.due}")
        parts.append(_one_line(e.text))
        return " ".join(p for p in parts if p)
    parts = [f"- {e.kind}", e.id or "", e.date or _today(), f"by:{e.by or 'user'}"]
    note = _one_line(e.text)
    if note:
        parts.append(note)
    return " ".join(p for p in parts if p)


def _synthetic_id(text: str, taken: set[str]) -> str:
    """Deterministic id for an id-less open line, avoiding stamped ids."""
    base = _norm(text)
    n = 0
    while True:
        seed = base if n == 0 else f"{base}#{n}"
        cand = hashlib.sha1(f"ask:{seed}".encode("utf-8")).hexdigest()[:6]
        if cand not in taken:
            return cand
        n += 1


def parse_events(content: str | None) -> list[Event]:
    """Every event in ``content`` in file order; id-less opens get their
    deterministic id (``synthetic=True``)."""
    raw: list[Event] = []
    for i, line in enumerate((content or "").split("\n")):
        e = parse_line(line, i)
        if e is not None:
            raw.append(e)
    defined = {e.id for e in raw if e.kind == "open" and e.id}
    out: list[Event] = []
    for e in raw:
        if e.kind == "open" and e.id is None and e.text:
            sid = _synthetic_id(e.text, defined)
            defined.add(sid)
            e = replace(e, id=sid, synthetic=True)
        out.append(e)
    return out


def fold(content: str | None) -> list[Ask]:
    """Current asks from an event log, in order of first appearance."""
    state: dict[str, dict] = {}
    for seq, e in enumerate(parse_events(content)):
        if e.kind == "open":
            if not e.id or not e.text:
                continue
            cur = state.get(e.id)
            if cur is None:
                state[e.id] = {
                    "id": e.id, "text": e.text, "due": e.due, "opened": e.date,
                    "state": "open", "updated": e.date, "closed_by": None,
                    "note": "", "seq": seq,
                }
            else:  # a second open for a known id acts as a reopen
                cur.update(state="open", updated=e.date or cur["updated"],
                           closed_by=None, note="", seq=seq)
            continue
        cur = state.get(e.id) if e.id else None
        if cur is None:
            continue  # a close for an ask this log never opened
        if e.kind == "reopen":
            cur.update(state="open", updated=e.date or cur["updated"],
                       closed_by=None, note=e.text, seq=seq)
        else:
            # A close with no by: was written outside Orbital; treat it as
            # not-the-user so it stays undoable.
            cur.update(state=e.kind, updated=e.date or cur["updated"],
                       closed_by=e.by or "agent", note=e.text, seq=seq)
    return [Ask(**v) for v in state.values()]


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

_IS_WINDOWS = sys.platform.startswith("win")
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    key = os.path.realpath(path)
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def asks_path(orbital_dir: str) -> str:
    return os.path.join(orbital_dir, ASKS_FILENAME)


def _read(path: str) -> str | None:
    """Decode-safe read for FOLDING (never for a rewrite): invalid bytes
    become U+FFFD so one bad byte can't blank a Workbench."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _read_exact(path: str) -> tuple[str | None, bool]:
    """``(text, decodable)`` for a file about to be rewritten. ``decodable``
    is False when the bytes aren't valid UTF-8 — the caller must then not
    rewrite the file, because writing the decoded text back would replace
    those bytes."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return None, True
    try:
        return raw.decode("utf-8"), True
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), False


def _atomic_write(path: str, content: str) -> None:
    """tmp file in the same dir + ``os.replace`` — never a half-written file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    for attempt in range(5):  # Windows: the target may be briefly open
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            if not _IS_WINDOWS or attempt == 4:
                raise
            time.sleep(0.05)


def read_asks(orbital_dir: str) -> list[Ask]:
    """Asks for a project's ``orbital/`` dir; ``[]`` when there is no file."""
    return fold(_read(asks_path(orbital_dir)))


def normalize(content: str | None, today: str | None = None) -> str:
    """Current header on line 1 + id-less opens stamped with their fold id.

    Only Orbital's own appends call this, never a read. Every other line is
    byte-identical; the result always ends with a newline.
    """
    today = today or _today()
    text = content or ""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for e in parse_events(text):
        if e.synthetic and 0 <= e.line < len(lines):
            lines[e.line] = render_event(replace(e, date=e.date or today))
    if lines and lines[0].lstrip().startswith("<!--format"):
        if lines[0] != FORMAT_HEADER and lines[0].rstrip().endswith("-->"):
            lines[0] = FORMAT_HEADER
    else:
        lines.insert(0, FORMAT_HEADER)
    return "\n".join(lines) + "\n"


def _mint_id(taken) -> str:
    while True:
        cand = secrets.token_hex(3)
        if cand not in taken:
            return cand


def _append_lines(orbital_dir: str, build, today: str) -> list:
    """Read → fold → ``build(asks_by_id)`` → write normalized + new lines.

    ``build`` returns ``(lines, result)``. One atomic write under the file's
    lock, so an append is all-or-nothing.
    """
    path = asks_path(orbital_dir)
    with _lock_for(path):
        content, decodable = _read_exact(path)
        by_id = {a.id: a for a in fold(content)}
        lines, result = build(by_id)
        if lines and decodable:
            base = normalize(content, today)
            _atomic_write(path, base + "\n".join(lines) + "\n")
        elif lines:
            # Bytes that aren't UTF-8 (a foreign writer): append as-is and
            # leave every existing byte alone rather than normalize.
            with open(path, "ab") as f:
                lead = b"" if not content or content.endswith("\n") else b"\n"
                f.write(lead + ("\n".join(lines) + "\n").encode("utf-8"))
        return result


def open_ask(
    orbital_dir: str,
    text: str,
    *,
    due: str | None = None,
    today: str | None = None,
) -> Ask | None:
    """Open one ask with a fresh id. ``None`` (no write) when an ask with the
    same text is already open."""
    today = today or _today()
    text = _one_line(text)
    if not text:
        raise ValueError("an ask needs text")

    def build(by_id):
        key = _norm(text)
        if any(a.is_open and _norm(a.text) == key for a in by_id.values()):
            return [], None
        e = Event("open", _mint_id(by_id), today, None, text, due)
        ask = Ask(e.id, text, due, today, "open", today, None, "", len(by_id))
        return [render_event(e)], ask

    return _append_lines(orbital_dir, build, today)


def append_event(
    orbital_dir: str,
    kind: str,
    ask_id: str,
    note: str = "",
    by: str = "user",
    *,
    today: str | None = None,
) -> str | None:
    """Append a ``done`` / ``dropped`` / ``reopen`` event for ``ask_id``.

    ``by`` is who acted: ``"user"`` (Workbench), ``"agent"``, or ``"editor"``
    (the memory editor). A close by anyone but the user should carry the
    user's own words in quotes as ``note`` — those closes are what the
    Workbench offers to undo.

    Returns the line written, or ``None`` when the ask is already in that
    state (closing a closed ask, reopening an open one). Raises ``KeyError``
    for an id this project's log never opened and ``ValueError`` for an
    unknown ``kind``/``by``.
    """
    if kind not in ("done", "dropped", "reopen"):
        raise ValueError(f"unknown ask event kind: {kind!r}")
    if by not in WRITERS:
        raise ValueError(f"unknown ask writer: {by!r}")
    today = today or _today()
    ask_id = (ask_id or "").lower()

    def build(by_id):
        ask = by_id.get(ask_id)
        if ask is None:
            raise KeyError(ask_id)
        if kind == "reopen" and ask.is_open:
            return [], None
        if kind in CLOSE_KINDS and not ask.is_open:
            return [], None
        line = render_event(Event(kind, ask_id, today, by, note))
        return [line], line

    return _append_lines(orbital_dir, build, today)


# ---------------------------------------------------------------------------
# Append-only write path (agent write/edit tools)
# ---------------------------------------------------------------------------

def _has_quote(note: str) -> bool:
    return any(q in note for q in ('"', "“", "”", "「", "」", "『", "』"))


def _legacy_open(line: str) -> Event | None:
    """``- [user …] text`` written into ASKS.md out of habit → an open."""
    m = _LEGACY_TAG_RE.match(line.strip())
    if not m or "user" not in m.group("tag").split():
        return None
    due = None
    for tok in m.group("tag").split():
        if tok.startswith("due:") and re.match(rf"^{_DUE_VALUE_RE}$", tok[4:]):
            due = tok[4:]
    text = _one_line(user_flags.strip_mem_comments(m.group("text")))
    return Event("open", None, None, None, text, due) if text else None


def process_write(
    prev: str | None, new: str, *, today: str | None = None,
) -> tuple[str, list[str]]:
    """Enforce append-only on an agent's write of ASKS.md.

    ``prev`` is the file on disk (``None`` if absent), ``new`` the content the
    write/edit tool computed. Every line of ``prev`` survives in place (the
    agent is told when it tried to remove or alter one); every line of
    ``new`` that ``prev`` does not already contain is validated, stamped
    (``id`` + today's date, ``by:agent``) and appended. Returns
    ``(content_to_write, warnings)``.
    """
    today = today or _today()
    prev_text = prev or ""
    prev_lines = [ln for ln in prev_text.split("\n")]
    new_lines = (new or "").split("\n")
    sm = difflib.SequenceMatcher(None, prev_lines, new_lines, autojunk=False)
    matched_prev: set[int] = set()
    matched_new: set[int] = set()
    for block in sm.get_matching_blocks():
        matched_prev.update(range(block.a, block.a + block.size))
        matched_new.update(range(block.b, block.b + block.size))
    new_line_set = set(new_lines)
    prev_line_set = set(prev_lines)
    restored = [
        ln for i, ln in enumerate(prev_lines)
        if i not in matched_prev and ln.strip() and ln not in new_line_set
        and not ln.lstrip().startswith("<!--format")
    ]
    candidates = [
        ln for j, ln in enumerate(new_lines)
        if j not in matched_new and ln not in prev_line_set
    ]

    base = normalize(prev_text, today)
    by_id = {a.id: a for a in fold(base)}
    appended: list[str] = []
    warnings: list[str] = []
    opened: list[str] = []
    closed: list[str] = []
    for raw in candidates:
        stripped = raw.strip()
        if not stripped or stripped.startswith("<!--"):
            continue
        ev = parse_line(raw)
        if ev is None:
            ev = _legacy_open(raw)
        if ev is None:
            warnings.append(
                f'ASKS.md: ignored "{stripped[:60]}" — not an ask event. Add '
                "`- open <text>`, or `- done <id> \"<the user's words>\"`."
            )
            continue
        if ev.kind == "open":
            if not ev.text:
                continue
            if ev.id and ev.id in by_id:
                # An existing open line, reworded. The original is restored
                # above; an ask's text is fixed once stamped.
                warnings.append(
                    f"ASKS.md: [{ev.id}] can't be reworded — close it and open "
                    "a new ask if it changed."
                )
                continue
            key = _norm(ev.text)
            same = [a for a in by_id.values() if _norm(a.text) == key]
            live = next((a for a in same if a.is_open), None)
            if live is not None:
                warnings.append(f"ASKS.md: [{live.id}] is already open with that text — not added again.")
                continue
            dropped = next((a for a in reversed(same) if a.state == "dropped"), None)
            if dropped is not None:
                warnings.append(
                    f"ASKS.md: not added — the user dropped this as [{dropped.id}] "
                    f"on {dropped.updated}; never re-propose it. If the user asked "
                    f"for it again, add `- reopen {dropped.id} \"<their words>\"`."
                )
                continue
            new_id = _mint_id(by_id)
            e = Event("open", new_id, today, None, ev.text, ev.due)
            by_id[new_id] = Ask(new_id, ev.text, ev.due, today, "open", today, None, "", len(by_id))
            appended.append(render_event(e))
            opened.append(new_id)
            continue

        ask = by_id.get(ev.id) if ev.id else None
        if ask is None:
            warnings.append(
                f"ASKS.md: ignored `{ev.kind}` — no ask with id "
                f"{ev.id or '(missing)'}; use an id from the Open asks list."
            )
            continue
        if ev.kind in CLOSE_KINDS and not ask.is_open:
            warnings.append(f"ASKS.md: [{ask.id}] is already {ask.state} — not closed again.")
            continue
        if ev.kind == "reopen" and ask.is_open:
            warnings.append(f"ASKS.md: [{ask.id}] is already open.")
            continue
        if ev.kind in CLOSE_KINDS and not _has_quote(ev.text):
            warnings.append(
                f"ASKS.md: closed [{ask.id}] without quoting the user — close "
                "an ask only with the user's own words in quotes."
            )
        e = Event(ev.kind, ask.id, today, "agent", ev.text)
        new_state = "open" if ev.kind == "reopen" else ev.kind
        by_id[ask.id] = replace(ask, state=new_state, updated=today,
                                closed_by=None if new_state == "open" else "agent",
                                note=ev.text)
        appended.append(render_event(e))
        closed.append(f"{ev.kind} [{ask.id}]")

    if restored:
        warnings.insert(0,
            f"ASKS.md is append-only: restored {len(restored)} line(s) you "
            "removed or changed. Record a change as a new line instead, e.g. "
            "`- done <id> \"<the user's words>\"`."
        )
    if opened or closed:
        done_bits = [f"opened [{i}]" for i in opened] + closed
        # Stamping changed the text the agent wrote, so a follow-up edit
        # anchored on it would miss — hand back the line actually on disk.
        warnings.append(
            "ASKS.md: " + ", ".join(done_bits) + f". The file now ends with: {appended[-1]}"
        )
    out = base + ("\n".join(appended) + "\n" if appended else "")
    return out, warnings


# ---------------------------------------------------------------------------
# Legacy [user] flags in PROJECT_STATE
# ---------------------------------------------------------------------------

def _block_end(lines: list[str], entry) -> int:
    """Last line of a flagged bullet's block: its mem-comment (``line_end``)
    plus any indented continuation lines right below it."""
    end = entry.line_end
    j = end + 1
    while j < len(lines) and lines[j].strip() and lines[j][:1] in (" ", "\t"):
        end = j
        j += 1
    return end


def extract_legacy_flags(state_content: str | None) -> tuple[str | None, list[Event]]:
    """Convert ``[user]``-flagged PROJECT_STATE bullets into ask events. Pure.

    Returns ``(state_without_flag_lines, open_events)``:

    - an open flagged bullet (``- [user] …``, ``- [user due:…] …``,
      ``3. [user] …``) becomes an ``open`` event — its mem-comment id (when it
      is a valid ask id) and ``created`` date carried over, ``due:`` kept,
      indented continuation lines folded into the one-line text — and its
      whole block leaves the state;
    - a flagged bullet whose mem-comment carries ``resolved:`` is a settled
      fact: the bracket tag is dropped, the line stays, no event;
    - anything else — plain lines, ``[due:]`` facts, checkboxes, indented
      (nested) flags, a flag with no text — stays exactly where it is.

    Events carry ``id``/``date`` only when the legacy entry had them; the
    caller assigns fresh ones otherwise.
    """
    if not state_content or "[" not in state_content:
        return state_content, []
    entries = [e for e in user_flags.parse_entries(state_content) if e.flagged]
    if not entries:
        return state_content, []
    lines = state_content.split("\n")
    remove: set[int] = set()
    retag: set[int] = set()
    events: list[Event] = []
    for e in entries:
        if e.resolved:
            retag.add(e.line_start)
            continue
        end = _block_end(lines, e)
        extra = user_flags.strip_mem_comments("\n".join(lines[e.line_start + 1:end + 1]))
        cont = [
            _LIST_MARKER_PREFIX_RE.sub("", ln).strip()
            for ln in extra.split("\n") if ln.strip()
        ]
        head = e.text
        if cont:
            head = head.rstrip() + " " + "; ".join(cont)
        text = _one_line(head)
        if not text:
            continue
        legacy_id = e.id.lower() if e.id and _ASK_ID_RE.match(e.id.lower()) else None
        events.append(Event("open", legacy_id, _valid_date(e.created), None, text, e.due))
        remove.update(range(e.line_start, end + 1))
    if not remove and not retag:
        return state_content, []
    out: list[str] = []
    for i, line in enumerate(lines):
        if i in remove:
            continue
        if i in retag:
            m = _LEGACY_TAG_RE.match(line)
            if m:
                line = m.group("prefix") + m.group("text")
        out.append(line)
    return "\n".join(out), events


def _plan_opens(by_id: dict[str, Ask], events: list[Event], today: str):
    """Lines + asks for legacy opens, skipping any whose text is already an
    ask in ANY state (a stale rewrite must not resurrect a closed ask)."""
    known = {_norm(a.text): a for a in by_id.values()}
    lines: list[str] = []
    made: list[Ask] = []
    skipped: list[tuple[Event, Ask]] = []
    for ev in events:
        key = _norm(ev.text)
        if key in known:
            skipped.append((ev, known[key]))
            continue
        ask_id = ev.id if ev.id and ev.id not in by_id else _mint_id(by_id)
        opened = ev.date or today
        e = Event("open", ask_id, opened, None, ev.text, ev.due)
        ask = Ask(ask_id, _one_line(ev.text), ev.due, opened, "open", opened, None, "", len(by_id))
        by_id[ask_id] = ask
        known[key] = ask
        lines.append(render_event(e))
        made.append(ask)
    return lines, made, skipped


def convert_legacy_on_write(
    orbital_dir: str, state_content: str, *, today: str | None = None,
) -> tuple[str, list[str]]:
    """PROJECT_STATE write-path hook: move ``[user]`` lines into ASKS.md.

    Never rejects the write. Returns ``(state_content_without_them, warnings)``.
    """
    today = today or _today()
    new_state, events = extract_legacy_flags(state_content)
    if not events:
        return new_state, []

    def build(by_id):
        lines, made, skipped = _plan_opens(by_id, events, today)
        return lines, (made, skipped)

    made, skipped = _append_lines(orbital_dir, build, today)
    warnings: list[str] = []
    if made:
        listed = "; ".join(f"[{a.id}] {a.text[:60]}" for a in made)
        warnings.append(
            f"Moved {len(made)} [user] line(s) out of PROJECT_STATE into "
            f"orbital/ASKS.md as open asks: {listed}. Things waiting on the "
            "user live in ASKS.md now (see its header)."
        )
    for ev, ask in skipped:
        warnings.append(
            f"Removed [user] line \"{ev.text[:60]}\" from PROJECT_STATE — it is "
            f"already ask [{ask.id}] ({ask.state}) in orbital/ASKS.md."
        )
    return new_state, warnings


# ---------------------------------------------------------------------------
# What the agent sees each call
# ---------------------------------------------------------------------------

def _as_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat((value or "")[:10])
    except ValueError:
        return None


def sort_open(asks_list: list[Ask]) -> list[Ask]:
    """Open asks: dated ones first, due-soonest (so overdue) first; then
    undated ones, newest first."""
    live = [a for a in asks_list if a.is_open]
    dated = sorted((a for a in live if a.due), key=lambda a: (a.due, a.seq))
    undated = sorted(
        (a for a in live if not a.due),
        key=lambda a: (a.opened or "", a.seq), reverse=True,
    )
    return dated + undated


def recently_dropped(
    asks_list: list[Ask], today: str, *,
    days: int = DROPPED_DAYS, limit: int = DROPPED_LIMIT,
) -> list[Ask]:
    """Asks dropped in the last ``days``, newest first, at most ``limit``."""
    t = _as_date(today) or date.today()
    cutoff = t - timedelta(days=days)
    rows = []
    for a in asks_list:
        if a.state != "dropped":
            continue
        d = _as_date(a.updated)
        if d is not None and d < cutoff:
            continue
        rows.append(a)
    rows.sort(key=lambda a: (a.updated or "", a.seq), reverse=True)
    return rows[:limit]


def recently_closed(
    asks_list: list[Ask], today: str, *, days: int = RECENTLY_CLOSED_DAYS,
) -> list[Ask]:
    """Asks closed by anyone but the user in the last ``days`` (the undoable
    ones), newest first."""
    t = _as_date(today) or date.today()
    cutoff = t - timedelta(days=days)
    rows = [
        a for a in asks_list
        if a.state in CLOSE_KINDS and a.closed_by != "user"
        and (_as_date(a.updated) or t) >= cutoff
    ]
    rows.sort(key=lambda a: (a.updated or "", a.seq), reverse=True)
    return rows


def _clip(text: str, limit: int = _RUNTIME_LINE_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def render_runtime_block(
    orbital_dir: str, *, today: str | None = None, empty_cue: bool = False,
) -> str:
    """The per-call runtime section: open asks, then recently dropped ones.

    With nothing open, ``empty_cue`` puts ``EMPTY_CUE`` where the list would
    be (the dropped section is still omitted when empty). ``""`` when there
    is nothing to show or the file can't be read — never raises, because it
    runs on every LLM call.
    """
    try:
        today = today or _today()
        folded = read_asks(orbital_dir)
        parts: list[str] = []
        live = sort_open(folded)
        if not live and empty_cue:
            parts.append(EMPTY_CUE)
        if live:
            lines = [f"Open asks ({len(live)}):"]
            for a in live[:OPEN_LIMIT]:
                suffix = ""
                if a.due:
                    late = ", overdue" if a.due[:10] < today else ""
                    suffix = f" (due {a.due}{late})"
                lines.append(f"[{a.id}] {_clip(a.text)}{suffix}")
            if len(live) > OPEN_LIMIT:
                lines.append(f"…and {len(live) - OPEN_LIMIT} more in orbital/ASKS.md")
            parts.append("\n".join(lines))
        dropped = recently_dropped(folded, today)
        if dropped:
            lines = ["Dropped by the user — never re-propose:"]
            lines.extend(f"- {_clip(a.text)}" for a in dropped)
            parts.append("\n".join(lines))
        return "\n\n".join(parts)
    except Exception:
        logger.warning("asks: runtime block failed for %s", orbital_dir, exc_info=True)
        return ""


def dropped_texts(
    orbital_dir: str, *, recent: bool = True, today: str | None = None,
) -> list[str]:
    """Texts of dropped asks — the "never re-propose" list for writers that
    don't get the runtime block. ``recent`` limits it to the runtime block's
    selection; ``False`` returns every dropped ask (the old retractions list)."""
    try:
        folded = read_asks(orbital_dir)
        if not recent:
            return [a.text for a in folded if a.state == "dropped"]
        return [a.text for a in recently_dropped(folded, today or _today())]
    except Exception:
        logger.warning("asks: dropped list failed for %s", orbital_dir, exc_info=True)
        return []


# ---------------------------------------------------------------------------
# One-time migration (runs at daemon start; idempotent afterwards)
# ---------------------------------------------------------------------------

class MigrationError(RuntimeError):
    pass


def _backup(orbital_dir: str, paths: list[str], now: datetime) -> str:
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = os.path.join(orbital_dir, BACKUPS_DIRNAME)
    dest = os.path.join(root, _BACKUP_PREFIX + stamp)
    n = 1
    while os.path.exists(dest):
        n += 1
        dest = os.path.join(root, f"{_BACKUP_PREFIX}{stamp}-{n}")
    os.makedirs(dest)
    for p in paths:
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(dest, os.path.basename(p)))
    return dest


def migrate_project(
    workspace: str, *, today: str | None = None, now: datetime | None = None,
) -> dict:
    """Move a project's asks into ``orbital/ASKS.md`` (spec 089 §3.4).

    First run (no marker): open ``[user]`` bullets → ``open`` asks (removed
    from PROJECT_STATE); flagged bullets already ``resolved:`` → plain settled
    facts; ``retractions.md`` entries → ``dropped`` asks (by:user). The
    PROJECT_STATE header is refreshed to the current contract. Later runs only
    sweep ``[user]`` lines that reached PROJECT_STATE outside the tools (an
    external agent, an older version after a downgrade) and never read
    ``retractions.md`` again.

    Guarantees: backup of every file it will touch before the first write
    (``orbital/backups/asks-migration-<utc>/``); ASKS.md is written and
    re-parsed BEFORE PROJECT_STATE is rewritten; asks are matched by text, so
    an interrupted run resumes without duplicates; nothing is deleted —
    ``retractions.md`` stays for older versions; the marker is written last.
    """
    from agent_os.agent import memory_entries, retractions
    from agent_os.agent.project_paths import ProjectPaths

    today = today or _today()
    now = now or datetime.now(timezone.utc)
    pp = ProjectPaths(workspace)
    orbital_dir = pp.orbital_dir
    if not os.path.isdir(orbital_dir):
        return {"status": "no_orbital"}
    marker = os.path.join(orbital_dir, MIGRATION_MARKER)
    first_run = not os.path.exists(marker)
    state_path = pp.project_state
    retractions_path = os.path.join(orbital_dir, retractions.RETRACTIONS_FILENAME)
    a_path = asks_path(orbital_dir)

    state, state_decodable = _read_exact(state_path)
    if not state_decodable:
        logger.warning(
            "asks migration: %s is not valid UTF-8 — left untouched", state_path
        )
        state = None
    new_state, legacy = extract_legacy_flags(state)
    settled = 0
    if state:
        settled = sum(
            1 for e in user_flags.parse_entries(state) if e.flagged and e.resolved
        )
        if first_run and new_state and new_state.strip():
            new_state = memory_entries.ensure_format_header(new_state, "state")
    retracted = retractions.list_retractions(orbital_dir) if first_run else []

    with _lock_for(a_path):
        asks_before, asks_decodable = _read_exact(a_path)
        if not asks_decodable:
            raise MigrationError(f"{a_path} is not valid UTF-8; not rewriting it")
        by_id = {a.id: a for a in fold(asks_before)}
        lines, made, _skipped = _plan_opens(by_id, legacy, today)
        known = {_norm(a.text) for a in by_id.values()}
        dropped: list[Ask] = []
        for r in retracted:
            text = _one_line(r.title)
            if not text or _norm(text) in known:
                continue
            rid = r.id.lower() if _ASK_ID_RE.match((r.id or "").lower()) else None
            ask_id = rid if rid and rid not in by_id else _mint_id(by_id)
            when = _valid_date(r.date) or today
            lines.append(render_event(Event("open", ask_id, when, None, text)))
            lines.append(render_event(Event("dropped", ask_id, when, "user", r.reason)))
            ask = Ask(ask_id, text, None, when, "dropped", when, "user", _one_line(r.reason), len(by_id))
            by_id[ask_id] = ask
            known.add(_norm(text))
            dropped.append(ask)

        state_changed = new_state != state
        counts = {"opened": len(made), "dropped": len(dropped), "settled": settled}
        if not lines and not state_changed:
            if first_run:
                _write_marker(marker, now, None, counts)
                return {"status": "migrated", "backup": None, **counts}
            return {"status": "unchanged", "backup": None, **counts}

        backup_dir = _backup(orbital_dir, [state_path, retractions_path, a_path], now)
        if lines:
            _atomic_write(a_path, normalize(asks_before, today) + "\n".join(lines) + "\n")
            written = {a.id: a for a in fold(_read(a_path))}
            for ask in made + dropped:
                got = written.get(ask.id)
                if got is None or got.state != ask.state:
                    raise MigrationError(
                        f"ASKS.md did not read back ask {ask.id} after writing it"
                    )
    if state_changed:
        _atomic_write(state_path, new_state)
    rel_backup = os.path.relpath(backup_dir, orbital_dir)
    if first_run:
        _write_marker(marker, now, rel_backup, counts)
    status = "migrated" if first_run else "swept"
    logger.info(
        "asks migration (%s) for %s: %d opened, %d dropped, %d settled; backup %s",
        status, workspace, counts["opened"], counts["dropped"], counts["settled"],
        rel_backup,
    )
    return {"status": status, "backup": rel_backup, **counts}


def _write_marker(path: str, now: datetime, backup: str | None, counts: dict) -> None:
    payload = {
        "version": 1,
        "migrated_at": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "backup": backup,
        **counts,
    }
    _atomic_write(path, json.dumps(payload, indent=2) + "\n")


def migrate_all(project_store) -> dict[str, dict]:
    """Run ``migrate_project`` for every project with a workspace on disk.

    One project's failure is logged and isolated — it never blocks the
    daemon or the other projects, and it retries on the next start (the
    marker is only written on success).
    """
    results: dict[str, dict] = {}
    try:
        projects = project_store.list_projects()
    except Exception:
        logger.exception("asks migration: could not list projects")
        return results
    for p in projects:
        pid = p.get("project_id") or "?"
        workspace = p.get("workspace") or ""
        if not workspace or not os.path.isdir(workspace):
            continue
        try:
            results[pid] = migrate_project(workspace)
        except Exception as exc:
            logger.exception("asks migration failed for project %s", pid)
            results[pid] = {"status": "error", "error": str(exc)}
    return results
