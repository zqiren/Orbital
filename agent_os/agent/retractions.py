# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The retractions store: the permanent "user said no" record (spec §3, §5.2).

A retraction is written once, from the Workbench's "Not relevant" exit
(Task 5) or a chokepoint-detected re-add warning (Task 2) — never from an
agent directly. ``orbital/retractions.md`` is **append-only from this
module's perspective**: nothing here ever rewrites or removes a prior line,
so the record survives regardless of what else happens to the project.

The store is consumed two ways:
- ``render_constraints`` / injection into every session (``context.py``) as
  a hard constraint block — the agent sees "the user said no to this" on
  every turn, not just when it happens to re-read the file.
- ``normalized_title_match`` — used by the write chokepoint (`flag_chokepoint
  .reconcile_flags`) to catch a retracted entry re-appearing under a
  rephrased title and keep it out (spec §5.2's layered matching: exact id,
  mechanical; normalized-title similarity, mechanical warn).

File format, one line per retraction (spec §5.2 example)::

    - [x7f3a2] "Send 宝玉 + Simon DM drafts" — retracted by user 2026-07-24: changed my mind

The title is double-quoted (with ``"``/``\\`` escaped, mirroring
``user_flags``'s comment-value quoting) so it can safely contain the literal
" — retracted by user " separator, CJK text, or punctuation. The reason is
free text to end of line — it does NOT need escaping, so colons or further
em dashes inside a reason round-trip untouched.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

RETRACTIONS_FILENAME = "retractions.md"

# Fuzzy title-match threshold — same tier the write chokepoint uses for
# re-association (spec §5.2 / flag_chokepoint._FUZZY_THRESHOLD).
_FUZZY_THRESHOLD = 0.75

_LINE_RE = re.compile(
    r'^-\s+\[(?P<id>[^\]]+)\]\s+"(?P<title>(?:[^"\\]|\\.)*)"\s+—\s+'
    r"retracted by user (?P<date>\d{4}-\d{2}-\d{2}):\s?(?P<reason>.*)$"
)


@dataclass(frozen=True)
class Retraction:
    id: str
    title: str
    reason: str
    date: str


def _unescape(raw: str) -> str:
    return raw.replace('\\"', '"').replace("\\\\", "\\")


def _escape(val: str) -> str:
    return val.replace("\\", "\\\\").replace('"', '\\"')


def _path(orbital_dir: Path | str) -> Path:
    return Path(orbital_dir) / RETRACTIONS_FILENAME


def _render_line(r: Retraction) -> str:
    return f'- [{r.id}] "{_escape(r.title)}" — retracted by user {r.date}: {r.reason}'


def add_retraction(orbital_dir: Path | str, r: Retraction) -> None:
    """Append ``r`` to ``<orbital_dir>/retractions.md``. Creates dir/file if needed.

    Never trims or rewrites existing lines — a fresh line is added at the end.
    """
    path = _path(orbital_dir)
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(_render_line(r) + "\n")


def list_retractions(orbital_dir: Path | str) -> list[Retraction]:
    """Return every retraction ever recorded, oldest first. `[]` if no file."""
    path = _path(orbital_dir)
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return []
    out: list[Retraction] = []
    for line in content.split("\n"):
        m = _LINE_RE.match(line)
        if not m:
            continue
        out.append(Retraction(
            id=m.group("id"),
            title=_unescape(m.group("title")),
            reason=m.group("reason"),
            date=m.group("date"),
        ))
    return out


def render_constraints(rs: list[Retraction]) -> str:
    """Render ``rs`` as a hard-constraint block for session context. `""` if empty."""
    if not rs:
        return ""
    lines = [
        "## Retracted by user — hard constraints",
        "",
        "The user explicitly said no to each item below. Do not re-propose, "
        "re-infer, or re-add any of them under any wording — they may only "
        "return by explicit user request.",
        "",
    ]
    for r in rs:
        lines.append(f'- "{r.title}" (retracted {r.date}: {r.reason})')
    return "\n".join(lines)


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation/whitespace/underscore (keep letters incl.
    CJK + digits) — same normalization idea as flag_chokepoint._norm_for_match."""
    return re.sub(r"[\W_]+", "", text.lower())


def normalized_title_match(title: str, rs: list[Retraction]) -> Retraction | None:
    """Match ``title`` against ``rs`` in two tiers: exact id, then fuzzy title.

    ``title`` may itself be an id (callers checking "was this entry already
    retracted?" often have the id in hand) — an exact ``id`` match wins
    first. Otherwise, normalized-title similarity (difflib ratio) is used;
    the best match at or above ``_FUZZY_THRESHOLD`` (0.75) is returned, else
    ``None``.
    """
    if not title or not rs:
        return None
    for r in rs:
        if r.id == title:
            return r
    target = _normalize(title)
    best: Retraction | None = None
    best_ratio = 0.0
    for r in rs:
        ratio = SequenceMatcher(None, target, _normalize(r.title)).ratio()
        if ratio > best_ratio:
            best_ratio, best = ratio, r
    return best if best is not None and best_ratio >= _FUZZY_THRESHOLD else None
