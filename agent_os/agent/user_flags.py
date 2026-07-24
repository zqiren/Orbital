# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The `[user]` flag grammar: the ONE shared parser (spec §3, §4).

One judgment, made by the agent at write time, on every layer-1 entry: does
this need to be surfaced to the user? If yes, the bullet carries a `[user]`
bracket tag (machine attributes live in a trailing ``<!--mem ...-->``
comment); a dated fact (no `[user]`, just `[due:...]`) is agent work with a
date and projects to the calendar without being addressed to the user.
Anything without a bracket tag at all is a plain entry — not surfaced here.

This module is consumed by the write chokepoint (lint + id-preserving
merge, Task 2), the Workbench read path (Task 5), and the calendar
`memory` source (Task 6) — one grammar, one place it is understood.

Grammar (spec §4, verbatim example)::

    - [user due:2026-07-28] Send 宝玉 + Simon DM drafts — only you can send
      from your accounts.
      <!--mem id:x7f3a2 from:orbital-marketing_7c045c40
          evidence:"EN 这边宝玉和 Simon 的 draft 写好就准备发" confidence:unconfirmed
          created:2026-07-19 touched:2026-07-23-->

The bracket tag carries ``user`` and/or ``due:<date>`` tokens only; every
other attribute lives in the mem-comment, which may wrap across multiple
indented lines (as above) or sit on a single line. Every existing
PROJECT_STATE (no bracket tags anywhere) must parse to zero entries and
round-trip byte-identical through `strip_mem_comments` — this module never
assumes a project has adopted the grammar yet.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------

import re

# A list marker: a dash bullet, or a numbered item ("3." / "12)"). Captured
# as its own group (named "prefix" in the regexes below) so a rewrite can
# reproduce the exact marker a line was found with — flagging must never
# convert a numbered item into a bullet (spec: tag-in-place grammar).
# Public (no leading underscore): the write chokepoint (flag_chokepoint.py)
# imports this so its own list-item regexes cannot drift from this grammar.
LIST_MARKER = r"(?:-|\d+[.)])\s+"

# A bracket-tagged list item: "<marker> [<tag tokens>] <sentence>". Deliberately
# does NOT match on tag content here — an empty/checkbox bracket ("- [ ]",
# "- [x]" from markdown task lists, common in plans/BACKLOG.md) must fall
# through to "not a tag entry" rather than be misparsed.
_BULLET_TAG_RE = re.compile(rf"^(?P<prefix>{LIST_MARKER})\[(?P<tag>[^\]]*)\]\s*(?P<text>.*)$")

# Any list item, tagged or not. Used (a) as the fallback for tag-less items
# that carry an adjacent mem-comment (a fulfilled Workbench exit rewrites the
# retired entry this way — spec §5.3 — dropping the whole `[user ...]` tag
# but keeping id/resolved in the comment, preserving whatever marker the
# entry was found with) and (b) to recognize "a new item starts here" so a
# comment lookahead never bleeds into the next entry.
_BULLET_RE = re.compile(rf"^(?P<prefix>{LIST_MARKER})(?P<text>.*)$")

# A "## <text>" section heading (exactly two hashes — "#"/"###+" don't count
# as section provenance). Entry.section is the nearest preceding one of these,
# or None above any heading.
_SECTION_HEADING_RE = re.compile(r"^##\s+(?P<text>.*)$")


def _clean_heading(raw: str) -> str:
    return raw.strip().rstrip("#").strip()


def iter_section_headings(content: str) -> list[tuple[int, str]]:
    """``(0-based line index, cleaned heading text)`` for every ``## <text>``
    heading in ``content``, in file order.

    Exposed for consumers that need section-BODY ranges (the Workbench
    in-flight digest — the text between one heading and the next); this same
    detection runs inline inside ``parse_entries`` to stamp ``Entry.section``.
    """
    if not content:
        return []
    out: list[tuple[int, str]] = []
    for i, line in enumerate(content.split("\n")):
        m = _SECTION_HEADING_RE.match(line)
        if m:
            out.append((i, _clean_heading(m.group("text"))))
    return out


# A mem-comment block, possibly wrapped across multiple lines (DOTALL lets
# "." span the embedded newlines of a wrapped comment).
_COMMENT_RE = re.compile(r"<!--\s*mem\s+(?P<body>.*?)-->", re.DOTALL)

# key:value or key:"quoted value with spaces / CJK / escaped quotes".
_ATTR_RE = re.compile(r'(?P<key>[A-Za-z_]+):(?P<val>"(?:[^"\\]|\\.)*"|\S+)')

_DUE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2})?$")

# Canonical field order for rendering (mirrors the spec §4 example).
_COMMENT_FIELD_ORDER = (
    "id", "from", "evidence", "confidence", "created", "touched", "resolved",
)


@dataclass(frozen=True)
class Entry:
    id: str | None
    text: str
    prefix: str                    # exact list marker as found: "- ", "3. ", "12) "
    flagged: bool
    due: str | None                # raw "2026-07-28" or "2026-07-28T20:00"
    from_session: str | None
    evidence: str | None
    confidence: str | None         # "stated" | "unconfirmed" | None
    created: str | None
    touched: str | None
    resolved: str | None
    section: str | None            # nearest preceding "## " heading text, or None
    line_start: int                # 0-based, inclusive
    line_end: int                  # 0-based, inclusive; comment line included


def _unquote(raw: str) -> str:
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        inner = raw[1:-1]
        return inner.replace('\\"', '"').replace("\\\\", "\\")
    return raw


def _quote(val: str) -> str:
    escaped = val.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _parse_tag_tokens(tag: str) -> tuple[bool, str | None]:
    """Return (flagged, raw_due) from a bracket tag's whitespace-split tokens.

    ``raw_due`` is returned even when malformed (e.g. "due:tomorrow") — the
    caller decides whether to surface it as ``Entry.due`` (only when valid)
    vs. a lint warning (always, when present but invalid).
    """
    flagged = False
    raw_due: str | None = None
    for tok in tag.split():
        if tok == "user":
            flagged = True
        elif tok.startswith("due:"):
            raw_due = tok[len("due:"):]
    return flagged, raw_due


def _parse_comment_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for m in _ATTR_RE.finditer(body):
        fields[m.group("key")] = _unquote(m.group("val"))
    return fields


def parse_entries(content: str) -> list[Entry]:
    """Parse every bullet that has a bracket tag OR an adjacent mem-comment.

    A bracket-tagged bullet (``[user]``/``[due:...]``) is always an entry,
    comment or not. A tag-less bullet — no ``[...]`` at all, including
    markdown checkbox syntax (``- [ ]``, ``- [x]``) — is an entry only when
    immediately followed by a ``<!--mem ...-->`` comment (this is how a
    fulfilled Workbench exit leaves a retired entry: the whole tag dropped,
    but id/``resolved:`` kept in the comment as the anti-resurrection trace —
    spec §5.3). A plain bullet with neither a tag nor a comment is not
    returned.
    """
    if not content:
        return []
    lines = content.split("\n")
    n = len(lines)
    entries: list[Entry] = []
    i = 0
    current_section: str | None = None
    while i < n:
        heading_m = _SECTION_HEADING_RE.match(lines[i])
        if heading_m:
            current_section = _clean_heading(heading_m.group("text"))
            i += 1
            continue

        tag_m = _BULLET_TAG_RE.match(lines[i])
        if tag_m:
            flagged, raw_due = _parse_tag_tokens(tag_m.group("tag"))
            has_tag = flagged or raw_due is not None
            text = tag_m.group("text")
            text_pos = tag_m.start("text")
            prefix = tag_m.group("prefix")
        else:
            plain_m = _BULLET_RE.match(lines[i])
            if not plain_m:
                i += 1
                continue
            flagged, raw_due, has_tag = False, None, False
            text = plain_m.group("text")
            text_pos = plain_m.start("text")
            prefix = plain_m.group("prefix")

        line_start = i
        line_end = i
        fields: dict[str, str] = {}
        comment_found = False

        # Same-line trailing comment (fully closed on the bullet's own line).
        same_line_cm = _COMMENT_RE.search(lines[i])
        if same_line_cm:
            text = lines[i][text_pos:same_line_cm.start()].strip()
            fields = _parse_comment_fields(same_line_cm.group("body"))
            comment_found = True
        else:
            text = text.strip()
            # Look ahead for a comment starting on the immediate next line,
            # possibly wrapped across several indented lines.
            j = i + 1
            collected: list[str] = []
            while j < n:
                cand = lines[j]
                if cand.strip() == "":
                    break
                if not collected and _BULLET_RE.match(cand):
                    break  # next bullet starts immediately — no comment here
                collected.append(cand)
                joined = "\n".join(collected)
                cm = _COMMENT_RE.search(joined)
                if cm and "-->" in joined:
                    fields = _parse_comment_fields(cm.group("body"))
                    line_end = j
                    comment_found = True
                    break
                j += 1

        if not has_tag and not comment_found:
            i += 1
            continue  # plain bullet, no tag, no comment — not an entry

        due = raw_due if raw_due and _DUE_RE.match(raw_due) else None
        entries.append(Entry(
            id=fields.get("id"),
            text=text,
            prefix=prefix,
            flagged=flagged,
            due=due,
            from_session=fields.get("from"),
            evidence=fields.get("evidence"),
            confidence=fields.get("confidence"),
            created=fields.get("created"),
            touched=fields.get("touched"),
            resolved=fields.get("resolved"),
            section=current_section,
            line_start=line_start,
            line_end=line_end,
        ))
        i = line_end + 1
    return entries


def strip_mem_comments(content: str) -> str:
    """Remove every ``<!--mem ...-->`` block from ``content``.

    A comment that occupies whole line(s) on its own (only leading/trailing
    whitespace besides the comment) has those line(s) removed entirely,
    including their newline, so no residual blank line is left behind.
    Content with no mem-comments at all is returned byte-identical (the
    common case: a project that hasn't adopted the grammar yet).
    """
    if not content:
        return content
    out: list[str] = []
    pos = 0
    for m in _COMMENT_RE.finditer(content):
        start, end = m.start(), m.end()
        line_start = content.rfind("\n", 0, start) + 1
        prefix_on_line = content[line_start:start]
        line_end = content.find("\n", end)
        if line_end == -1:
            line_end = len(content)
        suffix_on_line = content[end:line_end]
        if prefix_on_line.strip() == "" and suffix_on_line.strip() == "":
            remove_start = line_start
            remove_end = line_end + 1 if line_end < len(content) else line_end
        else:
            remove_start, remove_end = start, end
        out.append(content[pos:remove_start])
        pos = remove_end
    out.append(content[pos:])
    result = "".join(out)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


def render_comment(fields: dict[str, str]) -> str:
    """Render a canonical single-line ``<!--mem ...-->`` comment.

    Fields in ``_COMMENT_FIELD_ORDER`` come first in that order; any extra
    keys follow in insertion order. Values containing whitespace are quoted
    (with ``"``/``\\`` escaped); missing/empty fields are omitted.
    """
    parts: list[str] = []
    seen: set[str] = set()
    for key in _COMMENT_FIELD_ORDER:
        seen.add(key)
        val = fields.get(key)
        if not val:
            continue
        parts.append(f"{key}:{_quote(val) if ' ' in val else val}")
    for key, val in fields.items():
        if key in seen or not val:
            continue
        parts.append(f"{key}:{_quote(val) if ' ' in val else val}")
    return "<!--mem " + " ".join(parts) + "-->"


def new_entry_id() -> str:
    """A short, content-independent id — 6 hex chars, daemon-assigned."""
    return secrets.token_hex(3)


def lint(content: str) -> list[str]:
    """Grammar-level warnings for ``content``. Never rejects — warns only.

    v1 (Task 1) scope: malformed ``due:`` values, and flagged entries with
    no receipts at all — neither ``evidence`` (a verbatim quote) nor
    ``confidence`` (e.g. ``confidence:unconfirmed`` when no quote exists).
    The one-voice contract (spec §3, commit 050e6dd): evidence must be a
    real verbatim quote or absent — never fabricated — so a compliant
    quote-less entry carries ``confidence:unconfirmed`` instead and must
    warn zero times. ``from:`` is no longer instructed and is never linted.
    The omission heuristic (unflagged user-directed phrasing) is Task 2 —
    it needs section-heading context this module doesn't have.
    """
    if not content:
        return []
    warnings: list[str] = []
    for i, line in enumerate(content.split("\n")):
        m = _BULLET_TAG_RE.match(line)
        if not m:
            continue
        _, raw_due = _parse_tag_tokens(m.group("tag"))
        if raw_due is not None and not _DUE_RE.match(raw_due):
            warnings.append(
                f"line {i + 1}: malformed due '{raw_due}' — expected "
                "YYYY-MM-DD or YYYY-MM-DDTHH:MM"
            )
    for e in parse_entries(content):
        if not e.flagged:
            continue
        if not e.evidence and not e.confidence:
            warnings.append(
                f"line {e.line_start + 1}: flagged entry missing receipts "
                f"— add confidence:unconfirmed or a verbatim quote as "
                f"evidence (\"{e.text[:60]}\")"
            )
    return warnings
