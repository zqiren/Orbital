# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""PROJECT_STATE.md as a sequence of bullet blocks (spec 089 §4.1, §4.3).

One parser shared by the write chokepoint (which stamps every block with an
id + dates) and the deterministic floor / memory editor (which move blocks
to the archive by id). A block is the unit that moves, so it has to be the
same unit everywhere:

- it starts at a top-level list item (``- `` / ``3. `` / ``12) `` — the same
  ``user_flags.LIST_MARKER`` grammar the rest of the memory code uses);
- it continues through every following non-blank INDENTED line (wrapped
  text, nested bullets, the machine ``<!--mem-->`` comment);
- it ends at a blank line, a heading, or any non-indented line — so a
  pointer line or prose paragraph written straight after a bullet is never
  swallowed into it.

Everything that is not a block (headings, prose, pointer lines, the format
header) is left exactly where it is by every caller.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent_os.agent import user_flags

_BULLET_START_RE = re.compile(rf"^(?P<prefix>{user_flags.LIST_MARKER})(?P<text>\S.*)$")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
_COMMENT_RE = re.compile(r"<!--\s*mem\b(?P<body>.*?)-->", re.DOTALL)
_COMMENT_OPEN_RE = re.compile(r"<!--\s*mem\b")
# A leading `[user …]` / `[due:…]` bracket tag. Only these tokens make it a
# tag — a markdown checkbox (`[ ]`, `[x]`) is part of the sentence.
_TAG_RE = re.compile(r"^\[(?P<tag>[^\]]*)\]\s*")

# Pointer lines left behind by every archiving path (spec 089 §4.4):
#   [archived 2026-09-19 id:c9bf60] GOAI booth prep → PROJECT_STATE_ARCHIVE.md
# Legacy id-less stubs (`[archived 2026-07-28] … → DECISIONS_ARCHIVE.md`)
# match too — they are valid text and age out exactly like new ones.
POINTER_RE = re.compile(r"^\[archived (?P<date>\d{4}-\d{2}-\d{2})\b[^\]]*\]")

COMMENT_INDENT = "  "


@dataclass
class Block:
    start: int                     # line index of the bullet line
    end: int                       # inclusive last line index
    prefix: str                    # exact list marker, e.g. "- " / "3. "
    lines: list[str]               # the block's lines, verbatim
    fields: dict[str, str] = field(default_factory=dict)  # first mem-comment
    has_comment: bool = False
    # The block with every mem-comment removed (what the agent sees).
    content_lines: list[str] = field(default_factory=list)
    # A single-line comment sitting directly under the bullet, verbatim —
    # kept byte-for-byte when its fields do not change.
    comment_line: str | None = None

    @property
    def id(self) -> str | None:
        return self.fields.get("id") or None

    @property
    def created(self) -> str | None:
        return self.fields.get("created") or None

    @property
    def raw(self) -> str:
        return "\n".join(self.lines)

    @property
    def text(self) -> str:
        """The sentence(s) after the list marker, comment-free."""
        first = self.content_lines[0][len(self.prefix):] if self.content_lines else ""
        rest = [ln.strip() for ln in self.content_lines[1:]]
        return " ".join([first.strip(), *rest]).strip()

    @property
    def match_key(self) -> str:
        """Identity for exact-text carry-forward (spec 089 §4.1).

        Whitespace-normalised text with a leading `[user …]`/`[due:…]` tag
        dropped: toggling a tag is not a rewording, the words are the fact.
        """
        return match_key(self.text)


def match_key(text: str) -> str:
    text = text.strip()
    m = _TAG_RE.match(text)
    if m:
        flagged, raw_due = user_flags._parse_tag_tokens(m.group("tag"))
        if flagged or raw_due is not None:
            text = text[m.end():]
    return " ".join(text.split())


def _is_block_line(line: str) -> bool:
    return bool(line.strip()) and line[:1] in (" ", "\t")


def _opens_comment(line: str) -> bool:
    """True when ``line`` starts a mem-comment it does not close."""
    opened = _COMMENT_OPEN_RE.search(line)
    return bool(opened) and "-->" not in line[opened.start():]


def parse(content: str) -> tuple[list[str], list[Block]]:
    """Split ``content`` into lines and the bullet blocks found in it."""
    lines = content.split("\n") if content else []
    blocks: list[Block] = []
    i, n = 0, len(lines)
    while i < n:
        m = _BULLET_START_RE.match(lines[i])
        if not m:
            i += 1
            continue
        j = i + 1
        in_comment = _opens_comment(lines[i])
        opened_at = i
        while j < n:
            line = lines[j]
            if in_comment:
                # A wrapped comment runs to its closing "-->" whatever the
                # indentation of its continuation lines.
                j += 1
                if "-->" in line:
                    in_comment = False
                continue
            if not _is_block_line(line) or _HEADING_RE.match(line):
                break
            in_comment = _opens_comment(line)
            opened_at = j
            j += 1
        if in_comment:
            # Never closed: it is not a comment, and it must not swallow the
            # rest of the file into one movable block.
            j = opened_at + 1
        blocks.append(_make_block(lines, i, j - 1, m.group("prefix")))
        i = j
    return lines, blocks


def _make_block(lines: list[str], start: int, end: int, prefix: str) -> Block:
    raw_lines = lines[start:end + 1]
    joined = "\n".join(raw_lines)
    comments = list(_COMMENT_RE.finditer(joined))
    fields: dict[str, str] = {}
    if comments:
        fields = user_flags._parse_comment_fields(comments[0].group("body"))
    # Strip every comment; drop lines that held nothing but a comment.
    content_lines: list[str] = []
    comment_line: str | None = None
    if comments:
        stripped = _COMMENT_RE.sub("\x00", joined)
        for idx, ln in enumerate(stripped.split("\n")):
            if "\x00" in ln:
                rest = ln.replace("\x00", "")
                if not rest.strip():
                    continue
                ln = rest.rstrip()
            content_lines.append(ln)
        if (
            len(comments) == 1
            and len(raw_lines) > 1
            and _COMMENT_RE.fullmatch(raw_lines[1].strip() or "x")
        ):
            comment_line = raw_lines[1]
    else:
        content_lines = list(raw_lines)
    return Block(
        start=start,
        end=end,
        prefix=prefix,
        lines=raw_lines,
        fields=fields,
        has_comment=bool(comments),
        content_lines=content_lines,
        comment_line=comment_line,
    )


def render(block: Block, fields: dict[str, str]) -> list[str]:
    """The block's content lines with ONE canonical comment under the bullet.

    When the block already carries exactly these fields as a single-line
    comment directly under its bullet, that line is reused verbatim, so a
    no-op write is byte-identical.
    """
    if not block.content_lines:
        return []
    if block.comment_line is not None and block.fields == fields:
        comment = block.comment_line
    else:
        comment = COMMENT_INDENT + user_flags.render_comment(fields)
    return [block.content_lines[0], comment, *block.content_lines[1:]]


def pointer_date(line: str) -> str | None:
    m = POINTER_RE.match(line)
    return m.group("date") if m else None
