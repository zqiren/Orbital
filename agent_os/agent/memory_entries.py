# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Layer-1 memory: one entry shape, deterministic normalisation, ids, the
injected view, archive contents tables and the id-conservation guard
(spec 124).

The spine of the bounded-memory model. One place that:

  - defines the ONE entry shape every memory file uses (spec 124 §3.1): a
    top-level bullet line that stands on its own, optional indented detail,
    and a daemon-managed ``<!--mem id:… created:… touched:…-->`` comment on
    the line under the bullet. ``## `` headings are sections, not entries;
  - normalises whatever an agent (or a legacy daemon) wrote into that shape
    (``normalise`` — rules N1–N5, lossless and idempotent) at every write
    path and in the tidy pass's deterministic pre-step;
  - stamps ids that are unique across the live file AND its archive, never
    rewriting an existing one (``stamp``);
  - renders the injected view (``inject_view``): every bullet line always,
    indented detail only for the most recently touched / pinned entries, and
    folding by ``## `` section when the bullet lines alone overflow;
  - maintains the file-level pointers: the contents table at the top of
    every ``*_ARCHIVE.md`` and the one-line-per-archive entries in INDEX;
  - guards ids: an entry that leaves a live file must be in its archive or
    it comes back (``conserve_ids`` at the write path, ``restore_missing``
    after a pass) — spec 070's "absence never means removal".

Nothing here archives by age. Archiving is a judgement ("superseded or
finished") and belongs to the Memory session; the daemon only keeps the
shape, the ids and the budget of what is injected.

Token accounting uses ``len(text) / 4`` to match the rest of the codebase
(``token_utils.estimate_message_tokens``, ``context.py``).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import date

from agent_os.agent import state_blocks, user_flags
from agent_os.utils.text_io import read_text_lenient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Budgets (len/4). Floors are the measured clean-mature sizes + headroom
# (VERIFY-phase1-injection-cap.md Q2), set ABOVE the full clean set so a healthy
# project injects everything and only a runaway overflows.
# ---------------------------------------------------------------------------

FILE_BUDGETS: dict[str, dict[str, int]] = {
    "decisions": {"soft": 7000, "hard": 9000},
    "lessons": {"soft": 5000, "hard": 6000},
    # state soft bumped 1500 -> 1800 (spec §4.1 resolution 3): headroom for the
    # [user] register rule's sentence overhead, measured against the flagship
    # over-soft project. Hard cap unchanged.
    "state": {"soft": 1800, "hard": 2000},
    "index": {"soft": 1500, "hard": 2000},
}

# Consolidation aims BELOW the soft budget, never at it. A pass that lands
# exactly on the threshold re-trips the hygiene flag as soon as the next entry
# is appended, which is how a project ends up checkpointing continuously
# without ever getting quieter. The headroom is what makes a pass last.
CONSOLIDATION_HEADROOM_TOKENS = 1000

# ...but a flat 1000 would gut the small volatile files (state soft=1800,
# index soft=1500), so the target never demands more than a 40% cut.
_MIN_TARGET_FRACTION = 0.6


def consolidation_target(key: str) -> int:
    """Token count a tidy pass should bring ``key`` down to.

    Sits a full ``CONSOLIDATION_HEADROOM_TOKENS`` below the soft budget for the
    entry-structured files that grow by appending (DECISIONS, LESSONS), and a
    proportional margin below it for the small freeform ones.
    """
    budgets = FILE_BUDGETS.get(key)
    if not budgets:
        return 0
    soft = budgets["soft"]
    return max(soft - CONSOLIDATION_HEADROOM_TOKENS, int(soft * _MIN_TARGET_FRACTION))


# Guard for tiny-window models only: total injected Layer-1 must stay under this
# fraction of the active model's context window. For any window >= ~80k the
# floors above already fit, so the floor is what bites in practice.
WINDOW_SHARE = 0.25

# The three entry-structured files share one shape (spec 124 §3.1). INDEX has
# no entries and no ids.
ENTRY_KEYS = ("state", "decisions", "lessons")
DURABLE_KEYS = ("decisions", "lessons")   # kept for callers that iterate them
VOLATILE_KEYS = ("state", "index")        # freeform lint (shape_report) only

# Display names + on-disk filenames for the injection-omission note and flags.
DISPLAY_NAME: dict[str, str] = {
    "decisions": "DECISIONS",
    "lessons": "LESSONS",
    "state": "PROJECT_STATE",
    "index": "INDEX",
}
MEMORY_FILENAME: dict[str, str] = {
    "decisions": "DECISIONS.md",
    "lessons": "LESSONS.md",
    "state": "PROJECT_STATE.md",
    "index": "INDEX.md",
}
# Layer-1 file -> its Layer-2 archive file-key (read-on-demand, never
# injected, never budgeted). INDEX stays absent on purpose — it is
# regenerable navigation, so a stale line there is noise, not history.
ARCHIVE_OF: dict[str, str] = {
    "decisions": "decisions_archive",
    "lessons": "lessons_archive",
    "state": "state_archive",
}
ARCHIVE_FILENAME: dict[str, str] = {
    "decisions_archive": "DECISIONS_ARCHIVE.md",
    "lessons_archive": "LESSONS_ARCHIVE.md",
    "state_archive": "PROJECT_STATE_ARCHIVE.md",
}
ARCHIVE_KEYS = tuple(ARCHIVE_FILENAME)

# ---------------------------------------------------------------------------
# In-file format contracts. One-line HTML comments: invisible in rendered
# markdown, sit on line 1 so they survive the volatile head-trim, and are
# excluded from the file's own budget. Self-healing at BOTH write
# chokepoints (WorkspaceFileManager.write for system writes, process_on_write
# for the agent's write/edit tools). The format detail deliberately lives
# HERE rather than in the system prompt: adjacent to the content when the
# agent reads/edits the file, and visible to humans opening the raw file.
# ---------------------------------------------------------------------------

_ONE_SHAPE = (
    "One entry = one `- ` bullet line that stands on its own; optional detail "
    "goes on indented lines under it, below its mem-comment line when it has "
    "one (the exception, not the norm). `## ` "
    "headings group entries by topic and are not entries. Machine attributes "
    "(id, created, touched) live in a daemon-managed mem-comment on the line "
    "under the bullet — never write or edit these comments; leave them "
    "exactly where they are."
)

# Spec 182: how to replace an entry. The daemon's write path acts on the line
# (``resolve_supersedes``); the same words ride the record row (loop.py
# ``MEMORY_SHAPE_RULES``).
_SUPERSEDES_RULE = (
    " When a {noun} changes, keep the old entry as it is and write the new one "
    "with one indented line under it: supersedes: <the old bullet line, word "
    "for word>. The old entry then moves to the archive."
)

FORMAT_HEADERS: dict[str, str] = {
    "index": (
        "<!--format INDEX is a navigation map ONLY: '- path — one sentence' "
        "bullets under '## <area>' headings. List folders and topics, not "
        "individual dated files; a single file gets its own line only when it "
        "is the source of truth for a kind of output. No dates, status, "
        "decisions, or lessons here — those live in PROJECT_STATE.md / "
        "DECISIONS.md / LESSONS.md. Keep every *_ARCHIVE.md line: it says "
        "what that archive holds (the daemon fills its count and dates).-->"
    ),
    "state": (
        "<!--format PROJECT_STATE is what is true NOW: current focus, "
        "in-progress work, blockers, next steps. " + _ONE_SHAPE + " Overwrite "
        "stale lines; never append dated history. Every line must be "
        "understandable without this session's context: concrete names, no "
        "unexplained shorthand, no cross-references by list number. One fact "
        "= one entry, never duplicated into another section. A dated fact is "
        "`[due:YYYY-MM-DD]` right after the list marker (shows on the "
        "calendar). Write timeless (\"due Jul 28\", never \"tomorrow\"). "
        "Things waiting on the user (their decision, their action, something "
        "they said they will do) do NOT go here: they are asks in "
        "orbital/ASKS.md (see that file's header); a `[user]` line written "
        "here is moved there automatically. Never auto-decide: spending money, "
        "sending external messages as the user, or irreversible/destructive "
        "acts always go to the user, whatever the autonomy setting.-->"
    ),
    "decisions": (
        "<!--format DECISIONS holds durable decisions: the bullet line says "
        "what was chosen and why, in one sentence; rejected alternatives and "
        "longer reasoning go on indented lines under it. " + _ONE_SHAPE
        + _SUPERSEDES_RULE.format(noun="decision") + "-->"
    ),
    "lessons": (
        "<!--format LESSONS holds durable heuristics and playbooks: the bullet "
        "line is the rule itself, in one sentence; the incident and playbook "
        "steps go on indented lines under it. Add a lesson on error recovery "
        "or a non-obvious workaround. " + _ONE_SHAPE
        + _SUPERSEDES_RULE.format(noun="lesson") + "-->"
    ),
}


def ensure_format_header(content: str | None, key: str) -> str:
    """Prepend the file's <!--format--> contract, or upgrade a stale one.

    The contract is code-owned, so a divergent first-line header is replaced
    with the current contract. Content after the header is preserved
    byte-for-byte; an already-current header is returned untouched
    (idempotent).
    """
    header = FORMAT_HEADERS.get(key)
    text = content or ""
    if header is None:
        return text
    if text.lstrip().startswith("<!--format"):
        return force_format_header(text, key)
    return header + "\n" + text


def force_format_header(content: str | None, key: str) -> str:
    """Replace an existing ``<!--format-->`` header with the current template.

    Content after the header is preserved byte-for-byte; a file with no
    header gets one prepended.
    """
    header = FORMAT_HEADERS.get(key)
    text = content or ""
    if header is None:
        return text
    if text.lstrip().startswith("<!--format"):
        start = text.find("<!--format")
        end = text.find("-->", start)
        if end != -1:
            prefix = text[:start]
            after = text[end + 3:]
            if after.startswith("\n"):
                after = after[1:]
            return prefix + header + "\n" + after
    return header + "\n" + text


# --- Report-only shape lint (v1: volatile files only) -----------------------
# NEVER a hygiene-flag trigger: consumed exclusively by the tidy prompt
# ("FORMATTING TO FIX") so formatting tidy-up rides along a pass that runs
# anyway. Imposing format must not increase checkpoint frequency.

_SHAPE_DATE_RE = re.compile(r"\b\d{4}-\d{2}(-\d{2})?\b(?![-\w]|\.\w)")
_SHAPE_EMOJI_RE = re.compile(r"[🚨✅⚠]")
_INDEX_MAP_SHAPE_RE = re.compile(r"^(#{1,3} |- \S.* — )")
_STATE_CHANGELOG_RE = re.compile(r"^#{1,3} .*\b\d{4}-\d{2}\b")
_INDEX_NON_MAP_MAX_RATIO = 0.4


def shape_report(content: str | None, key: str) -> str | None:
    """One-line lint summary for a drifted volatile file, or None when clean.

    Conservative by design (few rules, counts not line-dumps) — a false
    positive here nags every tidy pass.
    """
    if key not in VOLATILE_KEYS or not content or not content.strip():
        return None
    lines = [
        ln for ln in content.splitlines()
        if ln.strip() and not ln.lstrip().startswith("<!--")
    ]
    if not lines:
        return None

    if key == "state":
        dated = sum(1 for ln in lines if _STATE_CHANGELOG_RE.match(ln))
        if dated:
            return (
                f"PROJECT_STATE: {dated} dated changelog-style header(s) — "
                "state is overwrite-in-place, not a history."
            )
        return None

    problems: list[str] = []
    dated = sum(1 for ln in lines if _SHAPE_DATE_RE.search(ln))
    emoji = sum(1 for ln in lines if _SHAPE_EMOJI_RE.search(ln))
    non_map = sum(1 for ln in lines if not _INDEX_MAP_SHAPE_RE.match(ln))
    if dated:
        problems.append(f"{dated} dated line(s)")
    if emoji:
        problems.append(f"{emoji} status-emoji line(s)")
    ratio = non_map / len(lines)
    if ratio > _INDEX_NON_MAP_MAX_RATIO:
        problems.append(
            f"{round(ratio * 100)}% of lines not in 'path — sentence' map shape"
        )
    if not problems:
        return None
    return "INDEX: " + ", ".join(problems)


def est_tokens(text: str | None) -> float:
    """len/4 token estimate — matches ``token_utils`` / ``context.py``."""
    return len(text) / 4 if text else 0.0


def strip_format_header(content: str | None) -> str:
    """Drop a leading ``<!--format ...-->`` contract line, if present.

    The contract is code-owned scaffolding: charging it to the file's own
    budget would mean every rail added to the contract silently steals space
    from real project memory.
    """
    text = content or ""
    stripped = text.lstrip()
    if not stripped.startswith("<!--format"):
        return text
    end = stripped.find("-->")
    if end == -1:
        return text
    rest = stripped[end + 3:]
    return rest[1:] if rest.startswith("\n") else rest


def _budget_text(content: str, key: str) -> str:
    """Content used for the soft-budget measure (spec 124 §3.4 step 5).

    For the three entry files the soft budget counts BULLET LINES only: the
    format header, every mem-comment and every indented detail line are
    excluded (detail is injected only for hot entries, so it must not trip
    the flag). Headings and any loose line count — they are injected. INDEX
    is measured header-stripped.
    """
    text = strip_format_header(content)
    if key not in ENTRY_KEYS:
        return text
    lines, blocks = state_blocks.parse(text)
    hidden: set[int] = set()
    for b in blocks:
        hidden.update(range(b.start + 1, b.end + 1))
    kept = []
    for i, ln in enumerate(lines):
        if i in hidden:
            continue
        if _is_comment_only(ln):
            continue
        kept.append(_strip_inline_comment(ln))
    return "\n".join(kept)


def _today() -> str:
    return date.today().isoformat()


# ---------------------------------------------------------------------------
# Budget derivation (WU0/WU3) — derived from the active model's window.
# ---------------------------------------------------------------------------

def budgets_for_window(context_window: int | None) -> dict[str, dict[str, int]]:
    """Per-file ``{soft, hard}`` budgets for the active model's window.

    Returns the measured floors unless a *tiny* window forces them lower via the
    ``WINDOW_SHARE`` guard. ``context_window`` is the value ``ContextManager``
    already derives from ``providers.json`` (``model_info.context_window``); a
    missing/zero window falls back to the floors with a warning.
    """
    base = {k: dict(v) for k, v in FILE_BUDGETS.items()}
    if not context_window or context_window <= 0:
        logger.warning(
            "memory budgets: no context_window for active model; using floors"
        )
        return base
    total_hard = sum(v["hard"] for v in base.values())
    cap = WINDOW_SHARE * context_window
    if total_hard > cap and total_hard > 0:
        scale = cap / total_hard
        for v in base.values():
            v["hard"] = int(v["hard"] * scale)
            v["soft"] = int(v["soft"] * scale)
        logger.warning(
            "memory budgets: window %d too small for floors; scaled by %.2f",
            context_window, scale,
        )
    return base


# ---------------------------------------------------------------------------
# Line grammar shared by normalise / stamp / view
# ---------------------------------------------------------------------------

_COMMENT_RE = state_blocks._COMMENT_RE                  # <!--mem …-->, DOTALL
_LIST_RE = re.compile(rf"^(?P<prefix>{user_flags.LIST_MARKER})(?P<text>\S.*)$")
_NUMBERED_RE = re.compile(r"^\d+[.)]\s+")
_BOLD_LEAD_RE = re.compile(r"^\*\*(?P<title>[^*\n]+?)\*\*(?P<rest>.*)$")
_MERGED_RE = re.compile(r"^\[merged \d{4}-\d{2}-\d{2}\b")
_VIEW_NOTE_RE = re.compile(r"^\[\.\.\. .*? \.\.\.\]")
_FORMAT_LINE_RE = re.compile(r"^\s*<!--format\b")
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_LEAD_DATE_RE = re.compile(r"^\d{4}-\d{2}(-\d{2})?[:：]?\s*")
_PATH_RE = re.compile(r"(?:[\w.\-]+/)+[\w.\-]*|[\w\-]+\.(?:md|json|csv|html|py|pdf|tsv|txt)\b")
_POINTER_ID_RE = re.compile(r"\bid:([^\s\]]+)")


def _is_comment_only(line: str) -> bool:
    s = line.strip()
    return bool(s) and _COMMENT_RE.fullmatch(s) is not None


def _strip_inline_comment(line: str) -> str:
    if "<!--" not in line:
        return line
    return _COMMENT_RE.sub("", line).rstrip()


def _comment_fields(line: str) -> dict[str, str]:
    m = _COMMENT_RE.search(line)
    return user_flags._parse_comment_fields(m.group("body")) if m else {}


def _is_pointer(line: str) -> bool:
    return state_blocks.pointer_date(line) is not None


def _is_indented(line: str) -> bool:
    return bool(line.strip()) and line[:1] in (" ", "\t")


def _is_blank(line: str) -> bool:
    return not line.strip()


def _strip_bold_lead(text: str) -> str:
    """``**Title** rest`` → ``Title rest``; ``**Title.**`` → ``Title.``."""
    m = _BOLD_LEAD_RE.match(text)
    if not m:
        return text
    rest = m.group("rest")
    title = m.group("title").strip()
    if rest and rest[:1].isascii() and rest[:1].isalnum():
        return f"{title} {rest}"
    return (title + rest).rstrip()


def _quote(line: str, limit: int = 160) -> str:
    text = line.rstrip()
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return f'"{text}"'


# ---------------------------------------------------------------------------
# normalise (spec 124 §3.2): N1–N5, lossless, idempotent
# ---------------------------------------------------------------------------

@dataclass
class PointerLine:
    """A legacy ``[archived DATE id:X] … → ARCHIVE`` line removed from a live
    file by rule N5. The caller appends it to the archive when its ids are not
    already there, archive first, then writes the live file."""
    line: str
    date: str
    ids: list[str]
    paths: list[str]


@dataclass
class NormaliseResult:
    text: str
    warnings: list[str] = field(default_factory=list)
    pointers: list[PointerLine] = field(default_factory=list)
    dropped_comment_ids: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.warnings) or bool(self.pointers)

    @property
    def pointer_paths(self) -> list[str]:
        """Paths the folded pointers named, for the tidy task message
        (spec 124 §3.2 N5): deduplicated, in file order."""
        seen: list[str] = []
        for p in self.pointers:
            for path in p.paths:
                if path not in seen:
                    seen.append(path)
        return seen


def _parse_pointer(line: str) -> PointerLine:
    day = state_blocks.pointer_date(line) or ""
    head_end = line.find("]")
    head = line[: head_end + 1] if head_end >= 0 else line
    ids = _POINTER_ID_RE.findall(head)
    body = line[head_end + 1:] if head_end >= 0 else line
    body = re.sub(r"\s*(?:→|->)\s*[A-Z_]+_ARCHIVE\.md\s*$", "", body)
    paths = []
    for m in _PATH_RE.finditer(body):
        tok = m.group(0).rstrip(".,;:")
        if tok and tok not in paths and not _ISO_DATE_RE.fullmatch(tok):
            paths.append(tok)
    return PointerLine(line=line, date=day, ids=ids, paths=paths)


def _has_structure(lines: list[str]) -> bool:
    """A file with at least one section heading or list item has a shape to
    normalise into. Pure prose has nothing to anchor on: it is left as-is
    with a note about the expected shape."""
    for ln in lines:
        if state_blocks.heading_level(ln) >= 2 or _LIST_RE.match(ln):
            return True
    return False


def _normalise_scaffolding(lines: list[str], warnings: list[str]) -> list[str]:
    """Rule N0: daemon scaffolding pasted back into the file is removed —
    a second ``<!--format-->`` line, or a view-trim note (``[... older
    content trimmed ... ]``) glued onto a line."""
    out: list[str] = []
    for i, ln in enumerate(lines):
        if i > 0 and _FORMAT_LINE_RE.match(ln):
            warnings.append("removed a duplicate <!--format--> header line")
            continue
        m = _VIEW_NOTE_RE.match(ln)
        if m:
            rest = ln[m.end():].lstrip()
            warnings.append("removed a pasted view note " + _quote(m.group(0), 60))
            if not rest:
                continue
            ln = rest
        out.append(ln)
    return out


def _heading_body(lines: list[str], i: int) -> int:
    """Index just past the body of the heading at ``i`` (next heading or EOF)."""
    j = i + 1
    while j < len(lines) and state_blocks.heading_level(lines[j]) == 0:
        j += 1
    return j


def _entry_body_end(lines: list[str], i: int, end: int) -> int:
    """A legacy heading entry's body runs to the next heading, except that a
    blank line followed by a top-level list item starts an independent entry."""
    blank_seen = False
    for k in range(i + 1, end):
        if _is_blank(lines[k]):
            blank_seen = True
            continue
        if blank_seen and _LIST_RE.match(lines[k]):
            return k
    return end


def _first_paragraph(body: list[str]) -> list[str]:
    para: list[str] = []
    for ln in body:
        if _is_blank(ln):
            if para:
                break
            continue
        para.append(ln)
    return para


def _normalise_heading_entries(lines: list[str], warnings: list[str]) -> list[str]:
    """Rule N1 (DECISIONS / LESSONS only): a legacy ``## Title`` entry becomes
    ``- Title`` with its mem-comment on the next line and its body indented.

    A heading is a legacy entry when it carries a mem-comment (inline, or on
    the line right under it), when its text is numbered (``## 2. Title``),
    or when the first paragraph under it is prose rather than list items.
    A heading whose first paragraph is top-level bullets is a section and
    is left alone. An entry's body runs to the next heading, except that a
    blank line followed by a top-level list item starts a new entry.
    """
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        level = state_blocks.heading_level(ln)
        if level < 2:
            out.append(ln)
            i += 1
            continue
        end = _heading_body(lines, i)
        own_comment: str | None = None
        inline = _COMMENT_RE.search(ln)
        text = state_blocks.heading_text(ln)
        if inline:
            own_comment = inline.group(0)
            text = _strip_inline_comment(text)
        body_start = i + 1
        if own_comment is None and body_start < end and _is_comment_only(lines[body_start]):
            own_comment = lines[body_start].strip()
            body_start += 1
        para = _first_paragraph(lines[body_start:end])
        has_text = any(
            not _is_pointer(b) and not _is_comment_only(b) and not _MERGED_RE.match(b)
            for b in para
        )
        has_items = any(_LIST_RE.match(b) for b in para)
        is_entry = (
            own_comment is not None
            or bool(_NUMBERED_RE.match(text))
            or (has_text and not has_items)
        )
        if not is_entry:
            out.append(ln)
            i += 1
            continue
        end = _entry_body_end(lines, body_start - 1, end)
        content_body = lines[body_start:end]
        title = _strip_bold_lead(_NUMBERED_RE.sub("", text).strip())
        bullet = f"- {title}"
        out.append(bullet)
        if own_comment:
            out.append(state_blocks.COMMENT_INDENT + own_comment)
        saw_blank = False
        for b in content_body:
            if _is_blank(b):
                saw_blank = True
                continue
            if _is_pointer(b):
                out.append(b)
                continue
            if _is_comment_only(b):
                warnings.append(
                    "dropped a mem-comment that was not under its bullet "
                    + _quote(b.strip(), 80)
                )
                continue
            if "<!--" in b and _COMMENT_RE.search(b):
                b = _strip_inline_comment(b)
                warnings.append("dropped a mem-comment inside the text of " + _quote(b, 80))
            out.append("  " + b)
        if saw_blank:
            out.append("")
        warnings.append(
            f"rewrote the heading entry {_quote(ln.rstrip(), 80)} as a bullet: "
            + _quote(bullet)
        )
        i = end
    return out


def _normalise_items_and_prose(
    lines: list[str], key: str, warnings: list[str],
) -> list[str]:
    """Rules N2 + N3 (+ inline-comment relocation).

    N2 (DECISIONS / LESSONS): ``N. **T** …`` / ``- **T** …`` → ``- T …``;
    non-indented lines that follow an entry without a blank line are its
    continuation and get indented (so numbered playbook steps are detail,
    not entries — spec §1.3).
    N3: a top-level prose line becomes ``- line``. PROJECT_STATE: every such
    line is its own entry (118 §3.1.1); DECISIONS / LESSONS: the lines that
    follow it without a blank are its continuation.
    """
    out: list[str] = []
    in_entry = False     # the previous emitted line belongs to an entry
    durable = key != "state"
    for ln in lines:
        if _is_blank(ln):
            out.append(ln)
            in_entry = False
            continue
        if (
            state_blocks.heading_level(ln) or _is_pointer(ln) or _MERGED_RE.match(ln)
            or _FORMAT_LINE_RE.match(ln)
        ):
            out.append(ln)
            in_entry = False
            continue
        if _is_comment_only(ln):
            out.append(ln)
            continue
        if _is_indented(ln):
            out.append(ln)
            continue
        m = _LIST_RE.match(ln)
        if m:
            prefix, text = m.group("prefix"), m.group("text")
            comment = None
            cm = _COMMENT_RE.search(text)
            if cm:
                comment = cm.group(0)
                text = _strip_inline_comment(text)
            new_line = ln
            if durable:
                plain_numbered = bool(_NUMBERED_RE.match(prefix)) and not text.startswith("**")
                if plain_numbered and in_entry:
                    # A numbered step under an entry (§1.3): detail, not an entry.
                    out.append("  " + _strip_inline_comment(ln))
                    if comment:
                        warnings.append("dropped a mem-comment inside the text of " + _quote(ln, 80))
                    continue
                new_text = _strip_bold_lead(text)
                new_line = f"- {new_text}"
            elif comment:
                new_line = f"{prefix}{text}".rstrip()
            if new_line != ln.rstrip():
                warnings.append(f"rewrote {_quote(ln, 80)} as {_quote(new_line)}")
            out.append(new_line)
            if comment:
                out.append(state_blocks.COMMENT_INDENT + comment)
            in_entry = True
            continue
        # A top-level prose line.
        if durable and in_entry:
            if "<!--" in ln and _COMMENT_RE.search(ln):
                warnings.append("dropped a mem-comment inside the text of " + _quote(ln, 80))
                ln = _strip_inline_comment(ln)
            out.append("  " + ln)
            continue
        text = _strip_inline_comment(ln) if _COMMENT_RE.search(ln) else ln
        if text != ln:
            warnings.append("dropped a mem-comment inside the text of " + _quote(text, 80))
        if durable:
            text = _strip_bold_lead(text)
        new_line = f"- {text}"
        warnings.append(f"rewrote the paragraph {_quote(ln, 80)} as a bullet: {_quote(new_line)}")
        out.append(new_line)
        in_entry = True
    return out


def _normalise_comments(
    lines: list[str], warnings: list[str], dropped_ids: list[str],
) -> list[str]:
    """Rule N4: machine metadata is allowed only on the line under a bullet.
    A comment anywhere else (under a pointer, under a heading, stacked under
    another comment, inside detail) is dropped; an inline comment inside a
    text line is stripped from it."""
    out: list[str] = []
    for ln in lines:
        if _is_comment_only(ln):
            prev = out[-1] if out else ""
            if _LIST_RE.match(prev):
                out.append(ln)
                continue
            eid = _comment_fields(ln).get("id")
            if eid:
                dropped_ids.append(eid)
            warnings.append(
                "dropped a mem-comment that was not under its bullet " + _quote(ln.strip(), 80)
            )
            continue
        if "<!--" in ln and _COMMENT_RE.search(ln) and not _LIST_RE.match(ln):
            if state_blocks._opens_comment(ln):
                out.append(ln)
                continue
            stripped = _strip_inline_comment(ln)
            warnings.append("dropped a mem-comment inside the text of " + _quote(stripped, 80))
            out.append(stripped)
            continue
        out.append(ln)
    return out


def _normalise_pointers(
    lines: list[str], pointers: list[PointerLine], warnings: list[str],
) -> list[str]:
    """Rule N5: legacy pointer lines leave the live file (the caller folds
    the ones whose ids are not in the archive into it, archive first)."""
    out: list[str] = []
    for ln in lines:
        if _is_pointer(ln):
            pointers.append(_parse_pointer(ln))
            continue
        out.append(ln)
    if pointers:
        warnings.append(
            f"folded {len(pointers)} legacy [archived …] pointer line(s) into the archive"
        )
    return out


def _collapse_blank_runs(lines: list[str]) -> list[str]:
    out: list[str] = []
    for ln in lines:
        if _is_blank(ln) and out and _is_blank(out[-1]):
            continue
        out.append(ln)
    return out


def reattach_hidden_lines(text: str | None) -> str | None:
    """Spec 192: put a stray hidden line back under its bullet. Pure.

    An agent that adds a detail line right under a DECISIONS / LESSONS
    bullet, above its ``<!--mem …-->`` line, leaves a shape rule N4 does not
    read as the same entry: ``normalise`` drops the hidden line, ``stamp``
    gives the bullet a fresh id, and ``conserve_ids`` puts the old block
    back, so every later write adds another copy. Run before ``normalise``
    on both the new and the previous text.

    In each block (as ``state_blocks.parse`` sees it) holding exactly one
    mem-comment, alone on a line that is not the one directly under the
    bullet, that line moves up to sit directly under the bullet. Every other
    byte stays as it was; a block with no hidden line, several, or one that
    spans lines or shares its line with text is left alone. Idempotent.
    Not for PROJECT_STATE (its lines reconcile by their words).
    """
    if not text or "<!--" not in text:
        return text
    lines, blocks = state_blocks.parse(text)
    changed = False
    for b in blocks:
        block = b.lines
        if len(_COMMENT_RE.findall("\n".join(block))) != 1:
            continue
        alone = [k for k, ln in enumerate(block)
                 if ln.strip() and _COMMENT_RE.fullmatch(ln.strip())]
        if len(alone) != 1 or alone[0] <= 1:
            continue
        k = alone[0]
        # Same number of lines: the other blocks' places do not move.
        lines[b.start:b.end + 1] = [block[0], block[k], *block[1:k], *block[k + 1:]]
        changed = True
    return "\n".join(lines) if changed else text


def normalise(text: str | None, key: str) -> NormaliseResult:
    """Bring ``text`` into the one entry shape (spec 124 §3.2). Pure.

    Lossless: every non-marker line keeps its content (only list markers,
    leading numbers, a leading bold title span and indentation change);
    idempotent; byte-identical for a file that already conforms. Legacy
    pointer lines are returned in ``pointers`` for the caller to fold.
    """
    content = text or ""
    if key not in ENTRY_KEYS or not content.strip():
        return NormaliseResult(content)
    warnings: list[str] = []
    lines = content.split("\n")
    header: list[str] = []
    if lines and _FORMAT_LINE_RE.match(lines[0]):
        header = [lines[0]]
        lines = lines[1:]
    if not _has_structure(lines):
        has_prose = any(
            not _is_blank(ln) and state_blocks.heading_level(ln) == 0
            and not _is_comment_only(ln) and not _is_pointer(ln)
            for ln in lines
        )
        if has_prose:
            return NormaliseResult(content, [
                f"{MEMORY_FILENAME[key]} was written as prose and stored as-is. "
                "Expected shape: one `- ` bullet line per entry that stands on "
                "its own, optional detail indented under it, `## ` headings "
                "to group by topic."
            ])
        return NormaliseResult(content)
    pointers: list[PointerLine] = []
    dropped_ids: list[str] = []
    lines = _normalise_scaffolding(lines, warnings)
    if key != "state":
        lines = _normalise_heading_entries(lines, warnings)
    lines = _normalise_items_and_prose(lines, key, warnings)
    lines = _normalise_comments(lines, warnings, dropped_ids)
    lines = _normalise_pointers(lines, pointers, warnings)
    lines = _collapse_blank_runs(lines)
    out = "\n".join(header + lines)
    if out == content:
        warnings = []
    return NormaliseResult(out, warnings, pointers, dropped_ids)


def normalise_entries(text: str | None, key: str) -> tuple[str, list[str]]:
    """``(normalised_text, warnings)`` — the spec's signature for the pure
    pass. Use ``normalise`` when the folded pointers are needed."""
    r = normalise(text, key)
    return r.text, r.warnings


# ---------------------------------------------------------------------------
# Ids (spec 124 §3.3): unique across live file + archive, never rewritten
# ---------------------------------------------------------------------------

def _bullet_key(block: state_blocks.Block) -> str:
    """Identity of an entry across a comment-stripped rewrite: its bullet line
    (bold and whitespace normalised) — detail may be edited under it."""
    first = block.content_lines[0][len(block.prefix):] if block.content_lines else ""
    return state_blocks.match_key(first.replace("**", ""))


def _entry_title(block: state_blocks.Block) -> str:
    first = block.content_lines[0][len(block.prefix):] if block.content_lines else ""
    return _strip_bold_lead(" ".join(first.split()))


def new_entry_id(title: str, taken: set[str]) -> str:
    """A content-derived id for a durable entry.

    ASCII titles slug to words (``keep-api-base-url``); a title with
    non-ASCII words gets ``<ascii slug>-<short hash>`` or just the hash when
    there is no ASCII at all — never a bare ``entry``. Uniqueness is against
    ``taken`` (the live file's and the archive's ids).
    """
    norm = _LEAD_DATE_RE.sub("", " ".join((title or "").replace("**", "").split()))
    norm = norm.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", norm).strip("-")[:32].strip("-")
    non_ascii = any(c.isalnum() and ord(c) > 127 for c in norm)
    if non_ascii or not slug:
        digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:6]
        base = f"{slug}-{digest}" if slug else digest
    else:
        base = slug
    eid = base
    n = 2
    while eid in taken:
        eid = f"{base}-{n}"
        n += 1
    return eid


def archive_ids(archive_text: str | None) -> set[str]:
    """Every ``id:`` an archive carries in its mem-comments."""
    out: set[str] = set()
    if not archive_text:
        return out
    for m in _COMMENT_RE.finditer(archive_text):
        eid = user_flags._parse_comment_fields(m.group("body")).get("id")
        if eid:
            out.add(eid)
    return out


def stamp(
    new_text: str, old_text: str | None, kind: str, *,
    today: str | None = None, archive_ids: set[str] | None = None,
) -> tuple[str, list[str]]:
    """Stamp ids + dates onto every DECISIONS / LESSONS entry (PROJECT_STATE
    goes through ``flag_chokepoint.reconcile_flags``, same shape).

    Pure: ``old_text`` is the previous on-disk content. An entry keeps its
    id when its comment is still attached, or — the comment-stripped
    rewrite — when its bullet line is the same text as before; ``touched``
    advances when the entry's text changed. A new entry gets a
    content-derived id unique across the live file and ``archive_ids``.
    Existing ids are never rewritten; a duplicate within the file gets a
    fresh one.
    """
    today = today or _today()
    if kind not in DURABLE_KEYS or not new_text:
        return new_text, []
    warnings: list[str] = []
    # A format-only rewrite (the normalisation pass) is not an edit: compare
    # against the old text in the same shape, bold and whitespace ignored.
    old_blocks = state_blocks.parse(normalise(old_text, kind).text)[1] if old_text else []
    old_by_id = {b.id: b for b in old_blocks if b.id}
    old_by_key: dict[str, list[state_blocks.Block]] = {}
    for b in old_blocks:
        old_by_key.setdefault(_bullet_key(b), []).append(b)

    lines, blocks = state_blocks.parse(new_text)
    taken: set[str] = set(archive_ids or ()) | set(old_by_id)
    taken.update(b.id for b in blocks if b.id)
    used: set[str] = set()
    claimed: set[int] = set()
    rendered: dict[int, list[str]] = {}

    def _content(b: state_blocks.Block) -> str:
        return " ".join(" ".join(b.content_lines).replace("**", "").split())

    for b in blocks:
        fields: dict[str, str] | None = None
        own = old_by_id.get(b.id) if b.id else None
        if b.id and b.id in used:
            warnings.append(f"entry id:{b.id} appeared twice; the second got a fresh id")
            own = None
            b.fields.pop("id", None)
        if own is not None and id(own) not in claimed:
            fields = {**own.fields, **b.fields}
            claimed.add(id(own))
            if _content(b) != _content(own):
                fields["touched"] = today
        elif b.id:
            fields = dict(b.fields)          # an id the old file never had: keep it
        else:
            for cand in old_by_key.get(_bullet_key(b), ()):
                if id(cand) in claimed or not cand.id or cand.id in used:
                    continue
                fields = dict(cand.fields)
                claimed.add(id(cand))
                if _content(b) != _content(cand):
                    fields["touched"] = today
                break
        if fields is None:
            fields = {"id": new_entry_id(_entry_title(b), taken | used)}
        fields.setdefault("created", today)
        fields.setdefault("touched", fields["created"])
        used.add(fields["id"])
        rendered[b.start] = state_blocks.render(b, fields)

    out: list[str] = []
    by_start = {b.start: b for b in blocks}
    i = 0
    while i < len(lines):
        b = by_start.get(i)
        if b is None:
            out.append(lines[i])
            i += 1
            continue
        out.extend(rendered[b.start])
        i = b.end + 1
    return "\n".join(out), warnings


def entry_count(content: str | None, key: str) -> int:
    if not content or key not in ENTRY_KEYS:
        return 0
    return len(state_blocks.parse(content)[1])


# ---------------------------------------------------------------------------
# Injected view (spec 124 §3.4): all bullet lines, hot detail, fold by section
# ---------------------------------------------------------------------------

_TRIM_NOTE = (
    "\n[... older content trimmed from this view — read the file on disk "
    "for the full text ...]"
)


def _head_within(content: str, budget_tokens: int) -> str:
    """Keep head lines under the token budget (INDEX, and the last resort).

    Returns the SAME object when it already fits (prefix-cache friendly).
    """
    char_budget = budget_tokens * 4
    if len(content) <= char_budget:
        return content
    body_budget = max(1, char_budget - len(_TRIM_NOTE))
    kept: list[str] = []
    used = 0
    for line in content.splitlines():
        if used + len(line) + 1 > body_budget:
            break
        kept.append(line)
        used += len(line) + 1
    return "\n".join(kept) + _TRIM_NOTE


def _fits_stripped(text: str, hard_budget: int) -> bool:
    """True if ``text`` fits ``hard_budget`` once mem-comments are excluded."""
    return len(user_flags.strip_mem_comments(text)) <= hard_budget * 4


# Lines naming an archive file are the ONLY route from a Layer-1 file to its
# Layer-2 archive. Pin them in every INDEX trim.
_ARCHIVE_POINTER = re.compile(r"_ARCHIVE\.md")


def _drop_tail_first(
    content: str, hard_budget: int, protected: set[int]
) -> tuple[str, bool]:
    lines = content.split("\n")
    n = len(lines)
    keep = [True] * n

    def _cur_fits() -> bool:
        return _fits_stripped(
            "\n".join(lines[j] for j in range(n) if keep[j]), hard_budget
        )

    i = n - 1
    while i >= 0 and not _cur_fits():
        if i not in protected:
            keep[i] = False
        i -= 1

    return "\n".join(lines[j] for j in range(n) if keep[j]), _cur_fits()


def _recency_key(b: state_blocks.Block, pos: int) -> tuple[str, str, int]:
    """Newest first: by ``touched``, then ``created``, then file position.
    An unstamped block (written outside the tools) counts as newest."""
    touched = b.fields.get("touched") or "9999-99-99"
    created = b.fields.get("created") or "9999-99-99"
    return (touched, created, pos)


def _is_pinned(b: state_blocks.Block) -> bool:
    return b.fields.get("tag") == "pinned"


def inject_view(content: str | None, key: str, hard_budget: int) -> str | None:
    """The injected view of a Layer-1 file (spec 124 §3.4).

    Entry files (PROJECT_STATE / DECISIONS / LESSONS), one algorithm:

    1. every bullet line is always injected (mem-comments stripped);
    2. indented detail is injected only for the most recently ``touched``
       entries that fit the budget, and always for ``tag:pinned`` ones;
    3. when the bullet lines alone overflow, the file folds by ``## ``
       section: each section keeps its newest entries (at least one) plus
       a count line pointing at the file; a file without sections keeps its
       newest entries plus one count line.

    Output keeps file order and is a pure function of the content, so it is
    byte-stable across calls (spec 112 reuses it within a run). INDEX is
    head-trimmed. Returns the SAME object when nothing had to change.
    """
    if not content:
        return content
    if key not in ENTRY_KEYS:
        return _head_within(content, hard_budget)

    char_budget = hard_budget * 4
    lines, blocks = state_blocks.parse(content)
    by_start = {b.start: b for b in blocks}
    in_block: dict[int, state_blocks.Block] = {}
    for b in blocks:
        for i in range(b.start, b.end + 1):
            in_block[i] = b

    # --- level 1 + 2 ------------------------------------------------------
    loose: dict[int, str] = {}            # non-block lines, comments stripped
    for i, ln in enumerate(lines):
        if i in in_block:
            continue
        if _is_comment_only(ln):
            continue
        loose[i] = _strip_inline_comment(ln) if "<!--" in ln else ln

    def _render(hot: set[int]) -> str:
        out: list[str] = []
        i = 0
        while i < len(lines):
            b = by_start.get(i)
            if b is not None:
                if b.content_lines:
                    out.append(b.content_lines[0])
                    if b.start in hot:
                        out.extend(b.content_lines[1:])
                i = b.end + 1
                continue
            if i in loose:
                out.append(loose[i])
            i += 1
        return "\n".join(out)

    all_hot = {b.start for b in blocks}
    full = _render(all_hot)
    if len(full) <= char_budget:
        return content if full == content else full

    pinned = {b.start for b in blocks if _is_pinned(b)}
    base = _render(pinned)
    if len(base) <= char_budget:
        ranked = sorted(
            (b for b in blocks if b.start not in pinned and len(b.content_lines) > 1),
            key=lambda b: _recency_key(b, b.start), reverse=True,
        )
        hot = set(pinned)
        used = len(base)
        for b in ranked:
            extra = sum(len(ln) + 1 for ln in b.content_lines[1:])
            if used + extra > char_budget:
                continue
            hot.add(b.start)
            used += extra
        return _render(hot)

    # --- level 3: fold by section -----------------------------------------
    return _fold_view(lines, blocks, loose, key, char_budget)


def _fold_view(
    lines: list[str], blocks: list[state_blocks.Block], loose: dict[int, str],
    key: str, char_budget: int,
) -> str:
    filename = f"orbital/{MEMORY_FILENAME[key]}"
    # Sections: (heading line index or None, [blocks]) in file order; the
    # preamble (lines before the first heading / block) is kept verbatim.
    first_heading = next(
        (i for i, ln in enumerate(lines) if state_blocks.heading_level(ln) >= 2), None
    )
    sections: list[tuple[int | None, list[state_blocks.Block]]] = [(None, [])]
    heading_at: dict[int, int] = {}
    for i, ln in enumerate(lines):
        if state_blocks.heading_level(ln) >= 2:
            heading_at[i] = len(sections)
            sections.append((i, []))
    for b in blocks:
        idx = 0
        for h, s_idx in heading_at.items():
            if h < b.start:
                idx = s_idx
        sections[idx][1].append(b)
    first_block = blocks[0].start if blocks else len(lines)
    preamble_end = min(first_heading if first_heading is not None else len(lines), first_block)
    preamble = [loose[i] for i in range(preamble_end) if i in loose]

    shown: set[int] = {b.start for b in blocks if _is_pinned(b)}
    for _h, members in sections:
        newest = sorted(
            (b for b in members if b.start not in shown),
            key=lambda b: _recency_key(b, b.start), reverse=True,
        )
        if newest and not any(b.start in shown for b in members):
            shown.add(newest[0].start)

    def _render(show: set[int]) -> str:
        out = list(preamble)
        for h, members in sections:
            if h is not None:
                out.append(loose.get(h, lines[h]))
            hidden = [b for b in members if b.start not in show]
            for b in members:
                if b.start in show and b.content_lines:
                    out.append(b.content_lines[0])
                    if _is_pinned(b):
                        out.extend(b.content_lines[1:])
            if hidden:
                n = len(hidden)
                where = "in this section" if h is not None else "above"
                out.append(
                    f"[{n} more entr{'y' if n == 1 else 'ies'} {where} not shown — "
                    f"read {filename} or grep it]"
                )
        return "\n".join(out)

    text = _render(shown)
    used = len(text)
    ranked = sorted(
        (b for b in blocks if b.start not in shown),
        key=lambda b: _recency_key(b, b.start), reverse=True,
    )
    for b in ranked:
        cost = len(b.content_lines[0]) + 1 if b.content_lines else 0
        if used + cost > char_budget:
            continue
        shown.add(b.start)
        used += cost
    text = _render(shown)
    if len(text) > char_budget:
        return _head_within(text, char_budget // 4)
    return text


# ---------------------------------------------------------------------------
# Soft-cap flag (WU3) — dynamic slot only.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RefreshView:
    """Tidy-pass scheduler state snapshot threaded into ``soft_flag``.

    The flag renders as a state machine driven by this view, not a repeating
    alarm. ``last_outcome`` uses run_session_end_routine's vocabulary
    (``EDITOR_OK_OUTCOMES`` / ``EDITOR_FAILED_OUTCOMES``), or None.
    """
    in_flight: bool = False
    in_flight_since_turn: int | None = None
    last_outcome: str | None = None
    last_turn: int | None = None


# Outcomes of a pass whose editor ran (the two legacy names are what a pass
# from before the editor reported; a session can carry one across an upgrade).
EDITOR_OK_OUTCOMES = ("edited", "no_change", "llm_merged", "llm_merged_archived")
# A pass whose editor could not run: only the deterministic floor did.
EDITOR_FAILED_OUTCOMES = ("backstop_only", "failed")

# Escalation band: within this fraction of the hard cap, the flag warns that
# the injected view will fold.
_HARD_CAP_WARN_FRACTION = 0.9


def over_soft_budget(content: str | None, key: str) -> bool:
    """True when ``key``'s content is over its soft budget — the tidy trigger,
    measured exactly the way the hygiene flag measures it (bullet lines)."""
    budgets = FILE_BUDGETS.get(key)
    if not content or not budgets:
        return False
    return est_tokens(_budget_text(content, key)) > budgets["soft"]


def over_hard_budget(content: str | None, key: str) -> bool:
    """True when the full text (header-stripped, comments stripped) is past
    the hard cap — the injected view folds then."""
    budgets = FILE_BUDGETS.get(key)
    if not content or not budgets:
        return False
    return est_tokens(user_flags.strip_mem_comments(strip_format_header(content))) > budgets["hard"]


def soft_flag(
    content: str | None, key: str, refresh: RefreshView | None = None
) -> str | None:
    """A persistent note while a file is over its soft threshold, or None.

    The caller MUST place this in the dynamic/uncached slot (never the
    cached prefix). Over-budget files are tidied automatically (a Memory
    pass archives what is finished or superseded; nothing is lost), so no
    state of this flag asks the agent to do anything but keep its lines
    true.
    """
    budgets = FILE_BUDGETS.get(key)
    if not content or not budgets:
        return None
    toks = est_tokens(_budget_text(content, key))
    soft = budgets["soft"]
    hard = budgets["hard"]
    if toks <= soft:
        return None
    suffix = f" ({entry_count(content, key)} entries)" if key in ENTRY_KEYS else ""
    head = f"{DISPLAY_NAME[key]} memory {toks/1000:.1f}k/{soft//1000}k tok{suffix}"

    r = refresh or RefreshView()
    if r.in_flight:
        since = (
            f" (started turn {r.in_flight_since_turn})"
            if r.in_flight_since_turn is not None else ""
        )
        body = (
            f"{head} — a memory pass is tidying it in the background{since}; "
            "no action needed. Do not trim it by hand; keep recording new facts "
            "as usual — the flag may persist until the pass lands."
        )
    elif r.last_outcome in EDITOR_FAILED_OUTCOMES:
        at = f" (turn {r.last_turn})" if r.last_turn is not None else ""
        body = (
            f"{head} — the last automatic tidy{at} could not run its editor "
            "(only the size floor ran); it retries on its own once the file "
            "changes again. No action needed beyond keeping lines true: "
            "overwrite any line that no longer is."
        )
    elif r.last_outcome in EDITOR_OK_OUTCOMES:
        at = f" at turn {r.last_turn}" if r.last_turn is not None else ""
        body = (
            f"{head} — tidied{at}; what remains is recent or still live. No "
            "action needed; overwrite any line that is no longer true."
        )
    else:
        body = (
            f"{head} — over its soft budget; a memory pass tidies it "
            "automatically in the background (finished or superseded entries "
            "move to the archive; nothing is lost). No action needed."
        )

    if toks >= hard * _HARD_CAP_WARN_FRACTION:
        body += (
            f" ⚠ {max(0, int(hard - toks))} tok from the hard cap ({hard}): past "
            "it, only the newest entries of each section are injected until a "
            "pass archives what is finished."
        )
    return body


# ---------------------------------------------------------------------------
# INDEX size floor (unchanged by spec 124: no entries, tail trim)
# ---------------------------------------------------------------------------

def trim_volatile(content: str, hard_budget: int) -> str:
    """INDEX.md's size floor: trim tail lines until the file fits.

    INDEX is regenerable navigation with no archive, so this is a plain
    head-keep / tail-drop — except for lines naming an ``*_ARCHIVE.md`` file,
    which are the only route to an archive and are never dropped. Fitting is
    measured comment-stripped. When the protected lines alone exceed the
    budget the content is returned unchanged.
    """
    if not content:
        return content

    if _fits_stripped(content, hard_budget):
        return content

    protected = {
        i for i, line in enumerate(content.split("\n"))
        if _ARCHIVE_POINTER.search(line) and not _FORMAT_LINE_RE.match(line)
    }
    if not protected:
        return _head_within(content, hard_budget)

    fit_budget = max(1, int(hard_budget - est_tokens(_TRIM_NOTE)))
    trimmed, fits = _drop_tail_first(content, fit_budget, protected)
    if not fits:
        return content

    return trimmed + _TRIM_NOTE


# ---------------------------------------------------------------------------
# Archive contents table + INDEX archive lines (spec 124 §3.6)
# ---------------------------------------------------------------------------

CONTENTS_OPEN = "<!--contents generated by Orbital; do not edit-->"
CONTENTS_CLOSE = "<!--/contents-->"
CONTENTS_HEADING = "## Contents"
_SECTION_RE = re.compile(
    r"^## \[(?P<kind>archived|trimmed|merged|replaced|pointers folded) (?P<date>\d{4}-\d{2}-\d{2})"
    r"(?: (?:into|by) id:(?P<into>\S+?))?\]\s*$"
)
_ROW_TITLE_CHARS = 80
ARCHIVE_TITLE_SUFFIX = " (archived entries, read-on-demand)"


@dataclass
class ArchiveRow:
    id: str
    kind: str            # archived | trimmed | merged | replaced | folded | ""
    date: str            # "" when the entry sits outside a dated section
    into: str
    title: str


def _row_title(text: str) -> str:
    t = " ".join(text.replace("**", "").split())
    m = _LIST_RE.match(t)
    if m:
        t = m.group("text")
    t = re.sub(r"^#{1,6}\s+", "", t)
    if len(t) > _ROW_TITLE_CHARS:
        t = t[: _ROW_TITLE_CHARS - 1].rstrip() + "…"
    return t


def _archive_body(archive_text: str) -> list[str]:
    """The archive's lines with the generated contents block removed."""
    lines = archive_text.split("\n")
    try:
        a = lines.index(CONTENTS_OPEN)
        b = lines.index(CONTENTS_CLOSE, a)
    except ValueError:
        return lines
    rest = lines[:a] + lines[b + 1:]
    return rest


def archive_rows(archive_text: str | None) -> list[ArchiveRow]:
    """One row per mem-id in the archive body, newest section first."""
    if not archive_text:
        return []
    lines = _archive_body(archive_text)
    rows: list[tuple[int, ArchiveRow]] = []
    kind, day, into = "", "", ""
    section_no = 0
    prev_text = ""
    for ln in lines:
        m = _SECTION_RE.match(ln)
        if m:
            kind = {"pointers folded": "folded"}.get(m.group("kind"), m.group("kind"))
            day, into = m.group("date"), m.group("into") or ""
            section_no += 1
            prev_text = ""
            continue
        for cm in _COMMENT_RE.finditer(ln):
            eid = user_flags._parse_comment_fields(cm.group("body")).get("id")
            if not eid:
                continue
            own = _strip_inline_comment(ln).strip()
            title = _row_title(own if own else prev_text)
            rows.append((section_no, ArchiveRow(eid, kind, day, into, title)))
        if ln.strip() and not _is_comment_only(ln):
            prev_text = _strip_inline_comment(ln)
    rows.sort(key=lambda r: (r[1].date, r[0]), reverse=True)
    return [r for _n, r in rows]


def contents_table(archive_text: str | None) -> str:
    """The generated block that sits at the top of an archive."""
    out = [CONTENTS_OPEN, CONTENTS_HEADING]
    rows = archive_rows(archive_text)
    if not rows:
        out.append("- (no entries carry an id yet — grep the file by topic)")
    for r in rows:
        when = f"{r.kind} {r.date}" if r.date else (r.kind or "")
        if r.kind in ("merged", "replaced") and r.into:
            when += f" → id:{r.into}"
        parts = [f"id:{r.id}"]
        if when:
            parts.append(when)
        parts.append(r.title or "(no text)")
        out.append("- " + " · ".join(parts))
    out.append(CONTENTS_CLOSE)
    return "\n".join(out)


def with_contents_table(archive_text: str | None) -> str:
    """``archive_text`` with its contents table regenerated (idempotent).

    The block is replaced in place when present, else inserted under the
    ``# `` title line (or at the very top).
    """
    text = archive_text or ""
    block = contents_table(text)
    lines = text.split("\n")
    try:
        a = lines.index(CONTENTS_OPEN)
        b = lines.index(CONTENTS_CLOSE, a)
        new_lines = lines[:a] + block.split("\n") + lines[b + 1:]
        return "\n".join(new_lines)
    except ValueError:
        pass
    if lines and lines[0].startswith("# "):
        head, rest = lines[:1], lines[1:]
        if rest and rest[0].strip():
            rest = [""] + rest
        return "\n".join(head + [""] + block.split("\n") + rest)
    return block + "\n\n" + text if text.strip() else block + "\n"


def archive_summary(archive_text: str | None) -> tuple[int, str, str]:
    """``(count of ids, earliest date, latest date)`` for the INDEX line."""
    rows = archive_rows(archive_text)
    dates = sorted(r.date for r in rows if r.date)
    return len(rows), (dates[0] if dates else ""), (dates[-1] if dates else "")


def append_archive_section(
    existing: str | None, archive_filename: str, heading: str, text: str,
) -> str:
    """``existing`` + a new ``heading`` section, contents table regenerated.
    Every archive write goes through this or ``with_contents_table``."""
    section = f"{heading}\n\n{text.strip(chr(10))}\n"
    body = existing or ""
    if body.strip():
        new_archive = body.rstrip() + "\n\n" + section
    else:
        new_archive = f"# {archive_filename}{ARCHIVE_TITLE_SUFFIX}\n\n" + section
    return with_contents_table(new_archive)


INDEX_SUBJECT_PLACEHOLDER = "to be described"
_INDEX_LINE_TAIL = "Contents table at the top; grep id:<id>."
_INDEX_ARCHIVE_LINE_RE = re.compile(
    r"^(?P<lead>\s*-\s+)`?(?:orbital/)?(?P<name>[A-Z_]+_ARCHIVE\.md)`?\s+[—-]+\s+(?P<rest>.*)$"
)
_INDEX_STATS_RE = re.compile(
    r"^(?P<n>\d+) archived entr(?:y|ies)(?:, (?P<a>\d{4}-\d{2}-\d{2}) to (?P<b>\d{4}-\d{2}-\d{2}))?: "
    r"(?P<subject>.*?)\s*(?:" + re.escape(_INDEX_LINE_TAIL) + r")?\s*$"
)
# The daemon's own pre-124 boilerplate ("older entries demoted from the live
# file / from DECISIONS.md (read on demand)") is not a subject; an
# agent-written description is kept.
_LEGACY_INDEX_DESCRIPTION_RE = re.compile(
    r"^(older entries demoted from (the live file|[a-z_]+\.md)|archived entries)"
    r"( \(read on demand\))?$"
)


def index_archive_line(archive_filename: str, archive_text: str | None,
                       existing_line: str | None = None) -> str:
    """The INDEX line for an archive: count + date range filled by the daemon,
    subject kept from ``existing_line`` (the Memory session writes it) or the
    placeholder until then."""
    n, a, b = archive_summary(archive_text)
    subject = INDEX_SUBJECT_PLACEHOLDER
    if existing_line:
        m = _INDEX_ARCHIVE_LINE_RE.match(existing_line)
        rest = (m.group("rest") if m else "").strip()
        sm = _INDEX_STATS_RE.match(rest)
        if sm:
            subject = sm.group("subject").strip() or INDEX_SUBJECT_PLACEHOLDER
        elif rest:
            plain = rest.rstrip(".。 ").lower()
            if not _LEGACY_INDEX_DESCRIPTION_RE.match(plain):
                subject = rest.rstrip(".。 ")
    subject = subject.rstrip(".。 ")
    span = f", {a} to {b}" if a and b else ""
    return (
        f"- orbital/{archive_filename} — {n} archived entr{'y' if n == 1 else 'ies'}"
        f"{span}: {subject}. {_INDEX_LINE_TAIL}"
    )


def refresh_index_archive_lines(index_text: str | None, archives: dict[str, str]) -> str:
    """INDEX with one daemon-maintained line per archive in ``archives``
    (``{filename: archive text}``): existing lines are refreshed in place,
    missing ones appended. Idempotent."""
    text = index_text or ""
    lines = text.split("\n")
    for filename, archive_text in archives.items():
        found = None
        for i, ln in enumerate(lines):
            m = _INDEX_ARCHIVE_LINE_RE.match(ln)
            if m and m.group("name") == filename:
                found = i
                break
        new_line = index_archive_line(filename, archive_text, lines[found] if found is not None else None)
        if found is not None:
            lines[found] = new_line
        else:
            while lines and not lines[-1].strip():
                lines.pop()
            if not lines:
                lines = ["# INDEX", ""]
            lines.extend([new_line, ""])
    out = "\n".join(lines)
    if text.endswith("\n") and not out.endswith("\n"):
        out += "\n"
    return out


# ---------------------------------------------------------------------------
# Id conservation (spec 124 §3.5 / §3.10): absence never means removal
# ---------------------------------------------------------------------------

def _insert_blocks(
    cur_lines: list[str], cur_blocks: list[state_blocks.Block],
    ref_blocks: list[state_blocks.Block], to_restore: list[state_blocks.Block],
) -> list[str]:
    """Put each block of ``to_restore`` (from the reference text) back into
    ``cur_lines``: after the nearest preceding reference block that still
    exists, else before the nearest following one, else at the end."""
    cur_by_id = {b.id: b for b in cur_blocks if b.id}
    ref_ids = [b.id for b in ref_blocks]
    inserts: list[tuple[int, list[str]]] = []
    for b in to_restore:
        pos = ref_ids.index(b.id)
        at: int | None = None
        for prev in reversed(ref_blocks[:pos]):
            if prev.id in cur_by_id:
                at = cur_by_id[prev.id].end + 1
                break
        if at is None:
            for nxt in ref_blocks[pos + 1:]:
                if nxt.id in cur_by_id:
                    at = cur_by_id[nxt.id].start
                    break
        if at is None:
            at = len(cur_lines)
            while at > 0 and not cur_lines[at - 1].strip():
                at -= 1
        inserts.append((at, list(b.lines)))
    for at, block_lines in sorted(inserts, key=lambda p: p[0], reverse=True):
        cur_lines[at:at] = block_lines
    return cur_lines


def restore_missing(
    reference: str | None, current: str | None, archive: set[str],
) -> tuple[str, list[str]]:
    """Strict conservation (after a pass): every id that ``reference`` had
    and ``current`` lacks must be in ``archive``, else its block comes back
    verbatim. Returns ``(text, restored_ids)``."""
    if not reference:
        return current or "", []
    ref_lines, ref_blocks = state_blocks.parse(reference)
    cur_lines, cur_blocks = state_blocks.parse(current or "")
    cur_ids = {b.id for b in cur_blocks if b.id}
    missing = [b for b in ref_blocks if b.id and b.id not in cur_ids and b.id not in archive]
    if not missing:
        return current or "", []
    out = _insert_blocks(cur_lines, cur_blocks, ref_blocks, missing)
    return "\n".join(out), [b.id for b in missing]


# The ids of PROJECT_STATE the running loop has seen (its run-start
# snapshot plus what its own writes produced), bound per task by the loop
# (``bind_seen_state_ids`` at every snapshot segment). The value is a mutable
# set so ``process_on_write`` — which runs in a worker thread with a COPY of
# the context — can add the ids a write just produced. None (no loop, an
# external writer, the Memory session reading from disk) falls back to the
# gap rule.
_SEEN_STATE_IDS: ContextVar[set[str] | None] = ContextVar("orbital_seen_state_ids", default=None)


def bind_seen_state_ids(ids) -> Token:
    """Bind the seen-set for the current task; returns the reset token."""
    return _SEEN_STATE_IDS.set(set(ids))


def unbind_seen_state_ids(token: Token) -> None:
    _SEEN_STATE_IDS.reset(token)


def seen_state_ids() -> set[str] | None:
    return _SEEN_STATE_IDS.get()


def conserve_ids(
    prev: str | None, new: str | None, archive: set[str], *,
    key: str = "state", seen: set[str] | None = None,
) -> tuple[str, list[str]]:
    """Write-path conservation (spec 124 §3.10): an entry never leaves a live
    file except through its archive — per file:

    - DECISIONS / LESSONS are STRICT: every id ``prev`` had and ``new`` lacks
      must be in ``archive``, else its block comes back.
    - PROJECT_STATE's contract is overwrite, so only ids the writer NEVER SAW
      are protected: with ``seen`` (the loop's snapshot plus its own writes)
      an id on disk but outside that set is restored on drop; everything the
      writer saw may be dropped. Without ``seen`` (an external writer, the
      Memory session reading from disk) the gap rule applies: a dropped id
      is left out only when it is archived or a NEW entry was written in the
      same gap of the file (a reworded line, a replaced status); a lost
      update from a stale snapshot or a bare deletion is restored.
    """
    if not prev or not new:
        return new or "", []
    if key in DURABLE_KEYS:
        return restore_missing(prev, new, archive)
    if seen is not None:
        return restore_missing(prev, new, archive | set(seen))
    prev_lines, prev_blocks = state_blocks.parse(prev)
    new_lines, new_blocks = state_blocks.parse(new)
    prev_ids = {b.id for b in prev_blocks if b.id}
    new_ids = {b.id for b in new_blocks if b.id}
    missing = [b for b in prev_blocks if b.id and b.id not in new_ids and b.id not in archive]
    if not missing:
        return new, []
    new_by_id = {b.id: b for b in new_blocks if b.id}
    fresh = [b for b in new_blocks if b.id not in prev_ids]

    def _gap(b: state_blocks.Block) -> tuple[int, int]:
        pos = prev_blocks.index(b)
        lo = -1
        for p in reversed(prev_blocks[:pos]):
            if p.id in new_by_id:
                lo = new_by_id[p.id].end
                break
        hi = len(new_lines)
        for p in prev_blocks[pos + 1:]:
            if p.id in new_by_id:
                hi = new_by_id[p.id].start
                break
        return lo, hi

    def _words(b: state_blocks.Block) -> set[str]:
        return set(_bullet_key(b).lower().split())

    # Within one gap, each fresh block replaces at most one dropped block —
    # the one whose bullet line it shares the most words with (a reworded
    # line), else the next one in order. Dropped blocks nothing replaced
    # come back.
    by_gap: dict[tuple[int, int], list[state_blocks.Block]] = {}
    for b in missing:
        by_gap.setdefault(_gap(b), []).append(b)
    to_restore: list[state_blocks.Block] = []
    for (lo, hi), dropped in by_gap.items():
        candidates = [f for f in fresh if lo < f.start < hi]
        remaining = list(dropped)
        for f in candidates:
            if not remaining:
                break
            fw = _words(f)
            best = max(remaining, key=lambda d: (len(fw & _words(d)), -remaining.index(d)))
            remaining.remove(best)
        to_restore.extend(remaining)
    to_restore.sort(key=lambda b: b.start)
    if not to_restore:
        return new, []
    out = _insert_blocks(new_lines, new_blocks, prev_blocks, to_restore)
    return "\n".join(out), [b.id for b in to_restore]


# ---------------------------------------------------------------------------
# Write-path entry point used by the write/edit tools (WU1 / spec 124 §3.10).
# ---------------------------------------------------------------------------

# basename -> file_key, for detecting a memory-file write by resolved path.
_BASENAME_TO_KEY = {v: k for k, v in MEMORY_FILENAME.items()}
_BASENAME_TO_KEY.update({v: k for k, v in ARCHIVE_FILENAME.items()})
# ASKS.md (spec 089) is an append-only event log, not an injected/budgeted
# Layer-1 file — so it is routed here without joining MEMORY_FILENAME, which
# the budget/injection/trim code iterates.
ASKS_KEY = "asks"
_BASENAME_TO_KEY["ASKS.md"] = ASKS_KEY


def memory_key_for_path(resolved_path: str, workspace: str) -> str | None:
    """Return the memory file_key if ``resolved_path`` is ``<ws>/orbital/<file>``."""
    try:
        rp = os.path.realpath(resolved_path)
        orbital = os.path.realpath(os.path.join(workspace, "orbital"))
        if os.path.dirname(rp) != orbital:
            return None
        return _BASENAME_TO_KEY.get(os.path.basename(rp))
    except OSError:
        return None


def _read(path: str) -> str | None:
    return read_text_lenient(path)


def _atomic_write(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def fold_pointers_into_archive(
    orbital_dir: str, key: str, pointers: list[PointerLine], ids_in_archive: set[str],
    *, today: str | None = None,
) -> list[PointerLine]:
    """Rule N5's archive half: pointer lines whose ids are not all in the
    archive are appended verbatim under ``## [pointers folded DATE]``
    (archive written FIRST — the caller writes the live file after).
    Returns the pointers that were appended."""
    unresolved = [p for p in pointers if not (p.ids and all(i in ids_in_archive for i in p.ids))]
    if not unresolved:
        return []
    archive_key = ARCHIVE_OF[key]
    filename = ARCHIVE_FILENAME[archive_key]
    path = os.path.join(orbital_dir, filename)
    existing = _read(path)
    new_text = append_archive_section(
        existing, filename, f"## [pointers folded {today or _today()}]",
        "\n".join(p.line for p in unresolved),
    )
    os.makedirs(orbital_dir, exist_ok=True)
    _atomic_write(path, new_text)
    return unresolved


# ---------------------------------------------------------------------------
# Spec 182: a declared replacement leaves the live file in the same write.
# A new DECISIONS / LESSONS entry carries an indented detail line quoting the
# old bullet word for word; the daemon moves that entry, intact, to the
# archive under ``## [replaced DATE by id:<new>]``. Exact match only.
# ---------------------------------------------------------------------------

_SUPERSEDES_RE = re.compile(r"^\s+supersedes\s*[:：]\s*(?P<quote>\S.*?)\s*$", re.IGNORECASE)


@dataclass
class Replacement:
    old: state_blocks.Block          # the entry that leaves (verbatim lines)
    new: state_blocks.Block          # the entry that declared it


def _quote_key(quote: str) -> str:
    """The quote compared the way ``_bullet_key`` compares a bullet line:
    a leading list marker and bold removed, whitespace collapsed."""
    text = " ".join(quote.replace("**", "").split())
    m = _LIST_RE.match(text)
    if m:
        text = m.group("text")
    return state_blocks.match_key(text)


def resolve_supersedes(
    text: str, key: str, ids_in_archive: set[str],
) -> tuple[str, list[Replacement], list[str]]:
    """Pure. ``(text without the moved entries and their resolved lines,
    replacements, notes)``. A ``supersedes:`` line acts only when its quote
    equals exactly one other live, unpinned entry whose id is not already
    archived; otherwise nothing changes and a note says why. Stateless: an
    unresolved line stays in the file and is tried again on the next write."""
    if key not in DURABLE_KEYS or not text or "supersedes" not in text.lower():
        return text, [], []
    lines, blocks = state_blocks.parse(text)
    by_key: dict[str, list[state_blocks.Block]] = {}
    for b in blocks:
        by_key.setdefault(_bullet_key(b), []).append(b)
    filename = MEMORY_FILENAME[key]
    archive_name = ARCHIVE_FILENAME[ARCHIVE_OF[key]]
    moved: set[int] = set()            # start lines of entries leaving
    drop: set[int] = set()             # resolved supersedes lines
    out: list[Replacement] = []
    notes: list[str] = []
    for n in blocks:
        if n.start in moved:
            continue
        for i in range(n.start + 1, n.end + 1):
            m = _SUPERSEDES_RE.match(lines[i])
            if not m:
                continue
            quote = m.group("quote")
            hits = by_key.get(_quote_key(quote), [])
            why = ""
            if not hits:
                why = f"matched no entry in {filename}"
            elif len(hits) > 1:
                why = f"matched {len(hits)} entries in {filename}"
            elif hits[0] is n:
                why = "names this entry itself"
            elif _is_pinned(hits[0]):
                why = "names a pinned entry, which is never moved"
            elif hits[0].id and hits[0].id in ids_in_archive:
                why = f"names id:{hits[0].id}, which is already in {archive_name}"
            elif hits[0].start in moved:
                why = "names an entry another line in this write already replaced"
            if why:
                notes.append(
                    f'supersedes: "{quote}" under "{_entry_title(n)}" {why}; nothing moved. '
                    "Quote the old bullet line word for word, or remove the line."
                )
                continue
            old = hits[0]
            moved.add(old.start)
            drop.add(i)
            out.append(Replacement(old=old, new=n))
    if not out:
        return text, [], notes
    gone: set[int] = set(drop)
    for b in blocks:
        if b.start in moved:
            gone.update(range(b.start, b.end + 1))
    kept = _collapse_blank_runs([ln for i, ln in enumerate(lines) if i not in gone])
    return "\n".join(kept), out, notes


def archive_replacements(
    orbital_dir: str, key: str, replacements: list[Replacement], *, today: str,
) -> None:
    """Append each replaced entry, verbatim, to the file's archive: one
    ``## [replaced DATE by id:<new>]`` section per replacing entry. One
    atomic write, made BEFORE the caller returns the live text."""
    if not replacements:
        return
    filename = ARCHIVE_FILENAME[ARCHIVE_OF[key]]
    path = os.path.join(orbital_dir, filename)
    archive = _read(path)
    groups: dict[str, list[state_blocks.Block]] = {}
    for r in replacements:
        groups.setdefault(r.new.id or "", []).append(r.old)
    for new_id, olds in groups.items():
        archive = append_archive_section(
            archive, filename, f"## [replaced {today} by id:{new_id}]",
            "\n".join(b.raw for b in olds),
        )
    os.makedirs(orbital_dir, exist_ok=True)
    _atomic_write(path, archive)


def process_on_write(workspace: str, resolved_path: str, content: str, *, today: str | None = None) -> tuple[str, list[str]]:
    """Normalise + stamp + guard a memory-file write (never rejects).

    Called by the ``write``/``edit`` tools after they resolve the target path
    and compute the content. Not a memory file → unchanged. Entry files get
    rule N1–N5 normalisation (warnings quote the rewritten lines so the
    agent's next exact-match edit finds them), ids, and the id-conservation
    guard; archives get their contents table regenerated; INDEX its header.
    """
    key = memory_key_for_path(resolved_path, workspace)
    if key is None:
        return content, []
    today = today or _today()
    if key == ASKS_KEY:
        from agent_os.agent import asks
        from agent_os.agent.project_paths import ProjectPaths
        return asks.process_write(
            _read(resolved_path), content, today=today,
            sessions_dir=ProjectPaths(workspace).sessions_dir,
        )
    if key in ARCHIVE_KEYS:
        return with_contents_table(content), []
    content = ensure_format_header(content, key)
    if key not in ENTRY_KEYS:
        return content, []

    orbital_dir = os.path.join(workspace, "orbital")
    prev = _read(resolved_path)
    if key in DURABLE_KEYS:
        # Spec 192: a hidden line left below a detail line goes back under
        # its bullet, in the new text and in the previous one (the disk may
        # hold the shape from a write no rule saw: an external worker's own
        # tools, a hand edit), so the entry keeps its id instead of a copy.
        content = reattach_hidden_lines(content)
        prev = reattach_hidden_lines(prev)
    archive_text = _read(os.path.join(orbital_dir, ARCHIVE_FILENAME[ARCHIVE_OF[key]]))
    ids_in_archive = archive_ids(archive_text)
    warnings: list[str] = []
    if key == "state":
        from agent_os.agent import asks
        content, ask_warnings = asks.convert_legacy_on_write(orbital_dir, content, today=today)
        warnings.extend(ask_warnings)

    res = normalise(content, key)
    if res.pointers:
        folded = fold_pointers_into_archive(orbital_dir, key, res.pointers, ids_in_archive, today=today)
        if folded:
            ids_in_archive = archive_ids(_read(os.path.join(orbital_dir, ARCHIVE_FILENAME[ARCHIVE_OF[key]])))
    if res.warnings:
        warnings.append(
            f"{MEMORY_FILENAME[key]} was normalised into the file's entry shape "
            "(re-read before the next edit): " + "; ".join(res.warnings)
        )
    content = res.text

    if key == "state":
        from agent_os.agent import asks, flag_chokepoint
        merged, w = flag_chokepoint.reconcile_flags(
            prev, content, today, asks.dropped_texts(orbital_dir, recent=False),
        )
        warnings.extend(x for x in w if "[user] flag" not in x)
    else:
        merged, w = stamp(content, prev, key, today=today, archive_ids=ids_in_archive)
        warnings.extend(w)

    seen = seen_state_ids() if key == "state" else None
    merged, restored = conserve_ids(prev, merged, ids_in_archive, key=key, seen=seen)
    if key in DURABLE_KEYS:
        # Spec 182: run after the guard, so an old entry the writer deleted
        # AND declared replaced is back in the text and moves the same way.
        merged, replaced, notes = resolve_supersedes(merged, key, ids_in_archive)
        warnings.extend(notes)
        if replaced:
            archive_replacements(orbital_dir, key, replaced, today=today)   # archive FIRST
            moved_ids = {r.old.id for r in replaced}
            restored = [r for r in restored if r not in moved_ids]
            for r in replaced:
                warnings.append(
                    f'id:{r.old.id} "{_entry_title(r.old)}" moved to '
                    f"{ARCHIVE_FILENAME[ARCHIVE_OF[key]]}, replaced by "
                    f'"{_entry_title(r.new)}".'
                )
    if seen is not None:
        # The writer has now seen what it produced (the restored entries it
        # has not: they came back without its knowledge).
        seen.update(b.id for b in state_blocks.parse(merged)[1] if b.id and b.id not in restored)
    if restored:
        logger.info(
            "memory write guard: %s dropped %d entr%s not in its archive; restored id=%s",
            MEMORY_FILENAME[key], len(restored), "y" if len(restored) == 1 else "ies",
            ",".join(restored),
        )
        warnings.append(
            f"{len(restored)} entr{'y' if len(restored) == 1 else 'ies'} that the write "
            f"dropped {'was' if len(restored) == 1 else 'were'} put back (id:"
            + ", id:".join(restored) + "): entries are never deleted from "
            f"{MEMORY_FILENAME[key]} — a memory pass archives what is finished."
        )
    return merged, warnings
