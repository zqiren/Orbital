# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Layer-1 memory: entry parsing, system-managed metadata, and injection budgets.

The spine of the bounded-memory model. One place that:

  - parses DECISIONS/LESSONS into entries and stamps system-managed metadata
    (``id`` / ``created`` / ``touched`` / ``tag``) so deduplication can run on
    recency, not on fragile entry position;
  - renumbers LESSONS contiguously (the on-disk files drift to ``1,4,5,7…`` as
    entries are deleted), keying by ``id`` rather than position;
  - computes per-file injection budgets from the *active model's* context window
    (derived from ``providers.json`` via ``ContextManager.model_context_limit`` —
    never hardcoded), floored at the measured clean-mature sizes;
  - bounds what is injected per turn (newest-within-budget plus the oldest few
    foundational entries for durable files);
  - is invoked by the ``write`` tool, the ``edit`` tool, and the session-end
    routine, so the caps can no longer be bypassed by a direct tool write (the
    old failure mode: caps lived only at session-end).

The active model (MiniMax-M3) has a 1,000,000-token window, so these bounds are
about **attention and a clean, non-contradictory project identity** for a weak
model — not context pressure or cost, both of which are negligible here.

Token accounting uses ``len(text) / 4`` to match the rest of the codebase
(``token_utils.estimate_message_tokens``, ``context.py``); the cap is enforced in
the same unit it is measured in.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import date

from agent_os.agent import user_flags

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
    """Token count a consolidation pass should bring ``key`` down to.

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

# Foundational entries cluster at the file head (oldest); always inject them even
# when newer entries overflow the budget (VERIFY-phase1-injection-cap.md Q5).
PROTECT_OLDEST = 3

DURABLE_KEYS = ("decisions", "lessons")   # entry-structured → demote to archive
VOLATILE_KEYS = ("state", "index")        # freeform → head-trim, no archive

ENTRY_MARKERS: dict[str, str] = {
    "decisions": r"^##\s+",
    "lessons": r"^\d+\.\s+",
}

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
# injected, never budgeted). PROJECT_STATE was absent here until 2026-07-27,
# which meant its overflow was DELETED rather than demoted: a real project lost
# 22 lines of briefing that way. INDEX stays absent on purpose — it is
# regenerable navigation, so a stale pointer is noise, not history.
ARCHIVE_OF: dict[str, str] = {
    "decisions": "decisions_archive",
    "lessons": "lessons_archive",
    "state": "state_archive",
}

# ---------------------------------------------------------------------------
# In-file format contracts. One-line HTML comments: invisible in rendered
# markdown, sit on line 1 so they survive the volatile head-trim, and count
# inside the file's own budget (~40 tok each). Self-healing at BOTH write
# chokepoints (WorkspaceFileManager.write for system writes, process_on_write
# for the agent's write/edit tools) — the same pattern stamp() uses to
# re-apply stripped <!--mem--> comments. The format detail deliberately lives
# HERE rather than in the system prompt: adjacent to the content when the
# agent reads/edits the file, and visible to humans opening the raw file.
# ---------------------------------------------------------------------------

FORMAT_HEADERS: dict[str, str] = {
    "index": (
        "<!--format INDEX is a navigation map ONLY: '- path — one sentence' "
        "bullets under '## <area>' headings. No dates, status, decisions, or "
        "lessons here — those live in PROJECT_STATE.md / DECISIONS.md / "
        "LESSONS.md.-->"
    ),
    "state": (
        '<!--format PROJECT_STATE is what is true NOW: current focus, in-progress work, blockers, next steps. Overwrite stale lines; never append dated history. Every line must be understandable without this session\'s context: concrete names, no unexplained shorthand, no cross-references by list number. [user] flag — one judgment per line: does this need the user (their decision, their action, or something they\'d be sorry to miss — including things they assigned to themselves)? If yes, insert [user] after the list marker of the line where the fact already lives: `- [user] <text>` or `3. [user] <text>`. Flagging marks a line, never creates one: one fact = one entry, never duplicated into another section. A dated commitment needing no decision is `[due:YYYY-MM-DD]` (shows on the calendar). Machine attributes (id, created, touched, resolved) live in a daemon-managed mem-comment on the next line — never write or edit these comments; leave them exactly where they are. Never auto-decide: spending money, sending external messages as the user, or irreversible/destructive acts are always surfaced, whatever the autonomy setting. Write timeless ("due Jul 28", never "tomorrow"). A line whose mem-comment carries resolved:<date> is settled — on consolidation rewrite it as the completed fact or drop it; never re-open or re-flag it. CLOSE THE LOOP THE SAME TURN: the moment the user answers a flagged line, decides it, or does it, remove the [user] flag from that line in this turn — rewrite the line as the settled fact (`- Chose option A.`) and leave the mem-comment alone. You are the only reader who can see both the flag and the user\'s answer; consolidation runs later, sees a truncated window, and cannot do this for you. A flagged line you leave behind after it is answered keeps nagging the user for something they already gave you. Never flag a question you asked during this session — flag the decision that is still genuinely open, written so someone who was not here can act on it.-->'
    ),
    "decisions": (
        "<!--format DECISIONS entries: '## <slug>' then Chose / Reason / "
        "Rejected. Supersede or replace old entries when a decision changes; "
        "never leave contradictions.-->"
    ),
    "lessons": (
        "<!--format LESSONS entries: numbered durable heuristics and "
        "playbooks. Add on error recovery or non-obvious workaround; keep "
        "real playbooks intact.-->"
    ),
}


def ensure_format_header(content: str | None, key: str) -> str:
    """Prepend the file's <!--format--> contract, or upgrade a stale one.

    The contract is code-owned. This used to only ever PREPEND when the header
    was missing, leaving any existing one alone — which meant every project
    already carrying a header was pinned to whatever contract shipped the day
    its file was created, and every rail added afterwards silently never
    applied. Only the Workbench migration endpoint (``force_format_header``)
    could ever refresh one, and nothing routes an active project there.

    So a divergent first-line header is now replaced with the current
    contract. Content after the header is preserved byte-for-byte; an
    already-current header is returned untouched (idempotent). An agent-edited
    variant is normalized rather than preserved — the header is the grammar
    contract we hand the agent, not agent content.
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

    Unlike ``ensure_format_header`` (self-heal only — leaves any existing
    header untouched), this REWRITES a stale/legacy header line to the current
    ``FORMAT_HEADERS[key]``. The Workbench migration endpoint uses it so a
    legacy project that already has an old PROJECT_STATE header actually
    receives the new ``[user]`` grammar rails (which ``ensure_format_header``
    would otherwise never apply, seeing a header already present). Content
    after the header is preserved byte-for-byte; a file with no header gets one
    prepended.
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
# NEVER a hygiene-flag trigger: consumed exclusively by the session-end merge
# prompt ("FORMATTING TO FIX") so formatting tidy-up rides along a pass that
# runs anyway. Imposing format must not increase checkpoint frequency.

# A date counts as drift only in PROSE — dates embedded in filename tokens
# (agent_output/2026-07-08-competitor-watch.md, ACTIVE-reframe-2026-04.md) are
# navigation, hence the negative lookahead for a continuing -word/path
# character or a file extension (`.md`). A sentence-ending period (dot NOT
# followed by a word char) still counts.
_SHAPE_DATE_RE = re.compile(r"\b\d{4}-\d{2}(-\d{2})?\b(?![-\w]|\.\w)")
_SHAPE_EMOJI_RE = re.compile(r"[🚨✅⚠]")
_INDEX_MAP_SHAPE_RE = re.compile(r"^(#{1,3} |- \S.* — )")
_STATE_CHANGELOG_RE = re.compile(r"^#{1,3} .*\b\d{4}-\d{2}\b")
_INDEX_NON_MAP_MAX_RATIO = 0.4


def shape_report(content: str | None, key: str) -> str | None:
    """One-line lint summary for a drifted volatile file, or None when clean.

    Conservative by design (few rules, counts not line-dumps) — a false
    positive here nags every consolidation pass.
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


def _budget_text(content: str, key: str) -> str:
    """Content used for budget counting/injection (spec §4.1 resolution 1).

    ``<!--mem ...-->`` comments are daemon-managed machine metadata (id,
    created/touched/resolved stamps; legacy files may still carry receipt
    attrs) on PROJECT_STATE bullets — they never
    compete with agent-visible context for budget, and are never shown to the
    agent at all. Only ``state`` carries this comment grammar; every other
    key (including DECISIONS/LESSONS, whose own ``<!--mem id:...-->`` stamp
    is a *different*, always-injected metadata convention) passes through
    unchanged so their budget/injection behavior stays byte-identical.
    Content with no mem-comments round-trips unchanged (no grammar adopted
    yet), so this is a no-op for the common case.
    """
    text = strip_format_header(content)
    if key != "state":
        return text
    return user_flags.strip_mem_comments(text)


def strip_format_header(content: str | None) -> str:
    """Drop a leading ``<!--format ...-->`` contract line, if present.

    The contract is code-owned scaffolding: we inject it, the user never wrote
    it, and the agent cannot remove it — so charging it to the file's own
    budget means every rail added to the contract silently steals space from
    real project memory. PROJECT_STATE's header reached 498 tokens, 46% of that
    file's 1080 consolidation target, leaving 582 for content and putting a
    real project permanently over budget with no route down.

    Hygiene only. Compaction fires on the provider's reported usage
    (``ContextManager.should_compact``), and context-window math runs through
    ``budgets_for_window``/``inject_view``, both of which measure the real
    injected text — header included. Nothing here can undercount those.
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
# Entry parsing / metadata
# ---------------------------------------------------------------------------

_META_RE = re.compile(r"\s*<!--\s*mem\s+(?P<body>.*?)\s*-->\s*$")


def _split_entries(content: str, marker: str) -> tuple[str, list[str]]:
    """Split content on the entry marker, preserving separators.

    Returns ``(preamble, entries)``. ``preamble`` is any leading text (file
    title) before the first marker. Each entry is ``marker + body`` including
    its trailing whitespace up to the next marker.
    """
    parts = re.split(f"({marker})", content, flags=re.MULTILINE)
    if len(parts) < 3:
        return content, []
    preamble = parts[0]
    entries: list[str] = []
    i = 1
    while i + 1 < len(parts):
        entries.append(parts[i] + parts[i + 1])
        i += 2
    return preamble, entries


def _parse_meta(first_line: str) -> tuple[str, dict[str, str]]:
    """Strip a trailing ``<!--mem ...-->`` comment off the header line.

    Returns ``(line_without_meta, meta_dict)``. Unknown/absent → ``({})``.
    """
    m = _META_RE.search(first_line)
    if not m:
        return first_line.rstrip("\n"), {}
    meta: dict[str, str] = {}
    for tok in m.group("body").split():
        if ":" in tok:
            k, _, v = tok.partition(":")
            if k:
                meta[k] = v
    clean = first_line[: m.start()].rstrip()
    return clean, meta


def _meta_comment(meta: dict[str, str]) -> str:
    fields = []
    for k in ("id", "created", "touched", "tag"):
        v = meta.get(k)
        if v:
            fields.append(f"{k}:{v}")
    return "<!--mem " + " ".join(fields) + "-->" if fields else ""


def _norm_title(first_line_clean: str, kind: str) -> str:
    """Normalized title used for dedup matching when an entry has no ``id``."""
    s = first_line_clean
    s = re.sub(ENTRY_MARKERS[kind], "", s)          # drop "## " / "N. "
    s = s.replace("**", "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    # drop a leading ISO date prefix ("2026-06-15: ") so a re-dated dup matches
    s = re.sub(r"^\d{4}-\d{2}(-\d{2})?:?\s*", "", s)
    return s


def _slug(first_line_clean: str, kind: str, seen: set[str]) -> str:
    base = _norm_title(first_line_clean, kind)
    base = re.sub(r"[^a-z0-9]+", "-", base).strip("-")[:32] or "entry"
    slug = base
    n = 2
    while slug in seen:
        slug = f"{base}-{n}"
        n += 1
    return slug


def _first_line_split(raw_entry: str) -> tuple[str, str]:
    nl = raw_entry.find("\n")
    if nl < 0:
        return raw_entry, ""
    return raw_entry[:nl], raw_entry[nl:]


def stamp(new_text: str, old_text: str | None, kind: str, *, today: str | None = None) -> tuple[str, list[str]]:
    """Stamp system-managed metadata onto a durable file's entries.

    Pure: caller supplies the previous on-disk ``old_text`` for id/created
    preservation. For matched entries (by ``id`` else normalized title) the
    ``id`` and ``created`` are preserved and ``touched`` is set to today when the
    body changed. New entries get ``created=touched=today``. LESSONS are
    renumbered contiguously. Non-durable files (state/index) pass through.

    Returns ``(stamped_text, warnings)``. If the agent stripped the metadata on
    an overwrite, it is re-applied here (so metadata survives agent rewrites).
    """
    today = today or _today()
    if kind not in ENTRY_MARKERS:
        return new_text, []

    marker = ENTRY_MARKERS[kind]
    warnings: list[str] = []

    # Index the previous file: id -> meta, normalized-title -> meta, id -> body.
    old_by_id: dict[str, dict[str, str]] = {}
    old_by_title: dict[str, dict[str, str]] = {}
    old_body_by_key: dict[str, str] = {}
    if old_text:
        _, old_raws = _split_entries(old_text, marker)
        for raw in old_raws:
            fl, body = _first_line_split(raw)
            clean, meta = _parse_meta(fl)
            key = meta.get("id") or _norm_title(clean, kind)
            if meta.get("id"):
                old_by_id[meta["id"]] = meta
            old_by_title[_norm_title(clean, kind)] = meta
            old_body_by_key[key] = body.strip()

    pre, raws = _split_entries(new_text, marker)
    if not raws:
        if new_text.strip():
            warnings.append(
                f"{kind}: could not parse into entries (expected '{marker}' "
                f"markers) — wrote as-is, metadata/caps not applied."
            )
        return new_text, warnings

    seen_ids: set[str] = set()
    out_entries: list[str] = []
    for idx, raw in enumerate(raws, start=1):
        fl, body = _first_line_split(raw)
        clean, meta = _parse_meta(fl)
        ntitle = _norm_title(clean, kind)
        prev = (meta.get("id") and old_by_id.get(meta["id"])) or old_by_title.get(ntitle)
        if prev:
            meta["id"] = prev.get("id") or meta.get("id") or _slug(clean, kind, seen_ids)
            meta["created"] = prev.get("created", today)
            prev_body = old_body_by_key.get(meta["id"]) or old_body_by_key.get(ntitle, "")
            changed = body.strip() != prev_body
            meta["touched"] = today if changed else prev.get("touched", today)
            if prev.get("tag") and "tag" not in meta:
                meta["tag"] = prev["tag"]
        else:
            meta.setdefault("id", _slug(clean, kind, seen_ids))
            meta["created"] = today
            meta["touched"] = today
        seen_ids.add(meta["id"])

        if kind == "lessons":
            # Renumber contiguously, keyed by id not position.
            clean = re.sub(r"^\d+\.\s+", f"{idx}. ", clean)
        comment = _meta_comment(meta)
        header = f"{clean} {comment}".rstrip() if comment else clean
        out_entries.append(header + body if body else header + "\n")

    rebuilt = (pre if pre.strip() else pre)
    # Ensure exactly the original preamble then entries, normalizing spacing so
    # entries are blank-line separated.
    body_text = "".join(out_entries)
    if pre and not pre.endswith("\n"):
        pre = pre + "\n"
    return (pre + body_text), warnings


# ---------------------------------------------------------------------------
# Injection cap (WU2) — bound what is injected per turn.
# ---------------------------------------------------------------------------

def _head_within(content: str, budget_tokens: int) -> str:
    """Keep head lines under the token budget (volatile/freeform files).

    For PROJECT_STATE/INDEX the newest/current content lives at the head
    (current status, overview), so head-keeping == newest-keeping. Returns the
    SAME object when it already fits (prefix-cache friendly).
    """
    char_budget = budget_tokens * 4
    if len(content) <= char_budget:
        return content
    # The note is part of the result, so it has to come out of the budget.
    # Leaving it unbudgeted put the result a note's width OVER the cap, which
    # the deterministic floor can never then satisfy — the file sits
    # permanently over target by that exact margin.
    body_budget = max(1, char_budget - len(_TRIM_NOTE))
    kept: list[str] = []
    used = 0
    for line in content.splitlines():
        if used + len(line) + 1 > body_budget:
            break
        kept.append(line)
        used += len(line) + 1
    return "\n".join(kept) + _TRIM_NOTE


_TRIM_NOTE = (
    "\n[... older content trimmed from this view — read the file on disk "
    "for the full text ...]"
)


def _fits_stripped(text: str, hard_budget: int) -> bool:
    """True if ``text`` fits ``hard_budget`` once mem-comments are excluded
    (spec §4.1 resolution 1). A no-op length check for text with no
    comments (e.g. content ``inject_view`` has already stripped)."""
    return len(user_flags.strip_mem_comments(text)) <= hard_budget * 4


# Lines naming an archive file are the ONLY route from a Layer-1 file to its
# Layer-2 archive. On the real project the LESSONS_ARCHIVE pointer was the last
# line of INDEX.md — the first casualty of a tail-dropping trim, which would
# leave the archive on disk and invisible. Pin them.
_ARCHIVE_POINTER = re.compile(r"_ARCHIVE\.md")


def _drop_tail_first(
    content: str, hard_budget: int, protected: set[int]
) -> tuple[str, bool]:
    """Drop lines NOT in ``protected``, tail-first, until ``content`` fits
    ``hard_budget`` (comment-stripped) or no droppable line remains.
    Returns ``(result, fits)``.
    """
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


_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _state_omitted_note(count: int) -> str:
    return (
        f"\n[... {count} older PROJECT_STATE line(s) omitted from this view — "
        "read orbital/PROJECT_STATE.md for the full text ...]"
    )


def _state_injected_view(raw: str, hard_budget: int) -> str:
    """PROJECT_STATE's per-turn view: everything, or oldest-first omission.

    A file over its hard budget hides its OLDEST lines from this turn's view
    (by the ``created`` stamp every bullet carries; old pointer lines by
    their date), never the newest. Agents append, so the old head-trim hid
    exactly the current work. Lines without a stamp (written outside the
    tools, not yet reconciled) count as newest. Nothing on disk changes —
    the view refreshes every call, and the editor / size floor do the real
    work. Only when that still cannot fit does it fall back to a head-trim.
    """
    stripped = user_flags.strip_mem_comments(raw)
    if _fits_stripped(stripped, hard_budget):
        return stripped

    from agent_os.agent import state_blocks

    lines, blocks = state_blocks.parse(raw)
    order: list[tuple[str, int, int]] = []        # (date, start, end)
    for b in blocks:
        if b.created and _ISO_DATE_RE.fullmatch(b.created):
            order.append((b.created, b.start, b.end))
    for idx, ln in enumerate(lines):
        d = state_blocks.pointer_date(ln)
        if d:
            order.append((d, idx, idx))
    order.sort()

    body_budget = max(1, int(hard_budget - est_tokens(_state_omitted_note(999))))
    hidden: set[int] = set()
    omitted = 0
    for _date, start, end in order:
        if _fits_stripped(
            "\n".join(ln for i, ln in enumerate(lines) if i not in hidden), body_budget
        ):
            break
        hidden.update(range(start, end + 1))
        omitted += 1
    kept = user_flags.strip_mem_comments(
        "\n".join(ln for i, ln in enumerate(lines) if i not in hidden)
    )
    if not _fits_stripped(kept, body_budget):
        return _head_within(kept, hard_budget)
    return kept.rstrip("\n") + _state_omitted_note(omitted) + "\n"


def inject_view(content: str | None, key: str, hard_budget: int) -> str | None:
    """Return the injected view of a Layer-1 file: newest-within-budget.

    Durable files (decisions/lessons): inject the newest entries that fit, PLUS
    the oldest ``PROTECT_OLDEST`` foundational entries. INDEX: head-trim.
    Returns the SAME object unchanged when everything fits, so a healthy
    project injects everything and the prefix cache is kept.

    PROJECT_STATE's ``<!--mem ...-->`` comments are daemon-managed machine
    metadata — stripped here before anything else, so the agent never sees
    them and they never count against the budget. Its overflow view drops the
    OLDEST lines first (see ``_state_injected_view``).
    """
    if not content:
        return content
    if key == "state":
        return _state_injected_view(content, hard_budget)
    if key in VOLATILE_KEYS or key not in ENTRY_MARKERS:
        return _head_within(content, hard_budget)

    marker = ENTRY_MARKERS[key]
    pre, raws = _split_entries(content, marker)
    if len(raws) < 2:
        return _head_within(content, hard_budget)

    char_budget = hard_budget * 4
    oldest = raws[:PROTECT_OLDEST]
    rest = raws[PROTECT_OLDEST:]
    used = len(pre) + sum(len(e) for e in oldest)
    newest_kept: list[str] = []
    for e in reversed(rest):
        if used + len(e) > char_budget:
            break
        newest_kept.insert(0, e)
        used += len(e)

    omitted = len(rest) - len(newest_kept)
    if omitted <= 0:
        return content  # everything fits — unchanged object

    note = (
        f"\n[... {omitted} older {DISPLAY_NAME[key]} entries omitted from this "
        f"view — read orbital/{MEMORY_FILENAME[key]} (and its archive, listed in "
        f"INDEX.md) for the full set ...]\n\n"
    )
    return pre + "".join(oldest) + note + "".join(newest_kept)


# ---------------------------------------------------------------------------
# Soft-cap flag (WU3) — dynamic slot only.
# ---------------------------------------------------------------------------

def entry_count(content: str | None, key: str) -> int:
    if not content or key not in ENTRY_MARKERS:
        return 0
    _, raws = _split_entries(content, ENTRY_MARKERS[key])
    return len(raws)


@dataclass(frozen=True)
class RefreshView:
    """Memory-editor scheduler state snapshot threaded into ``soft_flag``.

    The flag renders as a state machine driven by this view, not a repeating
    alarm. Incident (orbital-marketing, 2026-07-09): the flag re-fired
    identically every turn while a background pass was in flight, so the agent
    read "still over budget" as "checkpoint_state failed" and hand-trimmed the
    file mid-pass. ``last_outcome`` uses run_session_end_routine's vocabulary
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

# Escalation band: within this fraction of the hard cap, the flag warns about
# the deterministic floor that fires at the cap.
_HARD_CAP_WARN_FRACTION = 0.9


def over_soft_budget(content: str | None, key: str) -> bool:
    """True when ``key``'s content is over its soft budget — the editor's
    trigger, measured exactly the way the hygiene flag measures it."""
    budgets = FILE_BUDGETS.get(key)
    if not content or not budgets:
        return False
    return est_tokens(_budget_text(content, key)) > budgets["soft"]


def soft_flag(
    content: str | None, key: str, refresh: RefreshView | None = None
) -> str | None:
    """A persistent note while a file is over its soft threshold, or None.

    Token bound + entry count for legibility. The caller MUST place this in the
    dynamic/uncached slot (never the cached prefix) — a churning flag in the
    prefix busts the cache every turn.

    Over-budget files are tidied automatically (the memory editor runs in the
    background, archiving stale entries by id with a pointer left behind), so
    no state of this flag asks the agent to do anything but keep its lines
    true. It is state-aware via ``refresh`` so an in-flight pass never reads
    as a failure. Near the hard cap an escalation note is appended.
    """
    budgets = FILE_BUDGETS.get(key)
    if not content or not budgets:
        return None
    # Mem-comments never count against the state file's budget (spec §4.1
    # resolution 1) — a comment-heavy PROJECT_STATE that fits once its
    # machine metadata is excluded must not trip the soft-budget nudge.
    toks = est_tokens(_budget_text(content, key))
    soft = budgets["soft"]
    hard = budgets["hard"]
    if toks <= soft:
        return None
    suffix = f" ({entry_count(content, key)} entries)" if key in ENTRY_MARKERS else ""
    head = f"{DISPLAY_NAME[key]} memory {toks/1000:.1f}k/{soft//1000}k tok{suffix}"

    r = refresh or RefreshView()
    if r.in_flight:
        since = (
            f" (started turn {r.in_flight_since_turn})"
            if r.in_flight_since_turn is not None else ""
        )
        body = (
            f"{head} — the memory editor is tidying it in the background{since}; "
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
            f"{head} — over its soft budget; the memory editor tidies it "
            "automatically in the background (stale entries move to the archive, "
            "leaving an [archived … id:…] pointer). No action needed."
        )

    if toks >= hard * _HARD_CAP_WARN_FRACTION:
        body += (
            f" ⚠ {max(0, int(hard - toks))} tok from the hard cap ({hard}): past "
            "it, the oldest entries move to the archive automatically, each "
            "leaving a pointer."
        )
    return body


# ---------------------------------------------------------------------------
# Hard cap (WU5) — deterministic demote (durable) / trim (volatile).
# ---------------------------------------------------------------------------

def entry_manifest(content: str, key: str) -> list[dict]:
    """One row per entry: id, touched, tag, title — no bodies.

    Feeds the archive pass, which chooses entries by id. Keeping bodies out is
    the entire point: the daemon already has them, and making the model
    reproduce them is what made archiving unaffordable.
    """
    marker = ENTRY_MARKERS.get(key)
    if not content or not marker:
        return []
    _pre, raws = _split_entries(content, marker)
    rows = []
    for raw in raws:
        first_line, _rest = _first_line_split(raw)
        title, meta = _parse_meta(first_line)
        entry_id = meta.get("id")
        if not entry_id:
            continue                      # unstamped: only the floor can move it
        rows.append({
            "id": entry_id,
            "touched": meta.get("touched", ""),
            "tag": meta.get("tag", ""),
            "title": re.sub(r"^(##\s+|\d+\.\s+)", "", title).strip(),
        })
    return rows


def split_by_ids(content: str, key: str, ids: set[str]) -> tuple[str, str, set[str]]:
    """Split ``content`` into (kept, moved, matched_ids) by entry id.

    Byte-exact: the moved text is the original entry, not a reproduction. Ids
    that do not match, or that name a ``pinned`` entry, are simply not moved —
    the deterministic floor still guarantees the target, so a bad id costs
    quality, never content.
    """
    marker = ENTRY_MARKERS.get(key)
    if not content or not marker or not ids:
        return content, "", set()
    pre, raws = _split_entries(content, marker)
    kept, moved, matched = [], [], set()
    for raw in raws:
        first_line, _rest = _first_line_split(raw)
        _title, meta = _parse_meta(first_line)
        entry_id = meta.get("id")
        if entry_id in ids and meta.get("tag") != "pinned":
            moved.append(raw)
            matched.add(entry_id)
        else:
            kept.append(raw)
    if not moved:
        return content, "", set()
    return pre + "".join(kept), "".join(moved), matched


def split_for_demotion(content: str, key: str, hard_budget: int) -> tuple[str, str]:
    """Deterministically demote coldest entries until the file fits its budget.

    Returns ``(kept_text, demoted_text)``. Demotes by **coldest ``touched``
    first**; NEVER demotes the oldest ``PROTECT_OLDEST`` entries or any entry
    tagged ``pinned``. Moves, never deletes. Volatile files are not handled here
    (use ``trim_volatile``).
    """
    marker = ENTRY_MARKERS[key]
    pre, raws = _split_entries(content, marker)
    if len(raws) < 2 or est_tokens(content) <= hard_budget:
        return content, ""

    char_budget = hard_budget * 4
    n = len(raws)
    protected_idx = set(range(min(PROTECT_OLDEST, n)))
    # Parse touched + tag for ordering.
    info = []
    for i, raw in enumerate(raws):
        fl, _ = _first_line_split(raw)
        _, meta = _parse_meta(fl)
        info.append((i, meta.get("touched", "0000-00-00"), meta.get("tag", "")))
    for i, _, tag in info:
        if tag == "pinned":
            protected_idx.add(i)

    # Demote candidates: non-protected, ordered coldest touched first.
    candidates = sorted(
        (i for i, _, _ in info if i not in protected_idx),
        key=lambda i: info[i][1],
    )
    demote: set[int] = set()
    cur = len(pre) + sum(len(raws[i]) for i in range(n))
    for i in candidates:
        if cur <= char_budget:
            break
        demote.add(i)
        cur -= len(raws[i])

    if not demote:
        return content, ""
    kept = pre + "".join(raws[i] for i in range(n) if i not in demote)
    demoted = "".join(raws[i] for i in range(n) if i in demote)
    return kept, demoted


def trim_volatile(content: str, hard_budget: int) -> str:
    """INDEX.md's size floor: trim tail lines until the file fits.

    INDEX is regenerable navigation with no archive, so this is a plain
    head-keep / tail-drop — except for lines naming an ``*_ARCHIVE.md`` file,
    which are the only route to an archive and are never dropped. Fitting is
    measured comment-stripped. When the protected lines alone exceed the
    budget the content is returned unchanged. PROJECT_STATE no longer comes
    through here — its floor is age-based (``floor_state``).
    """
    if not content:
        return content

    if _fits_stripped(content, hard_budget):
        return content

    protected = {
        i for i, line in enumerate(content.split("\n"))
        if _ARCHIVE_POINTER.search(line)
    }
    if not protected:
        # _head_within appends its own note; leave its budgeting alone.
        return _head_within(content, hard_budget)

    # Below, the note is appended AFTER fitting, so budget for it up front —
    # otherwise the result lands a note's width OVER the target and the file
    # stays permanently over budget by exactly that margin.
    fit_budget = max(1, int(hard_budget - est_tokens(_TRIM_NOTE)))
    trimmed, fits = _drop_tail_first(content, fit_budget, protected)
    if not fits:
        return content

    return trimmed + _TRIM_NOTE


# ---------------------------------------------------------------------------
# Pointers + the deterministic floor (spec 089 §4.3, §4.4).
# ---------------------------------------------------------------------------

# PROJECT_STATE lines younger than this are never cut by the floor.
STATE_PROBATION_DAYS = 14

_POINTER_SUMMARY_CHARS = 60


def pointer_line(day: str, ids: list[str], summary: str, archive_filename: str) -> str:
    """``[archived YYYY-MM-DD id:<id>] <one-line pointer> → <ARCHIVE_FILE>``.

    The ONE pointer format every archiving path writes, so a future reader can
    grep the archive for ``id:<id>`` and read only that entry.
    """
    summary = " ".join((summary or "").replace("→", "->").split()) or "older entry"
    tag = " ".join(f"id:{i}" for i in ids if i)
    head = f"[archived {day} {tag}]" if tag else f"[archived {day}]"
    return f"{head} {summary} → {archive_filename}"


def pointer_summary(text: str, limit: int = _POINTER_SUMMARY_CHARS) -> str:
    """A pointer's default one-liner: the entry's own opening words."""
    text = " ".join(text.replace("**", "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _is_pointer(line: str) -> bool:
    from agent_os.agent import state_blocks
    return state_blocks.pointer_date(line) is not None


@dataclass
class FloorResult:
    content: str               # new live content (the input object when untouched)
    archived: str              # moved blocks, verbatim ("" when none)
    moved_ids: list[str]
    dropped_pointers: int
    over_by: int               # tokens still over target afterwards (0 = fits)


def floor_state(
    content: str, target_tokens: int, today: str,
    archive_filename: str = "PROJECT_STATE_ARCHIVE.md",
) -> FloorResult:
    """PROJECT_STATE's last-resort size floor: age-based, never recent.

    Candidates are bullets whose ``created`` stamp is older than
    ``STATE_PROBATION_DAYS`` and pointer lines dated older than that, oldest
    first. A bullet moves to the archive verbatim (its mem-comment keeps its
    id) and a pointer line takes its place; an old pointer is simply removed
    (its body is already archived). Nothing inside the probation window is
    ever a candidate, so when candidates run out before the target the file
    stays over budget — over budget beats losing current work — and
    ``over_by`` says by how much.
    """
    from datetime import timedelta
    from agent_os.agent import state_blocks

    def measure(text: str) -> float:
        return est_tokens(_budget_text(text, "state"))

    if not content or measure(content) <= target_tokens:
        return FloorResult(content, "", [], 0, 0)

    lines, blocks = state_blocks.parse(content)
    cutoff = (date.fromisoformat(today) - timedelta(days=STATE_PROBATION_DAYS)).isoformat()
    candidates: list[tuple[str, int, object]] = []
    for b in blocks:
        if b.id and b.created and _ISO_DATE_RE.fullmatch(b.created) and b.created < cutoff:
            candidates.append((b.created, b.start, b))
    for idx, line in enumerate(lines):
        d = state_blocks.pointer_date(line)
        if d and d < cutoff:
            candidates.append((d, idx, None))
    candidates.sort(key=lambda c: (c[0], c[1]))

    # start line -> (end line, replacement lines)
    replaced: dict[int, tuple[int, list[str]]] = {}

    def render() -> str:
        out: list[str] = []
        i = 0
        while i < len(lines):
            if i in replaced:
                end, repl = replaced[i]
                out.extend(repl)
                i = end + 1
                continue
            out.append(lines[i])
            i += 1
        return "\n".join(out)

    moved: list = []
    dropped = 0
    current = content
    for _d, start, block in candidates:
        if measure(current) <= target_tokens:
            break
        if block is None:
            replaced[start] = (start, [])
            dropped += 1
        else:
            ptr = pointer_line(
                today, [block.id], pointer_summary(block.text), archive_filename)
            if len(ptr) >= len("\n".join(block.content_lines)):
                continue       # a pointer this long would GROW the file
            replaced[start] = (block.end, [ptr])
            moved.append(block)
        current = render()

    over_by = max(0, int(measure(current) - target_tokens))
    if not moved and not dropped:
        return FloorResult(content, "", [], 0, over_by)
    moved.sort(key=lambda b: b.start)
    return FloorResult(
        current,
        "\n".join(b.raw for b in moved),
        [b.id for b in moved],
        dropped,
        over_by,
    )


def split_out_pointer_lines(raw: str) -> tuple[str, list[str]]:
    """Separate pointer lines from an entry chunk.

    In DECISIONS/LESSONS a pointer line left between two entries parses as
    the tail of the entry ABOVE it. When that entry later moves to the
    archive, its pointers must stay in the live file — they are the route to
    OTHER archived entries.
    """
    kept: list[str] = []
    pointers: list[str] = []
    for line in raw.split("\n"):
        (pointers if _is_pointer(line) else kept).append(line)
    return "\n".join(kept), pointers


def _entry_title(raw: str, key: str) -> str:
    first_line, _rest = _first_line_split(raw)
    title, _meta = _parse_meta(first_line)
    return re.sub(r"^(##\s+|\d+\.\s+)", "", title).strip()


def entry_id(raw: str) -> str | None:
    first_line, _rest = _first_line_split(raw)
    return _parse_meta(first_line)[1].get("id") or None


def demote_with_pointers(
    content: str, key: str, target_tokens: int, today: str, archive_filename: str,
) -> tuple[str, str, list[str]]:
    """DECISIONS/LESSONS floor: demote coldest-``touched`` entries, by id.

    Same order and protection as ``split_for_demotion`` (oldest
    ``PROTECT_OLDEST`` and ``pinned`` entries never move), but every demoted
    entry leaves a ``pointer_line`` in its place, pointer lines sitting in a
    demoted chunk stay behind, and an entry without an id is never moved
    (the caller stamps ids first; an unaddressable entry could not be
    recalled). Returns ``(kept, demoted_text, demoted_ids)``.
    """
    marker = ENTRY_MARKERS[key]

    def measure(text: str) -> float:
        return est_tokens(_budget_text(text, key))

    pre, raws = _split_entries(content, marker)
    if len(raws) < 2 or measure(content) <= target_tokens:
        return content, "", []

    protected = set(range(min(PROTECT_OLDEST, len(raws))))
    info = []
    for i, raw in enumerate(raws):
        first_line, _rest = _first_line_split(raw)
        _title, meta = _parse_meta(first_line)
        if meta.get("tag") == "pinned" or not meta.get("id"):
            protected.add(i)
        info.append((meta.get("touched", "0000-00-00"), i, meta.get("id")))
    candidates = sorted((t, i, eid) for t, i, eid in info if i not in protected)

    new_raws = list(raws)
    demoted: list[tuple[int, str]] = []
    ids: list[str] = []
    for _touched, i, eid in candidates:
        if measure(pre + "".join(new_raws)) <= target_tokens:
            break
        body, pointers = split_out_pointer_lines(raws[i])
        trailing = raws[i][len(raws[i].rstrip("\n")):] or "\n"
        ptr = pointer_line(
            today, [eid], pointer_summary(_entry_title(raws[i], key), 80),
            archive_filename,
        )
        if len(ptr) >= len(body.strip()):
            continue           # a pointer this long would GROW the file
        new_raws[i] = "\n".join([ptr, *[p for p in pointers if p.strip()]]) + trailing
        demoted.append((i, body.rstrip("\n") + "\n"))
        ids.append(eid)

    if not demoted:
        return content, "", []
    demoted.sort()
    return pre + "".join(new_raws), "\n".join(b for _i, b in demoted), ids


# ---------------------------------------------------------------------------
# Write-path entry point used by the write/edit tools (WU1).
# ---------------------------------------------------------------------------

# basename -> file_key, for detecting a memory-file write by resolved path.
_BASENAME_TO_KEY = {v: k for k, v in MEMORY_FILENAME.items()}


def memory_key_for_path(resolved_path: str, workspace: str) -> str | None:
    """Return the Layer-1 file_key if ``resolved_path`` is ``<ws>/orbital/<file>``."""
    try:
        rp = os.path.realpath(resolved_path)
        orbital = os.path.realpath(os.path.join(workspace, "orbital"))
        if os.path.dirname(rp) != orbital:
            return None
        return _BASENAME_TO_KEY.get(os.path.basename(rp))
    except OSError:
        return None


def process_on_write(workspace: str, resolved_path: str, content: str, *, today: str | None = None) -> tuple[str, list[str]]:
    """Stamp metadata + enforce format on a memory-file write (non-destructive).

    Called by the ``write``/``edit`` tools after they resolve the target path and
    compute the content. If the path is not a Layer-1 memory file, returns the
    content unchanged. The hard cap (demotion/trim) is NOT applied here — it runs
    deterministically at session-end (``apply_hard_cap``) so the write path stays
    cheap and never deletes/demotes mid-edit.
    """
    key = memory_key_for_path(resolved_path, workspace)
    if key is None:
        return content, []
    content = ensure_format_header(content, key)
    if key == "state":
        # PROJECT_STATE runs through the flag chokepoint: preserve ids + user
        # lifecycle decisions across the agent's (comment-less) rewrite by
        # diffing against the previous on-disk content. Lazy import avoids a
        # module-load cycle.
        from agent_os.agent import flag_chokepoint, retractions
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                prev = f.read()
        except OSError:
            prev = None
        orbital_dir = os.path.join(workspace, "orbital")
        retraction_titles = [r.title for r in retractions.list_retractions(orbital_dir)]
        return flag_chokepoint.reconcile_flags(
            prev, content, today or _today(), retraction_titles
        )
    if key not in ENTRY_MARKERS:
        # Other volatile files (index): header only, no entry stamping.
        return content, []
    old = None
    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            old = f.read()
    except OSError:
        old = None
    return stamp(content, old, key, today=today)
