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
# Durable file -> its archive file-key (read-on-demand, never injected).
ARCHIVE_OF: dict[str, str] = {
    "decisions": "decisions_archive",
    "lessons": "lessons_archive",
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
        "<!--format PROJECT_STATE is what is true NOW: current focus, "
        "in-progress work, blockers, next steps. Overwrite stale lines; never "
        "append a dated history. "
        "[user] flag — one judgment per entry: does this need the user? If yes, "
        "tag the bullet `- [user] <sentence>`; a dated commitment needing no "
        "decision is `- [due:YYYY-MM-DD] <text>` (a fact that also shows on the "
        "calendar, not addressed to the user); an untagged bullet is a plain "
        "fact. Machine attributes (id, from, evidence, confidence, due, created, "
        "touched, resolved) live in a daemon-managed mem-comment on the next "
        "line — never hand-write or edit ids. "
        "Rails: (1) Attention test — flag what a competent assistant would "
        "raise: awaiting the user's decision or action, a mid-flight plan they "
        "may want to continue, or something they'd be unpleasantly surprised to "
        "have missed. (2) Assignment, not capability — what the user took on is "
        "theirs even if you could do it. (3) Never auto-decide — spending money, "
        "sending external messages as the user, or irreversible or destructive "
        "acts are always surfaced, whatever the autonomy setting. (4) Decompose "
        "compounds — split agent-work + user-decision + user-action into "
        "separate entries. (5) Register — a flagged entry is one plain sentence "
        "addressed to the user, in their language, written timeless (\"due Jul "
        "28\", never \"tomorrow\"); unflagged entries stay dense shorthand. "
        "(6) Laundering guard — a flag must anchor to user-visible evidence "
        "(their own words or acts); if the only basis is your own wording, mark "
        "confidence:unconfirmed and phrase it as an ask-to-confirm, never a "
        "settled commitment.-->"
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
    """Prepend the file's <!--format--> contract if missing (idempotent).

    An existing header (even an agent-edited variant) is left alone — this is
    self-healing, not normalization.
    """
    header = FORMAT_HEADERS.get(key)
    text = content or ""
    if header is None or text.lstrip().startswith("<!--format"):
        return text
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
    evidence, confidence, timestamps) on PROJECT_STATE bullets — they never
    compete with agent-visible context for budget, and are never shown to the
    agent at all. Only ``state`` carries this comment grammar; every other
    key (including DECISIONS/LESSONS, whose own ``<!--mem id:...-->`` stamp
    is a *different*, always-injected metadata convention) passes through
    unchanged so their budget/injection behavior stays byte-identical.
    Content with no mem-comments round-trips unchanged (no grammar adopted
    yet), so this is a no-op for the common case.
    """
    if key != "state":
        return content
    return user_flags.strip_mem_comments(content)


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
    kept: list[str] = []
    used = 0
    for line in content.splitlines():
        if used + len(line) + 1 > char_budget:
            break
        kept.append(line)
        used += len(line) + 1
    note = (
        f"\n[... older content trimmed from this view — read the file on disk "
        f"for the full text ...]"
    )
    return "\n".join(kept) + note


def inject_view(content: str | None, key: str, hard_budget: int) -> str | None:
    """Return the injected view of a Layer-1 file: newest-within-budget.

    Durable files (decisions/lessons): inject the newest entries that fit, PLUS
    the oldest ``PROTECT_OLDEST`` foundational entries. Volatile files
    (state/index): head-trim. Returns the SAME object unchanged when everything
    fits, so a healthy project injects everything and the prefix cache is kept.

    PROJECT_STATE's ``<!--mem ...-->`` comments are daemon-managed machine
    metadata (spec §4.1) — stripped here before anything else, so the agent
    never sees them and they never count against (or get counted toward
    filling) the budget. A no-op for content without the grammar.
    """
    if not content:
        return content
    if key == "state":
        content = user_flags.strip_mem_comments(content)
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
    """Consolidation-scheduler state snapshot threaded into ``soft_flag``.

    The flag renders as a state machine driven by this view, not a repeating
    alarm. Incident (orbital-marketing, 2026-07-09): the flag re-fired
    identically every turn while a background pass was in flight, so the agent
    read "still over budget" as "checkpoint_state failed" and hand-trimmed the
    file mid-pass. ``last_outcome`` uses run_session_end_routine's vocabulary:
    "llm_merged" | "backstop_only" | "failed" | "no_delta" | None.
    """
    in_flight: bool = False
    in_flight_since_turn: int | None = None
    last_outcome: str | None = None
    last_turn: int | None = None


# Escalation band: within this fraction of the hard cap, the flag warns about
# the deterministic demote-to-archive that fires at the cap.
_HARD_CAP_WARN_FRACTION = 0.9


def soft_flag(
    content: str | None, key: str, refresh: RefreshView | None = None
) -> str | None:
    """A persistent nudge while a file is over its soft threshold, or None.

    Token bound + entry count for legibility. The caller MUST place this in the
    dynamic/uncached slot (never the cached prefix) — a churning flag in the
    prefix busts the cache every turn.

    The nudge is state-aware via ``refresh``: while a consolidation pass is in
    flight it says "no action needed" (never re-suggests the tool or a manual
    edit); after a pass that couldn't run its LLM merge, the manual edit is the
    sanctioned path (OCC makes concurrent hand-edits safe); after a successful
    merge that still leaves the file over budget, the remainder is genuinely
    large and only a manual trim can help. Near the hard cap an escalation
    warning is appended in every state.
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
            f"{head} — a consolidation pass is in flight{since}; no action "
            "needed. Do NOT re-trigger checkpoint_state or hand-edit this file "
            "— the flag may persist until the pass lands."
        )
    elif r.last_outcome in ("backstop_only", "failed"):
        at = f" (turn {r.last_turn})" if r.last_turn is not None else ""
        body = (
            f"{head} — the last background consolidation{at} could not run its "
            "LLM merge (deterministic backstop only), so re-triggering "
            "checkpoint_state will not reduce this file; edit the file directly "
            "to merge duplicates and trim stale entries."
        )
    elif r.last_outcome == "llm_merged":
        at = f" at turn {r.last_turn}" if r.last_turn is not None else ""
        body = (
            f"{head} — consolidation already ran{at} and this is what remains; "
            "the content is genuinely large. If entries are stale, edit the "
            "file directly to trim them; otherwise no action is useful."
        )
    else:
        body = (
            f"{head} — call the checkpoint_state tool to consolidate (merge "
            "duplicates, supersede stale entries), or edit the file directly."
        )

    if toks >= hard * _HARD_CAP_WARN_FRACTION:
        body += (
            f" ⚠ {max(0, int(hard - toks))} tok from the hard cap ({hard}): at "
            "the cap, over-budget durable entries are demoted to the archive "
            "automatically."
        )
    return body


# ---------------------------------------------------------------------------
# Hard cap (WU5) — deterministic demote (durable) / trim (volatile).
# ---------------------------------------------------------------------------

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
    """Trim oldest (tail) content of a volatile file to fit its budget.

    Volatile = disposable; no archive. Head-keep mirrors ``_head_within`` so the
    current status/overview survives.

    Overflow never deletes a ``[user]``-flagged entry (spec §4.1 resolution
    2): every line belonging to a flagged bullet (its comment line included)
    is protected. Unflagged prose is dropped tail-first — oldest first, same
    direction as the legacy head-keep/tail-drop shape — until the file fits.
    Budget fitting is measured on the comment-stripped length (resolution 1:
    mem-comments never count against the cap), so a comment-heavy file that
    already fits once its machine metadata is excluded is left untouched.
    If flagged entries alone still exceed the cap after every unflagged line
    is gone, the content is returned COMPLETELY UNCHANGED — no partial or
    silent deletion of a user-facing obligation — and the existing soft-cap
    hygiene nudge (``soft_flag``) keeps signalling that a manual/checkpoint
    pass is needed, exactly as it does today.

    A file with no ``[user]``-flagged entries (including any file that
    hasn't adopted the grammar at all, e.g. INDEX.md) falls back to the
    original head-trim behavior unchanged.
    """
    if not content:
        return content

    def _fits(text: str) -> bool:
        return len(user_flags.strip_mem_comments(text)) <= hard_budget * 4

    if _fits(content):
        return content

    flagged_lines: set[int] = set()
    for e in user_flags.parse_entries(content):
        if e.flagged:
            flagged_lines.update(range(e.line_start, e.line_end + 1))

    if not flagged_lines:
        return _head_within(content, hard_budget)

    lines = content.split("\n")
    n = len(lines)
    keep = [True] * n

    def _cur_fits() -> bool:
        return _fits("\n".join(lines[j] for j in range(n) if keep[j]))

    i = n - 1
    while i >= 0 and not _cur_fits():
        if i not in flagged_lines:
            keep[i] = False
        i -= 1

    if not _cur_fits():
        # Flagged entries alone exceed the cap even with every unflagged
        # line removed — leave the file untouched.
        return content

    trimmed = "\n".join(lines[j] for j in range(n) if keep[j])
    note = (
        "\n[... older content trimmed from this view — read the file on disk "
        "for the full text ...]"
    )
    return trimmed + note


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
