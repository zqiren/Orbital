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
from datetime import date

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Budgets (len/4). Floors are the measured clean-mature sizes + headroom
# (VERIFY-phase1-injection-cap.md Q2), set ABOVE the full clean set so a healthy
# project injects everything and only a runaway overflows.
# ---------------------------------------------------------------------------

FILE_BUDGETS: dict[str, dict[str, int]] = {
    "decisions": {"soft": 7000, "hard": 9000},
    "lessons": {"soft": 5000, "hard": 6000},
    "state": {"soft": 1500, "hard": 2000},
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


def est_tokens(text: str | None) -> float:
    """len/4 token estimate — matches ``token_utils`` / ``context.py``."""
    return len(text) / 4 if text else 0.0


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
    """
    if not content:
        return content
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


def soft_flag(content: str | None, key: str) -> str | None:
    """A persistent nudge while a file is over its soft threshold, or None.

    Token bound + entry count for legibility. The caller MUST place this in the
    dynamic/uncached slot (never the cached prefix) — a churning flag in the
    prefix busts the cache every turn.
    """
    budgets = FILE_BUDGETS.get(key)
    if not content or not budgets:
        return None
    toks = est_tokens(content)
    soft = budgets["soft"]
    if toks <= soft:
        return None
    suffix = f" ({entry_count(content, key)} entries)" if key in ENTRY_MARKERS else ""
    return (
        f"{DISPLAY_NAME[key]} memory {toks/1000:.1f}k/{soft//1000}k tok{suffix} "
        f"— consider consolidating (merge duplicates, supersede stale entries)."
    )


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
    """
    return _head_within(content, hard_budget)


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
    if key is None or key not in ENTRY_MARKERS:
        return content, []
    old = None
    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            old = f.read()
    except OSError:
        old = None
    return stamp(content, old, key, today=today)
