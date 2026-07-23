# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The write chokepoint: id-preserving, lifecycle-preserving merge for
PROJECT_STATE.md (spec §5, §5.2, §5.5, §8).

Agents rewrite PROJECT_STATE freely and never see the machine ``<!--mem-->``
comments (they are stripped from the injected view). So a raw agent rewrite
would drop every id, every ``created`` date, and every user decision. This
module is the seam every state write passes through: it diffs the agent's new
content against the previous on-disk content and re-attaches what the agent
could not have known about.

``reconcile_flags`` is pure: the caller supplies the previous on-disk bytes.
It is wired into both state-write paths — ``memory_entries.process_on_write``
(the agent write/edit tools) and ``WorkspaceFileManager.write`` (daemon/system
writes, e.g. the session-end merge) — so every PROJECT_STATE.md write is
reconciled regardless of which writer produced it.

Merge rules (spec §5.2 / §5.5):

- **id-matched** bullets keep their comment (id, created, from, evidence).
- **comment-less rewritten** bullets re-associate by fuzzy title match
  (normalized ratio >= 0.75) against the previous file.
- **new flagged** bullets get a fresh id + ``created:today``.
- a **text change** on a matched entry stamps ``touched:today``.
- **user lifecycle fields win**: ``resolved`` and ``confidence:stated`` are
  re-attached when the agent's (stale) rewrite drops or reverts them — a
  resolved entry rewritten as unresolved comes back resolved.
- an entry the user **retracted** that reappears (title-matched) is kept OUT
  and warned about loudly.

Lint (spec §8) is folded in as warnings that never block the write: the
grammar lint from ``user_flags.lint`` (malformed ``due``, flagged entry
missing ``evidence``/``from``) plus the omission heuristic (unflagged
user-directed content under a blocker/waiting/next-step heading, or carrying
``用户`` / "you must|need|should" phrasing).
"""

from __future__ import annotations

import re
from datetime import date
from difflib import SequenceMatcher

from agent_os.agent import user_flags

# Fuzzy re-association threshold (spec §5.2): normalized difflib ratio.
_FUZZY_THRESHOLD = 0.75

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(?P<text>.*)$")
_BULLET_LINE_RE = re.compile(r"^\s*[-*+]\s+\S")
# Headings that make an unflagged bullet suspicious (spec §8 omission list).
_OMISSION_HEADING_RE = re.compile(r"blocker|waiting|needs|next\s*step", re.IGNORECASE)
# User-directed phrasing an unflagged bullet should probably have flagged.
_OMISSION_PHRASE_RE = re.compile(r"用户|you\s+(?:must|need|should)", re.IGNORECASE)
# A trailing, single-line inline mem-comment on a bullet line.
_INLINE_COMMENT_RE = re.compile(r"\s*<!--\s*mem\b.*?-->\s*$", re.DOTALL)

# 2-space indent nests the machine comment under its bullet (matches spec §4).
_COMMENT_INDENT = "  "


def _today() -> str:
    return date.today().isoformat()


def _norm_for_match(text: str) -> str:
    """Lowercase, drop punctuation/whitespace/underscore (keep letters incl.
    CJK + digits) — the normalization used for fuzzy title re-association."""
    return re.sub(r"[\W_]+", "", text.lower())


def _ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _merge_fields(entry, match, today: str) -> dict[str, str]:
    """Compute the reconciled ``<!--mem-->`` fields for one flagged entry.

    ``match`` is the previous-file entry it re-associates to (or ``None`` for a
    genuinely new bullet). User lifecycle fields on ``match`` win over the
    agent's rewrite.
    """
    if match is not None:
        eid = match.id or entry.id or user_flags.new_entry_id()
        created = match.created or entry.created or today
        text_changed = _norm_for_match(entry.text) != _norm_for_match(match.text)
        touched = today if text_changed else (match.touched or today)
        from_session = entry.from_session or match.from_session
        evidence = entry.evidence or match.evidence
        # Lifecycle-wins: a user 'stated' confidence is never reverted to
        # 'unconfirmed' by a stale agent copy.
        confidence = "stated" if match.confidence == "stated" else (
            entry.confidence or match.confidence
        )
        # Lifecycle-wins: a resolved entry stays resolved.
        resolved = match.resolved or entry.resolved
    else:
        eid = entry.id or user_flags.new_entry_id()
        created = entry.created or today
        touched = entry.touched or today
        from_session = entry.from_session
        evidence = entry.evidence
        confidence = entry.confidence
        resolved = entry.resolved

    fields: dict[str, str] = {"id": eid}
    if from_session:
        fields["from"] = from_session
    if evidence:
        fields["evidence"] = evidence
    if confidence:
        fields["confidence"] = confidence
    if created:
        fields["created"] = created
    if touched:
        fields["touched"] = touched
    if resolved:
        fields["resolved"] = resolved
    return fields


def _strip_inline_comment(line: str) -> str:
    """Remove a trailing inline ``<!--mem-->`` from a bullet line, if any."""
    return _INLINE_COMMENT_RE.sub("", line).rstrip()


def _omission_warnings(lines: list[str], entry_starts: set[int]) -> list[str]:
    """Warn on unflagged bullets that read as user-facing (spec §8).

    An entry-start line (a parsed flagged/dated bullet) is never flagged — only
    genuinely untagged bullets under a suspicious heading or carrying
    user-directed phrasing.
    """
    warns: list[str] = []
    heading = ""
    for idx, line in enumerate(lines):
        hm = _HEADING_RE.match(line)
        if hm:
            heading = hm.group("text")
            continue
        if idx in entry_starts or not _BULLET_LINE_RE.match(line):
            continue
        text = line.strip().lstrip("-*+ ").strip()
        if _OMISSION_HEADING_RE.search(heading) or _OMISSION_PHRASE_RE.search(text):
            warns.append(
                f"line {idx + 1}: possible unflagged user-facing content "
                f"(\"{text[:60]}\") — consider a [user] flag."
            )
    return warns


def reconcile_flags(
    prev: str | None,
    new: str,
    today: str,
    retraction_titles: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Reconcile an agent's PROJECT_STATE rewrite against the previous content.

    Returns ``(merged_content, warnings)``. Warnings never block the write.
    """
    today = today or _today()
    retraction_titles = retraction_titles or []
    warnings: list[str] = []
    if not new:
        return new, warnings

    prev_entries = user_flags.parse_entries(prev) if prev else []
    new_entries = user_flags.parse_entries(new)
    prev_by_id = {e.id: e for e in prev_entries if e.id}
    norm_retractions = [_norm_for_match(t) for t in retraction_titles if t]

    # Classify each new entry in file order: "drop" (retracted), "keep_raw"
    # (unstamped dated fact), or "flag" (a flagged entry that needs a match).
    kinds: dict[int, str] = {}
    flagged_new: list = []
    for e in new_entries:
        norm_title = _norm_for_match(e.text)
        if any(nr and _ratio(norm_title, nr) >= _FUZZY_THRESHOLD for nr in norm_retractions):
            kinds[e.line_start] = "drop"
            warnings.append(
                f"RETRACTED: dropped re-added entry \"{e.text[:60]}\" — the user "
                f"retracted this; it may return only by explicit user request."
            )
            continue
        if not e.flagged:
            kinds[e.line_start] = "keep_raw"  # dated fact (spec §5.2)
            continue
        kinds[e.line_start] = "flag"
        flagged_new.append(e)

    # Re-associate flagged entries to previous entries. Exact-id matches take
    # precedence; the remainder are assigned best-ratio-first GLOBALLY (not
    # greedy by file order) so the highest-similarity bullet wins the id +
    # lifecycle fields when several clear the threshold — a weaker decoy that
    # happens to appear first in the file must not steal a better match.
    match_for: dict[int, object] = {}          # id(new Entry) -> prev Entry
    claimed_prev: set[int] = set()
    unmatched: list = []
    for e in flagged_new:
        pe = prev_by_id.get(e.id) if e.id else None
        if pe is not None and id(pe) not in claimed_prev:
            match_for[id(e)] = pe
            claimed_prev.add(id(pe))
        else:
            unmatched.append(e)

    candidates: list[tuple[float, int, int, object, object]] = []
    for ni, e in enumerate(unmatched):
        target = _norm_for_match(e.text)
        for pi, pe in enumerate(prev_entries):
            if id(pe) in claimed_prev:
                continue
            r = _ratio(target, _norm_for_match(pe.text))
            if r >= _FUZZY_THRESHOLD:
                candidates.append((r, ni, pi, e, pe))
    # Highest ratio first; ties broken by file order for determinism.
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    claimed_new: set[int] = set()
    for _r, _ni, _pi, e, pe in candidates:
        if id(e) in claimed_new or id(pe) in claimed_prev:
            continue
        match_for[id(e)] = pe
        claimed_new.add(id(e))
        claimed_prev.add(id(pe))

    # Build the per-entry decision (retracted entries drop, facts pass through,
    # flagged entries emit with reconciled comment fields).
    decisions: dict[int, tuple[str, dict[str, str] | None]] = {}
    for e in new_entries:
        kind = kinds[e.line_start]
        if kind == "flag":
            decisions[e.line_start] = ("emit", _merge_fields(e, match_for.get(id(e)), today))
        else:
            decisions[e.line_start] = (kind, None)

    # Rebuild content using `new` as the skeleton; only entry lines change.
    entry_at = {e.line_start: e for e in new_entries}
    lines = new.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        e = entry_at.get(i)
        if e is None:
            out.append(lines[i])
            i += 1
            continue
        kind, fields = decisions[e.line_start]
        end = e.line_end
        if kind == "drop":
            i = end + 1
            continue
        if kind == "keep_raw":
            out.extend(lines[e.line_start:end + 1])
            i = end + 1
            continue
        out.append(_strip_inline_comment(lines[e.line_start]))
        out.append(_COMMENT_INDENT + user_flags.render_comment(fields))
        i = end + 1
    merged = "\n".join(out)

    warnings.extend(_omission_warnings(lines, set(entry_at)))
    warnings.extend(user_flags.lint(merged))
    return merged, warnings
