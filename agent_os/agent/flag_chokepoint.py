# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The PROJECT_STATE write chokepoint: every bullet carries an id and dates
(spec 089 §4.1).

Agents rewrite PROJECT_STATE freely and never see the machine ``<!--mem-->``
comments (they are stripped from the injected view). So a raw agent rewrite
would drop every id and every ``created`` date. This module is the seam every
state write passes through: it diffs the new content against the previous
on-disk content and re-attaches what the writer could not have known about.

``reconcile_flags`` is pure: the caller supplies the previous on-disk bytes.
It is wired into both state-write paths — ``memory_entries.process_on_write``
(the agent write/edit tools) and ``WorkspaceFileManager.write`` (daemon and
editor writes) — so every PROJECT_STATE.md write is reconciled.

Rules:

- EVERY top-level bullet block (see ``state_blocks``) gets a comment with
  ``id``, ``created`` and ``touched``, directly under its bullet line.
- id and dates carry forward ONLY on an exact, whitespace-normalised text
  match. A reworded line is a new line — new id, today's dates. That errs
  toward keeping it: the size floor never cuts a line younger than 14 days,
  and the editor archives by id, so a line it has not seen cannot be cut by
  mistake.
- a comment the new content already carries wins field-by-field over the
  previous file's (daemon writers such as the editor or the Workbench write
  real metadata); an id is never emitted twice.

What used to live here and was retired by 089 v2: the fuzzy title matcher,
the ``resolved`` lock, the retraction drop and the omission lint. Items that
wait on the user now live in their own list, so the identity of a
PROJECT_STATE line no longer has to be reconstructed from reworded prose.
"""

from __future__ import annotations

from datetime import date

from agent_os.agent import state_blocks, user_flags


def _today() -> str:
    return date.today().isoformat()


def reconcile_flags(
    prev: str | None,
    new: str,
    today: str,
    retraction_titles: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Stamp every PROJECT_STATE bullet, carrying ids forward by exact text.

    Returns ``(merged_content, warnings)``. Warnings never block the write.
    ``retraction_titles`` is accepted for call-site compatibility and ignored.
    """
    del retraction_titles
    today = today or _today()
    if not new:
        return new, []

    prev_blocks = state_blocks.parse(prev)[1] if prev else []
    prev_by_id = {b.id: b for b in prev_blocks if b.id}
    # match_key -> previous blocks with that exact text, in file order.
    prev_by_key: dict[str, list] = {}
    for b in prev_blocks:
        prev_by_key.setdefault(b.match_key, []).append(b)

    lines, blocks = state_blocks.parse(new)
    used_ids: set[str] = set()
    claimed_prev: set[int] = set()
    rendered: dict[int, list[str]] = {}

    for b in blocks:
        key = b.match_key
        fields: dict[str, str] | None = None
        own = prev_by_id.get(b.id) if b.id else None
        if own is not None and id(own) not in claimed_prev and b.id not in used_ids:
            if own.match_key == key:
                # Same line, comment still attached (or re-sent by a daemon
                # writer): the writer's fields win, the old ones fill gaps.
                fields = {**own.fields, **b.fields}
                claimed_prev.add(id(own))
        carries_unknown_id = bool(b.id) and b.id not in prev_by_id and b.id not in used_ids
        if fields is None and not carries_unknown_id:
            # The comment-stripped rewrite: the same words, anywhere in the file.
            for cand in prev_by_key.get(key, ()):
                if id(cand) in claimed_prev or not cand.id or cand.id in used_ids:
                    continue
                fields = dict(cand.fields)
                claimed_prev.add(id(cand))
                break
        if fields is None and carries_unknown_id:
            # An id the previous file never had (restore from backup, a
            # block the editor re-inserted): keep it.
            fields = dict(b.fields)
        if fields is None:
            fields = {
                "id": _fresh_id(used_ids | set(prev_by_id)),
                "created": today,
                "touched": today,
            }
        fields.setdefault("created", today)
        fields.setdefault("touched", fields["created"])
        used_ids.add(fields["id"])
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
    merged = "\n".join(out)
    return merged, user_flags.lint(merged)


def _fresh_id(taken: set[str]) -> str:
    while True:
        eid = user_flags.new_entry_id()
        if eid not in taken:
            return eid
