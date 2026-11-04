# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Line diffs for file history (spec 065 §5.3, §5.6).

``count_changes`` gives a version's ``added``/``removed`` (``改了 N 行`` =
their sum). ``compose_view`` builds the tinted version view: the target
version's lines plus the deleted lines of ``from``, each changed line tagged
with the version that last introduced (or removed) it. The front end only
paints.
"""
from __future__ import annotations

import difflib


def split_lines(data: bytes | None) -> list[str]:
    if not data:
        return []
    return data.decode("utf-8", errors="replace").splitlines()


def count_changes(old: bytes | None, new: bytes | None) -> tuple[int, int]:
    """(added, removed) line counts from ``old`` to ``new``."""
    a, b = split_lines(old), split_lines(new)
    added = removed = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        removed += i2 - i1
        added += j2 - j1
    return added, removed


def compose_view(base: bytes | None, chain: list[bytes | None]) -> list[dict]:
    """The view of ``chain[-1]`` diffed against ``base``.

    ``chain`` = the contents of each version after ``base`` up to and
    including the selected one, oldest first (one element for a plain
    version; several for a turn group). Each output line is
    ``{text, op: same|add|del, editor}`` where ``editor`` is the index into
    ``chain`` of the version that introduced an added line or removed a
    deleted one (None for ``same``).
    """
    base_lines = split_lines(base)
    # Current state while walking: (text, origin) where origin is
    # ("b", k) for base line k or ("v", i) for a line introduced by chain[i].
    state: list[tuple[str, tuple[str, int]]] = [(t, ("b", k)) for k, t in enumerate(base_lines)]
    deleted_by: dict[int, int] = {}
    for i, content in enumerate(chain):
        new_lines = split_lines(content)
        old_texts = [t for t, _ in state]
        nxt: list[tuple[str, tuple[str, int]]] = []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                None, old_texts, new_lines, autojunk=False).get_opcodes():
            if tag == "equal":
                nxt.extend(state[i1:i2])
                continue
            for _t, origin in state[i1:i2]:
                if origin[0] == "b":
                    deleted_by[origin[1]] = i
            nxt.extend((t, ("v", i)) for t in new_lines[j1:j2])
        state = nxt

    final_texts = [t for t, _ in state]
    last = len(chain) - 1
    out: list[dict] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, base_lines, final_texts, autojunk=False).get_opcodes():
        if tag == "equal":
            out.extend({"text": t, "op": "same", "editor": None} for t in base_lines[i1:i2])
            continue
        for k in range(i1, i2):
            out.append({"text": base_lines[k], "op": "del", "editor": deleted_by.get(k, last)})
        for j in range(j1, j2):
            origin = state[j][1]
            out.append({"text": final_texts[j], "op": "add",
                        "editor": origin[1] if origin[0] == "v" else last})
    return out
