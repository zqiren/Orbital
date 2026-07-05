# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared workspace-aware path resolution for the path-taking tools.

Five tools (read/write/edit/glob/grep) used to each carry their own copy of a
private `_resolve_safe` helper. All five had the same bugs:

  * `path.lstrip("/")` + `os.path.join(workspace, path)` — when an agent emits
    an absolute path that already points inside the workspace, this strips
    the leading slash and concatenates again, doubling the workspace prefix.
    Writes silently land at a wrong path; reads return spurious "not found".
  * `resolved.startswith(self._workspace)` without an `os.sep` guard — a
    sibling like `/home/user-foo` shares the string prefix `/home/user` but
    is not inside it.

`resolve_safe(workspace, path)` returns the realpath of `path` interpreted
relative to `workspace`, or `None` if the resolved path falls outside the
workspace. Absolute paths that already point inside the workspace are passed
through (after realpath) without any concatenation. Callers should treat a
`None` return as "outside workspace, reject."
"""

import os


def resolve_safe(workspace: str, path: str) -> str | None:
    """Resolve `path` against `workspace`, returning None if it escapes.

    Behaviour:
      * Absolute `path` inside `workspace` → return its realpath (pass through).
      * Absolute `path` outside `workspace` → return None.
      * Relative `path` → join against workspace realpath, then realpath again
        (resolving symlinks and `..`); return None if the result escapes.
      * `path == "."` → workspace realpath itself.
    """
    workspace_real = os.path.realpath(workspace)

    if os.path.isabs(path):
        candidate = os.path.realpath(path)
    else:
        candidate = os.path.realpath(os.path.join(workspace_real, path))

    # The `+ os.sep` guard is essential: without it, `/home/user-foo` would
    # be accepted as "inside" `/home/user` because the string starts with it.
    if candidate == workspace_real or candidate.startswith(workspace_real + os.sep):
        return candidate
    return None


# Runtime/internal subtrees excluded from CROSS-PROJECT (secondary-root) reads.
# A project's ``orbital/`` holds agent internals (memory, sessions,
# credentials-adjacent material) and ``.git/`` its VCS internals — reference
# means *work product*, not another project's machinery (Spec 12 §3, §7-Q2).
# The primary root (the reader's own workspace) is NOT subject to these.
_SECONDARY_EXCLUDED_COMPONENTS = frozenset({"orbital", ".git"})


def _excluded_in_secondary(candidate_real: str, root_real: str) -> bool:
    """True if ``candidate_real`` (inside ``root_real``) sits under an excluded subtree."""
    rel = os.path.relpath(candidate_real, root_real)
    return any(part in _SECONDARY_EXCLUDED_COMPONENTS for part in rel.split(os.sep))


def resolve_safe_read(roots: list[str], path: str) -> str | None:
    """Resolve ``path`` for READ against a list of roots, or None if it escapes.

    ``roots[0]`` is the primary workspace; the rest are read-only reference
    roots (other projects). Semantics:

      * Relative ``path`` resolves against the PRIMARY root only (never a
        secondary) — same containment as :func:`resolve_safe`.
      * Absolute ``path`` passes if it lands inside ANY root (realpath + the
        ``os.sep`` guard). Inside a SECONDARY root, paths under ``orbital/`` or
        ``.git/`` are rejected (agent internals, not work product).

    With a single-element ``roots`` list this is byte-identical to
    ``resolve_safe(roots[0], path)`` — normal projects are unaffected.
    """
    if not roots:
        return None
    primary_real = os.path.realpath(roots[0])

    if not os.path.isabs(path):
        candidate = os.path.realpath(os.path.join(primary_real, path))
        if candidate == primary_real or candidate.startswith(primary_real + os.sep):
            return candidate
        return None

    candidate = os.path.realpath(path)
    # Primary root first — no exclusions (it's the reader's own workspace).
    if candidate == primary_real or candidate.startswith(primary_real + os.sep):
        return candidate
    # Secondary roots — allow if contained, minus excluded internal subtrees.
    for root in roots[1:]:
        root_real = os.path.realpath(root)
        if candidate == root_real or candidate.startswith(root_real + os.sep):
            if _excluded_in_secondary(candidate, root_real):
                return None
            return candidate
    return None
