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
