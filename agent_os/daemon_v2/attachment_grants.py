# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Read grants for files the user attached by their local path.

In the desktop app the composer attaches a file by its absolute path instead
of uploading a copy. Attaching it is the user's deliberate choice, so every
agent in that project may READ exactly that file: the manager's read tool in
safe mode, and each worker whose CLI would otherwise ask (Claude Code's
permission callback, an ACP agent's permission request, the ``claude -p``
fallback's ``--add-dir``). Nothing here widens write access.

The grant is the attached file itself (its realpath), held per project in
memory. It is recorded by the inject route the moment the message is
accepted, and re-read from a session's own ``<attached_files>`` rows when a
session is loaded after a daemon restart (``hydrate_from_messages``), so no
new file is written to disk.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Iterable

# The marker the inject route writes after the size on a local-file line of
# the ``<attached_files>`` block (``_attachment_formatter.format_prefix``).
LOCAL_FILE_MARKER = "the user's local file, not copied"

_BLOCK_RE = re.compile(r"^<attached_files>\n([\s\S]*?)\n</attached_files>")

_lock = threading.Lock()
_grants: dict[str, set[str]] = {}
_hydrated: set[tuple[str, str]] = set()


def grant(project_id: str, paths: Iterable[str]) -> None:
    """Let every agent of ``project_id`` read each of ``paths`` (realpaths)."""
    reals = {os.path.realpath(p) for p in paths if p}
    if not project_id or not reals:
        return
    with _lock:
        _grants.setdefault(project_id, set()).update(reals)


def granted(project_id: str) -> frozenset[str]:
    """The realpaths granted in ``project_id`` (empty when none)."""
    with _lock:
        return frozenset(_grants.get(project_id, ()))


def is_granted(project_id: str, path: str) -> bool:
    """True when ``path`` resolves to a file attached in ``project_id``."""
    if not project_id or not isinstance(path, str) or not path:
        return False
    if not os.path.isabs(path):
        return False
    real = os.path.realpath(path)
    with _lock:
        return real in _grants.get(project_id, ())


def local_paths_in(content: str) -> list[str]:
    """The local-file paths listed in a message's leading attachments block."""
    if not isinstance(content, str):
        return []
    m = _BLOCK_RE.match(content)
    if not m:
        return []
    out: list[str] = []
    suffix = f", {LOCAL_FILE_MARKER})"
    for line in m.group(1).split("\n"):
        line = line.strip()
        if not line.startswith("- ") or not line.endswith(suffix):
            continue
        body = line[2:]
        paren = body.rfind(" (")
        if paren <= 0:
            continue
        path = body[:paren]
        if os.path.isabs(path):
            out.append(path)
    return out


def hydrate_from_messages(project_id: str, session_id: str | None,
                          messages: Iterable[dict]) -> None:
    """Re-grant the local files a session's user rows attached (once per
    session per daemon run). Files that are gone are skipped."""
    key = (project_id, session_id or "")
    with _lock:
        if key in _hydrated:
            return
        _hydrated.add(key)
    paths: list[str] = []
    for msg in messages or ():
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        for p in local_paths_in(msg.get("content") or ""):
            if os.path.isfile(p):
                paths.append(p)
    grant(project_id, paths)


def reset() -> None:
    """Forget every grant (tests)."""
    with _lock:
        _grants.clear()
        _hydrated.clear()
