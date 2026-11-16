# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Atomic replace-a-file writes for the daemon's small JSON stores.

A store that rewrites its file with ``open(path, "w")`` truncates it first:
for the length of the write, and for ever after a crash in the middle of it,
the file is empty or half-written. Every store here answers an unparseable
file with its defaults, and its next save makes those defaults permanent
(the 2026-11-16 settings wipe). Writing a temp file in the same folder,
fsyncing it and ``os.replace``-ing it into place removes that window: a
reader sees the old file or the new one, and an interrupted write leaves the
old file exactly as it was.

The caller serialises (each store keeps its own JSON options, so file formats
do not change); this module only puts the text on disk.
"""

from __future__ import annotations

import os
import stat
import sys
import time
import uuid
from contextlib import suppress

# How long a write keeps retrying a replace the OS refuses. Windows refuses
# to replace a file another handle holds open (a sharing violation, surfaced
# as PermissionError) — an antivirus scan or an indexer, gone in
# milliseconds. A truncating write is never the fallback.
REPLACE_RETRY_S = 2.0

TMP_SUFFIX = ".tmp"


def tmp_prefix(path: str) -> str:
    """The name prefix of ``path``'s temp files (hidden, same folder)."""
    return "." + os.path.basename(path) + "."


def _replace(src: str, dst: str) -> None:
    """``os.replace``, retried briefly on PermissionError (Windows)."""
    deadline = time.monotonic() + REPLACE_RETRY_S
    while True:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def atomic_write_text(path: str, text: str, encoding: str | None = "utf-8") -> None:
    """Replace ``path`` with ``text`` atomically.

    Text mode, like the ``open(path, "w")`` it replaces, so the bytes are
    identical on every platform (CRLF on Windows, as before). A new file gets
    the mode ``open`` would have given it (0666 minus the umask); an existing
    file keeps its mode. On any failure the temp file is removed and the
    original is untouched.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, f"{tmp_prefix(path)}{uuid.uuid4().hex[:12]}{TMP_SUFFIX}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(tmp, flags, 0o666)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        with suppress(OSError):
            os.chmod(tmp, stat.S_IMODE(os.stat(path).st_mode))
        _replace(tmp, path)
    except BaseException:
        with suppress(OSError):
            os.remove(tmp)
        raise
    if sys.platform != "win32":
        # Make the rename itself durable. Best effort: not every filesystem
        # lets a directory be opened for fsync.
        with suppress(OSError):
            dfd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)


def sweep_stray_temps(path: str, older_than_s: float = 60.0) -> None:
    """Remove temp files a crashed write of ``path`` left behind. Readers
    never look at them; this only keeps the folder tidy."""
    directory = os.path.dirname(path) or "."
    prefix = tmp_prefix(path)
    try:
        names = os.listdir(directory)
    except OSError:
        return
    now = time.time()
    for name in names:
        if not (name.startswith(prefix) and name.endswith(TMP_SUFFIX)):
            continue
        p = os.path.join(directory, name)
        with suppress(OSError):
            if now - os.path.getmtime(p) > older_than_s:
                os.remove(p)
