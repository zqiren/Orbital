# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Helpers for formatting and validating chat-message attachments.

These helpers are deliberately FastAPI-free so they can be unit tested
without any app setup. The inject route imports both functions and calls
them in order: validate → format → prepend to user content.

Approach 1 (path-references only, no native multimodal): the agent reads
the attached file via its existing `read` tool when it needs the contents.
That keeps the feature uniform across all four sub-agent transports.

Two kinds of attachment share the block: a file uploaded into the
workspace's ``uploads/`` (browser, phone, paste) and, in the desktop app, a
file attached by its absolute local path (nothing copied, no size cap),
whose line ends with ``LOCAL_FILE_MARKER``.
"""

from __future__ import annotations

import mimetypes
import os
import stat
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from agent_os.daemon_v2.attachment_grants import LOCAL_FILE_MARKER

if TYPE_CHECKING:  # avoid runtime circular import with agents_v2
    from agent_os.api.routes.agents_v2 import InjectAttachment


def _human_size(bytes_count: int) -> str:
    """Render a byte count using binary (1024-based) units.

    Binary units match the upload cap (10 * 1024 * 1024).
    """
    if bytes_count < 1024:
        return f"{bytes_count} B"
    if bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.1f} KB"
    if bytes_count < 1024 * 1024 * 1024:
        return f"{bytes_count / (1024 * 1024):.1f} MB"
    return f"{bytes_count / (1024 * 1024 * 1024):.1f} GB"


@dataclass(frozen=True)
class ResolvedAttachment:
    """One validated attachment, as the prefix and the JSONL row show it.

    ``local`` marks a file attached by its absolute path in the desktop app:
    ``path`` is then its realpath and ``size`` the size on disk.
    """
    path: str
    mime: str
    size: int
    local: bool = False

    def as_row(self) -> dict:
        """The JSONL ``attachments`` entry. An uploaded file keeps today's
        three keys exactly; only a local file carries ``local``."""
        row = {"path": self.path, "mime": self.mime, "size": self.size}
        if self.local:
            row["local"] = True
        return row


def resolve_local_attachment(path: str) -> str:
    """Validate a local-file attachment and return its realpath.

    The path must be absolute and resolve (symlinks followed) to an existing
    regular file. No size cap: nothing is copied.
    """
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ValueError("local file path is empty or invalid")
    if not os.path.isabs(path):
        raise ValueError(f"local file path must be absolute: {path}")
    real = os.path.realpath(path)
    try:
        st = os.stat(real)
    except OSError:
        raise ValueError(f"file not found: {path}")
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"not a regular file: {path}")
    return real


def validate_attachments(
    workspace: str,
    attachments: "Sequence[InjectAttachment]",
) -> "list[ResolvedAttachment]":
    """Validate each attachment and return them as the prefix should show them.

    An uploaded file (the default) must exist inside the workspace and match
    its declared size. A ``local`` one (desktop app, attached by path) goes
    through :func:`resolve_local_attachment`; its realpath and on-disk size
    replace what the client sent.

    Stops at the first failure and raises ``ValueError`` with a message
    naming the offending path. Mirrors the realpath confinement check used
    by ``agent_os.agent.tools.browser._resolve_upload_path`` so symlink
    escapes are rejected.
    """
    workspace_real = os.path.realpath(workspace)
    out: list[ResolvedAttachment] = []
    for att in attachments:
        if getattr(att, "local", False):
            real = resolve_local_attachment(att.path)
            out.append(ResolvedAttachment(
                path=real, mime=att.mime, size=os.path.getsize(real), local=True))
            continue
        resolved = os.path.realpath(os.path.join(workspace, att.path))
        if not (
            resolved == workspace_real
            or resolved.startswith(workspace_real + os.sep)
        ):
            raise ValueError(
                f"path resolves outside workspace: {att.path}"
            )
        if not os.path.isfile(resolved):
            raise ValueError(f"file not found: {att.path}")
        actual = os.path.getsize(resolved)
        if actual != att.size:
            raise ValueError(
                f"size mismatch for {att.path}: "
                f"declared {att.size}, actual {actual}"
            )
        out.append(ResolvedAttachment(path=att.path, mime=att.mime, size=att.size))
    return out


def format_prefix(attachments: "Sequence[InjectAttachment]") -> str:
    """Build the ``<attached_files>...</attached_files>`` block.

    - Empty list → ``""``
    - Non-empty list → XML-tagged block with one bullet line per attachment

    Output for a non-empty list always ends with exactly ``\\n\\n`` so the
    user's content lands on its own paragraph below the prefix.
    """
    if not attachments:
        return ""

    lines = ["<attached_files>"]
    for a in attachments:
        if getattr(a, "local", False):
            # Desktop app: the user's own file, read in place. The agent
            # reads it at this absolute path (every agent is granted it).
            lines.append(f"- {a.path} ({a.mime}, {_human_size(a.size)}, "
                         f"{LOCAL_FILE_MARKER})")
        else:
            lines.append(f"- {a.path} ({a.mime}, {_human_size(a.size)})")
    lines.append("</attached_files>")
    return "\n".join(lines) + "\n\n"


@dataclass(frozen=True)
class _RefAttachment:
    """Lightweight stand-in for InjectAttachment built from a bare path ref."""
    path: str
    mime: str
    size: int


def format_prefix_from_refs(workspace: str, file_refs: "Sequence[str]") -> str:
    """Build the ``<attached_files>`` block from bare path references.

    Queue items persist only file paths (``file_refs``), not the rich
    InjectAttachment the chat composer sends. Derive mime/size by stat-ing each
    path under the workspace, then reuse ``format_prefix`` so a dispatched queue
    item carries the SAME block format a user's Send produces (attachment
    symmetry). Missing files degrade to size 0 rather than failing dispatch.
    """
    if not file_refs:
        return ""
    atts: list[_RefAttachment] = []
    for ref in file_refs:
        resolved = os.path.join(workspace, ref) if workspace else ref
        try:
            size = os.path.getsize(resolved)
        except OSError:
            size = 0
        mime = mimetypes.guess_type(ref)[0] or "application/octet-stream"
        atts.append(_RefAttachment(path=ref, mime=mime, size=size))
    return format_prefix(atts)
