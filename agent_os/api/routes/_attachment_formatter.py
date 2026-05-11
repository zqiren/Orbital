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
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Sequence

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


def validate_attachments(
    workspace: str,
    attachments: "Sequence[InjectAttachment]",
) -> None:
    """Confirm each attachment exists inside the workspace and matches size.

    Stops at the first failure and raises ``ValueError`` with a message
    naming the offending path. Mirrors the realpath confinement check used
    by ``agent_os.agent.tools.browser._resolve_upload_path`` so symlink
    escapes are rejected.
    """
    workspace_real = os.path.realpath(workspace)
    for att in attachments:
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
        lines.append(f"- {a.path} ({a.mime}, {_human_size(a.size)})")
    lines.append("</attached_files>")
    return "\n".join(lines) + "\n\n"
