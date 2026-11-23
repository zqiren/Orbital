# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Worker slugs whose manifest was retired, and the one that replaced each.

Applied where a session's stored pin is read (``Session.load`` and the
session-list row), so a chat pinned before the rename keeps working. Nothing
is rewritten on disk: the session file keeps the old slug, and an older
Orbital opens it unchanged. Drop an entry once no supported version can still
have written it.
"""

from __future__ import annotations

# Spec 193: the Gemini CLI seed manifest was replaced by Antigravity (v0.25).
RETIRED_SLUGS: dict[str, str] = {"gemini-cli": "antigravity"}


def current_slug(slug: str | None) -> str | None:
    """``slug``, or the slug that replaced it when its manifest was retired."""
    return RETIRED_SLUGS.get(slug, slug) if slug else slug
