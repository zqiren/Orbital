# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared resolver for which sub-agents an agent is TOLD it can delegate to.

A project's ``disabled_sub_agents`` denylist is the canonical source of truth
for hiding a sub-agent (e.g. an installed-but-unwanted Gemini CLI). It must be
honoured everywhere a management or peer agent is told about available
sub-agents — including the auto-detect branch used when no explicit allowlist
is configured. Centralising the logic here keeps the orchestrator-prompt path
(agent_manager) and the peer-list path (sub_agent_manager) consistent.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


def resolve_visible_sub_agent_slugs(
    *,
    enabled_sub_agents: Optional[Sequence[str]],
    disabled_sub_agents: Optional[Sequence[str]],
    setup_engine,
    enabled_agents_legacy: Optional[Sequence[str]] = None,
) -> list[str]:
    """Return the sub-agent slugs that should be visible to an agent.

    Resolution order for the candidate set:
      1. ``enabled_sub_agents`` (explicit allowlist), else
      2. ``enabled_agents_legacy`` (legacy field), else
      3. auto-detect: every installed, non-``built-in`` sub-agent reported by
         ``setup_engine.check_all()``.

    The project's ``disabled_sub_agents`` denylist is then subtracted from the
    candidate set in ALL cases — this is the fix for the auto-detect branch
    that previously ignored the denylist.
    """
    slugs = list(enabled_sub_agents or enabled_agents_legacy or [])
    if not slugs and setup_engine is not None:
        try:
            slugs = [
                s.slug
                for s in setup_engine.check_all()
                if getattr(s, "installed", False) and s.slug != "built-in"
            ]
        except Exception:
            logger.warning("check_all() failed while resolving sub-agents", exc_info=True)
            slugs = []
    disabled = set(disabled_sub_agents or [])
    return [s for s in slugs if s not in disabled]
