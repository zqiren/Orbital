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

from agent_os.agents.setup_types import handshake_failed

logger = logging.getLogger(__name__)


def worker_ready(status) -> bool:
    """A worker a send could reach right now: installed, dependencies met,
    signed in (``credentials_configured``), not below the version floor, and
    not failing its handshake — the dispatch gate's own test. Identity checks
    (``is True`` / ``is False``) so a mocked engine never trips it. The one
    readiness test for "who can this project's work go to": the pin list
    (``AgentManager.available_worker_slugs``) and the dispatch tool's agent
    list (``resolve_visible_sub_agent_slugs(ready_only=True)``)."""
    slug = getattr(status, "slug", None)
    if not isinstance(slug, str) or not slug or slug == "built-in":
        return False
    if getattr(status, "installed", False) is not True:
        return False
    if getattr(status, "dependencies_met", True) is False:
        return False
    if getattr(status, "credentials_configured", True) is False:
        return False
    if getattr(status, "update_needed", False) is True:
        return False
    return not handshake_failed(status)


def resolve_visible_sub_agent_slugs(
    *,
    enabled_sub_agents: Optional[Sequence[str]],
    disabled_sub_agents: Optional[Sequence[str]],
    setup_engine,
    enabled_agents_legacy: Optional[Sequence[str]] = None,
    ready_only: bool = False,
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

    ``ready_only`` (2026-11-09) also drops a worker that ``worker_ready``
    says cannot run now (not signed in, too old, …), so an agent is never
    told it can hand work to one. Without an engine answer the list is kept.
    """
    slugs = list(enabled_sub_agents or enabled_agents_legacy or [])
    statuses = None
    if setup_engine is not None and (ready_only or not slugs):
        try:
            statuses = list(setup_engine.check_all())
        except Exception:
            logger.warning("check_all() failed while resolving sub-agents", exc_info=True)
            statuses = None
    if not slugs and statuses is not None:
        slugs = [
            s.slug
            for s in statuses
            if getattr(s, "installed", False) and s.slug != "built-in"
        ]
    if ready_only and statuses is not None:
        ready = {s.slug for s in statuses if worker_ready(s)}
        slugs = [s for s in slugs if s in ready]
    disabled = set(disabled_sub_agents or [])
    return [s for s in slugs if s not in disabled]
