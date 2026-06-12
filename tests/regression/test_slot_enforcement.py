# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Single-slot is enforced at the MANAGER level (start_agent), not just the
HTTP route — so every caller (HTTP, dispatcher, triggers, internal auto-start)
is covered.

A project has at most one active-loop slot. `start_agent` raises ValueError
("Slot held by session …") when a DIFFERENT session already holds it. The
guard sits right after the same-session re-entrancy guard, before any provider
construction, so a held slot is rejected cheaply.

The "succeeds" cases use a sentinel: `_provider_registry.get_model_info` (the
first call after the guard) is made to raise `_PastGuard`, so reaching it
proves the slot guard let execution through.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager


class _PastGuard(Exception):
    """Sentinel raised just past the slot guard to prove it was not rejected."""


def _manager() -> AgentManager:
    mgr = AgentManager(
        project_store=MagicMock(),
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )
    mgr._ws.broadcast = MagicMock()
    # start_agent(session_id=None) mints a real session uuid from the project
    # name since 433912a (seam 3 / D1) — give the store a real dict so
    # _mint_session_uuid's re.sub doesn't choke on a MagicMock name.
    mgr._project_store.get_project.return_value = {"name": "proj"}
    # No platform isolation → skip the capability block before the provider.
    mgr._platform_provider = None
    # First call after the guard raises the sentinel.
    mgr._provider_registry = MagicMock()
    mgr._provider_registry.get_model_info.side_effect = _PastGuard()
    return mgr


def _running_handle() -> MagicMock:
    h = MagicMock()
    h.session.is_stopped.return_value = False
    h.session._paused_for_approval = False
    h.task = MagicMock()
    h.task.done.return_value = False  # live loop → holds the slot
    return h


def _stale_handle() -> MagicMock:
    h = MagicMock()
    h.session.is_stopped.return_value = False
    h.session._paused_for_approval = False
    h.task = MagicMock()
    h.task.done.return_value = True  # done → does not hold the slot
    return h


def _config() -> MagicMock:
    c = MagicMock()
    c.api_key = ""
    c.llm_fallback_models = []
    c.utility_model = None
    return c


@pytest.mark.asyncio
async def test_cross_session_slot_guard_rejects_second_session():
    """Session A holds the slot; starting session B raises."""
    mgr = _manager()
    mgr._handles[("proj", "sess-A")] = _running_handle()
    with pytest.raises(ValueError, match="Slot held by session"):
        await mgr.start_agent("proj", _config(), session_id="sess-B")


@pytest.mark.asyncio
async def test_same_session_restart_allowed_when_stale():
    """A done (stale) handle for session A does not block restarting session A."""
    mgr = _manager()
    sk = ("proj", "sess-A")
    mgr._handles[sk] = _stale_handle()
    # Passes the slot guard (holder is None after stale cleanup) → reaches the sentinel.
    with pytest.raises(_PastGuard):
        await mgr.start_agent("proj", _config(), session_id="sess-A")
    assert sk not in mgr._handles  # stale handle cleaned up by the re-entrancy guard


@pytest.mark.asyncio
async def test_slot_free_after_no_holder():
    """With no holder, starting a fresh session passes the slot guard."""
    mgr = _manager()
    with pytest.raises(_PastGuard):
        await mgr.start_agent("proj", _config(), session_id="sess-B")


@pytest.mark.asyncio
async def test_session_id_none_does_not_bypass_slot_guard():
    """session_id=None (now mints a fresh session uuid, 433912a) is still subject to the slot guard."""
    mgr = _manager()
    mgr._handles[("proj", "sess-A")] = _running_handle()
    with pytest.raises(ValueError, match="Slot held by session"):
        await mgr.start_agent("proj", _config(), session_id=None)


@pytest.mark.asyncio
async def test_dispatcher_path_blocked_by_slot():
    """The dispatcher's inject_message → start_agent path hits the manager guard."""
    mgr = _manager()
    mgr._handles[("proj", "sess-A")] = _running_handle()
    # inject_message Case 3 builds config + checks disk before start_agent.
    mgr._build_agent_config_from_project = MagicMock(return_value=_config())
    mgr._load_session_from_disk = MagicMock(return_value=None)
    with pytest.raises(ValueError, match="Slot held by session"):
        await mgr.inject_message("proj", "do the thing", session_id="sess-B")
