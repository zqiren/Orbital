# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""2026-11-09 keyless smoke F4: a worker unchecked for a project
(``disabled_sub_agents``) is not offered there, and a send the user started
to it (the pin, a queue item / automation) is refused by
``SubAgentManager.send`` with an ``Error: …`` string before anything spawns.
The manager's own dispatches are not gated here (its list already omits it).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.sub_agent_manager import SubAgentManager


def _mgr():
    store = MagicMock()
    store.get_project.return_value = {"workspace": "/nowhere", "disabled_sub_agents": ["pi"]}
    mgr = SubAgentManager(process_manager=MagicMock(), project_store=store)
    # Stop at the spawn: the gate under test runs before it.
    mgr.start = AsyncMock(return_value="Error: spawn stubbed")
    return mgr


@pytest.mark.asyncio
@pytest.mark.parametrize("initiator", ["user_pinned", "queue_item"])
async def test_a_user_started_send_to_a_disabled_worker_is_refused(initiator):
    mgr = _mgr()
    mgr._start_failures[("p1", "pi")] = {"reason": "not_authenticated"}
    result = await mgr.send("p1", "pi", "hi", session_id="s1", initiator=initiator)
    assert result.startswith("Error:")
    assert "turned off" in result
    mgr.start.assert_not_awaited()
    # The refusal's chat row must not inherit an older failure's reason.
    assert mgr.last_start_failure_meta("p1", "pi") is None


@pytest.mark.asyncio
async def test_an_enabled_worker_is_not_gated():
    mgr = _mgr()
    await mgr.send("p1", "codex", "hi", session_id="s1", initiator="user_pinned")
    mgr.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_managers_own_dispatch_is_not_gated_here():
    mgr = _mgr()
    await mgr.send("p1", "pi", "hi", session_id="s1", initiator="management_agent")
    mgr.start.assert_awaited_once()
