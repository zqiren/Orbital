# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Step 1 — lifecycle endpoints thread session_id to the holding session.

The recovery bug (deny/new-session don't unstick a session keyed under anything
but "default") is because these endpoints never passed session_id. These tests
lock in that each endpoint forwards a caller-supplied session_id to the
underlying AgentManager method, so the UI can act on the session it addresses
(not just the default sentinel). Plus the AgentLoop.run() re-entrancy guard.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.loop import AgentLoop
from agent_os.api.routes import agents_v2
from agent_os.api.routes.agents_v2 import (
    ApproveRequest,
    DenyRequest,
    SessionScopedRequest,
    approve,
    cancel_message,
    deny,
    new_session,
)


class _Globals:
    """Snapshot/restore agents_v2 module globals around a test."""

    def __init__(self, **patches):
        self._patches = patches
        self._orig = {}

    def __enter__(self):
        for k, v in self._patches.items():
            self._orig[k] = getattr(agents_v2, k)
            setattr(agents_v2, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self._orig.items():
            setattr(agents_v2, k, v)


def _mgr():
    m = MagicMock()
    m.new_session = AsyncMock(return_value={"status": "ok"})
    m.cancel_message = AsyncMock(return_value={"status": "cancelled"})
    m.approve = AsyncMock()
    m.deny = AsyncMock()
    return m


@pytest.mark.asyncio
async def test_new_session_forwards_session_id():
    m = _mgr()
    with _Globals(_agent_manager=m, _ws_manager=MagicMock()):
        await new_session("proj", SessionScopedRequest(session_id="sess_abc"))
    assert m.new_session.await_args.kwargs.get("session_id") == "sess_abc"


@pytest.mark.asyncio
async def test_cancel_forwards_session_id():
    m = _mgr()
    with _Globals(_agent_manager=m, _ws_manager=MagicMock()):
        await cancel_message("proj", SessionScopedRequest(session_id="sess_abc"))
    assert m.cancel_message.await_args.kwargs.get("session_id") == "sess_abc"


# NOTE: test_stop_forwards_session_id removed — the POST /stop route was removed
# (stop_agent is now an internal AgentManager method, not an HTTP endpoint). The
# route-removed coverage lives in tests/integration/test_stop_removed.py.


@pytest.mark.asyncio
async def test_approve_forwards_session_id():
    m = _mgr()
    with _Globals(_agent_manager=m, _ws_manager=MagicMock(), _sub_agent_manager=None):
        await approve("proj", ApproveRequest(tool_call_id="t", session_id="sess_abc"))
    assert m.approve.await_args.kwargs.get("session_id") == "sess_abc"


@pytest.mark.asyncio
async def test_deny_forwards_session_id():
    m = _mgr()
    with _Globals(_agent_manager=m, _ws_manager=MagicMock(), _sub_agent_manager=None):
        await deny("proj", DenyRequest(tool_call_id="t", reason="no", session_id="sess_abc"))
    assert m.deny.await_args.kwargs.get("session_id") == "sess_abc"


@pytest.mark.asyncio
async def test_run_reentrancy_guard_raises():
    """A second concurrent run() on the same loop must be rejected, not silently
    start a second loop writing the same session."""
    loop = AgentLoop.__new__(AgentLoop)
    loop._running = True
    loop._session = MagicMock(session_id="s")
    with pytest.raises(RuntimeError):
        await loop.run()
