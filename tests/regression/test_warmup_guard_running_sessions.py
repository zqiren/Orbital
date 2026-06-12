# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: POST /browser/warmup running-session guard + close-then-launch.

TASK-browser-signin-fallback.md: the Chromium profile dir is locked by
whichever instance holds it. The daemon's headless context, once lazily
launched, holds the lock until daemon shutdown — so the warmup cannot launch
while it lives. The route must therefore:

  (i)   refuse with 409 while ANY session has run status other than idle,
        WITHOUT touching the daemon browser context;
  (ii)  when all sessions are idle and the daemon context is live, close it
        (close_for_handoff) BEFORE launching warmup — releasing the lock;
  (iii) keep working when no context was ever launched (the setup-wizard-time
        case), where close_for_handoff is a no-op.

The guard is on running sessions, NOT on context existence — checking
"context exists" would refuse forever after first browser use.
"""

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.routes.platform import router, configure
from agent_os.daemon_v2.agent_manager import AgentManager, make_session_key
from agent_os.daemon_v2.browser_manager import BrowserManager

GUARD_DETAIL = "A session is currently running. Stop running sessions, then try again."


class _BM(BrowserManager):
    """BrowserManager with the real warmup_active property re-declared.

    Other test modules (test_api_smoke_bugfixes, test_browser_warmup_endpoint)
    assign exhausted PropertyMocks onto the BrowserManager CLASS and never
    restore them, so a real instance's warmup_active read raises StopIteration
    when those files run first. Shadowing the property here makes these tests
    order-independent.
    """

    @property
    def warmup_active(self) -> bool:
        return self._warmup_active


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _make_agent_manager() -> AgentManager:
    """Real AgentManager (the route reads its real _handles/list_sessions),
    with mocked collaborators — no loops actually run."""
    return AgentManager(
        project_store=MagicMock(),
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )


def _handle(running: bool) -> MagicMock:
    """Fake ProjectHandle whose session reads as running or idle."""
    h = MagicMock()
    h.session.is_stopped.return_value = False
    h.session._paused_for_approval = False
    h.task.done.return_value = not running
    return h


def _make_client(agent_manager, browser_manager) -> TestClient:
    configure(MagicMock(), agent_manager=agent_manager,
              browser_manager=browser_manager)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ----------------------------------------------------------------------
# (i) Non-idle session anywhere -> 409, daemon context untouched
# ----------------------------------------------------------------------

def test_warmup_refused_409_while_session_running(tmp_path):
    am = _make_agent_manager()
    # One idle session in proj-a, one RUNNING session in proj-b: the guard
    # must trip on ANY non-idle session across all projects.
    am._handles[make_session_key("proj-a", "sess-1")] = _handle(running=False)
    am._handles[make_session_key("proj-b", "sess-2")] = _handle(running=True)

    bm = _BM(profile_dir=str(tmp_path / "profile"), headless=True)
    calls = []

    async def tracked_close():
        calls.append("close_for_handoff")

    async def tracked_launch(url):
        calls.append("launch_warmup")

    bm.close_for_handoff = tracked_close
    bm.launch_warmup = tracked_launch

    client = _make_client(am, bm)
    resp = client.post("/api/v2/platform/browser/warmup")

    assert resp.status_code == 409
    assert resp.json()["detail"] == GUARD_DETAIL
    assert calls == [], (
        "409 must be raised BEFORE close_for_handoff/launch_warmup — the "
        f"daemon context must not be touched, got calls: {calls}"
    )


# ----------------------------------------------------------------------
# (ii) All sessions idle + daemon context live -> close BEFORE launch
# ----------------------------------------------------------------------

def test_warmup_closes_live_context_before_launch(tmp_path):
    am = _make_agent_manager()
    am._handles[make_session_key("proj-a", "sess-1")] = _handle(running=False)

    bm = _BM(profile_dir=str(tmp_path / "profile"), headless=True)
    # Simulate the daemon's headless context already lazily launched (it
    # holds the profile SingletonLock).
    live_ctx = AsyncMock()
    live_pw = MagicMock()
    live_pw.stop = AsyncMock()
    bm._context = live_ctx
    bm._browser = MagicMock()
    bm._playwright = live_pw

    order = []
    real_close = bm.close_for_handoff

    async def tracked_close():
        order.append("close_for_handoff")
        await real_close()

    async def tracked_launch(url):
        order.append("launch_warmup")
        bm._warmup_active = True

    bm.close_for_handoff = tracked_close
    bm.launch_warmup = tracked_launch

    client = _make_client(am, bm)
    resp = client.post("/api/v2/platform/browser/warmup")

    assert resp.status_code == 200
    assert resp.json()["status"] == "launched"
    assert order == ["close_for_handoff", "launch_warmup"], (
        f"daemon context must be closed BEFORE warmup launches, got: {order}"
    )
    # The real close_for_handoff ran: lock released, manager relaunchable.
    live_ctx.close.assert_awaited()
    live_pw.stop.assert_awaited()
    assert bm._context is None
    assert bm._playwright is None


# ----------------------------------------------------------------------
# (iii) No context ever launched -> still works (wizard-time equivalence)
# ----------------------------------------------------------------------

def test_warmup_works_when_context_never_launched(tmp_path):
    am = _make_agent_manager()  # no sessions at all

    bm = _BM(profile_dir=str(tmp_path / "profile"), headless=True)
    order = []
    real_close = bm.close_for_handoff

    async def tracked_close():
        order.append("close_for_handoff")
        await real_close()  # must be a clean no-op

    async def tracked_launch(url):
        order.append("launch_warmup")
        bm._warmup_active = True

    bm.close_for_handoff = tracked_close
    bm.launch_warmup = tracked_launch

    client = _make_client(am, bm)
    resp = client.post("/api/v2/platform/browser/warmup")

    assert resp.status_code == 200
    assert resp.json()["status"] == "launched"
    assert order == ["close_for_handoff", "launch_warmup"]
