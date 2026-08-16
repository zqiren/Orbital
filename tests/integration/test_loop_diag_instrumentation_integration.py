# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration test for loop diagnostic instrumentation across the real
AgentLoop ↔ AgentManager._on_loop_done seam.

Reproduces the original "silent death" class: a turn whose stream raises an
uncaught (non-LLMError) exception mid-flight. Pre-instrumentation this produced
NO row, NO terminal log, and NO daemon-log error. With instrumentation it must
be visible at BOTH layers on the real callback path:

  - loop layer (Point 3): a `run_terminal` diag with disposition=error,
    rows_appended=0 (the silent fingerprint), emitted from run()'s finally.
  - manager layer (Point 4): AgentManager._on_loop_done logs the exception
    type/message/traceback (real callback, not a mock).

This is the seam the unit tests cover separately; here they fire together on a
real loop Task wired to the real done-callback.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.loop import AgentLoop
from agent_os.agent.session import Session, persist_user_row
from agent_os.agent.context import ContextManager
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.prompt_builder import PromptContext, Autonomy
from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle


DIAG_LOGGER = "agent.diag"
MGR_LOGGER = "agent_os.daemon_v2.agent_manager"


class _BoomProvider:
    """Stream raises an uncaught RuntimeError mid-flight (not LLMError, not
    CancelledError) — the exact unhandled-exception shape."""

    async def stream(self, messages, tools=None):
        raise RuntimeError("simulated mid-stream blowup")
        yield  # pragma: no cover — makes this an async generator

    async def complete(self, messages, tools=None, *, disable_reasoning=False):
        raise RuntimeError("simulated mid-stream blowup")


class _Builder:
    def build(self, context):
        return ("cached-prefix", "semi-stable", "dynamic")


class _Registry:
    def schemas(self):
        return []

    def is_async(self, name):
        return False

    def execute(self, name, arguments):
        return ToolResult(content="x")

    def tool_names(self):
        return []

    def reset_run_state(self):
        pass


def _ctx(ws):
    return PromptContext(
        workspace=ws, model="test-model", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[], tool_names=[], os_type="linux",
        datetime_now="2026-01-01T00:00:00", context_usage_pct=0.0,
    )


def _make_manager():
    sub = MagicMock()
    sub.list_active = MagicMock(return_value=[])
    sub.stop_all = AsyncMock()
    mgr = AgentManager(
        project_store=MagicMock(), ws_manager=MagicMock(),
        sub_agent_manager=sub, activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )
    return mgr


@pytest.mark.asyncio
async def test_uncaught_exception_visible_at_both_layers(tmp_path, caplog):
    caplog.set_level(logging.INFO)  # captures agent.diag (INFO) + manager ERROR

    session = Session.new("proj_diagsess01", str(tmp_path))
    loop = AgentLoop(session, _BoomProvider(), _Registry(),
                     ContextManager(session, _Builder(), _ctx(str(tmp_path))))
    mgr = _make_manager()

    # Wire the REAL done-callback, exactly as agent_manager._start_loop does.
    persist_user_row(loop._session, "trigger blowup")
    task = asyncio.ensure_future(loop.run())
    task.add_done_callback(mgr._on_loop_done("proj", session_id="proj_diagsess01"))

    results = await asyncio.gather(task, return_exceptions=True)
    # The loop task surfaced the exception (run() has no except — by design).
    assert isinstance(results[0], RuntimeError)
    # Let the done-callback run.
    await asyncio.sleep(0)

    # Point 3 — loop emitted run_terminal disposition=error with 0 rows appended.
    terminals = [r.diag for r in caplog.records
                 if r.name == DIAG_LOGGER and hasattr(r, "diag")
                 and r.diag.get("event") == "run_terminal"]
    assert len(terminals) == 1, f"expected one run_terminal, got {terminals}"
    assert terminals[0]["disposition"] == "error"
    assert terminals[0]["rows_appended"] == 0

    # Point 4 — manager logged the exception type + message (real callback).
    mgr_errors = [r for r in caplog.records
                  if r.name == MGR_LOGGER and r.levelno >= logging.ERROR
                  and "RuntimeError" in r.getMessage()
                  and "simulated mid-stream blowup" in r.getMessage()]
    assert mgr_errors, "expected _on_loop_done to log the exception"
