# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""When the memory editor runs (spec 089 §4.2).

Trigger: a live Layer-1 file over its soft budget, noticed where the files
are already measured — ``ContextManager.prepare()`` — so a file written by an
external agent is caught too. Single-flight per project, background,
debounced, and gated on "something changed since the last pass and the last
pass is old enough". The every-50-iterations turn-count trigger is gone.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

import agent_os.agent.loop as loop_mod
from agent_os.agent import memory_editor as E
from agent_os.agent.context import ContextManager
from agent_os.agent.loop import AgentLoop, REFRESH_DEBOUNCE_S
from agent_os.agent.prompt_builder import Autonomy, PromptContext
from agent_os.agent.session import Session
from agent_os.agent.workspace_files import WorkspaceFileManager


class _Builder:
    def build(self, context):
        return ("prefix", "semi", "dynamic")


def _ctx_manager(workspace: str) -> ContextManager:
    session = Session.new("trigger-test", workspace)
    ctx = PromptContext(
        workspace=workspace, model="m", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[], tool_names=[], os_type="linux",
        datetime_now="2026-09-19T00:00:00", context_usage_pct=0.0,
    )
    return ContextManager(session, _Builder(), ctx)


def _big_state():
    return "## Now\n" + "\n".join(f"- line {i} " + "s" * 400 for i in range(25)) + "\n"


@pytest.fixture(autouse=True)
def _clean_registries():
    loop_mod._PROJECT_REFRESH.clear()
    E._RUNNING.clear()
    yield
    loop_mod._PROJECT_REFRESH.clear()
    E._RUNNING.clear()


# ---------------------------------------------------------------------------
# prepare() → hook
# ---------------------------------------------------------------------------

def test_prepare_reports_over_budget_files_to_the_hook(tmp_path):
    WorkspaceFileManager(str(tmp_path)).write("state", _big_state())
    cm = _ctx_manager(str(tmp_path))
    seen = []
    cm._on_memory_over_budget = seen.append
    cm.prepare()
    assert seen == [["state"]]


def test_prepare_does_not_call_the_hook_under_budget(tmp_path):
    WorkspaceFileManager(str(tmp_path)).write("state", "## Now\n- small\n")
    cm = _ctx_manager(str(tmp_path))
    seen = []
    cm._on_memory_over_budget = seen.append
    cm.prepare()
    assert seen == []


def test_a_failing_hook_never_breaks_prepare(tmp_path):
    WorkspaceFileManager(str(tmp_path)).write("state", _big_state())
    cm = _ctx_manager(str(tmp_path))

    def boom(_keys):
        raise RuntimeError("scheduler exploded")

    cm._on_memory_over_budget = boom
    assert cm.prepare()          # still builds the context


def test_hygiene_text_says_the_tidy_is_automatic(tmp_path):
    WorkspaceFileManager(str(tmp_path)).write("state", _big_state())
    cm = _ctx_manager(str(tmp_path))
    messages = cm.prepare()
    tail = messages[-1]["content"]
    assert "[MEMORY HYGIENE]" in tail
    assert "automatically" in tail
    assert "call the checkpoint_state tool" not in tail


# ---------------------------------------------------------------------------
# the loop's over-budget scheduler
# ---------------------------------------------------------------------------

def _loop(project_dir, calls, *, gate=None, uuid="sess"):
    async def refresh(trigger):
        calls.append((uuid, trigger))
        if gate is not None:
            await gate.wait()
        return "edited"

    session = MagicMock()
    session.session_uuid = uuid
    session._stopped = False
    session.is_stopped.side_effect = lambda: session._stopped
    session.stop.side_effect = lambda: setattr(session, "_stopped", True)
    return AgentLoop(
        session=session, provider=MagicMock(), tool_registry=MagicMock(),
        context_manager=MagicMock(), on_session_end_refresh=refresh,
        project_dir=project_dir,
    )


def test_turn_count_trigger_is_gone():
    assert not hasattr(loop_mod, "COOLDOWN_TURNS")


def test_loop_installs_its_scheduler_as_the_context_hook(tmp_path):
    lp = _loop(str(tmp_path), [])
    assert lp._context_manager._on_memory_over_budget == lp._schedule_over_budget_edit


@pytest.mark.asyncio
async def test_over_budget_spawns_one_background_pass(tmp_path):
    calls = []
    lp = _loop(str(tmp_path), calls)
    assert lp._schedule_over_budget_edit(["state"]) is True
    await lp.drain_refresh()
    assert calls == [("sess", "over_budget")]
    assert lp._refresh_dirty is False          # automatic trigger never queues


@pytest.mark.asyncio
async def test_over_budget_is_single_flight_across_sessions(tmp_path):
    calls = []
    gate = asyncio.Event()
    a = _loop(str(tmp_path), calls, gate=gate, uuid="A")
    b = _loop(str(tmp_path), calls, uuid="B")
    assert a._schedule_over_budget_edit(["state"]) is True
    await asyncio.sleep(0)
    assert b._schedule_over_budget_edit(["state"]) is False
    assert b._refresh_dirty is False
    gate.set()
    await a.drain_refresh()
    assert calls == [("A", "over_budget")]


@pytest.mark.asyncio
async def test_over_budget_respects_the_debounce(tmp_path):
    calls = []
    lp = _loop(str(tmp_path), calls)
    lp._schedule_over_budget_edit(["state"])
    await lp.drain_refresh()
    assert lp._schedule_over_budget_edit(["state"]) is False
    loop_mod._PROJECT_REFRESH[str(tmp_path)]["last_merge_at"] -= REFRESH_DEBOUNCE_S + 1
    assert lp._schedule_over_budget_edit(["state"]) is True
    await lp.drain_refresh()


@pytest.mark.asyncio
async def test_over_budget_waits_for_a_change_and_an_old_enough_watermark(tmp_path):
    wf = WorkspaceFileManager(str(tmp_path))
    wf.write("state", _big_state())
    E.write_marker(wf.dir, editor_ran_at=datetime.now(timezone.utc) - timedelta(minutes=1))
    calls = []
    lp = _loop(str(tmp_path), calls)
    assert lp._schedule_over_budget_edit(["state"]) is False     # nothing changed
    wf.write("state", _big_state() + "- new fact\n")
    assert lp._schedule_over_budget_edit(["state"]) is False     # changed, ran a minute ago
    E.write_marker(wf.dir, editor_ran_at=datetime.now(timezone.utc) - timedelta(hours=2))
    wf.write("state", _big_state() + "- newer fact\n")
    assert lp._schedule_over_budget_edit(["state"]) is True
    await lp.drain_refresh()


@pytest.mark.asyncio
async def test_over_budget_defers_to_a_pass_running_outside_the_loop(tmp_path):
    calls = []
    lp = _loop(str(tmp_path), calls)
    E.claim(str(tmp_path))                     # e.g. the pinned-chat coordinator
    assert lp._schedule_over_budget_edit(["state"]) is False
    E.release(str(tmp_path))


def test_over_budget_refuses_when_stopped(tmp_path):
    lp = _loop(str(tmp_path), [])
    lp._session.stop()
    assert lp._schedule_over_budget_edit(["state"]) is False


@pytest.mark.asyncio
async def test_checkpoint_state_still_schedules_a_pass(tmp_path):
    calls = []
    lp = _loop(str(tmp_path), calls)
    msg = await lp.trigger_checkpoint()
    assert "scheduled" in msg.lower()
    await lp.drain_refresh()
    assert calls == [("sess", "agent_decided")]
