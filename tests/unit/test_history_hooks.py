# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Spec 065: the write/edit tools record exact versions; the candidate
helper's session semantics (§5.4)."""
import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from agent_os.agent.tools.edit import EditTool
from agent_os.agent.tools.write import WriteTool
from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.history.service import FileHistory


class FakeProjects:
    def __init__(self, ws):
        self._p = {"p1": {"project_id": "p1", "workspace": str(ws)}}

    def list_projects(self):
        return list(self._p.values())

    def get_project(self, pid):
        return self._p.get(pid)


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "ws"
    w.mkdir()
    (w / "notes.md").write_text("alpha\nbeta\n", encoding="utf-8")
    return w


@pytest.fixture
async def svc(ws):
    s = FileHistory(FakeProjects(ws), lambda pid: [("other", ["codex"])], quiet=0.3, reconcile_every=0)
    await s.start()
    await s.wait_ready("p1")
    yield s
    await s.stop()


def changes(svc, rel):
    return [v for v in svc.get_project("p1").store.versions_for(rel) if v["kind"] != "baseline"]


class TestTools:
    async def test_write_tool_records_session_exactly(self, ws, svc):
        out = WriteTool(str(ws), session_id="sess-1").execute(path="notes.md", content="rewritten\n")
        assert json.loads(out.content)["status"] == "success"
        vs = changes(svc, "notes.md")
        assert len(vs) == 1
        assert vs[0]["attrib"] == {"kind": "tool", "sessions": [{"session_id": "sess-1", "agents": ["orbital"]}]}
        await asyncio.sleep(1.0)
        assert len(changes(svc, "notes.md")) == 1  # watcher echo: hash equal

    async def test_edit_tool_records_once(self, ws, svc):
        EditTool(str(ws), session_id="sess-1").execute(path="notes.md", old_text="beta", new_text="gamma")
        vs = changes(svc, "notes.md")
        assert len(vs) == 1 and (vs[0]["added"], vs[0]["removed"]) == (1, 1)
        assert (ws / "notes.md").read_text() == "alpha\ngamma\n"

    async def test_fanout_worker_is_tagged(self, ws, svc):
        WriteTool(str(ws), session_id="sess-1", worker="worker:f1-0").execute(path="w.md", content="w\n")
        v = changes(svc, "w.md")[0]
        assert v["attrib"]["worker"] == "worker:f1-0" and v["kind"] == "create"

    def test_tools_unchanged_without_history(self, ws):
        WriteTool(str(ws)).execute(path="a/b.md", content="x\n")
        assert (ws / "a" / "b.md").read_text() == "x\n"
        EditTool(str(ws)).execute(path="a/b.md", old_text="x", new_text="y")
        assert (ws / "a" / "b.md").read_text() == "y\n"
        assert not (ws / "orbital" / "history").exists()


class FakeSAM:
    def __init__(self, running, active):
        self._running, self._active = running, active

    def running_session_ids(self, pid):
        return set(self._running)

    def list_active(self, pid, *, session_id=None):
        return self._active.get(session_id, [])


def cands(holder, running=(), active=None):
    fake = SimpleNamespace(
        current_holder_session_id=lambda pid: holder,
        _sub_agent_manager=FakeSAM(running, active or {}),
    )
    return AgentManager.history_candidates(fake, "p1")


class TestCandidates:
    def test_manager_and_worker_in_one_session_is_one_candidate(self):
        got = cands("S", running=["S"], active={"S": [{"handle": "claude-code", "status": "running"}]})
        assert got == [("S", ["claude-code"])]

    def test_holder_without_workers_is_orbital(self):
        assert cands("S") == [("S", ["orbital"])]

    def test_idle_adapters_are_not_writers(self):
        got = cands(None, running=["T"], active={"T": [
            {"handle": "codex", "status": "idle"},
            {"handle": "pi", "status": "background-running"},
        ]})
        assert got == [("T", ["pi"])]

    def test_two_sessions(self):
        got = cands("S", running=["T"], active={"T": [{"handle": "codex", "status": "running"}]})
        assert got == [("S", ["orbital"]), ("T", ["codex"])]

    def test_nobody(self):
        assert cands(None) == []


def test_session_ids_threaded_to_tools():
    """Both construction sites pass the session (agent_manager.py)."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "..", "agent_os", "daemon_v2",
                            "agent_manager.py"), encoding="utf-8").read()
    assert "WriteTool(workspace=config.workspace, session_id=session_id)" in src
    assert "EditTool(workspace=config.workspace, session_id=session_id)" in src
    assert "WriteTool(workspace=workspace, session_id=session_id," in src
    assert "EditTool(workspace=workspace, session_id=session_id," in src
