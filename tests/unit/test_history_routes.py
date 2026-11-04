# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Spec 065 §5.6: the history routes over a live FileHistory."""
import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.routes import files_v2, history
from agent_os.history.service import FileHistory, recording, tool_attrib
from agent_os.history.store import sha256


class FakeProjects:
    def __init__(self, ws):
        self._p = {"p1": {"project_id": "p1", "workspace": str(ws)}}

    def list_projects(self):
        return list(self._p.values())

    def get_project(self, pid):
        return self._p.get(pid)


class FakeAgents:
    def __init__(self):
        self.sessions = [
            {"session_id": "sess-A", "name": "重写笔记", "session_uuid": "uuid-a"},
            {"session_id": "sess-B", "name": "Second", "session_uuid": "uuid-b"},
        ]

    def list_sessions(self, pid):
        return self.sessions


def iso(dt):
    return dt.isoformat()


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "ws"
    w.mkdir()
    (w / "notes.md").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (w / "plan.md").write_text("plan v1\n", encoding="utf-8")
    (w / "pic.png").write_bytes(b"\x89PNG\x00\x00")
    (w / "orbital" / "sessions").mkdir(parents=True)
    return w


@pytest.fixture
def client(ws):
    app = FastAPI()
    store = FakeProjects(ws)
    fh = FileHistory(store, lambda pid: [], quiet=0.3, reconcile_every=0)
    files_v2.configure(store)
    history.configure(store, agent_manager=FakeAgents(), file_history=fh)
    app.include_router(files_v2.router)
    app.include_router(history.router)

    @app.on_event("startup")
    async def _s():
        await fh.start()
        await fh.wait_ready("p1")

    @app.on_event("shutdown")
    async def _e():
        await fh.stop()

    with TestClient(app) as c:
        c.fh = fh
        yield c


def write_as(fh, ws, rel, text, sid):
    path = os.path.join(str(ws), rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with recording(path, tool_attrib(sid)):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def base(rel):
    return f"/api/v2/projects/p1/files/history{rel}"


class TestStatusAndList:
    def test_untouched_project_has_no_changes(self, client):
        r = client.get(base("/status")).json()
        assert r["status"] == "watching" and r["has_changes"] is False
        assert r["skipped"] == {"binary": 1}
        assert r["tracked_files"] == 2 and r["store_bytes"] > 0

    def test_list_versions_newest_first_with_names_and_groups(self, client, ws):
        now = datetime.now(timezone.utc)
        (ws / "orbital" / "sessions" / "uuid-a.jsonl").write_text("\n".join(json.dumps(r) for r in [
            {"role": "user", "source": "user", "content": "go", "timestamp": iso(now - timedelta(seconds=30))},
            {"role": "agent", "content": "ok", "timestamp": iso(now - timedelta(seconds=29))},
        ]) + "\n")
        write_as(client.fh, ws, "notes.md", "one\nTWO\nthree\n", "sess-A")
        write_as(client.fh, ws, "notes.md", "one\nTWO\nthree\nfour\n", "sess-A")
        r = client.get(base(""), params={"path": "notes.md"}).json()
        assert r["status"] == "watching"
        vs = r["versions"]
        assert [v["kind"] for v in vs] == ["modify", "modify", "baseline"]
        assert vs[0]["current"] is True and vs[1]["current"] is False
        assert vs[0]["attrib"]["sessions"][0]["name"] == "重写笔记"
        assert vs[0]["lines"] == 1 and vs[1]["lines"] == 2
        assert vs[0]["group"] == vs[1]["group"] != vs[2]["group"]  # one turn folds
        assert vs[0]["turn"]["session_id"] == "sess-A"
        assert client.get(base("/status")).json()["has_changes"] is True

    def test_skipped_reason_for_binary(self, client):
        r = client.get(base(""), params={"path": "pic.png"}).json()
        assert r["versions"] == [] and r["skipped_reason"] == "binary"

    def test_path_escape_refused(self, client):
        assert client.get(base(""), params={"path": "../outside.md"}).status_code == 400


class TestView:
    def test_view_ops_and_editor(self, client, ws):
        write_as(client.fh, ws, "notes.md", "one\nTWO\nthree\n", "sess-A")
        r = client.get(base("/view"), params={"path": "notes.md"}).json()
        ops = [(ln["op"], ln["text"]) for ln in r["lines"]]
        assert ops == [("same", "one"), ("del", "two"), ("add", "TWO"), ("same", "three")]
        assert r["first_change"] == 1
        ed = r["editors"][r["lines"][2]["editor"]]
        assert ed["agents"] == ["orbital"] and ed["session_name"] == "重写笔记"

    def test_group_view_from_before_group(self, client, ws):
        b = sha256(b"one\ntwo\nthree\n")
        write_as(client.fh, ws, "notes.md", "one\nTWO\nthree\n", "sess-A")
        write_as(client.fh, ws, "notes.md", "one\nTWO\nthree\nfour\n", "sess-B")
        r = client.get(base("/view"), params={"path": "notes.md", "from": b}).json()
        by_text = {ln["text"]: ln for ln in r["lines"]}
        assert r["editors"][by_text["TWO"]["editor"]]["session_id"] == "sess-A"
        assert r["editors"][by_text["four"]["editor"]]["session_id"] == "sess-B"
        assert by_text["two"]["op"] == "del"

    def test_view_unknown_version_404(self, client):
        assert client.get(base("/view"), params={"path": "notes.md", "sha": "0" * 64}).status_code == 404


class TestRestoreAndName:
    def test_restore_creates_user_version(self, client, ws):
        orig = sha256(b"one\ntwo\nthree\n")
        write_as(client.fh, ws, "notes.md", "overwritten\n", "sess-A")
        r = client.post(base("/restore"), json={"items": [
            {"path": "notes.md", "sha": orig, "expect_sha": sha256(b"overwritten\n")}]}).json()
        assert r["results"][0]["status"] == "ok"
        assert (ws / "notes.md").read_text() == "one\ntwo\nthree\n"
        top = client.get(base(""), params={"path": "notes.md"}).json()["versions"][0]
        assert top["kind"] == "restore" and top["current"] and top["attrib"]["by"] == "user"
        assert top["attrib"]["from_ts"]

    def test_restore_conflict(self, client, ws):
        r = client.post(base("/restore"), json={"items": [
            {"path": "notes.md", "sha": None, "expect_sha": "deadbeef"}]}).json()
        assert r["results"][0]["status"] == "conflict"
        assert (ws / "notes.md").exists()

    def test_name_a_version(self, client, ws):
        sha = sha256(b"one\ntwo\nthree\n")
        r = client.post(base("/name"), json={"path": "notes.md", "sha": sha, "name": " 发布稿 "})
        assert r.status_code == 200 and r.json()["name"] == "发布稿"
        v = client.get(base(""), params={"path": "notes.md"}).json()["versions"][-1]
        assert v["name"] == "发布稿"
        assert client.post(base("/name"), json={"path": "notes.md", "sha": "f" * 64,
                                                "name": "x"}).status_code == 404

    def test_files_panel_save_is_user(self, client, ws):
        r = client.put("/api/v2/projects/p1/files/content", json={"path": "notes.md", "content": "typed\n"})
        assert r.status_code == 200
        top = client.get(base(""), params={"path": "notes.md"}).json()["versions"][0]
        assert top["attrib"] == {"kind": "user"}


class TestTurn:
    def test_turn_lists_files_with_hash_check(self, client, ws):
        t0 = datetime.now(timezone.utc) - timedelta(seconds=1)
        write_as(client.fh, ws, "notes.md", "one\nTWO\nthree\n", "sess-A")
        write_as(client.fh, ws, "plan.md", "plan v2\n", "sess-A")
        write_as(client.fh, ws, "new.md", "created\n", "sess-A")
        time.sleep(0.01)
        t1 = datetime.now(timezone.utc)
        # after the turn the user edits plan.md in the Files panel
        client.put("/api/v2/projects/p1/files/content", json={"path": "plan.md", "content": "mine\n"})
        r = client.get(base("/turn"), params={"session_id": "sess-A", "from_ts": iso(t0),
                                              "to_ts": iso(t1)}).json()
        files = {f["path"]: f for f in r["files"]}
        assert set(files) == {"notes.md", "plan.md", "new.md"}
        assert files["notes.md"]["clean"] and files["notes.md"]["before_sha"] == sha256(b"one\ntwo\nthree\n")
        assert not files["plan.md"]["clean"] and files["plan.md"]["later_editor"]["kind"] == "user"
        assert files["new.md"]["kind"] == "create" and files["new.md"]["before_sha"] is None
        assert r["session_name"] == "重写笔记" and r["no_history"] == 0
        # undo the clean file through the restore route
        f = files["notes.md"]
        res = client.post(base("/restore"), json={"items": [
            {"path": "notes.md", "sha": f["before_sha"], "expect_sha": f["current_sha"]},
            {"path": "plan.md", "sha": files["plan.md"]["before_sha"],
             "expect_sha": files["plan.md"]["turn_sha"]},  # stale on purpose
        ]}).json()["results"]
        assert [x["status"] for x in res] == ["ok", "conflict"]
        assert (ws / "plan.md").read_text() == "mine\n"

    def test_turn_of_other_session_is_empty(self, client, ws):
        t0 = datetime.now(timezone.utc) - timedelta(seconds=1)
        write_as(client.fh, ws, "notes.md", "x\n", "sess-A")
        r = client.get(base("/turn"), params={"session_id": "sess-B", "from_ts": iso(t0)}).json()
        assert r["files"] == []


class TestUnavailable:
    def test_history_unavailable_state(self, ws, monkeypatch):
        from agent_os.history import rules
        monkeypatch.setattr(rules, "find_ripgrep", lambda: None)
        app = FastAPI()
        store = FakeProjects(ws)
        fh = FileHistory(store, None, reconcile_every=0)
        files_v2.configure(store)
        history.configure(store, agent_manager=FakeAgents(), file_history=fh)
        app.include_router(history.router)

        @app.on_event("startup")
        async def _s():
            await fh.start()
            await fh.wait_ready("p1")

        @app.on_event("shutdown")
        async def _e():
            await fh.stop()

        with TestClient(app) as c:
            r = c.get(base(""), params={"path": "notes.md"}).json()
            assert r["status"] == "unavailable" and r["versions"] == []
            assert c.post(base("/restore"), json={"items": [
                {"path": "notes.md", "sha": None, "expect_sha": None}]}).status_code == 409

    def test_not_running_at_all(self, ws):
        app = FastAPI()
        store = FakeProjects(ws)
        files_v2.configure(store)
        history.configure(store, agent_manager=None, file_history=None)
        app.include_router(history.router)
        c = TestClient(app)
        assert c.get(base("/status")).json()["status"] == "unavailable"
