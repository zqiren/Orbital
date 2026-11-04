# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Spec 065: the watcher service end to end on a real watchdog observer.

Covers the verification list items that need no model: baseline, external
writes credited to the running session(s), the debounce, binary/oversized
skips, Orbital's own exact write (the watcher echo records nothing), delete
and restore, the stale expected hash, the Layer-1 restore through
WorkspaceFileManager.write, the store excluding itself and sessions/, the
offline edit, and the watcher-not-running states.
"""
import asyncio
import os
import sys
import time

import pytest

from agent_os.history import rules, service
from agent_os.history.service import FileHistory, recording, tool_attrib, write_and_record
from agent_os.history.store import sha256

QUIET = 0.4


class FakeProjects:
    def __init__(self, **projects):
        self._p = {pid: {"project_id": pid, "workspace": str(ws)} for pid, ws in projects.items()}

    def list_projects(self):
        return list(self._p.values())

    def get_project(self, pid):
        return self._p.get(pid)

    def add(self, pid, ws):
        self._p[pid] = {"project_id": pid, "workspace": str(ws)}

    def remove(self, pid):
        self._p.pop(pid, None)


class Running:
    """Fake candidate predicate: {pid: [(session_id, [agents])]}."""

    def __init__(self):
        self.now: dict[str, list] = {}

    def __call__(self, pid):
        return list(self.now.get(pid, []))


async def until(cond, timeout=8.0, step=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        await asyncio.sleep(step)
    return cond()


def versions(svc, pid, rel):
    return svc.get_project(pid).store.versions_for(rel)


def changes(svc, pid, rel):
    return [v for v in versions(svc, pid, rel) if v["kind"] != "baseline"]


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "ws"
    w.mkdir()
    (w / "notes.md").write_text("# notes\nline one\n", encoding="utf-8", newline="\n")
    (w / "src").mkdir()
    (w / "src" / "a.py").write_text("print('a')\n", encoding="utf-8", newline="\n")
    (w / "node_modules" / "pkg").mkdir(parents=True)
    (w / "node_modules" / "pkg" / "index.js").write_text("x\n", newline="\n")
    (w / "nested" / "orbital").mkdir(parents=True)
    (w / "nested" / "orbital" / "PROJECT_STATE.md").write_text("other project\n", newline="\n")
    (w / "orbital" / "sessions").mkdir(parents=True)
    (w / "orbital" / "sessions" / "s1.jsonl").write_text("{}\n", newline="\n")
    (w / "big.txt").write_bytes(b"y" * (rules.TEXT_MAX_BYTES + 10))
    (w / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    return w


@pytest.fixture
async def svc(ws):
    running = Running()
    s = FileHistory(FakeProjects(p1=ws), running, quiet=QUIET, max_wait=3.0, reconcile_every=0)
    s.running = running
    await s.start()
    await s.wait_ready("p1")
    yield s
    await s.stop()


class TestBaseline:
    async def test_building_then_watching_text_only(self, ws, svc):
        assert svc.status("p1") == {"status": "watching"}
        proj = svc.get_project("p1")
        assert set(proj.store.index) == {"notes.md", "src/a.py"}
        assert proj.skipped == {"big.txt": "size", "pic.png": "binary"}
        assert not proj.store.has_changes()  # D10: a baseline is not a change
        assert [v["kind"] for v in versions(svc, "p1", "notes.md")] == ["baseline"]

    async def test_layer1_files_tracked(self, ws, tmp_path):
        (ws / "orbital" / "PROJECT_STATE.md").write_text("# Project State\n", newline="\n")
        s = FileHistory(FakeProjects(p1=ws), None, quiet=QUIET, reconcile_every=0)
        await s.start()
        await s.wait_ready("p1")
        try:
            assert "orbital/PROJECT_STATE.md" in s.get_project("p1").store.index
            assert "nested/orbital/PROJECT_STATE.md" not in s.get_project("p1").store.index
        finally:
            await s.stop()

    async def test_write_during_baseline_starts_history_there(self, ws, monkeypatch):
        real = rules.list_tracked

        def racing_listing(workspace, nested=()):
            # the watch is already scheduled; a write lands before the read
            (ws / "notes.md").write_text("written during baseline\n", newline="\n")
            return real(workspace, nested)

        monkeypatch.setattr(rules, "list_tracked", racing_listing)
        s = FileHistory(FakeProjects(p1=ws), None, quiet=QUIET, reconcile_every=0)
        await s.start()
        await s.wait_ready("p1")
        monkeypatch.setattr(rules, "list_tracked", real)
        try:
            await asyncio.sleep(QUIET * 4)
            vs = versions(s, "p1", "notes.md")
            assert [v["kind"] for v in vs] == ["baseline"]  # no duplicate version
            assert s.get_project("p1").store.get_blob(vs[0]["sha"]) == b"written during baseline\n"
        finally:
            await s.stop()


class TestWatcherAttribution:
    async def test_external_write_while_one_session_runs(self, ws, svc):
        svc.running.now["p1"] = [("sess-A", ["claude-code"])]
        (ws / "notes.md").write_text("# notes\nrewritten by a worker\n", newline="\n")
        assert await until(lambda: changes(svc, "p1", "notes.md"))
        v = changes(svc, "p1", "notes.md")[0]
        assert v["kind"] == "modify"
        assert v["attrib"] == {"kind": "running",
                               "sessions": [{"session_id": "sess-A", "agents": ["claude-code"]}]}
        assert (v["added"], v["removed"]) == (1, 1)

    async def test_two_sessions_running_is_ambiguous(self, ws, svc):
        svc.running.now["p1"] = [("sess-A", ["claude-code"]), ("sess-B", ["codex"])]
        (ws / "notes.md").write_text("both\n", newline="\n")
        assert await until(lambda: changes(svc, "p1", "notes.md"))
        a = changes(svc, "p1", "notes.md")[0]["attrib"]
        assert a["ambiguous"] is True and [s["session_id"] for s in a["sessions"]] == ["sess-A", "sess-B"]

    async def test_nobody_running_is_outside(self, ws, svc):
        (ws / "new.md").write_text("hello\n", newline="\n")
        assert await until(lambda: changes(svc, "p1", "new.md"))
        v = changes(svc, "p1", "new.md")[0]
        assert v["kind"] == "create" and v["attrib"] == {"kind": "outside"}

    async def test_burst_of_20_writes_in_2s_is_one_version(self, ws, tmp_path):
        s = FileHistory(FakeProjects(p1=ws), None, quiet=1.0, max_wait=30, reconcile_every=0)
        await s.start()
        await s.wait_ready("p1")
        try:
            for i in range(20):
                (ws / "notes.md").write_text(f"draft {i}\n", newline="\n")
                await asyncio.sleep(0.1)
            assert await until(lambda: changes(s, "p1", "notes.md"), timeout=6)
            await asyncio.sleep(1.5)
            vs = changes(s, "p1", "notes.md")
            assert len(vs) == 1
            assert s.get_project("p1").store.get_blob(vs[0]["sha"]) == b"draft 19\n"
        finally:
            await s.stop()

    @pytest.mark.xfail(
        sys.platform == "win32", strict=False,
        reason="watchdog 7's Windows emitter delivers no events until writes pause "
               "(~1 s); a file written without pause is caught when it pauses or "
               "by the reconcile pass")
    async def test_max_wait_flushes_a_file_written_without_pause(self, ws):
        s = FileHistory(FakeProjects(p1=ws), None, quiet=0.5, max_wait=1.0, reconcile_every=0)
        await s.start()
        await s.wait_ready("p1")
        try:
            t0 = time.monotonic()
            got_early = False
            i = 0
            while time.monotonic() - t0 < 3.0:
                with open(ws / "notes.md", "a") as f:
                    f.write(f"log {i}\n")
                i += 1
                await asyncio.sleep(0.1)
                if changes(s, "p1", "notes.md"):
                    got_early = True
                    break
            assert got_early  # a version while writes never paused
        finally:
            await s.stop()

    async def test_binary_and_oversized_skipped_and_reported(self, ws, svc):
        svc.running.now["p1"] = [("sess-A", ["codex"])]
        (ws / "photo.jpg").write_bytes(b"\xff\xd8\xff\x00" * 100)
        (ws / "huge.log").write_bytes(b"z" * (rules.TEXT_MAX_BYTES + 1))
        proj = svc.get_project("p1")
        assert await until(lambda: {"photo.jpg", "huge.log"} <= set(proj.skipped))
        assert proj.skipped["photo.jpg"] == "binary" and proj.skipped["huge.log"] == "size"
        assert "photo.jpg" not in proj.store.index and "huge.log" not in proj.store.index
        assert {e["path"] for e in proj.skip_events} >= {"photo.jpg", "huge.log"}

    async def test_office_document_is_tracked_whole(self, ws, svc):
        (ws / "deck.pptx").write_bytes(b"PK\x03\x04\x00\x00v1" * 50)
        assert await until(lambda: changes(svc, "p1", "deck.pptx"))
        v = changes(svc, "p1", "deck.pptx")[0]
        assert v["added"] is None and v["size"] == 400

    async def test_store_excludes_itself_sessions_and_deps(self, ws, svc):
        (ws / "orbital" / "sessions" / "s1.jsonl").write_text("{}\n{}\n", newline="\n")
        (ws / "orbital" / "backups").mkdir()
        (ws / "orbital" / "backups" / "PROJECT_STATE.md").write_text("old\n", newline="\n")
        (ws / "node_modules" / "pkg" / "index.js").write_text("y\n", newline="\n")
        (ws / "nested" / "orbital" / "PROJECT_STATE.md").write_text("changed\n", newline="\n")
        (ws / "notes.md").write_text("trigger a flush that writes the store\n", newline="\n")
        assert await until(lambda: changes(svc, "p1", "notes.md"))
        await asyncio.sleep(QUIET * 3)  # the store's own writes must not echo
        paths = {v["path"] for v in svc.get_project("p1").store.versions}
        assert paths == {"notes.md", "src/a.py"}

    async def test_gitignored_new_file_is_not_tracked(self, ws, svc):
        (ws / ".gitignore").write_text("out/\n", newline="\n")
        (ws / "out").mkdir()
        (ws / "out" / "gen.txt").write_text("generated\n", newline="\n")
        assert await until(lambda: changes(svc, "p1", ".gitignore"))
        await asyncio.sleep(QUIET * 3)
        assert not versions(svc, "p1", "out/gen.txt")

    async def test_delete_then_restore_recreates(self, ws, svc):
        before = (ws / "notes.md").read_bytes()
        os.remove(ws / "notes.md")
        assert await until(lambda: any(v["kind"] == "delete" for v in versions(svc, "p1", "notes.md")))
        res = await asyncio.to_thread(svc.restore, "p1",
                                      [{"path": "notes.md", "sha": sha256(before), "expect_sha": None}])
        assert res[0]["status"] == "ok"
        assert (ws / "notes.md").read_bytes() == before
        last = versions(svc, "p1", "notes.md")[-1]
        assert last["kind"] == "restore" and last["attrib"]["kind"] == "restore"
        assert last["attrib"]["by"] == "user"


class TestOwnWrites:
    async def test_tool_write_is_exact_and_echo_records_nothing(self, ws, svc):
        svc.running.now["p1"] = [("sess-A", ["orbital"]), ("sess-B", ["codex"])]
        path = os.path.join(str(ws), "notes.md")
        with recording(path, tool_attrib("sess-A")):
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write("written by orbital's write tool\n")
        vs = changes(svc, "p1", "notes.md")
        assert len(vs) == 1
        assert vs[0]["attrib"] == {"kind": "tool", "sessions": [{"session_id": "sess-A", "agents": ["orbital"]}]}
        await asyncio.sleep(QUIET * 4)  # the watcher's flush finds the hash equal
        assert len(changes(svc, "p1", "notes.md")) == 1

    async def test_tool_creates_new_file(self, ws, svc):
        path = os.path.join(str(ws), "docs", "new.md")
        write_and_record(path, b"fresh\n", tool_attrib("sess-A"))
        vs = changes(svc, "p1", "docs/new.md")
        assert len(vs) == 1 and vs[0]["kind"] == "create"

    async def test_no_active_history_is_a_plain_write(self, tmp_path):
        assert service.active() is None
        p = tmp_path / "x.md"
        write_and_record(str(p), b"ok\n", tool_attrib("s"))
        assert p.read_bytes() == b"ok\n"

    async def test_daemon_layer1_write_is_orbital(self, ws, svc):
        from agent_os.agent.workspace_files import WorkspaceFileManager
        WorkspaceFileManager(str(ws)).write("decisions", "## d\n- pick sqlite\n")
        vs = changes(svc, "p1", "orbital/DECISIONS.md")
        assert len(vs) == 1 and vs[0]["attrib"] == {"kind": "orbital"}
        await asyncio.sleep(QUIET * 3)
        assert len(changes(svc, "p1", "orbital/DECISIONS.md")) == 1


class TestRestore:
    async def test_stale_expected_hash_is_refused(self, ws, svc):
        base = versions(svc, "p1", "notes.md")[0]
        (ws / "notes.md").write_text("someone else wrote after the review\n", newline="\n")
        reviewed = sha256(b"# notes\nline one\n")
        res = await asyncio.to_thread(svc.restore, "p1",
                                      [{"path": "notes.md", "sha": base["sha"], "expect_sha": sha256(b"stale")}])
        assert res[0]["status"] == "conflict"
        assert res[0]["current_sha"] == sha256(b"someone else wrote after the review\n")
        assert (ws / "notes.md").read_text() == "someone else wrote after the review\n"
        assert reviewed == base["sha"]

    async def test_restore_records_what_it_replaces_first(self, ws, svc):
        base = versions(svc, "p1", "notes.md")[0]
        svc.running.now["p1"] = [("sess-A", ["pi"])]
        (ws / "notes.md").write_text("unflushed agent text\n", newline="\n")
        cur = sha256(b"unflushed agent text\n")
        # restore immediately, before the debounce flushes the write
        res = await asyncio.to_thread(svc.restore, "p1",
                                      [{"path": "notes.md", "sha": base["sha"], "expect_sha": cur}],
                                      capture_attrib=svc.current_attrib("p1"))
        assert res[0]["status"] == "ok"
        vs = versions(svc, "p1", "notes.md")
        assert [v["kind"] for v in vs] == ["baseline", "modify", "restore"]
        assert vs[1]["sha"] == cur and vs[1]["attrib"]["sessions"][0]["session_id"] == "sess-A"
        assert (ws / "notes.md").read_bytes() == b"# notes\nline one\n"
        await asyncio.sleep(QUIET * 4)
        assert len(versions(svc, "p1", "notes.md")) == 3  # the restore's echo adds nothing

    async def test_restore_of_a_restore(self, ws, svc):
        base = versions(svc, "p1", "notes.md")[0]
        (ws / "notes.md").write_text("v2\n", newline="\n")
        assert await until(lambda: changes(svc, "p1", "notes.md"))
        await asyncio.to_thread(svc.restore, "p1",
                                [{"path": "notes.md", "sha": base["sha"], "expect_sha": sha256(b"v2\n")}])
        res = await asyncio.to_thread(svc.restore, "p1",
                                      [{"path": "notes.md", "sha": sha256(b"v2\n"), "expect_sha": base["sha"]}])
        assert res[0]["status"] == "ok" and (ws / "notes.md").read_text() == "v2\n"

    async def test_refuses_store_and_sessions(self, ws, svc):
        res = await asyncio.to_thread(svc.restore, "p1", [
            {"path": "orbital/sessions/s1.jsonl", "sha": None, "expect_sha": None},
            {"path": "orbital/history/index.json", "sha": None, "expect_sha": None},
        ])
        assert [r["status"] for r in res] == ["refused", "refused"]
        assert (ws / "orbital" / "sessions" / "s1.jsonl").exists()

    async def test_created_file_restore_deletes_it(self, ws, svc):
        (ws / "made.md").write_text("by the turn\n", newline="\n")
        assert await until(lambda: changes(svc, "p1", "made.md"))
        res = await asyncio.to_thread(svc.restore, "p1",
                                      [{"path": "made.md", "sha": None, "expect_sha": sha256(b"by the turn\n")}])
        assert res[0]["status"] == "ok" and not (ws / "made.md").exists()
        assert svc.get_project("p1").store.get_blob(sha256(b"by the turn\n")) == b"by the turn\n"

    async def test_busy_when_replace_is_denied(self, ws, svc, monkeypatch):
        base = versions(svc, "p1", "notes.md")[0]
        (ws / "notes.md").write_text("locked\n", newline="\n")
        import agent_os.agent.workspace_files as wf

        def deny(src, dst):
            raise PermissionError("share lock")
        monkeypatch.setattr(wf, "_atomic_replace", deny)
        res = await asyncio.to_thread(svc.restore, "p1",
                                      [{"path": "notes.md", "sha": base["sha"], "expect_sha": sha256(b"locked\n")}])
        assert res[0]["status"] == "busy"
        assert (ws / "notes.md").read_text() == "locked\n"
        assert not [n for n in os.listdir(ws) if n.startswith(rules.RESTORE_TMP_PREFIX)]

    async def test_layer1_restore_goes_through_workspace_file_manager(self, ws, svc):
        from agent_os.agent import state_blocks
        from agent_os.agent.workspace_files import WorkspaceFileManager
        wfm = WorkspaceFileManager(str(ws))
        wfm.write("state", "## Now\n- shipping the history store\n")
        first = changes(svc, "p1", "orbital/PROJECT_STATE.md")[-1]
        first_bytes = (ws / "orbital" / "PROJECT_STATE.md").read_bytes()
        wfm.write("state", "## Now\n- something else entirely\n")
        cur = (ws / "orbital" / "PROJECT_STATE.md").read_bytes()
        calls = []
        real_write = WorkspaceFileManager.write

        def spy(self, key, content):
            calls.append(key)
            return real_write(self, key, content)
        import agent_os.agent.workspace_files as wf
        orig = wf.WorkspaceFileManager.write
        wf.WorkspaceFileManager.write = spy
        try:
            res = await asyncio.to_thread(svc.restore, "p1", [
                {"path": "orbital/PROJECT_STATE.md", "sha": first["sha"], "expect_sha": sha256(cur)}])
        finally:
            wf.WorkspaceFileManager.write = orig
        assert res[0]["status"] == "ok" and calls == ["state"]
        on_disk = (ws / "orbital" / "PROJECT_STATE.md").read_bytes()
        last = versions(svc, "p1", "orbital/PROJECT_STATE.md")[-1]
        assert last["kind"] == "restore" and last["sha"] == sha256(on_disk)
        assert "shipping the history store" in on_disk.decode()
        assert on_disk.startswith(first_bytes.split(b"\n", 1)[0])  # format header kept
        state_blocks.parse(on_disk.decode())  # reparses
        # exactly one restore version, no extra 'orbital' version from the inner write
        kinds = [v["kind"] for v in versions(svc, "p1", "orbital/PROJECT_STATE.md")]
        assert kinds.count("restore") == 1 and kinds[-2] != "restore"


class TestLifecycle:
    async def test_offline_edit_is_closed(self, ws):
        s = FileHistory(FakeProjects(p1=ws), None, quiet=QUIET, reconcile_every=0)
        await s.start()
        await s.wait_ready("p1")
        await s.stop()
        time.sleep(0.01)
        (ws / "notes.md").write_text("edited in TextEdit while Orbital was closed\n", newline="\n")
        os.remove(ws / "src" / "a.py")
        s2 = FileHistory(FakeProjects(p1=ws), None, quiet=QUIET, reconcile_every=0)
        await s2.start()
        await s2.wait_ready("p1")
        try:
            v = changes(s2, "p1", "notes.md")[-1]
            assert v["attrib"] == {"kind": "closed"} and v["kind"] == "modify"
            d = versions(s2, "p1", "src/a.py")[-1]
            assert d["kind"] == "delete" and d["attrib"] == {"kind": "closed"}
            assert s2.status("p1")["status"] == "watching"
        finally:
            await s2.stop()

    async def test_observer_failing_is_unavailable(self, ws):
        def broken():
            raise OSError("no fsevents")
        s = FileHistory(FakeProjects(p1=ws), None, observer_factory=broken, reconcile_every=0)
        await s.start()
        try:
            st = s.status("p1")
            assert st["status"] == "unavailable" and "observer_failed" in st["reason"]
        finally:
            await s.stop()

    async def test_missing_ripgrep_is_unavailable(self, ws, monkeypatch):
        monkeypatch.setattr(rules, "find_ripgrep", lambda: None)
        s = FileHistory(FakeProjects(p1=ws), None, reconcile_every=0)
        await s.start()
        await s.wait_ready("p1")
        try:
            st = s.status("p1")
            assert st["status"] == "unavailable" and st["reason"].startswith("listing_failed")
            with pytest.raises(service.HistoryUnavailable):
                s.restore("p1", [{"path": "notes.md", "sha": None, "expect_sha": None}])
        finally:
            await s.stop()

    async def test_missing_workspace_and_dead_observer(self, ws, tmp_path):
        projects = FakeProjects(p1=ws, gone=tmp_path / "nope")
        s = FileHistory(projects, None, quiet=QUIET, reconcile_every=0)
        await s.start()
        await s.wait_ready("p1")
        try:
            assert s.status("gone") == {"status": "unavailable", "reason": "workspace_missing"}
            s._observer.stop()
            s._observer.join(5)
            await s.reconcile_all()
            assert s.status("p1") == {"status": "unavailable", "reason": "observer_dead"}
        finally:
            await s.stop()

    async def test_remove_project_stops_watching(self, ws):
        projects = FakeProjects(p1=ws)
        s = FileHistory(projects, None, quiet=QUIET, reconcile_every=0)
        await s.start()
        await s.wait_ready("p1")
        proj = s.get_project("p1")
        projects.remove("p1")
        await s.remove_project("p1")
        (ws / "notes.md").write_text("after delete\n", newline="\n")
        await asyncio.sleep(QUIET * 3)
        try:
            assert proj.closed and not [v for v in proj.store.versions if v["kind"] != "baseline"]
            assert s.status("p1")["status"] == "unavailable"
        finally:
            await s.stop()

    async def test_nested_project_owns_its_files(self, ws, tmp_path):
        inner = ws / "inner"
        inner.mkdir()
        (inner / "own.md").write_text("inner file\n", newline="\n")
        running = Running()
        s = FileHistory(FakeProjects(outer=ws, inner=inner), running, quiet=QUIET, reconcile_every=0)
        await s.start()
        await s.wait_ready("outer")
        await s.wait_ready("inner")
        try:
            assert "inner/own.md" not in s.get_project("outer").store.index
            assert "own.md" in s.get_project("inner").store.index
            running.now["outer"] = [("outer-sess", ["codex"])]
            (inner / "own.md").write_text("written from the outer session\n", newline="\n")
            assert await until(lambda: changes(s, "inner", "own.md"))
            # candidates include enclosing projects' sessions
            assert changes(s, "inner", "own.md")[0]["attrib"]["sessions"][0]["session_id"] == "outer-sess"
            assert not versions(s, "outer", "inner/own.md")
        finally:
            await s.stop()
