# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Batch 1114: importing a project starts nothing.

A Windows user ticked a dozen projects in onboarding and Orbital started a
dozen Claude Code processes (spec 178 §4: the keyless background read ran one
pinned worker per project and nothing stopped them). The owner removed the
read:

* A. Import is a list write: no process, no model call, nothing written into
  the folder until the user works there. Projects an older build stamped for
  the read (pending / deferred / running / failed) are normal projects; the
  fields stay as they were (a downgrade to v0.22.0 reads them unchanged) and
  a read an upgrade cut off is never a "didn't finish" card.
* B. Orbital's own agent, on a session's first run in a project with no
  goals, gets one tail row naming the folder's own notes (paths only).
* C (the router's doc head) is tested in ``test_router_doc_head_1114.py``,
  kept apart so it can be merged or left out on its own.
* E. Git repos found on disk join the import list, ticked.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from agent_os.onboarding import import_scanner
from agent_os.onboarding.history import clear_git_cache
from agent_os.onboarding.history_paths import HistoryEnv


def _git_init(path) -> None:
    os.makedirs(path, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)


def _claude_transcript(home, workspace, sid="s1", text="hello"):
    slug = "".join(c if c.isalnum() else "-" for c in str(workspace))
    d = home / ".claude" / "projects" / slug
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{sid}.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "user", "sessionId": sid, "cwd": str(workspace),
                             "timestamp": "2026-11-10T10:00:00Z", "entrypoint": "cli",
                             "message": {"role": "user", "content": text}}) + "\n")
        fh.write(json.dumps({"type": "assistant", "sessionId": sid, "cwd": str(workspace),
                             "timestamp": "2026-11-10T10:00:05Z",
                             "message": {"model": "m", "content": [{"type": "text", "text": "ok"}]}}) + "\n")
    return d


# ---------------------------------------------------------------------------
# A. The read is gone
# ---------------------------------------------------------------------------


def test_the_read_machinery_stays_deleted():
    from agent_os.api.routes import agents_v2, settings
    from agent_os.onboarding import deferred_seed, import_scan
    for name in ("ImportScanQueue", "ScanRunner", "kick", "configure", "get_queue",
                 "pending_fields", "unscanned_count", "scan_agent", "turn_off_waiting",
                 "stamp_deferred_pending", "turn_off_deferred"):
        assert not hasattr(import_scan, name), name
    for name in ("start_import_scan", "_start_manager_scan", "_worker_onboarding"):
        assert not hasattr(agents_v2, name), name
    for name in ("_import_scan_kick", "_import_scan_switch"):
        assert not hasattr(settings, name), name
    assert not hasattr(deferred_seed, "seed_orbital_folder")
    with pytest.raises(ImportError):
        __import__("agent_os.daemon_v2.worker_onboarding")
    # The confirm card a first run's record turn posts is still here.
    assert callable(import_scan.classify_answer) and callable(agents_v2.post_goals_confirm)


@pytest.fixture
def keyless_import(tmp_path, monkeypatch):
    """The keyless harness (real AgentManager, recording sub-agent manager)
    behind the onboarding routes, over a staged home with 12 folders that
    have Claude Code history."""
    from agent_os.api.routes import onboarding as onboarding_routes
    from agent_os.onboarding.history import HistoryScanner
    from agent_os.onboarding.sources import claude_code
    from tests.unit.keyless_harness import build_client

    monkeypatch.delenv("AGENT_OS_API_KEY", raising=False)
    home = tmp_path / "home"
    folders = []
    for i in range(12):
        ws = home / "code" / f"proj-{i:02d}"
        _git_init(ws)
        (ws / "README.md").write_text(f"# proj {i}\n")
        _claude_transcript(home, ws, sid=f"s{i}", text=f"work on proj {i}")
        folders.append(str(ws))
    clear_git_cache()
    monkeypatch.setenv("ORBITAL_HISTORY_HOME", str(home))
    tc, deps = build_client(tmp_path, keyed=False)
    started = AsyncMock()
    monkeypatch.setattr(deps["agent_manager"], "start_agent", started)
    scanner = HistoryScanner(str(tmp_path / "orbital-data"), home=str(home),
                             sources=[claude_code.SOURCE], env=HistoryEnv.for_home(str(home)))
    onboarding_routes.configure(project_store=deps["project_store"], scanner=scanner)
    tc.app.include_router(onboarding_routes.router)
    tc.app.include_router(onboarding_routes.projects_router)
    deadline = time.time() + 10
    while tc.get("/api/v2/onboarding/history").json()["status"] != "ready":
        assert time.time() < deadline
        time.sleep(0.05)
    yield tc, deps, folders, started
    onboarding_routes.configure()


def test_a_keyless_import_of_twelve_starts_nothing_and_writes_nothing(keyless_import):
    tc, deps, folders, started = keyless_import
    # An older web build still sends the read's fields: accepted, ignored.
    r = tc.post("/api/v2/onboarding/import", json={
        "folders": folders, "lang": "zh", "import_scan": True, "defer_scan": False})
    assert r.status_code == 200, r.text
    assert len(r.json()["created"]) == 12
    time.sleep(0.3)  # anything an import scheduled would have started by now
    deps["sub_agent_manager"].send.assert_not_called()
    started.assert_not_called()
    for c in r.json()["created"]:
        p = deps["project_store"].get_project(c["project_id"])
        assert p["imported"] is True and p["deferred_seed"] is True
        assert not any(k.startswith("import_scan") for k in p), p
        assert "orbital_seeded" not in p
        assert not os.path.exists(os.path.join(c["workspace"], "orbital"))
        assert not os.path.exists(os.path.join(c["workspace"], "AGENTS.md"))


def test_first_use_still_seeds_the_whole_folder(keyless_import, monkeypatch):
    from agent_os.onboarding.deferred_seed import ensure_project_seeded

    tc, deps, folders, _ = keyless_import
    created = tc.post("/api/v2/onboarding/import", json={"folders": folders[:1]}).json()["created"]
    pid, ws = created[0]["project_id"], created[0]["workspace"]
    assert ensure_project_seeded(deps["project_store"], pid) is True
    assert os.path.isfile(os.path.join(ws, "AGENTS.md"))
    assert deps["project_store"].get_project(pid)["deferred_seed"] is False


def test_settings_carries_no_read_fields(tmp_path):
    from unittest.mock import MagicMock
    from fastapi import FastAPI
    from agent_os.api.routes import settings as settings_routes

    fake = MagicMock()
    fake.get_masked = MagicMock(return_value={"llm": {"api_key_set": False}})
    settings_routes.configure(fake, project_store=MagicMock())
    app = FastAPI()
    app.include_router(settings_routes.router)
    body = TestClient(app).get("/api/v2/settings").json()
    assert "unscanned_imports" not in body and "import_scan_agent" not in body


# -- upgrade: what an older build left on disk ------------------------------


def _session_file(ws, sid, rows, *, origin):
    from agent_os.agent.session import Session
    s = Session.new(sid, str(ws), session_id=sid, origin=origin)
    for row in rows:
        s.append(row)
    return s


@pytest.mark.parametrize("state", ["pending", "deferred", "running", "failed"])
def test_an_old_read_session_is_never_a_didnt_finish_card(tmp_path, state):
    """A v0.22.0 read cut off by the upgrade (running), or one that failed,
    left a session of daemon rows and the model's tool calls. It is not the
    user's work: no left-off card. The project record keeps its fields."""
    from agent_os.onboarding.left_off import left_off_for_project

    ws = tmp_path / "ws"
    ws.mkdir()
    _session_file(ws, "scan1", [
        {"role": "system", "source": "daemon", "content": "[WORKSPACE SKELETON]\nREADME.md"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "..."},
    ], origin="cold_start")
    project = {"project_id": "p1", "workspace": str(ws), "imported": True,
               "import_scan": state, "import_scan_session": "scan1"}
    out = left_off_for_project(project, None)
    assert out == {"last_time": None, "unfinished": []}
    # A session the user wrote in is theirs, cold start or not.
    _session_file(ws, "scan2", [
        {"role": "assistant", "content": "This is a parser library. Right?"},
        {"role": "user", "content": "add a CLI flag"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t2", "type": "function", "function": {"name": "edit", "arguments": "{}"}}]},
    ], origin="cold_start")
    out = left_off_for_project(project, None)
    assert [u["session_id"] for u in out["unfinished"]] == ["scan2"]


def test_an_ordinary_interrupted_session_is_still_a_card(tmp_path):
    from agent_os.onboarding.left_off import left_off_for_project

    ws = tmp_path / "ws"
    ws.mkdir()
    _session_file(ws, "chat1", [
        {"role": "user", "content": "fix the build"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "shell", "arguments": "{}"}}]},
    ], origin="chat")
    out = left_off_for_project({"project_id": "p", "workspace": str(ws)}, None)
    assert [u["session_id"] for u in out["unfinished"]] == ["chat1"]


# ---------------------------------------------------------------------------
# B. The first run's pointers
# ---------------------------------------------------------------------------


def test_pointers_name_what_exists_as_paths(tmp_path, monkeypatch):
    from agent_os.daemon_v2.agent_md_seeder import AGENT_MD_TEMPLATE
    from agent_os.onboarding.workspace_pointers import pointer_row, pointers

    home = tmp_path / "home"
    ws = home / "work" / "app"
    ws.mkdir(parents=True)
    monkeypatch.setenv("ORBITAL_HISTORY_HOME", str(home))
    assert pointers(str(ws)) == [] and pointer_row([]) is None

    (ws / "CLAUDE.md").write_text("# rules\n")
    (ws / "AGENTS.md").write_text(AGENT_MD_TEMPLATE)          # the one Orbital seeds
    (ws / "readme.txt").write_text("x")
    (ws / "README.md").write_text("# App\n")
    d = _claude_transcript(home, ws)
    assert pointers(str(ws)) == ["CLAUDE.md", "README.md", "readme.txt"]  # no notes yet
    (d / "memory").mkdir()
    (d / "memory" / "feedback_x.md").write_text("---\nname: x\n---\nbody\n")
    got = pointers(str(ws))
    assert got[:3] == ["CLAUDE.md", "README.md", "readme.txt"]
    assert got[3] == str(d / "memory").replace("\\", "/") + "/"
    (ws / "AGENTS.md").write_text("# Our agents\nBe brief.\n")  # user-written
    assert pointers(str(ws))[:2] == ["CLAUDE.md", "AGENTS.md"]
    row = pointer_row(pointers(str(ws)))
    assert row.splitlines()[1:3] == ["- CLAUDE.md", "- AGENTS.md"]
    assert len(row) < 600


def _rig(tmp_path, llm, *, files=None):
    from tests.unit.test_memory_record_turn import _rig as rec_rig
    for name, text in (files or {"README.md": "# Shop helpers\n"}).items():
        (tmp_path / name).write_text(text)
    return rec_rig(tmp_path, llm, seed=False)


def _pointer_rows(session):
    return [m for m in session.get_messages()
            if (m.get("_meta") or {}).get("event") == "workspace_pointers"]


@pytest.mark.asyncio
async def test_a_goal_less_first_run_gets_one_pointer_row_at_the_tail(tmp_path, monkeypatch):
    from tests.unit.test_memory_record_turn import _ScriptedLLM, _reply

    monkeypatch.setenv("ORBITAL_HISTORY_HOME", str(tmp_path / "nohome"))
    llm = _ScriptedLLM([_reply("Done."), _reply("nothing to record")])
    session, _tools, loop = _rig(tmp_path, llm)
    await loop.run()
    (row,) = _pointer_rows(session)
    assert row["role"] == "system" and "- README.md" in row["content"]
    msgs = session.get_messages()
    users = [i for i, m in enumerate(msgs) if m.get("role") == "user"]
    assert msgs.index(row) == users[0] + 1  # right after the user's message
    # On the wire: a [runtime] turn after the user's message, never the
    # system prompt (the cached prefix is untouched).
    first = llm.sent[0]
    assert all("- README.md" not in str(m.get("content")) for m in first if m.get("role") == "system")
    tail = [m for m in first if m.get("role") == "user"]
    assert "- README.md" in str(tail[-1]["content"]) and "[runtime]" in str(tail[-1]["content"])
    # A second run in the same session adds nothing.
    from agent_os.agent.session import persist_user_row
    persist_user_row(session, "and another thing")
    await loop.run()
    assert len(_pointer_rows(session)) == 1


@pytest.mark.asyncio
async def test_no_pointer_row_with_goals_or_with_nothing_to_name(tmp_path, monkeypatch):
    from tests.unit.test_memory_record_turn import _ScriptedLLM, _reply

    monkeypatch.setenv("ORBITAL_HISTORY_HOME", str(tmp_path / "nohome"))
    a = tmp_path / "a"
    a.mkdir()
    goals = a / "orbital" / "instructions" / "project_goals.md"
    goals.parent.mkdir(parents=True)
    goals.write_text("# Goals\nShip it.\n")
    session, _t, loop = _rig(a, _ScriptedLLM([_reply("Done.")]))
    await loop.run()
    assert _pointer_rows(session) == []

    b = tmp_path / "b"
    b.mkdir()
    session, _t, loop = _rig(b, _ScriptedLLM([_reply("Done."), _reply("nothing to record")]),
                             files={"notes.txt": "x"})
    await loop.run()
    assert _pointer_rows(session) == []


@pytest.mark.asyncio
async def test_no_pointer_row_in_quick_tasks(tmp_path, monkeypatch):
    from tests.unit.test_memory_record_turn import _ScriptedLLM, _reply

    monkeypatch.setenv("ORBITAL_HISTORY_HOME", str(tmp_path / "nohome"))
    session, _t, loop = _rig(tmp_path, _ScriptedLLM([_reply("Done.")]))
    loop._context_manager._base_ctx.is_scratch = True
    await loop.run()
    assert _pointer_rows(session) == []


# ---------------------------------------------------------------------------
# E. Git repos on disk
# ---------------------------------------------------------------------------


@pytest.fixture
def disk(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    clear_git_cache()
    return home


def _rel(path, root) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def _scan(home, **kw):
    return {_rel(c.path, home): c for c in
            import_scanner.scan_git_repos(str(home), env=HistoryEnv.for_home(str(home)), **kw)}


def test_repos_under_desktop_documents_downloads_and_to_depth_five(disk):
    for rel in ("Desktop/site", "Documents/thesis", "Downloads/clone",
                "a/b/c/d/five", "a/b/c/d/e/six", "code/app"):
        _git_init(disk / rel)
    got = _scan(disk)
    assert set(got) == {"Desktop/site", "Documents/thesis", "Downloads/clone",
                        "a/b/c/d/five", "code/app"}
    c = got["code/app"]
    assert c.source == "git" and c.is_git and not c.minor and c.session_count == 0
    assert c.by_agent == {} and c.last_activity is not None


def test_a_repo_is_not_walked_into_and_a_worktree_folds_onto_its_repo(disk):
    repo = disk / "code" / "app"
    _git_init(repo)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    _git_init(repo / "vendor" / "nested")          # inside a repo: never seen
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "-q",
                    str(disk / "wt" / "app-feature")], check=True)
    assert set(_scan(disk)) == {"code/app"}


def test_skipped_names_dot_folders_symlinks_and_a_dotfiles_home(disk, tmp_path):
    _git_init(disk)                                  # a dotfiles repo in ~
    for rel in ("Library/x", "node_modules/y", ".config/z", "Pictures/p", "AppData/q",
                "real/r"):
        _git_init(disk / rel)
    outside = tmp_path / "outside" / "linked"
    _git_init(outside)
    try:
        os.symlink(tmp_path / "outside", disk / "link", target_is_directory=True)
    except OSError:
        pass  # Windows without the symlink privilege: nothing to follow anyway
    assert set(_scan(disk)) == {"real/r"}


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="POSIX permissions as a normal user")
def test_an_unreadable_folder_is_skipped_silently(disk):
    _git_init(disk / "ok" / "repo")
    locked = disk / "locked"
    _git_init(locked / "hidden")
    os.chmod(locked, 0)
    try:
        assert set(_scan(disk)) == {"ok/repo"}
    finally:
        os.chmod(locked, 0o755)


def test_the_folder_cap_returns_what_was_found(disk):
    for i in range(30):
        (disk / f"empty{i:02d}").mkdir()
    _git_init(disk / "zz" / "repo")
    assert _scan(disk, max_dirs=5) == {}
    assert set(_scan(disk)) == {"zz/repo"}


def test_a_hanging_folder_cannot_hold_the_list_past_the_deadline(disk, monkeypatch):
    _git_init(disk / "aa" / "repo")
    (disk / "zz-hang").mkdir()
    release = threading.Event()
    real = os.scandir

    def scandir(path):
        if str(path).endswith("zz-hang"):
            release.wait(10)
        return real(path)

    monkeypatch.setattr(import_scanner.os, "scandir", scandir)
    t0 = time.monotonic()
    got = _scan(disk, deadline_s=0.5)
    elapsed = time.monotonic() - t0
    release.set()
    assert elapsed < 2.0
    assert set(got) == {"aa/repo"}


def test_windows_walks_every_fixed_drive_but_not_its_users_folder_twice(tmp_path, monkeypatch):
    # Windows' DirEntry.stat has no st_dev: the volume check must not run
    # there (it would skip every folder); junctions are skipped instead.
    monkeypatch.setattr(import_scanner, "_WINDOWS", True)
    home = tmp_path / "C" / "Users" / "me"
    _git_init(home / "Documents" / "proj")
    _git_init(tmp_path / "C" / "work" / "tool")
    _git_init(tmp_path / "C" / "Users" / "other" / "theirs")
    _git_init(tmp_path / "C" / "Windows" / "sys")
    _git_init(tmp_path / "D" / "code" / "game")
    drives = [str(tmp_path / "C"), str(tmp_path / "D")]
    assert import_scanner.walk_starts(str(home), platform="win32", drives=drives) == [
        (str(home), import_scanner.WALK_HOME_DEPTH),
        (drives[0], import_scanner.WALK_DRIVE_DEPTH),
        (drives[1], import_scanner.WALK_DRIVE_DEPTH)]
    assert import_scanner.walk_starts(str(home), platform="darwin") == [
        (str(home), import_scanner.WALK_HOME_DEPTH)]
    clear_git_cache()
    found = {os.path.relpath(c.path, tmp_path) for c in import_scanner.scan_git_repos(
        str(home), env=HistoryEnv.for_home(str(home)), platform="win32", drives=drives)}
    assert found == {os.path.join("C", "Users", "me", "Documents", "proj"),
                     os.path.join("C", "work", "tool"), os.path.join("D", "code", "game")}


def test_orbitals_own_data_dir_is_never_offered(disk):
    _git_init(disk / "orbital" / "scratch" / "repo")
    _git_init(disk / "code" / "app")
    got = import_scanner.scan_git_repos(str(disk), env=HistoryEnv.for_home(str(disk)),
                                        exclude=[str(disk / "orbital")])
    assert [_rel(c.path, disk) for c in got] == ["code/app"]


def test_repos_merge_under_history_rows_and_are_ticked(disk):
    from types import SimpleNamespace

    from agent_os.onboarding.history_types import HistorySession

    _git_init(disk / "code" / "app")
    _git_init(disk / "code" / "new")
    session = HistorySession(agent="claude-code", native_id="s1", path="x",
                             cwd=str(disk / "code" / "app"))
    session.ended_at = "2026-11-10T10:00:00+00:00"
    analysis = SimpleNamespace(projects={str(disk / "code" / "app"): SimpleNamespace(
        sessions=[session], is_git=True, has_agent_md=False, minor=False)})
    repos = import_scanner.scan_git_repos(str(disk), env=HistoryEnv.for_home(str(disk)))
    rows = import_scanner.candidates_from_analysis(
        analysis, existing_workspaces=[str(disk / "code" / "new")], repos=repos)
    by = {os.path.basename(c.path): c for c in rows}
    assert by["app"].source == "claude-code" and by["app"].by_agent == {"claude-code": 1}
    assert by["new"].source == "git" and by["new"].minor is False
    assert by["new"].already_imported is True


def test_the_list_carries_git_rows_once_the_history_is_ready(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from agent_os.api.routes import onboarding as onboarding_routes
    from agent_os.daemon_v2.project_store import ProjectStore
    from agent_os.onboarding.history import HistoryScanner
    from agent_os.onboarding.sources import claude_code

    home = tmp_path / "home"
    _git_init(home / "Desktop" / "side")
    clear_git_cache()
    monkeypatch.setenv("ORBITAL_HISTORY_HOME", str(home))
    store = ProjectStore(str(tmp_path / "data"))
    scanner = HistoryScanner(str(tmp_path / "data"), home=str(home),
                             sources=[claude_code.SOURCE], env=HistoryEnv.for_home(str(home)))
    onboarding_routes.configure(project_store=store, scanner=scanner, data_dir=str(tmp_path / "data"))
    onboarding_routes._reset_repo_walk()
    app = FastAPI()
    app.include_router(onboarding_routes.router)
    client = TestClient(app)
    try:
        deadline = time.time() + 10
        while client.get("/api/v2/onboarding/history").json()["status"] != "ready":
            assert time.time() < deadline
            time.sleep(0.05)
        # Batch 1115: the walk runs in the background; the list says
        # scanning until it is done, then carries the repo.
        while (body := client.get("/api/v2/onboarding/importable-projects").json())["status"] != "ready":
            assert time.time() < deadline
            time.sleep(0.05)
        (row,) = body["candidates"]
        assert row["source"] == "git" and row["path"] == str(home / "Desktop" / "side")
        assert row["minor"] is False and row["is_git"] is True and row["by_agent"] == {}
    finally:
        onboarding_routes.configure()


# ---------------------------------------------------------------------------
# A. Nothing in the folder until first use: the file history waits too
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_history_waits_for_an_untouched_imports_first_use(tmp_path):
    from agent_os.history.service import FileHistory

    fresh = tmp_path / "fresh"          # imported by this build: nothing written
    fresh.mkdir()
    (fresh / "README.md").write_text("# fresh\n")
    old = tmp_path / "old"              # imported by v0.20-v0.22: orbital/ exists
    (old / "orbital").mkdir(parents=True)
    (old / "README.md").write_text("# old\n")
    made = tmp_path / "made"            # created by hand
    made.mkdir()
    (made / "README.md").write_text("# made\n")
    records = {
        "p_fresh": {"project_id": "p_fresh", "workspace": str(fresh), "deferred_seed": True},
        "p_old": {"project_id": "p_old", "workspace": str(old), "deferred_seed": True},
        "p_made": {"project_id": "p_made", "workspace": str(made)},
    }

    class Store:
        def list_projects(self):
            return list(records.values())

        def get_project(self, pid):
            return records.get(pid)

    svc = FileHistory(Store(), None, quiet=0.2, reconcile_every=0)
    await svc.start()
    try:
        await svc.wait_ready("p_old")
        await svc.wait_ready("p_made")
        assert svc.get_project("p_fresh") is None
        assert not (fresh / "orbital").exists()
        assert svc.get_project("p_old") is not None and svc.get_project("p_made") is not None
        # First use writes orbital/; the next add watches it.
        (fresh / "orbital").mkdir()
        await svc.add_project("p_fresh")
        await svc.wait_ready("p_fresh")
        assert "README.md" in svc.get_project("p_fresh").store.index
    finally:
        await svc.stop()
