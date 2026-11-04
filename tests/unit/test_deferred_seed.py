# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 153 §4.3: the private-notes fold, the third step of the deferred
seed. Verbatim copies under ``orbital/imported/claude-code/``, an imported
``INDEX.md`` with the agent's own lines retargeted, ONE pointer line in
``orbital/INDEX.md`` through ``WorkspaceFileManager.write``; the seed field
stays set (and the next first use retries) until all of that succeeded."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.daemon_v2.project_store import ProjectStore
from agent_os.onboarding import deferred_seed
from agent_os.onboarding.deferred_seed import (
    IMPORTED_NOTES_DIR, ensure_project_seeded, fold_private_notes, workspace_notes,
)

from tests.unit.test_history_notes import FIXTURE, NOTE_FILES, stage_memory

POINTER_PREFIX = "- orbital/imported/claude-code/ — "


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("ORBITAL_HISTORY_HOME", str(h))
    return str(h)


@pytest.fixture
def store(tmp_path):
    return ProjectStore(str(tmp_path / "data"))


def _imported(store, tmp_path, name="app"):
    ws = tmp_path / f"ws-{name}"
    ws.mkdir()
    pid = store.create_project({"name": name, "workspace": str(ws), "deferred_seed": True,
                                "imported": True})
    return pid, ws


def _dest(ws):
    return ws / "orbital" / "imported" / "claude-code"


def test_fold_copies_verbatim_and_writes_both_indexes(store, home, tmp_path):
    pid, ws = _imported(store, tmp_path)
    mem = stage_memory(home, str(ws))
    assert ensure_project_seeded(store, pid, lang="en") is True
    dest = _dest(ws)
    assert {p.name for p in dest.iterdir()} == NOTE_FILES | {"INDEX.md"}
    for name in NOTE_FILES:
        assert (dest / name).read_bytes() == open(os.path.join(mem, name), "rb").read()
    lines = (dest / "INDEX.md").read_text(encoding="utf-8").splitlines()
    assert lines[0] == (f"{len(NOTE_FILES)} memories Claude Code kept about this project, "
                        f"imported from `{mem}`")
    body = [ln for ln in lines[1:] if ln.strip()]
    # The agent's own lines first, in its order (a line to a missing file is
    # dropped), then the notes it never indexed, built from title — hook.
    assert body[0] == "- [Coupled steps = one commit](feedback_coupled_steps.md) — both sides of one fix in one commit"
    assert "- [Not in the index](not_in_index.md) — A note MEMORY.md never listed" in body
    assert not any("deleted_note.md" in ln for ln in body)
    assert len(body) == len(NOTE_FILES)
    index = (ws / "orbital" / "INDEX.md").read_text(encoding="utf-8")
    pointers = [ln for ln in index.splitlines() if ln.startswith(POINTER_PREFIX)]
    assert pointers == [
        f"{POINTER_PREFIX}{len(NOTE_FILES)} memories Claude Code kept about this project "
        "(INDEX.md lists them; feedback entries are the user's own corrections)"]
    # Written through WorkspaceFileManager.write: the INDEX format contract.
    assert index.startswith("<!--format INDEX")
    assert store.get_project(pid)["deferred_seed"] is False


def test_zh_header(store, home, tmp_path):
    pid, ws = _imported(store, tmp_path)
    mem = stage_memory(home, str(ws))
    ensure_project_seeded(store, pid, lang="zh")
    first = (_dest(ws) / "INDEX.md").read_text(encoding="utf-8").splitlines()[0]
    assert first == f"Claude Code 对这个项目的 {len(NOTE_FILES)} 条记忆，导入自 `{mem}`"


def test_retargets_links_that_point_elsewhere(store, home, tmp_path):
    pid, ws = _imported(store, tmp_path)
    mem = stage_memory(home, str(ws))
    with open(os.path.join(mem, "MEMORY.md"), "w", encoding="utf-8") as fh:
        fh.write(f"- [Coupled]({mem}/feedback_coupled_steps.md) — absolute link\n")
    ensure_project_seeded(store, pid, lang="en")
    text = (_dest(ws) / "INDEX.md").read_text(encoding="utf-8")
    assert "- [Coupled](feedback_coupled_steps.md) — absolute link" in text


def test_pointer_appends_to_an_existing_index_and_is_never_doubled(store, home, tmp_path):
    pid, ws = _imported(store, tmp_path)
    stage_memory(home, str(ws))
    (ws / "orbital").mkdir()
    (ws / "orbital" / "INDEX.md").write_text("## Code\n- src/ — the app\n", encoding="utf-8")
    notes = workspace_notes(str(ws))
    fold_private_notes(str(ws), notes, lang="en")
    fold_private_notes(str(ws), notes, lang="en")  # a retry after a later failure
    index = (ws / "orbital" / "INDEX.md").read_text(encoding="utf-8")
    assert "- src/ — the app" in index
    assert sum(ln.startswith(POINTER_PREFIX) for ln in index.splitlines()) == 1


def test_no_notes_is_a_noop_and_still_clears_the_field(store, home, tmp_path):
    pid, ws = _imported(store, tmp_path)
    assert ensure_project_seeded(store, pid) is True
    assert not (ws / "orbital" / "imported").exists()
    index = ws / "orbital" / "INDEX.md"
    assert not index.exists() or POINTER_PREFIX not in index.read_text()
    assert store.get_project(pid)["deferred_seed"] is False


def test_failed_fold_keeps_the_field_and_the_next_use_retries(store, home, tmp_path,
                                                             monkeypatch, caplog):
    pid, ws = _imported(store, tmp_path)
    stage_memory(home, str(ws))
    real_copy = deferred_seed._copy_verbatim
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disk full")
        real_copy(src, dst)

    monkeypatch.setattr(deferred_seed, "_copy_verbatim", flaky)
    assert ensure_project_seeded(store, pid, lang="en") is True  # skills + AGENTS.md ran
    assert (ws / "AGENTS.md").is_file()
    assert store.get_project(pid)["deferred_seed"] is True
    assert "private notes fold failed" in caplog.text
    assert not (_dest(ws) / "INDEX.md").exists()
    # Next first use: the fold completes and the field clears.
    assert ensure_project_seeded(store, pid, lang="en") is True
    assert (_dest(ws) / "INDEX.md").is_file()
    assert {p.name for p in _dest(ws).iterdir()} == NOTE_FILES | {"INDEX.md"}
    assert store.get_project(pid)["deferred_seed"] is False
    # Opening again changes nothing.
    before = (ws / "orbital" / "INDEX.md").read_text()
    assert ensure_project_seeded(store, pid, lang="en") is False
    assert (ws / "orbital" / "INDEX.md").read_text() == before


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="POSIX permissions; root ignores them")
def test_read_only_orbital_folder_fails_the_fold_only(store, home, tmp_path):
    pid, ws = _imported(store, tmp_path)
    stage_memory(home, str(ws))
    ensure_project_seeded(store, pid)  # first pass: skills etc. create orbital/
    store.update_project(pid, {"deferred_seed": True})
    import shutil
    shutil.rmtree(ws / "orbital" / "imported")
    os.chmod(ws / "orbital", 0o555)
    try:
        ensure_project_seeded(store, pid, lang="en")
        assert store.get_project(pid)["deferred_seed"] is True
    finally:
        os.chmod(ws / "orbital", 0o755)
    ensure_project_seeded(store, pid, lang="en")
    assert store.get_project(pid)["deferred_seed"] is False
    assert (_dest(ws) / "INDEX.md").is_file()


def test_scanner_list_is_the_source_when_ready(store, home, tmp_path):
    from agent_os.onboarding.history import HistoryScanner
    from agent_os.onboarding.sources import claude_code

    pid, ws = _imported(store, tmp_path)
    mem = stage_memory(home, str(ws))
    sc = HistoryScanner(None, home=home, sources=[claude_code.SOURCE])
    assert sc.wait(10)
    # A note written after the scan is not folded: the scanner's list is the source.
    with open(os.path.join(mem, "later.md"), "w", encoding="utf-8") as fh:
        fh.write("written after the scan\n")
    assert len(workspace_notes(str(ws), sc)) == len(NOTE_FILES)
    ensure_project_seeded(store, pid, scanner=sc, lang="en")
    assert len([p for p in _dest(ws).iterdir() if p.name != "INDEX.md"]) == len(NOTE_FILES)


def test_name_clash_with_the_index_is_kept_apart(store, home, tmp_path):
    pid, ws = _imported(store, tmp_path)
    mem = stage_memory(home, str(ws))
    with open(os.path.join(mem, "index.md"), "w", encoding="utf-8") as fh:
        fh.write("a note that happens to be called index\n")
    ensure_project_seeded(store, pid, lang="en")
    names = {p.name for p in _dest(ws).iterdir()}
    assert "INDEX.md" in names and "index-note.md" in names
    assert (_dest(ws) / "index-note.md").read_text() == "a note that happens to be called index\n"


# ---- the route side: imported field, to_thread, project row ----------------


def test_import_marks_the_project_imported(tmp_path, monkeypatch):
    import asyncio

    from agent_os.api.routes import agents_v2
    from agent_os.api.routes import onboarding as onboarding_routes

    store = ProjectStore(str(tmp_path / "data"))
    manager = MagicMock()
    agents_v2.configure(project_store=store, agent_manager=manager, ws_manager=MagicMock())
    ws = tmp_path / "folder"
    ws.mkdir()
    project = asyncio.run(onboarding_routes._create_imported(str(ws), []))
    stored = store.get_project(project["project_id"])
    assert stored["imported"] is True and stored["deferred_seed"] is True


def test_project_row_carries_available_workers(tmp_path):
    from agent_os.api.routes import agents_v2

    store = ProjectStore(str(tmp_path / "data"))
    manager = MagicMock()
    manager.manager_ready = MagicMock(return_value=False)
    manager.effective_pinned_target = MagicMock(return_value="codex")
    manager.available_worker_slugs = MagicMock(return_value=["claude-code", "codex"])
    agents_v2.configure(project_store=store, agent_manager=manager, ws_manager=MagicMock())
    ws = tmp_path / "w"
    ws.mkdir()
    pid = store.create_project({"name": "w", "workspace": str(ws), "imported": True})
    app = FastAPI()
    app.include_router(agents_v2.router)
    row = TestClient(app).get(f"/api/v2/projects/{pid}").json()
    assert row["available_workers"] == ["claude-code", "codex"]
    assert row["imported"] is True


def test_first_use_seed_runs_off_the_event_loop(tmp_path, monkeypatch):
    import threading

    from agent_os.api.routes import agents_v2

    store = ProjectStore(str(tmp_path / "data"))
    manager = MagicMock()
    manager._ensure_dispatcher = AsyncMock()
    agents_v2.configure(project_store=store, agent_manager=manager, ws_manager=MagicMock())
    seen = {}

    def fake_seed(project_store, project_id, **kw):
        seen["thread"] = threading.current_thread().name
        return False

    monkeypatch.setattr(agents_v2, "ensure_project_seeded", fake_seed)
    import asyncio

    async def run():
        seen["loop_thread"] = threading.current_thread().name
        await agents_v2._ensure_seeded("proj_x")

    asyncio.run(run())
    assert seen["thread"] != seen["loop_thread"]


def test_new_session_folds_before_the_first_screen(tmp_path, home):
    """The wizard's Finish lands on POST new-session; the fold must have run
    by then so the card can say every agent can use the notes (spec 153 §4.3)."""
    from agent_os.api.routes import agents_v2

    store = ProjectStore(str(tmp_path / "data"))
    manager = MagicMock()
    manager._ensure_dispatcher = AsyncMock()
    manager.new_session = AsyncMock(return_value={"session_id": "s1", "session_uuid": "u1"})
    agents_v2.configure(project_store=store, agent_manager=manager, ws_manager=MagicMock())
    pid, ws = _imported(store, tmp_path)
    stage_memory(home, str(ws))
    app = FastAPI()
    app.include_router(agents_v2.router)
    r = TestClient(app).post(f"/api/v2/agents/{pid}/new-session", json={})
    assert r.status_code == 200, r.text
    assert (_dest(ws) / "INDEX.md").is_file()
    assert store.get_project(pid)["deferred_seed"] is False
