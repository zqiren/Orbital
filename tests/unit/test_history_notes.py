# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 153 §4.1: the Claude Code private-notes reader, the scanner's notes
list, ``GET /history``'s ``notes_by_agent`` / ``notes_by_folder`` and the
importable-projects ``notes`` field.

The fixture (``tests/fixtures/claude_memory_153``) reproduces the layout of
a real ``~/.claude/projects/<dir>/memory`` folder with invented content:
top-level ``type``, ``metadata.type``, an untyped note, one without a
description, one whose frontmatter does not parse, one with no frontmatter
and one the ``MEMORY.md`` index never lists. Never the real ``~/.claude``.
"""

from __future__ import annotations

import json
import os
import shutil
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.onboarding.sources import claude_code

FIXTURE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, "fixtures", "claude_memory_153"))
NOTE_FILES = {
    "feedback_coupled_steps.md", "project_release_train.md", "reference_realpath.md",
    "untyped.md", "no_description.md", "broken_yaml.md", "no_frontmatter.md",
    "not_in_index.md",
}


def encoded(path: str) -> str:
    """Claude Code's lossy dir name for a cwd."""
    return "".join("-" if ch in "/\\:." else ch for ch in path)


def transcript(path: str, cwd: str, sid: str = "s1") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [
        {"type": "user", "sessionId": sid, "cwd": cwd, "timestamp": "2026-11-01T10:00:00Z",
         "entrypoint": "cli", "message": {"role": "user", "content": "hello there"}},
        {"type": "assistant", "sessionId": sid, "cwd": cwd, "timestamp": "2026-11-01T10:00:05Z",
         "message": {"model": "m", "content": [{"type": "text", "text": "hi"}]}},
    ]
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def stage_memory(home: str, cwd: str | None, *, dirname: str | None = None,
                 sessions: int = 1, source: str = FIXTURE) -> str:
    """``home/.claude/projects/<dir>/memory`` holding a copy of ``source``;
    ``sessions`` transcripts recording ``cwd`` (none when cwd is None)."""
    d = os.path.join(home, ".claude", "projects", dirname or encoded(cwd or "/nowhere"))
    os.makedirs(d, exist_ok=True)
    mem = os.path.join(d, "memory")
    if source:
        shutil.copytree(source, mem)
    else:
        os.makedirs(mem)
    if cwd is not None:
        for i in range(sessions):
            transcript(os.path.join(d, f"s{i}.jsonl"), cwd, sid=f"s{i}-{os.path.basename(d)}")
    return mem


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    h.mkdir()
    return str(h)


# ---- the reader -------------------------------------------------------------


def test_every_md_but_the_index_is_a_note(home, tmp_path):
    ws = str(tmp_path / "ws")
    mem = stage_memory(home, ws)
    notes = claude_code.notes(home)
    assert {os.path.basename(n.path) for n in notes} == NOTE_FILES
    assert all(n.folder == ws for n in notes)
    assert all(os.path.dirname(n.path) == mem for n in notes)
    assert all(n.encoded == encoded(ws) for n in notes)


def test_title_hook_and_kind_follow_frontmatter_then_index(home, tmp_path):
    stage_memory(home, str(tmp_path / "ws"))
    by = {os.path.basename(n.path): n for n in claude_code.notes(home)}
    # Top-level ``type`` (the older layout) and ``metadata.type`` both count.
    assert (by["feedback_coupled_steps.md"].title, by["feedback_coupled_steps.md"].kind) == (
        "Coupled steps = one commit", "feedback")
    assert by["feedback_coupled_steps.md"].hook == (
        "Both sides of one fix land in one commit; rewrite old-contract tests")
    assert by["project_release_train.md"].kind == "project"
    assert by["reference_realpath.md"].kind == "reference"
    assert by["untyped.md"].kind is None
    # No description: the index line's text after the dash.
    assert by["no_description.md"].hook == "hook taken from this index line"
    # Unparseable frontmatter: file name for the title, index line for the hook.
    broken = by["broken_yaml.md"]
    assert (broken.title, broken.hook, broken.kind) == (
        "broken_yaml", "its YAML does not parse", None)
    plain = by["no_frontmatter.md"]
    assert (plain.title, plain.hook) == ("no_frontmatter", "no frontmatter at all")
    lone = by["not_in_index.md"]
    assert (lone.title, lone.hook, lone.index_line) == (
        "Not in the index", "A note MEMORY.md never listed", None)
    assert by["untyped.md"].index_line == "- [Untyped note](untyped.md) — a note with no type"


def test_frontmatter_that_raises_outside_yaml_errors_still_counts(home, tmp_path):
    """Well-formed YAML can still fail to load: an impossible date raises a
    plain ValueError from PyYAML's constructor, not a YAMLError. The note
    still counts (and so do its neighbours in the same folder)."""
    mem = stage_memory(home, str(tmp_path / "ws"))
    with open(os.path.join(mem, "bad_date.md"), "w", encoding="utf-8") as fh:
        fh.write("---\nname: Bad date\ncreated: 2026-13-45\n---\nbody\n")
    by = {os.path.basename(n.path): n for n in claude_code.notes(home)}
    assert set(by) == NOTE_FILES | {"bad_date.md"}
    assert (by["bad_date.md"].title, by["bad_date.md"].kind) == ("bad_date", None)


def test_dir_without_transcript_has_no_folder(home):
    stage_memory(home, None, dirname="-Users-x-old-repo")
    notes = claude_code.notes(home)
    assert len(notes) == len(NOTE_FILES)
    assert {n.folder for n in notes} == {None}
    assert {n.encoded for n in notes} == {"-Users-x-old-repo"}


def test_empty_memory_dir_and_missing_home_contribute_nothing(home, tmp_path):
    stage_memory(home, str(tmp_path / "ws"), source="")
    assert claude_code.notes(home) == []
    assert claude_code.notes(str(tmp_path / "no-such-home")) == []


def test_notes_for_dirs_reads_only_those_dirs(home, tmp_path):
    ws, other = str(tmp_path / "ws"), str(tmp_path / "other")
    mem = stage_memory(home, ws)
    stage_memory(home, other)
    notes = claude_code.notes_in_dirs([os.path.dirname(mem)], folder=ws)
    assert len(notes) == len(NOTE_FILES) and {n.folder for n in notes} == {ws}


# ---- the scanner and the routes --------------------------------------------


def _scanner(home, tmp_path):
    from agent_os.onboarding.history import HistoryScanner
    from agent_os.onboarding.history_paths import HistoryEnv
    return HistoryScanner(str(tmp_path / "data"), home=home, sources=[claude_code.SOURCE],
                          env=HistoryEnv.for_home(home))


def test_scanner_keeps_the_notes_and_reports_counts(home, tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    stage_memory(home, str(ws))
    stage_memory(home, None, dirname="-Users-x-old-repo")
    sc = _scanner(home, tmp_path)
    assert sc.wait(10)
    st = sc.status()
    assert st["notes_by_agent"] == {"claude-code": 2 * len(NOTE_FILES)}
    assert st["notes_by_folder"] == [
        {"agent": "claude-code", "folder": str(ws), "encoded": encoded(str(ws)),
         "n": len(NOTE_FILES)},
        {"agent": "claude-code", "folder": None, "encoded": "-Users-x-old-repo",
         "n": len(NOTE_FILES)},
    ]
    assert len(sc.notes_for(str(ws))) == len(NOTE_FILES)
    assert sc.notes_for(str(tmp_path / "elsewhere")) == []


def test_scanner_without_the_claude_reader_reads_no_notes(home, tmp_path):
    from agent_os.onboarding.history import HistoryScanner

    stage_memory(home, str(tmp_path / "ws"))

    class Other:
        agent = "codex"

        def session_files(self, home):
            return []

        def read(self, path):
            return None

    sc = HistoryScanner(None, home=home, sources=[Other()])
    assert sc.wait(10)
    assert sc.status()["notes_by_agent"] == {}
    assert sc.status()["notes_by_folder"] == []


@pytest.fixture
def routes(home, tmp_path, monkeypatch):
    from agent_os.api.routes import onboarding as onboarding_routes
    from agent_os.daemon_v2.project_store import ProjectStore

    monkeypatch.setenv("ORBITAL_HISTORY_HOME", home)
    store = ProjectStore(str(tmp_path / "data"))
    sc = _scanner(home, tmp_path)
    onboarding_routes.configure(project_store=store, scanner=sc)
    app = FastAPI()
    app.include_router(onboarding_routes.router)
    app.include_router(onboarding_routes.projects_router)
    yield TestClient(app), store, sc
    onboarding_routes.configure()


def _ready(client):
    deadline = time.time() + 10
    while time.time() < deadline:
        body = client.get("/api/v2/onboarding/history").json()
        if body["status"] == "ready":
            return body
        time.sleep(0.05)
    raise AssertionError("scan never finished")


def test_history_route_and_importable_rows_carry_notes(routes, home, tmp_path):
    client, _, _ = routes
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "CLAUDE.md").write_text("# rules\n")  # qualifies as a project with one session
    stage_memory(home, str(ws))
    stage_memory(home, None, dirname="-Users-x-old-repo")
    body = _ready(client)
    assert body["notes_by_agent"] == {"claude-code": 2 * len(NOTE_FILES)}
    assert [r["folder"] for r in body["notes_by_folder"]] == [str(ws), None]
    rows = client.get("/api/v2/onboarding/importable-projects").json()["candidates"]
    (row,) = [r for r in rows if r["path"] == str(ws)]
    assert row["notes"] == {"claude-code": len(NOTE_FILES)}


def test_imported_notes_endpoint(routes, home, tmp_path):
    client, store, _ = routes
    ws = tmp_path / "ws"
    ws.mkdir()
    stage_memory(home, str(ws))
    _ready(client)
    pid = store.create_project({"name": "ws", "workspace": str(ws), "imported": True,
                                "deferred_seed": True})
    r = client.get(f"/api/v2/projects/{pid}/imported-notes")
    assert r.status_code == 200
    assert r.json() == {"agent": "claude-code", "count": len(NOTE_FILES), "folded": False,
                        "index_path": "orbital/imported/claude-code/INDEX.md"}
    # Folded = the seed field is cleared AND the imported index exists.
    store.update_project(pid, {"deferred_seed": False})
    assert client.get(f"/api/v2/projects/{pid}/imported-notes").json()["folded"] is False
    idx = ws / "orbital" / "imported" / "claude-code" / "INDEX.md"
    idx.parent.mkdir(parents=True)
    idx.write_text("x")
    assert client.get(f"/api/v2/projects/{pid}/imported-notes").json()["folded"] is True
    assert client.get("/api/v2/projects/proj_nope/imported-notes").status_code == 404


def test_imported_notes_before_the_scan_reads_the_workspace_directly(routes, home, tmp_path):
    client, store, sc = routes
    ws = tmp_path / "ws"
    ws.mkdir()
    stage_memory(home, str(ws))
    pid = store.create_project({"name": "ws", "workspace": str(ws), "imported": True})
    assert not sc.is_ready()  # nobody started it
    assert client.get(f"/api/v2/projects/{pid}/imported-notes").json()["count"] == len(NOTE_FILES)


def test_notes_found_telemetry_on_scan_ready(home, tmp_path, monkeypatch):
    from agent_os import telemetry
    from agent_os.api.routes import onboarding as onboarding_routes

    stage_memory(home, str(tmp_path / "ws"))
    sent = []
    monkeypatch.setattr(telemetry, "emit", lambda event, fields=None: sent.append((event, fields)))
    sc = _scanner(home, tmp_path)
    sc.wait(10)
    onboarding_routes._report_scan(sc)
    assert ("notes_found", {"agent": "claude-code", "n": len(NOTE_FILES)}) in sent


def test_notes_found_rollup_keeps_the_latest_count_per_allowlisted_agent(tmp_path):
    from agent_os.telemetry.identity import InstallIdentity
    from agent_os.telemetry.rollup import build_ping
    from agent_os.telemetry.spool import Spool

    day = "2026-11-04"
    spool = Spool(tmp_path)
    spool._path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"event": "notes_found", "agent": "claude-code", "n": 140},
            {"event": "notes_found", "agent": "claude-code", "n": 145},
            {"event": "notes_found", "agent": "/Users/x/secret", "n": 3},
            {"event": "notes_found", "agent": "codex", "n": "many"}]
    with open(spool._path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"ts": f"{day}T10:00:00+00:00", **r}) + "\n")
    ping = build_ping(InstallIdentity(tmp_path), spool, day, version="x", os_name="darwin")
    assert ping["counters"]["notes_found"] == {"claude-code": 145, "other": 3, "codex": 0}
    empty = build_ping(InstallIdentity(tmp_path / "e"), Spool(tmp_path / "e"), day,
                       version="x", os_name="darwin")
    assert empty["counters"]["notes_found"] == {}
