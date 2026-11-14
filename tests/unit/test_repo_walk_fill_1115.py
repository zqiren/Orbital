# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Batch 1115: the git-repo walk runs in the background and the import list
keeps filling in (owner decision 2026-11-15). The route reports the existing
``scanning`` state until the walk is done, one walk serves every request,
its rows are reused for a while, and a hard bound ends a stuck walk."""

import logging
import os
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.routes import onboarding as routes
from agent_os.onboarding import import_scanner
from agent_os.onboarding.history import clear_git_cache
from agent_os.onboarding.history_paths import HistoryEnv


def _git_init(path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)


class _Hang:
    """``os.scandir`` that blocks on folders named ``zz-hang*`` until released."""

    def __init__(self, monkeypatch):
        self.release = threading.Event()
        real = os.scandir

        def scandir(path):
            if os.path.basename(str(path)).startswith("zz-hang"):
                self.release.wait(10)
            return real(path)

        monkeypatch.setattr(import_scanner.os, "scandir", scandir)


@pytest.fixture
def disk(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    clear_git_cache()
    return home


def _names(rows):
    return {os.path.basename(c.path) for c in rows}


def _until(cond, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


# ---------------------------------------------------------------------------
# The walk itself
# ---------------------------------------------------------------------------


def test_rows_fill_in_while_the_walk_runs(disk, monkeypatch):
    _git_init(disk / "aa" / "first")
    _git_init(disk / "zz-hang" / "later")
    hang = _Hang(monkeypatch)
    walk = import_scanner.RepoWalk(str(disk), env=HistoryEnv.for_home(str(disk))).start()
    try:
        _until(lambda: "first" in _names(walk.rows()))
        assert not walk.finished()
        hang.release.set()
        assert walk.wait(10)
        assert _names(walk.rows()) == {"first", "later"}
    finally:
        hang.release.set()


def test_the_bound_ends_a_stuck_walk_with_what_it_found_and_logs_one_cut_line(disk, monkeypatch, caplog):
    _git_init(disk / "aa" / "first")
    (disk / "zz-hang").mkdir()
    hang = _Hang(monkeypatch)
    caplog.set_level(logging.INFO, logger=import_scanner.logger.name)
    walk = import_scanner.RepoWalk(str(disk), env=HistoryEnv.for_home(str(disk)), bound_s=0.3).start()
    try:
        _until(lambda: "first" in _names(walk.rows()))
        t0 = time.monotonic()
        assert walk.wait(5)
        assert time.monotonic() - t0 < 2
        assert walk.cut and walk.alive()          # the stuck read is still stuck
        assert _names(walk.rows()) == {"first"}
        walk.finished()                           # asked again: no second line
        cut = [r for r in caplog.records if "cut" in r.getMessage()]
        assert len(cut) == 1
    finally:
        hang.release.set()


def test_rows_are_copies_the_list_can_mark_without_touching_the_walk(disk):
    _git_init(disk / "code" / "app")
    walk = import_scanner.RepoWalk(str(disk), env=HistoryEnv.for_home(str(disk))).start()
    assert walk.wait(10)
    (row,) = walk.rows()
    row.already_imported = True
    assert walk.rows()[0].already_imported is False


# ---------------------------------------------------------------------------
# The route: states, one shared walk, the cache rule
# ---------------------------------------------------------------------------


class _Scanner:
    """A history scanner whose readiness the test sets."""

    def __init__(self):
        self.state = "ready"
        self.analysis_obj = SimpleNamespace(projects={}, groups=[], group=lambda _id: None)

    def start(self):
        pass

    def refresh(self):
        pass

    def status(self):
        return {"status": self.state, "sessions_total": 0, "by_agent": {}}

    def analysis(self):
        return self.analysis_obj if self.state == "ready" else None

    def notes(self):
        return []


@pytest.fixture
def client(disk, tmp_path, monkeypatch):
    monkeypatch.setenv("ORBITAL_HISTORY_HOME", str(disk))
    scanner = _Scanner()
    routes.configure(scanner=scanner, data_dir=str(tmp_path / "data"))
    routes._reset_repo_walk()
    starts = []
    real_start = import_scanner.RepoWalk.start

    def counting_start(self):
        starts.append(self)
        return real_start(self)

    monkeypatch.setattr(import_scanner.RepoWalk, "start", counting_start)
    app = FastAPI()
    app.include_router(routes.router)
    c = TestClient(app)
    c.scanner = scanner
    c.starts = starts
    try:
        yield c
    finally:
        for w in starts:
            w.stop = True
        routes._reset_repo_walk()
        routes.configure()


def _get(client):
    return client.get("/api/v2/onboarding/importable-projects").json()


def _repo_names(body):
    return {r["name"] for r in body["candidates"] if r["source"] == "git"}


def test_scanning_with_the_repos_so_far_then_ready_with_all(client, disk, monkeypatch):
    _git_init(disk / "aa" / "first")
    _git_init(disk / "zz-hang" / "later")
    hang = _Hang(monkeypatch)
    try:
        _until(lambda: "first" in _repo_names(_get(client)))
        body = _get(client)
        assert body["status"] == "scanning"
        assert _repo_names(body) == {"first"}
        hang.release.set()
        _until(lambda: _get(client)["status"] == "ready")
        assert _repo_names(_get(client)) == {"first", "later"}
    finally:
        hang.release.set()
    assert len(client.starts) == 1


def test_the_walk_starts_with_the_first_request_while_the_history_is_still_read(client, disk):
    _git_init(disk / "code" / "app")
    client.scanner.state = "scanning"
    body = _get(client)
    assert body["status"] == "scanning" and _repo_names(body) == set()
    assert len(client.starts) == 1
    assert client.starts[0].wait(10)
    client.scanner.state = "ready"
    body = _get(client)
    assert body["status"] == "ready" and _repo_names(body) == {"app"}
    assert len(client.starts) == 1


def test_concurrent_requests_join_one_walk(client, disk, monkeypatch):
    _git_init(disk / "zz-hang" / "r")
    hang = _Hang(monkeypatch)
    try:
        threads = [threading.Thread(target=_get, args=(client,)) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        assert _get(client)["status"] == "scanning"
        assert len(client.starts) == 1
    finally:
        hang.release.set()


def test_a_finished_walk_is_reused_while_fresh_and_redone_after(client, disk, monkeypatch):
    _git_init(disk / "code" / "app")
    _until(lambda: _get(client)["status"] == "ready")
    _get(client)
    _get(client)
    assert len(client.starts) == 1                # reopening inside the window

    _git_init(disk / "code" / "cloned")           # cloned since the last walk
    _git_init(disk / "zz-hang" / "slow")
    hang = _Hang(monkeypatch)
    monkeypatch.setattr(routes, "_REPO_FRESH_SECONDS", 0.0)
    try:
        body = _get(client)
        # The new walk runs; the rows of the last one are offered meanwhile.
        assert body["status"] == "scanning" and "app" in _repo_names(body)
        assert len(client.starts) == 2
        monkeypatch.setattr(routes, "_REPO_FRESH_SECONDS", 120.0)
        _until(lambda: "cloned" in _repo_names(_get(client)))
        assert _get(client)["status"] == "scanning"
        hang.release.set()
        _until(lambda: _get(client)["status"] == "ready")
        assert _repo_names(_get(client)) == {"app", "cloned", "slow"}
    finally:
        hang.release.set()
    assert len(client.starts) == 2


def test_a_cut_walk_still_stuck_is_not_walked_again(client, disk, monkeypatch):
    _git_init(disk / "aa" / "first")
    (disk / "zz-hang").mkdir()
    hang = _Hang(monkeypatch)
    monkeypatch.setattr(import_scanner, "WALK_BOUND_S", 0.3)
    monkeypatch.setattr(routes, "_REPO_FRESH_SECONDS", 0.0)
    try:
        _until(lambda: _get(client)["status"] == "ready")
        body = _get(client)
        assert body["status"] == "ready" and _repo_names(body) == {"first"}
        assert len(client.starts) == 1            # stale, but its thread is stuck
    finally:
        hang.release.set()
    _until(lambda: not client.starts[0].alive())
    _get(client)
    assert len(client.starts) == 2                # once it ends, the next one may run
