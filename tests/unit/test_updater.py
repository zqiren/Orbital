# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""In-app upgrade core (agent_os/updater.py): feed, resumable download,
state machine, startup reconcile, busy probe. Downloads go through a real
loopback HTTP server (tests/unit/_update_http_stub.py)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from pathlib import Path

import httpx
import pytest

from agent_os import updater
from agent_os.updater import (
    ARTIFACTS, FeedEntry, UpdateError, Updater, clean_cache, download,
    resolve_github, resolve_oss, source_order,
)
from tests.unit._update_http_stub import StubServer

MAC = ARTIFACTS["darwin"]
WIN = ARTIFACTS["win32"]
BLOB = bytes(range(256)) * 4096          # 1 MiB
SHA = hashlib.sha256(BLOB).hexdigest()


def client():
    return httpx.AsyncClient(trust_env=False, timeout=10.0, follow_redirects=True)


@pytest.fixture
def stub():
    s = StubServer().start()
    yield s
    s.stop()


def entry(stub, *, sha=SHA, size=len(BLOB), name="Orbital-macOS.dmg", version="0.24.0"):
    return FeedEntry(version, stub.url(name), sha, size, "china")


# ---------------------------------------------------------------------------
# Feed
# ---------------------------------------------------------------------------

def oss_doc(version="v0.24.0", name="Orbital-macOS.dmg", sha=SHA, size=len(BLOB)):
    return {"version": version, "files": {
        "mac": {"name": name, "size": size, "sha256": sha},
        "win": {"name": "Orbital-Setup.exe", "size": size, "sha256": sha}}}


def gh_doc(tag="v0.24.0", digest=f"sha256:{SHA}", name="Orbital-macOS.dmg",
           url="https://github.com/zqiren/Orbital/releases/download/v0.24.0/Orbital-macOS.dmg",
           **extra):
    return {"tag_name": tag, "draft": False, "prerelease": False, **extra, "assets": [
        {"name": name, "size": len(BLOB), "digest": digest, "browser_download_url": url}]}


def fetch_of(mapping):
    async def fetch(url):
        value = mapping[url]
        if isinstance(value, Exception):
            raise value
        return value
    return fetch


class TestFeed:
    async def test_oss_entry(self):
        e = await resolve_oss("0.24.0", MAC, fetch_of({updater.OSS_LATEST_URL: oss_doc()}))
        assert e == FeedEntry("0.24.0", f"{updater.CHINA_DOWNLOAD_BASE}/Orbital-macOS.dmg",
                              SHA, len(BLOB), "china")

    async def test_windows_entry_reads_the_win_file(self):
        e = await resolve_oss("0.24.0", WIN, fetch_of({updater.OSS_LATEST_URL: oss_doc()}))
        assert e.url.endswith("/Orbital-Setup.exe")

    async def test_a_lagging_mirror_is_refused(self):
        # The OSS sync is a manual release step: it may still serve the
        # previous version. Never install something other than what was offered.
        with pytest.raises(UpdateError) as err:
            await resolve_oss("0.24.0", MAC, fetch_of({updater.OSS_LATEST_URL: oss_doc("v0.23.1")}))
        assert err.value.code == "feed"

    @pytest.mark.parametrize("doc", [
        oss_doc(sha="nothex"), oss_doc(sha=""), oss_doc(size=0),
        oss_doc(name="../evil.dmg"), oss_doc(name="Orbital-0.23.0-macOS.dmg"),
    ])
    async def test_bad_mirror_entries_are_refused(self, doc):
        with pytest.raises(UpdateError):
            await resolve_oss("0.24.0", MAC, fetch_of({updater.OSS_LATEST_URL: doc}))

    async def test_unreachable_mirror_is_a_feed_error(self):
        with pytest.raises(UpdateError) as err:
            await resolve_oss("0.24.0", MAC, fetch_of({updater.OSS_LATEST_URL: OSError("x")}))
        assert err.value.code == "feed"

    async def test_github_reads_the_offered_tag_and_its_digest(self):
        url = updater.GITHUB_TAG_API.format(version="0.24.0")
        e = await resolve_github("v0.24.0", MAC, fetch_of({url: gh_doc()}))
        assert e.sha256 == SHA and e.source == "global" and e.version == "0.24.0"

    async def test_github_falls_back_to_the_versioned_asset_name(self):
        url = updater.GITHUB_TAG_API.format(version="0.24.0")
        doc = gh_doc(name="Orbital-0.24.0-macOS.dmg",
                     url="https://github.com/zqiren/Orbital/releases/download/v0.24.0/Orbital-0.24.0-macOS.dmg")
        e = await resolve_github("0.24.0", MAC, fetch_of({url: doc}))
        assert e.url.endswith("Orbital-0.24.0-macOS.dmg")

    @pytest.mark.parametrize("doc", [
        gh_doc(digest=""),                                   # no hash, no install
        gh_doc(digest="md5:abc"),
        gh_doc(url="https://evil.example/Orbital-macOS.dmg"),
        gh_doc(tag="v0.25.0"),
        gh_doc(prerelease=True),
        gh_doc(name="Orbital-Setup.exe"),                    # nothing for this platform
    ])
    async def test_bad_github_entries_are_refused(self, doc):
        url = updater.GITHUB_TAG_API.format(version="0.24.0")
        with pytest.raises(UpdateError) as err:
            await resolve_github("0.24.0", MAC, fetch_of({url: doc}))
        assert err.value.code == "feed"

    def test_source_order_follows_the_ui_region(self):
        assert source_order("china") == ["china", "global"]
        assert source_order("global") == ["global", "china"]
        assert source_order(None) == ["global", "china"]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

class TestDownload:
    async def test_full_download_verifies_and_renames(self, stub, tmp_path):
        stub.files["Orbital-macOS.dmg"] = BLOB
        seen = []
        path = await download(entry(stub), tmp_path, MAC, client_factory=client,
                              on_progress=lambda r, t: seen.append(r))
        assert path == tmp_path / "Orbital-0.24.0.dmg"
        assert path.read_bytes() == BLOB
        assert seen[-1] == len(BLOB)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["Orbital-0.24.0.dmg"]

    async def test_resume_after_an_interrupted_download(self, stub, tmp_path):
        stub.files["Orbital-macOS.dmg"] = BLOB
        stub.cut_after["Orbital-macOS.dmg"] = 300_000
        with pytest.raises(UpdateError) as err:
            await download(entry(stub), tmp_path, MAC, client_factory=client)
        assert err.value.code == "download"
        part = tmp_path / "Orbital-0.24.0.dmg.part"
        kept = part.stat().st_size
        assert 0 < kept < len(BLOB)

        path = await download(entry(stub), tmp_path, MAC, client_factory=client)
        assert path.read_bytes() == BLOB
        last_headers = stub.requests[-1][1]
        assert last_headers["range"] == f"bytes={kept}-"
        assert last_headers["if-range"] == '"v1"'
        assert not part.exists()

    async def test_a_file_replaced_in_place_restarts_from_zero(self, stub, tmp_path):
        # The stable names are overwritten each release: If-Range makes the
        # server send the whole new file instead of splicing two versions.
        stub.files["Orbital-macOS.dmg"] = BLOB
        stub.cut_after["Orbital-macOS.dmg"] = 100_000
        with pytest.raises(UpdateError):
            await download(entry(stub), tmp_path, MAC, client_factory=client)
        stub.etags["Orbital-macOS.dmg"] = '"v2"'
        path = await download(entry(stub), tmp_path, MAC, client_factory=client)
        assert path.read_bytes() == BLOB

    async def test_hash_mismatch_fails_and_removes_the_file(self, stub, tmp_path):
        stub.files["Orbital-macOS.dmg"] = BLOB
        with pytest.raises(UpdateError) as err:
            await download(entry(stub, sha="0" * 64), tmp_path, MAC, client_factory=client)
        assert err.value.code == "hash"
        assert list(tmp_path.iterdir()) == []

    async def test_served_size_differing_from_the_feed_fails(self, stub, tmp_path):
        stub.files["Orbital-macOS.dmg"] = BLOB + b"x"
        with pytest.raises(UpdateError) as err:
            await download(entry(stub), tmp_path, MAC, client_factory=client)
        assert err.value.code == "download"
        assert not (tmp_path / "Orbital-0.24.0.dmg.part").exists()

    async def test_http_error(self, stub, tmp_path):
        with pytest.raises(UpdateError) as err:
            await download(entry(stub, name="missing.dmg"), tmp_path, MAC, client_factory=client)
        assert err.value.code == "download"

    async def test_a_finished_download_is_reused_without_the_network(self, stub, tmp_path):
        (tmp_path / "Orbital-0.24.0.dmg").write_bytes(BLOB)
        path = await download(entry(stub), tmp_path, MAC, client_factory=client)
        assert path.read_bytes() == BLOB and stub.requests == []

    async def test_a_part_for_another_hash_is_discarded(self, stub, tmp_path):
        stub.files["Orbital-macOS.dmg"] = BLOB
        (tmp_path / "Orbital-0.24.0.dmg.part").write_bytes(b"junk" * 100)
        (tmp_path / "Orbital-0.24.0.dmg.part.json").write_text(json.dumps({"sha256": "1" * 64}))
        path = await download(entry(stub), tmp_path, MAC, client_factory=client)
        assert path.read_bytes() == BLOB
        assert "range" not in stub.requests[-1][1]

    async def test_not_enough_space(self, stub, tmp_path, monkeypatch):
        import shutil as _shutil
        monkeypatch.setattr(updater.shutil, "disk_usage",
                            lambda p: _shutil._ntuple_diskusage(10, 10, 10))
        stub.files["Orbital-macOS.dmg"] = BLOB
        with pytest.raises(UpdateError) as err:
            await download(entry(stub), tmp_path, MAC, client_factory=client)
        assert err.value.code == "space" and stub.requests == []

    def test_clean_cache_keeps_only_the_wanted_version(self, tmp_path):
        for name in ("Orbital-0.23.0.dmg", "Orbital-0.24.0.dmg.part",
                     "Orbital-0.24.0.dmg.part.json", "Orbital-0.25.0.dmg", "notes.txt"):
            (tmp_path / name).write_text("x")
        clean_cache(tmp_path, "0.25.0")
        assert sorted(p.name for p in tmp_path.iterdir()) == ["Orbital-0.25.0.dmg", "notes.txt"]


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class FakeInstaller:
    def __init__(self, *, reason=None, prepare_exc=None, outcome="go"):
        self.reason = reason
        self.prepare_exc = prepare_exc
        self.outcome = outcome
        self.prepared = []
        self.handoffs = []
        self.startups = []
        self.cleanups = 0

    def unsupported_reason(self):
        return self.reason

    async def prepare(self, downloaded, version):
        if self.prepare_exc:
            raise self.prepare_exc
        self.prepared.append((downloaded, version))
        return f"staged-{version}"

    async def handoff(self, prepared, version, cache):
        self.handoffs.append((prepared, version))
        return self.outcome

    def on_startup(self, cache, pending, current):
        self.startups.append((pending, current))

    def cleanup(self, cache):
        self.cleanups += 1


class WS:
    def __init__(self):
        self.events = []

    def broadcast_global(self, event):
        self.events.append(event)

    def states(self):
        out = []
        for e in self.events:
            if not out or out[-1] != e["state"]:
                out.append(e["state"])
        return out


@pytest.fixture
def shell():
    calls = {"confirm": [], "exit": 0, "answer": True}

    def confirm():
        calls["confirm"].append(1)
        return calls["answer"]

    def exit_app():
        calls["exit"] += 1

    updater.register_shell(confirm_quit=confirm, exit_app=exit_app)
    yield calls
    updater.clear_shell()


def make(stub, tmp_path, *, installer=None, offered="0.24.0", current="0.23.1",
         china=None, github=None, frozen=True, platform="darwin"):
    ws = WS()
    feeds = {
        updater.OSS_LATEST_URL: china if china is not None else oss_doc(),
        updater.GITHUB_TAG_API.format(version=offered): github if github is not None else gh_doc(),
    }

    async def fetch(url):
        value = feeds[url]
        if isinstance(value, Exception):
            raise value
        return value

    u = Updater(ws, offered=lambda: {"update_available": True, "latest": offered},
                platform=platform, cache_dir=tmp_path, installer=installer or FakeInstaller(),
                fetch_json=fetch, client_factory=client, frozen=frozen,
                current_version=lambda: current)
    return u, ws


@pytest.fixture
def served(stub, monkeypatch):
    """Point both sources' download URLs at the stub."""
    stub.files["Orbital-macOS.dmg"] = BLOB
    monkeypatch.setattr(updater, "CHINA_DOWNLOAD_BASE", stub.base)
    monkeypatch.setattr(updater, "GITHUB_DOWNLOAD_PREFIX", stub.base + "/")
    return stub


async def settle(u):
    if u._task is not None:
        await asyncio.wait_for(u._task, 10)


def gh_on(stub, **kw):
    return gh_doc(url=stub.url("Orbital-macOS.dmg"), **kw)


class TestStateMachine:
    async def test_happy_path_walks_every_state_then_exits(self, served, tmp_path, shell):
        inst = FakeInstaller()
        u, ws = make(served, tmp_path, installer=inst)
        assert u.snapshot()["can_install"] is True
        await u.start("china")
        await settle(u)
        assert ws.states() == ["downloading", "verifying", "ready", "installing"]
        assert inst.handoffs == [("staged-0.24.0", "0.24.0")]
        assert shell["confirm"] == [1] and shell["exit"] == 1
        pending = json.loads((tmp_path / "pending.json").read_text())
        assert pending["target"] == "0.24.0" and pending["previous"] == "0.23.1"
        assert u.state["source"] == "china"
        progress = [e["progress"] for e in ws.events if e["state"] == "downloading"]
        assert progress[-1] == 100 and progress == sorted(progress)

    async def test_dev_build_is_inert(self, served, tmp_path, shell):
        u, _ = make(served, tmp_path, frozen=False)
        assert u.snapshot()["can_install"] is False
        assert u.unsupported_reason() == "dev"
        with pytest.raises(UpdateError) as err:
            await u.start("global")
        assert err.value.code == "unsupported"
        (tmp_path / "pending.json").write_text(json.dumps({"target": "0.23.1", "at": time.time()}))
        u.on_startup()  # touches nothing: the cache is shared with the installed app
        assert (tmp_path / "pending.json").exists()

    async def test_without_the_desktop_shell_nothing_installs(self, served, tmp_path):
        updater.clear_shell()
        u, _ = make(served, tmp_path)
        assert u.unsupported_reason() == "no-shell"
        with pytest.raises(UpdateError):
            await u.start("global")

    async def test_platform_refusal_keeps_the_manual_download(self, served, tmp_path, shell):
        u, _ = make(served, tmp_path, installer=FakeInstaller(reason="translocated"))
        assert u.snapshot()["can_install"] is False
        with pytest.raises(UpdateError):
            await u.start("global")

    @pytest.mark.parametrize("offered,current", [("0.23.1", "0.23.1"), ("0.23.0", "0.23.1")])
    async def test_never_a_downgrade_or_a_reinstall(self, served, tmp_path, shell, offered, current):
        u, _ = make(served, tmp_path, offered=offered, current=current)
        with pytest.raises(UpdateError) as err:
            await u.start("global")
        assert err.value.code == "not_newer"
        assert served.requests == []

    async def test_falls_back_when_the_mirror_lags(self, served, tmp_path, shell):
        u, ws = make(served, tmp_path, china=oss_doc("v0.23.1"), github=gh_on(served))
        await u.start("china")
        await settle(u)
        assert u.state["state"] == "installing" and u.state["source"] == "global"

    async def test_falls_back_when_the_primary_is_unreachable(self, served, tmp_path, shell):
        u, _ = make(served, tmp_path, github=OSError("blocked"))
        await u.start("global")
        await settle(u)
        assert u.state["source"] == "china" and shell["exit"] == 1

    async def test_falls_back_on_a_bad_hash_from_the_primary(self, served, tmp_path, shell):
        u, _ = make(served, tmp_path, china=oss_doc(sha="f" * 64), github=gh_on(served))
        await u.start("china")
        await settle(u)
        assert u.state["source"] == "global" and shell["exit"] == 1

    async def test_both_sources_failing_reports_the_hash(self, served, tmp_path, shell):
        u, ws = make(served, tmp_path, china=oss_doc(sha="f" * 64),
                     github=gh_on(served, digest="sha256:" + "e" * 64))
        await u.start("china")
        await settle(u)
        assert u.state["state"] == "failed" and u.state["error"] == "hash"
        assert shell["exit"] == 0
        assert not any(p.suffix in (".dmg", ".part") for p in tmp_path.iterdir())

    async def test_verify_failure_never_reaches_the_quit(self, served, tmp_path, shell):
        inst = FakeInstaller(prepare_exc=UpdateError("verify", "team id"))
        u, ws = make(served, tmp_path, installer=inst)
        await u.start("global")
        await settle(u)
        assert u.state["error"] == "verify" and shell["confirm"] == [] and shell["exit"] == 0
        assert ws.states()[-1] == "failed"
        assert not (tmp_path / "Orbital-0.24.0.dmg").exists()   # refused file deleted

    async def test_cancelling_the_quit_waits_ready_and_the_next_click_installs(
            self, served, tmp_path, shell):
        inst = FakeInstaller()
        u, ws = make(served, tmp_path, installer=inst)
        shell["answer"] = False
        await u.start("global")
        await settle(u)
        assert u.state["state"] == "ready" and shell["exit"] == 0
        assert not (tmp_path / "pending.json").exists()
        hits = len(served.requests)
        shell["answer"] = True
        await u.start("global")
        await settle(u)
        assert shell["exit"] == 1
        assert len(served.requests) == hits          # no second download
        assert len(inst.prepared) == 1                # no second unpack

    async def test_uac_declined_stays_on_the_old_version(self, served, tmp_path, shell):
        u, _ = make(served, tmp_path, installer=FakeInstaller(outcome="cancelled"))
        await u.start("global")
        await settle(u)
        assert u.state["state"] == "ready" and shell["exit"] == 0
        assert not (tmp_path / "pending.json").exists()

    async def test_a_click_while_downloading_is_idempotent(self, served, tmp_path, shell):
        u, _ = make(served, tmp_path)
        first = u._task
        await u.start("global")
        first = u._task
        await u.start("global")
        assert u._task is first
        await settle(u)

    async def test_a_retry_after_failure_resumes(self, served, tmp_path, shell):
        served.cut_after["Orbital-macOS.dmg"] = 200_000
        u, _ = make(served, tmp_path, github=OSError("x"))
        await u.start("china")
        await settle(u)
        assert (u.state["state"], u.state["error"]) == ("failed", "download")
        await u.start("china")
        await settle(u)
        assert shell["exit"] == 1
        assert served.requests[-1][1].get("range", "").startswith("bytes=")

    def test_force_dev_state(self, tmp_path, stub):
        u, _ = make(stub, tmp_path, frozen=False)
        snap = u.force_dev_state({"state": "downloading", "version": "0.24.0", "progress": 42})
        assert snap["state"] == "downloading" and snap["progress"] == 42
        assert u.force_dev_state(None)["state"] == "idle"


class TestStartupReconcile:
    def write(self, tmp_path, name, data):
        (tmp_path / name).write_text(json.dumps(data))

    def test_the_new_version_up_clears_everything(self, tmp_path, stub):
        inst = FakeInstaller()
        u, _ = make(stub, tmp_path, installer=inst, current="0.24.0")
        self.write(tmp_path, "pending.json", {"target": "0.24.0", "previous": "0.23.1", "at": time.time()})
        (tmp_path / "Orbital-0.24.0.dmg").write_bytes(b"x")
        u.on_startup()
        assert inst.startups and inst.startups[0][1] == "0.24.0"
        assert not (tmp_path / "pending.json").exists()
        assert not (tmp_path / "Orbital-0.24.0.dmg").exists()
        assert u.state["state"] == "idle" and inst.cleanups == 1

    def test_macos_rollback_reports_a_failed_launch(self, tmp_path, stub):
        u, _ = make(stub, tmp_path, current="0.23.1")
        self.write(tmp_path, "pending.json", {"target": "0.24.0", "previous": "0.23.1", "at": time.time()})
        self.write(tmp_path, "result.json", {"result": "rolled_back", "version": "0.24.0"})
        u.on_startup()
        assert u.state["state"] == "failed" and u.state["error"] == "launch"
        assert not (tmp_path / "pending.json").exists()

    def test_macos_old_version_without_a_verdict_leaves_the_helper_alone(self, tmp_path, stub):
        inst = FakeInstaller()
        u, _ = make(stub, tmp_path, installer=inst, current="0.23.1")
        self.write(tmp_path, "pending.json", {"target": "0.24.0", "previous": "0.23.1", "at": time.time()})
        u.on_startup()
        assert (tmp_path / "pending.json").exists() and inst.cleanups == 0
        assert u.state["state"] == "idle"

    def test_windows_setup_abort_reports_a_failed_install(self, tmp_path, stub):
        u, _ = make(stub, tmp_path, current="0.23.1", platform="win32")
        self.write(tmp_path, "pending.json", {"target": "0.24.0", "previous": "0.23.1", "at": time.time()})
        u.on_startup()
        assert u.state["error"] == "install" and not (tmp_path / "pending.json").exists()

    def test_stale_records_are_dropped(self, tmp_path, stub):
        u, _ = make(stub, tmp_path, current="0.23.1")
        self.write(tmp_path, "pending.json", {"target": "0.24.0", "previous": "0.23.1", "at": 1})
        u.on_startup()
        assert not (tmp_path / "pending.json").exists() and u.state["state"] == "idle"

    def test_downloads_not_newer_than_the_running_version_are_removed(self, tmp_path, stub):
        u, _ = make(stub, tmp_path, current="0.24.0")
        for name in ("Orbital-0.23.1.dmg", "Orbital-0.24.0.dmg.part", "Orbital-0.25.0.dmg.part"):
            (tmp_path / name).write_bytes(b"x")
        u.on_startup()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["Orbital-0.25.0.dmg.part"]


# ---------------------------------------------------------------------------
# Busy probe (confirm_quit counts agents mid-turn too)
# ---------------------------------------------------------------------------

class TestBusyProbe:
    async def test_counts_from_another_thread_on_the_daemon_loop(self):
        loop_thread = []

        def probe():
            loop_thread.append(threading.current_thread())
            return 2

        updater.register_work_probe(probe)
        try:
            result = await asyncio.to_thread(updater.busy_agent_count)
            assert result == 2
            assert loop_thread == [threading.main_thread()] or loop_thread[0] is not None
            assert updater.busy_agent_count() == 2  # on the loop itself: direct call
        finally:
            updater.clear_work_probe()

    async def test_a_broken_probe_counts_as_nothing(self):
        updater.register_work_probe(lambda: 1 / 0)
        try:
            assert await asyncio.to_thread(updater.busy_agent_count) == 0
        finally:
            updater.clear_work_probe()

    def test_no_probe_is_zero(self):
        updater.clear_work_probe()
        assert updater.busy_agent_count() == 0

    def test_count_busy_projects(self):
        class Store:
            def list_projects(self):
                return [{"project_id": "a"}, {"project_id": "b"}, {"project_id": "c"}, {}]

        class AM:
            def is_running(self, pid):
                return pid == "a"

        class SAM:
            def has_running_sub_agents(self, pid):
                if pid == "c":
                    raise RuntimeError("boom")
                return pid == "b"

        assert updater.count_busy_projects(Store(), AM(), SAM()) == 2
        assert updater.count_busy_projects(Store(), AM(), None) == 1
