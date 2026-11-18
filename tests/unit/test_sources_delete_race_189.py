# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 189: a source removed while one of its checks is still running stays
removed. The late check must not write its runtime file back, must not push a
``sources.changed`` row for it, and must not put it back in the list. The
invariant is "runtime state exists only for a configured source", held at
the store (``SourcesStore.rt``) and at the one broadcast (``_push_source``)."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from agent_os.sources.adapters import Adapter
from agent_os.sources.core import SourcesCore
from agent_os.sources.model import Item, SourceFailure, Verified
from agent_os.sources.registry import AdapterRegistry
from agent_os.sources.secrets import MemorySecretStore
from agent_os.sources.store import SourcesStore, empty_runtime

ENTRY = {
    "type": "telegram", "family": "bot", "platforms": ["darwin", "win32", "linux"],
    "interval_s": 300, "lossy": False,
    "connect": {"method": "fields", "fields": [
        {"key": "token", "label_key": "fl.btoken", "secret": True, "optional": False,
         "default": None, "placeholder": None, "pair": False}]},
    "kinds": [{"key": "dm", "label_key": "pick.dm", "default": True, "params": []}],
    "setup_link": None, "agent_prompt": None,
}


class SlowPoll(Adapter):
    """A polled adapter whose check waits on ``release`` (a slow service, or
    a calendar read stuck on EventKit) once ``hold`` is set."""
    mode = "poll"
    interval_s = 300
    DESCRIPTOR = ENTRY

    def __init__(self):
        self.items: list = []
        self.hold = False
        self.fail: Exception | None = None
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def verify(self, ctx):
        return Verified(label="bot")

    async def poll(self, ctx):
        if self.hold:
            self.entered.set()
            await self.release.wait()
        if self.fail:
            raise self.fail
        return list(self.items)


class Clock:
    def __init__(self):
        self.t = datetime(2026, 11, 18, 10, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.t


@pytest.fixture
def rig(tmp_path):
    poll = SlowPoll()
    pushed: list[dict] = []
    clock = Clock()
    core = SourcesCore(str(tmp_path), registry=AdapterRegistry(overrides={"telegram": poll}),
                       secret_store=MemorySecretStore(), broadcast=pushed.append, now_fn=clock)
    core.load()

    class R:
        pass
    r = R()
    r.core, r.poll, r.pushed, r.clock, r.dir = core, poll, pushed, clock, str(tmp_path)
    return r


def _items(*ids):
    return [Item(id=i, kind="dm", title=f"title {i}", body=f"body {i}") for i in ids]


def _runtime_file(r, sid):
    return os.path.join(r.dir, "sources", f"{sid}.json")


async def _connected(r):
    """A connected source past its baseline check, holding one recorded id."""
    sid = (await r.core.create({"type": "telegram", "kinds": {"dm": {}},
                                "secrets": {"token": "tok"}}))["id"]
    r.poll.items = _items("old")
    await r.core._poll_one(r.core.store.get(sid))
    assert os.path.exists(_runtime_file(r, sid))
    return sid


def _assert_gone(r, sid):
    assert not os.path.exists(_runtime_file(r, sid)), "the late check wrote the file back"
    removed_at = next(i for i, p in enumerate(r.pushed)
                      if p == {"type": "sources.removed", "source_id": sid})
    late = [p for p in r.pushed[removed_at + 1:]
            if p.get("type") == "sources.changed" and (p.get("source") or {}).get("id") == sid]
    assert late == [], "a row for the removed source was pushed after sources.removed"
    assert sid not in [s["id"] for s in r.core.list()["sources"]]
    assert sid not in r.core.store.runtime
    assert r.core.store.get(sid) is None
    # reading the folder again from disk finds nothing either
    fresh = SourcesStore(r.dir)
    fresh.load()
    assert fresh.get(sid) is None


async def _delete_mid_check(r, sid, start_check):
    r.poll.hold = True
    task = asyncio.create_task(start_check())
    await asyncio.wait_for(r.poll.entered.wait(), timeout=5)
    await r.core.delete(sid)
    r.poll.release.set()
    await asyncio.wait_for(task, timeout=5)


async def test_delete_during_a_timer_check_that_then_succeeds(rig):
    sid = await _connected(rig)
    rig.clock.t += timedelta(hours=1)  # due again
    rig.poll.items = _items("old", "new")  # the late check brings a new item
    await _delete_mid_check(rig, sid, rig.core.run_due_polls)
    _assert_gone(rig, sid)


async def test_delete_during_a_timer_check_that_then_fails(rig):
    sid = await _connected(rig)
    rig.clock.t += timedelta(hours=1)
    rig.poll.fail = SourceFailure("calendar_permission", {})
    await _delete_mid_check(rig, sid, rig.core.run_due_polls)
    _assert_gone(rig, sid)


async def test_delete_during_a_timer_check_that_errors(rig):
    sid = await _connected(rig)
    rig.clock.t += timedelta(hours=1)
    rig.poll.fail = RuntimeError("service fell over")
    await _delete_mid_check(rig, sid, rig.core.run_due_polls)
    _assert_gone(rig, sid)


async def test_delete_during_a_manual_recheck(rig):
    sid = await _connected(rig)
    rig.poll.items = _items("old", "new")

    async def recheck():
        # 「再检查」 answers with the detail; for a source deleted meanwhile
        # that read is a 404, which is what the panel gets today too.
        try:
            await rig.core.check(sid)
        except Exception as e:
            assert getattr(e, "status", None) == 404
    await _delete_mid_check(rig, sid, recheck)
    _assert_gone(rig, sid)


async def test_a_held_config_cannot_push_or_persist_a_removed_source(rig):
    """Every late writer goes through the same two guards, not only the poll:
    a caller holding the old config (stream ingest, route, catch-up) writes
    into a throwaway runtime and pushes nothing."""
    sid = await _connected(rig)
    cfg = rig.core.store.get(sid)
    await rig.core.delete(sid)
    n = len(rig.pushed)
    await rig.core._ingest(cfg, _items("late"), poll=False, persist=True)
    rig.core._set_failure(cfg, SourceFailure("error"))
    await rig.core._persist(sid)
    rig.core._push_source(cfg)
    assert [p for p in rig.pushed[n:] if p.get("type") == "sources.changed"] == []
    _assert_gone(rig, sid)


# ── the store boundary ─────────────────────────────────────────────────────

def test_rt_of_an_unknown_id_is_detached(tmp_path):
    store = SourcesStore(str(tmp_path))
    store.load()
    rt = store.rt("src_0123456789ab")
    assert rt == empty_runtime()
    rt["failure"] = {"code": "error"}
    assert store.runtime == {}
    assert store.rt("src_0123456789ab") == empty_runtime()  # a new throwaway each time


async def test_rt_of_an_unknown_id_never_reaches_disk(tmp_path):
    store = SourcesStore(str(tmp_path))
    store.load()
    store.rt("src_0123456789ab")["last_check_at"] = "2026-11-18T10:37:25Z"
    await store.save_runtime("src_0123456789ab")
    assert not os.path.exists(os.path.join(str(tmp_path), "sources", "src_0123456789ab.json"))


async def test_a_configured_source_without_a_runtime_file_still_polls_and_persists(rig):
    """Risk 1: a source in sources.json whose runtime file is missing still
    gets a stored runtime, as before."""
    sid = await _connected(rig)
    os.unlink(_runtime_file(rig, sid))
    core = SourcesCore(rig.dir, registry=AdapterRegistry(overrides={"telegram": rig.poll}),
                       secret_store=MemorySecretStore(), now_fn=rig.clock)
    core.load()
    core.store.runtime.pop(sid, None)  # as if load had found no entry at all
    assert core.store.get(sid) is not None
    rig.poll.items = _items("old", "new")
    await core._poll_one(core.store.get(sid))
    assert sid in core.store.runtime
    with open(_runtime_file(rig, sid), encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["last_check_at"] is not None
