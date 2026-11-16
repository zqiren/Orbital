# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""settings.json can never be replaced by defaults (2026-11-16 data loss).

Four browsers loaded the app at once; each load sends
``PUT /settings/ui-language``. The route read settings on the event loop and
wrote them from a worker thread, and the store wrote with a truncating
``open(path, "w")`` and answered any unreadable read with a fresh default
``GlobalSettings()``. A read that landed while another request's write had
truncated the file got defaults, set the language on them, and wrote the
defaults back: the credential cards, ``default_card_id`` and the skip flag
were gone and the app fell back to onboarding.

The invariants pinned here:
- a write is atomic (temp file + replace): a reader sees the old file or the
  new one, never a truncated one, and an interrupted write leaves the old
  file in place;
- a file that exists but cannot be read is never overwritten in place: an
  unparseable one is moved aside (bytes kept) and the store starts fresh, one
  that cannot be opened (permission, I/O) stays and every write refuses;
- every read-modify-write runs under one lock, so concurrent writers cannot
  lose each other's changes.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.daemon_v2 import settings_store as ss_mod
from agent_os.utils import atomic_write
from agent_os.daemon_v2.settings_store import SettingsStore


class _Keys:
    """In-memory per-card key store (a Keychain stand-in)."""

    def __init__(self):
        self.keys: dict[str, str] = {}

    def get(self, card_id):
        return self.keys.get(card_id)

    def set(self, card_id, key):
        if not key:
            self.keys.pop(card_id, None)
            return {"source": "none"}
        self.keys[card_id] = key
        return {"source": "keychain"}

    def delete(self, card_id):
        self.keys.pop(card_id, None)

    def source(self, card_id):
        return "keychain" if card_id in self.keys else "none"

    def get_legacy(self):
        return None

    def delete_legacy(self):
        return None


@pytest.fixture(autouse=True)
def _no_env_key(monkeypatch):
    monkeypatch.delenv("AGENT_OS_API_KEY", raising=False)


def _store(tmp_path) -> SettingsStore:
    return SettingsStore(data_dir=str(tmp_path / "data"),
                         card_key_store=_Keys(), migrate=False)


def _seed(store: SettingsStore):
    """A configured install: one card (the default) and the skip flag."""
    card = store.create_card(provider="deepseek", model="deepseek-chat",
                             api_key="sk-abcdefgh12345678")
    raw = json.loads(open(store._path, encoding="utf-8").read())
    raw["setup_api_key_skipped"] = True
    with open(store._path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)
    return card


def _on_disk(store: SettingsStore) -> dict:
    with open(store._path, encoding="utf-8") as f:
        return json.load(f)


def _client(store, monkeypatch):
    from agent_os.api.routes import settings as settings_routes
    settings_routes.configure(store)
    monkeypatch.setattr(settings_routes, "_refresh_agents_in_background",
                        lambda *a: None)
    monkeypatch.setattr(settings_routes, "_model_provider_slugs", lambda: [])
    app = FastAPI()
    app.include_router(settings_routes.router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# The reported race
# ---------------------------------------------------------------------------


def test_concurrent_ui_language_updates_keep_the_cards(tmp_path, monkeypatch):
    """The page-load storm: many ui-language PUTs with differing languages,
    interleaved with reads, must leave cards, default and skip flag intact."""
    from agent_os.api.routes import settings as settings_routes

    store = _store(tmp_path)
    card = _seed(store)
    settings_routes.configure(store)

    stop = threading.Event()
    reads_without_cards = []

    def reader():
        while not stop.is_set():
            s = store.get()
            if not s.credential_cards:
                reads_without_cards.append(1)

    readers = [threading.Thread(target=reader) for _ in range(3)]
    for t in readers:
        t.start()

    async def storm():
        reqs = [
            settings_routes.set_ui_language(settings_routes.UiLanguageRequest(
                ui_language=("zh" if i % 2 else "en"),
                ui_languages=(["zh-CN", "en"] if i % 3 else ["en-US"])))
            for i in range(300)
        ]
        await asyncio.gather(*reqs, return_exceptions=False)

    try:
        asyncio.run(storm())
    finally:
        stop.set()
        for t in readers:
            t.join()

    disk = _on_disk(store)
    assert [c["id"] for c in disk["credential_cards"]] == [card.id]
    assert disk["default_card_id"] == card.id
    assert disk["setup_api_key_skipped"] is True
    # No in-process reader ever saw an install without its cards.
    assert reads_without_cards == []


def test_a_reader_never_sees_a_partly_written_file(tmp_path):
    """While a write is in flight the path holds the old bytes or the new
    ones, never a truncated or half-written file."""
    store = _store(tmp_path)
    _seed(store)
    bad = []
    stop = threading.Event()

    def raw_reader():
        while not stop.is_set():
            try:
                with open(store._path, encoding="utf-8") as f:
                    json.load(f)
            except FileNotFoundError:
                bad.append("missing")
            except ValueError:
                bad.append("partial")

    t = threading.Thread(target=raw_reader)
    t.start()
    try:
        for i in range(300):
            store.mutate(lambda s, i=i: setattr(s, "ui_language", f"l{i}"))
    finally:
        stop.set()
        t.join()
    assert bad == []


# ---------------------------------------------------------------------------
# Unreadable is not "no settings"
# ---------------------------------------------------------------------------


def _backups(store) -> list[str]:
    d = os.path.dirname(store._path)
    return sorted(n for n in os.listdir(d)
                  if n.startswith("settings.json.unreadable-"))


def _write_bytes(store, blob: bytes) -> None:
    os.makedirs(os.path.dirname(store._path), exist_ok=True)
    with open(store._path, "wb") as f:
        f.write(blob)


@pytest.mark.parametrize("blob", [
    b"",
    b'{"credential_cards": [{"id": "card_',
    b"not json",
    b"[1, 2]",
    b"\xff\xfe{ not utf-8",
])
def test_an_unparseable_file_is_moved_aside_and_saves_work_again(
        tmp_path, monkeypatch, blob):
    """Disk damage or a hand edit: the bytes are kept beside it under a new
    name, never written over in place, and the app carries on fresh."""
    store = _store(tmp_path)
    _write_bytes(store, blob)

    c = _client(store, monkeypatch)
    r = c.put("/api/v2/settings/ui-language", json={"ui_language": "zh"})
    assert r.status_code == 200

    backups = _backups(store)
    assert len(backups) == 1
    with open(os.path.join(os.path.dirname(store._path), backups[0]), "rb") as f:
        assert f.read() == blob
    assert _on_disk(store)["ui_language"] == "zh"
    # Saves keep working from here on.
    store.create_card(provider="deepseek", model="m", api_key="sk-x")
    assert len(_on_disk(store)["credential_cards"]) == 1


def test_a_second_unparseable_file_gets_its_own_backup_name(tmp_path):
    store = _store(tmp_path)
    _write_bytes(store, b"first garbage")
    store.get()
    _write_bytes(store, b"second garbage")
    store.get()
    d = os.path.dirname(store._path)
    backups = _backups(store)
    assert len(backups) == 2
    contents = sorted(open(os.path.join(d, b), "rb").read() for b in backups)
    assert contents == [b"first garbage", b"second garbage"]


def test_a_file_with_a_bom_still_reads(tmp_path):
    """A hand edit in an editor that saves a UTF-8 BOM is not damage."""
    store = _store(tmp_path)
    card = _seed(store)
    raw = open(store._path, "rb").read()
    _write_bytes(store, b"\xef\xbb\xbf" + raw)
    assert store.get().default_card_id == card.id
    assert _backups(store) == []


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="needs POSIX permissions as a non-root user")
def test_a_file_that_cannot_be_opened_is_not_moved_and_refuses_writes(
        tmp_path, monkeypatch):
    """Permission / I/O errors may be transient and the bytes may be fine:
    no rename, and nothing is written over them."""
    store = _store(tmp_path)
    _seed(store)
    before = open(store._path, "rb").read()
    os.chmod(store._path, 0)
    try:
        c = _client(store, monkeypatch)
        r = c.put("/api/v2/settings/ui-language", json={"ui_language": "zh"})
        assert r.status_code == 500
        assert r.json()["detail"]["code"] == "settings_unreadable"
        with pytest.raises(ss_mod.SettingsUnreadableError):
            store.mutate(lambda s: setattr(s, "telemetry_enabled", False))
        with pytest.raises(ss_mod.SettingsUnreadableError):
            store.update(ss_mod.GlobalSettings())
        with pytest.raises(ss_mod.SettingsUnreadableError):
            store.create_card(provider="deepseek", model="m", api_key="sk-x")
        with pytest.raises(ss_mod.SettingsUnreadableError):
            store.set_default_card("card_x")
        # Readers still get an answer: the daemon keeps serving.
        assert store.get().credential_cards is None
    finally:
        os.chmod(store._path, 0o600)
    assert open(store._path, "rb").read() == before
    assert _backups(store) == []


def test_a_json_shape_this_version_cannot_load_is_kept_in_place(tmp_path):
    """Valid JSON this version cannot load may be a newer version's file
    after a downgrade: it is neither moved nor written over."""
    store = _store(tmp_path)
    blob = b'{"credential_cards": "a newer shape"}'
    _write_bytes(store, blob)
    with pytest.raises(ss_mod.SettingsUnreadableError):
        store.mutate(lambda s: setattr(s, "ui_language", "zh"))
    assert open(store._path, "rb").read() == blob
    assert _backups(store) == []


def test_an_unparseable_file_at_startup_is_moved_aside_and_projects_keep_their_cards(
        tmp_path):
    """The one-shot card migration then runs as on a fresh install, but a
    project already on a card keeps pointing at it, so restoring the backup
    by hand brings everything back."""
    from agent_os.daemon_v2.project_store import ProjectStore

    data = tmp_path / "data"
    data.mkdir()
    projects = ProjectStore(str(data))
    pid = projects.create_project({"name": "P", "workspace": str(tmp_path / "ws")})
    projects.update_project(pid, {"card_id": "card_abc12345"})
    path = data / "settings.json"
    path.write_bytes(b'{"credential_cards": [')

    SettingsStore(data_dir=str(data), card_key_store=_Keys(),
                  project_store=projects, migrate=True)

    backups = [n for n in os.listdir(data) if n.startswith("settings.json.unreadable-")]
    assert len(backups) == 1
    assert (data / backups[0]).read_bytes() == b'{"credential_cards": ['
    assert json.loads(path.read_text(encoding="utf-8"))["credential_cards"] == []
    assert ProjectStore(str(data)).get_project(pid)["card_id"] == "card_abc12345"


def test_an_absent_file_still_starts_from_defaults(tmp_path, monkeypatch):
    store = _store(tmp_path)
    assert not os.path.exists(store._path)
    c = _client(store, monkeypatch)
    r = c.put("/api/v2/settings/ui-language", json={"ui_language": "zh"})
    assert r.status_code == 200
    disk = _on_disk(store)
    assert disk["ui_language"] == "zh"
    assert disk["telemetry_enabled"] is True


# ---------------------------------------------------------------------------
# Interrupted writes
# ---------------------------------------------------------------------------


def test_an_interrupted_write_leaves_the_old_file(tmp_path, monkeypatch):
    """A failure between the temp write and the replace (a crash, a Windows
    replace that never succeeds) leaves the old file byte-identical and no
    temp file behind."""
    store = _store(tmp_path)
    card = _seed(store)
    before = open(store._path, "rb").read()

    def boom(src, dst):
        raise PermissionError(13, "simulated: target held open")

    monkeypatch.setattr(ss_mod.os, "replace", boom)
    monkeypatch.setattr(atomic_write, "REPLACE_RETRY_S", 0.05)
    with pytest.raises(OSError):
        store.mutate(lambda s: setattr(s, "ui_language", "zh"))
    monkeypatch.undo()

    assert open(store._path, "rb").read() == before
    leftovers = [n for n in os.listdir(os.path.dirname(store._path))
                 if n != "settings.json"]
    assert leftovers == []
    assert store.get().default_card_id == card.id


def test_a_stray_temp_file_from_a_crash_is_ignored_and_cleaned(tmp_path):
    """A process killed after writing the temp file but before the replace
    leaves a stray temp: the next start reads the real file and sweeps it."""
    store = _store(tmp_path)
    card = _seed(store)
    data_dir = os.path.dirname(store._path)
    stray = os.path.join(data_dir, ".settings.json.deadbeef.tmp")
    with open(stray, "w", encoding="utf-8") as f:
        f.write('{"half": ')
    old = time.time() - 3600
    os.utime(stray, (old, old))

    again = SettingsStore(data_dir=data_dir, card_key_store=_Keys(),
                          migrate=False)
    assert again.get().default_card_id == card.id
    assert not os.path.exists(stray)


def test_replace_retries_a_briefly_held_target(tmp_path, monkeypatch):
    """Windows: replacing a file another handle has open fails with a
    sharing violation for a moment. The write retries rather than falling
    back to a truncating write."""
    store = _store(tmp_path)
    _seed(store)
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(13, "simulated: sharing violation")
        return real(src, dst)

    monkeypatch.setattr(ss_mod.os, "replace", flaky)
    store.mutate(lambda s: setattr(s, "ui_language", "zh"))
    monkeypatch.undo()
    assert calls["n"] == 3
    assert _on_disk(store)["ui_language"] == "zh"


# ---------------------------------------------------------------------------
# One lock around every read-modify-write
# ---------------------------------------------------------------------------


def test_a_card_created_during_a_slow_mutation_is_not_lost(tmp_path):
    """Lost update: a mutation that read settings before a card was created
    must not write its stale snapshot over the new card."""
    store = _store(tmp_path)
    _seed(store)
    entered = threading.Event()
    release = threading.Event()

    def slow(s):
        entered.set()
        release.wait(5)
        s.ui_language = "zh"

    t = threading.Thread(target=store.mutate, args=(slow,))
    t.start()
    assert entered.wait(5)
    created = {}

    def create():
        created["card"] = store.create_card(provider="openai", model="gpt-x",
                                            api_key="sk-other-12345678")

    t2 = threading.Thread(target=create)
    t2.start()
    time.sleep(0.1)
    release.set()
    t.join(5)
    t2.join(5)

    disk = _on_disk(store)
    ids = [c["id"] for c in disk["credential_cards"]]
    assert created["card"].id in ids and len(ids) == 2
    assert disk["ui_language"] == "zh"
    assert disk["setup_api_key_skipped"] is True


def test_a_mutation_that_changes_nothing_does_not_write(tmp_path):
    store = _store(tmp_path)
    _seed(store)
    before = os.stat(store._path).st_mtime_ns
    time.sleep(0.01)
    store.mutate(lambda s: False)
    assert os.stat(store._path).st_mtime_ns == before


def test_the_file_format_is_unchanged(tmp_path):
    """Same JSON shape, same indent: a v0.22.0 store reads it as before."""
    store = _store(tmp_path)
    _seed(store)
    store.mutate(lambda s: setattr(s, "ui_language", "zh"))
    text = open(store._path, encoding="utf-8").read()
    data = json.loads(text)
    assert text == json.dumps(data, indent=2)
    assert data["llm"] == {"fallback_models": []}
