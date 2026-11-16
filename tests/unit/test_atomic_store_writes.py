# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The small JSON stores write atomically (2026-11-16 follow-up).

Each of these stores answers an unparseable file with its defaults and makes
them permanent on its next save. A truncating write (or a fixed-name temp
without fsync) is what makes "unparseable" reachable: a crash or a reader in
the middle of the write. They now all save through
``agent_os.utils.atomic_write.atomic_write_text``. Per store: an interrupted
write (the replace step fails) leaves the old file byte-identical with no
temp file beside it, and a normal save writes the same bytes as before.
"""

from __future__ import annotations

import json
import os

import pytest

from agent_os.utils import atomic_write


def _break_replace(monkeypatch):
    def boom(src, dst):
        raise OSError(5, "simulated crash between temp write and replace")
    monkeypatch.setattr(atomic_write, "_replace", boom)


def _assert_untouched(path: str, before: bytes) -> None:
    with open(path, "rb") as f:
        assert f.read() == before
    d = os.path.dirname(path)
    stray = [n for n in os.listdir(d)
             if n.startswith(atomic_write.tmp_prefix(path))]
    assert stray == []


def _text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


# -- the helper itself --------------------------------------------------------


def test_helper_keeps_an_existing_files_mode_and_cleans_up(tmp_path, monkeypatch):
    p = str(tmp_path / "x.json")
    atomic_write.atomic_write_text(p, "{}")
    os.chmod(p, 0o640)
    atomic_write.atomic_write_text(p, '{"a": 1}')
    assert _text(p) == '{"a": 1}'
    if os.name != "nt":
        assert os.stat(p).st_mode & 0o777 == 0o640
    _break_replace(monkeypatch)
    with pytest.raises(OSError):
        atomic_write.atomic_write_text(p, '{"a": 2}')
    _assert_untouched(p, b'{"a": 1}')


def test_helper_retries_a_briefly_held_target(tmp_path, monkeypatch):
    p = str(tmp_path / "x.json")
    atomic_write.atomic_write_text(p, "{}")
    real, calls = os.replace, {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(13, "simulated sharing violation")
        return real(src, dst)

    monkeypatch.setattr(atomic_write.os, "replace", flaky)
    atomic_write.atomic_write_text(p, '{"b": 1}')
    assert calls["n"] == 3 and _text(p) == '{"b": 1}'


# -- credential-meta.json -----------------------------------------------------


def test_credential_meta_interrupted_write_keeps_old_file(tmp_path, monkeypatch):
    from agent_os.daemon_v2.credential_store import UserCredentialStore

    path = str(tmp_path / "credential-meta.json")
    store = UserCredentialStore(meta_path=path)
    store._meta = {"github": {"domain": "github.com", "fields": ["password"]}}
    store._save_meta()
    assert _text(path) == json.dumps(store._meta, indent=2)
    before = open(path, "rb").read()

    _break_replace(monkeypatch)
    store._meta = {}
    with pytest.raises(OSError):
        store._save_meta()
    _assert_untouched(path, before)


# -- paired_devices.json ------------------------------------------------------


def test_paired_devices_interrupted_write_keeps_old_file(tmp_path, monkeypatch):
    from agent_os.api.routes import pairing

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    devices = [{"phone_id": "ph_1", "paired_at": "2026-11-16T00:00:00+00:00"}]
    pairing._save_devices(devices)
    path = str(pairing._devices_file())
    assert path.startswith(str(tmp_path))
    assert _text(path) == json.dumps(devices, indent=2)
    assert pairing._load_devices() == devices
    before = open(path, "rb").read()

    _break_replace(monkeypatch)
    with pytest.raises(OSError):
        pairing._save_devices([])
    _assert_untouched(path, before)


# -- connectors.json ----------------------------------------------------------


def test_connectors_interrupted_write_keeps_old_file(tmp_path, monkeypatch):
    from agent_os.connectors.manager import ConnectorManager
    from agent_os.connectors.manifest import custom_manifest

    mgr = ConnectorManager(catalog=[], data_dir=str(tmp_path))
    m = custom_manifest(name="Mine", url="https://mcp.example.com/mcp", auth_type="none")
    mgr._custom[m.id] = m
    mgr._save_custom()
    path = mgr._custom_path()
    assert _text(path) == json.dumps({"custom": [m.to_dict()]}, indent=2,
                                     ensure_ascii=False)
    before = open(path, "rb").read()

    _break_replace(monkeypatch)
    mgr._custom.clear()
    with pytest.raises(OSError):
        mgr._save_custom()
    _assert_untouched(path, before)


# -- sub_agent_config.json ----------------------------------------------------


def test_sub_agent_config_interrupted_write_keeps_old_file(tmp_path, monkeypatch):
    from agent_os.daemon_v2.sub_agent_config_store import SubAgentConfigStore

    path = str(tmp_path / "sub_agent_config.json")
    store = SubAgentConfigStore(path)
    store._data = {"codex": {"binary": "/opt/codex", "model": "gpt-6-sol"}}
    store._save()
    assert _text(path) == json.dumps(store._data, indent=2, ensure_ascii=False)
    before = open(path, "rb").read()

    _break_replace(monkeypatch)
    store._data = {}
    with pytest.raises(OSError):
        store._save()
    _assert_untouched(path, before)


# -- queue.json ---------------------------------------------------------------


def test_queue_interrupted_write_keeps_old_file(tmp_path, monkeypatch):
    from agent_os.queue.models import QueueRunState
    from agent_os.queue.store import QueueStore

    path = tmp_path / "orbital" / "queue.json"
    store = QueueStore(path)
    state = store.load()  # creates the file
    state.state = QueueRunState.PAUSED
    store.save()
    assert _text(str(path)) == state.model_dump_json(indent=2)
    before = path.read_bytes()

    _break_replace(monkeypatch)
    state.state = QueueRunState.RUNNING
    with pytest.raises(OSError):
        store.save()
    _assert_untouched(str(path), before)


# -- calendar_links.json ------------------------------------------------------


def test_calendar_links_interrupted_write_keeps_old_file(tmp_path, monkeypatch):
    from agent_os.calendar_hub.linkage import Linkage

    links = Linkage(data_dir=str(tmp_path))
    links.link("evt_1", "proj_1")
    path = links._path()
    assert _text(path) == json.dumps({"evt_1": "proj_1"}, indent=2,
                                     ensure_ascii=False)
    before = open(path, "rb").read()

    _break_replace(monkeypatch)
    with pytest.raises(OSError):
        links.link("evt_2", "proj_2")
    _assert_untouched(path, before)
