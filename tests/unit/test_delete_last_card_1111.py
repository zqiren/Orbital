# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""1111 D2: deleting the last card is allowed and leaves the install in the
existing keyless mode (``setup.api_key_skipped``), so the wizard never comes
back. The default card is still protected while other cards exist."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.daemon_v2.project_store import ProjectStore
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


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    from agent_os.api.routes import settings as settings_routes
    store = SettingsStore(data_dir=str(tmp_path / "data"),
                          card_key_store=_Keys(), migrate=False)
    projects = ProjectStore(str(tmp_path / "data"))
    settings_routes.configure(store, project_store=projects)
    monkeypatch.setattr(settings_routes, "_refresh_agents_in_background", lambda *a: None)
    monkeypatch.setattr(settings_routes, "_model_provider_slugs", lambda: [])
    app = FastAPI()
    app.include_router(settings_routes.router)
    return TestClient(app), store, projects


def _card(store, key="sk-abcdefgh12345678"):
    return store.create_card(provider="deepseek", model="deepseek-chat", api_key=key)


def test_deleting_the_last_card_goes_keyless(env, tmp_path):
    tc, store, projects = env
    card = _card(store)
    ws = tmp_path / "ws"
    ws.mkdir()
    pid = projects.create_project({"name": "p", "workspace": str(ws), "card_id": card.id})
    assert store.effective_default_card_id() == card.id

    r = tc.delete(f"/api/v2/settings/cards/{card.id}")
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] is True
    assert store.default_card() is None
    body = tc.get("/api/v2/settings").json()
    assert body["llm"]["api_key_set"] is False
    assert body["setup"]["api_key_skipped"] is True   # the wizard stays away
    p = projects.get_project(pid)
    assert p["card_id"] is None and p["migration_note"].startswith("card_deleted:")


def test_the_default_is_still_protected_while_other_cards_exist(env):
    tc, store, _projects = env
    first = _card(store)
    second = _card(store, key="sk-zyxwvuts87654321")
    default = store.effective_default_card_id()
    other = second.id if default == first.id else first.id

    r = tc.delete(f"/api/v2/settings/cards/{default}")
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "card_is_default"

    r = tc.delete(f"/api/v2/settings/cards/{other}")
    assert r.status_code == 200, r.text
    assert tc.get("/api/v2/settings").json()["setup"]["api_key_skipped"] is False
