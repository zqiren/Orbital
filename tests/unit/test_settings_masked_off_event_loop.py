# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The settings routes read card keys off the event loop.

Spec 171 A1: ``logs/hang.log`` on the owner's machine caught the loop thread
inside ``keyring … find_generic_password`` ← ``settings_store.key_for`` ←
``masked_card`` ← ``get_masked`` in ``GET /settings``: one Keychain read per
card, on the loop, so chat and every project froze while it ran. The masked
reads now run in a worker thread. No key cache (spec 171 §4 A1). The fake
store below records the thread each masked read ran on.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.api.routes import settings as settings_routes


def _recording_store(seen: list[int]) -> MagicMock:
    store = MagicMock()

    def record(value):
        def _read(*_a, **_k):
            seen.append(threading.get_ident())
            return value
        return _read

    store.get_masked.side_effect = record({"credential_cards": []})
    store.masked_cards.side_effect = record([])
    store.masked_card.side_effect = record({"id": "c1"})
    store.effective_default_card_id.return_value = "c1"
    store.resolve_card.return_value = SimpleNamespace(
        id="c1", provider="anthropic", model="m", base_url=None, name="n")
    return store


@pytest.fixture
def seen(monkeypatch):
    out: list[int] = []
    monkeypatch.setattr(settings_routes, "_settings_store", _recording_store(out))
    monkeypatch.setattr(settings_routes, "_test_and_record",
                        AsyncMock(return_value={"ok": False}))
    monkeypatch.setattr(settings_routes, "_refresh_agents_in_background",
                        lambda *_a, **_k: None)
    monkeypatch.setattr(settings_routes, "_import_scan_switch",
                        lambda *_a, **_k: None)
    return out


def _off_loop(seen: list[int]) -> None:
    assert seen, "the masked read did not run"
    assert threading.get_ident() not in seen, "a Keychain read ran on the event loop"


@pytest.mark.asyncio
async def test_get_settings_reads_keys_off_the_loop(seen, monkeypatch):
    from agent_os.onboarding import import_scan
    monkeypatch.setattr(import_scan, "unscanned_count", lambda _ps: 0)
    monkeypatch.setattr(import_scan, "scan_agent", lambda _am, _ps: None)
    out = await settings_routes.get_settings()
    assert out["credential_cards"] == []
    _off_loop(seen)


@pytest.mark.asyncio
async def test_put_settings_reads_keys_off_the_loop(seen):
    out = await settings_routes.update_settings(settings_routes.UpdateSettingsRequest())
    assert out == {"credential_cards": []}
    _off_loop(seen)


@pytest.mark.asyncio
async def test_list_cards_reads_keys_off_the_loop(seen):
    out = await settings_routes.list_cards()
    assert out == {"credential_cards": [], "default_card_id": "c1"}
    _off_loop(seen)


@pytest.mark.asyncio
async def test_card_test_reads_the_card_key_off_the_loop(seen):
    out = await settings_routes.test_card("c1")
    assert out["card"] == {"id": "c1"}
    _off_loop(seen)


@pytest.mark.asyncio
async def test_card_update_reads_the_card_key_off_the_loop(seen):
    out = await settings_routes.update_card(
        "c1", settings_routes.UpdateCardRequest(name="renamed"))
    assert out["card"] == {"id": "c1"}
    _off_loop(seen)
