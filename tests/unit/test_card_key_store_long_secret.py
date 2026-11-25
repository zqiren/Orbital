# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A card's secret longer than one Windows credential holds (2026-11-26).

Windows Credential Manager refuses a secret over 2560 bytes, which keyring
writes as UTF-16: 1280 characters (``CredWrite`` error 1783, measured on the
test PC: 1280 fits, 1300 does not). A ChatGPT sign-in stores its token set
(three JWTs) as the card's secret, several thousand characters, so the
sign-in ended in "连接失败 · 500" after the browser consent, on every Windows
machine. The store now splits a long secret over several items on Windows.
"""

import pytest

from agent_os.daemon_v2 import credential_store as cs

WIN_LIMIT = 1280


class _Vault:
    """An in-memory keyring with the Windows size limit."""

    def __init__(self, limit=WIN_LIMIT):
        self.items: dict[tuple[str, str], str] = {}
        self.limit = limit

    def get_password(self, service, name):
        return self.items.get((service, name))

    def set_password(self, service, name, value):
        if self.limit is not None and len(value) > self.limit:
            raise OSError(1783, "CredWrite", "The stub received bad data.")
        self.items[(service, name)] = value

    def delete_password(self, service, name):
        if (service, name) not in self.items:
            raise KeyError(name)
        del self.items[(service, name)]


@pytest.fixture
def vault(monkeypatch):
    v = _Vault()
    monkeypatch.setattr(cs, "keyring", v, raising=False)
    monkeypatch.setattr(cs, "_KEYRING_AVAILABLE", True)
    monkeypatch.setattr(cs.sys, "platform", "win32")
    return v


LONG = "eyJ" + "".join(chr(97 + i % 26) for i in range(5200)) + ".end"


def test_a_long_secret_round_trips_on_windows(vault):
    store = cs.CardKeyStore()
    assert store.set("card_tok", LONG) == {"source": "keychain"}
    assert store.get("card_tok") == LONG
    assert store.source("card_tok") == "keychain"
    assert all(len(v) <= WIN_LIMIT for v in vault.items.values())
    assert len(vault.items) > 1


def test_deleting_the_card_removes_every_part(vault):
    store = cs.CardKeyStore()
    store.set("card_tok", LONG)
    store.delete("card_tok")
    assert vault.items == {}
    assert store.get("card_tok") is None


def test_a_shorter_secret_replaces_a_long_one_and_leaves_no_parts(vault):
    store = cs.CardKeyStore()
    store.set("card_tok", LONG)
    store.set("card_tok", "sk-short")
    assert store.get("card_tok") == "sk-short"
    assert list(vault.items.values()) == ["sk-short"]


def test_a_longer_secret_replaces_a_long_one(vault):
    store = cs.CardKeyStore()
    store.set("card_tok", LONG)
    store.set("card_tok", LONG + LONG)
    assert store.get("card_tok") == LONG + LONG
    store.set("card_tok", LONG)
    assert store.get("card_tok") == LONG
    store.delete("card_tok")
    assert vault.items == {}


def test_a_missing_part_reads_as_no_secret(vault):
    store = cs.CardKeyStore()
    store.set("card_tok", LONG)
    part = next(k for k in vault.items if k[1].endswith("#2"))
    del vault.items[part]
    assert store.get("card_tok") is None


def test_an_ordinary_api_key_is_stored_as_one_item(vault):
    store = cs.CardKeyStore()
    store.set("card_key", "sk-" + "a" * 160)
    assert list(vault.items) == [(cs._SERVICE_NAME, cs._card_item("card_key"))]


def test_other_systems_keep_one_item(vault, monkeypatch):
    monkeypatch.setattr(cs.sys, "platform", "darwin")
    vault.limit = None
    store = cs.CardKeyStore()
    store.set("card_tok", LONG)
    assert list(vault.items.values()) == [LONG]
    assert store.get("card_tok") == LONG
