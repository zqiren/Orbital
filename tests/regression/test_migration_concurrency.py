# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 083 — the credential-card migration must never lose a key.

Reproduces the 2026-09-04 field incident. Two daemon instances started 377 ms
apart (the PID file let the second through because it judged the recorded PID
dead), and BOTH ran the one-shot migration. Each minted its own card ids, wrote
Keychain items under them, and saved its own settings; the last writer won. The
surviving settings.json named one instance's card ids while the keychain held
the other's, so every lookup missed — and because each instance's own writes had
succeeded, nothing logged an error. Meanwhile the migration had already stripped
the plaintext keys out of projects.json, which was the only remaining copy.

Every test here fails on the pre-fix code.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import pytest

from agent_os.daemon_v2.settings_store import SettingsStore
from agent_os.utils.pid_file import DaemonAlreadyRunning, acquire_pid_file


class _FakeCards:
    """A keychain double whose failures are controllable per card."""

    def __init__(self, fail_ids: set[str] | None = None, silent: bool = False):
        self.store: dict[str, str] = {}
        self.fail_ids = fail_ids or set()
        # `silent` models CI's null backend and a locked keychain: the write is
        # ACCEPTED and reads back empty. "No exception" is not evidence.
        self.silent = silent
        self.legacy_deleted = False

    def get(self, card_id):
        return self.store.get(card_id)

    def set(self, card_id, key):
        if card_id in self.fail_ids:
            raise RuntimeError("keychain unavailable")
        if not self.silent:
            self.store[card_id] = key
        return {"source": "keychain"}

    def delete(self, card_id):
        self.store.pop(card_id, None)

    def source(self, card_id):
        return "keychain" if card_id in self.store else "none"

    def get_legacy(self):
        return None

    def delete_legacy(self):
        self.legacy_deleted = True


class _Projects:
    """Project store double that strips legacy fields on save, as the real one
    does — that strip is what destroyed the keys."""

    LEGACY = ("api_key", "model", "provider", "base_url", "sdk")

    def __init__(self, rows):
        self.rows = rows

    def list_projects(self):
        return list(self.rows.values())

    def get_project(self, pid):
        return self.rows.get(pid)

    def update_project(self, pid, updates):
        row = self.rows[pid]
        row.update(updates)
        for f in self.LEGACY:
            row.pop(f, None)


def _legacy_dir(tmp_path, *, project_key="sk-project-SECRET-0001"):
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    (d / "settings.json").write_text(json.dumps({
        "llm": {"provider": "opencode-go", "model": "deepseek-v4-flash",
                "base_url": "https://opencode.ai/zen/go/v1", "sdk": "openai",
                "fallback_models": []},
    }))
    rows = {
        "proj_byok": {
            "project_id": "proj_byok", "name": "byok", "workspace": str(tmp_path / "ws"),
            "provider": "openrouter", "model": "deepseek/deepseek-chat-v3.1",
            "base_url": "https://openrouter.ai/api/v1", "api_key": project_key,
        },
    }
    (d / "projects.json").write_text(json.dumps(rows))
    return d, rows


def _store(data_dir, projects, cards):
    return SettingsStore(
        data_dir=str(data_dir), project_store=projects, card_key_store=cards,
    )


# ---------------------------------------------------------------------------
# The plaintext key must outlive a failed keychain write
# ---------------------------------------------------------------------------


def test_a_key_that_cannot_be_stored_is_not_dropped_from_disk(tmp_path):
    d, rows = _legacy_dir(tmp_path)
    projects = _Projects(rows)
    # Fail every write: the keychain is unavailable, as it effectively was.
    cards = _FakeCards(fail_ids=set())
    cards.set = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("keychain unavailable"))
    _store(d, projects, cards)

    row = projects.rows["proj_byok"]
    assert row.get("api_key") == "sk-project-SECRET-0001", (
        "the only copy of the key was destroyed even though it was never stored"
    )


def test_a_silently_discarded_write_is_caught_by_read_back(tmp_path):
    # The exact shape of the incident: the write raises nothing and stores
    # nothing. Only a read-back can tell the difference.
    d, rows = _legacy_dir(tmp_path)
    projects = _Projects(rows)
    _store(d, projects, _FakeCards(silent=True))

    assert projects.rows["proj_byok"].get("api_key") == "sk-project-SECRET-0001"


def test_a_successful_write_does_migrate_the_row(tmp_path):
    """The guard must not block the normal path."""
    d, rows = _legacy_dir(tmp_path)
    projects = _Projects(rows)
    cards = _FakeCards()
    _store(d, projects, cards)

    row = projects.rows["proj_byok"]
    assert "api_key" not in row, "a stored key should leave the plaintext behind"
    assert row.get("card_id"), "the project should now reference its card"
    assert "sk-project-SECRET-0001" in cards.store.values()


def test_backups_are_written_before_the_rewrite(tmp_path):
    """The pre-migration copies exist and are not world-readable.

    They already existed (`.pre-cards`) — the 2026-09-04 recovery just looked
    for the wrong name. What was missing is the mode: copy2 preserves 0644, so
    a file full of plaintext keys was readable by anything running as the user.
    """
    d, rows = _legacy_dir(tmp_path)
    _store(d, _Projects(rows), _FakeCards())

    for name in ("settings.json", "projects.json"):
        bak = d / f"{name}.pre-cards"
        assert bak.exists(), f"no pre-migration backup of {name}"
        assert oct(os.stat(bak).st_mode)[-3:] == "600", "backups hold plaintext keys"
    # The backup must be the PRE-migration content, not a copy of the result.
    saved = json.loads((d / "projects.json.pre-cards").read_text())
    assert saved["proj_byok"]["api_key"] == "sk-project-SECRET-0001"


# ---------------------------------------------------------------------------
# Two daemons must not both migrate
# ---------------------------------------------------------------------------


def test_two_concurrent_migrations_agree_on_one_set_of_cards(tmp_path):
    d, rows = _legacy_dir(tmp_path)
    projects = _Projects(rows)
    # ONE keychain, as two processes on one machine share: whoever writes,
    # writes here. The bug was two settings files disagreeing with it.
    cards = _FakeCards()
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def run():
        try:
            barrier.wait(timeout=10)
            _store(d, projects, cards)
        except BaseException as exc:  # noqa: BLE001 — surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors

    settings = json.loads((d / "settings.json").read_text())
    listed = settings.get("credential_cards", [])
    assert listed, "migration produced no cards"

    # THE INVARIANT: every card named on disk must have a readable key. This is
    # exactly what failed in the field — cards existed, keys were under other
    # ids, and nothing noticed.
    for card in listed:
        if card["id"] == settings.get("default_card_id") and not card.get("model"):
            continue
        assert cards.get(card["id"]) is not None or card.get("last_error"), (
            f"card {card['id']} is named in settings but has no key in the "
            "keychain and carries no error"
        )

    # And the project's key survived somewhere: either as a card, or still on
    # its row. Never neither.
    row = projects.rows["proj_byok"]
    secret = "sk-project-SECRET-0001"
    assert secret in cards.store.values() or row.get("api_key") == secret


def test_second_boot_does_not_re_migrate(tmp_path):
    d, rows = _legacy_dir(tmp_path)
    projects = _Projects(rows)
    cards = _FakeCards()
    _store(d, projects, cards)
    first = json.loads((d / "settings.json").read_text())["credential_cards"]

    _store(d, projects, cards)
    second = json.loads((d / "settings.json").read_text())["credential_cards"]
    assert [c["id"] for c in first] == [c["id"] for c in second]


# ---------------------------------------------------------------------------
# The race that let two daemons start at all
# ---------------------------------------------------------------------------


def test_a_stale_pid_file_admits_exactly_one_racer(tmp_path):
    """Two PROCESSES starting over a stale PID file: one wins, one is refused.

    Processes, not threads: the invariant is process-level, and two threads
    share a PID, so the "this file is already ours" path would mask the race
    being tested. Before the fix both racers read the file, judged the recorded
    PID dead, and each wrote its own — which is how two daemons came up 377 ms
    apart and both ran the one-shot migration.
    """
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text("999999")  # a PID that cannot be alive
    ready = tmp_path / "ready"

    script = textwrap.dedent(f"""
        import sys, time, pathlib
        sys.path.insert(0, {str(REPO_ROOT)!r})
        from agent_os.utils.pid_file import acquire_pid_file, DaemonAlreadyRunning
        # Line the two processes up as closely as the OS allows.
        gate = pathlib.Path({str(ready)!r})
        while not gate.exists():
            time.sleep(0.005)
        try:
            acquire_pid_file(pathlib.Path({str(pid_path)!r}))
            print("acquired")
        except DaemonAlreadyRunning:
            print("refused")
    """)
    procs = [
        subprocess.Popen([sys.executable, "-c", script],
                         stdout=subprocess.PIPE, text=True)
        for _ in range(2)
    ]
    time.sleep(0.4)
    ready.write_text("go")
    outs = [p.communicate(timeout=30)[0].strip() for p in procs]

    assert outs.count("acquired") == 1, (
        f"exactly one instance may claim a stale PID file, got {outs}"
    )
    assert outs.count("refused") == 1, outs


def test_a_live_pid_file_is_still_refused(tmp_path):
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text(str(os.getppid()))
    import agent_os.utils.pid_file as pf
    pf._active_pid_path = None
    with pytest.raises(DaemonAlreadyRunning):
        acquire_pid_file(pid_path)
    pf._active_pid_path = None
