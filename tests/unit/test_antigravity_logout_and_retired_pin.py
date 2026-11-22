# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 193 follow-ups.

1. Logout is offered only for an agent that has a logout action. The card
   reads ``supports_logout`` from the status entry, and that flag comes from
   the same table the logout route uses, so the button and the route can
   never disagree (Antigravity has no CLI logout: ``agy --help`` lists none).
2. A session whose stored pin is the retired ``gemini-cli`` handle loads as
   pinned to ``antigravity``. Read-time only: the session file keeps its old
   pin, so an older Orbital still opens it unchanged.
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_os.agents.registry import AgentRegistry
from agent_os.api.routes import settings as settings_routes
from tests.unit.keyless_harness import agent_status

MANIFESTS = os.path.normpath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "agent_os", "agents", "manifests"))


def _bundled() -> AgentRegistry:
    registry = AgentRegistry()
    registry.load_directory(MANIFESTS)
    return registry


# ---------------------------------------------------------------------------
# 1. supports_logout
# ---------------------------------------------------------------------------

@pytest.fixture
def engine(monkeypatch):
    registry = _bundled()
    eng = SimpleNamespace(
        _registry=registry,
        resolve_binary=lambda manifest: f"/usr/local/bin/{manifest.runtime.command}",
        _substitute_binary=lambda cmd, bare, binary: cmd.replace(bare, binary, 1),
    )
    monkeypatch.setattr(settings_routes, "_setup_engine", eng)
    monkeypatch.setattr(settings_routes, "_sub_agent_config_store", None)
    return eng


def _entry(slug):
    return settings_routes._build_sub_agent_status_entry(agent_status(slug))


class TestSupportsLogout:

    def test_antigravity_has_no_logout(self, engine):
        assert _entry("antigravity")["supports_logout"] is False

    @pytest.mark.parametrize("slug", ["claude-code", "codex", "cursor"])
    def test_clis_with_a_logout_subcommand_offer_it(self, engine, slug):
        assert _entry(slug)["supports_logout"] is True

    def test_the_flag_matches_the_route_for_every_bundled_agent(self, engine):
        """The card shows Logout exactly when the route would run a command
        rather than answer 400."""
        for manifest in engine._registry.list_all():
            slug = manifest.slug
            if manifest.runtime.adapter == "built_in":
                continue
            resolved = settings_routes._resolve_setup_command(slug, "logout")
            assert _entry(slug)["supports_logout"] is (resolved is not None), slug

    def test_an_unknown_slug_has_no_logout(self, engine):
        assert _entry("not-a-manifest")["supports_logout"] is False

    def test_the_antigravity_logout_route_still_400s(self, engine):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            asyncio.run(settings_routes.trigger_sub_agent_logout("antigravity"))
        assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# 2. A stored pin to the retired gemini-cli handle
# ---------------------------------------------------------------------------

STEM = "chat_retired_pin_0001"


def _write_session(workspace, pinned_target="gemini-cli", stem=STEM):
    sessions = os.path.join(str(workspace), "orbital", "sessions")
    os.makedirs(sessions, exist_ok=True)
    path = os.path.join(sessions, f"{stem}.jsonl")
    rows = [
        {"role": "meta", "event": "session_start", "session_id": stem,
         "session_uuid": stem, "origin": "chat", "provider": "unknown",
         "model": "unknown", "sdk": "unknown",
         "timestamp": "2026-06-01T00:00:00+00:00",
         "pinned_target": pinned_target},
        {"role": "user", "content": "hello", "source": "user",
         "target": pinned_target, "timestamp": "2026-06-01T00:00:01+00:00"},
    ]
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


class TestTodayAPinToAHandleWithNoManifest:
    """What a send to a handle with no manifest does on the real daemon path
    (registry + setup engine): an ``Error:`` answer, which the inject route
    writes as a ``[Sub-agent] … could not start`` row and a 404."""

    def test_registry_start_answers_unknown_agent(self, tmp_path):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        project_store = MagicMock()
        project_store.get_project.return_value = {"workspace": str(tmp_path)}
        mgr = SubAgentManager(
            process_manager=MagicMock(), registry=_bundled(),
            project_store=project_store, setup_engine=MagicMock())
        out = asyncio.run(mgr.start("p1", "gemini-cli", session_id="s1"))
        assert out == "Error: unknown agent 'gemini-cli'"


class TestRetiredPinResolves:

    def test_alias_table(self):
        from agent_os.agents.retired import current_slug
        assert current_slug("gemini-cli") == "antigravity"
        assert current_slug("claude-code") == "claude-code"
        assert current_slug(None) is None
        # Every alias points at a manifest that ships.
        from agent_os.agents.retired import RETIRED_SLUGS
        registry = _bundled()
        for old, new in RETIRED_SLUGS.items():
            assert registry.get(old) is None, old
            assert registry.get(new) is not None, new

    def test_session_load_reads_antigravity_and_keeps_the_file(self, tmp_path):
        from agent_os.agent.session import Session
        path = _write_session(tmp_path)
        before = open(path, encoding="utf-8").read()
        session = Session.load(path)
        assert session.pinned_target == "antigravity"
        # A rename (another session_start meta writer) leaves the old pin on
        # disk: only an explicit pin change rewrites it.
        session.set_name("renamed")
        meta = json.loads(open(path, encoding="utf-8").readline())
        assert meta["pinned_target"] == "gemini-cli"
        assert meta["name"] == "renamed"
        assert before.splitlines()[1] == open(path, encoding="utf-8").read().splitlines()[1]

    def test_a_current_pin_is_untouched(self, tmp_path):
        from agent_os.agent.session import Session
        path = _write_session(tmp_path, pinned_target="codex")
        assert Session.load(path).pinned_target == "codex"

    def test_the_session_list_row_reads_antigravity(self, tmp_path):
        from agent_os.daemon_v2.session_index import derive_disk_entry
        path = _write_session(tmp_path)
        assert derive_disk_entry(path, STEM)["pinned_target"] == "antigravity"

    def test_an_index_built_before_the_alias_is_rebuilt(self, tmp_path):
        """A v0.24 index cached the row with ``gemini-cli``; the version bump
        empties it, so the list re-derives the row with the alias."""
        import agent_os.daemon_v2.session_index as si
        path = _write_session(tmp_path)
        sessions_dir = os.path.dirname(path)
        st = os.stat(path)
        stale = si.derive_disk_entry(path, STEM)
        stale["pinned_target"] = "gemini-cli"
        old = si.SessionIndex(sessions_dir)
        old.load_rows()
        old.put(os.path.basename(path), st, stale)
        # Mark the file as written by the previous version.
        import sqlite3
        conn = sqlite3.connect(old.path, isolation_level=None)
        conn.execute("UPDATE index_meta SET value = '5' WHERE key = 'version'")
        conn.close()
        fresh = si.SessionIndex(sessions_dir)
        fresh.load_rows()
        hit, _row = fresh.get(os.path.basename(path), st)
        assert hit is False
