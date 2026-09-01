# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Resume-record persistence for sessions that are NOT hydrated in memory.

The pinned-worker chat flow (spec 074) never starts the management loop, so
the chat session exists only as an on-disk JSONL — there is no live handle in
``AgentManager._handles``. Before this fix:

  - ``record_sub_agent_thread`` looked up the session via ``get_session``
    (handles only), warned "no hydrated session", and DROPPED the worker's
    resume identity on every completed turn. The observed symptom: stop a
    pinned codex worker, send the next message, and the dispatch ack honestly
    reports "fresh session — first spawn" — the provider thread (and all its
    context) is lost even though codex reported its thread id.
  - The dispatch-side reader (``sub_agent_manager._session_resolver``, wired
    to ``get_session``) had the same blindness: even a persisted record was
    invisible while the session stayed unhydrated.

Fix: both sides fall back to the on-disk session. ``get_session_or_load`` is
the shared accessor — live handle first, then ``_load_session_from_disk``.
"""

from unittest.mock import MagicMock

import pytest

from agent_os.agent.session import Session
from agent_os.daemon_v2.agent_manager import AgentManager


@pytest.fixture
def manager(tmp_path):
    """AgentManager with minimal mocks and a real tmp workspace."""
    project_store = MagicMock()
    project_store.get_project.return_value = {"workspace": str(tmp_path)}
    mgr = AgentManager(
        project_store=project_store,
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )
    return mgr


def _mint_disk_session(tmp_path, session_id: str) -> str:
    """Create an on-disk session JSONL (never hydrated into the manager)."""
    session = Session.new(session_id, str(tmp_path))
    # A real (non-meta) row so the file exists on disk with content, exactly
    # like a pinned send's persisted user message.
    session.append({"role": "user", "content": "hello worker",
                    "target": "codex"})
    return session._filepath


class TestRecordSubAgentThreadDiskFallback:
    def test_record_persists_to_disk_only_session(self, manager, tmp_path):
        """A completed worker turn's thread id must land in the JSONL even
        when the chat session has no live handle (the pinned flow)."""
        sid = "proj-abc_11112222"
        path = _mint_disk_session(tmp_path, sid)

        manager.record_sub_agent_thread(
            "proj_1", "codex",
            claude_session_id="thread-777", model="gpt-5.3-codex",
            session_id=sid, rollout_path="/tmp/rollout-777.jsonl",
        )

        reloaded = Session.load(path)
        rec = reloaded.get_sub_agent_thread("codex")
        assert rec is not None
        assert rec["session_id"] == "thread-777"
        assert rec["rollout_path"] == "/tmp/rollout-777.jsonl"

    def test_record_prefers_live_handle_when_hydrated(self, manager):
        """The existing hydrated path is untouched: a live handle's session
        receives the record directly (no disk load)."""
        live = MagicMock()
        handle = MagicMock()
        handle.session = live
        manager._handles[("proj_1", "sid-live")] = handle

        manager.record_sub_agent_thread(
            "proj_1", "codex", claude_session_id="t-1", session_id="sid-live",
        )

        live.set_sub_agent_thread.assert_called_once()

    def test_record_still_warns_when_no_session_anywhere(self, manager, caplog):
        """No handle AND no JSONL: the drop stays loud, never a crash."""
        manager.record_sub_agent_thread(
            "proj_1", "codex", claude_session_id="t-2",
            session_id="never-existed",
        )
        assert any("not recorded" in r.message for r in caplog.records)


class TestGetSessionOrLoad:
    def test_returns_disk_session_when_not_hydrated(self, manager, tmp_path):
        sid = "proj-abc_33334444"
        _mint_disk_session(tmp_path, sid)

        session = manager.get_session_or_load("proj_1", session_id=sid)

        assert session is not None
        assert session.get_sub_agent_thread("codex") is None  # loads clean

    def test_returns_none_when_nothing_exists(self, manager):
        assert manager.get_session_or_load(
            "proj_1", session_id="ghost") is None

    def test_disk_loaded_session_carries_thread_records(self, manager, tmp_path):
        """The dispatch-side resume decision reads thread records through this
        accessor — a record persisted by a prior (unhydrated) completion must
        be visible to the next spawn."""
        sid = "proj-abc_55556666"
        _mint_disk_session(tmp_path, sid)
        manager.record_sub_agent_thread(
            "proj_1", "codex", claude_session_id="thread-999",
            session_id=sid,
        )

        session = manager.get_session_or_load("proj_1", session_id=sid)
        rec = session.get_sub_agent_thread("codex")
        assert rec is not None
        assert rec["session_id"] == "thread-999"

    def test_swallows_load_errors(self, manager, monkeypatch):
        """The resolver runs inside the dispatch path — a load failure must
        degrade to None (an honest fresh spawn), never propagate a 500."""
        monkeypatch.setattr(
            manager, "_load_session_from_disk",
            MagicMock(side_effect=OSError("locked")),
        )
        assert manager.get_session_or_load(
            "proj_1", session_id="whatever") is None
