# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 074 — route-level coverage: the sessions PATCH (pin/retarget/unpin
validation + side effects) and the inject route's pinned dispatch path
(initiator mapping + recap preamble placement).

Real FastAPI app via create_app (the test_sub_agent_memory_viewer pattern);
setup_engine.check_all patched to a deterministic installed set; the
sub-agent manager stubbed at dispatch so no real worker ever spawns.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from agent_os.agent.session import Session


def _fake_check_all_factory(installed_slugs: list[str]):
    from agent_os.agents.setup_types import AgentSetupStatus

    def _fake() -> list[AgentSetupStatus]:
        statuses = [AgentSetupStatus(
            slug="built-in", name="Built-in",
            installed=True, binary_path=None, version=None,
            dependencies_met=True, missing_dependencies=[],
            credentials_configured=True, missing_credentials=[],
            setup_actions=[],
        )]
        for slug in installed_slugs:
            statuses.append(AgentSetupStatus(
                slug=slug, name=slug.replace("-", " ").title(),
                installed=True, binary_path="/fake/" + slug, version="1.0.0",
                dependencies_met=True, missing_dependencies=[],
                credentials_configured=True, missing_credentials=[],
                setup_actions=[],
            ))
        return statuses

    return _fake


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    os.makedirs(str(tmp_path / "home"), exist_ok=True)

    from agent_os.api.app import create_app
    app = create_app(data_dir=str(tmp_path / "data"))

    from agent_os.api.routes import agents_v2 as routes_mod
    routes_mod._setup_engine.check_all = _fake_check_all_factory(
        ["claude-code", "codex"])
    return TestClient(app)


def _routes_mod():
    from agent_os.api.routes import agents_v2 as routes_mod
    return routes_mod


def _make_project(client, tmp_path, name="pin074") -> tuple[str, str]:
    ws = str(tmp_path / f"ws_{name}")
    os.makedirs(ws, exist_ok=True)
    resp = client.post("/api/v2/projects", json={
        "name": name, "workspace": ws,
        "model": "gpt-4", "api_key": "test-key",
    })
    assert resp.status_code == 201, resp.text
    return resp.json()["project_id"], ws


def _make_session(ws: str, stem: str = "pin074_sess_cafe0001") -> str:
    s = Session.new(stem, ws)
    s.append({"role": "user", "content": "earlier context message",
              "source": "user"})
    s.append({"role": "assistant", "content": "Earlier reply from the manager.",
              "source": "management"})
    return stem


def _session_rows(ws: str, stem: str) -> list[dict]:
    p = os.path.join(ws, "orbital", "sessions", f"{stem}.jsonl")
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# PATCH /agents/{pid}/sessions/{sid} — validation + persistence
# ---------------------------------------------------------------------------


class TestPatchValidation:

    def test_unknown_slug_rejected(self, client, tmp_path):
        pid, ws = _make_project(client, tmp_path, "unknown")
        sid = _make_session(ws)
        resp = client.patch(
            f"/api/v2/agents/{pid}/sessions/{sid}",
            json={"pinned_target": "not-a-real-agent"})
        assert resp.status_code == 422
        assert "not-a-real-agent" in resp.json()["detail"]

    @pytest.mark.parametrize("reserved", ["orbital", "@orbital", "Orbital"])
    def test_orbital_reserved(self, client, tmp_path, reserved):
        pid, ws = _make_project(client, tmp_path, "reserved")
        sid = _make_session(ws)
        resp = client.patch(
            f"/api/v2/agents/{pid}/sessions/{sid}",
            json={"pinned_target": reserved})
        assert resp.status_code == 422
        assert "reserved" in resp.json()["detail"]

    def test_pin_persists_and_lists(self, client, tmp_path):
        pid, ws = _make_project(client, tmp_path, "persist")
        sid = _make_session(ws)
        resp = client.patch(
            f"/api/v2/agents/{pid}/sessions/{sid}",
            json={"pinned_target": "codex"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["pinned_target"] == "codex"

        listed = client.get(f"/api/v2/projects/{pid}/sessions").json()["sessions"]
        entry = next(s for s in listed if s["session_uuid"] == sid)
        assert entry["pinned_target"] == "codex"

    def test_explicit_null_clears(self, client, tmp_path):
        pid, ws = _make_project(client, tmp_path, "clear")
        sid = _make_session(ws)
        client.patch(f"/api/v2/agents/{pid}/sessions/{sid}",
                     json={"pinned_target": "codex"})
        resp = client.patch(f"/api/v2/agents/{pid}/sessions/{sid}",
                            json={"pinned_target": None})
        assert resp.status_code == 200, resp.text
        assert resp.json()["pinned_target"] is None

        listed = client.get(f"/api/v2/projects/{pid}/sessions").json()["sessions"]
        entry = next(s for s in listed if s["session_uuid"] == sid)
        assert entry["pinned_target"] is None

    def test_absent_field_leaves_pin_untouched(self, client, tmp_path):
        """A rename-only PATCH must not read as an unpin (tri-state)."""
        pid, ws = _make_project(client, tmp_path, "tristate")
        sid = _make_session(ws)
        client.patch(f"/api/v2/agents/{pid}/sessions/{sid}",
                     json={"pinned_target": "codex"})
        resp = client.patch(f"/api/v2/agents/{pid}/sessions/{sid}",
                            json={"name": "renamed"})
        assert resp.status_code == 200

        listed = client.get(f"/api/v2/projects/{pid}/sessions").json()["sessions"]
        entry = next(s for s in listed if s["session_uuid"] == sid)
        assert entry["pinned_target"] == "codex"
        assert entry["name"] == "renamed"

    def test_empty_body_still_400(self, client, tmp_path):
        pid, ws = _make_project(client, tmp_path, "empty")
        sid = _make_session(ws)
        resp = client.patch(f"/api/v2/agents/{pid}/sessions/{sid}", json={})
        assert resp.status_code == 400


class TestPatchSideEffects:

    def test_retarget_and_unpin_fire_consolidation(self, client, tmp_path):
        pid, ws = _make_project(client, tmp_path, "kick")
        sid = _make_session(ws)
        routes_mod = _routes_mod()
        recorder = MagicMock()
        routes_mod._pinned_consolidation = recorder

        # First pin (previous None) → NO consolidation.
        client.patch(f"/api/v2/agents/{pid}/sessions/{sid}",
                     json={"pinned_target": "codex"})
        recorder.trigger.assert_not_called()

        # Retarget codex → claude-code → fires with reason=retarget.
        client.patch(f"/api/v2/agents/{pid}/sessions/{sid}",
                     json={"pinned_target": "claude-code"})
        recorder.trigger.assert_called_once_with(
            pid, sid, reason="retarget")

        # Unpin → fires with reason=unpin.
        recorder.trigger.reset_mock()
        client.patch(f"/api/v2/agents/{pid}/sessions/{sid}",
                     json={"pinned_target": None})
        recorder.trigger.assert_called_once_with(pid, sid, reason="unpin")

    def test_pin_time_reseed_never_touches_user_edited_agents_md(
            self, client, tmp_path):
        pid, ws = _make_project(client, tmp_path, "seedguard")
        sid = _make_session(ws)
        agents_md = os.path.join(ws, "AGENTS.md")
        # Project creation seeded it; the user then edits it.
        with open(agents_md, "r", encoding="utf-8") as f:
            seeded = f.read()
        edited = seeded + "\n## Hand-written section\nprecious\n"
        with open(agents_md, "w", encoding="utf-8") as f:
            f.write(edited)

        resp = client.patch(f"/api/v2/agents/{pid}/sessions/{sid}",
                            json={"pinned_target": "codex"})
        assert resp.status_code == 200

        with open(agents_md, "r", encoding="utf-8") as f:
            assert f.read() == edited

    def test_pin_time_reseed_restores_missing_agents_md(self, client, tmp_path):
        pid, ws = _make_project(client, tmp_path, "seedback")
        sid = _make_session(ws)
        agents_md = os.path.join(ws, "AGENTS.md")
        os.remove(agents_md)

        resp = client.patch(f"/api/v2/agents/{pid}/sessions/{sid}",
                            json={"pinned_target": "codex"})
        assert resp.status_code == 200
        assert os.path.exists(agents_md)


# ---------------------------------------------------------------------------
# POST /agents/{pid}/inject — pinned dispatch path
# ---------------------------------------------------------------------------


class _StubSubAgentManager:
    def __init__(self):
        self.sends: list[dict] = []

    async def send(self, project_id, handle, message, *, session_id=None,
                   dispatch_id=None, initiator="management_agent", **kwargs):
        self.sends.append({
            "project_id": project_id, "handle": handle, "message": message,
            "session_id": session_id, "dispatch_id": dispatch_id,
            "initiator": initiator,
        })
        return f"Message sent to {handle}. Transcript: /tmp/{handle}.jsonl"


@pytest.fixture
def dispatch_env(client, tmp_path):
    routes_mod = _routes_mod()
    stub = _StubSubAgentManager()
    routes_mod._sub_agent_manager = stub
    consolidation = MagicMock()
    routes_mod._pinned_consolidation = consolidation
    pid, ws = _make_project(client, tmp_path, "dispatch")
    sid = _make_session(ws)
    return client, stub, consolidation, pid, ws, sid


class TestPinnedInject:

    def test_pinned_inject_maps_to_user_pinned_and_prepends_recap(
            self, dispatch_env):
        client, stub, consolidation, pid, ws, sid = dispatch_env

        resp = client.post(f"/api/v2/agents/{pid}/inject", json={
            "content": "please fix the login bug",
            "target": "codex", "pinned": True, "session_id": sid,
        })
        assert resp.status_code == 200, resp.text

        assert len(stub.sends) == 1
        send = stub.sends[0]
        assert send["initiator"] == "user_pinned"
        # Recap preamble: codex never participated in this session, which has
        # history → the DISPATCHED body carries the context block…
        assert send["message"].startswith("Conversation so far")
        assert "earlier context message" in send["message"]
        assert send["message"].endswith("please fix the login bug")

        # …while the PERSISTED chat row stays the authored text only.
        rows = _session_rows(ws, sid)
        user_rows = [r for r in rows if r.get("role") == "user"]
        assert user_rows[-1]["content"] == "please fix the login bug"
        assert "Conversation so far" not in user_rows[-1]["content"]

        # A pinned dispatch resets the quiescence timer.
        consolidation.note_pinned_dispatch.assert_called_once()

    def test_mention_inject_keeps_user_mention_and_no_recap(
            self, dispatch_env):
        client, stub, consolidation, pid, ws, sid = dispatch_env

        resp = client.post(f"/api/v2/agents/{pid}/inject", json={
            "content": "please fix the login bug",
            "target": "codex", "session_id": sid,
        })
        assert resp.status_code == 200, resp.text

        send = stub.sends[0]
        assert send["initiator"] == "user_mention"
        assert send["message"] == "please fix the login bug"
        consolidation.note_pinned_dispatch.assert_not_called()

    def test_pinned_resume_of_own_thread_gets_no_recap(self, dispatch_env):
        """Second pinned message to the same worker: the first dispatch made
        it a participant, so the second body carries no context block."""
        client, stub, consolidation, pid, ws, sid = dispatch_env

        client.post(f"/api/v2/agents/{pid}/inject", json={
            "content": "first pinned message",
            "target": "codex", "pinned": True, "session_id": sid,
        })
        client.post(f"/api/v2/agents/{pid}/inject", json={
            "content": "second pinned message",
            "target": "codex", "pinned": True, "session_id": sid,
        })

        second = stub.sends[1]
        assert second["message"] == "second pinned message"


# ---------------------------------------------------------------------------
# Pin-PATCH vs. inject lock race (manual-verification bug, 2026-08-31): the
# composer's fire-and-forget pin PATCH write-locks the session JSONL for a
# load+meta-rewrite burst; a send landing inside that burst used to 500 (the
# recap peek degraded gracefully but persist_mention_message had no guard).
# ---------------------------------------------------------------------------


class TestPinnedInjectLockContention:

    @staticmethod
    def _lock(ws: str, sid: str):
        from agent_os.utils.file_lock import DirectFileLock
        path = os.path.join(ws, "orbital", "sessions", f"{sid}.jsonl")
        return DirectFileLock(path)

    def test_send_survives_briefly_held_session_lock(self, dispatch_env):
        """Lock held for ~300ms (a realistic patch_session burst): the route
        retries both the recap peek and the persist — 200, dispatched WITH
        the recap, and the authored row lands in the JSONL."""
        import threading

        client, stub, consolidation, pid, ws, sid = dispatch_env
        lock = self._lock(ws, sid)
        lock.acquire()
        timer = threading.Timer(0.3, lock.release)
        timer.start()
        try:
            resp = client.post(f"/api/v2/agents/{pid}/inject", json={
                "content": "racing the pin patch",
                "target": "codex", "pinned": True, "session_id": sid,
            })
        finally:
            timer.cancel()
            lock.release()  # no-op if the timer already released

        assert resp.status_code == 200, resp.text
        assert len(stub.sends) == 1
        # The peek retried too — recap preserved, not degraded away.
        assert stub.sends[0]["message"].startswith("Conversation so far")
        rows = _session_rows(ws, sid)
        user_rows = [r for r in rows if r.get("role") == "user"]
        assert user_rows[-1]["content"] == "racing the pin patch"

    def test_pin_patch_survives_briefly_held_session_lock(self, dispatch_env):
        """The mirror race (seen live 3/10 under contention): the pin PATCH's
        own Session.load loses the lock to a concurrent inject. It must retry
        the same way instead of 500ing the pin."""
        import threading

        client, stub, consolidation, pid, ws, sid = dispatch_env
        lock = self._lock(ws, sid)
        lock.acquire()
        timer = threading.Timer(0.3, lock.release)
        timer.start()
        try:
            resp = client.patch(
                f"/api/v2/agents/{pid}/sessions/{sid}",
                json={"pinned_target": "codex"},
            )
        finally:
            timer.cancel()
            lock.release()

        assert resp.status_code == 200, resp.text
        meta = [r for r in _session_rows(ws, sid)
                if r.get("event") == "session_start"]
        assert meta and meta[-1].get("pinned_target") == "codex"

    def test_stuck_lock_returns_503_not_500(self, dispatch_env, monkeypatch):
        """Lock held past the whole retry budget: a clean, retryable 503 —
        never an unhandled FileLockError/500 — and nothing dispatched or
        persisted."""
        client, stub, consolidation, pid, ws, sid = dispatch_env
        routes_mod = _routes_mod()
        monkeypatch.setattr(
            routes_mod, "_LOCK_RETRY_DELAYS", (0.01, 0.02), raising=False)

        lock = self._lock(ws, sid)
        lock.acquire()
        try:
            resp = client.post(f"/api/v2/agents/{pid}/inject", json={
                "content": "never lands",
                "target": "codex", "pinned": True, "session_id": sid,
            })
        finally:
            lock.release()

        assert resp.status_code == 503, resp.text
        assert stub.sends == []
        rows = _session_rows(ws, sid)
        assert all(r.get("content") != "never lands" for r in rows)
