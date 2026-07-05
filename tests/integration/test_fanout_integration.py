# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""End-to-end integration test for sub-agent fanout (spec 009, Task 7).

Drives the REAL wired pipeline that the unit suites cover only in stubbed
slices: ``create_app()`` wiring (Task 6) → ``dispatch_fanout`` (Task 2) →
``NativeWorkerAdapter`` running real ``AgentLoop`` turns on a stub provider
(Task 1) → ``_background_send`` → ``LifecycleObserver`` → ``FanoutRegistry``
join → single ``[Fanout …]`` injection into the management session → the
transcript endpoint (Task 4) serving both worker transcripts.

Only the LLM provider is stubbed; everything else is production code and
production wiring.
"""

import asyncio
import glob
import json
import os
import re

import pytest
from starlette.testclient import TestClient

from agent_os.agent.providers.types import StreamChunk, TokenUsage


class _TextProvider:
    """Single-shot text provider; every turn returns one text response.

    Matches the provider surface AgentLoop reads (provider/model/sdk
    identity attributes + async stream()). Fresh instance per LLMProvider()
    construction so management and worker loops don't share iteration state.
    """

    def __init__(self, *args, **kwargs):
        self.provider = "stub"
        self.model = "stub-model"
        self.sdk = "stub-sdk"

    async def stream(self, messages, tools=None):
        yield StreamChunk(text="stub turn complete")
        yield StreamChunk(
            is_final=True,
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )


@pytest.mark.asyncio
async def test_fanout_end_to_end_through_real_wiring(monkeypatch, tmp_path):
    # Headless keychain hang hazard (CLAUDE.md): create_app touches
    # credential storage during construction.
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "in-memory")
    monkeypatch.setenv("AGENT_OS_API_KEY", "test")

    data_dir = tmp_path / "data"
    workspace = tmp_path / "ws"
    workspace.mkdir()

    events: list[dict] = []

    from unittest.mock import patch

    from agent_os.api.ws import WebSocketManager

    def _recording_broadcast(self, project_id, payload):
        events.append(payload)

    # Class-level patches BEFORE create_app so the wiring captures the
    # recorder (FanoutRegistry binds ws.broadcast at construction time) and
    # every provider built — management main/utility AND worker deps — is a
    # stub. Everything else is real.
    with patch.object(WebSocketManager, "broadcast", _recording_broadcast), \
         patch("agent_os.daemon_v2.agent_manager.LLMProvider", _TextProvider), \
         patch("agent_os.daemon_v2.agent_manager.install_default_skills"):

        from agent_os.api.app import create_app
        from agent_os.api.routes import agents_v2

        app = create_app(data_dir=str(data_dir))
        client = TestClient(app)

        agent_manager = agents_v2._agent_manager
        sub_agent_manager = agents_v2._sub_agent_manager
        project_store = agents_v2._project_store

        pid = project_store.create_project({
            "name": "FanoutE2E",
            "workspace": str(workspace),
            "autonomy": "hands_off",
            "provider": "custom",
            "model": "stub-model",
            "api_key": "sk-test",
            "sdk": "openai",
            "enabled_sub_agents": [],
        })

        sid = "fanout_e2e_session"

        # ── Mint the management handle with one real (stubbed) turn ──────
        result = await agent_manager.inject_message(pid, "hello", session_id=sid)
        assert result in ("started", "delivered"), result

        async def _wait(predicate, timeout=10.0, what=""):
            deadline = asyncio.get_event_loop().time() + timeout
            while asyncio.get_event_loop().time() < deadline:
                if predicate():
                    return
                await asyncio.sleep(0.05)
            raise AssertionError(f"timed out waiting for {what}")

        handle = agent_manager._handles.get((pid, sid))
        assert handle is not None
        await _wait(lambda: handle.task is None or handle.task.done(),
                    what="management turn to complete")

        # ── Dispatch a real 2-task fanout ────────────────────────────────
        ack = await sub_agent_manager.dispatch_fanout(
            pid,
            [
                {"brief": "analyze part A", "label": "task A"},
                {"brief": "analyze part B", "label": "task B"},
            ],
            session_id=sid,
        )
        assert not ack.startswith("Error"), ack
        m = re.search(r"Fanout ([a-z0-9]{8})", ack)
        assert m, f"ack missing fanout id: {ack}"
        fanout_id = m.group(1)

        started = [e for e in events if e.get("type") == "fanout.started"]
        assert len(started) == 1
        assert started[0]["fanout_id"] == fanout_id
        worker_handles = [t["handle"] for t in started[0]["tasks"]]
        assert worker_handles == [
            f"worker:{fanout_id}-0", f"worker:{fanout_id}-1",
        ]

        # ── Join: both workers complete, group resolves exactly once ─────
        await _wait(
            lambda: any(e.get("type") == "fanout.completed" for e in events),
            timeout=15.0, what="fanout.completed broadcast",
        )
        updates = [e for e in events if e.get("type") == "fanout.task_update"]
        assert {u["handle"] for u in updates} == set(worker_handles)
        assert all(u["status"] == "completed" for u in updates), updates
        completed = [e for e in events if e.get("type") == "fanout.completed"]
        assert len(completed) == 1
        assert completed[0]["fanout_id"] == fanout_id

        # ── Exactly ONE [Fanout …] injection in the management JSONL ─────
        from agent_os.agent.project_paths import ProjectPaths
        pp = ProjectPaths(str(workspace))
        mgmt_jsonl = pp.session_file(handle.session.session_id)

        def _fanout_lines():
            if not os.path.isfile(mgmt_jsonl):
                return []
            with open(mgmt_jsonl, encoding="utf-8") as f:
                return [ln for ln in f if "[Fanout " in ln]

        await _wait(lambda: len(_fanout_lines()) >= 1, timeout=10.0,
                    what="join summary injection in management session")
        lines = _fanout_lines()
        assert len(lines) == 1, f"expected exactly one join injection, got {len(lines)}"
        assert "2/2 succeeded" in lines[0]
        assert "task A" in lines[0] and "task B" in lines[0]

        # ── Worker sessions: tagged on disk, hidden from the listing ─────
        session_files = glob.glob(os.path.join(pp.sessions_dir, "*.jsonl"))
        worker_files = []
        for path in session_files:
            with open(path, encoding="utf-8") as f:
                text = f.read()
            if '"session_kind"' in text and '"worker"' in text:
                worker_files.append(path)
        assert len(worker_files) == 2, (
            f"expected 2 worker-tagged session files, got {len(worker_files)} "
            f"of {len(session_files)} total"
        )

        resp = client.get(f"/api/v2/projects/{pid}/sessions")
        assert resp.status_code == 200
        listed = resp.json()
        listed_ids = {s["session_id"] for s in (
            listed if isinstance(listed, list) else listed.get("sessions", [])
        )}
        assert sid in listed_ids
        assert not any(s.startswith("worker:") for s in listed_ids)
        # No worker session uuid may leak into the listing either.
        worker_uuids = {os.path.splitext(os.path.basename(p))[0] for p in worker_files}
        listed_uuids = {s.get("session_uuid") for s in (
            listed if isinstance(listed, list) else listed.get("sessions", [])
        )}
        assert not (worker_uuids & listed_uuids)

        # ── Transcript endpoint serves both workers (drill-in contract) ──
        for wh in worker_handles:
            resp = client.get(
                f"/api/v2/agents/{pid}/sub-agents/{wh}/transcript",
                params={"session_id": sid},
            )
            assert resp.status_code == 200, (wh, resp.status_code, resp.text)
            body = resp.json()
            assert body["handle"] == wh
            assert body["kind"] == "worker"
            assert body["resumable"] is False
            assert any(
                "stub turn complete" in (e.get("content") or "")
                for e in body["entries"]
            ), body["entries"]

        # ── CRITICAL fix: completed workers are reaped, not left idle ────
        # Before the fix, both worker handles stay registered in
        # SubAgentManager._adapters forever (list_active never drops a
        # terminal one-shot worker), so they keep counting toward
        # MAX_CONCURRENT_SUBAGENTS and a second same-session fanout hits
        # the cap error with nothing actually running.
        status_resp = client.get(
            f"/api/v2/agents/{pid}/sub-agents/status",
            params={"session_id": sid},
        )
        assert status_resp.status_code == 200
        status_handles = {a["handle"] for a in status_resp.json()["agents"]}
        assert not (status_handles & set(worker_handles)), (
            f"completed fanout workers still listed as active: {status_handles}"
        )

        # ── Transcript endpoint still serves both workers post-reap ──────
        for wh in worker_handles:
            resp = client.get(
                f"/api/v2/agents/{pid}/sub-agents/{wh}/transcript",
                params={"session_id": sid},
            )
            assert resp.status_code == 200, (wh, resp.status_code, resp.text)

        # ── A SECOND fanout (5 tasks — full width) in the same session ───
        # must succeed: before the fix this fails with a capacity error
        # because the first batch's 2 completed-but-never-reaped adapters
        # still occupy slots.
        second_tasks = [
            {"brief": f"analyze part {i}", "label": f"task {i}"}
            for i in range(5)
        ]
        second_ack = await sub_agent_manager.dispatch_fanout(
            pid, second_tasks, session_id=sid,
        )
        assert not second_ack.startswith("Error"), second_ack

        await _wait(
            lambda: sum(
                1 for e in events
                if e.get("type") == "fanout.completed"
            ) >= 2,
            timeout=15.0, what="second fanout.completed broadcast",
        )

        client.close()
