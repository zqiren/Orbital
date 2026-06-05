# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Piece 2 regression suite: sub-agent resume persistence.

Contract (TASK-resume-persistence):
    A reaped sub-agent, when reused, resumes the SAME underlying agent
    session with full prior context. When that is impossible it starts
    fresh and SAYS SO — never silently.

The record lives in the existing SessionKey-keyed metadata store — the
``role: meta`` rows of the management session JSONL — as
``event: sub_agent_thread`` rows keyed per handle (composite
``(SessionKey, handle)``: the JSONL is SessionKey-scoped, the row carries
the handle). Last row wins on reload.

Resume-status taxonomy: ``resumed`` | ``fresh`` with
``reason in {first_spawn, resume_failed}``. No orchestrator branching on
``resume_failed`` (recorded for honesty/observability only).
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.session import Session
from agent_os.agent.transports.base import TransportEvent
from agent_os.agent.transports.sdk_transport import SDKTransport, HAS_SDK
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
from agent_os.daemon_v2.process_manager import ProcessManager
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_session(tmp_path, uuid="proj_test_ab12cd34"):
    session = Session.new(uuid, str(tmp_path), "proj_test")
    # Flush the deferred session_start meta: a session is a file on disk
    # only once it has a first message.
    session.append({"role": "user", "content": "hi", "source": "user"})
    return session


# ---------------------------------------------------------------------------
# 1. The record: Session meta-row store (survives daemon restart)
# ---------------------------------------------------------------------------


class TestSessionThreadStore:

    async def test_set_and_get_roundtrip_in_memory(self, tmp_path):
        s = _new_session(tmp_path)
        s.set_sub_agent_thread("claude-code", session_id="sid-1", model="m-1")
        rec = s.get_sub_agent_thread("claude-code")
        assert rec["session_id"] == "sid-1"
        assert rec["model"] == "m-1"
        assert rec["last_used_at"]

    async def test_record_survives_reload_from_disk(self, tmp_path):
        """TEST RULE 1 core: persist -> simulate daemon restart (reload from
        disk) -> the resume id is still there. Without persistence the id
        is lost."""
        s = _new_session(tmp_path)
        s.set_sub_agent_thread("claude-code", session_id="sid-42", model="m-1")

        reloaded = Session.load(s._filepath)
        rec = reloaded.get_sub_agent_thread("claude-code")
        assert rec is not None, "record lost across reload — persistence missing"
        assert rec["session_id"] == "sid-42"
        assert rec["model"] == "m-1"

    async def test_last_write_wins_per_handle(self, tmp_path):
        s = _new_session(tmp_path)
        s.set_sub_agent_thread("claude-code", session_id="sid-old", model="m")
        s.set_sub_agent_thread("claude-code", session_id="sid-new", model="m")
        reloaded = Session.load(s._filepath)
        assert reloaded.get_sub_agent_thread("claude-code")["session_id"] == "sid-new"

    async def test_handles_are_independent_rows(self, tmp_path):
        """One session owns both a claude-code and a codex thread — composite
        (SessionKey, handle) keying, both coexist."""
        s = _new_session(tmp_path)
        s.set_sub_agent_thread("claude-code", session_id="sid-cc", model="m1")
        s.set_sub_agent_thread("codex", session_id="thread-cx", model="gpt-5.4-mini")
        reloaded = Session.load(s._filepath)
        assert reloaded.get_sub_agent_thread("claude-code")["session_id"] == "sid-cc"
        assert reloaded.get_sub_agent_thread("codex")["session_id"] == "thread-cx"

    async def test_meta_rows_never_reach_the_llm(self, tmp_path):
        s = _new_session(tmp_path)
        s.set_sub_agent_thread("claude-code", session_id="sid-1", model="m")
        reloaded = Session.load(s._filepath)
        assert all(m.get("role") != "meta" for m in reloaded.get_messages())


# ---------------------------------------------------------------------------
# 2. Transport: thread identity travels on turn_complete; resume options pass
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
class TestTransportThreadIdentity:

    async def test_turn_complete_carries_session_id_and_model(self):
        from claude_agent_sdk.types import AssistantMessage, TextBlock, ResultMessage

        transport = SDKTransport()
        mock_client = AsyncMock()
        assistant = AssistantMessage(
            content=[TextBlock(text="done")], model="claude-test-model",
            parent_tool_use_id=None, error=None,
        )
        result = ResultMessage(
            subtype="success", duration_ms=1, duration_api_ms=1,
            is_error=False, num_turns=1, session_id="sid-9",
            total_cost_usd=0.0, usage={}, result=None, structured_output=None,
        )

        async def receive():
            yield assistant
            yield result

        mock_client.receive_response = lambda: receive()
        transport._client = mock_client
        transport._alive = True
        await transport.dispatch("go")
        await asyncio.gather(transport._bg_task, return_exceptions=True)

        events = []
        while True:
            try:
                events.append(transport._event_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        tc = [e for e in events if e.event_type == "turn_complete"][0]
        assert tc.data.get("cause") == "success"
        assert tc.data.get("session_id") == "sid-9", (
            "turn_complete must carry the resume session id — nothing reads "
            "transport.session_id today, so the id must travel with the event"
        )
        assert tc.data.get("model") == "claude-test-model"

    async def test_resume_options_passed_to_sdk_client(self, monkeypatch):
        """SDKTransport(resume_session_id=..., model=...) must reach
        ClaudeAgentOptions(resume=..., model=...)."""
        captured = {}

        class _StubClient:
            def __init__(self, options):
                captured["options"] = options

            async def connect(self):
                pass

        import agent_os.agent.transports.sdk_transport as st
        monkeypatch.setattr(st, "ClaudeSDKClient", _StubClient)

        transport = SDKTransport(resume_session_id="sid-resume-1",
                                 model="claude-test-model")
        transport._capture_process_handle = lambda: None
        await transport.start(command="claude", args=[], workspace="/tmp", env={})

        opts = captured["options"]
        assert getattr(opts, "resume", None) == "sid-resume-1"
        assert getattr(opts, "model", None) == "claude-test-model"

    async def test_no_resume_args_means_no_resume_option(self, monkeypatch):
        captured = {}

        class _StubClient:
            def __init__(self, options):
                captured["options"] = options

            async def connect(self):
                pass

        import agent_os.agent.transports.sdk_transport as st
        monkeypatch.setattr(st, "ClaudeSDKClient", _StubClient)

        transport = SDKTransport()
        transport._capture_process_handle = lambda: None
        await transport.start(command="claude", args=[], workspace="/tmp", env={})
        assert getattr(captured["options"], "resume", None) is None


@pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
class TestResumeSourcePrecheck:
    """Do NOT trust the resume call to error on a dead id — pre-check the
    store file (claude-code: ~/.claude/projects/<cwd-slug>/<sid>.jsonl)."""

    async def test_present_via_derived_slug(self, tmp_path):
        projects_root = tmp_path / "projects"
        workspace = tmp_path / "work" / "my-app"
        workspace.mkdir(parents=True)
        slug = SDKTransport.claude_project_slug(str(workspace))
        d = projects_root / slug
        d.mkdir(parents=True)
        (d / "sid-1.jsonl").write_text("{}\n")

        assert SDKTransport.resume_source_exists(
            str(workspace), "sid-1", projects_root=str(projects_root))

    async def test_absent_returns_false(self, tmp_path):
        projects_root = tmp_path / "projects"
        projects_root.mkdir()
        workspace = tmp_path / "work"
        workspace.mkdir()
        assert not SDKTransport.resume_source_exists(
            str(workspace), "sid-gone", projects_root=str(projects_root))

    async def test_glob_fallback_when_slug_derivation_differs(self, tmp_path):
        """If claude's slug derivation ever differs from ours, the glob
        fallback still finds the session file."""
        projects_root = tmp_path / "projects"
        other = projects_root / "-some-unexpected-slug"
        other.mkdir(parents=True)
        (other / "sid-7.jsonl").write_text("{}\n")
        workspace = tmp_path / "work"
        workspace.mkdir()
        assert SDKTransport.resume_source_exists(
            str(workspace), "sid-7", projects_root=str(projects_root))

    async def test_slug_derivation_matches_observed_layout(self):
        """Observed: /Users/x/Desktop/orbital-test -> -Users-x-Desktop-orbital-test
        and /tmp/... realpaths to /private/tmp/... on macOS."""
        slug = SDKTransport.claude_project_slug("/Users/x/Desktop/orbital-test")
        assert slug == "-Users-x-Desktop-orbital-test"


# ---------------------------------------------------------------------------
# 3. Capture: completion routes thread identity to the record
# ---------------------------------------------------------------------------


class _FakeTransport:
    def __init__(self):
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._alive = True

    async def read_stream(self):
        while True:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.2)
                yield event
            except asyncio.TimeoutError:
                if not self._alive:
                    break
                continue

    async def stop(self):
        self._alive = False

    def is_alive(self):
        return self._alive


class TestCaptureRouting:

    async def _run(self, events):
        from agent_os.agent.adapters.cli_adapter import CLIAdapter

        transport = _FakeTransport()
        adapter = CLIAdapter(handle="claude-code", display_name="CC",
                             transport=transport)
        observer = MagicMock()
        observer.on_completed = AsyncMock()
        observer.on_error = AsyncMock()
        observer.on_thread_update = AsyncMock()
        pm = ProcessManager(ws_manager=MagicMock(),
                            activity_translator=MagicMock(),
                            lifecycle_observer=observer)
        transcript = MagicMock()
        transcript.filepath = "/tmp/t.jsonl"
        for e in events:
            await transport._event_queue.put(e)
        await transport.stop()
        await pm.start("proj_x", "claude-code", adapter, transcript=transcript,
                       session_id="sess_x")
        await asyncio.wait_for(pm._tasks["proj_x:claude-code"], timeout=5.0)
        return observer

    async def test_success_completion_records_thread(self):
        observer = await self._run([
            TransportEvent("message", data={"text": "ok"}, raw_text="ok"),
            TransportEvent("turn_complete",
                           data={"cause": "success", "session_id": "sid-7",
                                 "model": "m-x"}),
        ])
        assert observer.on_thread_update.await_count == 1
        call = observer.on_thread_update.await_args
        assert call.args[0] == "proj_x"
        assert call.args[1] == "claude-code"
        assert call.kwargs.get("claude_session_id") == "sid-7"
        assert call.kwargs.get("model") == "m-x"
        assert call.kwargs.get("session_id") == "sess_x"

    async def test_error_completion_still_records_thread(self):
        """A failed turn still has a live, resumable session — record it so
        the next dispatch can resume with the context that led to the error."""
        observer = await self._run([
            TransportEvent("error", data={"error": "boom"}, raw_text="Error: boom"),
            TransportEvent("turn_complete",
                           data={"cause": "error", "session_id": "sid-8",
                                 "model": "m-x"}),
        ])
        assert observer.on_thread_update.await_count == 1
        assert observer.on_thread_update.await_args.kwargs.get(
            "claude_session_id") == "sid-8"

    async def test_no_session_id_means_no_record(self):
        """Reaped before first ResultMessage -> no id -> nothing recorded
        (the fresh/first_spawn edge is explicit, not an error)."""
        observer = await self._run([
            TransportEvent("turn_complete", data={"cause": "stopped"}),
        ])
        observer.on_thread_update.assert_not_awaited()


class TestLifecycleThreadUpdate:

    async def test_on_thread_update_routes_to_agent_manager(self):
        mock_am = MagicMock()
        mock_am.record_sub_agent_thread = MagicMock()
        observer = LifecycleObserver(agent_manager=mock_am, ws_manager=MagicMock())
        await observer.on_thread_update(
            "proj_x", "claude-code", claude_session_id="sid-1", model="m",
            session_id="sess_x",
        )
        mock_am.record_sub_agent_thread.assert_called_once_with(
            "proj_x", "claude-code", claude_session_id="sid-1", model="m",
            session_id="sess_x",
        )


class TestAgentManagerRecord:

    async def test_record_writes_into_hydrated_session(self, tmp_path):
        from agent_os.daemon_v2.agent_manager import AgentManager
        from agent_os.daemon_v2.models import make_session_key

        session = _new_session(tmp_path)
        am = AgentManager.__new__(AgentManager)
        am._handles = {
            make_session_key("proj_x", "sess_x"): SimpleNamespace(session=session)
        }
        am.record_sub_agent_thread(
            "proj_x", "claude-code", claude_session_id="sid-1", model="m",
            session_id="sess_x",
        )
        assert session.get_sub_agent_thread("claude-code")["session_id"] == "sid-1"
        # And it survives reload — the full capture->disk chain.
        assert Session.load(session._filepath).get_sub_agent_thread(
            "claude-code")["session_id"] == "sid-1"

    async def test_record_with_no_hydrated_session_is_a_noop(self):
        from agent_os.daemon_v2.agent_manager import AgentManager

        am = AgentManager.__new__(AgentManager)
        am._handles = {}
        # Must not raise — completion racing an eviction is logged, not fatal.
        am.record_sub_agent_thread(
            "proj_x", "claude-code", claude_session_id="sid-1", model="m",
            session_id="sess_x",
        )


# ---------------------------------------------------------------------------
# 4. Dispatch: resume decision + honesty
# ---------------------------------------------------------------------------


def _manager_with_resolver(session):
    mgr = SubAgentManager(process_manager=MagicMock())
    mgr._session_resolver = (lambda pid, sid: session)
    return mgr


class TestDetermineResume:

    async def test_no_record_is_fresh_first_spawn(self, tmp_path):
        session = _new_session(tmp_path)
        mgr = _manager_with_resolver(session)
        rec, status, reason = mgr._determine_resume(
            str(tmp_path), "proj_x", "claude-code", "sess_x")
        assert rec is None
        assert (status, reason) == ("fresh", "first_spawn")

    async def test_no_resolver_is_fresh_first_spawn(self, tmp_path):
        mgr = SubAgentManager(process_manager=MagicMock())
        rec, status, reason = mgr._determine_resume(
            str(tmp_path), "proj_x", "claude-code", "sess_x")
        assert (rec, status, reason) == (None, "fresh", "first_spawn")

    async def test_record_with_live_source_resumes(self, tmp_path, monkeypatch):
        session = _new_session(tmp_path)
        session.set_sub_agent_thread("claude-code", session_id="sid-1", model="m")
        mgr = _manager_with_resolver(session)
        monkeypatch.setattr(SDKTransport, "resume_source_exists",
                            staticmethod(lambda ws, sid, projects_root=None: True))
        rec, status, reason = mgr._determine_resume(
            str(tmp_path), "proj_x", "claude-code", "sess_x")
        assert rec["session_id"] == "sid-1"
        assert (status, reason) == ("resumed", None)

    async def test_record_with_dead_source_is_fresh_resume_failed(
            self, tmp_path, monkeypatch):
        """R4 honesty: pruned/missing store -> fresh AND says so. Pre-check,
        do not trust the resume call to fail loudly."""
        session = _new_session(tmp_path)
        session.set_sub_agent_thread("claude-code", session_id="sid-1", model="m")
        mgr = _manager_with_resolver(session)
        monkeypatch.setattr(SDKTransport, "resume_source_exists",
                            staticmethod(lambda ws, sid, projects_root=None: False))
        rec, status, reason = mgr._determine_resume(
            str(tmp_path), "proj_x", "claude-code", "sess_x")
        assert rec is None
        assert (status, reason) == ("fresh", "resume_failed")


class TestResumeStatusReporting:

    def test_clauses_are_honest(self):
        resumed = SubAgentManager._format_resume_clause("resumed", None)
        first = SubAgentManager._format_resume_clause("fresh", "first_spawn")
        failed = SubAgentManager._format_resume_clause("fresh", "resume_failed")

        assert "resumed" in resumed
        # A fresh start must NEVER read as a resume.
        assert "resumed" not in first
        assert "resumed" not in failed or "could not be resumed" in failed
        assert "fresh" in first
        assert "fresh" in failed
        # resume_failed is recorded distinctly (honesty/observability).
        assert "first spawn" in first
        assert "could not be resumed" in failed
