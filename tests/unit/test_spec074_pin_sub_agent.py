# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 074 — pin a sub-agent in the main chat (direct chat, manager out of
the loop).

Covers the spec §4 test list:

- ``pinned_target`` meta round-trip (survives load, coexists with
  name/pinned).
- PATCH validation: unknown slug rejected, explicit null clears, ``@orbital``
  rejected as a pin target; retarget/unpin kicks the consolidation trigger;
  pin-time AGENTS.md refresh never touches a user-edited file.
- Wake suppression at BOTH consumer sites in BOTH directions: a suppressed
  (pinned) terminal event appends but never starts the management loop; an
  unflagged event wakes exactly as today (idle path AND deferred drain).
- Producer side: the lifecycle observer stamps ``suppress_wake`` ONLY for
  dispatches recorded as ``user_pinned``; plain events stay unflagged.
- Recap preamble: built for a fresh thread / retarget, absent on own-thread
  resume, ~10k cap enforced, worker-anonymous.
- Consolidation triggers: retarget fires, quiescence fires once and resets on
  a new dispatch, single-flight + dirty-flag coalescing.
- Pinned spawn prompt: write contract + retraction titles present; the
  non-pinned ban is unchanged.
- AGENTS.md hash guard: seeded-and-unedited refreshes, user-edited untouched.
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.session import Session
from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
from agent_os.daemon_v2.pinned_consolidation import (
    PinnedConsolidationCoordinator,
)

SID = "sess-spec074-0001"


# ---------------------------------------------------------------------------
# 1. Session meta round-trip
# ---------------------------------------------------------------------------


def _mint(tmp_path, stem="chat_pin074_deadbeef"):
    s = Session.new(stem, str(tmp_path))
    s.append({"role": "user", "content": "hello there", "source": "user"})
    return s


def _start_meta(tmp_path, stem="chat_pin074_deadbeef"):
    p = tmp_path / "orbital" / "sessions" / f"{stem}.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("role") == "meta" and rec.get("event") == "session_start":
            return rec
    raise AssertionError("no session_start meta on disk")


def _path(tmp_path, stem="chat_pin074_deadbeef"):
    return str(tmp_path / "orbital" / "sessions" / f"{stem}.jsonl")


class TestPinnedTargetMetaRoundTrip:

    def test_defaults_none_and_writes_nothing(self, tmp_path):
        """Pre-074 logs carry no key; absence must read as unpinned (no
        migration)."""
        _mint(tmp_path)
        assert "pinned_target" not in _start_meta(tmp_path)
        assert Session.load(_path(tmp_path)).pinned_target is None

    def test_round_trips_through_disk_and_null_clears(self, tmp_path):
        s = _mint(tmp_path)
        s.set_pinned_target("claude-code")
        assert s.pinned_target == "claude-code"
        assert _start_meta(tmp_path)["pinned_target"] == "claude-code"
        assert Session.load(_path(tmp_path)).pinned_target == "claude-code"

        s.set_pinned_target(None)
        assert s.pinned_target is None
        assert _start_meta(tmp_path)["pinned_target"] is None
        assert Session.load(_path(tmp_path)).pinned_target is None

    def test_coexists_with_name_and_pinned(self, tmp_path):
        """Three writers on ONE meta line — none may clobber the others."""
        s = _mint(tmp_path)
        s.set_name("Direct codex chat")
        s.set_pinned(True)
        s.set_pinned_target("codex")
        meta = _start_meta(tmp_path)
        assert meta["name"] == "Direct codex chat"
        assert meta["pinned"] is True
        assert meta["pinned_target"] == "codex"

        loaded = Session.load(_path(tmp_path))
        assert loaded.name == "Direct codex chat"
        assert loaded.pinned is True
        assert loaded.pinned_target == "codex"


# ---------------------------------------------------------------------------
# 2. Wake suppression — consumer site A: inject_system_message
# ---------------------------------------------------------------------------


@pytest.fixture
def manager():
    ws = MagicMock()
    ws.broadcast = MagicMock()
    mgr = AgentManager(
        project_store=MagicMock(),
        ws_manager=ws,
        sub_agent_manager=MagicMock(list_active=MagicMock(return_value=[])),
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )
    return mgr


def _idle_handle(manager):
    session = MagicMock()
    session.append = MagicMock()
    handle = MagicMock()
    handle.session = session
    handle.task = None  # loop idle
    manager._handles[("proj_test", SID)] = handle
    return handle


TERMINAL_META = {"event": "sub_agent_terminal", "kind": "completed"}
SUPPRESSED_META = {"event": "sub_agent_terminal", "kind": "completed",
                   "suppress_wake": True}


class TestInjectSystemMessageSuppression:

    @pytest.mark.asyncio
    async def test_idle_suppressed_appends_but_never_starts_loop(self, manager):
        handle = _idle_handle(manager)
        manager._start_loop = AsyncMock()

        result = await manager.inject_system_message(
            "proj_test", "[Sub-agent] codex completed.",
            session_id=SID, meta=dict(SUPPRESSED_META),
        )

        assert result == "suppressed"
        handle.session.append.assert_called_once()
        appended = handle.session.append.call_args[0][0]
        assert appended["_meta"]["suppress_wake"] is True
        manager._start_loop.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_idle_unflagged_still_wakes(self, manager):
        """The other direction: an unflagged terminal must keep today's wake."""
        handle = _idle_handle(manager)
        manager._start_loop = AsyncMock()

        result = await manager.inject_system_message(
            "proj_test", "[Sub-agent] codex completed.",
            session_id=SID, meta=dict(TERMINAL_META),
        )

        assert result == "delivered"
        handle.session.append.assert_called_once()
        manager._start_loop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_handle_suppressed_persists_without_start_agent(self, manager):
        loaded = MagicMock()
        loaded.session_uuid = "chat_x_cafebabe"
        manager._load_session_from_disk = MagicMock(return_value=loaded)
        manager.start_agent = AsyncMock()

        result = await manager.inject_system_message(
            "proj_test", "[Sub-agent] codex completed.",
            session_id=SID, meta=dict(SUPPRESSED_META),
        )

        assert result == "suppressed"
        loaded.append.assert_called_once()  # row is durable on disk
        manager.start_agent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_handle_unflagged_still_hydrates_and_wakes(self, manager):
        loaded = MagicMock()
        loaded.session_uuid = "chat_x_cafebabe"
        manager._load_session_from_disk = MagicMock(return_value=loaded)
        manager._build_agent_config_from_project = MagicMock()
        manager.start_agent = AsyncMock()

        result = await manager.inject_system_message(
            "proj_test", "[Sub-agent] codex completed.",
            session_id=SID, meta=dict(TERMINAL_META),
        )

        assert result == "delivered"
        loaded.append.assert_called_once()
        manager.start_agent.assert_awaited_once()


# ---------------------------------------------------------------------------
# 2b. Wake suppression — consumer site B: the deferred drain
# ---------------------------------------------------------------------------


def _handle_with_deferred(manager, deferred_messages):
    session = MagicMock()
    session.is_stopped.return_value = False
    session._paused_for_approval = False
    session.pop_deferred_messages.return_value = deferred_messages
    session.pop_queued_messages.return_value = []
    session.append = MagicMock()

    handle = MagicMock()
    handle.session = session
    handle.loop = MagicMock(last_llm_error=None)
    task = MagicMock()
    task.exception.return_value = None
    handle.task = task
    manager._handles[("proj_test", SID)] = handle
    return handle, task


def _run_loop_done(manager, task):
    callback = manager._on_loop_done("proj_test", session_id=SID)
    mock_future = MagicMock()
    with patch("asyncio.ensure_future", return_value=mock_future) as mock_ensure:
        callback(task)
        if mock_ensure.call_args:
            mock_ensure.call_args[0][0].close()
    return mock_ensure


def _deferred_terminal(meta):
    return {
        "role": "system",
        "content": "[Sub-agent] codex completed.",
        "source": "daemon",
        "timestamp": "2026-08-30T00:00:00+00:00",
        "_meta": dict(meta),
    }


class TestDeferredDrainSuppression:

    def test_suppressed_terminal_appends_but_never_wakes(self, manager):
        handle, task = _handle_with_deferred(
            manager, [_deferred_terminal(SUPPRESSED_META)])

        mock_ensure = _run_loop_done(manager, task)

        handle.session.append.assert_called_once()
        appended = handle.session.append.call_args[0][0]
        assert appended["_meta"]["suppress_wake"] is True
        mock_ensure.assert_not_called()

    def test_unflagged_terminal_still_wakes(self, manager):
        handle, task = _handle_with_deferred(
            manager, [_deferred_terminal(TERMINAL_META)])

        mock_ensure = _run_loop_done(manager, task)

        handle.session.append.assert_called_once()
        mock_ensure.assert_called_once()

    def test_mixed_batch_wakes_for_the_unflagged_event(self, manager):
        """A suppressed pinned terminal must not mask an unflagged one that
        arrived in the same drain."""
        handle, task = _handle_with_deferred(
            manager,
            [_deferred_terminal(SUPPRESSED_META),
             _deferred_terminal(TERMINAL_META)],
        )

        mock_ensure = _run_loop_done(manager, task)

        assert handle.session.append.call_count == 2
        mock_ensure.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Producer side — LifecycleObserver stamps suppress_wake only for pinned
# ---------------------------------------------------------------------------


class _RecordingManager:
    def __init__(self):
        self.injections: list[tuple[str, dict]] = []

    async def inject_system_message(self, project_id, content, **kwargs):
        self.injections.append((content, kwargs))


def _observer():
    am = _RecordingManager()
    obs = LifecycleObserver(am, MagicMock())
    return obs, am


class TestObserverStamping:

    @pytest.mark.asyncio
    async def test_pinned_dispatch_stamps_suppress_wake_on_completed(self):
        obs, am = _observer()
        obs.set_dispatch_initiator(
            "proj_test", "codex", "user_pinned", session_id=SID)

        await obs.on_completed(
            "proj_test", "codex", "done", "/tmp/t.jsonl", session_id=SID)

        _content, kwargs = am.injections[0]
        meta = kwargs["meta"]
        assert meta["event"] == "sub_agent_terminal"
        assert meta["suppress_wake"] is True
        # The #24 display split survives the added key.
        assert meta["kind"] == "completed"
        assert "display_content" in meta

    @pytest.mark.asyncio
    @pytest.mark.parametrize("initiator", ["management_agent", "user_mention"])
    async def test_non_pinned_dispatch_stays_unflagged(self, initiator):
        """The seven producers' defaults are untouched: anything but
        user_pinned must not carry the flag (mention behavior unchanged)."""
        obs, am = _observer()
        obs.set_dispatch_initiator(
            "proj_test", "codex", initiator, session_id=SID)

        await obs.on_completed(
            "proj_test", "codex", "done", "/tmp/t.jsonl", session_id=SID)
        await obs.on_error(
            "proj_test", "codex", "boom", "/tmp/t.jsonl", session_id=SID)

        for _content, kwargs in am.injections:
            assert "suppress_wake" not in kwargs["meta"]

    @pytest.mark.asyncio
    async def test_unknown_dispatch_key_stays_unflagged(self):
        """Fail-open: an unattributable terminal must wake exactly as today."""
        obs, am = _observer()

        await obs.on_error(
            "proj_test", "codex", "boom", "/tmp/t.jsonl", session_id=SID)

        assert "suppress_wake" not in am.injections[0][1]["meta"]

    @pytest.mark.asyncio
    async def test_later_management_dispatch_clears_the_pin(self):
        obs, am = _observer()
        obs.set_dispatch_initiator(
            "proj_test", "codex", "user_pinned", session_id=SID)
        obs.set_dispatch_initiator(
            "proj_test", "codex", "management_agent", session_id=SID)

        await obs.on_completed(
            "proj_test", "codex", "done", "/tmp/t.jsonl", session_id=SID)

        assert "suppress_wake" not in am.injections[0][1]["meta"]

    @pytest.mark.asyncio
    async def test_pinned_routed_marker_is_suppressed_and_unsupervised(self):
        """Zero manager turns on DISPATCH too: the pinned 'Message sent to'
        marker rides suppress_wake and carries no supervision guidance."""
        obs, am = _observer()

        await obs.on_message_routed(
            "proj_test", "codex", "user_pinned", "fix the bug",
            "/tmp/t.jsonl", session_id=SID, dispatch_id="d1")

        content, kwargs = am.injections[0]
        assert kwargs["meta"]["suppress_wake"] is True
        assert "supervise" not in content

    @pytest.mark.asyncio
    async def test_mention_routed_marker_unchanged(self):
        obs, am = _observer()

        await obs.on_message_routed(
            "proj_test", "codex", "user_mention", "fix the bug",
            "/tmp/t.jsonl", session_id=SID, dispatch_id="d1")

        content, kwargs = am.injections[0]
        assert "suppress_wake" not in kwargs["meta"]
        assert "supervise or relay" in content

    @pytest.mark.asyncio
    async def test_pinned_terminal_fires_quiescence_hook(self):
        obs, _am = _observer()
        obs.set_dispatch_initiator(
            "proj_test", "codex", "user_pinned", session_id=SID)
        hook = MagicMock()
        obs.pinned_terminal_hook = hook

        await obs.on_completed(
            "proj_test", "codex", "done", "/tmp/t.jsonl", session_id=SID)

        hook.assert_called_once_with("proj_test", SID)

    @pytest.mark.asyncio
    async def test_unpinned_terminal_does_not_fire_hook(self):
        obs, _am = _observer()
        hook = MagicMock()
        obs.pinned_terminal_hook = hook

        await obs.on_completed(
            "proj_test", "codex", "done", "/tmp/t.jsonl", session_id=SID)

        hook.assert_not_called()

    @pytest.mark.asyncio
    async def test_queue_dropped_uses_the_dropped_prompts_own_initiator(self):
        obs, am = _observer()
        # Registry says the LAST dispatched turn was management-initiated…
        obs.set_dispatch_initiator(
            "proj_test", "codex", "management_agent", session_id=SID)

        # …but the dropped queued prompt itself was pinned.
        await obs.on_queue_dropped(
            "proj_test", "codex", why="stopped by user", session_id=SID,
            initiator="user_pinned")

        assert am.injections[0][1]["meta"]["suppress_wake"] is True


class TestDispatchPushesInitiator:

    @pytest.mark.asyncio
    async def test_dispatch_prompt_locked_records_initiator_on_observer(self):
        from agent_os.daemon_v2.sub_agent_manager import (
            SubAgentManager, _QueuedPrompt,
        )

        observer = MagicMock()
        observer.on_message_routed = AsyncMock()
        mgr = SubAgentManager(
            process_manager=MagicMock(),
            lifecycle_observer=observer,
        )
        mgr._dispatch_async = AsyncMock()

        prompt = _QueuedPrompt(
            message="fix it", dispatch_id="d1",
            transcript_path="/tmp/t.jsonl", initiator="user_pinned",
        )
        await mgr._dispatch_prompt_locked(
            MagicMock(), prompt, "proj_test", SID, "codex")

        observer.set_dispatch_initiator.assert_called_once_with(
            "proj_test", "codex", "user_pinned", session_id=SID)
        observer.on_message_routed.assert_awaited_once()
        assert observer.on_message_routed.await_args.kwargs[
            "initiator"] == "user_pinned"


# ---------------------------------------------------------------------------
# 4. Recap preamble
# ---------------------------------------------------------------------------


from agent_os.api.routes.agents_v2 import (  # noqa: E402
    _RECAP_CAP_CHARS,
    _build_recap_preamble,
)


class _FakeSession:
    def __init__(self, messages):
        self._messages = messages

    def get_messages(self):
        return list(self._messages)


def _user(content, target=None):
    m = {"role": "user", "content": content, "source": "user"}
    if target:
        m["target"] = target
    return m


def _assistant(content):
    return {"role": "assistant", "content": content, "source": "management"}


def _completed_row(handle, summary):
    display = (f"[Sub-agent] {handle} completed. Summary: {summary}. "
               f"Transcript: /tmp/{handle}.jsonl.")
    return {
        "role": "system", "content": display + " (guidance)",
        "source": "daemon",
        "_meta": {"event": "sub_agent_terminal", "kind": "completed",
                  "display_content": display},
    }


def _routed_marker(handle):
    return {
        "role": "system",
        "content": f'[Sub-agent] Message sent to {handle}: "hi".',
        "source": "daemon",
        "_meta": {"dispatch_id": "d0", "handle": handle,
                  "transcript_path": "/tmp/t.jsonl"},
    }


class TestRecapPreamble:

    def test_fresh_thread_gets_the_whole_session(self):
        session = _FakeSession([
            _user("set up the repo"),
            _assistant("Repo is set up."),
            _user("now add CI"),
            _assistant("CI added."),
        ])
        recap = _build_recap_preamble(session, "codex")
        assert recap.startswith("Conversation so far")
        assert "user: set up the repo" in recap
        assert "assistant: Repo is set up." in recap
        assert "assistant: CI added." in recap
        assert recap.endswith("--- end of conversation so far ---\n\n")

    def test_own_thread_resume_gets_no_recap(self):
        """Nothing after the worker's last participation → empty recap."""
        session = _FakeSession([
            _user("fix the bug", target="codex"),
            _routed_marker("codex"),
            _completed_row("codex", "Bug fixed"),
        ])
        assert _build_recap_preamble(session, "codex") == ""

    def test_retarget_covers_only_the_missed_middle(self):
        """Retarget BACK to a worker: recap = messages since ITS last
        participation, not the whole session."""
        session = _FakeSession([
            _user("early codex task", target="codex"),
            _routed_marker("codex"),
            _completed_row("codex", "Early task done"),
            _user("now you, claude", target="claude-code"),
            _routed_marker("claude-code"),
            _completed_row("claude-code", "Middle work finished"),
        ])
        recap = _build_recap_preamble(session, "codex")
        assert "early codex task" not in recap
        assert "user: now you, claude" in recap
        assert "assistant: Middle work finished" in recap

    def test_worker_anonymous(self):
        """Prior replies are labeled 'assistant'; the producing agent's name
        never appears in a recap label."""
        session = _FakeSession([
            _user("do the thing"),
            _completed_row("claude-code", "Thing is done"),
        ])
        recap = _build_recap_preamble(session, "codex")
        assert "assistant: Thing is done" in recap
        assert "claude-code" not in recap
        assert "[Sub-agent]" not in recap

    def test_cap_enforced_and_favors_newest(self):
        big = [_user(f"message number {i} " + "x" * 400) for i in range(60)]
        recap = _build_recap_preamble(_FakeSession(big), "codex")
        # Frame overhead is small; the body must respect the ~10k cap.
        assert len(recap) < _RECAP_CAP_CHARS + 500
        assert "message number 59" in recap  # newest kept
        assert "message number 0" not in recap  # oldest dropped

    def test_empty_session_gets_no_recap(self):
        assert _build_recap_preamble(_FakeSession([]), "codex") == ""


# ---------------------------------------------------------------------------
# 5. Consolidation triggers
# ---------------------------------------------------------------------------


class TestConsolidationCoordinator:

    def _coordinator(self, quiescence_s=0.05):
        c = PinnedConsolidationCoordinator(
            MagicMock(), MagicMock(), quiescence_s=quiescence_s)
        c.pass_log = []

        async def _fake_run_pass(key, reason):
            c.pass_log.append((key, reason))

        c._run_pass = _fake_run_pass
        return c

    @pytest.mark.asyncio
    async def test_retarget_trigger_fires_a_detached_pass(self):
        c = self._coordinator()
        c.trigger("proj", SID, reason="retarget")
        await asyncio.sleep(0.01)
        assert c.pass_log == [(("proj", SID), "retarget")]

    @pytest.mark.asyncio
    async def test_quiescence_fires_once_per_quiet_period(self):
        c = self._coordinator(quiescence_s=0.03)
        c.note_pinned_terminal("proj", SID)
        await asyncio.sleep(0.1)
        assert c.pass_log == [(("proj", SID), "quiescence")]
        # No re-fire without a new terminal event.
        await asyncio.sleep(0.08)
        assert len(c.pass_log) == 1

    @pytest.mark.asyncio
    async def test_new_dispatch_cancels_the_quiescence_timer(self):
        c = self._coordinator(quiescence_s=0.03)
        c.note_pinned_terminal("proj", SID)
        c.note_pinned_dispatch("proj", SID)
        await asyncio.sleep(0.1)
        assert c.pass_log == []

    @pytest.mark.asyncio
    async def test_terminal_resets_the_timer(self):
        c = self._coordinator(quiescence_s=0.06)
        c.note_pinned_terminal("proj", SID)
        first_timer = c._timers[("proj", SID)]
        await asyncio.sleep(0.02)
        c.note_pinned_terminal("proj", SID)
        second_timer = c._timers[("proj", SID)]
        assert second_timer is not first_timer
        await asyncio.sleep(0.01)  # let the cancellation settle
        # _quiescence_wait absorbs the CancelledError and returns, so the
        # cancelled timer reads as done — and it must never have fired.
        assert first_timer.done()
        assert c.pass_log == []
        c.note_pinned_dispatch("proj", SID)  # cleanup: no lingering timer

    @pytest.mark.asyncio
    async def test_single_flight_with_dirty_flag_coalescing(self):
        c = PinnedConsolidationCoordinator(
            MagicMock(), MagicMock(), quiescence_s=60)
        c.pass_log = []
        gate = asyncio.Event()

        async def _blocking_pass(key, reason):
            c.pass_log.append((key, reason))
            await gate.wait()

        c._run_pass = _blocking_pass
        c.trigger("proj", SID, reason="retarget")
        await asyncio.sleep(0.01)
        assert len(c.pass_log) == 1

        # Three triggers while running coalesce into ONE dirty re-run.
        c.trigger("proj", SID, reason="quiescence")
        c.trigger("proj", SID, reason="unpin")
        c.trigger("proj", SID, reason="unpin")
        await asyncio.sleep(0.01)
        assert len(c.pass_log) == 1  # still single-flight

        gate.set()
        await asyncio.sleep(0.05)
        assert len(c.pass_log) == 2  # exactly one re-run
        gate.set()

    @pytest.mark.asyncio
    async def test_pass_windows_since_last_pass(self):
        """The pass hands run_session_end_routine the message index of the
        previous pass (in-memory) so each pass distills only the new tail."""
        agent_manager = MagicMock()
        session = MagicMock()
        session.get_messages.return_value = [{"role": "user"}] * 7
        session.session_uuid = "chat_x_cafebabe"
        agent_manager.get_session.return_value = session
        cfg = MagicMock()
        cfg.workspace = "/tmp/ws074"
        agent_manager._build_agent_config_from_project.return_value = cfg
        agent_manager._build_llm_providers.return_value = (
            MagicMock(), [], MagicMock(), {})

        c = PinnedConsolidationCoordinator(agent_manager, MagicMock())
        calls = []

        async def _fake_routine(**kwargs):
            calls.append(kwargs)
            return "llm_merged"

        with patch(
            "agent_os.agent.workspace_files.run_session_end_routine",
            new=_fake_routine,
        ), patch("agent_os.agent.workspace_files.WorkspaceFileManager"):
            await c._run_pass(("proj", SID), "retarget")
            session.get_messages.return_value = [{"role": "user"}] * 12
            await c._run_pass(("proj", SID), "quiescence")

        assert calls[0]["since_index"] is None
        assert calls[0]["pinned_exchange"] is True
        assert calls[0]["bypass_idempotency"] is True
        assert calls[1]["since_index"] == 7


# ---------------------------------------------------------------------------
# 6. Pinned spawn prompt
# ---------------------------------------------------------------------------


from agent_os.agent.sub_agent_prompt import render_sub_agent_prompt  # noqa: E402


class TestPinnedSpawnPrompt:

    def test_non_pinned_ban_unchanged(self, tmp_path):
        ws = str(tmp_path)
        rendered = render_sub_agent_prompt(ws, None, "codex", ["codex"])
        assert "Do NOT modify these orbital-managed files:" in rendered
        assert "You can read them; do not write to them." in rendered
        assert "PINNED MODE" not in rendered
        # Default equals explicit pinned=False, byte for byte.
        assert rendered == render_sub_agent_prompt(
            ws, None, "codex", ["codex"], pinned=False)

    def test_pinned_prompt_carries_the_write_contract(self, tmp_path):
        ws = str(tmp_path)
        rendered = render_sub_agent_prompt(
            ws, None, "codex", ["codex"], pinned=True)
        assert "PINNED MODE" in rendered
        assert "no manager is mediating" in rendered
        assert "APPEND entries to" in rendered
        assert "TARGETED EDITS" in rendered
        assert "NO metadata comment grammar" in rendered
        assert "NEVER reorganize, dedupe, trim, or rewrite" in rendered
        assert "INDEX.md" in rendered
        # The standard ban is REPLACED, not doubled.
        assert "Do NOT modify these orbital-managed files:" not in rendered

    def test_pinned_prompt_includes_retraction_titles(self, tmp_path):
        from agent_os.agent.project_paths import ProjectPaths
        from agent_os.agent.retractions import Retraction, add_retraction

        ws = str(tmp_path)
        orbital = ProjectPaths(ws).orbital_dir
        add_retraction(orbital, Retraction(
            id="r1", title="the emoji rebrand", reason="user said no",
            date="2026-08-01"))
        add_retraction(orbital, Retraction(
            id="r2", title="weekly digest emails", reason="retracted",
            date="2026-08-02"))

        rendered = render_sub_agent_prompt(
            ws, None, "codex", ["codex"], pinned=True)
        assert "NEVER add entries about" in rendered
        assert "the emoji rebrand" in rendered
        assert "weekly digest emails" in rendered

        # No retractions → no dangling clause.
        rendered_plain = render_sub_agent_prompt(
            str(tmp_path / "other"), None, "codex", ["codex"], pinned=True)
        assert "NEVER add entries about" not in rendered_plain


# ---------------------------------------------------------------------------
# 7. AGENTS.md hash-guarded re-seed
# ---------------------------------------------------------------------------


from agent_os.daemon_v2 import agent_md_seeder  # noqa: E402


def _store_for(tmp_path, name="Proj074"):
    ws = tmp_path / "ws074"
    ws.mkdir(exist_ok=True)
    store = MagicMock()
    store.get_project.return_value = {
        "project_id": "proj_test", "name": name, "agent_name": name,
        "workspace": str(ws), "is_scratch": False,
    }
    return store, ws


class TestAgentsMdHashGuard:

    def test_missing_file_is_seeded(self, tmp_path):
        store, ws = _store_for(tmp_path)
        result = agent_md_seeder.reseed_project_agent_md(store, "proj_test")
        assert result["status"] == "seeded"
        assert (ws / "AGENTS.md").exists()

    def test_unedited_current_seed_is_unchanged(self, tmp_path):
        store, ws = _store_for(tmp_path)
        agent_md_seeder.reseed_project_agent_md(store, "proj_test")
        result = agent_md_seeder.reseed_project_agent_md(store, "proj_test")
        assert result["status"] == "unchanged"

    def test_user_edited_file_is_never_touched(self, tmp_path):
        """THE guard: an unconditional rewrite of a user-edited file is a
        data-loss bug."""
        store, ws = _store_for(tmp_path)
        agent_md_seeder.reseed_project_agent_md(store, "proj_test")
        # encoding pinned: the seeder writes utf-8; encoding-less read_text()
        # decodes cp1252 on Windows and garbles the template's em-dash.
        edited = (ws / "AGENTS.md").read_text(
            encoding="utf-8") + "\n## My own notes\nkeep me\n"
        (ws / "AGENTS.md").write_text(edited, encoding="utf-8")

        result = agent_md_seeder.reseed_project_agent_md(store, "proj_test")
        assert result["status"] == "skipped_user_modified"
        assert (ws / "AGENTS.md").read_text(encoding="utf-8") == edited

    def test_historical_seed_is_refreshed(self, tmp_path, monkeypatch):
        store, ws = _store_for(tmp_path)
        old_template = "# AGENTS.md v0\nProject: {project_name}\n"
        (ws / "AGENTS.md").write_text(old_template.format(
            project_name="Proj074", agent_name="Proj074"), encoding="utf-8")
        monkeypatch.setattr(
            agent_md_seeder, "_HISTORICAL_TEMPLATES", (old_template,))

        result = agent_md_seeder.reseed_project_agent_md(store, "proj_test")
        assert result["status"] == "reseeded"
        content = (ws / "AGENTS.md").read_text(encoding="utf-8")
        assert content.startswith("# AGENTS.md — read this first")

    def test_scratch_is_skipped(self, tmp_path):
        store, _ws = _store_for(tmp_path)
        store.get_project.return_value["is_scratch"] = True
        result = agent_md_seeder.reseed_project_agent_md(store, "proj_test")
        assert result["status"] == "skipped_scratch"
