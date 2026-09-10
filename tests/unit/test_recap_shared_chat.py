# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The chat is ONE conversation shared by every agent (manager + workers).

Two lossy bridges used to sit between the management session and a worker's
own thread: the recap block cut every message at 500 chars (a 3.9k-char
pasted application reached claude-code as its first paragraph), and the
completion row kept 500 chars of the worker's reply (the manager saw a
glimpse of each worker turn). Both directions now carry the user-visible
messages in full:

* ``agent_os/daemon_v2/recap.py`` builds the block from the session file at
  DISPATCH time (drain time for a queued prompt) for every entry path —
  pinned, @mention, manager dispatch. The watermark is the worker's last
  dispatch marker; rows the worker already holds natively (messages
  addressed to it, its own completion rows) are skipped; a FRESH spawn gets
  the whole session because its thread is empty.
* ``LifecycleObserver.on_completed`` stores the worker's full reply on the
  row (``_meta.reply`` + LLM-facing ``content``); ``display_content`` keeps
  the short timeline text.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.recap import (
    RECAP_CAP_CHARS,
    REPLY_CAP_CHARS,
    build_recap,
)

SID = "proj_r_1a2b3c4d"


# ---------------------------------------------------------------------------
# Row builders (shape of the real session JSONL rows)
# ---------------------------------------------------------------------------


def _user(content, target=None):
    m = {"role": "user", "content": content, "source": "user"}
    if target:
        m["target"] = target
    return m


def _assistant(content):
    return {"role": "assistant", "content": content, "source": "management"}


def _marker(handle, dispatch_id="d0"):
    return {
        "role": "system", "source": "daemon",
        "content": f'[Sub-agent] Message sent to {handle}: "hi". Transcript: /tmp/t.jsonl',
        "_meta": {"dispatch_id": dispatch_id, "handle": handle,
                  "transcript_path": "/tmp/t.jsonl"},
    }


def _completed(handle, reply, *, legacy=False):
    short = reply[:500]
    display = (f"[Sub-agent] {handle} completed. Summary: {short}. "
               f"Transcript: /tmp/{handle}.jsonl.")
    meta = {"event": "sub_agent_terminal", "kind": "completed",
            "display_content": display}
    if not legacy:
        meta["reply"] = reply
    return {"role": "system", "source": "daemon",
            "content": display + " (guidance)", "_meta": meta}


def _interaction_required(handle):
    return {
        "role": "system", "source": "daemon",
        "content": f"[Sub-agent] {handle} requires input (question): pick one",
        "_meta": {"event": "interaction_required", "handle": handle,
                  "interaction_id": "i1", "kind": "question"},
    }


def _tool_row():
    return {"role": "tool", "content": "grep output", "tool_call_id": "t1",
            "source": "management"}


# ---------------------------------------------------------------------------
# A. build_recap — the gap, the skips, the caps
# ---------------------------------------------------------------------------


class TestRecapGap:

    def test_gap_starts_after_last_marker_and_skips_native_rows(self):
        """U1→cc, C1, U2→orbit, O2, U3→cc: the block for U3 is exactly U2+O2.
        U1 and C1 live in claude-code's own thread; U3 is the message itself."""
        rows = [
            _user("U1 for claude", target="claude-code"),
            _marker("claude-code", "d1"),
            _completed("claude-code", "C1 reply"),
            _user("U2 for orbit"),
            _assistant("O2 from orbit"),
            _user("U3 for claude", target="claude-code"),
        ]
        recap = build_recap(rows, "claude-code", fresh=False)
        assert "user: U2 for orbit" in recap
        assert "assistant: O2 from orbit" in recap
        assert "U1 for claude" not in recap
        assert "C1 reply" not in recap
        assert "U3 for claude" not in recap

    def test_rows_landed_during_in_flight_turn_reach_the_next_dispatch(self):
        """The completion row is NOT the watermark: anything that lands while
        the worker is still running (between its marker and its completion)
        is delivered at its next dispatch."""
        rows = [
            _user("U1", target="codex"),
            _marker("codex", "d1"),
            _user("U2 for orbit while codex runs"),
            _assistant("O2 while codex runs"),
            _completed("codex", "X1 done"),
            _user("U3", target="codex"),
        ]
        recap = build_recap(rows, "codex", fresh=False)
        assert "user: U2 for orbit while codex runs" in recap
        assert "assistant: O2 while codex runs" in recap
        assert "X1 done" not in recap

    def test_other_workers_full_reply_is_included_as_assistant(self):
        rows = [
            _marker("codex", "d1"),
            _completed("codex", "X1 done"),
            _user("now you, claude", target="claude-code"),
            _marker("claude-code", "d2"),
            _completed("claude-code", "long reply " + "y" * 3000),
            _user("back to you", target="codex"),
        ]
        recap = build_recap(rows, "codex", fresh=False)
        assert "user: now you, claude" in recap
        assert "assistant: long reply " + "y" * 3000 in recap
        assert "X1 done" not in recap
        assert "claude-code" not in recap  # worker-anonymous
        assert "[Sub-agent]" not in recap

    def test_three_agents_each_get_only_their_own_gap(self):
        rows = [
            _user("U1", target="claude-code"), _marker("claude-code", "d1"),
            _completed("claude-code", "C1"),
            _user("U2"), _assistant("O2"),
            _user("U3", target="codex"), _marker("codex", "d3"),
            _completed("codex", "X3"),
            _user("U4", target="claude-code"), _marker("claude-code", "d4"),
            _completed("claude-code", "C4"),
        ]
        codex = build_recap(rows, "codex", fresh=False)
        assert "user: U4" in codex and "assistant: C4" in codex
        assert "U2" not in codex and "O2" not in codex  # already sent at d3
        claude = build_recap(rows, "claude-code", fresh=False)
        assert claude == ""  # nothing after d4 but its own completion

    def test_fresh_spawn_gets_whole_session_including_own_rows(self):
        rows = [
            _user("U1", target="codex"), _marker("codex", "d1"),
            _completed("codex", "X1 from a thread that is now gone"),
            _user("U2"), _assistant("O2"),
        ]
        recap = build_recap(rows, "codex", fresh=True)
        assert "user: U1" in recap
        assert "assistant: X1 from a thread that is now gone" in recap
        assert "user: U2" in recap and "assistant: O2" in recap

    def test_no_marker_yet_resumed_thread_gets_session_minus_native_rows(self):
        rows = [_user("U1"), _assistant("O1"), _user("U2", target="codex")]
        recap = build_recap(rows, "codex", fresh=False)
        assert "user: U1" in recap and "assistant: O1" in recap
        assert "U2" not in recap

    def test_interaction_required_row_is_not_a_watermark(self):
        rows = [
            _marker("codex", "d1"),
            _user("U2 during the turn"),
            _interaction_required("codex"),
            _completed("codex", "X1"),
            _user("U3", target="codex"),
        ]
        recap = build_recap(rows, "codex", fresh=False)
        assert "user: U2 during the turn" in recap
        assert "requires input" not in recap

    def test_legacy_completion_row_falls_back_to_display_summary(self):
        rows = [_completed("claude-code", "old style", legacy=True),
                _user("hi", target="codex")]
        recap = build_recap(rows, "codex", fresh=False)
        assert "assistant: old style" in recap

    def test_non_visible_rows_are_excluded(self):
        rows = [_user("U1"), _tool_row(), _assistant("O1"),
                {"role": "system", "source": "daemon",
                 "content": "[Sub-agent] codex started"},
                _assistant("")]
        recap = build_recap(rows, "codex", fresh=False)
        assert "grep output" not in recap
        assert "started" not in recap
        assert "user: U1" in recap and "assistant: O1" in recap

    def test_empty_gap_and_empty_session_give_no_block(self):
        assert build_recap([], "codex", fresh=False) == ""
        assert build_recap([_marker("codex"), _completed("codex", "X")],
                           "codex", fresh=False) == ""


class TestRecapSizeAndOrder:

    def test_no_per_message_truncation(self):
        pasted = "application form\n" + "字" * 3900
        recap = build_recap([_user(pasted), _assistant("ok " + "z" * 3000)],
                            "codex", fresh=False)
        assert pasted in recap
        assert "ok " + "z" * 3000 in recap

    def test_total_cap_evicts_oldest_first(self):
        rows = [_user(f"message number {i} " + "x" * 400) for i in range(200)]
        recap = build_recap(rows, "codex", fresh=False)
        assert len(recap) < RECAP_CAP_CHARS + 500
        assert "message number 199 " in recap
        assert "message number 0 " not in recap

    def test_chronological_order_and_frame(self):
        rows = [_user("first"), _assistant("second"), _user("third")]
        recap = build_recap(rows, "codex", fresh=False)
        assert recap.startswith("Conversation so far")
        assert recap.endswith("--- end of conversation so far ---\n\n")
        assert recap.index("user: first") < recap.index("assistant: second") \
            < recap.index("user: third")


# ---------------------------------------------------------------------------
# B. on_completed — the full reply lands on the row
# ---------------------------------------------------------------------------


class _AgentManager:
    def __init__(self):
        self.injections = []

    async def inject_system_message(self, project_id, content, **kwargs):
        self.injections.append((project_id, content, kwargs))


class _WS:
    def __init__(self):
        self.events = []

    def broadcast(self, project_id, payload):
        self.events.append(payload)


class TestCompletionRowCarriesFullReply:

    @pytest.mark.asyncio
    async def test_full_reply_on_row_short_display_short_ws(self):
        from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
        am, ws = _AgentManager(), _WS()
        obs = LifecycleObserver(am, ws)
        reply = "Q1 answer.\n" + "a" * 2000 + "\nQ8 answer."
        await obs.on_completed("proj", "claude-code", reply, "/tmp/cc.jsonl",
                               session_id=SID)
        _, content, kwargs = am.injections[0]
        meta = kwargs["meta"]
        assert meta["reply"] == reply
        assert reply in content  # the manager reads the whole reply
        assert "do NOT repeat" in content  # guidance still agent-facing
        assert len(meta["display_content"]) < 700  # timeline text stays short
        assert reply not in meta["display_content"]
        assert ws.events[-1]["summary"] == reply[:500]

    @pytest.mark.asyncio
    async def test_reply_is_capped_but_never_below_the_old_summary(self):
        from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
        am, ws = _AgentManager(), _WS()
        obs = LifecycleObserver(am, ws)
        huge = "h" * (REPLY_CAP_CHARS + 5000)
        await obs.on_completed("proj", "codex", huge, "/tmp/x.jsonl",
                               session_id=SID)
        meta = am.injections[0][2]["meta"]
        assert len(meta["reply"]) == REPLY_CAP_CHARS
        assert REPLY_CAP_CHARS > 500


# ---------------------------------------------------------------------------
# C. Dispatch — the block is built at dispatch time for every entry path
# ---------------------------------------------------------------------------


class _Session:
    def __init__(self, rows):
        self.rows = rows

    def get_messages(self):
        return list(self.rows)


def _manager(rows, observer=None):
    from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
    observer = observer or MagicMock(on_message_routed=AsyncMock())
    mgr = SubAgentManager(process_manager=MagicMock(),
                          lifecycle_observer=observer)
    mgr._dispatch_async = AsyncMock()
    session = _Session(rows)
    mgr._session_resolver = lambda pid, sid: session
    return mgr, session, observer


def _prompt(message, initiator="management_agent", dispatch_id="d9"):
    from agent_os.daemon_v2.sub_agent_manager import _QueuedPrompt
    return _QueuedPrompt(message=message, dispatch_id=dispatch_id,
                         transcript_path="/tmp/t.jsonl", initiator=initiator)


def _adapter(status="resumed"):
    return SimpleNamespace(_resume_status=(status, None))


class TestDispatchBuildsTheBlock:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("initiator",
                             ["user_pinned", "user_mention", "management_agent"])
    async def test_every_entry_path_gets_the_block_and_a_raw_preview(
            self, initiator):
        mgr, session, observer = _manager([_user("earlier"), _assistant("reply")])
        await mgr._dispatch_prompt_locked(
            _adapter(), _prompt("fix it", initiator), "proj", SID, "codex")
        sent = mgr._dispatch_async.await_args.args[3]
        assert sent.startswith("Conversation so far")
        assert "user: earlier" in sent and "assistant: reply" in sent
        assert sent.endswith("fix it")
        # The timeline marker previews the user's own words, not the block.
        preview = observer.on_message_routed.await_args.kwargs["message_preview"]
        assert preview == "fix it"

    @pytest.mark.asyncio
    async def test_block_is_computed_at_drain_time(self):
        """A queued prompt's block covers rows that landed while it waited."""
        mgr, session, _ = _manager([_marker("codex", "d1")])
        session.rows.append(_user("landed while codex was busy"))
        session.rows.append(_assistant("orbit answered meanwhile"))
        await mgr._dispatch_prompt_locked(
            _adapter(), _prompt("next"), "proj", SID, "codex")
        sent = mgr._dispatch_async.await_args.args[3]
        assert "user: landed while codex was busy" in sent
        assert "assistant: orbit answered meanwhile" in sent

    @pytest.mark.asyncio
    async def test_fresh_spawn_first_dispatch_whole_session_then_gaps(self):
        rows = [
            _user("U1", target="codex"), _marker("codex", "d1"),
            _completed("codex", "X1 from the lost thread"),
        ]
        mgr, session, _ = _manager(rows)
        adapter = _adapter("fresh")
        await mgr._dispatch_prompt_locked(
            adapter, _prompt("U2"), "proj", SID, "codex")
        first = mgr._dispatch_async.await_args.args[3]
        assert "user: U1" in first
        assert "assistant: X1 from the lost thread" in first
        # Second dispatch on the same (now primed) adapter: gap only.
        session.rows.append(_marker("codex", "d2"))
        session.rows.append(_completed("codex", "X2"))
        session.rows.append(_user("U3 for orbit"))
        await mgr._dispatch_prompt_locked(
            adapter, _prompt("U4"), "proj", SID, "codex")
        second = mgr._dispatch_async.await_args.args[3]
        assert "user: U3 for orbit" in second
        assert "U1" not in second and "X1" not in second and "X2" not in second

    @pytest.mark.asyncio
    async def test_nothing_missed_means_bare_message(self):
        mgr, _, _ = _manager([_marker("codex", "d1"), _completed("codex", "X")])
        await mgr._dispatch_prompt_locked(
            _adapter(), _prompt("again"), "proj", SID, "codex")
        assert mgr._dispatch_async.await_args.args[3] == "again"

    @pytest.mark.asyncio
    async def test_no_session_dispatches_bare_message(self):
        mgr, _, _ = _manager([])
        mgr._session_resolver = lambda pid, sid: None
        await mgr._dispatch_prompt_locked(
            _adapter(), _prompt("bare"), "proj", SID, "codex")
        assert mgr._dispatch_async.await_args.args[3] == "bare"

    @pytest.mark.asyncio
    async def test_resolver_failure_never_blocks_the_dispatch(self):
        mgr, _, _ = _manager([])

        def _boom(pid, sid):
            raise RuntimeError("lock contended")

        mgr._session_resolver = _boom
        await mgr._dispatch_prompt_locked(
            _adapter(), _prompt("still goes"), "proj", SID, "codex")
        assert mgr._dispatch_async.await_args.args[3] == "still goes"


# ---------------------------------------------------------------------------
# D. The pipe/ACP completion path hands the FULL reply to on_completed
# ---------------------------------------------------------------------------


class TestPipeCompletionPassesFullReply:

    def test_no_200_char_summary_cut_in_sub_agent_manager(self):
        import inspect
        from agent_os.daemon_v2 import sub_agent_manager
        src = inspect.getsource(sub_agent_manager)
        assert "summary=response[:200]" not in src


# ---------------------------------------------------------------------------
# E. Session reads and thread records survive an evicted manager session
# ---------------------------------------------------------------------------


class TestSessionAccessWithoutHydration:

    def test_resolve_session_for_read_falls_back_to_disk(self):
        from agent_os.daemon_v2.agent_manager import AgentManager
        disk = object()
        fake = SimpleNamespace(
            get_session=lambda pid, *, session_id=None: None,
            _load_session_from_disk=lambda pid, sid: disk,
        )
        assert AgentManager.resolve_session_for_read(fake, "proj", SID) is disk

    def test_resolve_session_for_read_prefers_live_handle(self):
        from agent_os.daemon_v2.agent_manager import AgentManager
        live = object()
        fake = SimpleNamespace(
            get_session=lambda pid, *, session_id=None: live,
            _load_session_from_disk=lambda pid, sid: pytest.fail("no disk"),
        )
        assert AgentManager.resolve_session_for_read(fake, "proj", SID) is live

    def test_record_sub_agent_thread_writes_to_disk_session_when_evicted(self):
        """While pinned the manager loop is often evicted; the thread record
        must still land (it is what makes the next resume possible)."""
        from agent_os.daemon_v2.agent_manager import AgentManager
        disk = MagicMock()
        fake = SimpleNamespace(
            get_session=lambda pid, *, session_id=None: None,
            _load_session_from_disk=lambda pid, sid: disk,
        )
        fake.resolve_session_for_read = (
            lambda pid, sid: AgentManager.resolve_session_for_read(fake, pid, sid))
        AgentManager.record_sub_agent_thread(
            fake, "proj", "claude-code", claude_session_id="c-1",
            model="m", session_id=SID, proc_pid=1, proc_create_time=2.0,
            rollout_path=None,
        )
        disk.set_sub_agent_thread.assert_called_once_with(
            "claude-code", session_id="c-1", model="m", proc_pid=1,
            proc_create_time=2.0, rollout_path=None,
        )
