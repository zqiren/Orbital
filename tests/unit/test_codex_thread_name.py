# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Orbital's Codex threads get a real name (2026-11-25).

A worker's thread lives in the user's ``~/.codex``, so the ChatGPT desktop
app lists it. Without a name the app titles it from the first user message,
which starts with Orbital's launcher paragraph ("Long-running commands: never
wa…") for every thread. After the thread's first turn starts, the transport
sends ``thread/name/set`` with the name the manager offered (the Orbital
session's title, else the first line of the user's own text), once, in the
background. Only a thread Codex hands back without a name is named (a fresh
start, a fork, a resumed thread nobody named), so a name the user set in the
app is never overwritten. One-off ``ephemeral`` threads are never named.
No codex process is spawned: ``_request`` is faked.
"""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.transports.codex_transport import (
    CodexRpcError,
    CodexTransport,
)
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

HELD = "01a0a825-7f33-75d0-b789-bb62c9c3e8be"
FORK = "019f0000-aaaa-7bbb-8ccc-dddddddddddd"
FRESH = "019f1111-aaaa-7bbb-8ccc-dddddddddddd"
WRITER_ERROR = {"code": -32600,
                "message": f"thread {HELD} already has an active writer"}


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch):
    """The retries after a refused name wait seconds in production."""
    from agent_os.agent.transports import codex_transport
    monkeypatch.setattr(codex_transport, "_THREAD_NAME_RETRY_S", (0.0, 0.0, 0.0),
                        raising=False)


EMPTY_ROLLOUT = {"code": -32603, "message": (
    "failed to set thread name: Fatal error: failed to update thread metadata "
    "019f1111-aaaa-7bbb-8ccc-dddddddddddd: thread-store internal error: failed to read "
    "canonical session metadata: rollout at C:/Users/u/.codex/sessions/x.jsonl is empty")}


def _fake_request(script: dict, calls: list):
    """``script``: method -> list of outcomes in order; an outcome is a dict
    (the result), an Exception (raised) or an asyncio.Event (wait on it)."""
    async def fake(method, params=None, timeout=30.0):
        calls.append((method, dict(params or {}), timeout))
        outcomes = script[method]
        outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if isinstance(outcome, asyncio.Event):
            await outcome.wait()
            return {}
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    return fake


def _thread(tid, name=None):
    thread = {"id": tid, "path": f"/codex/sessions/rollout-{tid}.jsonl"}
    if name is not None:
        thread["name"] = name
    return {"thread": thread, "model": "gpt-6-sol"}


async def _opened(script, *, resume=False, **kwargs):
    record = {"session_id": HELD, "rollout_path": "/x"} if resume else None
    t = CodexTransport(autonomy=Autonomy.HANDS_OFF, resume_record=record, **kwargs)
    t._model = "gpt-6-sol"
    calls: list = []
    script.setdefault("turn/start", [{"turn": {"id": "turn-1"}}])
    script.setdefault("thread/name/set", [{}])
    t._request = _fake_request(script, calls)
    await t._open_thread("/ws")
    t._alive = True
    return t, calls


async def _settle(t):
    task = getattr(t, "_name_task", None)
    if task is not None:
        await asyncio.wait_for(task, 2.0)


def _names(calls):
    return [c[1] for c in calls if c[0] == "thread/name/set"]


class TestWhichThreadsAreNamed:
    @pytest.mark.asyncio
    async def test_a_fresh_thread_is_named_after_its_first_turn_starts(self):
        t, calls = await _opened({"thread/start": [_thread(FRESH)]})
        assert t.wants_thread_name is True
        t.name_thread("Fix the login bug")
        await t.dispatch("Long-running commands: …\n\nFix the login bug")
        await _settle(t)
        methods = [c[0] for c in calls]
        assert methods.index("turn/start") < methods.index("thread/name/set")
        assert _names(calls) == [{"threadId": FRESH, "name": "Fix the login bug"}]
        assert t.wants_thread_name is False

    @pytest.mark.asyncio
    async def test_the_name_goes_out_once(self):
        t, calls = await _opened({"thread/start": [_thread(FRESH)]})
        t.name_thread("Fix the login bug")
        await t.dispatch("one")
        await _settle(t)
        t.name_thread("Something else")
        await t.dispatch("two")
        await _settle(t)
        assert _names(calls) == [{"threadId": FRESH, "name": "Fix the login bug"}]

    @pytest.mark.asyncio
    async def test_a_fork_names_the_new_thread(self):
        t, calls = await _opened({
            "thread/resume": [CodexRpcError("thread/resume", WRITER_ERROR)],
            "thread/fork": [_thread(FORK)],
        }, resume=True)
        assert t.wants_thread_name is True
        t.name_thread("Weekly report")
        await t.dispatch("go")
        await _settle(t)
        assert _names(calls) == [{"threadId": FORK, "name": "Weekly report"}]

    @pytest.mark.asyncio
    async def test_a_resumed_thread_that_has_a_name_keeps_it(self):
        t, calls = await _opened({"thread/resume": [_thread(HELD, name="My own title")]},
                                 resume=True)
        assert t.wants_thread_name is False
        t.name_thread("Orbital's title")
        await t.dispatch("go")
        await _settle(t)
        assert _names(calls) == []

    @pytest.mark.asyncio
    async def test_a_fork_that_carries_a_name_keeps_it(self):
        t, calls = await _opened({
            "thread/resume": [CodexRpcError("thread/resume", WRITER_ERROR)],
            "thread/fork": [_thread(FORK, name="Named in the app")],
        }, resume=True)
        assert t.wants_thread_name is False

    @pytest.mark.asyncio
    async def test_a_resumed_thread_nobody_named_is_named(self):
        # Threads Orbital opened before this change carry no name; their
        # next resume gives them one.
        t, calls = await _opened({"thread/resume": [_thread(HELD)]}, resume=True)
        assert t.wants_thread_name is True
        t.name_thread("Old chat")
        await t.dispatch("go")
        await _settle(t)
        assert _names(calls) == [{"threadId": HELD, "name": "Old chat"}]

    @pytest.mark.asyncio
    async def test_an_ephemeral_thread_is_never_named(self):
        t, calls = await _opened({"thread/start": [_thread(FRESH)]}, ephemeral=True)
        assert t.wants_thread_name is False
        t.name_thread("route this")
        await t.dispatch("route this")
        await _settle(t)
        assert _names(calls) == []

    @pytest.mark.asyncio
    async def test_no_name_offered_sends_nothing(self):
        t, calls = await _opened({"thread/start": [_thread(FRESH)]})
        for blank in ("", "   ", None):
            t.name_thread(blank)
        await t.dispatch("go")
        await _settle(t)
        assert _names(calls) == []

    @pytest.mark.asyncio
    async def test_whitespace_is_collapsed_and_the_name_is_capped(self):
        t, calls = await _opened({"thread/start": [_thread(FRESH)]})
        t.name_thread("  Fix\n the   bug  " + "x" * 300)
        await t.dispatch("go")
        await _settle(t)
        (sent,) = _names(calls)
        assert sent["name"].startswith("Fix the bug x")
        assert "\n" not in sent["name"] and len(sent["name"]) <= 100


class TestTheNameIsRetried:
    """Codex 0.171 on Windows refuses ``thread/name/set`` in the first moments
    of a thread's first turn: its rollout file is still empty (seen
    2026-11-26, 50 ms after ``turn/start``). The name is tried again."""

    @pytest.mark.asyncio
    async def test_a_refusal_while_the_rollout_is_empty_is_retried(self, caplog):
        t, calls = await _opened({
            "thread/start": [_thread(FRESH)],
            "thread/name/set": [CodexRpcError("thread/name/set", EMPTY_ROLLOUT),
                                CodexRpcError("thread/name/set", EMPTY_ROLLOUT), {}],
        })
        t.name_thread("Fix the login bug")
        with caplog.at_level(logging.INFO):
            await t.dispatch("go")
            await _settle(t)
        assert _names(calls) == [{"threadId": FRESH, "name": "Fix the login bug"}] * 3
        assert any("named thread" in r.getMessage() for r in caplog.records)
        assert t._event_queue.empty()

    @pytest.mark.asyncio
    async def test_the_retries_are_bounded(self):
        t, calls = await _opened({
            "thread/start": [_thread(FRESH)],
            "thread/name/set": [CodexRpcError("thread/name/set", EMPTY_ROLLOUT)],
        })
        t.name_thread("A title")
        await t.dispatch("go")
        await _settle(t)
        assert len(_names(calls)) == 4  # the first try and three more
        assert t._event_queue.empty()

    @pytest.mark.asyncio
    async def test_an_older_codex_without_the_method_is_not_retried(self):
        t, calls = await _opened({
            "thread/start": [_thread(FRESH)],
            "thread/name/set": [CodexRpcError("thread/name/set",
                                              {"code": -32601, "message": "method not found"})],
        })
        t.name_thread("A title")
        await t.dispatch("go")
        await _settle(t)
        assert len(_names(calls)) == 1

    @pytest.mark.asyncio
    async def test_a_timeout_is_not_retried(self):
        t, calls = await _opened({
            "thread/start": [_thread(FRESH)],
            "thread/name/set": [asyncio.TimeoutError()],
        })
        t.name_thread("A title")
        await t.dispatch("go")
        await _settle(t)
        assert len(_names(calls)) == 1


class TestNamingIsHarmless:
    @pytest.mark.asyncio
    async def test_a_naming_error_does_not_fail_the_turn(self, caplog):
        t, calls = await _opened({
            "thread/start": [_thread(FRESH)],
            "thread/name/set": [CodexRpcError("thread/name/set",
                                              {"code": -32603, "message": "boom"})],
        })
        t.name_thread("A title")
        with caplog.at_level(logging.DEBUG):
            await t.dispatch("go")
            await _settle(t)
        assert t._turn_open is True and t._turn_id == "turn-1"
        assert t._event_queue.empty()  # no error event, no turn_complete
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    @pytest.mark.asyncio
    async def test_method_not_found_is_swallowed(self, caplog):
        t, calls = await _opened({
            "thread/start": [_thread(FRESH)],
            "thread/name/set": [CodexRpcError("thread/name/set",
                                              {"code": -32601, "message": "method not found"})],
        })
        t.name_thread("A title")
        with caplog.at_level(logging.DEBUG):
            await t.dispatch("go")
            await _settle(t)
        assert t._event_queue.empty()
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    @pytest.mark.asyncio
    async def test_a_timeout_does_not_fail_the_turn(self):
        t, calls = await _opened({
            "thread/start": [_thread(FRESH)],
            "thread/name/set": [asyncio.TimeoutError()],
        })
        t.name_thread("A title")
        await t.dispatch("go")
        await _settle(t)
        assert t._event_queue.empty()
        (call,) = [c for c in calls if c[0] == "thread/name/set"]
        assert call[2] <= 15.0  # a short bound of its own

    @pytest.mark.asyncio
    async def test_a_hanging_name_call_never_delays_the_turn(self):
        never = asyncio.Event()
        t, calls = await _opened({
            "thread/start": [_thread(FRESH)],
            "thread/name/set": [never],
        })
        t.name_thread("A title")
        await asyncio.wait_for(t.dispatch("go"), 1.0)
        assert t._turn_id == "turn-1"
        t._name_task.cancel()

    @pytest.mark.asyncio
    async def test_a_failed_turn_start_keeps_the_name_for_the_next_turn(self):
        t, calls = await _opened({
            "thread/start": [_thread(FRESH)],
            "turn/start": [RuntimeError("down"), {"turn": {"id": "turn-2"}}],
        })
        t.name_thread("A title")
        with pytest.raises(RuntimeError):
            await t.dispatch("go")
        assert _names(calls) == []
        await t.dispatch("go again")
        await _settle(t)
        assert _names(calls) == [{"threadId": FRESH, "name": "A title"}]


# ── The manager offers the name ─────────────────────────────────────────

class _Transport:
    def __init__(self, wants=True):
        self.wants_thread_name = wants
        self.offered: list = []

    def name_thread(self, name):
        self.offered.append(name)


def _manager(session=None):
    mgr = SubAgentManager.__new__(SubAgentManager)
    mgr._lifecycle_observer = None
    mgr._project_store = None
    mgr._process_manager = MagicMock()
    mgr._build_recap = AsyncMock(return_value="RECAP\n")
    mgr._dispatch_async = AsyncMock()
    mgr._session_resolver = lambda pid, sid: session
    return mgr


def _prompt(message, initiator="user_pinned"):
    return SimpleNamespace(message=message, dispatch_id="d1", transcript_path="t",
                           initiator=initiator, daemon_turn=False)


class TestManagerOffersTheName:
    @pytest.mark.asyncio
    async def test_the_orbital_session_title_wins(self):
        mgr = _manager(SimpleNamespace(name="Plan the launch"))
        transport = _Transport()
        adapter = SimpleNamespace(_transport=transport,
                                  _orbital_first_turn_note="Long-running commands: …")
        await mgr._dispatch_prompt_locked(adapter, _prompt("write the post"), "p", "s", "codex")
        assert transport.offered == ["Plan the launch"]
        # The composed prompt is untouched: preamble first, then the text.
        payload = mgr._dispatch_async.await_args.args[3]
        assert payload == "Long-running commands: …\n\nRECAP\nwrite the post"

    @pytest.mark.asyncio
    async def test_without_a_session_title_the_first_line_of_the_users_text(self):
        mgr = _manager(SimpleNamespace(name=None))
        transport = _Transport()
        adapter = SimpleNamespace(_transport=transport)
        await mgr._dispatch_prompt_locked(
            adapter, _prompt("Fix the flaky upload test\nIt fails on CI only."),
            "p", "s", "codex")
        assert transport.offered == ["Fix the flaky upload test"]

    @pytest.mark.asyncio
    async def test_a_long_first_line_is_trimmed(self):
        mgr = _manager(None)
        transport = _Transport()
        adapter = SimpleNamespace(_transport=transport)
        await mgr._dispatch_prompt_locked(adapter, _prompt("word " * 40), "p", "s", "codex")
        (name,) = transport.offered
        assert name.endswith("…") and len(name) <= 51

    @pytest.mark.asyncio
    async def test_a_resolver_failure_falls_back_to_the_text(self):
        mgr = _manager(None)

        def boom(pid, sid):
            raise OSError("disk")
        mgr._session_resolver = boom
        transport = _Transport()
        adapter = SimpleNamespace(_transport=transport)
        await mgr._dispatch_prompt_locked(adapter, _prompt("Tidy the README"), "p", "s", "codex")
        assert transport.offered == ["Tidy the README"]
        mgr._dispatch_async.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_nothing_is_offered_when_the_transport_wants_no_name(self):
        mgr = _manager(SimpleNamespace(name="Plan the launch"))
        resolver = MagicMock(return_value=SimpleNamespace(name="x"))
        mgr._session_resolver = resolver
        transport = _Transport(wants=False)
        adapter = SimpleNamespace(_transport=transport)
        await mgr._dispatch_prompt_locked(adapter, _prompt("go"), "p", "s", "codex")
        assert transport.offered == []
        resolver.assert_not_called()  # no session read for a named thread

    @pytest.mark.asyncio
    async def test_other_transports_are_left_alone(self):
        mgr = _manager(SimpleNamespace(name="Plan the launch"))
        for transport in (MagicMock(), None, object()):
            adapter = SimpleNamespace(_transport=transport)
            await mgr._dispatch_prompt_locked(adapter, _prompt("go"), "p", "s", "x")
        assert mgr._dispatch_async.await_count == 3
