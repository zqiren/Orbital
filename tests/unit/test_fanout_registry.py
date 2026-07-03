# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for FanoutRegistry: join/gather semantics, partial failure
reporting, non-fanout handle passthrough, and the stall watchdog.

Spec 009 (subagent fanout), Task 2 brief.

Round-3 review (CRITICAL 1): the ``inject``/``stop_worker`` doubles below
are deliberately SUSPENDING (``await asyncio.sleep(0)``) rather than plain
synchronous-under-the-hood coroutines — production's real collaborators
(``AgentManager.inject_system_message`` via ``_start_loop``,
``SubAgentManager.stop`` via ``wait_for(shield(...))``) both genuinely
suspend, and a non-suspending double would silently absorb the
self-cancellation bug this round fixed (see ``resolve_group``'s
``asyncio.current_task()`` guard and task-2-report.md's "Round 3" section
for the full analysis + RED/GREEN evidence).
"""

import asyncio
import re
import time

import pytest

from agent_os.daemon_v2.fanout import FanoutRegistry, FanoutTask


def make_registry(events):
    async def inject(pid, content, session_id=None):
        await asyncio.sleep(0)   # genuine suspension — see module docstring
        events.append(("inject", pid, content, session_id))
    def broadcast(pid, payload):
        events.append(("ws", pid, payload))
    async def stop_worker(pid, handle, session_id=None):
        await asyncio.sleep(0)   # genuine suspension — see module docstring
        events.append(("stop", pid, handle))
    return FanoutRegistry(inject=inject, broadcast=broadcast,
                          stop_worker=stop_worker)


async def _wait_until(cond, *, timeout: float = 2.0, interval: float = 0.005):
    """Poll ``cond`` until true, rather than guessing a fixed number of
    ``asyncio.sleep(0)`` yields — the honest (suspending) stub doubles above
    need more than one event-loop tick to fully resolve a group."""
    loop = asyncio.get_event_loop()
    start = loop.time()
    while not cond():
        if loop.time() - start > timeout:
            raise AssertionError(f"condition not met within {timeout}s")
        await asyncio.sleep(interval)


@pytest.mark.asyncio
async def test_join_injects_once_when_all_complete():
    events = []
    r = make_registry(events)
    g = r.create_group("p1", "s1",
        [FanoutTask(handle="worker:f-0", label="a", brief="x"),
         FanoutTask(handle="worker:f-1", label="b", brief="y")],
        max_runtime_s=3600)
    assert r.absorb_terminal("p1", "worker:f-0", "s1", kind="completed",
                             summary="ok0", transcript_path="t0") is True
    injections = [e for e in events if e[0] == "inject"]
    assert injections == []                      # no inject yet
    assert r.absorb_terminal("p1", "worker:f-1", "s1", kind="completed",
                             summary="ok1", transcript_path="t1") is True
    await _wait_until(lambda: any(e[0] == "inject" for e in events))
    injections = [e for e in events if e[0] == "inject"]
    assert len(injections) == 1
    assert "2/2 succeeded" in injections[0][2]
    assert "t0" in injections[0][2] and "t1" in injections[0][2]
    completed = [e for e in events if e[0] == "ws"
                 and e[2]["type"] == "fanout.completed"]
    assert len(completed) == 1


@pytest.mark.asyncio
async def test_partial_failure_reported():
    events = []
    r = make_registry(events)
    g = r.create_group("p1", "s1",
        [FanoutTask(handle="worker:f-0", label="a", brief="x"),
         FanoutTask(handle="worker:f-1", label="b", brief="y")],
        max_runtime_s=3600)
    r.absorb_terminal("p1", "worker:f-0", "s1", kind="completed",
                      summary="ok", transcript_path="t0")
    r.absorb_terminal("p1", "worker:f-1", "s1", kind="error",
                      summary="ProviderError: 429", transcript_path="t1")
    await _wait_until(lambda: any(e[0] == "inject" for e in events))
    inj = [e for e in events if e[0] == "inject"][0][2]
    assert "1/2 succeeded" in inj and "429" in inj


def test_note_activity_bumps_running_task_only():
    r = make_registry([])
    g = r.create_group("p1", "s1",
        [FanoutTask(handle="worker:f-0", label="a", brief="x")],
        max_runtime_s=3600)
    task = g.tasks["worker:f-0"]
    assert task.last_activity == 0.0
    r.note_activity("p1", "worker:f-0", "s1")
    assert task.last_activity > 0.0

    # A non-fanout handle and an unknown session are both no-ops.
    r.note_activity("p1", "claude-code", "s1")
    r.note_activity("p1", "worker:f-0", "other-session")


def test_non_fanout_handle_not_absorbed():
    r = make_registry([])
    assert r.absorb_terminal("p1", "claude-code", "s1", kind="completed",
                             summary="ok", transcript_path="t") is False


@pytest.mark.asyncio
async def test_join_summary_matches_frozen_format():
    """Pins the FROZEN line-oriented join-summary format (team-lead spec,
    2026-07-04) the frontend parses — not just substrings. Header line,
    then one task line per task in ORIGINAL task order (not sorted by
    handle), each ``- [<status>] <label> (<handle>): <text> | transcript:
    <path>``, status in {completed, error, stalled, interrupted}."""
    events = []
    r = make_registry(events)
    g = r.create_group("p1", "s1", [
        FanoutTask(handle="worker:a1b2c3d4-0", label="research auth options",
                   brief="x"),
        FanoutTask(handle="worker:a1b2c3d4-1", label="scan dependencies",
                   brief="y"),
        FanoutTask(handle="worker:a1b2c3d4-2", label="draft migration plan",
                   brief="z"),
    ], max_runtime_s=3600)

    r.absorb_terminal("p1", "worker:a1b2c3d4-0", "s1", kind="completed",
                      summary="OAuth2 with PKCE recommended because...",
                      transcript_path="/path/t0.jsonl")
    r.absorb_terminal("p1", "worker:a1b2c3d4-1", "s1", kind="error",
                      summary="Error: provider 429 rate limit",
                      transcript_path="/path/t1.jsonl")
    r.absorb_terminal("p1", "worker:a1b2c3d4-2", "s1", kind="completed",
                      summary="Plan drafted in docs/plan.md",
                      transcript_path="/path/t2.jsonl")
    await _wait_until(lambda: any(e[0] == "inject" for e in events))

    content = [e for e in events if e[0] == "inject"][0][2]
    lines = content.split("\n")

    assert re.match(
        r"^\[Fanout [0-9a-f]{8}\] 2/3 succeeded\.$", lines[0],
    ), lines[0]

    task_line_re = re.compile(
        r"^- \[(completed|error|stalled|interrupted)\] (.+) \((worker:[^)]+)\): "
        r"(.*) \| transcript: (.+)$"
    )
    task_lines = lines[1:4]
    m0 = task_line_re.match(task_lines[0])
    assert m0 is not None, task_lines[0]
    assert m0.group(1) == "completed"
    assert m0.group(2) == "research auth options"
    assert m0.group(3) == "worker:a1b2c3d4-0"
    assert "OAuth2 with PKCE" in m0.group(4)
    assert m0.group(5) == "/path/t0.jsonl"

    m1 = task_line_re.match(task_lines[1])
    assert m1 is not None, task_lines[1]
    assert m1.group(1) == "error"
    assert m1.group(2) == "scan dependencies"
    assert "429" in m1.group(4)

    m2 = task_line_re.match(task_lines[2])
    assert m2 is not None, task_lines[2]
    assert m2.group(1) == "completed"
    assert m2.group(2) == "draft migration plan"

    # Original task order preserved (0, 1, 2), not re-sorted by handle/status.
    assert [m0.group(3), m1.group(3), m2.group(3)] == [
        "worker:a1b2c3d4-0", "worker:a1b2c3d4-1", "worker:a1b2c3d4-2",
    ]


def test_clean_summary_strips_newlines_and_truncates():
    from agent_os.daemon_v2.fanout import _clean_summary
    assert _clean_summary("line1\nline2\r\nline3") == "line1 line2  line3"
    long_text = "x" * 500
    assert len(_clean_summary(long_text)) == 200


@pytest.mark.asyncio
async def test_resolve_group_guards_inject_failure():
    """Teardown requirement (Task 2 brief): resolve_group must not raise into
    the caller when the session is dead / inject fails — it logs and still
    completes the broadcast, so a stray exception here can never orphan the
    watchdog's caller or leave the group unresolved."""
    events = []

    async def raising_inject(pid, content, session_id=None):
        await asyncio.sleep(0)
        raise RuntimeError("session is gone")
    def broadcast(pid, payload):
        events.append(("ws", pid, payload))
    async def stop_worker(pid, handle, session_id=None):
        await asyncio.sleep(0)
        events.append(("stop", pid, handle))

    r = FanoutRegistry(inject=raising_inject, broadcast=broadcast,
                       stop_worker=stop_worker)
    g = r.create_group("p1", "s1",
        [FanoutTask(handle="worker:f-0", label="a", brief="x")],
        max_runtime_s=3600)
    await r.resolve_group(g, reason="test")  # must not raise
    assert g.resolved is True
    completed = [e for e in events if e[0] == "ws"
                 and e[2]["type"] == "fanout.completed"]
    assert len(completed) == 1


# ---------------------------------------------------------------------------
# Watchdog-internal resolve_group call sites (CRITICAL 1 — self-cancellation)
#
# All three of these call `resolve_group` from INSIDE `_watchdog_loop`, i.e.
# from the very task `group._watchdog_task` refers to. `resolve_group`
# unconditionally tries to cancel `group._watchdog_task` — cancelling the
# CURRENTLY RUNNING task is a self-cancel: the CancelledError doesn't fire
# immediately, it fires at the next genuine suspension point (a `BaseException`,
# so it blows past every `except Exception` guard), aborting resolve_group
# mid-flight and leaking the group. The `inject`/`stop_worker` doubles must
# genuinely suspend (see module docstring) for these tests to actually
# exercise that failure mode instead of silently absorbing it.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_watchdog_stalls_silent_worker():
    events = []
    r = make_registry(events)
    g = r.create_group("p1", "s1",
        [FanoutTask(handle="worker:f-0", label="a", brief="x")],
        max_runtime_s=3600)
    g.stall_after_s = 0.01                       # test override
    await r.start_watchdog(g)
    await _wait_until(lambda: g.resolved)
    assert [e for e in events if e[0] == "stop"]  # straggler stopped
    inj = [e for e in events if e[0] == "inject"]
    assert len(inj) == 1 and "stalled" in inj[0][2]
    completed = [e for e in events if e[0] == "ws"
                 and e[2]["type"] == "fanout.completed"]
    assert len(completed) == 1
    # No leak: a resolved group's routing entries must be freed.
    assert r._groups == {}
    assert r._by_handle == {}


@pytest.mark.asyncio
async def test_watchdog_max_runtime_ceiling_resolves_group():
    """The hard max_runtime_s ceiling fires (and must resolve the group)
    even when the task is NOT stalled by activity — the SECOND distinct
    watchdog-internal resolve_group call site."""
    events = []
    r = make_registry(events)
    g = r.create_group("p1", "s1",
        [FanoutTask(handle="worker:f-0", label="a", brief="x",
                    last_activity=time.monotonic())],
        max_runtime_s=3600)
    g.stall_after_s = 1.0     # keeps the poll interval sane (~50ms) so the
                              # stall check (fresh last_activity) never fires
    g.max_runtime_s = 0.01    # ceiling fires on the very first poll
    await r.start_watchdog(g)
    await _wait_until(lambda: g.resolved)
    assert [e for e in events if e[0] == "stop"]
    inj = [e for e in events if e[0] == "inject"]
    assert len(inj) == 1
    completed = [e for e in events if e[0] == "ws"
                 and e[2]["type"] == "fanout.completed"]
    assert len(completed) == 1
    assert r._groups == {}
    assert r._by_handle == {}


@pytest.mark.asyncio
async def test_watchdog_error_still_resolves_group():
    """A crash INSIDE the watchdog's own poll loop (not a guarded
    stop_worker/inject call — those are individually try/excepted already)
    must still resolve the group (R4: never orphan). Synthesized via a
    deliberately malformed task (``last_activity=None``) that TypeErrors
    during the stall-time comparison — landing in the loop's own
    ``except Exception`` handler, the THIRD watchdog-internal resolve_group
    call site."""
    events = []
    r = make_registry(events)
    g = r.create_group("p1", "s1",
        [FanoutTask(handle="worker:f-0", label="a", brief="x")],
        max_runtime_s=3600)
    g.tasks["worker:f-0"].last_activity = None  # forces TypeError in the loop
    g.stall_after_s = 0.01
    await r.start_watchdog(g)
    await _wait_until(lambda: g.resolved)
    inj = [e for e in events if e[0] == "inject"]
    assert len(inj) == 1
    completed = [e for e in events if e[0] == "ws"
                 and e[2]["type"] == "fanout.completed"]
    assert len(completed) == 1
    assert r._groups == {}
    assert r._by_handle == {}


@pytest.mark.asyncio
async def test_stop_worker_called_keyword_only_session_id():
    """Regression: the real collaborator this is wired to in production,
    ``SubAgentManager.stop``, declares ``session_id`` KEYWORD-ONLY
    (``async def stop(self, project_id, handle, *, session_id=None)``). A
    stub that only accepts it positionally would mask a call-site bug that
    TypeErrors the moment ``stop_worker=sub_agent_manager.stop`` is wired for
    real — this stub raises if session_id ever arrives positionally."""
    events = []

    async def strict_stop_worker(project_id, handle, *, session_id=None):
        await asyncio.sleep(0)
        events.append(("stop", project_id, handle, session_id))

    async def inject(pid, content, session_id=None):
        await asyncio.sleep(0)
        events.append(("inject", pid, content, session_id))
    def broadcast(pid, payload):
        events.append(("ws", pid, payload))

    r = FanoutRegistry(inject=inject, broadcast=broadcast,
                       stop_worker=strict_stop_worker)
    g = r.create_group("p1", "s1",
        [FanoutTask(handle="worker:f-0", label="a", brief="x")],
        max_runtime_s=3600)
    g.stall_after_s = 0.01
    await r.start_watchdog(g)
    await _wait_until(lambda: g.resolved)
    stops = [e for e in events if e[0] == "stop"]
    assert len(stops) == 1
    assert stops[0][3] == "s1"  # session_id landed correctly via keyword
