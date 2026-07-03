# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for FanoutRegistry: join/gather semantics, partial failure
reporting, non-fanout handle passthrough, and the stall watchdog.

Spec 009 (subagent fanout), Task 2 brief.
"""

import asyncio
import re

import pytest

from agent_os.daemon_v2.fanout import FanoutRegistry, FanoutTask


def make_registry(events):
    async def inject(pid, content, session_id=None):
        events.append(("inject", pid, content, session_id))
    def broadcast(pid, payload):
        events.append(("ws", pid, payload))
    async def stop_worker(pid, handle, session_id=None):
        events.append(("stop", pid, handle))
    return FanoutRegistry(inject=inject, broadcast=broadcast,
                          stop_worker=stop_worker)


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
    await asyncio.sleep(0)                       # let resolve task run
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
    await asyncio.sleep(0)
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
    await asyncio.sleep(0)

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
        raise RuntimeError("session is gone")
    def broadcast(pid, payload):
        events.append(("ws", pid, payload))
    async def stop_worker(pid, handle, session_id=None):
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


@pytest.mark.asyncio
async def test_watchdog_stalls_silent_worker():
    events = []
    r = make_registry(events)
    g = r.create_group("p1", "s1",
        [FanoutTask(handle="worker:f-0", label="a", brief="x")],
        max_runtime_s=3600)
    g.stall_after_s = 0.01                       # test override
    await r.start_watchdog(g)
    await asyncio.sleep(0.05)
    assert [e for e in events if e[0] == "stop"]  # straggler stopped
    inj = [e for e in events if e[0] == "inject"]
    assert len(inj) == 1 and "stalled" in inj[0][2]


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
        events.append(("stop", project_id, handle, session_id))

    async def inject(pid, content, session_id=None):
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
    await asyncio.sleep(0.05)
    stops = [e for e in events if e[0] == "stop"]
    assert len(stops) == 1
    assert stops[0][3] == "s1"  # session_id landed correctly via keyword
