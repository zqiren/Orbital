# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 126 B — a refused ``thread/resume`` forks the held thread.

The Codex desktop app holds a thread's writer lock while it has the thread
loaded, not only during a turn, so Orbital's resume of an imported thread is
refused with ``-32600 … already has an active writer``. The transport then
retries once with ``thread/fork`` (same governance params: the fork applies
its own defaults for omitted ones), adopts the fork as its thread so the
turn's resume record points at the writable copy, and only when the fork
also fails opens a fresh thread and says so through ``resume_outcome``.
No codex process is spawned — ``_request`` is faked.
"""

import logging

import pytest

from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.transports.codex_transport import (
    CodexRpcError,
    CodexTransport,
)

HELD = "01a0a825-7f33-75d0-b789-bb62c9c3e8be"
FORK = "019f0000-aaaa-7bbb-8ccc-dddddddddddd"
WRITER_ERROR = {"code": -32600,
                "message": f"thread {HELD} already has an active writer"}


def _fake_request(script: dict, calls: list):
    """``script``: method -> list of outcomes in order; an outcome is a dict
    (the result) or a CodexRpcError (raised)."""
    async def fake(method, params=None, timeout=30.0):
        calls.append((method, dict(params or {}), timeout))
        outcomes = script[method]
        outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    return fake


def _thread_result(tid, path=None, model="gpt-5.4-mini"):
    return {"thread": {"id": tid, "path": path or f"/codex/sessions/rollout-{tid}.jsonl"},
            "model": model}


def _transport(**kwargs):
    t = CodexTransport(autonomy=Autonomy.CHECK_IN,
                       resume_record={"session_id": HELD, "rollout_path": "/x"},
                       **kwargs)
    return t


class TestRpcError:
    def test_request_errors_carry_code_and_message_without_changing_the_text(self):
        err = CodexRpcError("thread/resume", WRITER_ERROR)
        assert str(err) == f"codex thread/resume failed: {WRITER_ERROR}"
        assert isinstance(err, RuntimeError)
        assert err.code == -32600
        assert err.is_active_writer_refusal()
        assert not CodexRpcError("thread/resume", {"code": -32600, "message": "bad"}
                                 ).is_active_writer_refusal()
        assert not CodexRpcError("x", "plain string").is_active_writer_refusal()


class TestForkOnHeldThread:
    @pytest.mark.asyncio
    async def test_refused_resume_forks_with_identical_governance_params(self, caplog):
        t = _transport()
        t._model = "gpt-5.4"
        calls = []
        t._request = _fake_request({
            "thread/resume": [CodexRpcError("thread/resume", WRITER_ERROR)],
            "thread/fork": [_thread_result(FORK)],
        }, calls)
        with caplog.at_level(logging.INFO):
            await t._open_thread("/ws")
        methods = [c[0] for c in calls]
        assert methods == ["thread/resume", "thread/fork"]
        resume_params, fork_params = calls[0][1], calls[1][1]
        assert resume_params == {"cwd": "/ws", "approvalPolicy": "on-request",
                                 "sandbox": "workspace-write", "model": "gpt-5.4",
                                 "threadId": HELD}
        assert fork_params == resume_params, "fork must carry the same governance"
        # The fork is the thread from now on: turn_complete persists it as
        # the resume record, so the next fire resumes the fork.
        assert t._thread_id == FORK
        assert t._rollout_path == f"/codex/sessions/rollout-{FORK}.jsonl"
        assert t._turn_meta("success")["session_id"] == FORK
        assert t.resume_outcome == ("resumed", None)
        assert any(HELD in r.message and FORK in r.message and "held" in r.message
                   for r in caplog.records if r.levelno == logging.INFO)

    @pytest.mark.asyncio
    async def test_fork_gets_a_longer_timeout_than_resume(self):
        t = _transport()
        calls = []
        t._request = _fake_request({
            "thread/resume": [CodexRpcError("thread/resume", WRITER_ERROR)],
            "thread/fork": [_thread_result(FORK)],
        }, calls)
        await t._open_thread("/ws")
        assert calls[1][2] >= 120.0

    @pytest.mark.asyncio
    async def test_fork_failure_falls_back_to_a_fresh_thread_and_says_so(self, caplog):
        t = _transport()
        calls = []
        t._request = _fake_request({
            "thread/resume": [CodexRpcError("thread/resume", WRITER_ERROR)],
            "thread/fork": [CodexRpcError("thread/fork", {"code": -32603, "message": "boom"})],
            # Spec 172 B: the user's Codex config names no model here.
            "config/read": [{"config": {"model": None}}],
            "model/list": [{"data": [{"id": "gpt-5.4-mini"}]}],
            "thread/start": [_thread_result("fresh-1")],
        }, calls)
        with caplog.at_level(logging.WARNING):
            await t._open_thread("/ws")
        methods = [c[0] for c in calls]
        assert methods == ["thread/resume", "thread/fork", "config/read",
                           "model/list", "thread/start"]
        start_params = calls[4][1]
        assert "threadId" not in start_params
        # A FRESH thread never opens on the unqualified server default.
        assert start_params["model"] == "gpt-5.4-mini"
        assert start_params["cwd"] == "/ws"
        assert t._thread_id == "fresh-1"
        assert t.resume_outcome == ("fresh", "resume_failed")
        assert any("fork" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_other_resume_errors_keep_raising(self):
        t = _transport()
        calls = []
        t._request = _fake_request({
            "thread/resume": [CodexRpcError("thread/resume",
                                            {"code": -32600, "message": "no such thread"})],
        }, calls)
        with pytest.raises(RuntimeError, match="no such thread"):
            await t._open_thread("/ws")
        assert [c[0] for c in calls] == ["thread/resume"]

    @pytest.mark.asyncio
    async def test_fresh_start_never_forks(self):
        t = CodexTransport(autonomy=Autonomy.CHECK_IN)
        t._model = "gpt-5.4"
        calls = []
        t._request = _fake_request({
            "thread/start": [CodexRpcError("thread/start", WRITER_ERROR)],
        }, calls)
        with pytest.raises(RuntimeError):
            await t._open_thread("/ws")
        assert [c[0] for c in calls] == ["thread/start"]

    @pytest.mark.asyncio
    async def test_plain_resume_reports_resumed(self):
        t = _transport()
        calls = []
        t._request = _fake_request({"thread/resume": [_thread_result(HELD)]}, calls)
        await t._open_thread("/ws")
        assert t._thread_id == HELD
        assert t.resume_outcome == ("resumed", None)
        assert [c[0] for c in calls] == ["thread/resume"]


class TestManagerHonoursTheOutcome:
    def test_provider_confirmed_outcome_reads_the_codex_transport(self):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        t = _transport()
        t._resume_outcome = ("fresh", "resume_failed")
        assert SubAgentManager._provider_confirmed_resume_outcome(
            {"session_id": HELD}, "resumed", None, t) == ("fresh", "resume_failed")
        t._resume_outcome = ("resumed", None)
        assert SubAgentManager._provider_confirmed_resume_outcome(
            {"session_id": HELD}, "resumed", None, t) == ("resumed", None)
