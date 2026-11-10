"""Batch 1110 item 13: the native thread id is recorded the moment the worker
has one (Claude's init frame, Codex's thread/start), not only when a turn
completes, so a daemon restart during the FIRST turn can still resume."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.transports.base import AgentTransport
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager


class _T(AgentTransport):
    """The smallest concrete transport: only the sink plumbing is exercised."""

    async def start(self, *a, **k):  # pragma: no cover - never called
        pass

    async def stop(self, *a, **k):  # pragma: no cover
        pass

    async def dispatch(self, *a, **k):  # pragma: no cover
        pass

    async def send(self, *a, **k):  # pragma: no cover
        pass

    async def read_stream(self, *a, **k):  # pragma: no cover
        pass

    def is_alive(self):  # pragma: no cover
        return True


def _sam(observer):
    sam = SubAgentManager(process_manager=MagicMock(), registry=MagicMock(),
                          setup_engine=MagicMock(), lifecycle_observer=observer)
    return sam


def test_publish_without_a_sink_or_without_an_id_is_a_noop():
    t = _T()
    t._publish_thread("abc")  # no sink: nothing happens
    seen = []
    t.set_thread_sink(lambda nid, rollout_path=None: seen.append((nid, rollout_path)))
    t._publish_thread(None)
    t._publish_thread("")
    assert seen == []
    t._publish_thread("thr-1", rollout_path="/r.jsonl")
    assert seen == [("thr-1", "/r.jsonl")]


def test_a_raising_sink_never_reaches_the_transport():
    t = _T()

    def boom(nid, rollout_path=None):
        raise RuntimeError("x")
    t.set_thread_sink(boom)
    t._publish_thread("thr-1")  # swallowed, logged


@pytest.mark.asyncio
async def test_the_manager_records_the_early_id_through_on_thread_update():
    observer = MagicMock()
    observer.on_thread_update = AsyncMock()
    sam = _sam(observer)
    t = _T()
    sam._attach_thread_sink(t, "proj", "sess-1", "codex")

    t._publish_thread("01a1c191-thread", rollout_path="C:/r/rollout.jsonl")
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    observer.on_thread_update.assert_awaited_once_with(
        "proj", "codex", claude_session_id="01a1c191-thread", session_id="sess-1",
        rollout_path="C:/r/rollout.jsonl")
    assert sam._thread_sink_tasks == set()


@pytest.mark.asyncio
async def test_no_observer_means_no_task():
    sam = _sam(None)
    t = _T()
    sam._attach_thread_sink(t, "proj", None, "claude-code")
    t._publish_thread("sid-1")
    await asyncio.sleep(0)
    assert sam._thread_sink_tasks == set()


def test_transports_without_the_hook_are_left_alone():
    sam = _sam(MagicMock())
    plain = object()
    sam._attach_thread_sink(plain, "proj", None, "pipe")  # no AttributeError
