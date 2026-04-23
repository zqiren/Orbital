# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for TASK-fix-background-send-exception-handling.

Covers two coupled defects in SubAgentManager._dispatch_async:

1. The background task returned by asyncio.create_task() was discarded
   with no strong reference — making it vulnerable to premature GC.
2. Exceptions inside _background_send were swallowed at WARNING level
   with no state reconciliation, leaving broken adapters silently reusable.

Tests use pure asyncio mocks — portable across macOS and Windows.
"""

import asyncio
import gc
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.adapters.cli_adapter import AdapterBrokenError, CLIAdapter
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
from agent_os.daemon_v2.sub_agent_transcript import SubAgentTranscript


def _make_manager_with_adapter(tmp_path, adapter, handle: str = "test-agent",
                                project_id: str = "proj1",
                                lifecycle_observer=None):
    """Build a SubAgentManager with a single registered adapter + transcript."""
    pm = MagicMock()
    pm._ws = MagicMock()
    pm._ws.broadcast = MagicMock()
    sam = SubAgentManager(pm, lifecycle_observer=lifecycle_observer)
    sam._adapters[project_id] = {handle: adapter}
    transcript = SubAgentTranscript(str(tmp_path), handle, "t001")
    sam._transcripts[(project_id, handle)] = transcript
    return sam


class TestBackgroundSendStrongRef:
    @pytest.mark.asyncio
    async def test_background_send_task_strongly_referenced(self, tmp_path):
        """The bg task must be held on adapter._background_send_task so GC
        cannot collect it before it completes.

        Must fail on pre-fix code where `asyncio.create_task(...)` return
        value was discarded.
        """
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_send(msg):
            started.set()
            await release.wait()

        adapter = MagicMock(spec=CLIAdapter)
        adapter.handle = "test-agent"
        adapter._transport = None
        adapter._broken = False
        adapter._background_send_task = None
        adapter.send = slow_send
        adapter._last_response = None

        sam = _make_manager_with_adapter(tmp_path, adapter)

        await sam.send("proj1", "test-agent", "hello")

        # Wait until the coroutine actually started, guaranteeing the task
        # exists and holds a reference to the coroutine.
        await asyncio.wait_for(started.wait(), timeout=2.0)

        assert adapter._background_send_task is not None, \
            "bg task must be stored on adapter as strong reference"
        task = adapter._background_send_task
        assert isinstance(task, asyncio.Task)
        assert not task.done()

        # Aggressive GC: if the code stored only a weak reference, the loop
        # could reap the task here.
        gc.collect()
        gc.collect()
        gc.collect()

        assert not task.done(), "task was collected by GC — strong ref missing"

        # Let the task finish cleanly so the test does not leak.
        release.set()
        await asyncio.wait_for(task, timeout=2.0)


class TestBackgroundSendExceptionHandling:
    @pytest.mark.asyncio
    async def test_background_send_exception_surfaces_at_error_level(
        self, tmp_path, caplog
    ):
        """When adapter.send() raises, the background task must:
           - log at ERROR with exc_info,
           - mark adapter._broken = True,
           - call lifecycle_observer.on_failed with reason="background_send_exception".

        Must fail on pre-fix code which logged at WARNING and did no
        state reconciliation.
        """
        async def failing_send(msg):
            raise RuntimeError("synthetic transport failure")

        adapter = MagicMock(spec=CLIAdapter)
        adapter.handle = "broken-agent"
        adapter._transport = None
        adapter._broken = False
        adapter._background_send_task = None
        adapter.send = failing_send
        adapter._last_response = None

        observer = MagicMock()
        observer.on_failed = MagicMock()  # on_failed is sync (see LifecycleObserver)
        observer.on_message_routed = AsyncMock()
        observer.on_completed = AsyncMock()

        sam = _make_manager_with_adapter(
            tmp_path, adapter,
            handle="broken-agent",
            lifecycle_observer=observer,
        )

        with caplog.at_level(logging.ERROR, logger="agent_os.daemon_v2.sub_agent_manager"):
            await sam.send("proj1", "broken-agent", "hello")
            # Wait for the bg task to complete.
            if adapter._background_send_task is not None:
                try:
                    await asyncio.wait_for(
                        adapter._background_send_task, timeout=2.0
                    )
                except Exception:
                    pass

        # Small yield so done_callback finishes clearing the ref.
        await asyncio.sleep(0)

        # ERROR record with exc_info containing the synthetic message.
        matching = [
            r for r in caplog.records
            if r.levelno == logging.ERROR
            and "_background_send failed" in r.getMessage()
        ]
        assert matching, (
            "expected ERROR log 'Background_send failed ...' — "
            f"got records: {[r.getMessage() for r in caplog.records]}"
        )
        assert matching[0].exc_info is not None, "exc_info missing from log record"
        # Format the exception so we can substring-match reliably.
        import traceback
        exc_text = "".join(traceback.format_exception(*matching[0].exc_info))
        assert "synthetic transport failure" in exc_text

        assert adapter._broken is True

        observer.on_failed.assert_called_once()
        _, kwargs = observer.on_failed.call_args
        # Accept either positional or kwarg style.
        if "reason" in kwargs:
            assert kwargs["reason"] == "background_send_exception"
        else:
            args = observer.on_failed.call_args.args
            assert "background_send_exception" in args

    @pytest.mark.asyncio
    async def test_cancelled_background_send_does_not_mark_broken(
        self, tmp_path, caplog
    ):
        """Cancelling the bg task must NOT flip the adapter to broken
        and must NOT log at ERROR — cancellation is a normal shutdown path.
        """
        started = asyncio.Event()

        async def slow_send(msg):
            started.set()
            # Wait much longer than the test so cancellation definitely happens.
            await asyncio.sleep(60)

        adapter = MagicMock(spec=CLIAdapter)
        adapter.handle = "cancelled-agent"
        adapter._transport = None
        adapter._broken = False
        adapter._background_send_task = None
        adapter.send = slow_send
        adapter._last_response = None

        observer = MagicMock()
        observer.on_failed = MagicMock()
        observer.on_message_routed = AsyncMock()
        observer.on_completed = AsyncMock()

        sam = _make_manager_with_adapter(
            tmp_path, adapter,
            handle="cancelled-agent",
            lifecycle_observer=observer,
        )

        with caplog.at_level(logging.ERROR, logger="agent_os.daemon_v2.sub_agent_manager"):
            await sam.send("proj1", "cancelled-agent", "hello")
            await asyncio.wait_for(started.wait(), timeout=2.0)

            task = adapter._background_send_task
            assert task is not None
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            # Yield to let the done_callback run.
            await asyncio.sleep(0)

        assert adapter._broken is False, \
            "cancellation must NOT mark adapter as broken"
        observer.on_failed.assert_not_called()
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR
                         and "_background_send failed" in r.getMessage()]
        assert not error_records, \
            f"unexpected ERROR logs on cancel: {[r.getMessage() for r in error_records]}"


class TestBrokenAdapterBehaviour:
    @pytest.mark.asyncio
    async def test_broken_adapter_rejects_next_send(self):
        """CLIAdapter.send() must raise AdapterBrokenError when _broken=True.

        Fail-loud semantics: no silent reuse of a poisoned adapter.
        """
        adapter = CLIAdapter(handle="broken", display_name="broken")
        adapter._broken = True

        with pytest.raises(AdapterBrokenError) as exc_info:
            await adapter.send("hello")
        assert "broken" in str(exc_info.value).lower()

    def test_broken_adapter_reports_idle(self):
        """is_idle() must return True when _broken=True, even if
        _pending_response is also True, so the UI stops showing a spinner.
        """
        adapter = CLIAdapter(handle="broken", display_name="broken")
        adapter._broken = True
        adapter._idle = False
        adapter._pending_response = True

        assert adapter.is_idle() is True
