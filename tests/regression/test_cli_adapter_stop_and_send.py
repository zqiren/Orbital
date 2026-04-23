# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for TASK-fix-cli-adapter-stop-and-send-cleanup.

Covers two coupled defects in CLIAdapter:

Part A — stop() did not cancel an in-flight transport.send(), causing
         _send_lock to be held indefinitely. Root cause of B3 "Stop
         button has no effect" until the 5-minute force-idle watchdog.

Part B — send() had no exception guard. A raised transport.send() left
         _pending_response=True permanently so the adapter was stuck
         busy until daemon restart.

Tests use pure asyncio mocks — portable across macOS and Windows.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock

import pytest

from agent_os.agent.adapters.cli_adapter import CLIAdapter


def _make_transport_mock():
    """Return a MagicMock pretending to be an AgentTransport.

    send() is left unset so each test can install the behaviour it needs.
    """
    transport = MagicMock()
    transport.stop = MagicMock()
    # transport.stop() is awaited by CLIAdapter.stop(); provide an awaitable.

    async def _stop():
        return None

    transport.stop = _stop
    return transport


def _make_adapter_with_transport(transport) -> CLIAdapter:
    adapter = CLIAdapter(handle="test", display_name="test", transport=transport)
    return adapter


# --------------------------------------------------------------------- #
# Part A — stop cancels in-flight send
# --------------------------------------------------------------------- #


class TestStopCancelsInflightSend:
    @pytest.mark.asyncio
    async def test_stop_cancels_blocked_send(self):
        """stop() must cancel an in-flight send() so the caller unblocks
        within 3s and _send_lock is released."""
        transport = _make_transport_mock()

        never_future: asyncio.Future = asyncio.get_event_loop().create_future()

        async def hanging_send(_msg):
            # Never resolves on its own; must be cancelled.
            return await never_future

        transport.send = hanging_send
        adapter = _make_adapter_with_transport(transport)

        send_task = asyncio.create_task(adapter.send("hello"))

        # Let send() acquire the lock and create the inflight task.
        await asyncio.sleep(0.1)
        assert adapter._inflight_send is not None
        assert adapter._send_lock.locked(), "send should have acquired the lock"

        # Calling stop() must return within 3s even though send() would hang.
        await asyncio.wait_for(adapter.stop(), timeout=3.0)

        # send() task must have been cancelled.
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(send_task, timeout=1.0)

        # Lock must be released (finally-block unwinds the context manager).
        assert adapter._send_lock.locked() is False
        assert adapter._inflight_send is None

    @pytest.mark.asyncio
    async def test_send_after_stop_does_not_deadlock(self):
        """After a stop-cancel cycle the adapter must still accept new
        sends — the lock must not remain permanently held."""
        transport = _make_transport_mock()

        never_future: asyncio.Future = asyncio.get_event_loop().create_future()

        async def hanging_send(_msg):
            return await never_future

        transport.send = hanging_send
        adapter = _make_adapter_with_transport(transport)

        send_task = asyncio.create_task(adapter.send("one"))
        await asyncio.sleep(0.1)
        await asyncio.wait_for(adapter.stop(), timeout=3.0)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(send_task, timeout=1.0)

        # Now install a fast-completing send and fire a new request.
        async def quick_send(_msg):
            return "ok"

        transport.send = quick_send

        # This must acquire the lock immediately; if the lock leaked it
        # would hang past our 2s budget.
        await asyncio.wait_for(adapter.send("two"), timeout=2.0)
        assert adapter._send_lock.locked() is False
        assert adapter._inflight_send is None

    @pytest.mark.asyncio
    async def test_repeated_stop_send_cycles(self):
        """10 sequential send+stop cycles must each complete within 5s
        and never leak the lock."""
        transport = _make_transport_mock()
        adapter = _make_adapter_with_transport(transport)

        for _ in range(10):
            never_future: asyncio.Future = (
                asyncio.get_event_loop().create_future()
            )

            async def hanging_send(_msg, _f=never_future):
                return await _f

            transport.send = hanging_send

            send_task = asyncio.create_task(adapter.send("hi"))
            await asyncio.sleep(0.05)
            await asyncio.wait_for(adapter.stop(), timeout=5.0)
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(send_task, timeout=1.0)
            assert adapter._send_lock.locked() is False
            assert adapter._inflight_send is None


# --------------------------------------------------------------------- #
# Part B — send exception cleanup
# --------------------------------------------------------------------- #


class TestSendExceptionCleanup:
    @pytest.mark.asyncio
    async def test_send_exception_resets_pending_response(self):
        """If transport.send raises, _pending_response must be cleared
        and is_idle() must become True so the adapter is reusable."""
        transport = _make_transport_mock()

        async def failing_send(_msg):
            raise RuntimeError("synthetic")

        transport.send = failing_send
        adapter = _make_adapter_with_transport(transport)

        with pytest.raises(RuntimeError, match="synthetic"):
            await adapter.send("hello")

        assert adapter._pending_response is False
        assert adapter.is_idle() is True
        assert adapter._inflight_send is None

    @pytest.mark.asyncio
    async def test_send_exception_logged_at_error(self, caplog):
        """transport.send raising must produce an ERROR log with exc_info."""
        transport = _make_transport_mock()

        async def failing_send(_msg):
            raise RuntimeError("synthetic")

        transport.send = failing_send
        adapter = _make_adapter_with_transport(transport)

        with caplog.at_level(
            logging.ERROR, logger="agent_os.agent.adapters.cli_adapter"
        ):
            with pytest.raises(RuntimeError, match="synthetic"):
                await adapter.send("hello")

        matching = [
            r
            for r in caplog.records
            if r.levelno == logging.ERROR
            and "transport.send" in r.getMessage()
        ]
        assert matching, (
            "expected ERROR log from cli_adapter on send failure — "
            f"got: {[r.getMessage() for r in caplog.records]}"
        )
        assert matching[0].exc_info is not None, "exc_info missing"
        import traceback

        exc_text = "".join(traceback.format_exception(*matching[0].exc_info))
        assert "synthetic" in exc_text

    @pytest.mark.asyncio
    async def test_cancelled_send_does_not_log_as_error(self, caplog):
        """External cancellation of a send must propagate CancelledError
        and must NOT produce an ERROR log — cancellation is a normal path."""
        transport = _make_transport_mock()

        never_future: asyncio.Future = asyncio.get_event_loop().create_future()

        async def hanging_send(_msg):
            return await never_future

        transport.send = hanging_send
        adapter = _make_adapter_with_transport(transport)

        with caplog.at_level(
            logging.ERROR, logger="agent_os.agent.adapters.cli_adapter"
        ):
            send_task = asyncio.create_task(adapter.send("hi"))
            await asyncio.sleep(0.1)
            # Cancel the outer task; asyncio propagates through the await
            # on the inflight task.
            send_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await send_task

        error_records = [
            r for r in caplog.records if r.levelno == logging.ERROR
        ]
        assert not error_records, (
            f"unexpected ERROR log on cancellation: "
            f"{[r.getMessage() for r in error_records]}"
        )
        # Clean up the dangling future.
        if not never_future.done():
            never_future.cancel()

    @pytest.mark.asyncio
    async def test_inflight_send_cleared_in_all_paths(self):
        """_inflight_send and _pending_response must be None/False after
        success, exception, and cancellation."""
        # Success path.
        transport = _make_transport_mock()

        async def ok_send(_msg):
            return "ok"

        transport.send = ok_send
        adapter = _make_adapter_with_transport(transport)
        await adapter.send("hi")
        assert adapter._inflight_send is None
        assert adapter._pending_response is False

        # Exception path.
        async def boom_send(_msg):
            raise RuntimeError("boom")

        transport.send = boom_send
        adapter2 = _make_adapter_with_transport(transport)
        with pytest.raises(RuntimeError):
            await adapter2.send("hi")
        assert adapter2._inflight_send is None
        assert adapter2._pending_response is False

        # Cancellation path.
        never_future: asyncio.Future = asyncio.get_event_loop().create_future()

        async def hanging_send(_msg):
            return await never_future

        transport.send = hanging_send
        adapter3 = _make_adapter_with_transport(transport)
        send_task = asyncio.create_task(adapter3.send("hi"))
        await asyncio.sleep(0.05)
        send_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await send_task
        assert adapter3._inflight_send is None
        assert adapter3._pending_response is False
        if not never_future.done():
            never_future.cancel()
