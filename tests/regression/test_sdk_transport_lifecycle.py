# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for SDKTransport background-task lifecycle.

These tests cover two coupled defects in
``agent_os/agent/transports/sdk_transport.py`` (see
``TASK-fix-sdk-transport-lifecycle.md``):

1. ``dispatch()`` must keep a strong reference to the background
   response-consumption task, otherwise the event loop can GC it
   mid-stream and emit ``"Task was destroyed but it is pending"`` warnings.
2. ``stop()`` must cancel (and await) the background task BEFORE
   disconnecting the SDK client, so the task does not observe a None
   client mid-iteration and surface spurious errors.

All tests use asyncio mocks — no real subprocess, no network, no SDK
binary. Behavior is identical on macOS and Windows.
"""

from __future__ import annotations

import asyncio
import gc
import logging

import pytest

from agent_os.agent.transports.sdk_transport import SDKTransport


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


class _SlowReceiveClient:
    """Mock ClaudeSDKClient whose ``receive_response`` yields very slowly.

    Each iteration step sleeps ``delay`` seconds. The generator never
    terminates on its own — useful for exercising cancellation paths.
    ``disconnect`` is awaitable and records the call.
    """

    def __init__(self, delay: float = 0.05) -> None:
        self._delay = delay
        self.query_calls: list[str] = []
        self.disconnect_called = False
        self._receive_started = asyncio.Event()

    async def query(self, message: str) -> None:
        self.query_calls.append(message)

    def receive_response(self):
        return self._receive_gen()

    async def _receive_gen(self):
        self._receive_started.set()
        # Yield forever; each iteration awaits so cancellation can take
        # effect between yields.
        while True:
            await asyncio.sleep(self._delay)
            # We intentionally yield nothing that maps to an event — the
            # transport's _message_to_events() returns [] for unknown
            # message types, so the queue stays empty.
            yield object()

    async def disconnect(self) -> None:
        self.disconnect_called = True


class _ImmediateFailClient:
    """Mock client whose ``receive_response`` raises synchronously inside
    the async-generator body. Used to exercise the error-logging path."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.disconnect_called = False

    async def query(self, message: str) -> None:
        pass

    def receive_response(self):
        return self._gen()

    async def _gen(self):
        raise self._exc
        yield  # pragma: no cover - make it a generator

    async def disconnect(self) -> None:
        self.disconnect_called = True


def _install_slow_client(transport: SDKTransport, delay: float = 0.05) -> _SlowReceiveClient:
    client = _SlowReceiveClient(delay=delay)
    transport._client = client  # type: ignore[assignment]
    transport._alive = True
    return client


# --------------------------------------------------------------------- #
# 1. Background task is strongly referenced
# --------------------------------------------------------------------- #


async def test_bg_task_strongly_referenced_after_dispatch():
    """dispatch() must store the background task on the transport and
    survive multiple gc.collect() cycles without being GC'd.

    Fails on pre-fix code: orphan ``asyncio.create_task(...)`` only
    weakly referenced by the loop can be collected under memory
    pressure / gc."""
    transport = SDKTransport()
    _install_slow_client(transport, delay=0.05)

    await transport.dispatch("hello")

    # Strong reference exists and task is still running.
    assert transport._bg_task is not None, "dispatch() must store bg task on self"
    assert not transport._bg_task.done(), "bg task should still be running"

    # Force three gc cycles. If the task were only weakly referenced (as
    # on pre-fix code via a bare create_task() whose return value was
    # discarded) the event loop still holds it, but any higher-level
    # code that relies on a strong reference on `self` would fail here.
    for _ in range(3):
        gc.collect()

    # Task is still live and visible in all_tasks().
    assert transport._bg_task is not None
    assert not transport._bg_task.done()
    assert transport._bg_task in asyncio.all_tasks()

    # Clean up — cancel and let the event loop settle so the test
    # doesn't leak a pending task into the next test.
    transport._bg_task.cancel()
    try:
        await asyncio.wait_for(
            asyncio.gather(transport._bg_task, return_exceptions=True),
            timeout=2.0,
        )
    except asyncio.TimeoutError:
        pass


# --------------------------------------------------------------------- #
# 2. stop() cancels the bg task within its timeout
# --------------------------------------------------------------------- #


async def test_stop_cancels_bg_task_within_timeout():
    """stop() must cancel the in-flight background task and return
    within the 5s budget. Fails on pre-fix stop() which never touched
    the background task — the task lingered until GC."""
    transport = SDKTransport()
    client = _install_slow_client(transport, delay=0.05)

    await transport.dispatch("hello")
    bg_task = transport._bg_task
    assert bg_task is not None

    # Wait until the receive_response generator has actually started so
    # the cancellation path is exercised mid-iteration.
    await asyncio.wait_for(client._receive_started.wait(), timeout=2.0)

    loop = asyncio.get_running_loop()
    started = loop.time()
    await transport.stop()
    elapsed = loop.time() - started

    assert elapsed < 6.0, f"stop() took {elapsed:.2f}s, expected <6s"
    assert bg_task.done(), "bg task should be done after stop()"
    # Either cancelled cleanly OR finished (cancellation raced with
    # natural completion). Both are acceptable outcomes per spec.
    assert bg_task.cancelled() or bg_task.exception() is None

    # Transport state is fully torn down.
    assert transport._bg_task is None
    assert transport._client is None
    assert transport._alive is False
    assert client.disconnect_called


# --------------------------------------------------------------------- #
# 3. stop() with no dispatch is a no-op
# --------------------------------------------------------------------- #


async def test_stop_without_dispatch_is_noop():
    """Calling stop() before any dispatch() must complete quickly and
    not raise — no bg task exists to cancel."""
    transport = SDKTransport()

    loop = asyncio.get_running_loop()
    started = loop.time()
    await transport.stop()
    elapsed = loop.time() - started

    assert elapsed < 1.0
    assert transport._bg_task is None
    assert transport._alive is False


# --------------------------------------------------------------------- #
# 4. Background-task exceptions are logged, not silently swallowed
# --------------------------------------------------------------------- #


async def test_bg_task_exception_logged_not_swallowed(caplog):
    """When the background task raises, the error and stack trace must
    surface via ``logger.error(..., exc_info=...)`` and no
    ``"Task was destroyed"`` warning must appear. Pre-fix: the orphan
    task's exception was silently dropped by the event loop."""
    transport = SDKTransport()

    # Replace the consumer coroutine with one that raises a synthetic
    # error immediately. We have to patch the bound method so dispatch()
    # wires the same object.
    synthetic = RuntimeError("synthetic")

    async def boom() -> None:
        raise synthetic

    transport._consume_response_background = boom  # type: ignore[assignment]

    # Install a client with a no-op query so dispatch() reaches the
    # create_task line.
    _install_slow_client(transport, delay=0.05)

    with caplog.at_level(logging.ERROR, logger="agent_os.agent.transports.sdk_transport"):
        await transport.dispatch("hello")

        # Wait for the task to finish and the done-callback to fire.
        assert transport._bg_task is not None
        try:
            await asyncio.wait_for(
                asyncio.gather(transport._bg_task, return_exceptions=True),
                timeout=2.0,
            )
        except asyncio.TimeoutError:
            pytest.fail("bg task did not terminate")

        # Yield once so the loop can run the done-callback scheduled via
        # add_done_callback (fires on the next iteration after .done()).
        await asyncio.sleep(0)

    # The done-callback should have logged an ERROR containing our
    # synthetic marker and a traceback (via exc_info).
    matching = [
        r for r in caplog.records
        if r.levelno >= logging.ERROR and "synthetic" in (r.getMessage() + repr(r.exc_info))
    ]
    assert matching, (
        f"expected ERROR log with 'synthetic'; got {[r.getMessage() for r in caplog.records]}"
    )
    # exc_info was supplied on the record.
    assert any(r.exc_info is not None for r in matching), (
        "expected at least one ERROR record to carry exc_info for a traceback"
    )

    # No "Task was destroyed but it is pending" must appear in the log.
    task_destroyed = [
        r for r in caplog.records
        if "Task was destroyed but it is pending" in r.getMessage()
    ]
    assert not task_destroyed, "bg-task handling must not leak a GC warning"


# --------------------------------------------------------------------- #
# 5. stop() is idempotent
# --------------------------------------------------------------------- #


async def test_stop_is_idempotent():
    """Calling stop() twice must not raise. The second call sees a None
    bg_task / client and returns cleanly."""
    transport = SDKTransport()
    _install_slow_client(transport, delay=0.05)

    await transport.dispatch("hello")
    await transport.stop()

    # Second stop is a no-op — bg_task is None, client is None.
    await transport.stop()

    assert transport._bg_task is None
    assert transport._client is None
    assert transport._alive is False
