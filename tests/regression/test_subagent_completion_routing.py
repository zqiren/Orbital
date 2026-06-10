# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""SDK-transport completion path must route ``on_completed`` to the right session.

Background:
    Fix ``7d08d1b`` threaded ``session_id`` through the AgentMessageTool,
    LifecycleObserver, SubAgentManager, and AgentManager — but missed the
    ``ProcessManager.consume()`` background task, which is the ONLY path that
    fires ``on_completed`` for SDK-transport sub-agents (claude-code's default
    transport). Without ``session_id``, the lifecycle push silently dropped on
    any non-default management session.

    See ``docs/investigations/INVESTIGATION-2026-05-28-backend-still-broken.md``
    Section 2 (B1 root cause).

This file provides two complementary tests:

  1. Trace test — wires a REAL ProcessManager + REAL LifecycleObserver and
     verifies that a ``turn_complete`` chunk emitted by an adapter propagates
     ``session_id`` all the way to ``inject_system_message``. No mock of
     ProcessManager — that's the explicit requirement, because the missed bug
     lived inside ProcessManager itself and a mocked-ProcessManager test would
     have happily passed against the buggy code.

  2. Static-analysis test — greps every ``on_completed(`` / ``on_error(`` call
     site in ``agent_os/`` and asserts each one passes ``session_id=``. This is
     the safety net: when someone adds a NEW completion path in the future, the
     test fires before the silent drop ships.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.adapters.base import OutputChunk
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
from agent_os.daemon_v2.process_manager import ProcessManager


# ---------------------------------------------------------------------------
# (a) Trace test — real ProcessManager + real LifecycleObserver
# ---------------------------------------------------------------------------


class _StubAdapter:
    """Minimal adapter whose ``read_stream`` yields one ``turn_complete``.

    Designed to exercise the SDK-transport completion path inside
    ``ProcessManager.consume()``. We do not mock ProcessManager (the explicit
    requirement of this test) — instead we provide just enough adapter surface
    to drive the real consumer loop to its completion branch.
    """

    def __init__(self) -> None:
        self.handle = "claude-code"
        self._idle = False
        self._done = asyncio.Event()

    def is_alive(self) -> bool:
        return True

    def is_idle(self) -> bool:
        return self._idle

    async def read_stream(self) -> AsyncIterator[OutputChunk]:
        # Emit one response chunk so ``last_response_text`` is populated,
        # then a ``turn_complete`` sentinel — exactly the SDK transport's
        # documented end-of-turn signal. cause="success" because only a
        # verified success routes to on_completed under the honest-completion
        # contract (tests/regression/test_honest_completion_reporting.py).
        yield OutputChunk(text="primes are 2 3 5 7", chunk_type="response")
        self._idle = True
        yield OutputChunk(text="", chunk_type="turn_complete",
                          metadata={"cause": "success"})
        self._done.set()


@pytest.mark.asyncio
async def test_process_manager_threads_session_id_to_on_completed():
    """SDK transport's turn_complete must propagate session_id end-to-end.

    Wires REAL ``ProcessManager`` + REAL ``LifecycleObserver``. The only mocks
    are at the leaves: ``agent_manager`` (so we can assert the kwarg landed),
    the ``ws_manager`` and ``activity_translator`` (no behaviour under test),
    and the transcript object (only ``.filepath`` is read).

    We bypass ``SubAgentManager.start()`` here because its real construction
    requires a registry, setup_engine, project_store, platform_provider, and
    workspace machinery — none of which are under test. The path that matters
    for this regression is exactly ``ProcessManager.start(..., session_id=)``
    forwarding to ``LifecycleObserver.on_completed(..., session_id=)``, which
    is what we exercise directly. The companion file
    ``test_subagent_session_id_routing.py`` already verifies SubAgentManager's
    own threading; the bug being regressed here is the
    SubAgentManager → ProcessManager → LifecycleObserver hop.
    """
    mock_agent_mgr = MagicMock()
    mock_agent_mgr.inject_system_message = AsyncMock(return_value="delivered")
    mock_ws = MagicMock()
    mock_ws.broadcast = MagicMock()
    mock_activity = MagicMock()
    mock_activity.on_message = MagicMock()

    observer = LifecycleObserver(agent_manager=mock_agent_mgr, ws_manager=mock_ws)
    pm = ProcessManager(
        ws_manager=mock_ws,
        activity_translator=mock_activity,
        lifecycle_observer=observer,
    )

    adapter = _StubAdapter()
    transcript = MagicMock()
    transcript.filepath = "/tmp/fake_transcript.jsonl"
    transcript.append = MagicMock()

    # Drive the real consumer loop via the real ProcessManager.start().
    await pm.start(
        "proj_abc",
        "claude-code",
        adapter,
        transcript=transcript,
        session_id="sess_X",
    )

    # Wait for the background consume() task to drain (it terminates after
    # the adapter's async generator returns).
    key = "proj_abc:sess_X:claude-code"  # session-scoped key (Piece 3 Part E)
    task = pm._tasks[key]
    await asyncio.wait_for(task, timeout=2.0)

    # The completion push must have fired AND carried session_id=sess_X.
    # Exactly one on_completed is expected — the turn_complete chunk. (The
    # stream-end boundary after a closed turn emits nothing under the
    # honest-completion contract.)
    assert mock_agent_mgr.inject_system_message.await_count >= 1, (
        "on_completed never fired — the consumer didn't reach the lifecycle "
        "branch; check the adapter stub or ProcessManager.consume() logic"
    )
    for await_call in mock_agent_mgr.inject_system_message.await_args_list:
        assert await_call.kwargs.get("session_id") == "sess_X", (
            f"inject_system_message received session_id="
            f"{await_call.kwargs.get('session_id')!r}, expected 'sess_X'. "
            f"This is the silent-drop bug from "
            f"INVESTIGATION-2026-05-28-backend-still-broken.md §2."
        )


@pytest.mark.asyncio
async def test_process_manager_threads_session_id_to_on_error():
    """``on_error`` must also carry ``session_id`` so errored sub-agents wake
    the right management session.
    """
    mock_agent_mgr = MagicMock()
    mock_agent_mgr.inject_system_message = AsyncMock(return_value="delivered")
    mock_ws = MagicMock()
    mock_ws.broadcast = MagicMock()
    mock_activity = MagicMock()
    mock_activity.on_message = MagicMock()

    observer = LifecycleObserver(agent_manager=mock_agent_mgr, ws_manager=mock_ws)
    pm = ProcessManager(
        ws_manager=mock_ws,
        activity_translator=mock_activity,
        lifecycle_observer=observer,
    )

    class _ExplodingAdapter:
        handle = "claude-code"

        async def read_stream(self):
            # Force the consume() except branch.
            if False:
                yield  # pragma: no cover  # mark as async generator
            raise RuntimeError("transport blew up")

    transcript = MagicMock()
    transcript.filepath = "/tmp/fake_transcript.jsonl"
    transcript.append = MagicMock()

    await pm.start(
        "proj_abc",
        "claude-code",
        _ExplodingAdapter(),
        transcript=transcript,
        session_id="sess_X",
    )

    key = "proj_abc:sess_X:claude-code"  # session-scoped key (Piece 3 Part E)
    await asyncio.wait_for(pm._tasks[key], timeout=2.0)

    # on_error → _inject → inject_system_message with session_id=sess_X.
    assert mock_agent_mgr.inject_system_message.await_count == 1
    call = mock_agent_mgr.inject_system_message.await_args
    assert call.kwargs.get("session_id") == "sess_X"
    # And the content should be the error-flavoured system message.
    content = call.args[1]
    assert "stopped with error" in content


# ---------------------------------------------------------------------------
# (b) Static analysis test — every on_completed/on_error call passes session_id
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[2]
_AGENT_OS = _REPO_ROOT / "agent_os"

# These call patterns find INVOCATIONS, not DEFINITIONS. We explicitly skip
# lines that look like function definitions (``def on_completed(``,
# ``async def on_completed(``) and type-annotation-only mentions
# (``on_completed:``).
_CALL_RE = re.compile(r"(?<![A-Za-z0-9_])(on_completed|on_error|on_failed)\s*\(")
_DEF_RE = re.compile(r"^\s*(async\s+)?def\s+(on_completed|on_error|on_failed)\b")


def _iter_call_sites():
    """Yield (path, lineno, name, window) for every on_completed/on_error call."""
    for py in _AGENT_OS.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            if _DEF_RE.search(line):
                continue
            # Look for the call pattern in this line.
            m = _CALL_RE.search(line)
            if not m:
                continue
            # Pull a ~400-char window starting at the match so we can scan
            # multi-line argument lists for session_id.
            offset = sum(len(l) + 1 for l in lines[:idx]) + m.start()
            window = text[offset:offset + 400]
            yield py, idx + 1, m.group(1), window


def test_every_on_completed_call_passes_session_id():
    """Static guard: every ``on_completed(`` and ``on_error(`` call site in
    ``agent_os/`` must forward ``session_id=`` somewhere in its arg list.

    This catches the entire class of bug from
    INVESTIGATION-2026-05-28-backend-still-broken.md §2: a new completion path
    is added, the author forgets the session_id kwarg, and lifecycle pushes
    silently route to the wrong session. The runtime trace test above catches
    the specific path that bug lived in; this test catches every FUTURE path
    that hasn't been added yet.
    """
    offenders: list[str] = []
    for path, lineno, name, window in _iter_call_sites():
        # The call's argument list runs from the first ``(`` to its matching
        # ``)`` — for a quick static check, we just scan the window. A real
        # parser would be over-engineered for this surface.
        if "session_id" not in window:
            rel = path.relative_to(_REPO_ROOT)
            offenders.append(f"  {rel}:{lineno}  {name}(")

    assert not offenders, (
        "These call sites do not pass session_id=. Every lifecycle completion "
        "path must thread the management session_id so pushes land in the "
        "right SessionKey bucket. See "
        "docs/investigations/INVESTIGATION-2026-05-28-backend-still-broken.md "
        "§2 for the silent-drop bug this guards against.\n\n"
        + "\n".join(offenders)
    )
