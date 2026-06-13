# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the turn_complete boundary row ProcessManager appends to a
sub-agent transcript (TASK-subagent-last-message-display, Step 2.1).

A boundary row (`chunk_type="turn_complete"`, empty content) is appended to the
sub-agent transcript at each terminal turn so `read_sub_agent_summary` can split
per-turn. It is NOT appended for a clean reap (`cause="stopped"`). The existing
lifecycle calls and `last_response_text` reset are unchanged.
"""

from __future__ import annotations

import asyncio

from agent_os.daemon_v2.process_manager import ProcessManager


class _Chunk:
    def __init__(self, text, chunk_type, metadata=None):
        self.text = text
        self.chunk_type = chunk_type
        self.timestamp = "2026-06-13T10:00:00+00:00"
        self.metadata = metadata or {}


class _Adapter:
    def __init__(self, script):
        self._script = script
        self._transport = None

    async def read_stream(self):
        for c in self._script:
            yield c


class _Transcript:
    def __init__(self):
        self.entries = []
        self.filepath = "/tmp/_pm_boundary_test.jsonl"

    def append(self, entry):
        self.entries.append(entry)


class _Lifecycle:
    def __init__(self):
        self.completed, self.errors, self.interrupted = [], [], []

    async def on_completed(self, p, h, summary, transcript_path, *, session_id=None):
        self.completed.append(summary)

    async def on_error(self, *a, **k):
        self.errors.append(a)

    async def on_turn_interrupted(self, *a, **k):
        self.interrupted.append(a)

    async def on_thread_update(self, *a, **k):
        pass


class _WS:
    def broadcast(self, *a, **k):
        pass


class _Activity:
    def on_message(self, *a, **k):
        pass


async def _run(script):
    pm = ProcessManager(_WS(), _Activity(), _Lifecycle())
    tx = _Transcript()
    await pm.start("proj", "claude-code", _Adapter(script), transcript=tx, session_id="sess")
    key = pm._key("proj", "sess", "claude-code")
    await pm._tasks[key]
    return pm, tx


def test_boundary_row_appended_per_completed_turn():
    script = [
        _Chunk("[Using tool: Read]", "tool_activity", {"tool_name": "Read"}),
        _Chunk("TURN-1 final", "response"),
        _Chunk("", "turn_complete", {"cause": "success", "session_id": "s1"}),
        _Chunk("TURN-2 final", "response"),
        _Chunk("", "turn_complete", {"cause": "success", "session_id": "s1"}),
    ]
    pm, tx = asyncio.run(_run(script))
    types = [e.get("chunk_type") for e in tx.entries]
    # Two boundary rows, one closing each turn, interleaved after each turn's content.
    assert types == ["tool_activity", "response", "turn_complete", "response", "turn_complete"]
    boundaries = [e for e in tx.entries if e.get("chunk_type") == "turn_complete"]
    assert len(boundaries) == 2
    assert all((b.get("content") or "") == "" for b in boundaries)   # empty content (must not render)


def test_stopped_turn_appends_boundary_to_stay_aligned_with_its_marker():
    """A stopped turn (e.g. SDK per-turn cancel when a new dispatch arrives) HAS
    a preceding message_routed marker. It MUST get a boundary too, or the i-th
    marker ↔ i-th slice pairing drifts and aliases the next turn's text into the
    stopped dispatch's bubble (review finding: cause=stopped drift)."""
    script = [
        _Chunk("partial work", "response"),
        _Chunk("", "turn_complete", {"cause": "stopped", "session_id": "s1"}),
    ]
    pm, tx = asyncio.run(_run(script))
    types = [e.get("chunk_type") for e in tx.entries]
    assert types == ["response", "turn_complete"]   # boundary IS appended for stopped
    boundary = [e for e in tx.entries if e.get("chunk_type") == "turn_complete"][0]
    assert (boundary.get("content") or "") == ""


def test_errored_turn_appends_boundary():
    script = [
        _Chunk("[Using tool: Bash]", "tool_activity", {"tool_name": "Bash"}),
        _Chunk("boom", "error"),
        _Chunk("", "turn_complete", {"cause": "error", "session_id": "s1"}),
    ]
    pm, tx = asyncio.run(_run(script))
    assert [e.get("chunk_type") for e in tx.entries][-1] == "turn_complete"


def test_interrupted_turn_appends_boundary():
    """An interrupted turn (Codex cancel; thread lives on) also gets a boundary
    — every terminal turn_complete delimits a turn (review finding: interrupted
    boundary was untested)."""
    script = [
        _Chunk("[Using tool: Read]", "tool_activity", {"tool_name": "Read"}),
        _Chunk("partial", "response"),
        _Chunk("", "turn_complete", {"cause": "interrupted", "session_id": "s1"}),
    ]
    pm, tx = asyncio.run(_run(script))
    assert [e.get("chunk_type") for e in tx.entries][-1] == "turn_complete"


def test_note_turn_closed_appends_boundary_for_blocking_transports():
    """Blocking transports (Pipe/ACP/PTY) never emit turn_complete — they call
    note_turn_closed after each dispatch. That hook MUST append the turn
    boundary, or a blocking-transport transcript has ZERO boundaries → the whole
    file collapses to one slice → every dispatch but the first is dropped and the
    last turn's text is aliased onto the first (review finding: blocking
    transports never append a boundary)."""
    async def run():
        pm = ProcessManager(_WS(), _Activity(), _Lifecycle())
        tx = _Transcript()
        # Empty-stream adapter (mirrors PipeTransport.read_stream()).
        await pm.start("proj", "claude-code", _Adapter([]), transcript=tx, session_id="sess")
        key = pm._key("proj", "sess", "claude-code")
        await pm._tasks[key]
        # Blocking path appends the response itself, then closes the turn:
        tx.append({"source": "claude-code", "content": "pipe result", "chunk_type": "response",
                   "timestamp": "2026-06-13T10:00:00+00:00"})
        pm.note_turn_closed("proj", "claude-code", session_id="sess")
        return tx
    tx = asyncio.run(run())
    types = [e.get("chunk_type") for e in tx.entries]
    assert types == ["response", "turn_complete"]   # note_turn_closed appended the boundary
