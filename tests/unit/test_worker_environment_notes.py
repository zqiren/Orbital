# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A transport's environment notes (a sandbox the user must fix in the CLI's
own config) ride the worker turn's result, whatever its cause, so the manager
relays them (Windows smoke 2026-11-03, decision 1)."""

from __future__ import annotations

import asyncio

from agent_os.daemon_v2.process_manager import ProcessManager


class _Chunk:
    def __init__(self, text, chunk_type, metadata=None):
        self.text = text
        self.chunk_type = chunk_type
        self.timestamp = "2026-11-03T10:00:00+00:00"
        self.metadata = metadata or {}


class _Transport:
    def __init__(self, notes):
        self._notes = list(notes)

    def take_notes(self):
        notes, self._notes = self._notes, []
        return notes


class _Adapter:
    def __init__(self, script, transport):
        self._script = script
        self._transport = transport

    async def read_stream(self):
        for c in self._script:
            yield c


class _Transcript:
    filepath = "/tmp/_pm_notes_test.jsonl"

    def __init__(self):
        self.entries = []

    def append(self, entry):
        self.entries.append(entry)


class _Lifecycle:
    def __init__(self):
        self.completed, self.errors = [], []

    async def on_completed(self, p, h, summary, transcript_path, *, session_id=None):
        self.completed.append(summary)

    async def on_error(self, p, h, text, transcript_path, *, session_id=None):
        self.errors.append(text)

    async def on_turn_interrupted(self, *a, **k):
        pass

    async def on_thread_update(self, *a, **k):
        pass


class _WS:
    def broadcast(self, *a, **k):
        pass


class _Activity:
    def on_message(self, *a, **k):
        pass


async def _run(script, transport):
    lc = _Lifecycle()
    pm = ProcessManager(_WS(), _Activity(), lc)
    await pm.start("proj", "codex", _Adapter(script, transport), transcript=_Transcript(),
                   session_id="sess")
    await pm._tasks[pm._key("proj", "sess", "codex")]
    return lc


def test_note_follows_the_answer_of_a_completed_turn():
    script = [
        _Chunk("I could not run any command.", "response"),
        _Chunk("", "turn_complete", {"cause": "success", "session_id": "T1"}),
    ]
    lc = asyncio.run(_run(script, _Transport(["[Orbital] fix it in Codex"])))
    assert lc.completed == ["I could not run any command.\n\n[Orbital] fix it in Codex"]


def test_note_is_the_summary_when_the_turn_said_nothing():
    script = [_Chunk("", "turn_complete", {"cause": "success", "session_id": "T1"})]
    lc = asyncio.run(_run(script, _Transport(["[Orbital] fix it in Codex"])))
    assert lc.completed == ["[Orbital] fix it in Codex"]


def test_note_rides_an_errored_turn_too():
    script = [
        _Chunk("boom", "error"),
        _Chunk("", "turn_complete", {"cause": "error", "session_id": "T1"}),
    ]
    lc = asyncio.run(_run(script, _Transport(["[Orbital] fix it in Codex"])))
    assert lc.errors == ["boom\n\n[Orbital] fix it in Codex"]


def test_a_transport_without_notes_is_unchanged():
    script = [
        _Chunk("done", "response"),
        _Chunk("", "turn_complete", {"cause": "success", "session_id": "T1"}),
    ]
    lc = asyncio.run(_run(script, object()))
    assert lc.completed == ["done"]
