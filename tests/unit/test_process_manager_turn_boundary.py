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


def test_boundary_carries_dispatch_id_set_before_the_turn():
    """TASK-dispatch-id-pairing: ``set_active_dispatch`` records the id that
    started this turn; the closing boundary row is stamped with it, and the
    slot is cleared so a later turn WITHOUT a fresh dispatch never inherits a
    stale id."""
    script = [
        _Chunk("TURN-1 final", "response"),
        _Chunk("", "turn_complete", {"cause": "success", "session_id": "s1"}),
    ]

    async def run():
        pm = ProcessManager(_WS(), _Activity(), _Lifecycle())
        tx = _Transcript()
        pm.set_active_dispatch("proj", "claude-code", "sess:aaaa1111", session_id="sess")
        await pm.start("proj", "claude-code", _Adapter(script), transcript=tx, session_id="sess")
        key = pm._key("proj", "sess", "claude-code")
        await pm._tasks[key]
        return pm, tx, key

    pm, tx, key = asyncio.run(run())
    boundary = [e for e in tx.entries if e.get("chunk_type") == "turn_complete"][0]
    assert boundary["dispatch_id"] == "sess:aaaa1111"
    # Consumed — a later boundary with no fresh set_active_dispatch call must
    # not inherit this id (would misattribute the next turn to this dispatch).
    assert key not in pm._active_dispatch_id


def test_boundary_has_no_dispatch_id_key_when_none_was_set():
    """No set_active_dispatch call before the turn (e.g. a stray/legacy path)
    → the boundary carries no dispatch_id key at all (not even None) —
    read_sub_agent_summary treats a missing key the same as None."""
    script = [
        _Chunk("TURN-1 final", "response"),
        _Chunk("", "turn_complete", {"cause": "success", "session_id": "s1"}),
    ]
    pm, tx = asyncio.run(_run(script))
    boundary = [e for e in tx.entries if e.get("chunk_type") == "turn_complete"][0]
    assert "dispatch_id" not in boundary


def test_second_turn_without_new_dispatch_gets_no_stale_id():
    """Two turns, but set_active_dispatch is only called once (for turn 1).
    Turn 2's boundary must NOT silently carry turn 1's id."""
    script = [
        _Chunk("TURN-1", "response"),
        _Chunk("", "turn_complete", {"cause": "success", "session_id": "s1"}),
        _Chunk("TURN-2", "response"),
        _Chunk("", "turn_complete", {"cause": "success", "session_id": "s1"}),
    ]

    async def run():
        pm = ProcessManager(_WS(), _Activity(), _Lifecycle())
        tx = _Transcript()
        pm.set_active_dispatch("proj", "claude-code", "sess:one", session_id="sess")
        await pm.start("proj", "claude-code", _Adapter(script), transcript=tx, session_id="sess")
        key = pm._key("proj", "sess", "claude-code")
        await pm._tasks[key]
        return tx

    tx = asyncio.run(run())
    boundaries = [e for e in tx.entries if e.get("chunk_type") == "turn_complete"]
    assert boundaries[0]["dispatch_id"] == "sess:one"
    assert "dispatch_id" not in boundaries[1]


def test_note_turn_closed_stamps_dispatch_id_for_blocking_transports():
    """The blocking-transport boundary path (note_turn_closed) also consumes
    the active dispatch id, mirroring the streaming consume() path."""
    async def run():
        pm = ProcessManager(_WS(), _Activity(), _Lifecycle())
        tx = _Transcript()
        await pm.start("proj", "claude-code", _Adapter([]), transcript=tx, session_id="sess")
        key = pm._key("proj", "sess", "claude-code")
        await pm._tasks[key]
        tx.append({"source": "claude-code", "content": "pipe result", "chunk_type": "response",
                   "timestamp": "2026-06-13T10:00:00+00:00"})
        pm.set_active_dispatch("proj", "claude-code", "sess:blocking1", session_id="sess")
        pm.note_turn_closed("proj", "claude-code", session_id="sess")
        return pm, tx, key

    pm, tx, key = asyncio.run(run())
    boundary = [e for e in tx.entries if e.get("chunk_type") == "turn_complete"][0]
    assert boundary["dispatch_id"] == "sess:blocking1"
    assert key not in pm._active_dispatch_id


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


# ---------------------------------------------------------------------------
# FIFO dispatch-id queue (Important review finding on TASK-dispatch-id-pairing):
# a single last-writer-wins slot lets a second dispatch's set_active_dispatch
# overwrite the first's id before the first turn's (cancelled) boundary has
# been written — reachable via rapid re-dispatch to the same handle (no
# busy-guard on the real dispatch path). The transport cancels the prior
# turn (and its "stopped" turn_complete is enqueued) BEFORE the next turn's
# query() runs, so boundaries close in dispatch order — a per-key FIFO
# queue therefore restores 1:1 pairing by construction: whichever id was
# enqueued FIRST is popped by whichever boundary closes FIRST.
# ---------------------------------------------------------------------------

class _HangingAdapter:
    """An adapter whose stream never ends on its own — used to keep a
    consumer task genuinely alive (not yet done) so ProcessManager.start()'s
    'cancel the still-live prior consumer' branch actually fires, rather
    than racing a real timing window."""

    def __init__(self):
        self._transport = None

    async def read_stream(self):
        await asyncio.Event().wait()
        yield  # pragma: no cover — unreachable; only makes this an async generator


def test_fifo_prevents_stale_id_aliasing_on_rapid_second_dispatch():
    """The reviewer's exact race, simulated: BOTH ids are enqueued (mirrors
    send() #2 calling set_active_dispatch before the transport has even
    cancelled send() #1's turn) before EITHER boundary is processed. The
    FIRST boundary to close (the cancelled turn's "stopped" turn_complete)
    must consume the FIRST-enqueued id; the real turn's boundary must get
    the SECOND. Under the old single-slot dict, set_active_dispatch(idB)
    overwrites idA — boundary 1 would incorrectly pop idB (stamping
    dispatch 2's id on dispatch 1's aborted/partial turn — content aliasing
    if that turn has partial text) and boundary 2 would pop nothing (its
    bubble permanently lost)."""
    script = [
        _Chunk("partial (aborted)", "response"),
        _Chunk("", "turn_complete", {"cause": "stopped", "session_id": "s1"}),
        _Chunk("REAL RESPONSE", "response"),
        _Chunk("", "turn_complete", {"cause": "success", "session_id": "s1"}),
    ]

    async def run():
        pm = ProcessManager(_WS(), _Activity(), _Lifecycle())
        tx = _Transcript()
        pm.set_active_dispatch("proj", "claude-code", "sess:idA", session_id="sess")
        pm.set_active_dispatch("proj", "claude-code", "sess:idB", session_id="sess")
        await pm.start("proj", "claude-code", _Adapter(script), transcript=tx, session_id="sess")
        key = pm._key("proj", "sess", "claude-code")
        await pm._tasks[key]
        return tx

    tx = asyncio.run(run())
    boundaries = [e for e in tx.entries if e.get("chunk_type") == "turn_complete"]
    assert len(boundaries) == 2
    assert boundaries[0]["dispatch_id"] == "sess:idA"   # aborted turn -> FIRST-enqueued id
    assert boundaries[1]["dispatch_id"] == "sess:idB"   # real turn -> SECOND-enqueued id


def test_clear_dispatch_removes_specific_id_without_touching_others():
    """Guard (a): a dispatch that fails after enqueueing but before the
    transport ever owns the turn must have its id removed — surgically, by
    VALUE, not by position — so an older still-pending id (a real turn
    still in flight ahead of it) is untouched."""
    pm = ProcessManager(_WS(), _Activity(), _Lifecycle())
    pm.set_active_dispatch("proj", "claude-code", "idA", session_id="sess")
    pm.set_active_dispatch("proj", "claude-code", "idB", session_id="sess")
    pm.clear_dispatch("proj", "claude-code", "idB", session_id="sess")
    key = pm._key("proj", "sess", "claude-code")
    assert list(pm._active_dispatch_id.get(key, [])) == ["idA"]


def test_clear_dispatch_on_unknown_id_is_a_noop():
    pm = ProcessManager(_WS(), _Activity(), _Lifecycle())
    pm.set_active_dispatch("proj", "claude-code", "idA", session_id="sess")
    pm.clear_dispatch("proj", "claude-code", "never-enqueued", session_id="sess")
    key = pm._key("proj", "sess", "claude-code")
    assert list(pm._active_dispatch_id.get(key, [])) == ["idA"]


def test_dispatch_after_a_cleared_failed_dispatch_still_pairs_correctly():
    """Guard (a) end-to-end: a failed dispatch's id is cleared before it can
    ever be popped; the NEXT (successful) dispatch's id is still popped by
    ITS OWN boundary, unaffected by the failed one."""
    script = [
        _Chunk("real response", "response"),
        _Chunk("", "turn_complete", {"cause": "success", "session_id": "s1"}),
    ]

    async def run():
        pm = ProcessManager(_WS(), _Activity(), _Lifecycle())
        tx = _Transcript()
        # Dispatch 1 enqueues then "fails" before the transport owns the turn.
        pm.set_active_dispatch("proj", "claude-code", "failed-id", session_id="sess")
        pm.clear_dispatch("proj", "claude-code", "failed-id", session_id="sess")
        # Dispatch 2 enqueues and actually runs.
        pm.set_active_dispatch("proj", "claude-code", "real-id", session_id="sess")
        await pm.start("proj", "claude-code", _Adapter(script), transcript=tx, session_id="sess")
        key = pm._key("proj", "sess", "claude-code")
        await pm._tasks[key]
        return tx

    tx = asyncio.run(run())
    boundary = [e for e in tx.entries if e.get("chunk_type") == "turn_complete"][0]
    assert boundary["dispatch_id"] == "real-id"


def test_stop_clears_entire_queue_not_just_one_id():
    """Guard (b), reap side: stop() must clear EVERY pending id for the key,
    not just one — a still-queued id from a turn that will never close (the
    consumer is being torn down) must not leak into a later respawn."""
    async def run():
        pm = ProcessManager(_WS(), _Activity(), _Lifecycle())
        tx = _Transcript()
        await pm.start("proj", "claude-code", _Adapter([]), transcript=tx, session_id="sess")
        pm.set_active_dispatch("proj", "claude-code", "idA", session_id="sess")
        pm.set_active_dispatch("proj", "claude-code", "idB", session_id="sess")
        await pm.stop("proj", "claude-code", session_id="sess")
        return pm
    pm = asyncio.run(run())
    key = pm._key("proj", "sess", "claude-code")
    assert key not in pm._active_dispatch_id


def test_start_respawn_clears_stale_queue_from_prior_incarnation():
    """Guard (b), respawn side: start() called again for a key with a still-
    live consumer tears the OLD consumer down and replaces it. Any
    dispatch_id(s) still queued for the OLD incarnation will never be
    popped by a boundary from it (that transport is gone) — fail closed by
    clearing them, rather than letting them leak into the NEW incarnation's
    pairing."""
    async def run():
        pm = ProcessManager(_WS(), _Activity(), _Lifecycle())
        tx1 = _Transcript()
        await pm.start("proj", "claude-code", _HangingAdapter(), transcript=tx1, session_id="sess")
        pm.set_active_dispatch("proj", "claude-code", "stale-id", session_id="sess")

        tx2 = _Transcript()
        # Respawn: the still-live consumer above gets cancelled and replaced.
        await pm.start("proj", "claude-code", _Adapter([]), transcript=tx2, session_id="sess")
        return pm

    pm = asyncio.run(run())
    key = pm._key("proj", "sess", "claude-code")
    assert key not in pm._active_dispatch_id


# ---------------------------------------------------------------------------
# Additional "enqueue without a guaranteed consume-or-clear" branches found
# on re-review of the FIFO fix: a dispatch can END without EITHER a
# turn_complete boundary OR an exception the earlier guard (a) catches.
# ---------------------------------------------------------------------------

class _CrashingAdapter:
    """Yields some chunks, then the stream itself raises — models an
    unexpected exception inside the consumer loop (as opposed to the
    transport honestly reporting a turn_complete with cause='error')."""

    def __init__(self, script):
        self._script = script
        self._transport = None

    async def read_stream(self):
        for c in self._script:
            yield c
        raise RuntimeError("transport crashed mid-stream")


def test_stream_death_without_turn_complete_clears_stale_id():
    """Closure requested after re-review: the transport's stream can end
    abnormally — process died mid-turn — WITHOUT ever emitting a
    turn_complete chunk at all (the existing 'path c' on_error handling
    right after the read loop). No boundary is written for this turn, so
    the id enqueued for it must be discarded here too, or it leaks into
    whatever dispatch runs next on this key (permanent off-by-one, same
    failure class as the falsy-response gap)."""
    script = [
        _Chunk("partial, then the process died", "response"),
        # NO closing turn_complete — the stream just ends here.
    ]

    async def run():
        pm = ProcessManager(_WS(), _Activity(), _Lifecycle())
        tx = _Transcript()
        pm.set_active_dispatch("proj", "claude-code", "dead-turn-id", session_id="sess")
        await pm.start("proj", "claude-code", _Adapter(script), transcript=tx, session_id="sess")
        key = pm._key("proj", "sess", "claude-code")
        await pm._tasks[key]
        return pm, key, tx

    pm, key, tx = asyncio.run(run())
    assert key not in pm._active_dispatch_id
    # No boundary is written either (unchanged pre-existing behavior) — the
    # marker for this dispatch simply renders no bubble on read, honestly.
    assert not any(e.get("chunk_type") == "turn_complete" for e in tx.entries)


def test_consume_loop_exception_clears_stale_id():
    """Closure requested after re-review: an unexpected exception INSIDE the
    consumer loop itself (not a transport-reported error chunk) also means
    this turn's boundary will never be written — the enqueued id must be
    discarded here too."""
    script = [_Chunk("partial", "response")]

    async def run():
        pm = ProcessManager(_WS(), _Activity(), _Lifecycle())
        tx = _Transcript()
        pm.set_active_dispatch("proj", "claude-code", "crash-id", session_id="sess")
        await pm.start("proj", "claude-code", _CrashingAdapter(script), transcript=tx, session_id="sess")
        key = pm._key("proj", "sess", "claude-code")
        await pm._tasks[key]
        return pm, key

    pm, key = asyncio.run(run())
    assert key not in pm._active_dispatch_id
