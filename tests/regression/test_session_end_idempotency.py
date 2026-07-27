# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: run_session_end_routine idempotency + no-delta gate.

Two callers can fire the routine for the same session boundary:
  - loop.py fire-and-forget after loop exits (via _on_session_end callback)
  - agent_manager.new_session() synchronous pre-archival

Two gates keep it cheap and safe under that duplicate-dispatch:

  1. Idempotency guard, keyed by ``session_uuid``: the second non-bypass call
     short-circuits with an INFO log so the LLM merge runs at most once per
     boundary (no double-stamped DECISIONS / INDEX entries).

  2. Persisted NO-DELTA gate (Layer-1 redesign): the routine is a no-op with NO
     LLM call when no Layer-1 file (state/decisions/lessons/index) changed since
     the last successful cleanup. To exercise the routine in a test you must
     therefore FORCE a delta — write a Layer-1 file before calling — otherwise
     the LLM is never invoked.

The completion marker is recorded only AFTER a run reaches the end of the
routine. ``bypass_idempotency=True`` skips both reading and writing the
completion set, so periodic mid-session refresh triggers can always re-run.
"""

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.testutils import streamable

from agent_os.agent import workspace_files as wsf_module
from agent_os.agent.workspace_files import (
    WorkspaceFileManager,
    run_session_end_routine,
)


# ---------------------------------------------------------------------------
# Helpers (mirrors tests/unit/test_workspace_files.py patterns)
# ---------------------------------------------------------------------------

def _mock_session(messages=None, session_id="sess_test"):
    session = MagicMock()
    # Set both F1 and F2; idempotency keys on F2 (session_uuid) per the
    # F7 canonical rename. Tests pre-rename used session_id as the key —
    # we mirror that into session_uuid for compatibility.
    session.session_id = session_id
    session.session_uuid = session_id
    session.get_messages.return_value = messages or [
        {"role": "user", "content": "Hello"},
    ]
    return session


def _mock_provider(response_text):
    provider = streamable(AsyncMock())
    resp = MagicMock()
    resp.text = response_text
    provider.complete.return_value = resp
    return provider


def _valid_llm_response(tag="x"):
    """A well-formed session-end merge payload in the NEW Layer-1 schema.

    Keys are the post-redesign set: project_state / decisions / lessons / index.
    The old ``session_log_entry`` and ``context`` keys are gone (SESSION_LOG was
    removed; CONTEXT.md became INDEX.md).
    """
    return json.dumps({
        "project_state": f"# State\nstate-{tag}",
        "decisions": f"## Decision {tag}\n**Chose:** A",
        "lessons": f"## Lesson {tag}\n**Problem:** p\n**Fix:** f",
        "index": f"## People\n- Person {tag}",
    })


def _force_delta(ws: WorkspaceFileManager, tag="seed"):
    """Make the no-delta gate report a delta so the routine actually runs.

    The routine is a NO-OP (no LLM call) when no Layer-1 file changed since the
    last cleanup marker. A fresh workspace has no marker → delta present; but to
    be robust against a marker written by a prior call in the same test, write a
    Layer-1 file here to guarantee a fresh delta.
    """
    ws.ensure_dir()
    ws.write("decisions", f"## 2026-06-15: Seed {tag}\n**Chose:** prior\n\n")
    assert wsf_module._has_cleanup_delta(ws) is True


@pytest.fixture(autouse=True)
def _reset_completion_set():
    """Reset the module-level idempotency set between tests."""
    wsf_module._completed_session_ends.clear()
    yield
    wsf_module._completed_session_ends.clear()


# ---------------------------------------------------------------------------
# Test 0: the no-delta gate must be primed for the routine to run at all
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_delta_gate_blocks_llm_without_a_change(tmp_path):
    """Sanity: with the cleanup marker current (no delta), the routine is a
    no-op and never calls the LLM. This is the gate every other test below
    primes around."""
    ws = WorkspaceFileManager(str(tmp_path))
    ws.ensure_dir()
    ws.write("decisions", "## seed\n**Chose:** A\n\n")
    # Mark as already cleaned → no un-consolidated delta remains.
    wsf_module._write_cleanup_marker(ws)
    assert wsf_module._has_cleanup_delta(ws) is False

    session = _mock_session(session_id="s_nodelta")
    provider = _mock_provider(_valid_llm_response("s_nodelta"))
    await run_session_end_routine(session, provider, ws, session_uuid="s_nodelta")

    # No delta → no LLM call, and nothing recorded for the boundary.
    assert provider.complete.call_count == 0
    assert "s_nodelta" not in wsf_module._completed_session_ends


# ---------------------------------------------------------------------------
# Test 1: Duplicate session_id short-circuits (sequential)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_duplicate_session_id_short_circuits(tmp_path, caplog):
    """Calling twice with same session_id runs the LLM exactly once."""
    ws = WorkspaceFileManager(str(tmp_path))
    _force_delta(ws, "s1")
    session = _mock_session(session_id="s1")
    provider = _mock_provider(_valid_llm_response("s1"))

    await run_session_end_routine(session, provider, ws, session_uuid="s1")
    assert provider.complete.call_count == 1
    # First successful run records the boundary.
    assert "s1" in wsf_module._completed_session_ends

    with caplog.at_level(logging.INFO, logger="agent_os.agent.workspace_files"):
        await run_session_end_routine(session, provider, ws, session_uuid="s1")

    # Second call must NOT have hit the LLM (idempotency short-circuit).
    assert provider.complete.call_count == 1
    # Short-circuit must log at INFO.
    assert any(
        "already completed" in rec.message and rec.levelno == logging.INFO
        for rec in caplog.records
    ), f"Expected 'already completed' INFO log, got: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Test 2: Different session_ids both run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_different_session_ids_both_run(tmp_path):
    """Different session_ids do not interfere — both runs complete.

    Each run is given its own workspace so the second run sees a fresh delta
    (the no-delta gate is per-workspace, not per-session)."""
    ws1 = WorkspaceFileManager(str(tmp_path / "p1"))
    _force_delta(ws1, "s1")
    session1 = _mock_session(session_id="s1")
    provider1 = _mock_provider(_valid_llm_response("s1"))
    await run_session_end_routine(session1, provider1, ws1, session_uuid="s1")

    ws2 = WorkspaceFileManager(str(tmp_path / "p2"))
    _force_delta(ws2, "s2")
    session2 = _mock_session(session_id="s2")
    provider2 = _mock_provider(_valid_llm_response("s2"))
    await run_session_end_routine(session2, provider2, ws2, session_uuid="s2")

    assert provider1.complete.call_count == 1
    assert provider2.complete.call_count == 1

    # Both session_ids are in the completion set
    assert "s1" in wsf_module._completed_session_ends
    assert "s2" in wsf_module._completed_session_ends


# ---------------------------------------------------------------------------
# Test 3: Concurrent calls for the same session_id — exactly one runs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_calls_same_session_id(tmp_path):
    """Two asyncio.gather tasks for the same session_id — LLM called once.

    The second task short-circuits on either the idempotency flag or the
    no-delta marker that the first task writes on completion; either way the
    merge runs at most once and no append-style file gets a duplicate entry."""
    ws = WorkspaceFileManager(str(tmp_path))
    _force_delta(ws, "s_conc")
    session = _mock_session(session_id="s_conc")

    # Shared provider so we can count total calls across both tasks.
    provider = _mock_provider(_valid_llm_response("s_conc"))

    results = await asyncio.gather(
        run_session_end_routine(session, provider, ws, session_uuid="s_conc"),
        run_session_end_routine(session, provider, ws, session_uuid="s_conc"),
        return_exceptions=True,
    )

    # Neither task raised
    for r in results:
        assert not isinstance(r, Exception), f"unexpected exception: {r!r}"

    # LLM was called exactly once — second task short-circuited (idempotency
    # flag or no-delta marker set by the first).
    assert provider.complete.call_count == 1, (
        f"expected LLM call count=1, got {provider.complete.call_count}. "
        f"The idempotency / no-delta guard did not prevent the duplicate run."
    )

    # No file corruption: each entry-structured file contains exactly ONE entry.
    # DECISIONS is stamped with metadata on the header line by the persist path,
    # so match the entry title rather than exact bytes.
    decisions = ws.read("decisions") or ""
    index = ws.read("index") or ""

    assert decisions.count("## Decision s_conc") == 1, (
        f"DECISIONS has duplicate entries:\n{decisions}"
    )
    assert index.count("## People") == 1, (
        f"INDEX has duplicate entries:\n{index}"
    )

    # Volatile/overwrite files reflect the single merge (no truncation / torn
    # writes). STATE and INDEX are overwrite-intent; LESSONS is stamped but its
    # body is byte-intact.
    assert ws.read("state") == "# State\nstate-s_conc"
    lessons = ws.read("lessons") or ""
    assert "## Lesson s_conc" in lessons
    assert "**Problem:** p" in lessons and "**Fix:** f" in lessons


# ---------------------------------------------------------------------------
# Test 4: idempotency records only after a real run; bypass always re-runs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_failure_runs_backstop_and_does_not_raise(tmp_path):
    """New contract: an LLM error during the merge no longer propagates. The
    routine logs it and falls through to the deterministic backstop, never
    deleting durable content. (Old behavior raised + left the boundary
    un-recorded; the redesign moved durability off the LLM path.)"""
    ws = WorkspaceFileManager(str(tmp_path))
    _force_delta(ws, "s_fail")
    session = _mock_session(session_id="s_fail")

    failing_provider = streamable(AsyncMock())
    failing_provider.complete.side_effect = RuntimeError("LLM exploded")

    # Must NOT raise — the deterministic backstop owns durability now.
    await run_session_end_routine(
        session, failing_provider, ws, session_uuid="s_fail",
    )

    assert failing_provider.complete.call_count == 1
    # Durable content survives the LLM failure (never deleted by the backstop).
    decisions = ws.read("decisions") or ""
    assert "Seed s_fail" in decisions
    # The routine still reached the end → boundary recorded.
    assert "s_fail" in wsf_module._completed_session_ends


@pytest.mark.asyncio
async def test_bypass_idempotency_reruns_and_does_not_record(tmp_path):
    """bypass_idempotency=True is the mid-session-refresh path: it ignores the
    completion set on read AND write, so the routine can re-run for the same
    boundary as often as a fresh delta allows — never blocked, never recorded."""
    ws = WorkspaceFileManager(str(tmp_path))

    # First bypass run with a delta primed.
    _force_delta(ws, "first")
    session = _mock_session(session_id="s_bypass")
    provider1 = _mock_provider(_valid_llm_response("first"))
    await run_session_end_routine(
        session, provider1, ws, session_uuid="s_bypass", bypass_idempotency=True,
    )
    assert provider1.complete.call_count == 1
    # bypass must NOT record the boundary.
    assert "s_bypass" not in wsf_module._completed_session_ends

    # Prime a new delta and run again under the SAME session_uuid — bypass means
    # it runs again rather than short-circuiting.
    _force_delta(ws, "second")
    provider2 = _mock_provider(_valid_llm_response("second"))
    await run_session_end_routine(
        session, provider2, ws, session_uuid="s_bypass", bypass_idempotency=True,
    )
    assert provider2.complete.call_count == 1
    assert "s_bypass" not in wsf_module._completed_session_ends

    # The second merge's content is what persisted.
    assert ws.read("state") == "# State\nstate-second"


# ---------------------------------------------------------------------------
# Test 5: session_uuid keyword-only and required
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_uuid_is_required_keyword_only(tmp_path):
    """run_session_end_routine must raise TypeError if session_uuid is missing
    (enforces deliberate caller updates; no silent regressions)."""
    ws = WorkspaceFileManager(str(tmp_path))
    session = _mock_session(session_id="s_kw")
    provider = _mock_provider(_valid_llm_response("s_kw"))

    with pytest.raises(TypeError):
        # Missing session_uuid kwarg
        await run_session_end_routine(session, provider, ws)
