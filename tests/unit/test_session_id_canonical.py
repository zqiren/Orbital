# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: canonical ``session_id`` (Format 1) vs ``session_uuid`` (Format 2).

Per ``TASK/ACTIVE-session-and-queue-model.md`` and the F7 audit
(``TASK/INVESTIGATION-session-id-canonical-audit.md``):

* ``session_id`` is the **user-facing chat thread identity** (Format 1, F1).
  Stable within a chat thread; rotates when ``/new-session`` is invoked.
  Stamped onto every appended message and read by the frontend at
  ``web/src/utils/chatTransform.ts`` as an opaque equality key to draw
  session-boundary separators.

* ``session_uuid`` is the **JSONL filename stem** (Format 2, F2).
  ``{sanitized_project_name}_{uuid4().hex[:8]}``. Used for file paths,
  ``tool-results/`` directories, and idempotency keys. Always distinct
  across sessions, never user-facing.

This module asserts the five canonical-rename invariants from the F7 audit:

  1. F1 reaches the frontend (in the message ``session_id`` field).
  2. F2 retains its file-naming role (filename derives from ``session_uuid``).
  3. Old JSONLs load cleanly under the Option B compat layer (no migration).
  4. ``/new-session`` rotates F1 (new chat thread).
  5. The API surfaces ``session_id`` on ``/agents/start`` and ``/inject``
     even though internal storage paths use ``session_uuid``.

The Option B compat layer (F7 §Q5) is the load-bearing constraint these
tests defend: filenames stay F2 forever; the F1 chat id flips to a distinct
value for new sessions only; old JSONLs are not rewritten on disk.
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.project_paths import ProjectPaths
from agent_os.agent.session import Session
from agent_os.api.routes import agents_v2


# ---------------------------------------------------------------------------
# Invariant 1: F1 reaches the frontend (message ``session_id`` field carries F1)
# ---------------------------------------------------------------------------

def test_appended_messages_carry_f1_session_id(tmp_path):
    """When ``Session.new`` is invoked with an explicit F1 chat id, every
    appended message has that F1 value in its ``session_id`` field.

    The frontend reads ``msg.session_id`` and treats it as an opaque equality
    key (see ``web/src/utils/chatTransform.ts``); the canonical rename means
    the field must carry F1, not F2.
    """
    session = Session.new(
        "myproject_a1b2c3d4",      # F2 storage stem
        str(tmp_path),
        session_id="sess_research_openhuman",  # F1 chat id
    )
    session.append({"role": "user", "content": "hello"})
    session.append({"role": "assistant", "content": "hi"})

    # The in-memory field must be F1.
    assert session.session_id == "sess_research_openhuman"
    # The F2 stem stays accessible as a separate field for filename derivation.
    assert session.session_uuid == "myproject_a1b2c3d4"

    # Every persisted message's ``session_id`` must be the F1 value (what the
    # frontend reads). Skip the JSONL ``role: meta`` lifecycle records.
    sessions_dir = ProjectPaths(str(tmp_path)).sessions_dir
    jsonl_path = os.path.join(sessions_dir, "myproject_a1b2c3d4.jsonl")
    with open(jsonl_path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    convo = [m for m in lines if m.get("role") != "meta"]
    assert len(convo) == 2
    for m in convo:
        assert m["session_id"] == "sess_research_openhuman", (
            f"message must carry F1 session_id, got {m['session_id']!r}"
        )


# ---------------------------------------------------------------------------
# Invariant 2: F2 retains its file-naming role
# ---------------------------------------------------------------------------

def test_filename_derives_from_session_uuid(tmp_path):
    """``Session.new(session_uuid, ...)`` writes the JSONL at the path
    derived from the F2 stem, NOT the F1 chat id. The
    ``ProjectPaths.session_file`` helper must agree.
    """
    session = Session.new(
        "proj_aabbccdd",
        str(tmp_path),
        session_id="sess_user_thread",
    )
    # Materialize the file — deferred creation means nothing is on disk until
    # the first message; the path it lands at is what this test pins down.
    session.append({"role": "user", "content": "x"})

    pp = ProjectPaths(str(tmp_path))
    expected = pp.session_file("proj_aabbccdd")
    assert os.path.isfile(expected), (
        f"JSONL must be written at the F2-derived path, expected {expected}"
    )

    # The F1 chat id MUST NOT be used to derive the filename.
    f1_path = pp.session_file("sess_user_thread")
    assert not os.path.isfile(f1_path), (
        "JSONL filename must derive from F2 (session_uuid), not F1 (session_id)"
    )

    # The Session's internal _filepath must match the F2-derived path.
    assert os.path.normpath(session._filepath) == os.path.normpath(expected)


# ---------------------------------------------------------------------------
# Invariant 3: Old JSONLs load cleanly under the Option B compat layer
# ---------------------------------------------------------------------------

def test_legacy_jsonl_loads_with_session_id_falling_back_to_uuid(tmp_path):
    """Pre-canonical-rename JSONLs were written with the F2 stem in the
    ``session_id`` field (because the two values were one and the same).
    The Option B compat layer (F7 §Q5) does NOT rewrite these files on disk
    — they continue to read cleanly, and ``Session.load()`` populates the
    in-memory ``session_uuid`` from the filename stem AND keeps
    ``session_id`` falling back to that same stem (so the frontend treats it
    as an opaque equality key with the same value-space it already saw).
    """
    sessions_dir = tmp_path / "orbital" / "sessions"
    sessions_dir.mkdir(parents=True)
    legacy_stem = "legacy_project_deadbeef"
    legacy_path = sessions_dir / f"{legacy_stem}.jsonl"
    legacy_messages = [
        {"role": "user", "content": "old msg", "session_id": legacy_stem,
         "timestamp": "2025-01-01T00:00:00+00:00"},
        {"role": "assistant", "content": "old reply", "session_id": legacy_stem,
         "timestamp": "2025-01-01T00:00:01+00:00"},
    ]
    legacy_path.write_text(
        "\n".join(json.dumps(m) for m in legacy_messages) + "\n",
        encoding="utf-8",
    )

    # No migration runs: just load the file directly.
    session = Session.load(str(legacy_path))

    # Compat-layer fallback: both identifiers come from the filename stem.
    assert session.session_uuid == legacy_stem
    assert session.session_id == legacy_stem

    # The historical messages survive verbatim. The ``session_id`` field on
    # each old message keeps its original (F2-shaped) value — the frontend
    # treats it as opaque, so this remains correct.
    messages = session.get_messages()
    assert len(messages) == 2
    for m in messages:
        assert m["session_id"] == legacy_stem


# ---------------------------------------------------------------------------
# Invariant 4: ``/new-session`` rotates F1 (new chat thread)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_new_session_mints_fresh_f1_and_f2(tmp_path, monkeypatch):
    """new_session is pure-create: it mints a fresh F1 chat id (``sess_…``) AND
    a fresh F2 storage stem, returns both, and takes NO action on any existing
    session — no rotation, no handle swap, no termination.
    """
    from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle

    # Minimal AgentManager wired with mocked collaborators; we exercise
    # ``new_session`` end-to-end against an existing handle.
    project_store = MagicMock()
    project_store.get_project.return_value = {
        "project_id": "p1", "name": "Demo", "workspace": str(tmp_path),
    }
    ws_mgr = MagicMock()
    setup_engine = None

    sub_agent_manager = MagicMock()
    sub_agent_manager.stop_all = AsyncMock()
    mgr = AgentManager(
        project_store=project_store,
        ws_manager=ws_mgr,
        sub_agent_manager=sub_agent_manager,
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
        platform_provider=None,
        registry=None,
        setup_engine=setup_engine,
        settings_store=None,
        credential_store=None,
        browser_manager=None,
        user_credential_store=None,
        trigger_manager=None,
        provider_registry=MagicMock(),
    )

    # Seed a project handle holding an existing session with explicit F1/F2.
    old_session = Session.new(
        "demo_oldstem1",
        str(tmp_path),
        session_id="default",  # F1 = the default chat id
    )
    handle = ProjectHandle(
        session=old_session,
        loop=MagicMock(),
        provider=MagicMock(),
        registry=MagicMock(),
        context_manager=MagicMock(),
        interceptor=MagicMock(),
        config_snapshot={
            "workspace": str(tmp_path),
            "provider": "custom",
            "model": "gpt-4",
            "sdk": "openai",
            "fallback_models": [],
        },
        task=None,
    )
    # ``_handles`` is keyed by SessionKey == (project_id, session_id) per
    # the multi-session foundation (PR #22). The seeded handle holds the
    # default F1 session ("default"), so we key it accordingly.
    mgr._handles[("p1", "default")] = handle

    result = await mgr.new_session("p1")

    assert result["status"] == "ok"
    # Seam 3 / decision D2: new_session mints the UUID only — session_id IS the
    # uuid (the 'sess_' F1 mint is retired). Distinct from the seeded session.
    assert result["session_uuid"]
    assert result["session_id"] == result["session_uuid"]
    assert not result["session_id"].startswith("sess_")
    assert result["session_id"] != "default"
    assert result["session_uuid"] != "demo_oldstem1"

    # Pure-create takes no action on the existing default session — its handle
    # and Session object are untouched (no rotation, no swap).
    assert mgr._handles[("p1", "default")].session is old_session
    assert old_session.session_id == "default"


# ---------------------------------------------------------------------------
# Invariant 5: API surface accepts ``session_id`` on /agents/start and /inject
# ---------------------------------------------------------------------------

def test_api_surface_exposes_session_id_on_start_and_inject():
    """``StartAgentRequest`` and ``InjectRequest`` must accept an optional
    ``session_id`` (F1). Omitting it means "use ``DEFAULT_SESSION_ID``" per
    dispatch §4.2.5 — i.e. preserve backward compatibility with callers that
    do not know about the F1/F2 split.

    This is the API-surface counterpart of the on-disk Option B compat
    layer: the API can be addressed by old clients (no session_id) and new
    clients (explicit session_id) identically, without any wire-format
    migration.
    """
    # Both request models must accept the field.
    req = agents_v2.StartAgentRequest(project_id="p1")
    assert req.session_id is None, "omitting session_id must yield None"

    req_with_id = agents_v2.StartAgentRequest(
        project_id="p1", session_id="sess_explicit",
    )
    assert req_with_id.session_id == "sess_explicit"

    inj = agents_v2.InjectRequest(content="hello")
    assert inj.session_id is None

    inj_with_id = agents_v2.InjectRequest(
        content="hello", session_id="sess_explicit",
    )
    assert inj_with_id.session_id == "sess_explicit"
    # Seam 3 / D1: DEFAULT_SESSION_ID is retired; "no session specified" is None,
    # resolved per caller class (see test_d1_resolver_no_default.py).


# ---------------------------------------------------------------------------
# Bonus: Session.new defaults session_id to session_uuid when omitted
# ---------------------------------------------------------------------------

def test_session_new_defaults_f1_to_f2_when_session_id_omitted(tmp_path):
    """Backward compatibility for the in-process ``Session.new`` API: callers
    that pass only ``session_uuid`` (the old positional ``session_id``
    parameter) keep working — the F1 chat id defaults to the F2 stem.
    """
    session = Session.new("legacy_stem", str(tmp_path))
    assert session.session_uuid == "legacy_stem"
    assert session.session_id == "legacy_stem", (
        "When session_id is omitted, F1 must fall back to the F2 stem for "
        "backward compatibility with pre-rename callers."
    )
