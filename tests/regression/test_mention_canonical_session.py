# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression (seam 3 / D1): an @mention persists the authored user message to
the project's REAL chat session, never a fabricated ``subagent_<hex>`` log.

Before this fix the @mention inject branch resolved its PERSISTENCE session via
``_get_or_create_session`` -> ``get_session(project_id)`` (no session_id), which
under the seam-3 None-policy ALWAYS misses (``_resolve_session_id(None)`` is
passthrough -> ``(project_id, None)`` matches no concrete-uuid handle). On the
miss it minted ``subagent_{hex}`` and appended the user message there, so the
authored turn went missing from the real session's history and rendered as a
standalone sidebar session. Dispatch, meanwhile, correctly used
``req.session_id`` — so persistence and dispatch DIVERGED.

The fix routes the @mention through the canonical inject resolution exactly once
(``_sid_inject`` -> ``_ensure_chat_session`` — passthrough / disk-hydrate /
canonical mint) and threads the ONE concrete id to persistence + dispatch +
lifecycle, WITHOUT waking the management loop.

Three session-state variants (T1 live, T2 cold-disk, T3 fresh-mint) exercise the
three funnel branches; T4 pins append-before-dispatch ordering; T5 guards that
the mention never auto-wakes / re-dispatches the management agent.

Each test is RED-before-GREEN in isolation. Against PRE-fix code the module-
level ``_sub_agent_sessions`` cache (keyed by the shared PID) bleeds across
tests, so the pre-fix failure mode is run-order-dependent; post-fix the cache is
gone and the suite is deterministic. (T5's red driver is specifically its
canonical-persistence assertion — pre-fix already does not auto-wake, so the
``start_agent`` / empty-handles asserts are a forward regression guard for the
no-re-dispatch constraint.)
"""

from __future__ import annotations

import glob
import json
import os
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.project_paths import ProjectPaths
from agent_os.agent.session import Session
from agent_os.api.routes import agents_v2
from agent_os.api.routes.agents_v2 import InjectRequest
from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.models import make_session_key

PID = "proj_test"
TARGET = "claude-code"


# ---------------------------------------------------------------------------
# Harness — a REAL AgentManager behind the route so the canonical funnel runs.
# ---------------------------------------------------------------------------

def _wire(workspace):
    project_store = MagicMock()
    project_store.get_project.return_value = {
        "project_id": PID, "name": "Proj", "workspace": workspace,
    }
    mgr = AgentManager(
        project_store=project_store, ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(), activity_translator=MagicMock(),
        process_manager=MagicMock(), platform_provider=None,
        registry=MagicMock(), setup_engine=MagicMock(),
        settings_store=None, credential_store=None,
    )
    # Spy: a mention must NEVER auto-start a management loop.
    mgr.start_agent = AsyncMock()

    sub = MagicMock()
    sub.send = AsyncMock(return_value=f"Message sent to {TARGET}")
    sub.get_transcript = MagicMock(return_value=None)
    ws = MagicMock()
    ws.broadcast = MagicMock()
    lifecycle = MagicMock()
    lifecycle.on_message_routed = AsyncMock()

    agents_v2.configure(project_store, mgr, ws, sub, lifecycle_observer=lifecycle)
    return mgr, sub, ws, lifecycle


def _sessions_dir(workspace):
    return ProjectPaths(workspace).sessions_dir


def _read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _authored(messages):
    """Authored @mention records: role=user with the target set."""
    return [m for m in messages if m.get("role") == "user" and m.get("target") == TARGET]


def _subagent_files(workspace):
    return glob.glob(os.path.join(_sessions_dir(workspace), "subagent_*.jsonl"))


def _plant_idle_handle(mgr, workspace, sid):
    """A live (idle, not-running) management session under ``sid``."""
    sess = Session.new(sid, workspace)
    handle = MagicMock()
    handle.session = sess
    handle.task = MagicMock()
    handle.task.done.return_value = True
    mgr._handles[make_session_key(PID, sid)] = handle
    return sess


def _seed_disk_session(workspace, uuid, *, f1=None):
    """A cold session on disk (no live handle) with prior history."""
    f1 = f1 or uuid
    sdir = _sessions_dir(workspace)
    os.makedirs(sdir, exist_ok=True)
    rows = [
        {"role": "meta", "event": "session_start", "session_id": f1,
         "session_uuid": uuid, "provider": "x", "model": "y", "sdk": "z",
         "fallback_models": [], "timestamp": "2026-06-23T00:00:00+00:00"},
        {"role": "user", "content": "earlier note", "session_id": f1,
         "session_uuid": uuid, "timestamp": "2026-06-23T00:00:01+00:00"},
    ]
    with open(os.path.join(sdir, f"{uuid}.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")


# ---------------------------------------------------------------------------
# T1 — @mention into an active, live session (passthrough branch)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_t1_mention_into_live_session(tmp_path):
    ws = str(tmp_path)
    mgr, sub, _ws, _lc = _wire(ws)
    sid = "proj_live_0001"
    _plant_idle_handle(mgr, ws, sid)

    req = InjectRequest(content="hey @claude-code", target=TARGET, session_id=sid)
    await agents_v2.inject_message(PID, req)

    path = os.path.join(_sessions_dir(ws), f"{sid}.jsonl")
    msgs = _read_jsonl(path)
    # (a) authored record exists exactly once in the RESOLVED session
    assert len(_authored(msgs)) == 1
    # (b) no fabricated subagent_*.jsonl minted
    assert _subagent_files(ws) == []
    # (c) dispatch targets the SAME resolved session id persistence used
    assert sub.send.await_args.kwargs.get("session_id") == sid
    # guard: no management loop auto-started
    mgr.start_agent.assert_not_awaited()


# ---------------------------------------------------------------------------
# T2 — @mention into a cold, disk-only session (hydrate branch)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_t2_mention_into_cold_disk_session(tmp_path):
    ws = str(tmp_path)
    mgr, sub, _ws, _lc = _wire(ws)
    uuid = "proj_cold_0002"
    _seed_disk_session(ws, uuid)  # JSONL on disk, no live handle

    req = InjectRequest(content="resume @claude-code", target=TARGET, session_id=uuid)
    await agents_v2.inject_message(PID, req)

    path = os.path.join(_sessions_dir(ws), f"{uuid}.jsonl")
    msgs = _read_jsonl(path)
    assert len(_authored(msgs)) == 1                         # (a) appended once
    assert _subagent_files(ws) == []                         # (b) no fabricated log
    assert sub.send.await_args.kwargs.get("session_id") == uuid  # (c) same session
    # continued, not forked: prior history is still in the same file
    assert any(m.get("content") == "earlier note" for m in msgs)
    mgr.start_agent.assert_not_awaited()


# ---------------------------------------------------------------------------
# T2b — @mention into a disk session addressed by a legacy F1 (slow-path scan)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_t2b_mention_into_disk_session_addressed_by_legacy_f1(tmp_path):
    """The hydrate branch's F1-scan slow path: session_id is a legacy F1 that is
    NOT the JSONL stem. The authored record must still continue the SAME physical
    {uuid} file (not fork), and dispatch echoes the resolved id — locking the
    self-consistent slow-path behavior persist_mention_message inherits from
    _load_session_from_disk."""
    ws = str(tmp_path)
    mgr, sub, _ws, _lc = _wire(ws)
    uuid = "proj_cold_f1b0"
    f1 = "legacy_default"
    _seed_disk_session(ws, uuid, f1=f1)  # stem=uuid; records carry F1=legacy_default

    req = InjectRequest(content="resume @claude-code", target=TARGET, session_id=f1)
    await agents_v2.inject_message(PID, req)

    path = os.path.join(_sessions_dir(ws), f"{uuid}.jsonl")  # the stem file, not {f1}.jsonl
    msgs = _read_jsonl(path)
    assert len(_authored(msgs)) == 1                          # appended once, here
    assert any(m.get("content") == "earlier note" for m in msgs)  # continued, not forked
    assert _subagent_files(ws) == []
    # dispatch echoes the resolved id — the same session the record was persisted to
    assert sub.send.await_args.kwargs.get("session_id") == f1
    mgr.start_agent.assert_not_awaited()


# ---------------------------------------------------------------------------
# T3 — @mention as the first message of a fresh session (canonical-mint branch)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_t3_mention_first_message_mints_canonical_session(tmp_path):
    ws = str(tmp_path)
    mgr, sub, _ws, _lc = _wire(ws)

    req = InjectRequest(content="first @claude-code", target=TARGET)  # session_id omitted
    await agents_v2.inject_message(PID, req)

    files = glob.glob(os.path.join(_sessions_dir(ws), "*.jsonl"))
    assert _subagent_files(ws) == []          # (b) no fabricated subagent_ log
    assert len(files) == 1                     # exactly one canonical session minted
    stem = os.path.splitext(os.path.basename(files[0]))[0]
    # (d) minted id is a CANONICAL {project}_{hex8} uuid, NOT subagent_-prefixed
    assert not stem.startswith("subagent_")
    assert re.match(r".+_[0-9a-f]{8}$", stem), stem
    msgs = _read_jsonl(files[0])
    assert len(_authored(msgs)) == 1                          # (a)
    assert sub.send.await_args.kwargs.get("session_id") == stem  # (c)
    mgr.start_agent.assert_not_awaited()


# ---------------------------------------------------------------------------
# T4 — ordering: authored user record is appended BEFORE dispatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_t4_user_record_persisted_before_dispatch(tmp_path):
    ws = str(tmp_path)
    mgr, sub, _ws, _lc = _wire(ws)
    sid = "proj_order_0001"
    _plant_idle_handle(mgr, ws, sid)

    seen = {}

    async def _send(*args, **kwargs):
        # At dispatch time the authored record must already be in the JSONL.
        path = os.path.join(_sessions_dir(ws), f"{sid}.jsonl")
        seen["present"] = os.path.isfile(path) and bool(_authored(_read_jsonl(path)))
        return f"Message sent to {TARGET}"

    sub.send = AsyncMock(side_effect=_send)

    req = InjectRequest(content="q @claude-code", target=TARGET, session_id=sid)
    await agents_v2.inject_message(PID, req)

    assert seen.get("present") is True


# ---------------------------------------------------------------------------
# T5 — a mention never auto-wakes / re-dispatches the management agent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_t5_mention_does_not_auto_wake_management(tmp_path):
    ws = str(tmp_path)
    mgr, sub, _ws, _lc = _wire(ws)

    req = InjectRequest(content="hello @claude-code", target=TARGET)  # fresh, no session
    await agents_v2.inject_message(PID, req)

    # red-before-green: the authored record landed in a canonical session,
    # never a fabricated subagent_ log
    assert _subagent_files(ws) == []
    files = glob.glob(os.path.join(_sessions_dir(ws), "*.jsonl"))
    assert len(files) == 1 and not os.path.basename(files[0]).startswith("subagent_")
    # guard: no management loop started, no management handle spun up
    mgr.start_agent.assert_not_awaited()
    assert [k for k in mgr._handles if k[0] == PID] == []
