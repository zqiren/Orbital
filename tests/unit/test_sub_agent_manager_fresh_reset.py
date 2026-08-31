# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for SubAgentManager.send(fresh=...) — BACKLOG 005 §4d reset behavior.

The load-bearing part of §4d is in ``send``: a fresh reset must actually
re-spawn. A handle that is already running (live adapter in the session slate)
would otherwise keep its in-memory provider session and the reset would be a
silent no-op. So ``fresh=True`` tears the live adapter down first, then the
spawn-on-demand block re-spawns with ``fresh=True`` (no resume, new transcript).
The default path keeps reusing the live adapter — no teardown, no respawn.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
from agent_os.daemon_v2.models import make_session_key


class _FakeTransport:
    def __init__(self):
        self.dispatched = []

    async def dispatch(self, message):
        self.dispatched.append(message)


class _FakeAdapter:
    def __init__(self, resume_status=None):
        self._transport = _FakeTransport()
        self._idle = True
        self._resume_status = resume_status


def _mgr():
    # process_manager is only touched via _ws.broadcast on the blocking path;
    # the non-blocking dispatch path used here never reaches it.
    return SubAgentManager(process_manager=MagicMock())


def test_fresh_send_stops_live_adapter_then_respawns_fresh():
    mgr = _mgr()
    sk = make_session_key("proj", "sess")
    live = _FakeAdapter(resume_status=("resumed", None))
    mgr._adapters[sk] = {"researcher": live}

    calls = {"stop": [], "start": []}

    async def fake_stop(project_id, handle, *, session_id=None):
        calls["stop"].append((project_id, handle, session_id))
        mgr._adapters.get(make_session_key(project_id, session_id), {}).pop(handle, None)
        return f"Stopped {handle}"

    async def fake_start(project_id, handle, depth=0, *, session_id=None,
                         announce=True, fresh=False, pinned=False):
        calls["start"].append({"handle": handle, "fresh": fresh, "announce": announce})
        skk = make_session_key(project_id, session_id)
        mgr._adapters.setdefault(skk, {})[handle] = _FakeAdapter(
            resume_status=("fresh", "explicit_reset"))
        return "Started researcher (fresh session — reset requested ...)"

    mgr.stop = fake_stop
    mgr.start = fake_start

    result = asyncio.run(mgr.send("proj", "researcher", "new task",
                                  session_id="sess", fresh=True))

    # The live adapter was torn down before the respawn.
    assert calls["stop"] == [("proj", "researcher", "sess")]
    # The respawn carried fresh=True (so resume is skipped + new transcript).
    assert calls["start"] and calls["start"][0]["fresh"] is True
    # The NEW (fresh) adapter received the dispatch, not the old one.
    new_adapter = mgr._adapters[sk]["researcher"]
    assert new_adapter is not live
    assert new_adapter._transport.dispatched == ["new task"]
    assert live._transport.dispatched == []
    # The reset clause is surfaced to the management agent.
    assert "reset requested" in result


def test_default_send_reuses_live_adapter_no_stop():
    mgr = _mgr()
    sk = make_session_key("proj", "sess")
    live = _FakeAdapter(resume_status=("resumed", None))
    mgr._adapters[sk] = {"researcher": live}

    calls = {"stop": 0, "start": 0}

    async def fake_stop(*a, **k):
        calls["stop"] += 1

    async def fake_start(*a, **k):
        calls["start"] += 1
        return "Started"

    mgr.stop = fake_stop
    mgr.start = fake_start

    asyncio.run(mgr.send("proj", "researcher", "continue the work",
                         session_id="sess"))

    # Default (resume) path reuses the live adapter — never stops/respawns.
    assert calls["stop"] == 0
    assert calls["start"] == 0
    assert live._transport.dispatched == ["continue the work"]


# ---------------------------------------------------------------------------
# §4d resume-SKIP branch inside _start_from_registry (the decision itself, not
# the transcript-mint side effect): fresh=True must NOT consult
# _determine_resume and must hand the transport NO resume session id.
# ---------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402


def _registry_mgr(workspace=""):
    """A SubAgentManager wired just far enough to reach _start_from_registry's
    resume decision + transport build. workspace="" deliberately skips the
    system-prompt / claudemd / transcript blocks so the test isolates the
    resume-skip branch."""
    mgr = SubAgentManager(process_manager=MagicMock())
    manifest = SimpleNamespace(
        name="Researcher", slug="researcher",
        runtime=SimpleNamespace(
            adapter="cli", mode="pipe", transport="sdk",
            prompt_flag="-p", resume_flag="--resume", session_id_pattern=None),
    )
    mgr._registry = MagicMock()
    mgr._registry.get.return_value = manifest
    mgr._setup_engine = MagicMock()
    mgr._setup_engine.get_adapter_config.return_value = {
        "command": "claude", "workspace": workspace, "args": [], "env": {},
        "approval_patterns": [], "network_domains": [],
    }
    mgr._project_store = MagicMock()
    mgr._project_store.get_project.return_value = {"workspace": workspace}
    mgr._platform_provider = None
    return mgr


def _instrument(mgr, *, determine_returns):
    """Spy on _determine_resume (record calls) and capture the resume_record
    handed to _resolve_transport, then short-circuit by raising ValueError so
    the method returns before adapter/process-manager wiring runs."""
    state = {"determine_calls": [], "resume_record": "UNSET"}

    def _spy_determine(workspace, project_id, handle, session_id):
        state["determine_calls"].append((project_id, handle, session_id))
        return determine_returns

    def _capture_resolve(manifest, config_dict, autonomy=None,
                         system_prompt=None, resume_record=None):
        state["resume_record"] = resume_record
        # Bail out cleanly — the method catches ValueError from transport
        # resolution and returns, so nothing past this point executes.
        raise ValueError("test-sentinel")

    mgr._determine_resume = _spy_determine
    mgr._resolve_transport = _capture_resolve
    return state


def test_fresh_start_skips_determine_resume_and_passes_no_resume_id():
    mgr = _registry_mgr()
    # If _determine_resume WERE consulted it would yield a live resume record.
    state = _instrument(mgr, determine_returns=(
        {"session_id": "OLD-SESSION"}, "resumed", None))

    asyncio.run(mgr._start_from_registry(
        "proj", "researcher", session_id="sess", fresh=True))

    # The resume decision was force-skipped — _determine_resume never ran.
    assert state["determine_calls"] == []
    # And the transport was handed NO resume record (so no provider resume id).
    assert state["resume_record"] is None


def test_default_start_consults_determine_resume_and_forwards_record():
    mgr = _registry_mgr()
    live = ({"session_id": "OLD-SESSION"}, "resumed", None)
    state = _instrument(mgr, determine_returns=live)

    asyncio.run(mgr._start_from_registry(
        "proj", "researcher", session_id="sess"))  # fresh defaults False

    # Default path consults the resume source once...
    assert state["determine_calls"] == [("proj", "researcher", "sess")]
    # ...and forwards the live record to the transport (so resume happens).
    assert state["resume_record"] == {"session_id": "OLD-SESSION"}


def test_explicit_reset_clause_text():
    """fresh=True yields ('fresh', 'explicit_reset'); the clause must read as a
    deliberate reset, never as a first spawn or a failed resume."""
    clause = SubAgentManager._format_resume_clause("fresh", "explicit_reset")
    assert "reset requested" in clause
    assert "first spawn" not in clause
    assert "could not be resumed" not in clause
