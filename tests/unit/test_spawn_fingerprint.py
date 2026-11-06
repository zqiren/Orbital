# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A live worker must not outlive the settings it was spawned with.

Observed 2026-10-26: the user saved a new DeepSeek key for dsh, and the next
dispatch still failed with the OLD key (``****0e18 is invalid``) — the credential
route wrote the keychain, but ``send()`` reused the dsh process spawned ten
minutes earlier, whose environment still carried the old value.

The invariant: every spawn records a fingerprint of what it was launched with
(binary, argv, injected env, the agent's Settings params). A dispatch to an
IDLE worker whose fingerprint no longer matches tears it down so
spawn-on-demand rebuilds it (resuming the thread). A worker mid-turn is never
interrupted: its queued prompts drain through a rebuild once the turn closes.
"""

from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent_os.agents.setup_engine import spawn_fingerprint
from agent_os.daemon_v2.models import make_session_key
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager, _QueuedPrompt


_BASE = {"command": "/bin/dsh", "args": ["--profile", "acp"],
         "env": {"DEEPSEEK_API_KEY": "sk-old", "PATH": "/usr/bin"}}


def _cfg(**over):
    cfg = {k: (dict(v) if isinstance(v, dict) else list(v)) for k, v in _BASE.items()}
    cfg.update(over)
    return cfg


# ---------------------------------------------------------------------------
# The fingerprint itself
# ---------------------------------------------------------------------------

def test_fingerprint_is_stable_for_identical_spawns():
    assert spawn_fingerprint(_cfg(), {"model": "a"}) == spawn_fingerprint(_cfg(), {"model": "a"})


def test_fingerprint_ignores_env_ordering():
    reordered = _cfg(env={"PATH": "/usr/bin", "DEEPSEEK_API_KEY": "sk-old"})
    assert spawn_fingerprint(reordered, {}) == spawn_fingerprint(_cfg(), {})


def test_fingerprint_changes_with_a_credential():
    rotated = _cfg(env={"DEEPSEEK_API_KEY": "sk-new", "PATH": "/usr/bin"})
    assert spawn_fingerprint(rotated, {}) != spawn_fingerprint(_cfg(), {})


def test_fingerprint_changes_with_binary_args_and_params():
    base = spawn_fingerprint(_cfg(), {"model": "a"})
    assert spawn_fingerprint(_cfg(command="/other/dsh"), {"model": "a"}) != base
    assert spawn_fingerprint(_cfg(args=["--profile", "x"]), {"model": "a"}) != base
    # UI-only params (dsh's model never reaches argv) still count.
    assert spawn_fingerprint(_cfg(), {"model": "b"}) != base


def test_fingerprint_never_contains_the_secret():
    assert "sk-old" not in spawn_fingerprint(_cfg(), {})


# ---------------------------------------------------------------------------
# send(): idle stale worker is rebuilt, everything else is reused
# ---------------------------------------------------------------------------

class _FakeTransport:
    def __init__(self):
        self.dispatched = []

    async def dispatch(self, message):
        self.dispatched.append(message)


class _FakeAdapter:
    def __init__(self, fingerprint=None):
        self._transport = _FakeTransport()
        self._idle = True
        self._resume_status = ("resumed", None)
        if fingerprint is not None:
            self._spawn_fingerprint = fingerprint

    def is_alive(self):
        return True


def _mgr(current_fingerprint="fp-current"):
    mgr = SubAgentManager(process_manager=MagicMock())
    mgr._setup_engine = MagicMock()
    if isinstance(current_fingerprint, Exception):
        mgr._setup_engine.spawn_fingerprint.side_effect = current_fingerprint
    else:
        mgr._setup_engine.spawn_fingerprint.return_value = current_fingerprint
    mgr._project_store = MagicMock()
    mgr._project_store.get_project.return_value = {"workspace": "/ws"}
    calls = {"stop": [], "start": []}

    async def fake_stop(project_id, handle, *, session_id=None):
        calls["stop"].append(handle)
        mgr._adapters.get(make_session_key(project_id, session_id), {}).pop(handle, None)
        return f"Stopped {handle}"

    async def fake_start(project_id, handle, depth=0, *, session_id=None,
                         announce=True, fresh=False, pinned=False, initiator="management_agent"):
        calls["start"].append({"handle": handle, "fresh": fresh})
        mgr._adapters.setdefault(make_session_key(project_id, session_id), {})[handle] = (
            _FakeAdapter("fp-current"))
        return f"Started {handle}"

    mgr.stop = fake_stop
    mgr.start = fake_start
    return mgr, calls


def _install(mgr, adapter, handle="dsh"):
    mgr._adapters[make_session_key("proj", "sess")] = {handle: adapter}


def test_idle_worker_with_stale_fingerprint_is_rebuilt_before_dispatch():
    mgr, calls = _mgr("fp-current")
    old = _FakeAdapter("fp-old")
    _install(mgr, old)

    asyncio.run(mgr.send("proj", "dsh", "task", session_id="sess"))

    assert calls["stop"] == ["dsh"]
    # A settings rebuild resumes the thread — never a fresh reset.
    assert calls["start"] == [{"handle": "dsh", "fresh": False}]
    new = mgr._adapters[make_session_key("proj", "sess")]["dsh"]
    assert new is not old
    assert new._transport.dispatched == ["task"]
    assert old._transport.dispatched == []
    mgr._setup_engine.spawn_fingerprint.assert_called_with("dsh", "/ws")


def test_idle_worker_with_matching_fingerprint_is_reused():
    mgr, calls = _mgr("fp-current")
    live = _FakeAdapter("fp-current")
    _install(mgr, live)

    asyncio.run(mgr.send("proj", "dsh", "task", session_id="sess"))

    assert calls["stop"] == [] and calls["start"] == []
    assert live._transport.dispatched == ["task"]


def test_worker_without_a_recorded_fingerprint_is_reused():
    mgr, calls = _mgr("fp-current")
    legacy = _FakeAdapter()  # spawned by a path that records none
    _install(mgr, legacy)

    asyncio.run(mgr.send("proj", "dsh", "task", session_id="sess"))

    assert calls["stop"] == []
    assert legacy._transport.dispatched == ["task"]


def test_unresolvable_current_fingerprint_keeps_the_live_worker():
    # The binary vanished or the config can't be built right now: the running
    # process is the best thing we have — never tear it down on a guess.
    for current in (None, ValueError("Agent 'dsh' not installed")):
        mgr, calls = _mgr(current)
        live = _FakeAdapter("fp-old")
        _install(mgr, live)

        asyncio.run(mgr.send("proj", "dsh", "task", session_id="sess"))

        assert calls["stop"] == []
        assert live._transport.dispatched == ["task"]


def test_busy_worker_with_stale_fingerprint_is_not_interrupted():
    mgr, calls = _mgr("fp-current")
    busy = _FakeAdapter("fp-old")
    _install(mgr, busy)
    mgr._prompt_active.add(("proj", "sess", "dsh"))

    result = asyncio.run(mgr.send("proj", "dsh", "next", session_id="sess"))

    assert calls["stop"] == []
    assert "queued" in result
    assert [p.message for p in mgr._prompt_queues[("proj", "sess", "dsh")]] == ["next"]


def test_turn_close_with_stale_fingerprint_drains_through_a_rebuild():
    mgr, _ = _mgr("fp-current")
    busy = _FakeAdapter("fp-old")
    _install(mgr, busy)
    key = ("proj", "sess", "dsh")
    mgr._prompt_active.add(key)
    mgr._prompt_queues[key] = deque([_QueuedPrompt(
        message="next", dispatch_id="d1", transcript_path="t", initiator="user_pinned")])
    rebuilt = []

    async def fake_rebuild(project_id, handle, session_id):
        rebuilt.append((project_id, handle, session_id))

    mgr._rebuild_and_drain = fake_rebuild

    async def run():
        await mgr._on_prompt_turn_closed("proj", "dsh", session_id="sess")
        await asyncio.sleep(0)

    asyncio.run(run())

    assert rebuilt == [("proj", "dsh", "sess")]
    # The stale process never received the queued prompt.
    assert busy._transport.dispatched == []


# ---------------------------------------------------------------------------
# _start_from_registry records the fingerprint of what it actually spawned
# ---------------------------------------------------------------------------

def test_registry_spawn_records_its_fingerprint(monkeypatch):
    mgr = SubAgentManager(process_manager=MagicMock())
    manifest = SimpleNamespace(
        name="DeepSeek Harness", slug="dsh",
        runtime=SimpleNamespace(
            adapter="cli", mode="pipe", transport="acp-sdk", config_template=None,
            prompt_flag=None, resume_flag=None, session_id_pattern=None),
    )
    mgr._registry = MagicMock()
    mgr._registry.get.return_value = manifest
    config_dict = {"command": "/bin/dsh", "workspace": "", "args": [], "env": {},
                   "approval_patterns": [], "network_domains": []}
    mgr._setup_engine = MagicMock()
    mgr._setup_engine.get_adapter_config.return_value = config_dict
    mgr._setup_engine.fingerprint_config.return_value = "fp-spawned"
    mgr._project_store = MagicMock()
    mgr._project_store.get_project.return_value = {"workspace": ""}
    mgr._platform_provider = None

    async def no_gate(*a, **k):
        return None

    mgr._pre_dispatch_gate = no_gate
    mgr._determine_resume = lambda *a: (None, "fresh", "first_spawn")
    mgr._resolve_transport = lambda *a, **k: SimpleNamespace(_resume_session_id=None)

    started = []

    class _StubAdapter:
        def __init__(self, **kw):
            pass

        async def start(self, config):
            started.append(config)

        async def stop(self):
            pass

    monkeypatch.setattr("agent_os.daemon_v2.sub_agent_manager.CLIAdapter", _StubAdapter)

    async def run():
        try:
            await mgr._start_from_registry("proj", "dsh", session_id="sess")
        except Exception:
            pass  # transcript/broadcast wiring past registration is not under test

    asyncio.run(run())

    assert started, "adapter.start was never reached"
    adapter = mgr._adapters[make_session_key("proj", "sess")]["dsh"]
    assert adapter._spawn_fingerprint == "fp-spawned"
    mgr._setup_engine.fingerprint_config.assert_called_once_with("dsh", config_dict)
