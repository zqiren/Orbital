# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 156 phase B: Claude Code forks natively at the edited message.

Every dispatch to Claude Code carries a native prompt uuid derived from its
dispatch id; the fork route turns the first dispatch after the edit into a
cut (``parentUuid`` of that prompt in the parent's native log); the fork's
first spawn resumes the PARENT's native session with ``fork_session`` +
``resume_session_at`` (a new native id, the parent's file untouched); the
parent's live process is never killed; a fork that cannot start retries
once fresh, with the honest clause and the recap.
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os import telemetry
from agent_os.agent.session import Session
from agent_os.agent.transports import sdk_transport as st
from agent_os.agent.transports.sdk_transport import (
    HAS_SDK, SDKTransport, prompt_uuid_for,
)
from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

pytestmark = pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")

PID = "proj156b"
PARENT = "proj156b_aaaa1111"
NATIVE = "11111111-aaaa-4bbb-8ccc-000000000001"


def _manifest(transport="sdk"):
    return SimpleNamespace(
        name="Claude Code", slug="claude-code",
        runtime=SimpleNamespace(
            adapter="cli", mode="pipe", transport=transport, prompt_flag="-p",
            resume_flag="--resume", session_id_pattern=None,
            config_template=None, config_flag=None),
    )


def _sam(manifests=None):
    sam = SubAgentManager(process_manager=MagicMock())
    manifests = manifests or {"claude-code": _manifest(), "codex": _manifest("codex-appserver")}
    sam._registry = MagicMock()
    sam._registry.get.side_effect = lambda h: manifests.get(h)
    return sam


def _native_log(store: str, ws: str, entries: list[dict]) -> str:
    d = os.path.join(store, "projects", SDKTransport.claude_project_slug(ws))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{NATIVE}.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    return path


def _marker(dispatch_id, handle="claude-code", ts="T"):
    return {"role": "system", "content": f"[Sub-agent] {handle} dispatched",
            "timestamp": ts, "_meta": {"dispatch_id": dispatch_id, "handle": handle,
                                       "transcript_path": "/x/t.jsonl"}}


@pytest.fixture
def store(tmp_path, monkeypatch):
    root = str(tmp_path / "claude-store")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", root)
    return root


# ---------------------------------------------------------------------------
# 1. The stamped prompt
# ---------------------------------------------------------------------------


class _StubClient:
    instances: list = []
    fail_fork = False

    def __init__(self, options):
        self.options = options
        self.queries: list = []
        _StubClient.instances.append(self)

    async def connect(self):
        if _StubClient.fail_fork and getattr(self.options, "fork_session", False):
            raise RuntimeError("No message found with message.uuid of: x")

    async def query(self, prompt):
        if isinstance(prompt, str):
            self.queries.append(prompt)
        else:
            self.queries.append([m async for m in prompt])


@pytest.fixture
def stub_client(monkeypatch):
    _StubClient.instances = []
    _StubClient.fail_fork = False
    monkeypatch.setattr(st, "ClaudeSDKClient", _StubClient)
    monkeypatch.setattr(SDKTransport, "_capture_process_handle", lambda self: None)
    monkeypatch.setattr(SDKTransport, "_ensure_reader", lambda self: None)
    return _StubClient


def test_prompt_uuid_is_a_fixed_uuid5_of_the_dispatch_id():
    import uuid
    a = prompt_uuid_for("sess:abcd1234")
    assert a == prompt_uuid_for("sess:abcd1234") != prompt_uuid_for("sess:abcd1235")
    assert uuid.UUID(a).version == 5
    assert a == str(uuid.uuid5(st.ORBITAL_PROMPT_NS, "sess:abcd1234"))


@pytest.mark.asyncio
async def test_dispatch_with_a_prompt_uuid_streams_one_stamped_user_message(stub_client):
    t = SDKTransport()
    await t.start("claude", [], "/tmp", {})
    await t.dispatch("hello", prompt_uuid="u-1")
    await t.dispatch("plain")
    q = stub_client.instances[0].queries
    assert q[0] == [{"type": "user", "message": {"role": "user", "content": "hello"},
                     "parent_tool_use_id": None, "uuid": "u-1"}]
    assert q[1] == "plain"


@pytest.mark.asyncio
async def test_every_daemon_dispatch_to_claude_code_is_stamped(stub_client, monkeypatch):
    monkeypatch.setattr(telemetry, "emit", lambda *a, **k: None)
    sam = _sam()
    t = SDKTransport()
    await t.start("claude", [], "/tmp", {})
    adapter = SimpleNamespace(_transport=t, _idle=True)
    await sam._dispatch_async(adapter, PID, "claude-code", "do it",
                              session_id="s", dispatch_id="s:0001")
    sent = stub_client.instances[0].queries[0]
    assert sent[0]["uuid"] == prompt_uuid_for("s:0001")
    assert sent[0]["message"]["content"] == "do it"


@pytest.mark.asyncio
async def test_other_transports_keep_the_plain_dispatch(monkeypatch):
    monkeypatch.setattr(telemetry, "emit", lambda *a, **k: None)

    class _Codex:
        def __init__(self):
            self.calls = []

        async def dispatch(self, message):
            self.calls.append(message)

    codex = _Codex()
    await _sam()._dispatch_async(SimpleNamespace(_transport=codex, _idle=True),
                                 PID, "codex", "go", session_id="s", dispatch_id="s:1")
    assert codex.calls == ["go"]


# ---------------------------------------------------------------------------
# 2. Fork points
# ---------------------------------------------------------------------------


def _native_entries(*dispatch_ids):
    """A native log: each stamped prompt followed by two entries of its turn."""
    out, parent = [{"type": "session-owner"}], None
    for i, did in enumerate(dispatch_ids):
        pu = prompt_uuid_for(did)
        out.append({"type": "user", "uuid": pu, "parentUuid": parent})
        out.append({"type": "assistant", "uuid": f"a-{i}", "parentUuid": pu})
        out.append({"type": "attachment", "uuid": f"t-{i}", "parentUuid": f"a-{i}"})
        parent = f"t-{i}"
    return out


def test_fork_point_is_the_parent_of_the_first_discarded_prompt(tmp_path):
    path = str(tmp_path / "n.jsonl")
    with open(path, "w") as fh:
        for e in _native_entries("d1", "d2", "d3"):
            fh.write(json.dumps(e) + "\n")
    assert SDKTransport.fork_point(path, prompt_uuid_for("d2")) == (True, "t-0")
    assert SDKTransport.fork_point(path, prompt_uuid_for("d1")) == (True, None)
    assert SDKTransport.fork_point(path, prompt_uuid_for("zz")) == (False, None)
    assert SDKTransport.fork_point(str(tmp_path / "missing"), "x") == (False, None)


def test_claude_fork_rows(tmp_path, store):
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    _native_log(store, ws, _native_entries("d1", "d2"))
    sam = _sam()
    threads = {"claude-code": {"session_id": NATIVE, "proc_pid": 1},
               "codex": {"session_id": "thr-1"}}
    # The first dispatch after the edit is d2: cut at the end of d1's turn.
    after = [_marker("other", handle="codex"), _marker("d2"), _marker("d3")]
    rows = sam.claude_fork_rows(ws, threads, after)
    assert rows == [{"role": "meta", "event": "sub_agent_fork", "handle": "claude-code",
                     "from_session_id": NATIVE, "at_uuid": "t-0"}]
    # No dispatch after the edit: the whole native thread is kept.
    assert sam.claude_fork_rows(ws, threads, [])[0]["at_uuid"] is None
    # A prompt sent before stamping: no row (recap path).
    assert sam.claude_fork_rows(ws, threads, [_marker("unstamped")]) == []
    # The first discarded prompt opened the native session: nothing to keep.
    assert sam.claude_fork_rows(ws, threads, [_marker("d1")]) == []
    # The parent's native log is gone: no row.
    assert sam.claude_fork_rows(ws, {"claude-code": {"session_id": "gone"}}, []) == []


@pytest.mark.asyncio
async def test_fork_route_writes_fork_rows_and_never_a_thread_record(
        tmp_path, store, monkeypatch):
    monkeypatch.setattr(telemetry, "emit", lambda *a, **k: None)
    monkeypatch.setattr(telemetry, "latch", lambda *a, **k: None)
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    native = _native_log(store, ws, _native_entries(f"{PARENT}:1", f"{PARENT}:2"))
    native_sha = open(native, "rb").read()
    s = Session.new(PARENT, ws, session_id=PARENT)
    for i in (1, 2):
        ts = f"2026-11-09T11:0{i}:00.000001+00:00"
        s.append({"role": "user", "content": f"m{i}", "timestamp": ts,
                  "target": "claude-code", "source": "user"})
        s.append(_marker(f"{PARENT}:{i}", ts=ts.replace(":00.", ":01.")))
        s.set_sub_agent_thread("claude-code", session_id=NATIVE, model="opus",
                               proc_pid=os.getpid(), proc_create_time=1.0)
    sam = _sam()
    mgr = AgentManager(project_store=MagicMock(), ws_manager=MagicMock(),
                       sub_agent_manager=sam, activity_translator=MagicMock(),
                       process_manager=MagicMock())
    mgr._project_store.get_project.return_value = {"name": "p", "workspace": ws}
    mgr._platform_provider = None
    sam.list_active = MagicMock(return_value=[])

    result = await mgr.fork_session(PID, PARENT, at="2026-11-09T11:02:00.000001+00:00")

    rows = [json.loads(line) for line in open(os.path.join(
        ws, "orbital", "sessions", f"{result['session_id']}.jsonl"))]
    metas = [r for r in rows if r["role"] == "meta"]
    assert [m["event"] for m in metas] == ["session_start", "sub_agent_fork"]
    assert metas[1] == {"role": "meta", "event": "sub_agent_fork", "handle": "claude-code",
                        "from_session_id": NATIVE, "at_uuid": "t-0"}
    assert "proc_pid" not in json.dumps(rows)
    assert open(native, "rb").read() == native_sha


# ---------------------------------------------------------------------------
# 3. The spawn decision
# ---------------------------------------------------------------------------


def _fork_session(tmp_path, *, thread=None):
    s = Session.new("proj156b_ffff0000", str(tmp_path), session_id="proj156b_ffff0000")
    s.sub_agent_forks = {"claude-code": {"from_session_id": NATIVE, "at_uuid": "t-0"}}
    if thread:
        s.set_sub_agent_thread("claude-code", session_id=thread, model="m")
    return s


def test_determine_resume_forks_the_parent_session(tmp_path, monkeypatch):
    sam = _sam()
    sam._session_resolver = lambda pid, sid: _fork_session(tmp_path)
    monkeypatch.setattr(SDKTransport, "resume_source_exists",
                        staticmethod(lambda ws, sid, projects_root=None: sid == NATIVE))
    rec, status, reason = sam._determine_resume(str(tmp_path), PID, "claude-code", "f")
    assert rec == {"session_id": NATIVE, "fork": True, "fork_at": "t-0"}
    assert (status, reason) == ("resumed", None)
    assert "proc_pid" not in rec


def test_determine_resume_fork_with_the_parent_log_gone_is_fresh(tmp_path, monkeypatch):
    sam = _sam()
    sam._session_resolver = lambda pid, sid: _fork_session(tmp_path)
    monkeypatch.setattr(SDKTransport, "resume_source_exists",
                        staticmethod(lambda ws, sid, projects_root=None: False))
    assert sam._determine_resume(str(tmp_path), PID, "claude-code", "f") == (
        None, "fresh", "resume_failed")


def test_the_forks_own_thread_record_wins_over_the_fork_row(tmp_path, monkeypatch):
    sam = _sam()
    sam._session_resolver = lambda pid, sid: _fork_session(tmp_path, thread="own-id")
    monkeypatch.setattr(SDKTransport, "resume_source_exists",
                        staticmethod(lambda ws, sid, projects_root=None: True))
    rec, status, _ = sam._determine_resume(str(tmp_path), PID, "claude-code", "f")
    assert rec["session_id"] == "own-id" and "fork" not in rec
    assert status == "resumed"


def test_part_f_never_kills_the_parents_live_process(tmp_path, monkeypatch):
    """The parent's record names a LIVE pid (this test process); the fork's
    record carries none, so the live-attachment backstop is a no-op."""
    sam = _sam()
    sam._session_resolver = lambda pid, sid: _fork_session(tmp_path)
    monkeypatch.setattr(SDKTransport, "resume_source_exists",
                        staticmethod(lambda ws, sid, projects_root=None: True))
    reaped = []
    monkeypatch.setattr(sam, "_reap_stragglers", lambda v, g: reaped.append(v) or True)
    rec, status, _ = sam._determine_resume(str(tmp_path), PID, "claude-code", "f")
    assert status == "resumed" and reaped == []
    assert sam._ensure_no_live_attachment(PID, "claude-code", rec) is True and reaped == []


@pytest.mark.asyncio
async def test_fork_record_reaches_claude_agent_options(stub_client):
    sam = _sam()
    t = sam._resolve_transport(_manifest(), {"args": []},
                               resume_record={"session_id": NATIVE, "fork": True,
                                              "fork_at": "t-0"})
    await t.start("claude", [], "/tmp", {})
    opts = stub_client.instances[-1].options
    assert (opts.resume, opts.fork_session, opts.resume_session_at) == (NATIVE, True, "t-0")
    assert opts.resume_drops_turn is None
    # A whole-thread fork (no cut) forks without resume_session_at.
    t2 = sam._resolve_transport(_manifest(), {"args": []},
                                resume_record={"session_id": NATIVE, "fork": True,
                                               "fork_at": None})
    await t2.start("claude", [], "/tmp", {})
    opts = stub_client.instances[-1].options
    assert (opts.fork_session, opts.resume_session_at) == (True, None)
    # An ordinary resume never forks.
    t3 = sam._resolve_transport(_manifest(), {"args": []},
                                resume_record={"session_id": NATIVE})
    await t3.start("claude", [], "/tmp", {})
    assert stub_client.instances[-1].options.fork_session is False


@pytest.mark.asyncio
async def test_an_sdk_without_fork_options_refuses_instead_of_resuming_the_parent(
        stub_client, monkeypatch):
    @__import__("dataclasses").dataclass
    class _OldOptions:
        cwd: object = None
        permission_mode: object = None
        can_use_tool: object = None
        cli_path: object = None
        env: object = None
        max_buffer_size: object = None
        resume: object = None

    monkeypatch.setattr(st, "ClaudeAgentOptions", _OldOptions)
    t = SDKTransport(resume_session_id=NATIVE, fork_session=True, resume_session_at="t-0")
    with pytest.raises(RuntimeError, match="fork_session"):
        await t.start("claude", [], "/tmp", {})
    assert stub_client.instances == []


# ---------------------------------------------------------------------------
# 4. Start failure → one fresh retry
# ---------------------------------------------------------------------------


def _spawn_sam(determine):
    sam = _sam()
    sam._setup_engine = MagicMock()
    sam._setup_engine.get_adapter_config.return_value = {
        "command": "claude", "workspace": "", "args": [], "env": {},
        "approval_patterns": [], "network_domains": [],
    }
    sam._project_store = MagicMock()
    sam._project_store.get_project.return_value = {"workspace": ""}
    sam._platform_provider = None
    sam._process_manager.start = AsyncMock()
    sam._determine_resume = determine
    return sam


@pytest.mark.asyncio
async def test_a_fork_that_cannot_start_retries_once_fresh(stub_client):
    stub_client.fail_fork = True
    fork = ({"session_id": NATIVE, "fork": True, "fork_at": "t-0"}, "resumed", None)
    sam = _spawn_sam(lambda *a: fork)
    result = await sam._start_from_registry(PID, "claude-code", session_id="f")
    assert [getattr(c.options, "fork_session", False) for c in stub_client.instances] == [
        True, False]
    assert stub_client.instances[1].options.resume is None
    adapter = sam._adapters[(PID, "f")]["claude-code"]
    assert adapter._resume_status == ("fresh", "resume_failed")
    assert "could not be resumed" in result


@pytest.mark.asyncio
async def test_an_ordinary_start_failure_is_not_retried(stub_client, monkeypatch):
    calls = []

    async def boom(self):
        calls.append(1)
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(_StubClient, "connect", boom)
    sam = _spawn_sam(lambda *a: ({"session_id": NATIVE}, "resumed", None))
    result = await sam._start_from_registry(PID, "claude-code", session_id="f")
    assert result.startswith("Error: adapter start failed")
    assert calls == [1]
