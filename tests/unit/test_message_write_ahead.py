# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bug #59 — the user's message must be on disk BEFORE the start window.

On the auto-start inject path the message used to travel only as
``start_agent(initial_message=…)`` and did not reach disk until
``AgentLoop.run()`` appended it — the last thing ``start_agent`` schedules,
after ~450 lines of setup. Anything that raised in that window, or a supersede
before the task's first execution slice, destroyed the message with no JSONL
row, no session, no error row and no traceback. One was lost for real on
2026-08-15.

The fix has one invariant: **whoever injects, persists.** Exactly one writer
(``agent_os.agent.session.persist_user_row``), called before the start window;
``AgentLoop.run()`` appends nothing and takes no message. These tests pin:

1. all three ``inject_message`` auto-start branches write ahead;
2. the row survives a start that raises, and a start that is cancelled;
3. the failure is surfaced (session system row + classified broadcast);
4. a slot conflict — a rejection, not a failure — writes NOTHING, because the
   route answers it with a pending-inject enqueue that appends the message
   itself later;
5. ``run()`` cannot re-grow its append: no message parameter, no second row.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from unittest.mock import MagicMock

import pytest

from agent_os.agent.loop import AgentLoop
from agent_os.agent.session import Session, persist_user_row
from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.models import AgentConfig
from agent_os.daemon_v2.provider_errors import ProviderConfigError

SESSION_ID = "proj_deadbeef"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def _config(workspace: str) -> AgentConfig:
    return AgentConfig(
        workspace=workspace,
        model="stub-model",
        api_key="k",
        provider="stub-provider",
        sdk="openai",
        project_name="proj",
    )


def _manager(workspace: str) -> AgentManager:
    mgr = AgentManager(
        project_store=MagicMock(),
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )
    mgr._ws.broadcast = MagicMock()
    mgr._project_store.get_project.return_value = {
        "name": "proj", "workspace": workspace,
    }
    mgr._platform_provider = None
    mgr._build_agent_config_from_project = lambda pid: _config(workspace)
    return mgr


def _sessions_dir(workspace: str) -> str:
    return os.path.join(workspace, "orbital", "sessions")


def _rows(workspace: str, session_id: str) -> list[dict]:
    """Every JSONL record for a session, read fresh off disk."""
    path = os.path.join(_sessions_dir(workspace), f"{session_id}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _user_rows(workspace: str, session_id: str) -> list[dict]:
    return [r for r in _rows(workspace, session_id) if r.get("role") == "user"]


def _seed_session_on_disk(workspace: str, session_id: str) -> None:
    """Write a prior turn so ``_load_session_from_disk`` hydrates instead of
    falling through to the fresh-start branch."""
    os.makedirs(_sessions_dir(workspace), exist_ok=True)
    path = os.path.join(_sessions_dir(workspace), f"{session_id}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "role": "meta", "event": "session_start",
            "session_id": session_id, "session_uuid": session_id,
            "origin": "chat", "provider": "stub-provider",
            "model": "stub-model", "sdk": "openai", "fallback_models": [],
            "timestamp": "2026-08-16T00:00:00+00:00",
        }) + "\n")
        f.write(json.dumps({
            "role": "user", "content": "an earlier turn", "source": "user",
            "session_id": session_id, "session_uuid": session_id,
            "timestamp": "2026-08-16T00:00:01+00:00",
        }) + "\n")


class _CapturingStart:
    """Stands in for ``start_agent`` and records what the JSONL looked like at
    the moment it was entered — the whole point of the fix is that the row is
    already there. Optionally raises to model a failure inside the window."""

    def __init__(self, workspace: str, raises: BaseException | None = None):
        self._workspace = workspace
        self._raises = raises
        self.calls: list[dict] = []
        self.rows_at_entry: list[dict] = []

    async def __call__(self, project_id, config, **kwargs):
        self.calls.append(kwargs)
        sid = kwargs.get("session_id")
        self.rows_at_entry = _rows(self._workspace, sid)
        if self._raises is not None:
            raise self._raises


# ---------------------------------------------------------------------------
# 1. Write-ahead from all three auto-start inject branches
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_case3_fresh_start_persists_before_start_agent(tmp_path):
    """No handle, no session on disk (agent_manager.py Case 3) — the row is on
    disk before start_agent is entered, and start_agent is no longer handed an
    initial_message."""
    ws = str(tmp_path)
    mgr = _manager(ws)
    start = _CapturingStart(ws)
    mgr.start_agent = start

    result = await mgr.inject_message("proj", "the lost message",
                                      nonce="n-1", session_id=SESSION_ID)

    assert result == "started"
    # Persisted BEFORE start_agent ran, not after it returned.
    users = [r for r in start.rows_at_entry if r.get("role") == "user"]
    assert [r["content"] for r in users] == ["the lost message"]
    assert users[0]["nonce"] == "n-1"
    # start_agent no longer carries the message; it receives the session that
    # already holds it.
    kwargs = start.calls[0]
    assert "initial_message" not in kwargs
    assert "initial_nonce" not in kwargs
    assert kwargs["session"].session_uuid == SESSION_ID


@pytest.mark.asyncio
async def test_hydrate_from_disk_path_persists_before_start_agent(tmp_path):
    """The hydrate-on-inject branch appends onto the loaded session's JSONL
    before the start window, keeping the prior turn intact."""
    ws = str(tmp_path)
    _seed_session_on_disk(ws, SESSION_ID)
    mgr = _manager(ws)
    start = _CapturingStart(ws)
    mgr.start_agent = start

    result = await mgr.inject_message("proj", "second turn",
                                      nonce="n-2", session_id=SESSION_ID)

    assert result == "started"
    users = [r for r in start.rows_at_entry if r.get("role") == "user"]
    assert [r["content"] for r in users] == ["an earlier turn", "second turn"]
    assert users[-1]["nonce"] == "n-2"
    assert "initial_message" not in start.calls[0]


@pytest.mark.asyncio
async def test_stopped_handle_path_persists_before_start_agent(tmp_path):
    """The third auto-start branch — a handle existed but its session was
    stopped, so it is dropped and a fresh agent started."""
    ws = str(tmp_path)
    mgr = _manager(ws)
    stopped = MagicMock()
    stopped.session.is_stopped.return_value = True
    stopped.session._paused_for_approval = False
    stopped.task = MagicMock()
    stopped.task.done.return_value = True
    stopped.interceptor = None
    mgr._handles[("proj", SESSION_ID)] = stopped
    start = _CapturingStart(ws)
    mgr.start_agent = start

    result = await mgr.inject_message("proj", "after a stop",
                                      nonce="n-3", session_id=SESSION_ID)

    assert result == "started"
    users = [r for r in start.rows_at_entry if r.get("role") == "user"]
    assert [r["content"] for r in users] == ["after a stop"]
    assert users[0]["nonce"] == "n-3"


@pytest.mark.asyncio
async def test_persisted_row_echoes_chat_user_message(tmp_path):
    """The write-ahead still fires the canonical ``chat.user_message`` echo
    (with the nonce) — observers are wired before the write, so the frontend's
    optimistic-bubble dedup keeps working now that the row lands earlier."""
    ws = str(tmp_path)
    mgr = _manager(ws)
    mgr.start_agent = _CapturingStart(ws)
    seen: list[dict] = []
    mgr._activity_translator.on_message.side_effect = (
        lambda msg, pid, session_id=None: seen.append(msg)
    )

    await mgr.inject_message("proj", "echo me", nonce="n-echo",
                             session_id=SESSION_ID)

    assert [m["content"] for m in seen] == ["echo me"]
    assert seen[0]["nonce"] == "n-echo"


# ---------------------------------------------------------------------------
# 2 + 3. The message survives a failed / superseded start, loudly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_message_survives_raising_start(tmp_path):
    """The incident, reproduced: the provider blows up inside the start
    window. The message must still be on disk, the session must be listed, and
    the failure must be visible."""
    ws = str(tmp_path)
    mgr = _manager(ws)
    boom = ProviderConfigError("invalid_api_key", "MiniMax rejected the key")
    mgr.start_agent = _CapturingStart(ws, raises=boom)

    with pytest.raises(ProviderConfigError):
        await mgr.inject_message("proj", "do not lose me",
                                 nonce="n-boom", session_id=SESSION_ID)

    # 1. The user's words exist.
    users = _user_rows(ws, SESSION_ID)
    assert [r["content"] for r in users] == ["do not lose me"]
    assert users[0]["nonce"] == "n-boom"
    # 2. The session is real, not "minted but unmaterialized".
    listed = [s["session_uuid"] for s in mgr.list_sessions("proj")]
    assert SESSION_ID in listed
    # 3. The transcript says what happened.
    systems = [r for r in _rows(ws, SESSION_ID) if r.get("role") == "system"]
    assert len(systems) == 1
    assert "could not be started" in systems[0]["content"]
    # 4. A classified error the UI can render.
    terminal = mgr.get_last_terminal_event("proj", session_id=SESSION_ID)
    assert terminal["type"] == "error"
    assert terminal["error_code"] == "invalid_api_key"
    errors = [
        c.args[1] for c in mgr._ws.broadcast.call_args_list
        if c.args[1].get("type") == "agent.status"
        and c.args[1].get("status") == "error"
    ]
    assert len(errors) == 1
    assert errors[0]["error_code"] == "invalid_api_key"


@pytest.mark.asyncio
async def test_message_survives_cancelled_start(tmp_path):
    """The supersede route: the start is cancelled before the loop can run.
    Identical outcome — content preserved, failure surfaced with a non-empty
    reason (CancelledError stringifies to "")."""
    ws = str(tmp_path)
    mgr = _manager(ws)
    mgr.start_agent = _CapturingStart(ws, raises=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await mgr.inject_message("proj", "superseded but saved",
                                 nonce="n-cancel", session_id=SESSION_ID)

    assert [r["content"] for r in _user_rows(ws, SESSION_ID)] == [
        "superseded but saved",
    ]
    systems = [r for r in _rows(ws, SESSION_ID) if r.get("role") == "system"]
    assert "cancelled or superseded" in systems[0]["content"]
    terminal = mgr.get_last_terminal_event("proj", session_id=SESSION_ID)
    assert terminal["type"] == "error"
    assert terminal["details"]          # never the empty str(CancelledError())
    assert "superseded" in terminal["details"]


@pytest.mark.asyncio
async def test_task_cancelled_before_first_slice_keeps_the_message(tmp_path):
    """The narrower supersede: ``start_agent`` returns fine but the loop task
    it created is cancelled before its first execution slice. That used to be
    total loss because ``run()`` owned the append; now the row predates the
    task entirely."""
    ws = str(tmp_path)
    mgr = _manager(ws)
    ran = asyncio.Event()

    async def _never_runs():
        ran.set()
        await asyncio.sleep(3600)

    created: list[asyncio.Task] = []

    async def _start(project_id, config, **kwargs):
        created.append(asyncio.create_task(_never_runs()))

    mgr.start_agent = _start

    await mgr.inject_message("proj", "killed before the first slice",
                             nonce="n-slice", session_id=SESSION_ID)
    created[0].cancel()                 # supersede before any slice runs
    with pytest.raises(asyncio.CancelledError):
        await created[0]
    assert ran.is_set() is False        # the coroutine never executed

    assert [r["content"] for r in _user_rows(ws, SESSION_ID)] == [
        "killed before the first slice",
    ]


@pytest.mark.asyncio
async def test_slot_conflict_writes_nothing(tmp_path):
    """A slot conflict is a REJECTION, not a failure: the inject route turns it
    into a 202 and enqueues the message as a pending inject, which appends it
    for real when the slot frees. Writing ahead here too would render it
    twice — so the write-ahead pre-flights the slot guard and writes nothing.
    """
    ws = str(tmp_path)
    mgr = _manager(ws)
    holder = MagicMock()
    holder.session.is_stopped.return_value = False
    holder.session._paused_for_approval = False
    holder.task = MagicMock()
    holder.task.done.return_value = False
    mgr._handles[("proj", "someone_else")] = holder
    mgr.start_agent = _CapturingStart(ws)

    with pytest.raises(ValueError, match="Slot held by session"):
        await mgr.inject_message("proj", "should not be written",
                                 nonce="n-slot", session_id=SESSION_ID)

    assert _rows(ws, SESSION_ID) == []
    assert not os.path.exists(
        os.path.join(_sessions_dir(ws), f"{SESSION_ID}.jsonl"),
    )
    # And no start-failure noise for what is a normal queueing outcome.
    assert mgr.get_last_terminal_event("proj", session_id=SESSION_ID) is None


# ---------------------------------------------------------------------------
# 4. run() cannot re-grow its append
# ---------------------------------------------------------------------------

class _TextProvider:
    """One text response, then done."""

    def __init__(self):
        self.provider = "stub"
        self.model = "stub-model"
        self.sdk = "openai"

    async def stream(self, messages, tools=None):
        from agent_os.agent.providers.types import StreamChunk, TokenUsage
        yield StreamChunk(text="ack")
        yield StreamChunk(is_final=True,
                          usage=TokenUsage(input_tokens=1, output_tokens=1))


def _real_loop(workspace: str, session_id: str) -> tuple[AgentLoop, Session]:
    from agent_os.agent.context import ContextManager
    from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext

    session = Session.new(session_id, workspace, session_id=session_id)
    registry = MagicMock()
    registry.tool_names.return_value = []
    registry.get_schemas.return_value = []
    registry.reset_run_state.return_value = None
    ctx = PromptContext(
        workspace=workspace, model="stub-model", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[], tool_names=[], os_type="macos",
        datetime_now="2026-08-16T00:00:00",
    )
    cm = ContextManager(session, PromptBuilder(workspace=workspace), ctx)
    return AgentLoop(session, _TextProvider(), registry, cm), session


def test_run_takes_no_message_parameter():
    """Structural guard: the parameters are gone, so nothing can accidentally
    hand ``run()`` a message again and reopen the two-writer split."""
    params = inspect.signature(AgentLoop.run).parameters
    assert list(params) == ["self"]


@pytest.mark.asyncio
async def test_run_appends_no_user_row_of_its_own(tmp_path):
    """Behavioural half of the same guard: one persist + one run == one user
    row, on disk and in memory."""
    ws = str(tmp_path)
    loop, session = _real_loop(ws, SESSION_ID)

    persist_user_row(session, "only once", "n-once")
    await loop.run()

    assert [r["content"] for r in _user_rows(ws, SESSION_ID)] == ["only once"]
    in_memory = [m for m in session.get_messages() if m.get("role") == "user"]
    assert len(in_memory) == 1
    assert in_memory[0]["nonce"] == "n-once"


@pytest.mark.asyncio
async def test_start_agent_initial_message_persists_exactly_once(tmp_path):
    """``POST /agents/start`` and the trigger manager still pass
    ``initial_message`` to ``start_agent``. It goes through the same single
    writer, and with ``run()``'s append gone it lands exactly once."""
    ws = str(tmp_path)
    mgr = _manager(ws)
    session = Session.new(SESSION_ID, ws, session_id=SESSION_ID)
    loop, _ = _real_loop(ws, "throwaway")

    # Cut start_agent off just after the persist branch: everything past it is
    # provider/registry/sandbox wiring this assertion does not need.
    class _Stop(Exception):
        pass

    mgr._build_llm_providers = MagicMock(side_effect=_Stop())
    with pytest.raises(_Stop):
        await mgr.start_agent("proj", _config(ws),
                              initial_message="via /agents/start",
                              initial_nonce="n-http",
                              session_id=SESSION_ID, session=session)

    rows = _user_rows(ws, SESSION_ID)
    assert [r["content"] for r in rows] == ["via /agents/start"]
    assert rows[0]["nonce"] == "n-http"

    # Now let the loop run over that same session: still one row.
    loop._session = session
    await loop.run()
    assert len(_user_rows(ws, SESSION_ID)) == 1


# ---------------------------------------------------------------------------
# 5. The worker side of the one-writer invariant
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_worker_persists_brief_exactly_once(tmp_path):
    """``NativeWorkerAdapter.send`` took over the append when ``run()`` lost
    it — the brief is written once, and it is written before the turn."""
    from tests.unit.test_native_worker import (  # noqa: PLC0415 — shared harness
        _make_registry_factory, _StubProvider, _StubToolRegistry,
    )
    from agent_os.daemon_v2.native_worker import NativeWorkerAdapter, WorkerDeps

    deps = WorkerDeps(
        provider=_StubProvider("done"),
        workspace=str(tmp_path),
        project_id="proj-1",
        parent_session_id="parent-1",
        make_tool_registry=_make_registry_factory(_StubToolRegistry()),
    )
    adapter = NativeWorkerAdapter(deps=deps, handle="worker:w-0",
                                  display_name="w0", allowed_paths=None,
                                  forbidden_paths=None)
    await adapter.send("the task brief")

    with open(adapter.session_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    briefs = [r for r in rows if r.get("role") == "user"]
    assert [r["content"] for r in briefs] == ["the task brief"]


@pytest.mark.asyncio
async def test_worker_brief_survives_a_failing_turn(tmp_path):
    """A worker whose very first provider call raises still leaves the task it
    was given on disk instead of an empty session."""
    from tests.unit.test_native_worker import (  # noqa: PLC0415 — shared harness
        _make_registry_factory, _RaisingProvider, _StubToolRegistry,
    )
    from agent_os.daemon_v2.native_worker import NativeWorkerAdapter, WorkerDeps

    deps = WorkerDeps(
        provider=_RaisingProvider(),
        workspace=str(tmp_path),
        project_id="proj-1",
        parent_session_id="parent-1",
        make_tool_registry=_make_registry_factory(_StubToolRegistry()),
    )
    adapter = NativeWorkerAdapter(deps=deps, handle="worker:w-1",
                                  display_name="w1", allowed_paths=None,
                                  forbidden_paths=None)
    await adapter.send("brief that must survive")
    assert adapter._last_response.startswith("Error")

    with open(adapter.session_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert [r["content"] for r in rows if r.get("role") == "user"] == [
        "brief that must survive",
    ]


# ---------------------------------------------------------------------------
# The writer itself
# ---------------------------------------------------------------------------

def test_persist_user_row_shape_and_durability(tmp_path):
    """The row shape is byte-identical to what ``AgentLoop.run()`` used to
    write, and it is on disk the moment the call returns — a falsy nonce is
    omitted rather than stored as an empty string."""
    ws = str(tmp_path)
    session = Session.new(SESSION_ID, ws, session_id=SESSION_ID)

    msg = persist_user_row(session, "with a nonce", "n-1")
    assert msg["role"] == "user"
    assert msg["source"] == "user"
    assert msg["nonce"] == "n-1"

    persist_user_row(session, "no nonce", None)
    persist_user_row(session, "empty nonce", "")

    rows = _user_rows(ws, SESSION_ID)
    assert [r["content"] for r in rows] == [
        "with a nonce", "no nonce", "empty nonce",
    ]
    assert "nonce" not in rows[1] and "nonce" not in rows[2]
    assert all(r["session_id"] == SESSION_ID for r in rows)
