# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The onboarding gate: once a run creates orbital/instructions/project_goals.md,
the daemon refuses side-effect tools in that session until the user's next
message, so onboarding ends with a proposal and the user's go starts the work.

Daemon state, not prompt state: the flag lives on the Session, set from the
tool-dispatch path and cleared by a real user row.
"""

import json
import os

import pytest

from agent_os.agent import onboarding_gate as gate
from agent_os.agent.context import ContextManager
from agent_os.agent.loop import AgentLoop
from agent_os.agent.prompt_builder import Autonomy, PromptContext
from agent_os.agent.providers.types import StreamChunk, TokenUsage
from agent_os.agent.session import Session, persist_user_row
from agent_os.agent.tools.base import ToolResult

_USAGE = TokenUsage(input_tokens=10, output_tokens=5)
GOALS = "orbital/instructions/project_goals.md"


# ---------------------------------------------------------------------------
# rig
# ---------------------------------------------------------------------------

class _PromptBuilder:
    def build(self, context):
        return ("a", "b", "c")


def _ctx(ws):
    return PromptContext(workspace=ws, model="m", autonomy=Autonomy.HANDS_OFF,
                         enabled_agents=[], tool_names=[], os_type="linux",
                         datetime_now="2026-01-01T00:00:00", context_usage_pct=0.0)


def _calls(*calls):
    """One assistant message carrying the given (name, args) tool calls."""
    return [
        StreamChunk(tool_calls_delta=[{
            "index": i, "id": f"c{_calls.n}_{i}", "type": "function",
            "function": {"name": n, "arguments": json.dumps(a)},
        } for i, (n, a) in enumerate(calls)]),
        StreamChunk(is_final=True, usage=_USAGE, finish_reason="tool_calls"),
    ]


_calls.n = 0


def _batch(*calls):
    _calls.n += 1
    return _calls(*calls)


def _reply(text):
    return [StreamChunk(text=text), StreamChunk(is_final=True, usage=_USAGE,
                                                finish_reason="stop")]


class _LLM:
    provider = "test"
    model = "m"
    sdk = "openai"

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    async def stream(self, messages, tools=None):
        idx = self.calls
        self.calls += 1
        if idx >= len(self.script):
            raise AssertionError(f"unexpected LLM call #{idx + 1}")
        for chunk in self.script[idx]:
            yield chunk


class _Tools:
    """write/edit really touch the workspace; everything else just reports."""

    NAMES = ("read", "glob", "grep", "write", "edit", "shell", "agent_message",
             "fanout", "create_trigger", "list_triggers", "notify", "browser",
             "mcp_slack_post")

    def __init__(self, ws):
        self.ws = ws
        self.executed: list[str] = []

    def schemas(self):
        return [{"type": "function", "function": {
            "name": n, "description": n, "parameters": {"type": "object", "properties": {}},
        }} for n in self.NAMES]

    def tool_names(self):
        return list(self.NAMES)

    def is_async(self, name):
        return False

    def execute(self, name, arguments):
        self.executed.append(name)
        if name in ("write", "edit"):
            path = arguments["path"]
            full = path if os.path.isabs(path) else os.path.join(self.ws, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "a" if name == "edit" else "w", encoding="utf-8") as f:
                f.write(arguments.get("content", "x"))
            return ToolResult(content=json.dumps({"path": full, "status": "success"}))
        return ToolResult(content=f"{name} ok")

    async def execute_async(self, name, arguments):
        raise AssertionError

    def reset_run_state(self):
        pass


def _loop(ws, session, llm, tools, **kw):
    ctx = ContextManager(session, _PromptBuilder(), _ctx(ws))
    kw.setdefault("memory_record_enabled", False)
    return AgentLoop(session, llm, tools, ctx, project_dir=ws, max_iterations=20, **kw)


def _rig(tmp_path, script, *, first_user="给团队看，三百字左右", **kw):
    ws = str(tmp_path)
    session = Session.new("sess-gate", ws)
    persist_user_row(session, first_user)
    tools = _Tools(ws)
    llm = _LLM(script)
    return ws, session, tools, llm, _loop(ws, session, llm, tools, **kw)


def _tool_rows(session):
    return {m["tool_call_id"]: m["content"] for m in session.get_messages()
            if m.get("role") == "tool"}


def _goals_write():
    return ("write", {"path": GOALS, "content": "# Mission\n周报\n"})


# ---------------------------------------------------------------------------
# 1. classification (pure)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,args", [
    ("read", {"path": "notes-mon.md"}),
    ("glob", {"pattern": "**/*.md"}),
    ("grep", {"pattern": "周报"}),
    ("list_sessions", {}),
    ("read_session", {"session_id": "x"}),
    ("list_triggers", {}),
    ("calendar_read", {}),
    ("agent_message", {"action": "list"}),
    ("agent_message", {"action": "status", "agent": "claude-code"}),
    ("write", {"path": "orbital/INDEX.md", "content": "x"}),
    ("write", {"path": "orbital/ASKS.md", "content": "x"}),
    ("edit", {"path": "orbital/PROJECT_STATE.md", "old_string": "a", "new_string": "b"}),
    ("edit", {"path": GOALS, "old_string": "a", "new_string": "b"}),
    ("remember_about_user", {"fact": "prefers Chinese"}),
    ("checkpoint_state", {}),
])
def test_allowed_under_the_gate(tmp_path, name, args):
    assert gate.allowed(str(tmp_path), {"name": name, "arguments": args}), (name, args)


@pytest.mark.parametrize("name,args", [
    ("shell", {"command": "ls"}),
    ("write", {"path": "周报-2026-11-02.md", "content": "x"}),
    ("write", {"path": "orbital/../周报.md", "content": "x"}),
    ("write", {"path": "/tmp/elsewhere.md", "content": "x"}),
    ("edit", {"path": "notes-mon.md", "old_string": "a", "new_string": "b"}),
    ("write", {"path": "", "content": "x"}),
    ("write", {}),
    ("agent_message", {"action": "send", "agent": "claude-code", "message": "写周报"}),
    ("agent_message", {"action": "respond", "agent": "claude-code", "message": "ok"}),
    ("agent_message", {"action": "stop", "agent": "claude-code"}),
    ("agent_message", {}),
    ("fanout", {"tasks": []}),
    ("create_trigger", {"name": "weekly", "type": "schedule"}),
    ("update_trigger", {"name": "weekly"}),
    ("delete_trigger", {"name": "weekly"}),
    ("notify", {"message": "done"}),
    ("browser", {"action": "navigate", "url": "https://example.com"}),
    ("browser", {"action": "click", "ref": "e5"}),
    ("request_access", {"path": "/x", "mode": "read"}),
    ("request_credential", {"name": "slack"}),
    ("request_network_access", {"domain": "example.com"}),
    ("mcp_slack_post", {"text": "hi"}),        # unknown tools fail closed
])
def test_refused_under_the_gate(tmp_path, name, args):
    assert not gate.allowed(str(tmp_path), {"name": name, "arguments": args}), (name, args)


def test_browser_reads_stay_allowed(tmp_path):
    for action in ("search", "fetch"):
        assert gate.allowed(str(tmp_path), {"name": "browser", "arguments": {"action": action}})


def test_creates_goals_only_for_a_write_of_a_missing_goals_file(tmp_path):
    ws = str(tmp_path)
    assert gate.creates_goals(ws, {"name": "write", "arguments": {"path": GOALS}})
    assert gate.creates_goals(ws, {"name": "write", "arguments": {
        "path": os.path.join(ws, "orbital", "instructions", "project_goals.md")}})
    assert not gate.creates_goals(ws, {"name": "write", "arguments": {"path": "orbital/INDEX.md"}})
    assert not gate.creates_goals(ws, {"name": "edit", "arguments": {"path": GOALS}})
    os.makedirs(os.path.join(ws, "orbital", "instructions"))
    open(os.path.join(ws, GOALS), "w").close()
    assert not gate.creates_goals(ws, {"name": "write", "arguments": {"path": GOALS}})


# ---------------------------------------------------------------------------
# 2. the loop
# ---------------------------------------------------------------------------

async def test_side_effect_tools_are_refused_after_the_goals_write_and_two_refusals_end_the_run(tmp_path):
    ws, session, tools, llm, loop = _rig(tmp_path, [
        _batch(_goals_write()),
        _batch(("shell", {"command": "cat notes-mon.md"})),
        _batch(("write", {"path": "周报.md", "content": "本周"})),
        _reply("never asked"),
    ])
    await loop.run()
    assert tools.executed == ["write"]                  # only the goals file
    assert not os.path.exists(os.path.join(ws, "周报.md"))
    refusals = [c for c in _tool_rows(session).values() if c == gate.REFUSAL]
    assert len(refusals) == 2
    assert "\n" not in gate.REFUSAL
    assert llm.calls == 3
    assert loop._loop_exit_path == "onboarding_gate"
    assert session.onboarding_gate is True


async def test_a_side_effect_call_in_the_same_batch_as_the_goals_write_is_refused(tmp_path):
    ws, session, tools, llm, loop = _rig(tmp_path, [
        _batch(_goals_write(), ("agent_message", {"action": "send", "agent": "claude-code",
                                                  "message": "写周报"})),
        _reply("建议：我先读本周笔记写初稿，你审核。"),
    ])
    await loop.run()
    assert tools.executed == ["write"]
    rows = _tool_rows(session)
    assert list(rows.values())[1] == gate.REFUSAL
    assert loop._loop_exit_path == "text_complete"      # one refusal does not end it


async def test_reads_and_orbital_writes_still_run_under_the_gate(tmp_path):
    ws, session, tools, llm, loop = _rig(tmp_path, [
        _batch(_goals_write()),
        _batch(("read", {"path": "notes-mon.md"}), ("glob", {"pattern": "*.md"}),
               ("grep", {"pattern": "x"}), ("agent_message", {"action": "list"}),
               ("write", {"path": "orbital/INDEX.md", "content": "# Index"}),
               ("edit", {"path": "orbital/PROJECT_STATE.md", "content": "focus"}),
               ("write", {"path": "orbital/ASKS.md", "content": "# Asks"})),
        _reply("第一步：我来写周报初稿。"),
    ])
    await loop.run()
    assert tools.executed == ["write", "read", "glob", "grep", "agent_message",
                              "write", "edit", "write"]
    assert gate.REFUSAL not in _tool_rows(session).values()
    assert os.path.exists(os.path.join(ws, "orbital", "INDEX.md"))
    assert session.onboarding_gate is True


async def test_the_next_user_message_lifts_the_gate(tmp_path):
    ws, session, tools, llm, loop = _rig(tmp_path, [
        _batch(_goals_write()),
        _reply("第一步：我来写周报初稿。说“开始”我就动手。"),
        _batch(("shell", {"command": "ls"}), ("write", {"path": "周报.md", "content": "本周"})),
        _reply("写好了"),
    ])
    await loop.run()
    assert session.onboarding_gate is True
    persist_user_row(session, "好，开始吧")
    assert session.onboarding_gate is False
    await loop.run()
    assert tools.executed == ["write", "shell", "write"]
    assert os.path.exists(os.path.join(ws, "周报.md"))


async def test_system_daemon_and_trigger_rows_do_not_lift_the_gate(tmp_path):
    ws, session, tools, llm, loop = _rig(tmp_path, [
        _batch(_goals_write()),
        _reply("第一步：我来写周报初稿。"),
        _batch(("shell", {"command": "ls"})),
        _reply("等你的回复"),
    ])
    await loop.run()
    session.append_system("[Sub-agent] claude-code finished")
    session.append({"role": "user", "content": "[Orbital memory] record", "source": "daemon"})
    session.append({"role": "user", "content": "[Triggered by schedule 'weekly' (every Monday)]\n\nGo",
                    "source": "user"})
    assert session.onboarding_gate is True
    # The gate is daemon state: a brand-new loop over the same session (a hot
    # resume / a prompt rebuilt from scratch) still refuses.
    loop2 = _loop(ws, session, llm, tools)
    await loop2.run()
    assert tools.executed == ["write"]
    assert gate.REFUSAL in _tool_rows(session).values()


async def test_a_project_whose_goals_already_exist_never_hits_the_gate(tmp_path):
    os.makedirs(os.path.join(str(tmp_path), "orbital", "instructions"))
    with open(os.path.join(str(tmp_path), GOALS), "w", encoding="utf-8") as f:
        f.write("# Mission\nold\n")
    ws, session, tools, llm, loop = _rig(tmp_path, [
        _batch(_goals_write()),                         # an update, not a creation
        _batch(("shell", {"command": "ls"}), ("write", {"path": "周报.md", "content": "x"})),
        _reply("done"),
    ])
    await loop.run()
    assert tools.executed == ["write", "shell", "write"]
    assert not getattr(session, "onboarding_gate", False)


async def test_a_trigger_run_session_is_never_gated(tmp_path):
    ws, session, tools, llm, loop = _rig(
        tmp_path,
        [_batch(_goals_write()), _batch(("shell", {"command": "ls"})), _reply("done")],
        first_user="[Triggered by schedule 'weekly' (every Monday at 9)]\n\n写周报",
    )
    assert session.trigger_type == "schedule"
    await loop.run()
    assert tools.executed == ["write", "shell"]
    assert not session.onboarding_gate


async def test_a_queue_run_is_never_gated(tmp_path):
    ws, session, tools, llm, loop = _rig(
        tmp_path, [_batch(_goals_write()), _batch(("shell", {"command": "ls"})), _reply("done")])
    loop._queue_state = "running"
    await loop.run()
    assert tools.executed == ["write", "shell"]
    assert not session.onboarding_gate


async def test_a_failed_goals_write_does_not_arm_the_gate(tmp_path):
    class _FailingWrite(_Tools):
        def execute(self, name, arguments):
            if name == "write":
                self.executed.append(name)
                return ToolResult(content="Error: disk full")
            return super().execute(name, arguments)

    ws = str(tmp_path)
    session = Session.new("sess-gate", ws)
    persist_user_row(session, "go")
    tools = _FailingWrite(ws)
    llm = _LLM([_batch(_goals_write()), _batch(("shell", {"command": "ls"})), _reply("x")])
    await _loop(ws, session, llm, tools).run()
    assert tools.executed == ["write", "shell"]
    assert not session.onboarding_gate


async def test_the_record_turn_still_writes_memory_under_the_gate(tmp_path):
    ws, session, tools, llm, loop = _rig(tmp_path, [
        _batch(_goals_write()),
        _reply("第一步：我来写周报初稿。"),
        # the [MEMORY] record turn: its own executor, memory files only
        _batch(("write", {"path": "orbital/PROJECT_STATE.md", "content": "focus: 周报"})),
        _reply("recorded"),
    ], memory_record_enabled=True)
    await loop.run()
    assert session.onboarding_gate is True
    assert open(os.path.join(ws, "orbital", "PROJECT_STATE.md"), encoding="utf-8").read() == "focus: 周报"
    assert gate.REFUSAL not in _tool_rows(session).values()


# ---------------------------------------------------------------------------
# 3. durable across eviction: the daemon drops an idle handle seconds after a
#    run and the next event hydrates a fresh Session from the JSONL
# ---------------------------------------------------------------------------

async def test_the_gate_survives_a_reload_from_disk(tmp_path):
    ws, session, tools, llm, loop = _rig(tmp_path, [
        _batch(_goals_write()),
        _reply("第一步：我来写周报初稿。"),
        _batch(("shell", {"command": "ls"})),
        _reply("等你的回复"),
    ])
    await loop.run()
    reloaded = Session.load(session._filepath)
    assert reloaded.onboarding_gate is True
    # a wake that is not the user's reply (e.g. a background job finishing)
    reloaded.append_system("[Background job] finished")
    await _loop(ws, reloaded, llm, tools).run()
    assert tools.executed == ["write"]
    assert gate.REFUSAL in _tool_rows(reloaded).values()


async def test_a_lifted_gate_stays_lifted_after_a_reload(tmp_path):
    ws, session, tools, llm, loop = _rig(tmp_path, [_batch(_goals_write()), _reply("第一步…")])
    await loop.run()
    persist_user_row(session, "开始")
    assert Session.load(session._filepath).onboarding_gate is False


def test_a_session_without_the_field_loads_ungated(tmp_path):
    session = Session.new("sess-old", str(tmp_path))
    persist_user_row(session, "hi")
    assert Session.load(session._filepath).onboarding_gate is False
