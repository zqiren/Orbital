# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""PiRPCTransport — the native client for ``pi --mode rpc`` (spec 087 §5.2).

Routing tests replay records captured from the released pi 0.85.1 (redacted,
tests/fixtures/pi_rpc/). Process tests drive tests/fixtures/pi_rpc/fake_pi.py,
a JSONL child that replays those same records — no provider is ever called.
"""

import asyncio
import json
import os
import sys
import time

import psutil
import pytest

from agent_os.agent.transports.pi_rpc_transport import (
    PROMPT_FILE_PREFIX,
    PiProtocolError,
    PiRPCTransport,
    iter_records,
)

FIXTURES = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "fixtures", "pi_rpc"))
FAKE_PI = os.path.join(FIXTURES, "fake_pi.py")
SECRET = "sk-test-secret-value-0123456789"
MODEL_ARGS = ["--mode", "rpc", "--no-approve",
              "--model", "opencode-go/deepseek-v4-flash", "--thinking", "low"]


def _fixture(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _drain(t):
    events = []
    while not t._event_queue.empty():
        events.append(t._event_queue.get_nowait())
    return events


def _routed():
    """A transport mid-session with an open turn, fed records directly."""
    t = PiRPCTransport()
    t._session_id = "S1"
    t._model = "opencode-go/deepseek-v4-flash"
    t._session_file = "/tmp/pi-sessions/S1.jsonl"
    t._begin_turn()
    return t


async def _replay(t, records):
    for rec in records:
        await t._route_record(rec)


def _dead(pid, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.05)
    return False


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


async def _records_from(chunks, **kwargs):
    reader = asyncio.StreamReader()

    async def feed():
        for chunk in chunks:
            reader.feed_data(chunk)
            await asyncio.sleep(0)
        reader.feed_eof()

    feeder = asyncio.create_task(feed())
    try:
        return [rec async for rec in iter_records(reader, **kwargs)]
    finally:
        await feeder


class TestFraming:
    async def test_partial_and_multiple_records_split_on_lf_only(self):
        # U+2028/U+2029 are valid inside JSON strings; a generic line reader
        # that treats them as newlines corrupts the record (pi docs, Framing).
        text = "a b c"
        payload = (json.dumps({"type": "x", "text": text}, ensure_ascii=False)
                   + "\n" + json.dumps({"type": "y"}) + "\r\n").encode("utf-8")
        chunks = [payload[i:i + 5] for i in range(0, len(payload), 5)]
        assert await _records_from(chunks) == [
            {"type": "x", "text": text}, {"type": "y"}]

    async def test_malformed_record_raises(self):
        with pytest.raises(PiProtocolError, match="unparseable"):
            await _records_from([b'{"type": "ok"}\n{not json\n'])

    async def test_non_object_record_raises(self):
        with pytest.raises(PiProtocolError):
            await _records_from([b"[1, 2]\n"])

    async def test_oversize_record_raises(self):
        line = b'{"type": "x", "d": "' + b"y" * 200 + b'"}\n'
        with pytest.raises(PiProtocolError, match="exceeded"):
            await _records_from([line], max_line_bytes=64)


# ---------------------------------------------------------------------------
# Event routing (recorded protocol)
# ---------------------------------------------------------------------------


class TestTurnRouting:
    async def test_text_turn_with_auto_retry_streams_once_and_settles_once(self):
        t = _routed()
        await _replay(t, _fixture("turn_retry_then_text.jsonl"))
        events = _drain(t)
        assert [e.event_type for e in events] == ["message", "turn_complete"]
        assert events[0].data["text"] == "PELICAN-42 ZEBRA-7"
        assert events[1].data == {
            "cause": "success", "session_id": "S1",
            "model": "opencode-go/deepseek-v4-flash",
            "rollout_path": "/tmp/pi-sessions/S1.jsonl",
        }

    async def test_agent_end_and_retry_events_never_close_the_turn(self):
        t = _routed()
        records = _fixture("turn_retry_then_text.jsonl")
        await _replay(t, [r for r in records if r.get("type") != "agent_settled"])
        await _replay(t, [
            {"type": "agent_end", "messages": [], "willRetry": False},
            {"type": "compaction_start", "reason": "overflow"},
            {"type": "compaction_end", "reason": "overflow", "result": None,
             "aborted": False, "willRetry": True},
            {"type": "auto_retry_start", "attempt": 1, "maxAttempts": 3,
             "delayMs": 1, "errorMessage": "overloaded"},
        ])
        assert "turn_complete" not in [e.event_type for e in _drain(t)]
        assert t._turn_open
        await _replay(t, [{"type": "agent_settled"}, {"type": "agent_settled"}])
        assert [e.event_type for e in _drain(t)] == ["turn_complete"]

    async def test_correlated_responses_never_reach_the_transcript(self):
        t = _routed()
        fut = asyncio.get_running_loop().create_future()
        t._pending["orbital-7"] = fut
        await t._route_record({"id": "orbital-7", "type": "response",
                               "command": "get_state", "success": True, "data": {}})
        await t._route_record({"type": "response", "command": "not_a_command",
                               "success": False, "error": "Unknown command"})
        assert fut.result()["command"] == "get_state"
        assert _drain(t) == []

    async def test_tool_lifecycle_is_activity_not_a_boundary(self):
        t = _routed()
        records = _fixture("turn_tool.jsonl")
        end = next(i for i, r in enumerate(records)
                   if r.get("type") == "tool_execution_end")
        await _replay(t, records[:end + 1])
        start, finish = _drain(t)
        assert (start.event_type, finish.event_type) == ("tool_use", "tool_use")
        assert start.data["tool_name"] == "bash"
        assert start.data["tool_input"] == {"command": "echo hello > probe.txt"}
        assert start.data["status"] == "running"
        assert finish.data["tool_id"] == start.data["tool_id"]
        assert finish.data["status"] == "completed"
        assert t._turn_open
        await _replay(t, records[end + 1:])
        rest = _drain(t)
        assert [e.event_type for e in rest] == ["message", "turn_complete"]
        assert rest[0].data["text"] == "DONE"
        assert rest[1].data["cause"] == "success"

    async def test_provider_error_is_an_error_turn(self):
        t = _routed()
        await _replay(t, _fixture("turn_provider_error.jsonl"))
        events = _drain(t)
        assert [e.event_type for e in events] == ["error", "turn_complete"]
        assert "Model no-such-model-xyz is not supported" in events[0].raw_text
        assert events[1].data["cause"] == "error"

    async def test_failed_auto_retry_is_an_error_turn(self):
        t = _routed()
        await _replay(t, [
            {"type": "agent_start"},
            {"type": "auto_retry_end", "success": False, "attempt": 3,
             "finalError": "529 overloaded_error: Overloaded"},
            {"type": "agent_end", "messages": [], "willRetry": False},
            {"type": "agent_settled"},
        ])
        events = _drain(t)
        assert [e.event_type for e in events] == ["error", "turn_complete"]
        assert "overloaded" in events[0].raw_text
        assert events[1].data["cause"] == "error"

    async def test_abort_settles_as_interrupted_without_an_error(self):
        t = _routed()
        t._abort_requested = True
        await _replay(t, _fixture("turn_abort.jsonl"))
        events = _drain(t)
        types = [e.event_type for e in events]
        assert "error" not in types
        assert types[-1] == "turn_complete"
        assert types.count("turn_complete") == 1
        assert events[-1].data["cause"] == "interrupted"
        assert [e.data["status"] for e in events
                if e.event_type == "tool_use"] == ["running", "error"]

    async def test_abort_during_teardown_settles_as_stopped(self):
        t = _routed()
        t._abort_requested = True
        t._stopping = True
        await _replay(t, _fixture("turn_abort.jsonl"))
        assert _drain(t)[-1].data["cause"] == "stopped"

    async def test_text_block_is_not_repeated_by_message_end(self):
        t = _routed()
        await _replay(t, [
            {"type": "message_start", "message": {"role": "assistant", "content": []}},
            {"type": "message_update", "assistantMessageEvent": {
                "type": "text_start", "contentIndex": 0}},
            {"type": "message_update", "assistantMessageEvent": {
                "type": "text_delta", "contentIndex": 0, "delta": "Hel"}},
            {"type": "message_update", "assistantMessageEvent": {
                "type": "text_delta", "contentIndex": 0, "delta": "lo"}},
            {"type": "message_update", "assistantMessageEvent": {
                "type": "text_end", "contentIndex": 0, "content": "Hello"}},
            {"type": "message_end", "message": {
                "role": "assistant", "stopReason": "stop",
                "content": [{"type": "text", "text": "Hello"}]}},
            {"type": "agent_settled"},
        ])
        assert [(e.event_type, e.data.get("text")) for e in _drain(t)] == [
            ("message", "Hello"), ("turn_complete", None)]

    async def test_partial_text_is_flushed_when_the_turn_settles(self):
        t = _routed()
        await _replay(t, [
            {"type": "message_start", "message": {"role": "assistant", "content": []}},
            {"type": "message_update", "assistantMessageEvent": {
                "type": "text_delta", "contentIndex": 0, "delta": "Partial "}},
            {"type": "message_update", "assistantMessageEvent": {
                "type": "text_delta", "contentIndex": 0, "delta": "answer"}},
            {"type": "agent_settled"},
        ])
        assert [(e.event_type, e.data.get("text")) for e in _drain(t)] == [
            ("message", "Partial answer"), ("turn_complete", None)]

    async def test_message_end_recovers_text_that_never_streamed(self):
        t = _routed()
        await _replay(t, [
            {"type": "message_start", "message": {"role": "assistant", "content": []}},
            {"type": "message_end", "message": {
                "role": "assistant", "stopReason": "stop",
                "content": [{"type": "thinking", "thinking": "hmm"},
                            {"type": "text", "text": "Final"}]}},
            {"type": "agent_settled"},
        ])
        assert [(e.event_type, e.data.get("text")) for e in _drain(t)] == [
            ("message", "Final"), ("turn_complete", None)]

    async def test_read_stream_delivers_events_queued_before_the_process_died(self):
        t = _routed()
        await _replay(t, _fixture("turn_provider_error.jsonl"))
        t._alive = False
        events = [e async for e in t.read_stream()]
        assert [e.event_type for e in events] == ["error", "turn_complete"]


# ---------------------------------------------------------------------------
# Process-level behaviour against the fake child
# ---------------------------------------------------------------------------


def _record(tmp_path):
    path = tmp_path / "record.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _spawns(tmp_path):
    return [r for r in _record(tmp_path) if r["event"] == "spawn"]


def _commands(tmp_path, ctype):
    return [r["cmd"] for r in _record(tmp_path)
            if r["event"] == "command" and r["cmd"].get("type") == ctype]


def _scenario(tmp_path, records):
    path = tmp_path / "scenario.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                    encoding="utf-8")
    return str(path)


async def _started(tmp_path, *, env=None, resume=None,
                   system_prompt="Orbital brief\nPeers: codex\n", **kwargs):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    options = {"command_timeout": 10.0, "abort_timeout": 3.0, **kwargs}
    t = PiRPCTransport(
        system_prompt=system_prompt, resume_record=resume,
        session_dir=str(tmp_path / "pi-sessions"), **options)
    full_env = {"FAKE_PI_RECORD": str(tmp_path / "record.jsonl"),
                "FAKE_PI_ENV_KEYS": "OPENCODE_API_KEY",
                "OPENCODE_API_KEY": SECRET}
    full_env.update(env or {})
    await asyncio.wait_for(
        t.start(sys.executable, [FAKE_PI, *MODEL_ARGS], str(ws), full_env), 30)
    return t


async def _until_turn_complete(t, timeout=20.0):
    events = []

    async def pump():
        while True:
            event = await t._event_queue.get()
            events.append(event)
            if event.event_type == "turn_complete":
                return

    await asyncio.wait_for(pump(), timeout)
    return events


async def _next_event(t, event_type, timeout=20.0):
    async def pump():
        while True:
            event = await t._event_queue.get()
            if event.event_type == event_type:
                return event

    return await asyncio.wait_for(pump(), timeout)


async def _child_pid(tmp_path, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        children = [r for r in _record(tmp_path) if r["event"] == "child"]
        if children:
            return children[0]["pid"]
        await asyncio.sleep(0.05)
    raise AssertionError("fake pi never started its tool child")


async def _until_dead(t, timeout=10.0):
    deadline = time.monotonic() + timeout
    while t.is_alive() and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    return not t.is_alive()


class TestSpawn:
    async def test_every_value_reaches_the_child_separately(self, tmp_path):
        t = await _started(tmp_path)
        try:
            [spawn] = _spawns(tmp_path)
            argv = spawn["argv"]
            assert argv[:len(MODEL_ARGS)] == MODEL_ARGS
            assert argv[argv.index("--session-dir") + 1] == str(tmp_path / "pi-sessions")
            prompt_path = argv[argv.index("--append-system-prompt") + 1]
            assert os.path.dirname(prompt_path) == str(tmp_path / "pi-sessions")
            assert os.path.basename(prompt_path).startswith(PROMPT_FILE_PREFIX)
            # Multi-line prompt text travels as a file (npm's Windows .cmd shim
            # cannot carry newlines in argv); pi reads a path's contents.
            assert spawn["append_prompt"] == "Orbital brief\nPeers: codex\n"
            assert "--session" not in argv
            assert os.path.realpath(spawn["cwd"]) == os.path.realpath(str(tmp_path / "ws"))
            assert spawn["env"]["OPENCODE_API_KEY"] == SECRET
            assert not any(SECRET in arg for arg in argv)
            [started] = [e for e in _drain(t) if e.event_type == "thread_started"]
            assert started.data["session_id"] == "fresh-session-1"
            assert started.data["model"] == "opencode-go/deepseek-v4-flash"
            assert started.data["provider"] == "opencode-go"
            assert started.data["resume_status"] == "fresh"
            assert t.resume_outcome == ("fresh", None)
            assert t.session_id == "fresh-session-1"
            assert t._proc is not None and t._proc.pid == spawn["pid"]
        finally:
            await t.stop()
        assert not os.path.exists(prompt_path)
        assert not t.is_alive()
        assert _dead(spawn["pid"])


class TestResume:
    async def test_matching_session_is_resumed(self, tmp_path):
        t = await _started(tmp_path, resume={"session_id": "S-known"},
                           env={"FAKE_PI_KNOWN_SESSIONS": "S-known"})
        try:
            [spawn] = _spawns(tmp_path)
            assert spawn["argv"][spawn["argv"].index("--session") + 1] == "S-known"
            assert t.resume_outcome == ("resumed", None)
            assert t.session_id == "S-known"
            [started] = [e for e in _drain(t) if e.event_type == "thread_started"]
            assert started.data["resume_status"] == "resumed"
        finally:
            await t.stop()

    async def test_unknown_session_restarts_fresh_exactly_once(self, tmp_path):
        t = await _started(tmp_path, resume={"session_id": "S-gone"})
        try:
            first, second = _spawns(tmp_path)
            assert "--session" in first["argv"]
            assert "--session" not in second["argv"]
            assert _dead(first["pid"])
            assert t.resume_outcome == ("fresh", "resume_failed")
            assert t._resume_session_id is None
            assert t.session_id == "fresh-session-1"
            [started] = [e for e in _drain(t) if e.event_type == "thread_started"]
            assert (started.data["resume_status"], started.data["resume_reason"]) == (
                "fresh", "resume_failed")
        finally:
            await t.stop()

    async def test_mismatched_session_restarts_fresh_exactly_once(self, tmp_path):
        t = await _started(tmp_path, resume={"session_id": "S-wanted"},
                           env={"FAKE_PI_RESUME_MODE": "mismatch",
                                "FAKE_PI_NEW_SESSION_ID": "S-other"})
        try:
            first, second = _spawns(tmp_path)
            assert "--session" in first["argv"]
            assert "--session" not in second["argv"]
            assert _dead(first["pid"])
            assert t.resume_outcome == ("fresh", "resume_failed")
            assert t.session_id == "S-other"
        finally:
            await t.stop()

    async def test_failed_fresh_restart_is_not_retried_again(self, tmp_path):
        with pytest.raises(RuntimeError):
            await _started(tmp_path, resume={"session_id": "S-gone"},
                           env={"FAKE_PI_EXIT_WITHOUT_SESSION": "1"})
        assert len(_spawns(tmp_path)) == 2
        leftovers = [f for f in os.listdir(tmp_path / "pi-sessions")
                     if f.startswith(PROMPT_FILE_PREFIX)]
        assert leftovers == []

    async def test_startup_failure_is_loud_and_redacted(self, tmp_path):
        with pytest.raises(RuntimeError) as excinfo:
            await _started(tmp_path, env={
                "FAKE_PI_EXIT_AT_START": "2",
                "FAKE_PI_STDERR": f"boom: provider rejected {SECRET}"})
        message = str(excinfo.value)
        assert "boom" in message
        assert "exit code 2" in message
        assert SECRET not in message
        [spawn] = _spawns(tmp_path)
        assert _dead(spawn["pid"])


class TestTurns:
    async def test_tool_turn_through_the_child(self, tmp_path):
        t = await _started(tmp_path, env={
            "FAKE_PI_SCENARIO": os.path.join(FIXTURES, "turn_tool.jsonl")})
        try:
            _drain(t)
            await t.dispatch("Use the bash tool")
            events = await _until_turn_complete(t)
            assert [e.event_type for e in events] == [
                "tool_use", "tool_use", "message", "turn_complete"]
            assert events[-1].data["cause"] == "success"
            assert events[-1].data["session_id"] == "fresh-session-1"
            [prompt] = _commands(tmp_path, "prompt")
            assert prompt["message"] == "Use the bash tool"
            assert prompt["id"]
            assert t.is_alive()
        finally:
            await t.stop()

    async def test_rejected_prompt_is_an_error_turn(self, tmp_path):
        t = await _started(tmp_path, env={
            "FAKE_PI_SCENARIO": os.path.join(FIXTURES, "prompt_rejected_no_key.jsonl")})
        try:
            _drain(t)
            await t.dispatch("hi")
            events = await _until_turn_complete(t)
            assert [e.event_type for e in events] == ["error", "turn_complete"]
            assert "No API key found for openrouter" in events[0].raw_text
            assert events[1].data["cause"] == "error"
            assert t.is_alive()
        finally:
            await t.stop()

    async def test_second_dispatch_while_a_turn_runs_is_refused(self, tmp_path):
        scenario = _scenario(tmp_path, [
            {"type": "response", "command": "prompt", "success": True},
            {"type": "agent_start"},
            {"__wait_for__": "abort"},
            {"type": "agent_settled"},
            {"type": "response", "command": "abort", "success": True},
        ])
        t = await _started(tmp_path, env={"FAKE_PI_SCENARIO": scenario})
        try:
            await t.dispatch("first")
            with pytest.raises(RuntimeError):
                await t.dispatch("second")
        finally:
            await t.stop()
        assert len(_commands(tmp_path, "prompt")) == 1

    async def test_unicode_line_separators_survive_the_pipe(self, tmp_path):
        text = "line sep para"
        scenario = _scenario(tmp_path, [
            {"type": "response", "command": "prompt", "success": True},
            {"type": "agent_start"},
            {"type": "message_start", "message": {"role": "assistant", "content": []}},
            {"type": "message_update", "assistantMessageEvent": {
                "type": "text_end", "contentIndex": 0, "content": text}},
            {"type": "message_end", "message": {
                "role": "assistant", "stopReason": "stop",
                "content": [{"type": "text", "text": text}]}},
            {"type": "agent_settled"},
        ])
        t = await _started(tmp_path, env={"FAKE_PI_SCENARIO": scenario})
        try:
            _drain(t)
            await t.dispatch("go")
            events = await _until_turn_complete(t)
            assert [(e.event_type, e.data.get("text")) for e in events] == [
                ("message", text), ("turn_complete", None)]
        finally:
            await t.stop()

    async def test_extension_dialogs_are_cancelled_not_left_blocking(self, tmp_path):
        scenario = _scenario(tmp_path, [
            {"type": "response", "command": "prompt", "success": True},
            {"type": "agent_start"},
            {"type": "extension_ui_request", "id": "ui-1", "method": "confirm",
             "title": "Allow?", "message": "rm -rf build"},
            {"__wait_for__": "extension_ui_response"},
            {"type": "agent_settled"},
        ])
        t = await _started(tmp_path, env={"FAKE_PI_SCENARIO": scenario})
        try:
            _drain(t)
            await t.dispatch("go")
            events = await _until_turn_complete(t)
            assert events[-1].data["cause"] == "success"
            assert _commands(tmp_path, "extension_ui_response") == [
                {"type": "extension_ui_response", "id": "ui-1", "cancelled": True}]
        finally:
            await t.stop()


class TestFailures:
    async def test_child_exit_mid_turn_is_an_error_turn(self, tmp_path):
        scenario = _scenario(tmp_path, [
            {"type": "response", "command": "prompt", "success": True},
            {"type": "agent_start"},
            {"__exit__": 3},
        ])
        t = await _started(tmp_path, env={"FAKE_PI_SCENARIO": scenario})
        try:
            _drain(t)
            await t.dispatch("go")
            events = await _until_turn_complete(t)
            assert [e.event_type for e in events] == ["error", "turn_complete"]
            assert "exit code 3" in events[0].raw_text
            assert events[1].data["cause"] == "error"
            assert await _until_dead(t)
            assert t._pending == {}
        finally:
            await t.stop()

    async def test_malformed_record_fails_the_turn_and_kills_pi(self, tmp_path):
        scenario = _scenario(tmp_path, [
            {"type": "response", "command": "prompt", "success": True},
            {"type": "agent_start"},
            {"__raw__": "{not json"},
            {"__sleep__": 60},
        ])
        t = await _started(tmp_path, env={"FAKE_PI_SCENARIO": scenario})
        try:
            _drain(t)
            await t.dispatch("go")
            events = await _until_turn_complete(t)
            assert [e.event_type for e in events] == ["error", "turn_complete"]
            assert "unparseable" in events[0].raw_text
            assert events[1].data["cause"] == "error"
            assert await _until_dead(t)
            [spawn] = _spawns(tmp_path)
            assert _dead(spawn["pid"])
        finally:
            await t.stop()

    async def test_oversize_record_fails_the_turn_loudly(self, tmp_path):
        scenario = _scenario(tmp_path, [
            {"type": "response", "command": "prompt", "success": True},
            {"type": "agent_start"},
            {"__big_line__": 20000},
            {"__sleep__": 60},
        ])
        t = await _started(tmp_path, env={"FAKE_PI_SCENARIO": scenario},
                           max_line_bytes=4096)
        try:
            _drain(t)
            await t.dispatch("go")
            events = await _until_turn_complete(t)
            assert [e.event_type for e in events] == ["error", "turn_complete"]
            assert "exceeded" in events[0].raw_text
            assert await _until_dead(t)
        finally:
            await t.stop()


class TestStop:
    async def test_stop_mid_tool_aborts_and_leaves_no_process(self, tmp_path):
        recorded = [r for r in _fixture("turn_abort.jsonl") if r.get("id") != "p3b"]
        tool_start = next(i for i, r in enumerate(recorded)
                          if r.get("type") == "tool_execution_start")
        scenario = _scenario(tmp_path, recorded[:tool_start + 1] + [
            {"__spawn_child__": True}, {"__wait_for__": "abort"},
        ] + recorded[tool_start + 1:])
        t = await _started(tmp_path, env={"FAKE_PI_SCENARIO": scenario})
        _drain(t)
        await t.dispatch("sleep for a while")
        await _next_event(t, "tool_use")
        child = await _child_pid(tmp_path)
        [spawn] = _spawns(tmp_path)
        await asyncio.wait_for(t.stop(), 30)
        assert len(_commands(tmp_path, "abort")) == 1
        assert _dead(child)
        assert _dead(spawn["pid"])
        assert not t.is_alive()
        assert t._pending == {}
        causes = [e.data["cause"] for e in _drain(t) if e.event_type == "turn_complete"]
        assert causes == ["stopped"]

    async def test_stop_with_an_unanswered_abort_still_kills_the_tree(self, tmp_path):
        scenario = _scenario(tmp_path, [
            {"type": "response", "command": "prompt", "success": True},
            {"type": "agent_start"},
            {"type": "tool_execution_start", "toolCallId": "call_1",
             "toolName": "bash", "args": {"command": "sleep 120"}},
            {"__spawn_child__": True},
            {"__sleep__": 120},
        ])
        t = await _started(tmp_path, env={"FAKE_PI_SCENARIO": scenario},
                           abort_timeout=1.0)
        _drain(t)
        await t.dispatch("hang")
        await _next_event(t, "tool_use")
        child = await _child_pid(tmp_path)
        [spawn] = _spawns(tmp_path)
        started = time.monotonic()
        await asyncio.wait_for(t.stop(), 30)
        assert time.monotonic() - started < 15
        assert _dead(child)
        assert _dead(spawn["pid"])
        assert not t.is_alive()
        assert t._pending == {}
