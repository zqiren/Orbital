# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 166: each transport publishes its worker's own command list.

Claude Code (SDK) reads the stored ``initialize`` response after connect;
Codex asks ``skills/list`` in the background, never delaying thread open;
an ACP agent's ``available_commands_update`` is kept whenever it arrives and
never becomes an event. Protocol-boundary fakes only: a mocked SDK client,
a fake ``codex app-server`` executable, the dummy ACP agent fixture.
"""

import asyncio
import json
import os
import stat
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.prompt_builder import Autonomy


def _sink():
    got: list = []
    return got, got.append


# ------------------------------------------------------------------- SDK

class TestSDK:
    @pytest.mark.asyncio
    async def test_connect_publishes_the_initialize_commands(self):
        from agent_os.agent.transports.sdk_transport import SDKTransport
        t = SDKTransport()
        got, sink = _sink()
        t.set_commands_sink(sink)
        with patch("agent_os.agent.transports.sdk_transport.ClaudeSDKClient") as Client, \
             patch("agent_os.agent.transports.sdk_transport.ClaudeAgentOptions"):
            Client.return_value.connect = AsyncMock()
            Client.return_value.get_server_info = AsyncMock(return_value={"commands": [
                {"name": "orbital-probe", "description": "Probe (project)", "argumentHint": "<word>"}]})
            await t.start("claude", [], "/tmp/ws")
        assert got == [[{"name": "orbital-probe", "description": "Probe (project)",
                         "invoke": "/orbital-probe ", "hint": "<word>", "enabled": True}]]

    @pytest.mark.asyncio
    async def test_no_server_info_publishes_nothing_and_start_succeeds(self):
        from agent_os.agent.transports.sdk_transport import SDKTransport
        for info in (None, {}, RuntimeError("old sdk")):
            t = SDKTransport()
            got, sink = _sink()
            t.set_commands_sink(sink)
            with patch("agent_os.agent.transports.sdk_transport.ClaudeSDKClient") as Client, \
                 patch("agent_os.agent.transports.sdk_transport.ClaudeAgentOptions"):
                Client.return_value.connect = AsyncMock()
                Client.return_value.get_server_info = (
                    AsyncMock(side_effect=info) if isinstance(info, Exception)
                    else AsyncMock(return_value=info))
                await t.start("claude", [], "/tmp/ws")
            assert got == []
            assert t._alive is True

    def test_sink_failure_never_raises(self):
        from agent_os.agent.transports.sdk_transport import SDKTransport
        t = SDKTransport()

        def boom(_):
            raise RuntimeError("sink down")
        t.set_commands_sink(boom)
        t._publish_commands([])  # logged, not raised
        t._publish_commands(None)  # malformed reports never reach the sink


# ----------------------------------------------------------------- Codex

FAKE_APP_SERVER = textwrap.dedent("""\
    import json, sys, threading, time
    delay = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    mode = sys.argv[3] if len(sys.argv) > 3 else "ok"
    lock = threading.Lock()
    def write(obj):
        with lock:
            sys.stdout.write(json.dumps(obj) + "\\n")
            sys.stdout.flush()
    def skills(i):
        # Answered late and concurrently, like the real app-server.
        time.sleep(delay)
        if mode == "error":
            write({"jsonrpc": "2.0", "id": i,
                   "error": {"code": -32601, "message": "no such method"}})
            return
        write({"jsonrpc": "2.0", "id": i, "result": {"data": [{"cwd": ".", "skills": [
            {"name": "imagegen", "description": "Make images", "enabled": True},
            {"name": "off", "description": "", "enabled": False}]}]}})
    for line in sys.stdin:
        msg = json.loads(line)
        m, i = msg.get("method"), msg.get("id")
        if i is None:
            continue
        if m == "initialize":
            r = {"userAgent": "codex_cli_rs/0.157.1"}
        elif m == "model/list":
            r = {"data": [{"id": "gpt-5.4"}]}
        elif m == "thread/start":
            r = {"thread": {"id": "T1", "path": "/x/rollout-T1.jsonl"}, "model": "gpt-5.4"}
        elif m == "skills/list":
            threading.Thread(target=skills, args=(i,), daemon=True).start()
            continue
        else:
            r = {}
        write({"jsonrpc": "2.0", "id": i, "result": r})
""")


def _fake_codex(tmp_path: Path, delay: float, mode: str = "ok") -> str:
    script = tmp_path / "fake_app_server.py"
    script.write_text(FAKE_APP_SERVER)
    exe = tmp_path / "codex"
    exe.write_text(f"#!/bin/sh\nexec '{sys.executable}' '{script}' \"$1\" {delay} {mode}\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell shim")
class TestCodex:
    @pytest.mark.asyncio
    async def test_skills_list_publishes_in_the_background(self, tmp_path):
        from agent_os.agent.transports.codex_transport import CodexTransport
        t = CodexTransport(autonomy=Autonomy.HANDS_OFF)
        got, sink = _sink()
        t.set_commands_sink(sink)
        began = time.monotonic()
        await t.start(_fake_codex(tmp_path, delay=1.5), [], str(tmp_path))
        try:
            # Thread open did not wait for the slow skills/list answer.
            assert time.monotonic() - began < 1.4
            assert t._thread_id == "T1"
            assert got == []
            for _ in range(60):
                if got:
                    break
                await asyncio.sleep(0.05)
            assert got == [[
                {"name": "imagegen", "description": "Make images",
                 "invoke": "$imagegen ", "hint": "", "enabled": True},
                {"name": "off", "description": "", "invoke": "$off ", "hint": "",
                 "enabled": False},
            ]]
        finally:
            await t.stop()

    @pytest.mark.asyncio
    async def test_error_response_publishes_nothing(self, tmp_path):
        from agent_os.agent.transports.codex_transport import CodexTransport
        t = CodexTransport(autonomy=Autonomy.HANDS_OFF)
        got, sink = _sink()
        t.set_commands_sink(sink)
        await t.start(_fake_codex(tmp_path, delay=0.0, mode="error"), [], str(tmp_path))
        try:
            await asyncio.sleep(0.5)
            assert got == []
            assert t._thread_id == "T1"
        finally:
            await t.stop()


# ------------------------------------------------------------------- ACP

FIXTURE = Path(__file__).parents[1] / "fixtures" / "dummy_acp_sdk_agent.py"


class TestACP:
    @pytest.mark.asyncio
    async def test_startup_announcement_is_published_and_emits_nothing(self, tmp_path):
        from agent_os.agent.transports.acp_sdk_transport import ACPSDKTransport
        t = ACPSDKTransport()
        got, sink = _sink()
        t.set_commands_sink(sink)
        await t.start(sys.executable, [str(FIXTURE)], str(tmp_path))
        try:
            for _ in range(40):
                if got:
                    break
                await asyncio.sleep(0.05)
            assert got[0] == [
                {"name": "compact", "description": "compact command",
                 "invoke": "/compact ", "hint": "instructions", "enabled": True},
                {"name": "status", "description": "status command",
                 "invoke": "/status ", "hint": "", "enabled": True},
            ]
            events = []
            while not t._event_queue.empty():
                events.append(t._event_queue.get_nowait().event_type)
            assert events == ["thread_started"]
        finally:
            await t.stop()

    @pytest.mark.asyncio
    async def test_mid_turn_reannouncement_is_published_not_an_event(self, tmp_path):
        from agent_os.agent.transports.acp_sdk_transport import ACPSDKTransport
        t = ACPSDKTransport()
        got, sink = _sink()
        t.set_commands_sink(sink)
        await t.start(sys.executable, [str(FIXTURE)], str(tmp_path))
        try:
            while not t._event_queue.empty():
                t._event_queue.get_nowait()
            await t.dispatch("commands")
            types = []
            while True:
                ev = await asyncio.wait_for(t._event_queue.get(), timeout=5)
                types.append(ev.event_type)
                if ev.event_type == "turn_complete":
                    break
            assert [c["name"] for c in got[-1]] == ["compact", "status", "review"]
            assert "status" not in types
            assert types == ["message", "turn_complete"]
        finally:
            await t.stop()


# ------------------------------------------------------------ cold probes

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell shim")
class TestColdProbes:
    @pytest.mark.asyncio
    async def test_claude_probe_runs_in_the_workspace_and_reads_commands(self, tmp_path):
        from agent_os.agent.transports.claude_models import fetch_claude_commands
        ws = tmp_path / "ws"
        ws.mkdir()
        script = tmp_path / "claude"
        # The fake names its cwd as the one command, proving where it ran.
        script.write_text(
            "#!/bin/sh\n"
            "read line\n"
            "d=$(basename \"$PWD\")\n"
            "printf '{\"type\":\"control_response\",\"response\":{\"subtype\":\"success\","
            "\"request_id\":\"orbital-models\",\"response\":{\"models\":[],\"commands\":"
            "[{\"name\":\"%s\",\"description\":\"d\",\"argumentHint\":\"\"}]}}}\\n' \"$d\"\n"
            "sleep 5\n")
        script.chmod(0o755)
        got = await fetch_claude_commands(str(script), str(ws), timeout=5)
        assert got == [{"name": "ws", "description": "d", "invoke": "/ws ",
                        "hint": "", "enabled": True}]

    @pytest.mark.asyncio
    async def test_claude_probe_failure_is_none(self, tmp_path):
        from agent_os.agent.transports.claude_models import fetch_claude_commands
        assert await fetch_claude_commands("/nonexistent/claude", str(tmp_path), timeout=2) is None

    @pytest.mark.asyncio
    async def test_codex_probe_reads_skills_list(self, tmp_path):
        from agent_os.agent.transports.codex_models import fetch_codex_skills
        got = await fetch_codex_skills(_fake_codex(tmp_path, delay=0.0), str(tmp_path), timeout=5)
        assert [c["invoke"] for c in got] == ["$imagegen ", "$off "]

    @pytest.mark.asyncio
    async def test_codex_probe_error_is_none(self, tmp_path):
        from agent_os.agent.transports.codex_models import fetch_codex_skills
        exe = _fake_codex(tmp_path, delay=0.0, mode="error")
        assert await fetch_codex_skills(exe, str(tmp_path), timeout=5) is None
