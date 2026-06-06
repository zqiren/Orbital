# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for CodexTransport (codex app-server JSON-RPC, pinned 0.125.0).

All payloads are verbatim from the probe traces in
artifacts-2026-06-06-codex-lifecycle/codex_appserver/traces/. No codex
process is spawned — _route_server_message is fed directly.
"""

import asyncio
import json

import pytest

from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.transports.base import TransportEvent
from agent_os.agent.transports.codex_transport import (
    CodexTransport,
    _POLICY_BY_AUTONOMY,
)


def _drain(transport) -> list[TransportEvent]:
    events = []
    while not transport._event_queue.empty():
        events.append(transport._event_queue.get_nowait())
    return events


def _transport(**kwargs) -> CodexTransport:
    t = CodexTransport(**kwargs)
    t._thread_id = "T1"
    t._effective_model = "gpt-5.4-mini"
    t._rollout_path = "/tmp/rollout-T1.jsonl"
    return t


class TestPolicyMapping:
    def test_untrusted_is_never_produced(self):
        # FINDINGS A4a: `untrusted` silently auto-rejects ALL escalation.
        for policy, _sandbox in _POLICY_BY_AUTONOMY.values():
            assert policy != "untrusted"

    def test_locked_mapping(self):
        assert _POLICY_BY_AUTONOMY[Autonomy.HANDS_OFF] == ("never", "workspace-write")
        assert _POLICY_BY_AUTONOMY[Autonomy.CHECK_IN] == ("on-request", "workspace-write")
        assert _POLICY_BY_AUTONOMY[Autonomy.SUPERVISED] == ("on-request", "workspace-write")


class TestCapability:
    def test_two_state_only(self):
        # LOCKED: truthful two-state — never flip this for UI parity.
        assert getattr(CodexTransport, "supports_background_status", False) is False


class TestNotificationRouting:
    @pytest.mark.asyncio
    async def test_turn_started_captures_turn_id_and_emits_nothing(self):
        t = _transport()
        await t._route_server_message({"jsonrpc": "2.0", "method": "turn/started", "params": {
            "threadId": "T1", "turn": {"id": "U1", "status": "inProgress"}}})
        assert t._turn_id == "U1"
        assert _drain(t) == []

    @pytest.mark.asyncio
    async def test_supplementary_status_never_emits(self):
        # TEST RULE 1 ingredient: these must not produce ANY event (and
        # therefore can never flip idle).
        t = _transport()
        for method in ("thread/status/changed", "thread/tokenUsage/updated",
                       "account/rateLimits/updated", "mcpServer/startupStatus/updated"):
            await t._route_server_message({"jsonrpc": "2.0", "method": method, "params": {}})
        assert _drain(t) == []

    @pytest.mark.asyncio
    async def test_turn_completed_emits_turn_complete_with_resume_identity(self):
        t = _transport()
        t._begin_turn()
        t._turn_id = "U1"
        await t._route_server_message({"jsonrpc": "2.0", "method": "turn/completed", "params": {
            "threadId": "T1",
            "turn": {"id": "U1", "status": "completed", "durationMs": 9841}}})
        events = _drain(t)
        assert [e.event_type for e in events] == ["turn_complete"]
        assert events[0].data == {
            "cause": "success", "session_id": "T1",
            "model": "gpt-5.4-mini", "rollout_path": "/tmp/rollout-T1.jsonl",
        }
        assert t._turn_id is None  # cleared — interrupt now impossible

    @pytest.mark.asyncio
    async def test_turn_interrupted_while_alive_maps_to_interrupted(self):
        # Review correction: a cancel decision ends the turn `interrupted`
        # with NO teardown — the management session may be awaiting, and
        # cause="stopped" routes to silence (the Part-C hang class).
        t = _transport()
        t._begin_turn()
        await t._route_server_message({"jsonrpc": "2.0", "method": "turn/completed", "params": {
            "turn": {"id": "U1", "status": "interrupted", "durationMs": 11489}}})
        [event] = _drain(t)
        assert event.data["cause"] == "interrupted"

    @pytest.mark.asyncio
    async def test_turn_interrupted_during_teardown_maps_to_stopped(self):
        # Teardown interruptions stay silent on this channel: stop_for_user's
        # on_user_stopped speaks there — a second notice would double-report.
        t = _transport()
        t._begin_turn()
        t._stopping = True
        await t._route_server_message({"jsonrpc": "2.0", "method": "turn/completed", "params": {
            "turn": {"id": "U1", "status": "interrupted"}}})
        [event] = _drain(t)
        assert event.data["cause"] == "stopped"

    @pytest.mark.asyncio
    async def test_turn_failed_maps_to_error(self):
        t = _transport()
        t._begin_turn()
        await t._route_server_message({"jsonrpc": "2.0", "method": "turn/completed", "params": {
            "turn": {"id": "U1", "status": "failed"}}})
        [event] = _drain(t)
        assert event.data["cause"] == "error"


class TestItemRouting:
    @pytest.mark.asyncio
    async def test_command_execution_items_map_to_tool_use(self):
        t = _transport()
        item = {"type": "commandExecution", "id": "call_6MP",
                "command": "/bin/zsh -lc \"python3 -c 'print(6*7)'\"",
                "cwd": "/tmp/ws", "processId": "29276",
                "source": "unifiedExecStartup", "status": "completed",
                "aggregatedOutput": "42\n", "exitCode": 0, "durationMs": 0}
        await t._route_server_message({"jsonrpc": "2.0", "method": "item/completed",
                                       "params": {"item": item}})
        [event] = _drain(t)
        assert event.event_type == "tool_use"
        assert event.data["tool_name"] == "commandExecution"
        assert event.data["tool_input"]["command"] == item["command"]
        assert "run_in_background" not in event.data["tool_input"]  # provenance inert

    @pytest.mark.asyncio
    async def test_file_change_items_map_to_tool_use_with_diff(self):
        t = _transport()
        item = {"type": "fileChange", "id": "call_Fbr", "status": "completed",
                "changes": [{"path": "/tmp/ws/hello.txt",
                             "kind": {"type": "add"},
                             "diff": "hello from codex probe\n"}]}
        await t._route_server_message({"jsonrpc": "2.0", "method": "item/completed",
                                       "params": {"item": item}})
        [event] = _drain(t)
        assert event.event_type == "tool_use"
        assert event.data["tool_name"] == "fileChange"
        assert event.data["tool_input"]["changes"][0]["diff"] == "hello from codex probe\n"

    @pytest.mark.asyncio
    async def test_agent_message_deltas_accumulate_full_text_emits_once(self):
        # ProcessManager treats every "message" chunk as a complete message
        # (broadcast + summary) — per-token events would corrupt summaries.
        t = _transport()
        for delta in ("I", "'ll", " run"):
            await t._route_server_message({"jsonrpc": "2.0",
                "method": "item/agentMessage/delta",
                "params": {"threadId": "T1", "turnId": "U1",
                           "itemId": "msg_1", "delta": delta}})
        assert _drain(t) == []  # nothing emitted yet
        await t._route_server_message({"jsonrpc": "2.0", "method": "item/completed",
            "params": {"item": {"type": "agentMessage", "id": "msg_1",
                                "text": "I'll run", "phase": "commentary"}}})
        [event] = _drain(t)
        assert event.event_type == "message"
        assert event.raw_text == "I'll run"  # item.text authoritative

    @pytest.mark.asyncio
    async def test_final_answer_is_a_message_event(self):
        t = _transport()
        await t._route_server_message({"jsonrpc": "2.0", "method": "item/completed",
            "params": {"item": {"type": "agentMessage", "id": "msg_2",
                                "text": "Done.", "phase": "final_answer"}}})
        [event] = _drain(t)
        assert (event.event_type, event.raw_text) == ("message", "Done.")
        assert event.data["phase"] == "final_answer"

    @pytest.mark.asyncio
    async def test_interrupted_mid_message_flushes_partial_text(self):
        t = _transport()
        t._begin_turn()
        await t._route_server_message({"jsonrpc": "2.0",
            "method": "item/agentMessage/delta",
            "params": {"itemId": "msg_3", "delta": "partial answe"}})
        await t._route_server_message({"jsonrpc": "2.0", "method": "turn/completed",
            "params": {"turn": {"id": "U1", "status": "interrupted"}}})
        events = _drain(t)
        assert [e.event_type for e in events] == ["message", "turn_complete"]
        assert events[0].raw_text == "partial answe"


class TestErrors:
    @pytest.mark.asyncio
    async def test_error_notification_maps_to_error_event(self):
        t = _transport()
        await t._route_server_message({"jsonrpc": "2.0", "method": "error",
                                       "params": {"message": "boom"}})
        [event] = _drain(t)
        assert event.event_type == "error"
