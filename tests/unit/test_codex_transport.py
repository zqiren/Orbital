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
import os

import psutil
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


class _FakeStdin:
    """Captures JSON-RPC frames written by the transport."""
    def __init__(self):
        self.frames: list[dict] = []
    def write(self, data: bytes) -> None:
        self.frames.append(json.loads(data.decode("utf-8")))
    async def drain(self) -> None:
        pass
    def close(self) -> None:
        pass


def _wired(**kwargs) -> tuple[CodexTransport, _FakeStdin]:
    t = _transport(**kwargs)
    stdin = _FakeStdin()

    class _P:  # minimal Popen stand-in
        pid = 4242
        returncode = None
        def kill(self):  # real asyncio.subprocess.Process has this
            pass
    p = _P()
    p.stdin = stdin
    t._popen = p
    t._alive = True
    return t, stdin


class TestDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_sends_turn_start_and_captures_turn_id(self):
        t, stdin = _wired()

        async def run():
            await t.dispatch("do the thing")
        task = asyncio.create_task(run())
        await asyncio.sleep(0.05)
        [frame] = stdin.frames
        assert frame["method"] == "turn/start"
        assert frame["params"] == {
            "threadId": "T1",
            "input": [{"type": "text", "text": "do the thing"}],
        }
        # turn/start response carries the ack + turn id (FINDINGS A1)
        await t._route_server_message({"jsonrpc": "2.0", "id": frame["id"],
            "result": {"turn": {"id": "U7", "status": "inProgress"}}})
        await asyncio.wait_for(task, timeout=2.0)
        assert t._turn_id == "U7"

    @pytest.mark.asyncio
    async def test_dispatch_error_response_closes_turn_honestly(self):
        t, stdin = _wired()
        task = asyncio.create_task(t.dispatch("x"))
        await asyncio.sleep(0.05)
        await t._route_server_message({"jsonrpc": "2.0", "id": stdin.frames[0]["id"],
            "error": {"code": -32600, "message": "Invalid request: missing field `x`"}})
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(task, timeout=2.0)
        events = _drain(t)
        assert [e.event_type for e in events] == ["error", "turn_complete"]
        assert events[1].data["cause"] == "error"

    @pytest.mark.asyncio
    async def test_external_sandbox_adds_turn_sandbox_policy(self):
        # Constraint 7: the tagged object exists ONLY on turn/start.
        t, stdin = _wired(external_sandbox=True)
        task = asyncio.create_task(t.dispatch("x"))
        await asyncio.sleep(0.05)
        assert stdin.frames[0]["params"]["sandboxPolicy"] == {"type": "externalSandbox"}
        await t._route_server_message({"jsonrpc": "2.0", "id": stdin.frames[0]["id"],
            "result": {"turn": {"id": "U1"}}})
        await asyncio.wait_for(task, timeout=2.0)


class TestApprovals:
    REQUEST = {"jsonrpc": "2.0", "id": 0,
               "method": "item/commandExecution/requestApproval",
               "params": {"threadId": "T1", "turnId": "U1", "itemId": "call_B",
                          "reason": "Do you want to allow creating the file?",
                          "command": "/bin/zsh -lc 'touch marker.txt'",
                          "cwd": "/tmp/ws",
                          "availableDecisions": [
                              "accept",
                              {"acceptWithExecpolicyAmendment": {
                                  "execpolicy_amendment": ["touch", "marker.txt"]}},
                              "cancel"]}}

    @pytest.mark.asyncio
    async def test_request_surfaces_permission_event_with_card_payload(self):
        t, stdin = _wired(autonomy=Autonomy.CHECK_IN)  # not auto-approved
        await t._route_server_message(dict(self.REQUEST))
        [event] = _drain(t)
        assert event.event_type == "permission_request"
        rid = event.data["request_id"]
        assert rid in t._pending_approvals
        assert t._pending_approval_data[rid]["tool_name"] == "commandExecution"
        ti = t._pending_approval_data[rid]["tool_input"]
        assert ti["command"] == "/bin/zsh -lc 'touch marker.txt'"
        assert ti["reason"].startswith("Do you want")
        # non-string decision variants (amendment objects) are filtered
        assert ti["availableDecisions"] == ["accept", "cancel"]
        assert stdin.frames == []  # no response until the user decides

    @pytest.mark.asyncio
    async def test_bool_deny_maps_to_decline_turn_continues(self):
        # FINDINGS A4d: decline -> item declined, turn CONTINUES.
        t, stdin = _wired(autonomy=Autonomy.CHECK_IN)
        await t._route_server_message(dict(self.REQUEST))
        rid = _drain(t)[0].data["request_id"]
        await t.respond_to_permission(rid, approved=False)
        assert stdin.frames[-1] == {"jsonrpc": "2.0", "id": 0,
                                    "result": {"decision": "decline"}}
        assert rid not in t._pending_approvals

    @pytest.mark.asyncio
    async def test_bool_approve_maps_to_accept(self):
        t, stdin = _wired(autonomy=Autonomy.CHECK_IN)
        await t._route_server_message(dict(self.REQUEST))
        rid = _drain(t)[0].data["request_id"]
        await t.respond_to_permission(rid, approved=True)
        assert stdin.frames[-1]["result"] == {"decision": "accept"}

    @pytest.mark.asyncio
    async def test_explicit_cancel_decision(self):
        t, stdin = _wired(autonomy=Autonomy.CHECK_IN)
        await t._route_server_message(dict(self.REQUEST))
        rid = _drain(t)[0].data["request_id"]
        await t.respond_to_permission_decision(rid, "cancel")
        assert stdin.frames[-1]["result"] == {"decision": "cancel"}

    @pytest.mark.asyncio
    async def test_hands_off_auto_accepts_without_surfacing(self):
        t, stdin = _wired(autonomy=Autonomy.HANDS_OFF)
        await t._route_server_message(dict(self.REQUEST))
        assert _drain(t) == []
        assert stdin.frames[-1]["result"] == {"decision": "accept"}

    @pytest.mark.asyncio
    async def test_file_change_surface_handled(self):
        t, stdin = _wired(autonomy=Autonomy.CHECK_IN)
        await t._route_server_message({"jsonrpc": "2.0", "id": 5,
            "method": "item/fileChange/requestApproval",
            "params": {"threadId": "T1", "turnId": "U1", "itemId": "call_F",
                       "reason": "write outside workspace", "grantRoot": "/etc"}})
        [event] = _drain(t)
        rid = event.data["request_id"]
        assert t._pending_approval_data[rid]["tool_name"] == "fileChange"
        await t.respond_to_permission(rid, approved=False)
        assert stdin.frames[-1] == {"jsonrpc": "2.0", "id": 5,
                                    "result": {"decision": "decline"}}

    @pytest.mark.asyncio
    async def test_unknown_surface_answered_with_error_never_hangs(self):
        # item/permissions/requestApproval has a non-decision response shape;
        # an unanswered server request blocks codex forever.
        t, stdin = _wired(autonomy=Autonomy.CHECK_IN)
        await t._route_server_message({"jsonrpc": "2.0", "id": 9,
            "method": "item/permissions/requestApproval",
            "params": {"threadId": "T1", "turnId": "U1"}})
        [event] = _drain(t)
        assert event.event_type == "error"
        assert stdin.frames[-1]["id"] == 9
        assert "error" in stdin.frames[-1]


class TestStop:
    @pytest.mark.asyncio
    async def test_stop_cancels_pending_approvals_then_interrupts(self, monkeypatch):
        # Stop while a question is open == "Deny & stop": cancel, then
        # turn/interrupt (threadId AND turnId), then unsubscribe, then
        # tree-kill. We assert frame ORDER; the kill is patched out.
        t, stdin = _wired(autonomy=Autonomy.CHECK_IN)
        t._turn_id = "U1"
        await t._route_server_message(dict(TestApprovals.REQUEST))
        _drain(t)

        async def fake_kill(proc, **kw):
            from agent_os.agent.transports.process_kill import KillOutcome
            return KillOutcome(parent_dead=True)
        monkeypatch.setattr(
            "agent_os.agent.transports.process_kill.kill_process_tree", fake_kill)
        t._proc = psutil.Process()  # self — never signalled (kill patched)

        async def answer_requests():
            # resolve interrupt + unsubscribe requests as the reader would
            for _ in range(40):
                await asyncio.sleep(0.05)
                for f in list(stdin.frames):
                    rid = f.get("id")
                    if f.get("method") in ("turn/interrupt", "thread/unsubscribe") \
                            and rid in t._response_futures \
                            and not t._response_futures[rid].done():
                        await t._route_server_message(
                            {"jsonrpc": "2.0", "id": rid, "result": {}})
        answer = asyncio.create_task(answer_requests())
        try:
            await asyncio.wait_for(t.stop(), timeout=10.0)
        finally:
            answer.cancel()
            await asyncio.gather(answer, return_exceptions=True)

        methods = [f.get("method") or ("response" if "result" in f else "?")
                   for f in stdin.frames]
        # 1st frame: cancel response to the pending approval
        assert stdin.frames[0] == {"jsonrpc": "2.0", "id": 0,
                                   "result": {"decision": "cancel"}}
        assert "turn/interrupt" in methods and "thread/unsubscribe" in methods
        assert methods.index("turn/interrupt") < methods.index("thread/unsubscribe")
        interrupt = next(f for f in stdin.frames if f.get("method") == "turn/interrupt")
        assert interrupt["params"] == {"threadId": "T1", "turnId": "U1"}
        assert t.is_alive() is False

    @pytest.mark.asyncio
    async def test_stop_without_open_turn_skips_interrupt(self, monkeypatch):
        t, stdin = _wired()
        t._turn_id = None

        async def fake_kill(proc, **kw):
            from agent_os.agent.transports.process_kill import KillOutcome
            return KillOutcome(parent_dead=True)
        monkeypatch.setattr(
            "agent_os.agent.transports.process_kill.kill_process_tree", fake_kill)

        async def answer():
            for _ in range(40):
                await asyncio.sleep(0.05)
                for f in list(stdin.frames):
                    rid = f.get("id")
                    if f.get("method") == "thread/unsubscribe" \
                            and rid in t._response_futures \
                            and not t._response_futures[rid].done():
                        await t._route_server_message(
                            {"jsonrpc": "2.0", "id": rid, "result": {}})
        ans = asyncio.create_task(answer())
        try:
            await asyncio.wait_for(t.stop(), timeout=10.0)
        finally:
            ans.cancel()
            await asyncio.gather(ans, return_exceptions=True)
        assert all(f.get("method") != "turn/interrupt" for f in stdin.frames)


class TestManagerResolution:
    def _manager(self):
        from unittest.mock import MagicMock
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        return SubAgentManager(
            process_manager=MagicMock(), registry=MagicMock(),
            setup_engine=MagicMock(), project_store=MagicMock(),
        )

    def test_resolve_transport_codex_appserver(self):
        from unittest.mock import MagicMock
        mgr = self._manager()
        manifest = MagicMock()
        manifest.runtime.transport = "codex-appserver"
        manifest.runtime.mode = "interactive"
        manifest.runtime.command = "codex"
        transport = mgr._resolve_transport(
            manifest, {}, autonomy=Autonomy.HANDS_OFF,
            resume_record={"session_id": "T9", "model": "gpt-5.4-mini"},
        )
        assert isinstance(transport, CodexTransport)
        # the manager's honesty downgrade reads this attribute by name
        assert transport._resume_session_id == "T9"
        # Model is config (AMENDS piece 2): the record's model must NOT be
        # consulted — it resolves from argv at start().
        assert transport._model is None

    def test_codex_manifest_declares_appserver_transport(self):
        import os
        from agent_os.agents.registry import AgentRegistry
        manifests = os.path.join(os.path.dirname(__file__), "..", "..",
                                 "agent_os", "agents", "manifests")
        registry = AgentRegistry()
        registry.load_directory(manifests)
        manifest = registry.get("codex")
        assert manifest is not None
        assert manifest.runtime.transport == "codex-appserver"


class TestResumePreCheck:
    def test_rollout_path_hit(self, tmp_path):
        rollout = tmp_path / "rollout-2026-06-06T21-19-11-T9.jsonl"
        rollout.write_text("{}\n")
        assert CodexTransport.resume_source_exists(
            {"session_id": "T9", "rollout_path": str(rollout)}) is True

    def test_rollout_path_pruned_falls_to_glob_then_false(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))  # empty sessions dir
        assert CodexTransport.resume_source_exists(
            {"session_id": "T9", "rollout_path": str(tmp_path / "gone.jsonl")}) is False

    def test_glob_fallback_finds_by_thread_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))
        d = tmp_path / "sessions" / "2026" / "06" / "06"
        d.mkdir(parents=True)
        (d / "rollout-2026-06-06T21-19-11-T9.jsonl").write_text("{}\n")
        assert CodexTransport.resume_source_exists({"session_id": "T9"}) is True


class TestRolloutPersistenceChain:
    def test_set_sub_agent_thread_accepts_rollout_path(self, tmp_path):
        from agent_os.agent.session import Session
        s = Session.new(
            "sess_x_ab12cd34", str(tmp_path), "proj_test"
        )
        s.append({"role": "user", "content": "hi", "source": "user"})
        s.set_sub_agent_thread("codex", session_id="T9",
                               model="gpt-5.4-mini",
                               rollout_path="/x/rollout-T9.jsonl")
        rec = s.get_sub_agent_thread("codex")
        assert rec["rollout_path"] == "/x/rollout-T9.jsonl"
        assert rec["session_id"] == "T9"

    def test_determine_resume_dispatches_codex_pre_check(self, tmp_path, monkeypatch):
        """codex-appserver manifests pre-check the ROLLOUT file, not the
        claude store. A live rollout -> ('resumed'); a pruned one -> fresh."""
        from unittest.mock import MagicMock
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        mgr = SubAgentManager(
            process_manager=MagicMock(), registry=MagicMock(),
            setup_engine=MagicMock(), project_store=MagicMock(),
        )
        rollout = tmp_path / "rollout-2026-06-06T21-19-11-T9.jsonl"
        rollout.write_text("{}\n")
        record = {"session_id": "T9", "model": "gpt-5.4-mini",
                  "rollout_path": str(rollout)}
        session = MagicMock()
        session.get_sub_agent_thread.return_value = record
        mgr._session_resolver = lambda pid, sid: session
        manifest = MagicMock()
        manifest.runtime.transport = "codex-appserver"
        mgr._registry.get.return_value = manifest
        # Part-F backstop: no live attachment recorded -> safe to resume
        monkeypatch.setattr(mgr, "_ensure_no_live_attachment",
                            lambda *a, **k: True)
        rec, status, reason = mgr._determine_resume(
            str(tmp_path), "proj_x", "codex", "sess_1")
        assert (status, reason) == ("resumed", None)
        assert rec is record
        # pruned rollout (and no glob fallback hit) -> honest fresh
        monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty"))
        rollout.unlink()
        rec2, status2, reason2 = mgr._determine_resume(
            str(tmp_path), "proj_x", "codex", "sess_1")
        assert (rec2, status2, reason2) == (None, "fresh", "resume_failed")


class TestDenyAndStopWire:
    @pytest.mark.asyncio
    async def test_decision_pass_through_reaches_transport(self):
        from unittest.mock import AsyncMock, MagicMock
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        mgr = SubAgentManager(
            process_manager=MagicMock(), registry=MagicMock(),
            setup_engine=MagicMock(), project_store=MagicMock(),
        )
        transport = MagicMock()
        transport._pending_approvals = {"req-1": {"rpc_id": 0}}
        transport.respond_to_permission = AsyncMock()
        transport.respond_to_permission_decision = AsyncMock()
        adapter = MagicMock()
        adapter._transport = transport
        mgr._adapters[("p1", "s1")] = {"codex": adapter}

        routed = await mgr.resolve_sub_agent_approval(
            "p1", "req-1", approved=False, session_id="s1", decision="cancel")
        assert routed is True
        transport.respond_to_permission_decision.assert_awaited_once_with(
            "req-1", "cancel")
        transport.respond_to_permission.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_decision_keeps_bool_path(self):
        from unittest.mock import AsyncMock, MagicMock
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        mgr = SubAgentManager(
            process_manager=MagicMock(), registry=MagicMock(),
            setup_engine=MagicMock(), project_store=MagicMock(),
        )
        transport = MagicMock()
        transport._pending_approvals = {"req-1": {}}
        transport.respond_to_permission = AsyncMock()
        adapter = MagicMock()
        adapter._transport = transport
        mgr._adapters[("p1", "s1")] = {"codex": adapter}
        await mgr.resolve_sub_agent_approval("p1", "req-1", approved=False,
                                             session_id="s1")
        transport.respond_to_permission.assert_awaited_once_with("req-1", False)


class TestStartupModelResolution:
    """Cold-start 400 killed at the source (TASK-codex-startup-model):
    a FRESH thread never opens on the unqualified server default. The
    preference order over the live model/list is the point — a hardcoded
    single model would re-plant the trap on the next OpenAI model churn."""

    def _stub_list(self, t, ids):
        async def fake_request(method, params=None, timeout=30.0):
            assert method == "model/list"
            return {"data": [{"id": i} for i in ids]}
        t._request = fake_request

    @pytest.mark.asyncio
    async def test_picks_first_preference_when_present(self):
        t = _transport()
        self._stub_list(t, ["gpt-5.3-codex", "gpt-5.5", "gpt-5.2",
                            "gpt-5.4", "gpt-5.4-mini"])
        assert await t._resolve_startup_model() == "gpt-5.4-mini"

    @pytest.mark.asyncio
    async def test_skips_to_next_available_on_churn(self):
        # THE churn-robustness case — the reason for querying vs hardcoding.
        t = _transport()
        self._stub_list(t, ["gpt-5.3-codex", "gpt-5.5", "gpt-5.4"])
        assert await t._resolve_startup_model() == "gpt-5.4"
        self._stub_list(t, ["gpt-5.3-codex", "gpt-5.5"])
        assert await t._resolve_startup_model() == "gpt-5.5"

    @pytest.mark.asyncio
    async def test_non_codex_fallback_when_preferences_all_retired(self):
        # codex-class ids are the known-rejected class on ChatGPT auth.
        t = _transport()
        self._stub_list(t, ["gpt-9.9-codex", "gpt-9.9"])
        assert await t._resolve_startup_model() == "gpt-9.9"

    @pytest.mark.asyncio
    async def test_hard_fallback_when_list_empty(self):
        t = _transport()
        self._stub_list(t, [])
        assert await t._resolve_startup_model() == "gpt-5.4-mini"

    @pytest.mark.asyncio
    async def test_model_list_failure_degrades_loudly_never_blocks(self, caplog):
        import logging as _logging
        t = _transport()
        async def boom(method, params=None, timeout=30.0):
            raise RuntimeError("network down")
        t._request = boom
        with caplog.at_level(_logging.WARNING):
            assert await t._resolve_startup_model() == "gpt-5.4-mini"
        assert any("model/list" in r.message for r in caplog.records), \
            "degradation must be surfaced, not silent"


# ---------------------------------------------------------------------------
# P3-B: display-only sub-agent usage capture (thread/tokenUsage/updated)
# ---------------------------------------------------------------------------

def _token_usage_notif(thread_id, turn_id, *, last, total=None):
    """A verbatim-shaped thread/tokenUsage/updated notification.

    Schema [OBSERVED] in artifacts-2026-06-06-codex-lifecycle/.../traces:
    params = {threadId, turnId, tokenUsage: {last, total, modelContextWindow?}}.
    """
    return {
        "jsonrpc": "2.0",
        "method": "thread/tokenUsage/updated",
        "params": {
            "threadId": thread_id,
            "turnId": turn_id,
            "tokenUsage": {
                "last": last,
                "total": total if total is not None else last,
                "modelContextWindow": 258400,
            },
        },
    }


def _ledger_lines(workspace):
    from agent_os.budget.ledger import ledger_path
    path = ledger_path(workspace)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _capturing_transport(workspace):
    """A transport wired to a real workspace so append_event lands on disk."""
    t = _transport()
    t._workspace = str(workspace)
    return t


class TestCodexUsageCapture:
    @pytest.mark.asyncio
    async def test_repeated_notifications_one_event_with_final_last(self, tmp_path):
        """The notification fires repeatedly per turn with cumulative `total`
        and per-turn `last`; we ledger exactly ONE event for the turn using the
        LAST `last` value seen (replace-tracking, emit on turn boundary)."""
        t = _capturing_transport(tmp_path)
        t._begin_turn()
        t._turn_id = "U1"
        # First notification (turn so far): last == total.
        await t._route_server_message(_token_usage_notif(
            "T1", "U1",
            last={"totalTokens": 12189, "inputTokens": 11914,
                  "cachedInputTokens": 9088, "outputTokens": 275,
                  "reasoningOutputTokens": 137}))
        # Second notification (later in same turn): cumulative total grows, the
        # per-turn `last` is the FINAL per-turn breakdown — this is what we keep.
        await t._route_server_message(_token_usage_notif(
            "T1", "U1",
            last={"totalTokens": 12308, "inputTokens": 12223,
                  "cachedInputTokens": 11648, "outputTokens": 85,
                  "reasoningOutputTokens": 56},
            total={"totalTokens": 24497, "inputTokens": 24137,
                   "cachedInputTokens": 20736, "outputTokens": 360,
                   "reasoningOutputTokens": 193}))
        # No ledger line until the turn boundary.
        assert _ledger_lines(tmp_path) == []
        # Turn completes → exactly one event from the LAST `last`.
        await t._route_server_message({"jsonrpc": "2.0", "method": "turn/completed",
            "params": {"threadId": "T1",
                       "turn": {"id": "U1", "status": "completed"}}})
        lines = _ledger_lines(tmp_path)
        assert len(lines) == 1
        ev = lines[0]
        assert ev["source"] == "subagent:codex"
        assert ev["provider"] == "openai"
        assert ev["session_id"] == "T1"
        # Subset semantics: uncached = inputTokens - cachedInputTokens
        # = 12223 - 11648 = 575; cache_read = 11648; cache_write = 0; out = 85.
        assert ev["uncached_input"] == 575
        assert ev["cache_read"] == 11648
        assert ev["cache_write"] == 0
        assert ev["output"] == 85
        # Disjoint sum reconstructs the per-turn totalTokens (12308).
        assert (ev["uncached_input"] + ev["cache_read"]
                + ev["cache_write"] + ev["output"]) == 12308
        # Codex emits no cost.
        assert "reported_cost" not in ev

    @pytest.mark.asyncio
    async def test_multi_turn_one_event_per_turn(self, tmp_path):
        """Two turns in one thread → one ledger event per (threadId, turnId)."""
        t = _capturing_transport(tmp_path)
        # Turn 1.
        t._begin_turn()
        t._turn_id = "U1"
        await t._route_server_message(_token_usage_notif(
            "T1", "U1",
            last={"totalTokens": 12104, "inputTokens": 11989,
                  "cachedInputTokens": 9088, "outputTokens": 115,
                  "reasoningOutputTokens": 0}))
        await t._route_server_message({"jsonrpc": "2.0", "method": "turn/completed",
            "params": {"threadId": "T1",
                       "turn": {"id": "U1", "status": "completed"}}})
        # Turn 2.
        t._begin_turn()
        t._turn_id = "U2"
        await t._route_server_message(_token_usage_notif(
            "T1", "U2",
            last={"totalTokens": 12257, "inputTokens": 12251,
                  "cachedInputTokens": 12160, "outputTokens": 6,
                  "reasoningOutputTokens": 0}))
        await t._route_server_message({"jsonrpc": "2.0", "method": "turn/completed",
            "params": {"threadId": "T1",
                       "turn": {"id": "U2", "status": "completed"}}})
        lines = _ledger_lines(tmp_path)
        assert len(lines) == 2
        # Turn 1: uncached = 11989 - 9088 = 2901, out 115.
        assert lines[0]["uncached_input"] == 2901
        assert lines[0]["output"] == 115
        # Turn 2: uncached = 12251 - 12160 = 91, out 6.
        assert lines[1]["uncached_input"] == 91
        assert lines[1]["output"] == 6

    @pytest.mark.asyncio
    async def test_turn_with_no_usage_emits_no_ledger_line(self, tmp_path):
        """A turn that never received a tokenUsage notification ledgers nothing
        (and still completes cleanly)."""
        t = _capturing_transport(tmp_path)
        t._begin_turn()
        t._turn_id = "U1"
        await t._route_server_message({"jsonrpc": "2.0", "method": "turn/completed",
            "params": {"threadId": "T1",
                       "turn": {"id": "U1", "status": "completed"}}})
        assert _ledger_lines(tmp_path) == []
        # turn_complete still flowed.
        assert any(e.event_type == "turn_complete" for e in _drain(t))

    @pytest.mark.asyncio
    async def test_malformed_notification_skipped_turn_unaffected(self, tmp_path):
        """A tokenUsage notification missing the `last` breakdown is skipped;
        the turn is unaffected and ledgers nothing."""
        t = _capturing_transport(tmp_path)
        t._begin_turn()
        t._turn_id = "U1"
        # Missing tokenUsage.last entirely.
        await t._route_server_message({"jsonrpc": "2.0",
            "method": "thread/tokenUsage/updated",
            "params": {"threadId": "T1", "turnId": "U1",
                       "tokenUsage": {"total": {"totalTokens": 5}}}})
        await t._route_server_message({"jsonrpc": "2.0", "method": "turn/completed",
            "params": {"threadId": "T1",
                       "turn": {"id": "U1", "status": "completed"}}})
        assert _ledger_lines(tmp_path) == []

    @pytest.mark.asyncio
    async def test_tokenusage_notification_emits_no_transport_event(self, tmp_path):
        """The notification must NEVER produce a transport event (it cannot
        drive idle) — only the deferred ledger write happens on turn boundary."""
        t = _capturing_transport(tmp_path)
        t._begin_turn()
        t._turn_id = "U1"
        await t._route_server_message(_token_usage_notif(
            "T1", "U1",
            last={"totalTokens": 12104, "inputTokens": 11989,
                  "cachedInputTokens": 9088, "outputTokens": 115,
                  "reasoningOutputTokens": 0}))
        assert _drain(t) == []  # no event queued by the notification itself

    @pytest.mark.asyncio
    async def test_capture_never_raises_on_bad_workspace(self):
        """An unwritable workspace must not propagate out of capture."""
        t = _transport()
        t._workspace = ""  # ledger_path("") unwritable; swallowed
        t._begin_turn()
        t._turn_id = "U1"
        await t._route_server_message(_token_usage_notif(
            "T1", "U1",
            last={"totalTokens": 100, "inputTokens": 90,
                  "cachedInputTokens": 10, "outputTokens": 10,
                  "reasoningOutputTokens": 0}))
        # Must NOT raise.
        await t._route_server_message({"jsonrpc": "2.0", "method": "turn/completed",
            "params": {"threadId": "T1",
                       "turn": {"id": "U1", "status": "completed"}}})
