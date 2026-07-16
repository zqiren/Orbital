from types import SimpleNamespace

import pytest

from agent_os.agent.adapters.base import OutputChunk
from agent_os.daemon_v2.models import make_session_key
from agent_os.daemon_v2.process_manager import ProcessManager
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager


class _ProcessManager:
    def set_turn_closed_callback(self, callback):
        self.turn_closed = callback

    def set_permission_request_callback(self, callback):
        self.permission = callback


class _PermissionTransport:
    def __init__(self):
        self._pending_approvals = {
            "perm-1": object(), "perm-2": object(), "perm-3": object()}
        self._pending_approval_data = {
            "perm-3": {
                "request_id": "perm-3",
                "tool_name": "shell",
                "tool_input": {"command": "pwd"},
                "options": [
                    {"optionId": "allow-once", "kind": "allow_once"},
                    {"optionId": "reject-once", "kind": "reject_once"},
                ],
            }
        }
        self.responses = []

    async def respond_to_permission(self, permission_id, approved):
        self.responses.append((permission_id, approved, None, None, None))
        self._pending_approvals.pop(permission_id, None)

    async def respond_to_permission_response(
        self, permission_id, *, approved, reply_text, temporary_allow_s,
        decision,
    ):
        self.responses.append(
            (permission_id, approved, reply_text, temporary_allow_s, decision))
        self._pending_approvals.pop(permission_id, None)


@pytest.mark.asyncio
async def test_permission_card_preserves_guidance_and_temporary_approve_all():
    manager = SubAgentManager(_ProcessManager())
    transport = _PermissionTransport()
    adapter = SimpleNamespace(_transport=transport)
    manager._adapters[make_session_key("p1", "s1")] = {"cursor": adapter}

    routed = await manager.resolve_sub_agent_approval(
        "p1", "perm-1", True, session_id="s1",
        reply_text="只修改这个文件", approve_all=True,
    )

    assert routed is True
    assert transport.responses == [
        ("perm-1", True, "只修改这个文件", 600, None)]

    # A later ask-mode request inside the same task is auto-approved by
    # Orbital's temporary bypass, without asking the provider for a permanent
    # allow-always grant.
    handled = await manager._on_permission_request(
        "p1", "cursor", "perm-2", session_id="s1")
    assert handled is True
    assert transport.responses[-1] == ("perm-2", True, None, None, None)


@pytest.mark.asyncio
async def test_deny_and_stop_preserves_cancel_decision():
    manager = SubAgentManager(_ProcessManager())
    transport = _PermissionTransport()
    manager._adapters[make_session_key("p1", "s1")] = {
        "cursor": SimpleNamespace(_transport=transport)}

    routed = await manager.resolve_sub_agent_approval(
        "p1", "perm-3", False, session_id="s1", decision="cancel")

    assert routed is True
    assert transport.responses[-1] == (
        "perm-3", False, None, None, "cancel")


def test_pending_permission_recovery_keeps_provider_options():
    manager = SubAgentManager(_ProcessManager())
    transport = _PermissionTransport()
    manager._adapters[make_session_key("p1", "s1")] = {
        "cursor": SimpleNamespace(_transport=transport)}

    # Narrow to the ACP request that carries recovery metadata.
    transport._pending_approvals = {"perm-3": object()}
    pending = manager.get_pending_sub_agent_approval("p1", session_id="s1")
    assert pending["tool_args"]["command"] == "pwd"
    assert pending["tool_args"]["permission_options"][0]["optionId"] == "allow-once"


class _WS:
    def __init__(self):
        self.events = []

    def broadcast(self, project_id, payload):
        self.events.append(payload)


class _Activity:
    def on_message(self, *args, **kwargs):
        pass


class _PermissionStream:
    _transport = None

    async def read_stream(self):
        yield OutputChunk(
            text="permission",
            chunk_type="approval_request",
            metadata={
                "request_id": "perm-card",
                "tool_name": "shell",
                "tool_input": {"command": "pwd"},
                "options": [
                    {"optionId": "allow-once", "kind": "allow_once"},
                    {"optionId": "reject-once", "kind": "reject_once"},
                ],
            },
        )
        yield OutputChunk(
            text="", chunk_type="turn_complete",
            metadata={"cause": "success"})


@pytest.mark.asyncio
async def test_permission_websocket_card_keeps_provider_options():
    ws = _WS()
    pm = ProcessManager(ws, _Activity())
    await pm.start("p1", "cursor", _PermissionStream(), session_id="s1")
    await pm._tasks[pm._key("p1", "s1", "cursor")]

    card = next(event for event in ws.events if event["type"] == "approval.request")
    assert card["tool_args"]["command"] == "pwd"
    assert card["tool_args"]["permission_options"][1]["optionId"] == "reject-once"
