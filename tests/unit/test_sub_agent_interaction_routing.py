import pytest

from agent_os.agent.tools.agent_message import AgentMessageTool
from agent_os.agent.transports.base import TransportEvent, transport_event_to_chunk
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
from agent_os.daemon_v2.process_manager import ProcessManager


class _ToolManager:
    def __init__(self):
        self.call = None

    async def respond_to_interaction(self, *args, **kwargs):
        self.call = (args, kwargs)
        return True


@pytest.mark.asyncio
async def test_sub_agent_interaction_tool_responds_on_same_request_and_yields():
    manager = _ToolManager()
    tool = AgentMessageTool(
        manager, project_id="p1", session_id="s1", depth=0)

    result = await tool.execute(
        "respond", "cursor", "继续执行", interaction_id="i-7",
        selection=["accept"],
    )

    assert manager.call == (
        ("p1", "cursor", "i-7"),
        {"session_id": "s1", "text": "继续执行", "selection": ["accept"]},
    )
    assert result.meta["yield_turn"] is True


def test_interaction_transport_event_stays_control_plane_chunk():
    chunk = transport_event_to_chunk(TransportEvent(
        "interaction_required",
        {"interaction_id": "i-1", "prompt": "继续吗？"},
        "继续吗？",
    ))

    assert chunk.chunk_type == "interaction_required"
    assert chunk.metadata["interaction_id"] == "i-1"


def test_acp_session_created_stays_control_plane_chunk():
    chunk = transport_event_to_chunk(TransportEvent(
        "session_created", {"session_id": "cursor-session-1"}, ""))

    assert chunk.chunk_type == "thread_started"
    assert chunk.metadata["session_id"] == "cursor-session-1"


class _AgentManager:
    def __init__(self):
        self.injections = []

    async def inject_system_message(self, project_id, content, **kwargs):
        self.injections.append((project_id, content, kwargs))


class _WS:
    def __init__(self):
        self.events = []

    def broadcast(self, project_id, payload):
        self.events.append(payload)


@pytest.mark.asyncio
async def test_interaction_lifecycle_injection_contains_opaque_response_id():
    agent_manager = _AgentManager()
    ws = _WS()
    observer = LifecycleObserver(agent_manager, ws)

    await observer.on_interaction_required(
        "p1", "cursor", interaction_id="ask-中文", kind="question",
        prompt="是否继续？", options=[{"id": "yes", "label": "继续"}],
        session_id="s1",
    )

    _, content, kwargs = agent_manager.injections[0]
    assert "agent_message(action=\"respond\"" in content
    assert "ask-中文" in content
    assert "selection=\"option-id\"" in content
    assert "message only when" in content
    assert kwargs["session_id"] == "s1"
    assert kwargs["meta"]["event"] == "interaction_required"
    assert ws.events[0]["type"] == "sub_agent.interaction_required"


@pytest.mark.asyncio
async def test_plan_interaction_explains_accept_reject_without_freeform_default():
    agent_manager = _AgentManager()
    observer = LifecycleObserver(agent_manager, _WS())

    await observer.on_interaction_required(
        "p1", "cursor", interaction_id="plan-1", kind="cursor/create_plan",
        prompt="Review plan", plan={"steps": ["修改", "测试"]},
        session_id="s1",
    )

    content = agent_manager.injections[0][1]
    assert 'selection="accept"' in content
    assert 'selection="reject"' in content
    assert "optional rejection reason" in content
