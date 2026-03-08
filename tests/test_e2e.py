# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase 3 e2e tests: full daemon running, REST + WebSocket client, real LLM.

Simulates what a frontend would do: create project, start agent, receive WS
events, read chat history, inject messages, stop agent.

Uses httpx AsyncClient with ASGITransport for REST calls, and httpx_ws with
ASGIWebSocketTransport for WebSocket tests — both fully async.

Skip all tests if AGENT_OS_TEST_API_KEY is not set.
"""

import asyncio
import os

import pytest
import pytest_asyncio
import httpx
from httpx import ASGITransport

# ---------------------------------------------------------------------------
# Config: read from env, skip entire module if not set
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("AGENT_OS_TEST_API_KEY", "")
BASE_URL = os.environ.get("AGENT_OS_TEST_BASE_URL", "https://api.moonshot.cn/v1")
MODEL = os.environ.get("AGENT_OS_TEST_MODEL", "kimi-k2.5")

skip_no_key = pytest.mark.skipif(
    not API_KEY,
    reason="AGENT_OS_TEST_API_KEY not set — skipping e2e tests",
)

pytestmark = [skip_no_key, pytest.mark.timeout(180)]

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from agent_os.api.app import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace directory with a test file."""
    ws = str(tmp_path / "workspace")
    os.makedirs(ws, exist_ok=True)
    test_file = os.path.join(ws, "hello.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Hello from e2e test!")
    return ws


@pytest.fixture
def app(tmp_path):
    """Create a fresh FastAPI app with temp data directory."""
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    return create_app(data_dir=data_dir)


@pytest_asyncio.fixture
async def client(app):
    """Async HTTP client using ASGI transport."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _create_project(client: httpx.AsyncClient, workspace: str) -> str:
    """Helper: create project via REST and return project_id."""
    resp = await client.post("/api/v2/projects", json={
        "name": "E2E Test Project",
        "workspace": workspace,
        "model": MODEL,
        "api_key": API_KEY,
        "base_url": BASE_URL,
    })
    assert resp.status_code == 201, f"Create project failed: {resp.status_code} {resp.text}"
    data = resp.json()
    assert "project_id" in data
    return data["project_id"]


async def _wait_for_agent_idle(client: httpx.AsyncClient, pid: str,
                               max_wait: float = 60.0) -> list[dict]:
    """Poll chat history until an assistant text response appears or timeout."""
    import time
    start = time.time()
    while time.time() - start < max_wait:
        resp = await client.get(f"/api/v2/agents/{pid}/chat")
        if resp.status_code == 200:
            msgs = resp.json()
            assistant_text = [
                m for m in msgs
                if m["role"] == "assistant" and m.get("content")
            ]
            if assistant_text:
                return msgs
        await asyncio.sleep(2)
    # Final attempt
    resp = await client.get(f"/api/v2/agents/{pid}/chat")
    return resp.json() if resp.status_code == 200 else []


# ---------------------------------------------------------------------------
# Test 1: Create project
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_create_project(client, workspace):
    """POST /api/v2/projects -> 201, project_id returned."""
    resp = await client.post("/api/v2/projects", json={
        "name": "E2E Test",
        "workspace": workspace,
        "model": MODEL,
        "api_key": API_KEY,
        "base_url": BASE_URL,
    })
    assert resp.status_code == 201
    data = resp.json()
    assert "project_id" in data
    assert data["project_id"].startswith("proj_")
    assert data["name"] == "E2E Test"
    assert data["workspace"] == workspace


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_list_projects_after_create(client, workspace):
    """GET /api/v2/projects -> list includes created project."""
    pid = await _create_project(client, workspace)
    resp = await client.get("/api/v2/projects")
    assert resp.status_code == 200
    projects = resp.json()
    assert any(p["project_id"] == pid for p in projects)


# ---------------------------------------------------------------------------
# Test 2: Start agent and verify chat history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_start_agent_and_chat(client, workspace):
    """POST /api/v2/agents/start -> started; chat has user + assistant messages."""
    pid = await _create_project(client, workspace)

    resp = await client.post("/api/v2/agents/start", json={
        "project_id": pid,
        "initial_message": "What is 2+2? Reply with just the number, nothing else.",
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"

    messages = await _wait_for_agent_idle(client, pid, max_wait=60)

    assert len(messages) >= 2, f"Expected >= 2 messages, got {len(messages)}"
    assert messages[0]["role"] == "user"
    assert "2+2" in messages[0]["content"]

    assistant_msgs = [m for m in messages if m["role"] == "assistant" and m.get("content")]
    assert len(assistant_msgs) >= 1, "No assistant text response found"


# ---------------------------------------------------------------------------
# Test 3: WebSocket events during agent run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_ws_events_during_agent_run(tmp_path, workspace):
    """Connect to /ws, subscribe, start agent, receive agent.status and
    chat.stream_delta events."""
    from httpx_ws import aconnect_ws
    from httpx_ws.transport import ASGIWebSocketTransport

    data_dir = str(tmp_path / "data_ws")
    os.makedirs(data_dir, exist_ok=True)
    app = create_app(data_dir=data_dir)

    async with ASGIWebSocketTransport(app=app) as transport:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            pid = await _create_project(client, workspace)

            async with aconnect_ws("http://test/ws", client) as ws:
                # Subscribe
                await ws.send_json({"type": "subscribe", "project_ids": [pid]})
                sub_resp = await asyncio.wait_for(ws.receive_json(), timeout=5)
                assert sub_resp["type"] == "subscribed"
                assert pid in sub_resp["project_ids"]

                # Start agent
                resp = await client.post("/api/v2/agents/start", json={
                    "project_id": pid,
                    "initial_message": "What is 1+1? Reply with just the number.",
                })
                assert resp.status_code == 200

                # Collect WS events
                events = []
                import time
                start = time.time()
                while time.time() - start < 60:
                    try:
                        data = await asyncio.wait_for(ws.receive_json(), timeout=5)
                        events.append(data)
                        if (data.get("type") == "agent.status"
                                and data.get("status") in ("idle", "stopped", "error")):
                            break
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        break

    # Assertions
    assert len(events) >= 2, f"Expected >= 2 WS events, got {len(events)}"

    # agent.status "running"
    status_events = [e for e in events if e.get("type") == "agent.status"]
    running_events = [e for e in status_events if e.get("status") == "running"]
    assert len(running_events) >= 1, (
        f"Expected agent.status running, got: {status_events}"
    )

    # Terminal status (idle or stopped)
    idle_or_stopped = [
        e for e in status_events
        if e.get("status") in ("idle", "stopped")
    ]
    assert len(idle_or_stopped) >= 1, (
        f"Expected terminal status, got: {status_events}"
    )

    # chat.stream_delta events
    stream_events = [e for e in events if e.get("type") == "chat.stream_delta"]
    assert len(stream_events) >= 1, "Expected chat.stream_delta events"
    for se in stream_events:
        assert "project_id" in se
        assert "text" in se
        assert "source" in se
        assert "is_final" in se

    # All events use snake_case keys
    for evt in events:
        for key in evt.keys():
            assert key == key.lower(), (
                f"Key '{key}' has uppercase in event type={evt.get('type')}"
            )


# ---------------------------------------------------------------------------
# Test 4: Agent activity events (tool usage)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_ws_agent_activity_events(tmp_path, workspace):
    """When the agent uses a tool, WS should receive agent.activity events
    with proper category, tool_name, source fields."""
    from httpx_ws import aconnect_ws
    from httpx_ws.transport import ASGIWebSocketTransport

    data_dir = str(tmp_path / "data_activity")
    os.makedirs(data_dir, exist_ok=True)
    app = create_app(data_dir=data_dir)

    async with ASGIWebSocketTransport(app=app) as transport:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            pid = await _create_project(client, workspace)

            async with aconnect_ws("http://test/ws", client) as ws:
                await ws.send_json({"type": "subscribe", "project_ids": [pid]})
                await asyncio.wait_for(ws.receive_json(), timeout=5)

                # Ask agent to read a file (should trigger tool usage)
                resp = await client.post("/api/v2/agents/start", json={
                    "project_id": pid,
                    "initial_message": (
                        "Read the file hello.txt and tell me what it says. "
                        "Use the read tool."
                    ),
                })
                assert resp.status_code == 200

                events = []
                import time
                start = time.time()
                while time.time() - start < 60:
                    try:
                        data = await asyncio.wait_for(ws.receive_json(), timeout=5)
                        events.append(data)
                        if (data.get("type") == "agent.status"
                                and data.get("status") in ("idle", "stopped", "error")):
                            break
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        break

    # Should have agent.activity events from tool usage
    activities = [e for e in events if e.get("type") == "agent.activity"]
    assert len(activities) >= 1, (
        f"Expected agent.activity events, got types: "
        f"{[e.get('type') for e in events]}"
    )

    # Check activity event structure
    for act in activities:
        assert "project_id" in act
        assert "id" in act
        assert "category" in act
        assert "source" in act
        assert "timestamp" in act
        assert act["project_id"] == pid

    # Should have a file_read category event
    file_reads = [a for a in activities if a.get("category") == "file_read"]
    assert len(file_reads) >= 1, (
        f"Expected file_read activity, got: "
        f"{[a.get('category') for a in activities]}"
    )
    assert file_reads[0]["source"] == "management"
    assert file_reads[0]["tool_name"] == "read"


# ---------------------------------------------------------------------------
# Test 5: Chat history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_chat_history(client, workspace):
    """GET /api/v2/agents/{pid}/chat -> messages from session."""
    pid = await _create_project(client, workspace)

    resp = await client.post("/api/v2/agents/start", json={
        "project_id": pid,
        "initial_message": "Say 'hello world'. Reply with just those two words.",
    })
    assert resp.status_code == 200

    messages = await _wait_for_agent_idle(client, pid, max_wait=60)

    assert len(messages) >= 2
    for msg in messages:
        assert "role" in msg
        assert "timestamp" in msg

    user_msgs = [m for m in messages if m["role"] == "user"]
    assert len(user_msgs) >= 1

    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    assert len(assistant_msgs) >= 1


# ---------------------------------------------------------------------------
# Test 6: Inject message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_inject_message(client, workspace):
    """POST /api/v2/agents/{pid}/inject -> queued or delivered."""
    pid = await _create_project(client, workspace)

    # Start agent
    resp = await client.post("/api/v2/agents/start", json={
        "project_id": pid,
        "initial_message": "What is 3+3? Reply with just the number.",
    })
    assert resp.status_code == 200

    # Wait for agent to finish
    msgs_before = await _wait_for_agent_idle(client, pid, max_wait=60)
    user_count_before = len([m for m in msgs_before if m["role"] == "user"])

    # Inject a follow-up message
    resp = await client.post(f"/api/v2/agents/{pid}/inject", json={
        "content": "What is 5+5? Reply with just the number.",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("queued", "delivered"), (
        f"Unexpected inject status: {data}"
    )

    # Wait for the second response
    await asyncio.sleep(15)

    resp = await client.get(f"/api/v2/agents/{pid}/chat")
    assert resp.status_code == 200
    messages = resp.json()

    user_msgs = [m for m in messages if m["role"] == "user"]
    assert len(user_msgs) >= user_count_before + 1, (
        f"Expected injected message in history. "
        f"Before: {user_count_before}, after: {len(user_msgs)}"
    )


# ---------------------------------------------------------------------------
# Test 7: Stop agent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_stop_agent(tmp_path, workspace):
    """POST /api/v2/agents/{pid}/stop -> agent.status stopped via WS."""
    from httpx_ws import aconnect_ws
    from httpx_ws.transport import ASGIWebSocketTransport

    data_dir = str(tmp_path / "data_stop")
    os.makedirs(data_dir, exist_ok=True)
    app = create_app(data_dir=data_dir)

    async with ASGIWebSocketTransport(app=app) as transport:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            pid = await _create_project(client, workspace)

            async with aconnect_ws("http://test/ws", client) as ws:
                await ws.send_json({"type": "subscribe", "project_ids": [pid]})
                await asyncio.wait_for(ws.receive_json(), timeout=5)

                # Start agent
                resp = await client.post("/api/v2/agents/start", json={
                    "project_id": pid,
                    "initial_message": "Count from 1 to 100, one number per line.",
                })
                assert resp.status_code == 200

                # Wait briefly for agent to start
                import time
                events = []
                start = time.time()
                while time.time() - start < 5:
                    try:
                        data = await asyncio.wait_for(ws.receive_json(), timeout=2)
                        events.append(data)
                        if (data.get("type") == "agent.status"
                                and data.get("status") in ("idle", "error")):
                            break
                    except asyncio.TimeoutError:
                        break
                    except Exception:
                        break

                # Stop agent
                resp = await client.post(f"/api/v2/agents/{pid}/stop")
                assert resp.status_code == 200
                assert resp.json()["status"] == "stopping"

                # Collect stopped event
                stop_start = time.time()
                while time.time() - stop_start < 15:
                    try:
                        data = await asyncio.wait_for(ws.receive_json(), timeout=3)
                        events.append(data)
                        if (data.get("type") == "agent.status"
                                and data.get("status") == "stopped"):
                            break
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        break

    status_events = [e for e in events if e.get("type") == "agent.status"]
    stopped_events = [e for e in status_events if e.get("status") == "stopped"]
    assert len(stopped_events) >= 1, (
        f"Expected agent.status stopped, got: "
        f"{[(e.get('status'), e.get('reason', '')) for e in status_events]}"
    )


# ---------------------------------------------------------------------------
# Test 8: Full workflow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_full_workflow(tmp_path, workspace):
    """Full e2e: create -> start -> WS events -> chat -> inject -> stop."""
    from httpx_ws import aconnect_ws
    from httpx_ws.transport import ASGIWebSocketTransport

    data_dir = str(tmp_path / "data_full")
    os.makedirs(data_dir, exist_ok=True)
    app = create_app(data_dir=data_dir)

    async with ASGIWebSocketTransport(app=app) as transport:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Create project
            pid = await _create_project(client, workspace)

            async with aconnect_ws("http://test/ws", client) as ws:
                # 2. Subscribe
                await ws.send_json({"type": "subscribe", "project_ids": [pid]})
                sub = await asyncio.wait_for(ws.receive_json(), timeout=5)
                assert sub["type"] == "subscribed"

                # 3. Start agent (ask to read file — triggers tool usage)
                resp = await client.post("/api/v2/agents/start", json={
                    "project_id": pid,
                    "initial_message": (
                        "Read hello.txt and tell me what it says. "
                        "Use the read tool."
                    ),
                })
                assert resp.status_code == 200

                # 4. Collect WS events until idle
                events = []
                import time
                start = time.time()
                while time.time() - start < 60:
                    try:
                        data = await asyncio.wait_for(ws.receive_json(), timeout=5)
                        events.append(data)
                        if (data.get("type") == "agent.status"
                                and data.get("status") in ("idle", "stopped", "error")):
                            break
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        break

                # 5. Verify WS events received
                types_seen = {e.get("type") for e in events}
                assert "agent.status" in types_seen
                assert "chat.stream_delta" in types_seen

                # 6. Check chat history via REST
                resp = await client.get(f"/api/v2/agents/{pid}/chat")
                assert resp.status_code == 200
                messages = resp.json()
                assert len(messages) >= 2

                user_msgs = [m for m in messages if m["role"] == "user"]
                assert len(user_msgs) >= 1
                assert "hello.txt" in user_msgs[0]["content"]

                # Check tool was used
                tool_msgs = [m for m in messages if m["role"] == "tool"]
                assert len(tool_msgs) >= 1, "Expected tool result in chat"

                # Verify file content in tool result
                tool_contents = [m.get("content", "") for m in tool_msgs]
                assert any("Hello from e2e test" in c for c in tool_contents), (
                    f"Expected file content in tool results: {tool_contents}"
                )

                # 7. Stop agent (it may already be idle)
                resp = await client.post(f"/api/v2/agents/{pid}/stop")
                if resp.status_code == 200:
                    stop_start = time.time()
                    while time.time() - stop_start < 15:
                        try:
                            data = await asyncio.wait_for(ws.receive_json(), timeout=3)
                            events.append(data)
                            if (data.get("type") == "agent.status"
                                    and data.get("status") == "stopped"):
                                break
                        except asyncio.TimeoutError:
                            continue
                        except Exception:
                            break

    # Final: all events snake_case
    for evt in events:
        for key in evt.keys():
            assert key == key.lower(), (
                f"Key '{key}' has uppercase in event type={evt.get('type')}"
            )


# ---------------------------------------------------------------------------
# Test 9: Error cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_start_nonexistent_project(client):
    """POST /api/v2/agents/start with bad project_id -> 404."""
    resp = await client.post("/api/v2/agents/start", json={
        "project_id": "nonexistent_project",
    })
    assert resp.status_code == 404
    assert "detail" in resp.json()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_inject_no_session(client):
    """POST /api/v2/agents/{pid}/inject with no session -> 404."""
    resp = await client.post("/api/v2/agents/nonexistent/inject", json={
        "content": "hello",
    })
    assert resp.status_code == 404


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_chat_no_session(client):
    """GET /api/v2/agents/{pid}/chat with no session -> 404."""
    resp = await client.get("/api/v2/agents/nonexistent/chat")
    assert resp.status_code == 404


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_create_project_invalid_workspace(client):
    """POST /api/v2/projects with bad workspace -> 400."""
    resp = await client.post("/api/v2/projects", json={
        "name": "Bad Workspace",
        "workspace": "/nonexistent/path/xyz_e2e",
        "model": MODEL,
        "api_key": API_KEY,
    })
    assert resp.status_code == 400
    assert "detail" in resp.json()


# ---------------------------------------------------------------------------
# Test 10: WS subscribe protocol
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_ws_subscribe_protocol(tmp_path):
    """WS subscribe -> subscribed response with correct project_ids."""
    from httpx_ws import aconnect_ws
    from httpx_ws.transport import ASGIWebSocketTransport

    data_dir = str(tmp_path / "data_wssub")
    os.makedirs(data_dir, exist_ok=True)
    app = create_app(data_dir=data_dir)

    async with ASGIWebSocketTransport(app=app) as transport:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            async with aconnect_ws("http://test/ws", client) as ws:
                await ws.send_json({
                    "type": "subscribe",
                    "project_ids": ["proj_a", "proj_b"],
                })
                data = await asyncio.wait_for(ws.receive_json(), timeout=5)
                assert data["type"] == "subscribed"
                assert set(data["project_ids"]) == {"proj_a", "proj_b"}

                # Resubscribe
                await ws.send_json({
                    "type": "subscribe",
                    "project_ids": ["proj_c"],
                })
                data = await asyncio.wait_for(ws.receive_json(), timeout=5)
                assert data["type"] == "subscribed"
                assert data["project_ids"] == ["proj_c"]


# ---------------------------------------------------------------------------
# Test 11: No v1 routes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_no_v1_routes(client):
    """Zero /api/v1/ route registrations."""
    resp = await client.get("/api/v1/projects")
    assert resp.status_code in (404, 405)


# ---------------------------------------------------------------------------
# Test 12: No SQLite references
# ---------------------------------------------------------------------------


def test_no_sqlite_references():
    """App factory and daemon have zero SQLite references."""
    import inspect
    from agent_os.api import app as app_mod

    source = inspect.getsource(app_mod)
    assert "sqlite" not in source.lower(), "SQLite reference found in app.py"
    assert "from agent_os.database" not in source, "database import in app.py"
