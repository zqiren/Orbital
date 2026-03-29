# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""User-story e2e tests with full tracing.

Each test simulates a real user workflow against the full daemon stack
with a real LLM (kimi-k2.5). Every API call, WS event, message, and
timing is captured into a trace object. After each test, the trace is
dumped to a shared report list for post-run documentation.

User stories are derived from ACTIVE-project-background.md:
  - US-1: Onboard a project and get an answer
  - US-2: Agent reads workspace files and reports
  - US-3: Agent creates a file in the workspace
  - US-4: Multi-turn conversation via inject
  - US-5: Real-time monitoring via WebSocket events
  - US-6: Stop agent mid-task
  - US-7: Session persistence across hot resume
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime

import pytest
import pytest_asyncio
import httpx
from httpx import ASGITransport

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("AGENT_OS_TEST_API_KEY", "")
BASE_URL = os.environ.get("AGENT_OS_TEST_BASE_URL", "https://api.moonshot.cn/v1")
MODEL = os.environ.get("AGENT_OS_TEST_MODEL", "kimi-k2.5")

skip_no_key = pytest.mark.skipif(
    not API_KEY,
    reason="AGENT_OS_TEST_API_KEY not set — skipping user story tests",
)

pytestmark = [skip_no_key, pytest.mark.timeout(180)]

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from agent_os.api.app import create_app

# ---------------------------------------------------------------------------
# Trace infrastructure
# ---------------------------------------------------------------------------

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
REPORT_PATH = os.path.join(REPORT_DIR, "user-story-test-report.md")
BUGS_PATH = os.path.join(REPORT_DIR, "user-story-bugs.md")

# Accumulated traces across all tests in this module
_all_traces: list[dict] = []
_all_bugs: list[dict] = []


@dataclass
class Trace:
    """Captures every step of a user story test."""
    story_id: str
    story_title: str
    steps: list[dict] = field(default_factory=list)
    ws_events: list[dict] = field(default_factory=list)
    chat_messages: list[dict] = field(default_factory=list)
    bugs: list[dict] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    passed: bool = False

    def step(self, action: str, detail: str = "", **kwargs):
        entry = {
            "t": round(time.time() - self.started_at, 2),
            "action": action,
            "detail": detail,
        }
        entry.update(kwargs)
        self.steps.append(entry)

    def bug(self, title: str, detail: str, severity: str = "minor"):
        b = {"title": title, "detail": detail, "severity": severity,
             "story": self.story_id}
        self.bugs.append(b)
        _all_bugs.append(b)

    def to_dict(self) -> dict:
        return {
            "story_id": self.story_id,
            "story_title": self.story_title,
            "duration_s": round(self.finished_at - self.started_at, 2),
            "passed": self.passed,
            "step_count": len(self.steps),
            "ws_event_count": len(self.ws_events),
            "chat_message_count": len(self.chat_messages),
            "bug_count": len(self.bugs),
            "steps": self.steps,
            "ws_events": self.ws_events,
            "chat_messages": self.chat_messages,
            "bugs": self.bugs,
        }


def _start_trace(story_id: str, title: str) -> Trace:
    t = Trace(story_id=story_id, story_title=title, started_at=time.time())
    return t


def _finish_trace(trace: Trace, passed: bool):
    trace.finished_at = time.time()
    trace.passed = passed
    _all_traces.append(trace.to_dict())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_project(client, workspace, trace: Trace) -> str:
    trace.step("REST", "POST /api/v2/projects")
    resp = await client.post("/api/v2/projects", json={
        "name": "User Story Test",
        "workspace": workspace,
        "model": MODEL,
        "api_key": API_KEY,
        "base_url": BASE_URL,
    })
    trace.step("REST_RESPONSE", f"status={resp.status_code}", body=resp.json())
    assert resp.status_code == 201
    pid = resp.json()["project_id"]
    trace.step("PROJECT_CREATED", pid)
    return pid


async def _start_agent(client, pid, message, trace: Trace):
    trace.step("REST", f"POST /api/v2/agents/start  message={message[:60]}...")
    resp = await client.post("/api/v2/agents/start", json={
        "project_id": pid,
        "initial_message": message,
    })
    trace.step("REST_RESPONSE", f"status={resp.status_code}", body=resp.json())
    assert resp.status_code == 200


async def _wait_for_idle(client, pid, trace: Trace, max_wait=60) -> list[dict]:
    trace.step("POLL_START", f"Polling chat history for up to {max_wait}s")
    start = time.time()
    msgs = []
    while time.time() - start < max_wait:
        resp = await client.get(f"/api/v2/agents/{pid}/chat")
        if resp.status_code == 200:
            msgs = resp.json()
            assistant_text = [
                m for m in msgs
                if m["role"] == "assistant" and m.get("content")
            ]
            if assistant_text:
                trace.step("POLL_DONE", f"Got {len(msgs)} messages after {round(time.time()-start,1)}s")
                trace.chat_messages = msgs
                return msgs
        await asyncio.sleep(2)
    trace.step("POLL_TIMEOUT", f"Timed out after {max_wait}s, got {len(msgs)} messages")
    resp = await client.get(f"/api/v2/agents/{pid}/chat")
    if resp.status_code == 200:
        msgs = resp.json()
    trace.chat_messages = msgs
    return msgs


async def _collect_ws_events(ws, trace: Trace, timeout=60) -> list[dict]:
    """Collect WS events until terminal status or timeout."""
    events = []
    start = time.time()
    while time.time() - start < timeout:
        try:
            data = await asyncio.wait_for(ws.receive_json(), timeout=5)
            events.append(data)
            trace.ws_events.append({
                "t": round(time.time() - trace.started_at, 2),
                **data,
            })
            if (data.get("type") == "agent.status"
                    and data.get("status") in ("idle", "stopped", "error")):
                break
        except asyncio.TimeoutError:
            continue
        except Exception:
            break
    return events


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path):
    ws = str(tmp_path / "workspace")
    os.makedirs(ws, exist_ok=True)
    return ws


@pytest.fixture
def rich_workspace(tmp_path):
    """Workspace with several pre-existing files."""
    ws = str(tmp_path / "workspace")
    os.makedirs(ws, exist_ok=True)

    with open(os.path.join(ws, "README.md"), "w", encoding="utf-8") as f:
        f.write("# My Project\n\nA sample project for testing.\n")

    with open(os.path.join(ws, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"version": "1.0", "name": "test-app", "debug": True}, f, indent=2)

    src_dir = os.path.join(ws, "src")
    os.makedirs(src_dir, exist_ok=True)
    with open(os.path.join(src_dir, "main.py"), "w", encoding="utf-8") as f:
        f.write('def greet(name: str) -> str:\n    return f"Hello, {name}!"\n\n'
                'if __name__ == "__main__":\n    print(greet("World"))\n')

    return ws


@pytest.fixture
def app(tmp_path):
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    return create_app(data_dir=data_dir)


@pytest_asyncio.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# US-1: Developer onboards a new project and gets an answer
#
# "As a developer, I create a project pointing at my workspace, then
#  ask the agent a question. I expect a correct answer back."
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_us1_onboard_project_and_ask_question(client, workspace):
    """US-1: Create project -> start agent with question -> get correct answer."""
    trace = _start_trace("US-1", "Onboard project and ask question")
    passed = False
    try:
        # Step 1: Create project
        pid = await _create_project(client, workspace, trace)

        # Step 2: Start agent with a math question
        await _start_agent(client, pid,
                           "What is 17 * 3? Reply with just the number.", trace)

        # Step 3: Wait for response
        msgs = await _wait_for_idle(client, pid, trace)

        # Step 4: Verify
        trace.step("VERIFY", "Checking assistant response contains '51'")
        assistant_msgs = [m for m in msgs if m["role"] == "assistant" and m.get("content")]
        assert len(assistant_msgs) >= 1, f"No assistant response. Messages: {[m['role'] for m in msgs]}"

        last = assistant_msgs[-1]["content"]
        trace.step("ASSISTANT_RESPONSE", last[:200])
        assert "51" in last, f"Expected '51' in response: {last[:200]}"

        # Step 5: Verify message structure
        trace.step("VERIFY", "Checking message structure (role, timestamp)")
        for m in msgs:
            assert "role" in m
            assert "timestamp" in m

        passed = True
        trace.step("RESULT", "PASSED")

    except Exception as e:
        trace.step("RESULT", f"FAILED: {e}")
        raise
    finally:
        _finish_trace(trace, passed)


# ---------------------------------------------------------------------------
# US-2: Agent reads workspace files and reports content
#
# "As a developer, I have source files in my workspace. I ask the agent
#  to read a specific file and tell me what's in it."
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_us2_agent_reads_files(client, rich_workspace):
    """US-2: Agent reads a file and reports its content."""
    trace = _start_trace("US-2", "Agent reads workspace files")
    passed = False
    try:
        pid = await _create_project(client, rich_workspace, trace)

        await _start_agent(client, pid,
            "Read the file src/main.py and tell me what the greet function does. "
            "Use the read tool.", trace)

        msgs = await _wait_for_idle(client, pid, trace)

        # Verify tool was used
        trace.step("VERIFY", "Checking that read tool was called")
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        if not tool_msgs:
            trace.bug("NO_TOOL_CALL",
                       "Agent did not use the read tool despite being asked",
                       severity="major")
        else:
            trace.step("TOOL_RESULT", f"{len(tool_msgs)} tool result(s) found")
            # Check tool result contains the file content
            tool_content = " ".join(m.get("content", "") for m in tool_msgs)
            assert "greet" in tool_content or "Hello" in tool_content, (
                f"Tool result missing expected content: {tool_content[:200]}"
            )

        # Verify assistant describes the function
        trace.step("VERIFY", "Checking assistant describes the greet function")
        assistant_msgs = [m for m in msgs if m["role"] == "assistant" and m.get("content")]
        assert len(assistant_msgs) >= 1
        final_text = assistant_msgs[-1]["content"]
        trace.step("ASSISTANT_RESPONSE", final_text[:300])

        # The response should reference greeting/hello/name
        text_lower = final_text.lower()
        assert any(kw in text_lower for kw in ["greet", "hello", "name", "return"]), (
            f"Response doesn't describe function: {final_text[:200]}"
        )

        passed = True
        trace.step("RESULT", "PASSED")

    except Exception as e:
        trace.step("RESULT", f"FAILED: {e}")
        raise
    finally:
        _finish_trace(trace, passed)


# ---------------------------------------------------------------------------
# US-3: Agent creates a file in the workspace
#
# "As a developer, I ask the agent to create a new file with specific
#  content. After the agent finishes, the file should exist on disk."
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_us3_agent_creates_file(client, workspace):
    """US-3: Agent writes a file to the workspace."""
    trace = _start_trace("US-3", "Agent creates a file in workspace")
    passed = False
    try:
        pid = await _create_project(client, workspace, trace)

        await _start_agent(client, pid,
            "Create a file called notes.txt in the workspace root with the content: "
            "'Meeting notes: discuss API design, deadline Friday'. "
            "Use the write tool.", trace)

        msgs = await _wait_for_idle(client, pid, trace)

        # Verify write tool was called
        trace.step("VERIFY", "Checking that write tool was called")
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        trace.step("TOOL_RESULTS", f"{len(tool_msgs)} tool result(s)")

        # Verify file exists on disk
        trace.step("VERIFY", "Checking file exists on disk")
        notes_path = os.path.join(workspace, "notes.txt")
        file_exists = os.path.exists(notes_path)
        trace.step("FILE_CHECK", f"notes.txt exists={file_exists}")

        if not file_exists:
            trace.bug("FILE_NOT_CREATED",
                       "Agent was asked to create notes.txt but file does not exist on disk",
                       severity="major")
            # Still check if a different filename was used
            files = os.listdir(workspace)
            trace.step("WORKSPACE_FILES", str(files))
        else:
            with open(notes_path, "r", encoding="utf-8") as f:
                content = f.read()
            trace.step("FILE_CONTENT", content[:200])
            assert "Meeting notes" in content or "API design" in content or "deadline" in content, (
                f"File content doesn't match request: {content[:200]}"
            )

        assert file_exists, "Expected notes.txt to be created"
        passed = True
        trace.step("RESULT", "PASSED")

    except Exception as e:
        trace.step("RESULT", f"FAILED: {e}")
        raise
    finally:
        _finish_trace(trace, passed)


# ---------------------------------------------------------------------------
# US-4: Multi-turn conversation via inject
#
# "As a developer, I ask the agent a question, get an answer, then
#  send a follow-up. Both Q&A pairs should be in chat history."
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_us4_multi_turn_conversation(client, workspace):
    """US-4: Two-turn conversation via start + inject."""
    trace = _start_trace("US-4", "Multi-turn conversation via inject")
    passed = False
    try:
        pid = await _create_project(client, workspace, trace)

        # Turn 1
        trace.step("TURN_1", "Asking about Python lists")
        await _start_agent(client, pid,
            "What Python method adds an item to the end of a list? "
            "Reply with just the method name.", trace)

        msgs_t1 = await _wait_for_idle(client, pid, trace, max_wait=60)
        assistant_t1 = [m for m in msgs_t1 if m["role"] == "assistant" and m.get("content")]
        assert len(assistant_t1) >= 1, "No response to turn 1"
        t1_response = assistant_t1[-1]["content"]
        trace.step("TURN_1_RESPONSE", t1_response[:200])
        assert "append" in t1_response.lower(), f"Expected 'append' in: {t1_response[:200]}"

        # Turn 2: inject follow-up
        trace.step("TURN_2", "Injecting follow-up question")
        resp = await client.post(f"/api/v2/agents/{pid}/inject", json={
            "content": "And what method removes the last item? Reply with just the method name.",
        })
        trace.step("INJECT_RESPONSE", f"status={resp.status_code}", body=resp.json())
        assert resp.status_code == 200

        # Wait for second response
        await asyncio.sleep(15)
        resp = await client.get(f"/api/v2/agents/{pid}/chat")
        assert resp.status_code == 200
        msgs_t2 = resp.json()
        trace.chat_messages = msgs_t2

        # Verify both turns present
        trace.step("VERIFY", "Checking both turns in chat history")
        user_msgs = [m for m in msgs_t2 if m["role"] == "user"]
        trace.step("USER_MESSAGES", f"{len(user_msgs)} user messages found")
        assert len(user_msgs) >= 2, f"Expected >= 2 user messages, got {len(user_msgs)}"

        assistant_all = [m for m in msgs_t2 if m["role"] == "assistant" and m.get("content")]
        trace.step("ASSISTANT_MESSAGES", f"{len(assistant_all)} assistant messages found")

        # Check second response mentions "pop"
        if len(assistant_all) >= 2:
            t2_response = assistant_all[-1]["content"]
            trace.step("TURN_2_RESPONSE", t2_response[:200])
            assert "pop" in t2_response.lower(), f"Expected 'pop' in: {t2_response[:200]}"
        else:
            trace.bug("MISSING_TURN2_RESPONSE",
                       f"Only {len(assistant_all)} assistant messages after 2 turns",
                       severity="minor")

        passed = True
        trace.step("RESULT", "PASSED")

    except Exception as e:
        trace.step("RESULT", f"FAILED: {e}")
        raise
    finally:
        _finish_trace(trace, passed)


# ---------------------------------------------------------------------------
# US-5: Real-time monitoring via WebSocket
#
# "As a developer, I connect to WebSocket, subscribe to my project,
#  start the agent, and watch streaming tokens + activity events."
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_us5_realtime_ws_monitoring(tmp_path, rich_workspace):
    """US-5: WS events during agent run — status, stream, activity."""
    from httpx_ws import aconnect_ws
    from httpx_ws.transport import ASGIWebSocketTransport

    trace = _start_trace("US-5", "Real-time WebSocket monitoring")
    passed = False
    try:
        data_dir = str(tmp_path / "data_us5")
        os.makedirs(data_dir, exist_ok=True)
        app = create_app(data_dir=data_dir)

        async with ASGIWebSocketTransport(app=app) as transport:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                pid = await _create_project(client, rich_workspace, trace)

                async with aconnect_ws("http://test/ws", client) as ws:
                    # Subscribe
                    trace.step("WS", "subscribe")
                    await ws.send_json({"type": "subscribe", "project_ids": [pid]})
                    sub = await asyncio.wait_for(ws.receive_json(), timeout=5)
                    trace.step("WS_SUBSCRIBED", str(sub))

                    # Start agent — ask to read a file (triggers tool activity)
                    await _start_agent(client, pid,
                        "Read the file config.json and tell me the app name and version. "
                        "Use the read tool.", trace)

                    # Collect events
                    trace.step("WS_COLLECTING", "Collecting events...")
                    events = await _collect_ws_events(ws, trace, timeout=60)
                    trace.step("WS_COLLECTED", f"{len(events)} events total")

        # Analyze events
        trace.step("VERIFY", "Analyzing WS events")
        event_types = [e.get("type") for e in events]
        trace.step("EVENT_TYPES", str(event_types))

        # Must have agent.status running
        status_events = [e for e in events if e.get("type") == "agent.status"]
        running = [e for e in status_events if e.get("status") == "running"]
        assert len(running) >= 1, f"No 'running' status event. Status events: {status_events}"
        trace.step("CHECK", "agent.status 'running' found")

        # Must have terminal status
        terminal = [e for e in status_events if e.get("status") in ("idle", "stopped")]
        assert len(terminal) >= 1, f"No terminal status. Got: {status_events}"
        trace.step("CHECK", f"Terminal status '{terminal[0].get('status')}' found")

        # Should have stream delta events
        stream_events = [e for e in events if e.get("type") == "chat.stream_delta"]
        trace.step("STREAM_EVENTS", f"{len(stream_events)} stream delta events")
        if stream_events:
            # Reconstruct streamed text
            streamed_text = "".join(e.get("text", "") for e in stream_events)
            trace.step("STREAMED_TEXT_PREVIEW", streamed_text[:200])

            # Verify structure
            for se in stream_events:
                assert "project_id" in se
                assert "text" in se
                assert "source" in se
        else:
            trace.bug("NO_STREAM_EVENTS",
                       "No chat.stream_delta events received during agent run",
                       severity="minor")

        # Should have activity events (tool usage)
        activity_events = [e for e in events if e.get("type") == "agent.activity"]
        trace.step("ACTIVITY_EVENTS", f"{len(activity_events)} activity events")
        for ae in activity_events:
            trace.step("ACTIVITY", f"category={ae.get('category')} tool={ae.get('tool_name')} "
                        f"desc={ae.get('description', '')[:60]}")

        # All events snake_case
        for evt in events:
            for key in evt.keys():
                assert key == key.lower(), f"Non-snake_case key '{key}'"
        trace.step("CHECK", "All event keys are snake_case")

        passed = True
        trace.step("RESULT", "PASSED")

    except Exception as e:
        trace.step("RESULT", f"FAILED: {e}")
        raise
    finally:
        _finish_trace(trace, passed)


# ---------------------------------------------------------------------------
# US-6: Developer stops agent mid-task
#
# "As a developer, I start the agent on a long task, then stop it.
#  The agent should halt and I should see the stopped status."
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_us6_stop_agent(tmp_path, workspace):
    """US-6: Start agent, then stop it, verify stopped event."""
    from httpx_ws import aconnect_ws
    from httpx_ws.transport import ASGIWebSocketTransport

    trace = _start_trace("US-6", "Stop agent mid-task")
    passed = False
    try:
        data_dir = str(tmp_path / "data_us6")
        os.makedirs(data_dir, exist_ok=True)
        app = create_app(data_dir=data_dir)

        async with ASGIWebSocketTransport(app=app) as transport:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                pid = await _create_project(client, workspace, trace)

                async with aconnect_ws("http://test/ws", client) as ws:
                    await ws.send_json({"type": "subscribe", "project_ids": [pid]})
                    await asyncio.wait_for(ws.receive_json(), timeout=5)

                    # Start agent with a long task
                    await _start_agent(client, pid,
                        "Count slowly from 1 to 1000, one number per line. Take your time.", trace)

                    # Wait a moment to let agent start
                    events = []
                    start = time.time()
                    while time.time() - start < 5:
                        try:
                            data = await asyncio.wait_for(ws.receive_json(), timeout=2)
                            events.append(data)
                            trace.ws_events.append({"t": round(time.time() - trace.started_at, 2), **data})
                            if (data.get("type") == "agent.status"
                                    and data.get("status") in ("idle", "error")):
                                break
                        except asyncio.TimeoutError:
                            break
                        except Exception:
                            break

                    trace.step("EVENTS_BEFORE_STOP", f"{len(events)} events collected before stop")

                    # Stop the agent
                    trace.step("REST", "POST /api/v2/agents/{pid}/stop")
                    resp = await client.post(f"/api/v2/agents/{pid}/stop")
                    trace.step("REST_RESPONSE", f"status={resp.status_code}", body=resp.json())
                    assert resp.status_code == 200
                    assert resp.json()["status"] == "stopping"

                    # Collect stopped event
                    trace.step("WS_COLLECTING", "Waiting for stopped event")
                    stop_start = time.time()
                    while time.time() - stop_start < 15:
                        try:
                            data = await asyncio.wait_for(ws.receive_json(), timeout=3)
                            events.append(data)
                            trace.ws_events.append({"t": round(time.time() - trace.started_at, 2), **data})
                            if (data.get("type") == "agent.status"
                                    and data.get("status") == "stopped"):
                                trace.step("STOPPED_EVENT", "Received agent.status stopped")
                                break
                        except asyncio.TimeoutError:
                            continue
                        except Exception:
                            break

        # Verify stopped
        status_events = [e for e in events if e.get("type") == "agent.status"]
        stopped = [e for e in status_events if e.get("status") == "stopped"]
        trace.step("VERIFY", f"Status events: {[(e.get('status'), e.get('reason','')) for e in status_events]}")
        assert len(stopped) >= 1, f"No stopped event found. Status events: {status_events}"

        passed = True
        trace.step("RESULT", "PASSED")

    except Exception as e:
        trace.step("RESULT", f"FAILED: {e}")
        raise
    finally:
        _finish_trace(trace, passed)


# ---------------------------------------------------------------------------
# US-7: Session persistence — JSONL on disk matches in-memory
#
# "As a developer, after the agent finishes, I expect the full
#  conversation to be persisted to disk and recoverable."
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_us7_session_persistence(client, workspace):
    """US-7: Session JSONL file matches in-memory messages."""
    trace = _start_trace("US-7", "Session persistence across hot resume")
    passed = False
    try:
        pid = await _create_project(client, workspace, trace)

        await _start_agent(client, pid,
            "What is the capital of France? Reply with just the city name.", trace)

        msgs = await _wait_for_idle(client, pid, trace)

        # Find the JSONL file
        trace.step("VERIFY", "Looking for session JSONL file")
        sessions_dir = os.path.join(workspace, "orbital", "sessions")
        trace.step("SESSIONS_DIR", sessions_dir)

        if not os.path.isdir(sessions_dir):
            trace.bug("NO_SESSIONS_DIR",
                       f"Sessions directory does not exist: {sessions_dir}",
                       severity="major")
            assert False, "Sessions directory missing"

        jsonl_files = [f for f in os.listdir(sessions_dir) if f.endswith(".jsonl")]
        trace.step("JSONL_FILES", str(jsonl_files))
        assert len(jsonl_files) >= 1, f"No JSONL files found in {sessions_dir}"

        jsonl_path = os.path.join(sessions_dir, jsonl_files[0])
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        persisted = [json.loads(line) for line in lines]

        trace.step("PERSISTENCE_CHECK",
                    f"In-memory: {len(msgs)} messages, JSONL: {len(persisted)} records")

        assert len(persisted) == len(msgs), (
            f"Mismatch: JSONL has {len(persisted)} vs in-memory {len(msgs)}"
        )

        # Verify roles match
        for i, (mem, disk) in enumerate(zip(msgs, persisted)):
            assert mem["role"] == disk["role"], (
                f"Role mismatch at index {i}: memory={mem['role']} disk={disk['role']}"
            )
        trace.step("CHECK", "All roles match between memory and disk")

        # Verify content of user message
        user_on_disk = [p for p in persisted if p["role"] == "user"]
        assert len(user_on_disk) >= 1
        assert "capital" in user_on_disk[0].get("content", "").lower() or \
               "France" in user_on_disk[0].get("content", "")
        trace.step("CHECK", "User message content preserved on disk")

        passed = True
        trace.step("RESULT", "PASSED")

    except Exception as e:
        trace.step("RESULT", f"FAILED: {e}")
        raise
    finally:
        _finish_trace(trace, passed)


# ---------------------------------------------------------------------------
# Report generation — runs after all tests
# ---------------------------------------------------------------------------


def _render_report():
    """Generate markdown report from collected traces."""
    os.makedirs(REPORT_DIR, exist_ok=True)

    lines = [
        "# User Story Test Report",
        "",
        f"*Generated: {datetime.now().isoformat()}*",
        f"*Model: {MODEL}*",
        f"*Base URL: {BASE_URL}*",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Stories tested | {len(_all_traces)} |",
        f"| Passed | {sum(1 for t in _all_traces if t['passed'])} |",
        f"| Failed | {sum(1 for t in _all_traces if not t['passed'])} |",
        f"| Bugs found | {len(_all_bugs)} |",
        "",
    ]

    # Per-story summary table
    lines += [
        "## Results by Story",
        "",
        "| ID | Title | Duration | Steps | WS Events | Messages | Bugs | Result |",
        "|----|-------|----------|-------|-----------|----------|------|--------|",
    ]
    for t in _all_traces:
        result = "PASS" if t["passed"] else "FAIL"
        lines.append(
            f"| {t['story_id']} | {t['story_title']} | {t['duration_s']}s | "
            f"{t['step_count']} | {t['ws_event_count']} | {t['chat_message_count']} | "
            f"{t['bug_count']} | {result} |"
        )
    lines.append("")

    # Detailed traces
    lines += ["---", "", "## Detailed Traces", ""]
    for t in _all_traces:
        lines.append(f"### {t['story_id']}: {t['story_title']}")
        lines.append("")
        lines.append(f"**Result:** {'PASS' if t['passed'] else 'FAIL'}  ")
        lines.append(f"**Duration:** {t['duration_s']}s  ")
        lines.append(f"**WS events:** {t['ws_event_count']}  ")
        lines.append(f"**Chat messages:** {t['chat_message_count']}")
        lines.append("")

        # Steps
        lines.append("#### Steps")
        lines.append("")
        lines.append("```")
        for s in t["steps"]:
            detail = s.get("detail", "")
            lines.append(f"[{s['t']:6.1f}s] {s['action']}: {detail}")
        lines.append("```")
        lines.append("")

        # Chat messages
        if t["chat_messages"]:
            lines.append("#### Chat Messages")
            lines.append("")
            for m in t["chat_messages"]:
                role = m.get("role", "?")
                content = m.get("content", "")
                if content:
                    preview = content[:200].replace("\n", " ")
                    lines.append(f"- **{role}**: {preview}")
                elif m.get("tool_calls"):
                    tc_names = []
                    for tc in m["tool_calls"]:
                        if "function" in tc:
                            tc_names.append(tc["function"].get("name", "?"))
                        else:
                            tc_names.append(tc.get("name", "?"))
                    lines.append(f"- **{role}**: [tool_calls: {', '.join(tc_names)}]")
                else:
                    lines.append(f"- **{role}**: (no content)")
            lines.append("")

        # WS events
        if t["ws_events"]:
            lines.append("#### WebSocket Events")
            lines.append("")
            lines.append("```")
            for e in t["ws_events"]:
                evt_type = e.get("type", "?")
                ts = e.get("t", 0)
                if evt_type == "agent.status":
                    lines.append(f"[{ts:6.1f}s] {evt_type}: status={e.get('status')}")
                elif evt_type == "chat.stream_delta":
                    text = e.get("text", "")
                    if text.strip():
                        lines.append(f"[{ts:6.1f}s] {evt_type}: \"{text[:50]}\"")
                elif evt_type == "agent.activity":
                    lines.append(f"[{ts:6.1f}s] {evt_type}: "
                                 f"category={e.get('category')} tool={e.get('tool_name')}")
                else:
                    lines.append(f"[{ts:6.1f}s] {evt_type}")
            lines.append("```")
            lines.append("")

        # Bugs
        if t["bugs"]:
            lines.append("#### Bugs")
            lines.append("")
            for b in t["bugs"]:
                lines.append(f"- **[{b['severity'].upper()}]** {b['title']}: {b['detail']}")
            lines.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Bug report (separate file)
    if _all_bugs:
        bug_lines = [
            "# User Story Bugs",
            "",
            f"*Generated: {datetime.now().isoformat()}*",
            "",
            "---",
            "",
            f"Total bugs found: {len(_all_bugs)}",
            "",
            "| # | Story | Severity | Title | Detail |",
            "|---|-------|----------|-------|--------|",
        ]
        for i, b in enumerate(_all_bugs, 1):
            bug_lines.append(
                f"| {i} | {b['story']} | {b['severity']} | {b['title']} | {b['detail']} |"
            )
        bug_lines += [
            "",
            "---",
            "",
            "## Diagnosis & Recommended Fixes",
            "",
            "*To be filled by Architect if bugs require architectural decisions.*",
            "",
        ]
        for i, b in enumerate(_all_bugs, 1):
            bug_lines += [
                f"### Bug #{i}: {b['title']}",
                "",
                f"- **Story:** {b['story']}",
                f"- **Severity:** {b['severity']}",
                f"- **Detail:** {b['detail']}",
                f"- **Diagnosis:** (pending)",
                f"- **Fix:** (pending)",
                "",
            ]

        with open(BUGS_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(bug_lines))


@pytest.fixture(scope="session", autouse=True)
def write_report_on_exit():
    """After all tests run, write the report."""
    yield
    _render_report()
