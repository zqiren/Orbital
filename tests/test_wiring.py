# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase 2 wiring tests: real components A + B + C + E with a real LLM.

No mocks, no daemon. Tests verify the agent loop works end-to-end
using a real LLM API (kimi-k2.5 via Moonshot).

Skip all tests if AGENT_OS_TEST_API_KEY is not set.
"""

import asyncio
import json
import os

import pytest

# ---------------------------------------------------------------------------
# Config: read from env, skip entire module if not set
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("AGENT_OS_TEST_API_KEY", "")
BASE_URL = os.environ.get("AGENT_OS_TEST_BASE_URL", "https://api.moonshot.cn/v1")
MODEL = os.environ.get("AGENT_OS_TEST_MODEL", "kimi-k2.5")

skip_no_key = pytest.mark.skipif(
    not API_KEY,
    reason="AGENT_OS_TEST_API_KEY not set — skipping real LLM wiring tests",
)

pytestmark = [skip_no_key, pytest.mark.timeout(120)]

# ---------------------------------------------------------------------------
# Imports (only evaluated if tests run)
# ---------------------------------------------------------------------------

from agent_os.agent.providers.openai_compat import LLMProvider
from agent_os.agent.tools.registry import ToolRegistry
from agent_os.agent.tools.read import ReadTool
from agent_os.agent.tools.write import WriteTool
from agent_os.agent.tools.shell import ShellTool
from agent_os.agent.prompt_builder import PromptBuilder, PromptContext, Autonomy
from agent_os.agent.context import ContextManager
from agent_os.agent.session import Session
from agent_os.agent.loop import AgentLoop
from agent_os.agent.providers.types import StreamChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path):
    """Create a temp workspace directory."""
    return str(tmp_path)


@pytest.fixture
def provider():
    """Real LLM provider pointed at kimi-k2.5."""
    return LLMProvider(model=MODEL, api_key=API_KEY, base_url=BASE_URL)


@pytest.fixture
def registry(workspace):
    """Real tool registry with ReadTool, WriteTool, ShellTool."""
    reg = ToolRegistry()
    reg.register(ReadTool(workspace=workspace))
    reg.register(WriteTool(workspace=workspace))
    reg.register(ShellTool(workspace=workspace, os_type="windows"))
    return reg


@pytest.fixture
def prompt_builder(workspace):
    """Real prompt builder."""
    return PromptBuilder(workspace=workspace)


@pytest.fixture
def base_prompt_context(workspace, registry):
    """PromptContext with all fields filled."""
    return PromptContext(
        workspace=workspace,
        model=MODEL,
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=registry.tool_names(),
        os_type="windows",
        datetime_now="",
    )


@pytest.fixture
def session(workspace):
    """Fresh session in the temp workspace."""
    return Session.new(session_id="wiring-test", workspace=workspace)


@pytest.fixture
def context_manager(session, prompt_builder, base_prompt_context):
    """Real context manager."""
    return ContextManager(
        session=session,
        prompt_builder=prompt_builder,
        base_prompt_context=base_prompt_context,
        model_context_limit=128_000,
    )


@pytest.fixture
def agent_loop(session, provider, registry, context_manager):
    """Real agent loop wired with all real components."""
    return AgentLoop(
        session=session,
        provider=provider,
        tool_registry=registry,
        context_manager=context_manager,
        max_iterations=5,
        token_budget=500_000,
    )


# ---------------------------------------------------------------------------
# Test 1: Basic round-trip — simple question, text-only response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_basic_round_trip(agent_loop, session):
    """Send a trivial math question, verify the LLM answers correctly."""
    await agent_loop.run(initial_message="What is 2+2? Reply with just the number.")

    messages = session.get_messages()
    # Should have at least: user message + assistant response
    assert len(messages) >= 2, f"Expected >= 2 messages, got {len(messages)}"

    # First message is the user message
    assert messages[0]["role"] == "user"
    assert "2+2" in messages[0]["content"]

    # Find the final assistant message
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    assert len(assistant_msgs) >= 1, "No assistant message found"

    last_assistant = assistant_msgs[-1]
    assert last_assistant["content"] is not None
    assert "4" in last_assistant["content"], (
        f"Expected '4' in response, got: {last_assistant['content'][:200]}"
    )


# ---------------------------------------------------------------------------
# Test 2: Tool usage round-trip — agent reads a file
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_tool_usage_round_trip(workspace, provider, prompt_builder):
    """Create a test file, ask the agent to read it, verify tool was used."""
    # Create the test file in the workspace
    test_content = "Hello from wiring test 42"
    test_file = os.path.join(workspace, "test.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_content)

    # Build fresh components for this test
    reg = ToolRegistry()
    reg.register(ReadTool(workspace=workspace))
    reg.register(WriteTool(workspace=workspace))
    reg.register(ShellTool(workspace=workspace, os_type="windows"))

    sess = Session.new(session_id="wiring-tool-test", workspace=workspace)
    ctx = PromptContext(
        workspace=workspace,
        model=MODEL,
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=reg.tool_names(),
        os_type="windows",
        datetime_now="",
    )
    cm = ContextManager(sess, prompt_builder, ctx, model_context_limit=128_000)
    loop = AgentLoop(sess, provider, reg, cm, max_iterations=5, token_budget=500_000)

    await loop.run(
        initial_message="Read the file test.txt and tell me what it says. Use the read tool."
    )

    messages = sess.get_messages()

    # Check that a tool role message exists (meaning the agent called a tool)
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) >= 1, (
        f"Expected at least one tool result message, got {len(tool_msgs)}. "
        f"Messages: {[m['role'] for m in messages]}"
    )

    # Verify the tool result contains the file content
    tool_contents = [m.get("content", "") for m in tool_msgs]
    found_content = any(test_content in c for c in tool_contents)
    assert found_content, (
        f"Expected '{test_content}' in tool results, got: {tool_contents}"
    )

    # Verify the agent completed: either produced a text response referencing
    # the content, OR reached the tool result stage successfully.
    # The key wiring verification is that the tool was called and returned
    # the correct content — the LLM may or may not produce a final text
    # summary depending on model behavior and iteration limits.
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    assert len(assistant_msgs) >= 1, "No assistant message found"

    # Check if there's a text response mentioning file content
    text_assistant_msgs = [m for m in assistant_msgs if m.get("content")]
    if text_assistant_msgs:
        response_text = text_assistant_msgs[-1]["content"]
        assert "42" in response_text or "Hello" in response_text or test_content in response_text, (
            f"Expected file content reference in response, got: {response_text[:300]}"
        )
    else:
        # No text response, but the tool was called and returned content.
        # This is still a valid wiring test — the tool integration worked.
        # The LLM just didn't produce a follow-up text message (may have
        # hit iteration limits or the model's behavior). Verify the tool
        # result was appended correctly.
        assert found_content, "Tool was called but did not return expected content"


# ---------------------------------------------------------------------------
# Test 3: Session persistence — JSONL file contains all messages
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_session_persistence(agent_loop, session, workspace):
    """After a loop run, the JSONL file should contain all messages."""
    await agent_loop.run(initial_message="Say 'hello'. Reply with just that one word.")

    messages = session.get_messages()
    assert len(messages) >= 2

    # Find the JSONL file
    sessions_dir = os.path.join(workspace, "orbital", "sessions")
    jsonl_files = [f for f in os.listdir(sessions_dir) if f.endswith(".jsonl")]
    assert len(jsonl_files) == 1, f"Expected 1 JSONL file, got {jsonl_files}"

    jsonl_path = os.path.join(sessions_dir, jsonl_files[0])
    assert os.path.exists(jsonl_path)

    # Read and parse the JSONL
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    persisted = [json.loads(line) for line in lines]
    assert len(persisted) == len(messages), (
        f"JSONL has {len(persisted)} records but in-memory has {len(messages)}"
    )

    # Verify roles match
    for mem_msg, disk_msg in zip(messages, persisted):
        assert mem_msg["role"] == disk_msg["role"]


# ---------------------------------------------------------------------------
# Test 4: Streaming — on_stream callback receives chunks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_streaming_callback(agent_loop, session):
    """Attach an on_stream callback and verify chunks are received."""
    chunks_received: list[StreamChunk] = []

    def on_stream(chunk: StreamChunk):
        chunks_received.append(chunk)

    session.on_stream = on_stream

    await agent_loop.run(initial_message="What is 1+1? Reply with just the number.")

    # We should have received streaming chunks
    assert len(chunks_received) > 0, "No stream chunks received"

    # At least one chunk should have text content
    text_chunks = [c for c in chunks_received if c.text]
    assert len(text_chunks) > 0, "No text chunks received in stream"

    # The final chunk should be marked is_final
    final_chunks = [c for c in chunks_received if c.is_final]
    assert len(final_chunks) >= 1, "No final chunk received"
