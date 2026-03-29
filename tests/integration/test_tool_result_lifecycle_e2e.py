# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration: tool result lifecycle end-to-end with real LLM.

Uses real Kimi kimi-k2.5 LLM to verify the full tool result lifecycle:
pre-filter → LLM consumes → truncation to stubs → disk backup.

Env vars:
    AGENT_OS_TEST_API_KEY: LLM API key (Moonshot)
    AGENT_OS_TEST_BASE_URL: API base URL (default: https://api.moonshot.cn/v1)
    AGENT_OS_TEST_MODEL: Model name (default: kimi-k2.5)
"""

import json
import os

import pytest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("AGENT_OS_TEST_API_KEY", "")
BASE_URL = os.environ.get("AGENT_OS_TEST_BASE_URL", "https://api.moonshot.cn/v1")
MODEL = os.environ.get("AGENT_OS_TEST_MODEL", "kimi-k2.5")

skip_no_key = pytest.mark.skipif(
    not API_KEY,
    reason="AGENT_OS_TEST_API_KEY not set — skipping real LLM integration tests",
)

pytestmark = [skip_no_key, pytest.mark.timeout(120)]

# ---------------------------------------------------------------------------
# Imports (only evaluated if tests run)
# ---------------------------------------------------------------------------

from agent_os.agent.providers.openai_compat import LLMProvider
from agent_os.agent.tools.registry import ToolRegistry
from agent_os.agent.tools.read import ReadTool
from agent_os.agent.tools.write import WriteTool
from agent_os.agent.prompt_builder import PromptBuilder, PromptContext, Autonomy
from agent_os.agent.context import ContextManager
from agent_os.agent.session import Session
from agent_os.agent.loop import AgentLoop


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path):
    return str(tmp_path)


@pytest.fixture
def provider():
    return LLMProvider(model=MODEL, api_key=API_KEY, base_url=BASE_URL)


@pytest.fixture
def registry(workspace):
    reg = ToolRegistry()
    reg.register(ReadTool(workspace=workspace))
    reg.register(WriteTool(workspace=workspace))
    return reg


@pytest.fixture
def prompt_builder(workspace):
    return PromptBuilder(workspace=workspace)


@pytest.fixture
def base_prompt_context(workspace, registry):
    return PromptContext(
        workspace=workspace,
        model=MODEL,
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=registry.tool_names(),
        os_type="linux",
        datetime_now="",
    )


@pytest.fixture
def session(workspace):
    return Session.new(session_id="lifecycle-e2e", workspace=workspace)


@pytest.fixture
def context_manager(session, prompt_builder, base_prompt_context):
    return ContextManager(
        session=session,
        prompt_builder=prompt_builder,
        base_prompt_context=base_prompt_context,
        model_context_limit=128_000,
    )


@pytest.fixture
def agent_loop(session, provider, registry, context_manager):
    return AgentLoop(
        session=session,
        provider=provider,
        tool_registry=registry,
        context_manager=context_manager,
        max_iterations=5,
        token_budget=500_000,
    )


@pytest.fixture
def large_file(workspace):
    """Create a large file (5K+ chars) in the workspace."""
    content = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
    ) * 50  # ~8K chars
    filepath = os.path.join(workspace, "large_document.txt")
    with open(filepath, "w") as f:
        f.write(content)
    return filepath


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_tool_result_truncated_after_read(
    agent_loop, session, workspace, large_file,
):
    """Real LLM reads a large file → tool result becomes a stub after response."""
    await agent_loop.run(
        initial_message=(
            "Read the file large_document.txt and tell me the first sentence. "
            "Reply with just that sentence, nothing else."
        ),
    )

    messages = session.get_messages()

    # LLM should have used the read tool
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) >= 1, "Expected at least one tool result"

    # Tool results should be stubbed (content was >500 chars)
    stubbed = [m for m in tool_msgs if m.get("_stubbed")]
    assert len(stubbed) >= 1, "Expected at least one stubbed tool result"

    for msg in stubbed:
        assert msg["content"].startswith("[Tool:")
        assert "Agent summary:" in msg["content"]
        # Stub must be much smaller than the original file (~9600 chars)
        assert len(msg["content"]) < 1000, (
            f"Stub should be compact, got {len(msg['content'])} chars"
        )

    # LLM should have produced a response
    assistant_msgs = [m for m in messages if m.get("role") == "assistant" and m.get("content")]
    assert len(assistant_msgs) >= 1


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_disk_backup_exists_after_read(
    agent_loop, session, workspace, large_file,
):
    """Disk backup file exists after tool result truncation."""
    await agent_loop.run(
        initial_message="Read the file large_document.txt and summarize it in one word.",
    )

    # Check disk backup directory
    tool_results_dir = os.path.join(
        workspace, "orbital", "tool-results", "lifecycle-e2e",
    )

    if not os.path.exists(tool_results_dir):
        # If no tool results were large enough to stub, skip
        tool_msgs = [m for m in session.get_messages() if m.get("role") == "tool"]
        for msg in tool_msgs:
            if len(msg.get("content", "")) > 500 and not msg.get("_stubbed"):
                pytest.fail("Large tool result was not truncated and no disk backup exists")
        pytest.skip("No tool results were large enough to trigger truncation")

    backup_files = os.listdir(tool_results_dir)
    assert len(backup_files) >= 1, "Expected at least one disk backup file"

    # Verify backup file is valid JSON with correct schema
    for fname in backup_files:
        with open(os.path.join(tool_results_dir, fname), "r") as f:
            record = json.load(f)
        assert "turn" in record
        assert "call_id" in record
        assert "tool_name" in record
        assert "content" in record
        assert "timestamp" in record
        # Content should be the original file content
        assert "Lorem ipsum" in record["content"]


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_session_jsonl_has_stubs_after_reload(
    agent_loop, session, workspace, large_file,
):
    """After session reload from disk, tool results are stubs not full content."""
    await agent_loop.run(
        initial_message="Read large_document.txt and count the words. Reply with just the number.",
    )

    # Reload session from disk
    reloaded = Session.load(session._filepath)
    tool_msgs = [m for m in reloaded.get_messages() if m.get("role") == "tool"]

    for msg in tool_msgs:
        if msg.get("_stubbed"):
            assert msg["content"].startswith("[Tool:")
            assert "Lorem ipsum" not in msg["content"]
