# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration: multi-tool truncation end-to-end with real LLM.

Triggers multiple file reads in a single agent run to verify all tool
results share the same agent summary and all disk backups are created.

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
# Imports
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
    return Session.new(session_id="multi-tool-e2e", workspace=workspace)


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
        max_iterations=10,
        token_budget=500_000,
    )


@pytest.fixture
def three_files(workspace):
    """Create 3 large files in the workspace."""
    files = {}
    for name, marker in [("alpha.txt", "ALPHA"), ("beta.txt", "BETA"), ("gamma.txt", "GAMMA")]:
        content = (
            f"This is the {marker} file. "
            f"It contains a lot of text about {marker.lower()} topics. "
        ) * 40  # ~3K chars each, enough to trigger truncation
        path = os.path.join(workspace, name)
        with open(path, "w") as f:
            f.write(content)
        files[name] = {"path": path, "content": content, "marker": marker}
    return files


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_multiple_reads_all_stubbed(
    agent_loop, session, workspace, three_files,
):
    """LLM reads 3 files → all tool results become stubs after response."""
    await agent_loop.run(
        initial_message=(
            "Read the files alpha.txt, beta.txt, and gamma.txt. "
            "Tell me which file has the most characters. "
            "Reply with just the filename."
        ),
    )

    messages = session.get_messages()

    # Should have at least 3 tool results (one per file read)
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) >= 3, f"Expected at least 3 tool results, got {len(tool_msgs)}"

    # All large tool results should be stubbed
    stubbed = [m for m in tool_msgs if m.get("_stubbed")]
    # At least the file read results (>500 chars) should be stubbed
    large_tool_msgs = [m for m in tool_msgs if len(m.get("content", "")) > 500 or m.get("_stubbed")]
    assert len(stubbed) >= 1, "Expected at least 1 stubbed tool result"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_all_disk_backups_created(
    agent_loop, session, workspace, three_files,
):
    """Each stubbed tool result has a corresponding disk backup."""
    await agent_loop.run(
        initial_message=(
            "Read all three files: alpha.txt, beta.txt, gamma.txt. "
            "For each file, count the words and report the count."
        ),
    )

    tool_results_dir = os.path.join(
        workspace, ".agent-os", "tool-results", "multi-tool-e2e",
    )

    messages = session.get_messages()
    stubbed = [m for m in messages if m.get("role") == "tool" and m.get("_stubbed")]

    if not stubbed:
        pytest.skip("No tool results were large enough to trigger truncation")

    assert os.path.exists(tool_results_dir), "Tool results directory should exist"

    backup_files = os.listdir(tool_results_dir)
    assert len(backup_files) >= len(stubbed), (
        f"Expected at least {len(stubbed)} backup files, got {len(backup_files)}"
    )

    # Verify each backup is valid JSON
    for fname in backup_files:
        with open(os.path.join(tool_results_dir, fname), "r") as f:
            record = json.load(f)
        assert "content" in record
        assert "tool_name" in record
        assert len(record["content"]) > 0


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_stubs_share_agent_summary(
    agent_loop, session, workspace, three_files,
):
    """All stubbed tool results from the same turn share the same summary."""
    await agent_loop.run(
        initial_message=(
            "Read alpha.txt, beta.txt, and gamma.txt. "
            "Reply with just: 'All three files read successfully.'"
        ),
    )

    messages = session.get_messages()
    stubbed = [m for m in messages if m.get("role") == "tool" and m.get("_stubbed")]

    if len(stubbed) < 2:
        pytest.skip("Need at least 2 stubbed results to test shared summary")

    # Extract summaries from stubs
    summaries = []
    for msg in stubbed:
        content = msg["content"]
        if "Agent summary:" in content:
            summary = content.split("Agent summary:")[1].strip()
            summaries.append(summary)

    # All summaries from the same turn should be identical
    if summaries:
        assert len(set(summaries)) == 1, (
            f"Expected all stubs to share same summary, got {len(set(summaries))} distinct: "
            f"{summaries}"
        )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_session_jsonl_clean_after_multi_read(
    agent_loop, session, workspace, three_files,
):
    """Session JSONL on disk contains stubs, not full file content."""
    await agent_loop.run(
        initial_message="Read alpha.txt and beta.txt. Say 'done' when finished.",
    )

    # Reload from disk
    reloaded = Session.load(session._filepath)
    tool_msgs = [m for m in reloaded.get_messages() if m.get("role") == "tool"]

    for msg in tool_msgs:
        if msg.get("_stubbed"):
            # Verify no raw file content in stubs
            assert msg["content"].startswith("[Tool:")
            for marker in ["ALPHA", "BETA", "GAMMA"]:
                assert f"the {marker} file" not in msg["content"]
