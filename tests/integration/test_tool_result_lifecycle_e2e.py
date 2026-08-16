# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration: tool result lifecycle end-to-end with real LLM.

Uses real Kimi kimi-k2.5 LLM to verify the full tool result lifecycle:
pre-filter → LLM consumes → disk archive → history left intact.

The lifecycle no longer stubs consumed tool results: blanket history mutation
took documents away from the agent mid-turn. What is asserted here is the new
contract — the read result stays readable in history, AND the full content is
archived to orbital/tool-results/ regardless.

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

# The wall-clock a turn needs depends entirely on the model under test, so a
# hard-coded number is only ever right for one of them. 120s fits the default
# kimi-k2.5; MiniMax-M3 needs far more, because its reasoning is locked on
# (model_only — `disable_reasoning` is a no-op) and every iteration pays for a
# think block. Override with AGENT_OS_TEST_TIMEOUT when pointing these at a
# slower model, e.g. AGENT_OS_TEST_TIMEOUT=600 for M3.
TIMEOUT = int(os.environ.get("AGENT_OS_TEST_TIMEOUT", "120"))

pytestmark = [skip_no_key, pytest.mark.timeout(TIMEOUT)]

# ---------------------------------------------------------------------------
# Imports (only evaluated if tests run)
# ---------------------------------------------------------------------------

from agent_os.agent.providers.openai_compat import LLMProvider
from agent_os.agent.tools.registry import ToolRegistry
from agent_os.agent.tools.read import ReadTool
from agent_os.agent.tools.write import WriteTool
from agent_os.agent.prompt_builder import PromptBuilder, PromptContext, Autonomy
from agent_os.agent.context import ContextManager
from agent_os.agent.session import Session, persist_user_row
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
    return Session.new(session_uuid="lifecycle-e2e", workspace=workspace)


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
@pytest.mark.timeout(TIMEOUT)
async def test_tool_result_stays_live_after_read(
    agent_loop, session, workspace, large_file,
):
    """Real LLM reads a large file → the tool result stays readable in history.

    This is the inversion of the old assertion. A single read is never
    superseded, so nothing about it is rewritten: the agent can still see the
    document it just read on every subsequent iteration.
    """
    persist_user_row(
        agent_loop._session,
        "Read the file large_document.txt and tell me the first sentence. "
        "Reply with just that sentence, nothing else."
    )
    await agent_loop.run()

    messages = session.get_messages()

    # LLM should have used the read tool
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) >= 1, "Expected at least one tool result"

    # No tool result may be stubbed: the file was read once, so nothing
    # superseded it.
    stubbed = [m for m in tool_msgs if m.get("_stubbed")]
    assert not stubbed, (
        f"A single read must not be stubbed, got {len(stubbed)} stub(s): "
        f"{[m['content'][:120] for m in stubbed]}"
    )

    # The document itself is still in history.
    assert any(
        "Lorem ipsum" in m.get("content", "")
        for m in tool_msgs
        if isinstance(m.get("content"), str)
    ), "The file content must remain in the session after the LLM responds"

    # LLM should have produced a response
    assistant_msgs = [m for m in messages if m.get("role") == "assistant" and m.get("content")]
    assert len(assistant_msgs) >= 1


@pytest.mark.asyncio
@pytest.mark.timeout(TIMEOUT)
async def test_disk_backup_exists_after_read(
    agent_loop, session, workspace, large_file,
):
    """Disk archive exists even though the tool result was never stubbed.

    The archive is unconditional — it does not depend on history being
    rewritten. It is the corpus the lifecycle work is measured against.
    """
    persist_user_row(
        agent_loop._session,
        "Read the file large_document.txt and summarize it in one word.",
    )
    await agent_loop.run()

    # Check disk archive directory
    tool_results_dir = os.path.join(
        workspace, "orbital", "tool-results", "lifecycle-e2e",
    )

    tool_msgs = [m for m in session.get_messages() if m.get("role") == "tool"]
    large = [
        m for m in tool_msgs
        if isinstance(m.get("content"), str) and len(m["content"]) > 500
    ]
    if not large:
        pytest.skip("No tool results were large enough to trigger archiving")

    assert os.path.exists(tool_results_dir), (
        "Large tool results must be archived to disk even when not stubbed"
    )

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
@pytest.mark.timeout(TIMEOUT)
async def test_session_jsonl_keeps_content_after_reload(
    agent_loop, session, workspace, large_file,
):
    """After session reload from disk, the read result still holds the content."""
    persist_user_row(
        agent_loop._session,
        "Read large_document.txt and count the words. Reply with just the number.",
    )
    await agent_loop.run()

    # Reload session from disk
    reloaded = Session.load(session._filepath)
    tool_msgs = [m for m in reloaded.get_messages() if m.get("role") == "tool"]

    assert not any(m.get("_stubbed") for m in tool_msgs), (
        "A file read once must not be stubbed in the persisted session"
    )
    assert any(
        "Lorem ipsum" in m.get("content", "")
        for m in tool_msgs
        if isinstance(m.get("content"), str)
    )


@pytest.mark.asyncio
@pytest.mark.timeout(TIMEOUT)
async def test_reread_supersedes_the_earlier_copy(
    agent_loop, session, workspace, large_file,
):
    """Two reads of one path → the earlier copy is stubbed, the newest survives.

    Whether the model actually re-reads is up to the model, so this skips
    rather than fails when only one read happened.
    """
    persist_user_row(
        agent_loop._session,
        "Read the file large_document.txt. Then read large_document.txt "
        "again to double-check. Reply with just: done."
    )
    await agent_loop.run()

    tool_msgs = [
        m for m in session.get_messages()
        if m.get("role") == "tool" and isinstance(m.get("content"), str)
    ]
    if len(tool_msgs) < 2:
        pytest.skip("Model did not re-read the file — nothing to supersede")

    stubbed = [m for m in tool_msgs if m.get("_stubbed")]
    assert stubbed, "A re-read of the same path must supersede the earlier copy"

    for msg in stubbed:
        assert msg["content"].startswith("[SUPERSEDED")
        assert "NOT the content" in msg["content"]
        assert "Agent summary:" not in msg["content"]

    # Exactly one live copy of the file remains, and it is the newest.
    live = [m for m in tool_msgs if not m.get("_stubbed")]
    assert live, "The newest copy must survive"
    assert live[-1] is tool_msgs[-1]
