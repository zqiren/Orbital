# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration: multi-tool result lifecycle end-to-end with real LLM.

Triggers multiple file reads in a single agent run to verify that reads of
DIFFERENT files all stay live in history (none supersedes another) while every
one of them is still archived to disk.

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
# Imports
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
    return Session.new(session_uuid="multi-tool-e2e", workspace=workspace)


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
@pytest.mark.timeout(TIMEOUT)
async def test_multiple_reads_of_distinct_files_all_stay_live(
    agent_loop, session, workspace, three_files,
):
    """LLM reads 3 different files → all three results stay in history.

    Three distinct paths are three distinct targets, so nothing supersedes
    anything. The comparison the model was asked to make needs all three
    present at once — that is precisely what blanket stubbing broke.
    """
    persist_user_row(
        agent_loop._session,
        "Read the files alpha.txt, beta.txt, and gamma.txt. "
        "Tell me which file has the most characters. "
        "Reply with just the filename."
    )
    await agent_loop.run()

    messages = session.get_messages()

    # Should have at least 3 tool results (one per file read)
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) >= 3, f"Expected at least 3 tool results, got {len(tool_msgs)}"

    stubbed = [m for m in tool_msgs if m.get("_stubbed")]
    assert not stubbed, (
        f"Reads of distinct paths must not stub each other, got {len(stubbed)}"
    )

    # Each file's content is still visible.
    blob = " ".join(
        m["content"] for m in tool_msgs if isinstance(m.get("content"), str)
    )
    for marker in ("ALPHA", "BETA", "GAMMA"):
        assert f"the {marker} file" in blob, f"{marker} content missing from history"


@pytest.mark.asyncio
@pytest.mark.timeout(TIMEOUT)
async def test_all_disk_backups_created(
    agent_loop, session, workspace, three_files,
):
    """Every large tool result has a disk archive, stubbed or not."""
    persist_user_row(
        agent_loop._session,
        "Read all three files: alpha.txt, beta.txt, gamma.txt. "
        "For each file, count the words and report the count."
    )
    await agent_loop.run()

    tool_results_dir = os.path.join(
        workspace, "orbital", "tool-results", "multi-tool-e2e",
    )

    messages = session.get_messages()
    large = [
        m for m in messages
        if m.get("role") == "tool"
        and (m.get("_stubbed") or (
            isinstance(m.get("content"), str) and len(m["content"]) > 500
        ))
    ]

    if not large:
        pytest.skip("No tool results were large enough to trigger archiving")

    assert os.path.exists(tool_results_dir), "Tool results directory should exist"

    backup_files = os.listdir(tool_results_dir)
    assert len(backup_files) >= len(large), (
        f"Expected at least {len(large)} backup files, got {len(backup_files)}"
    )

    # Verify each backup is valid JSON
    for fname in backup_files:
        with open(os.path.join(tool_results_dir, fname), "r") as f:
            record = json.load(f)
        assert "content" in record
        assert "tool_name" in record
        assert len(record["content"]) > 0


@pytest.mark.asyncio
@pytest.mark.timeout(TIMEOUT)
async def test_any_stub_is_a_supersession_stub_and_is_honest(
    agent_loop, session, workspace, three_files,
):
    """The only stub reachable is the supersession stub, and it must be honest:
    it leads with the absence and never carries the model's narration."""
    persist_user_row(
        agent_loop._session,
        "Read alpha.txt, beta.txt, and gamma.txt. "
        "Reply with just: 'All three files read successfully.'"
    )
    await agent_loop.run()

    messages = session.get_messages()
    stubbed = [m for m in messages if m.get("role") == "tool" and m.get("_stubbed")]

    if not stubbed:
        pytest.skip("No result was re-fetched, so nothing was superseded")

    narrations = [
        m["content"] for m in messages
        if m.get("role") == "assistant" and isinstance(m.get("content"), str)
        and m["content"].strip()
    ]

    for msg in stubbed:
        content = msg["content"]
        assert content.startswith("[SUPERSEDED")
        assert "NOT the content" in content
        assert "Agent summary:" not in content
        for narration in narrations:
            assert narration[:40] not in content


@pytest.mark.asyncio
@pytest.mark.timeout(TIMEOUT)
async def test_session_jsonl_keeps_content_after_multi_read(
    agent_loop, session, workspace, three_files,
):
    """Session JSONL on disk keeps the file content for un-superseded reads."""
    persist_user_row(agent_loop._session, "Read alpha.txt and beta.txt. Say 'done' when finished.")
    await agent_loop.run()

    # Reload from disk
    reloaded = Session.load(session._filepath)
    tool_msgs = [m for m in reloaded.get_messages() if m.get("role") == "tool"]

    live_blob = " ".join(
        m["content"] for m in tool_msgs
        if not m.get("_stubbed") and isinstance(m.get("content"), str)
    )
    for marker in ["ALPHA", "BETA"]:
        assert f"the {marker} file" in live_blob, (
            f"{marker} content must survive in the persisted session"
        )

    for msg in tool_msgs:
        if msg.get("_stubbed"):
            # A stub never carries file content — that is the whole point.
            assert msg["content"].startswith("[SUPERSEDED")
            for marker in ["ALPHA", "BETA", "GAMMA"]:
                assert f"the {marker} file" not in msg["content"]
