# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration smoke tests for multimodal image pipeline.

Same pattern as test_wiring.py — uses real Kimi K2.5 via Moonshot.
Skip all tests if AGENT_OS_TEST_API_KEY is not set.
"""

import json
import os
import struct
import zlib

import pytest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("AGENT_OS_TEST_API_KEY", "")
BASE_URL = os.environ.get("AGENT_OS_TEST_BASE_URL", "https://api.moonshot.cn/v1")
MODEL = os.environ.get("AGENT_OS_TEST_MODEL", "kimi-k2.5")

skip_no_key = pytest.mark.skipif(
    not API_KEY,
    reason="AGENT_OS_TEST_API_KEY not set — skipping multimodal smoke tests",
)

pytestmark = [skip_no_key, pytest.mark.timeout(120)]

# ---------------------------------------------------------------------------
# Imports (only evaluated if tests run)
# ---------------------------------------------------------------------------

from agent_os.agent.providers.openai_compat import LLMProvider, _flatten_multimodal_content
from agent_os.agent.tools.registry import ToolRegistry
from agent_os.agent.tools.read import ReadTool
from agent_os.agent.prompt_builder import PromptBuilder, PromptContext, Autonomy
from agent_os.agent.context import ContextManager
from agent_os.agent.session import Session
from agent_os.agent.loop import AgentLoop
from agent_os.config.provider_registry import ModelCapabilities


# ---------------------------------------------------------------------------
# Test PNG generator (no PIL)
# ---------------------------------------------------------------------------

def make_test_png(path, width=100, height=100, r=255, g=0, b=0):
    """Write a solid-color PNG. No dependencies."""
    raw = b""
    for _ in range(height):
        raw += b"\x00" + bytes([r, g, b]) * width

    def chunk(ctype, data):
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", zlib.compress(raw)))
        f.write(chunk(b"IEND", b""))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path):
    return str(tmp_path)


@pytest.fixture
def vision_provider():
    """LLM provider with vision capabilities."""
    caps = ModelCapabilities(vision=True, tool_use=True, streaming=True)
    return LLMProvider(
        model=MODEL, api_key=API_KEY, base_url=BASE_URL,
        capabilities=caps,
    )


@pytest.fixture
def non_vision_provider():
    """LLM provider without vision capabilities."""
    caps = ModelCapabilities(vision=False, tool_use=True, streaming=True)
    return LLMProvider(
        model=MODEL, api_key=API_KEY, base_url=BASE_URL,
        capabilities=caps,
    )


@pytest.fixture
def registry(workspace):
    reg = ToolRegistry()
    reg.register(ReadTool(workspace=workspace))
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
    return Session.new(session_id="multimodal-smoke", workspace=workspace)


@pytest.fixture
def context_manager(session, prompt_builder, base_prompt_context):
    return ContextManager(
        session=session,
        prompt_builder=prompt_builder,
        base_prompt_context=base_prompt_context,
        model_context_limit=128_000,
    )


@pytest.fixture
def agent_loop(session, vision_provider, registry, context_manager):
    return AgentLoop(
        session=session,
        provider=vision_provider,
        tool_registry=registry,
        context_manager=context_manager,
        max_iterations=5,
        token_budget=500_000,
    )


# ---------------------------------------------------------------------------
# Test 1: Pipeline e2e — ReadTool → LLM sees image
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_read_image_pipeline(workspace, agent_loop, session):
    """Create a red PNG, ask the LLM to read and describe it.

    Verifies:
    - Tool result in session has multimodal content (list)
    - LLM response is substantive (>20 words)
    - No 'cannot read' / 'garbled' errors
    - Response references something visual
    """
    # Create 100x100 solid red PNG
    png_path = os.path.join(workspace, "test_image.png")
    make_test_png(png_path, 100, 100, 255, 0, 0)

    await agent_loop.run(
        initial_message="Read the file test_image.png and describe what you see. "
                       "What color is it? What shape?"
    )

    messages = session.get_messages()

    # Find tool result
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) >= 1, "Expected at least one tool result"

    # Verify multimodal content in tool result
    tool_content = tool_msgs[0]["content"]
    assert isinstance(tool_content, list), (
        f"Tool result should be multimodal list, got: {type(tool_content)}"
    )
    assert any(b.get("type") == "image_url" for b in tool_content), (
        "Tool result should contain image_url block"
    )

    # Find assistant response after tool result
    assistant_msgs = [m for m in messages if m.get("role") == "assistant" and m.get("content")]
    assert len(assistant_msgs) >= 1, "No assistant response found"

    last_response = assistant_msgs[-1]["content"]
    assert last_response is not None

    # Response should be substantive
    word_count = len(last_response.split())
    assert word_count > 20, f"Response too short ({word_count} words): {last_response[:200]}"

    # Response should NOT contain error indicators
    error_phrases = ["cannot read", "garbled", "unable to process", "error reading", "binary"]
    for phrase in error_phrases:
        assert phrase not in last_response.lower(), (
            f"Response contains error phrase '{phrase}': {last_response[:200]}"
        )

    # Response should reference something visual
    visual_words = ["red", "color", "image", "square", "pixel", "solid", "bright", "picture"]
    assert any(w in last_response.lower() for w in visual_words), (
        f"Response doesn't mention anything visual: {last_response[:200]}"
    )


# ---------------------------------------------------------------------------
# Test 2: Non-vision fallback
# ---------------------------------------------------------------------------

class TestNonVisionFallback:
    def test_non_vision_flattens_multimodal_tool_result(self, workspace, non_vision_provider):
        """Non-vision provider should flatten image blocks to text descriptions."""
        # Create a multimodal tool result (as ReadTool would produce)
        tool_msg = {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {"type": "text", "text": "Image file: photo.png, 100x100, 5KB, image/png"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBOR",
                        "detail": "low",
                    },
                },
            ],
        }

        result = non_vision_provider._prepare_messages_openai([tool_msg])

        flattened = result[0]
        assert isinstance(flattened["content"], str), "Non-vision should flatten to string"
        assert "[image omitted]" in flattened["content"]
        assert "photo.png" in flattened["content"]
        # Should NOT contain base64 data
        assert "iVBOR" not in flattened["content"]


# ---------------------------------------------------------------------------
# Test 3: History pruning round-trip
# ---------------------------------------------------------------------------

class TestHistoryPruning:
    def test_image_replaced_with_reference_in_turn_2(
        self, workspace, session, prompt_builder, base_prompt_context
    ):
        """After image is analyzed in turn 1, turn 2 context should have text reference, not base64."""
        # Simulate turn 1: user → assistant calls read → tool returns image → assistant describes
        session.append({
            "role": "user",
            "content": "Read test_image.png",
        })
        session.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_img", "type": "function",
                 "function": {"name": "read", "arguments": '{"path":"test_image.png"}'}}
            ],
        })
        session.append_tool_result(
            "call_img",
            [
                {"type": "text", "text": "Image file: test_image.png, 100x100, 5KB, image/png"},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + "A" * 5000,
                    "detail": "low",
                }},
            ],
            meta={"image_path": os.path.join(workspace, "test_image.png"), "mime": "image/png", "size": 5000},
        )
        session.append({
            "role": "assistant",
            "content": "I see a solid red square, 100x100 pixels.",
        })

        # Simulate turn 2: user asks follow-up
        session.append({
            "role": "user",
            "content": "What color was that image?",
        })
        session.append({
            "role": "assistant",
            "content": "The image was red.",
        })

        # Now simulate turn 3 context assembly
        session.append({
            "role": "user",
            "content": "Can you remind me of the dimensions?",
        })

        cm = ContextManager(
            session=session,
            prompt_builder=prompt_builder,
            base_prompt_context=base_prompt_context,
            model_context_limit=128_000,
        )
        context = cm.prepare()

        # Find the tool result in the assembled context
        tool_msgs = [m for m in context if m.get("role") == "tool"]
        assert len(tool_msgs) >= 1, "Tool result should appear in context"

        tool_content = tool_msgs[0]["content"]

        # Should be pruned to text reference (image is >1 turn ago)
        assert isinstance(tool_content, str), (
            f"Image should be pruned to string in turn 3, got: {type(tool_content)}"
        )
        assert "[Image:" in tool_content, f"Should have [Image: prefix, got: {tool_content}"
        assert "test_image.png" in tool_content

        # Verify NO base64 data in the entire context
        context_str = json.dumps(context)
        assert "AAAAA" not in context_str, "Base64 data should not appear in turn 3 context"
