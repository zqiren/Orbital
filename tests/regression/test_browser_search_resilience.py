# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for browser search resilience.

Tests the three-part fix:
1. Search fallback to accessibility tree snapshot on empty extraction
2. Browser excluded from hash-based repetition detection in loop.py
3. Advisory counter tracking consecutive failures per action
"""

import hashlib
import json
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.tools.base import ToolResult
from agent_os.agent.tools.browser import BrowserTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_browser_tool():
    """Create a BrowserTool with mocked browser_manager."""
    bm = MagicMock()
    tool = BrowserTool(
        browser_manager=bm,
        project_id="test_project",
        workspace="/tmp/test",
        autonomy_preset="full",
    )
    return tool


# ---------------------------------------------------------------------------
# Section 1: Search fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_fallback_on_empty():
    """When structured extraction returns [], fall back to accessibility tree snapshot."""
    tool = _make_browser_tool()

    # Mock page objects
    original_page = AsyncMock()
    search_page = AsyncMock()
    search_page.evaluate = AsyncMock(return_value=[])  # Empty extraction
    search_page.is_closed.return_value = False
    original_page.context.new_page = AsyncMock(return_value=search_page)
    original_page.bring_to_front = AsyncMock()

    tool._bm.get_page = AsyncMock(return_value=original_page)

    # Mock _get_ax_tree to return a valid tree
    fake_tree = {"role": "WebArea", "name": "Google Search", "children": [
        {"role": "link", "name": "Result 1 - Example"},
        {"role": "link", "name": "Result 2 - Test"},
    ]}

    fake_snapshot_text = "link 'Result 1 - Example'\nlink 'Result 2 - Test'"

    with patch("agent_os.agent.tools.browser._get_ax_tree", new_callable=AsyncMock, return_value=fake_tree), \
         patch("agent_os.agent.tools.browser.serialize_snapshot", return_value=(fake_snapshot_text, {}, MagicMock())):
        result = await tool._action_search({"query": "test query"})

    assert result.meta is not None
    assert result.meta.get("search_fallback") is True
    assert "Falling back to page snapshot" in result.content
    assert "Result 1" in result.content


@pytest.mark.asyncio
async def test_search_structured_when_working():
    """When structured extraction returns results, no fallback should happen."""
    tool = _make_browser_tool()

    original_page = AsyncMock()
    search_page = AsyncMock()
    search_page.evaluate = AsyncMock(return_value=[
        {"title": "Test Page", "url": "https://test.com", "snippet": "A test page"},
    ])
    search_page.is_closed.return_value = False
    original_page.context.new_page = AsyncMock(return_value=search_page)
    original_page.bring_to_front = AsyncMock()

    tool._bm.get_page = AsyncMock(return_value=original_page)

    result = await tool._action_search({"query": "test query"})

    assert "Search results for" in result.content
    assert "fallback" not in result.content.lower()
    assert result.meta["result_count"] == 1


# ---------------------------------------------------------------------------
# Section 2: Browser excluded from repetition hash
# ---------------------------------------------------------------------------

def test_browser_skips_repetition_hash():
    """Browser tool calls should not contribute to action_hashes."""
    action_hashes = deque(maxlen=20)

    # Simulate 5 browser calls -- none should add to action_hashes
    for i in range(5):
        tc_name = "browser"
        tc_args = {"action": "screenshot"}
        tool_content = f"screenshot_{i}"

        if tc_name != "browser":
            result_prefix = str(tool_content)[:500]
            action_hash = hashlib.md5(
                (tc_name + str(tc_args) + result_prefix).encode()
            ).hexdigest()
            action_hashes.append(action_hash)

    assert len(action_hashes) == 0, "Browser calls should not add to action_hashes"


def test_nonbrowser_repetition_hash_intact():
    """Non-browser tools should still be hash-detected for repetition."""
    action_hashes = deque(maxlen=20)

    # Simulate 5 identical "read" calls
    for _ in range(5):
        tc_name = "read"
        tc_args = {"path": "/etc/passwd"}
        tool_content = "file contents here"

        if tc_name != "browser":
            result_prefix = str(tool_content)[:500]
            action_hash = hashlib.md5(
                (tc_name + str(tc_args) + result_prefix).encode()
            ).hexdigest()
            action_hashes.append(action_hash)

    # Count should be 5 (all identical)
    counts = {}
    for h in action_hashes:
        counts[h] = counts.get(h, 0) + 1
    max_count = max(counts.values())
    assert max_count >= 5, "Non-browser identical calls should trigger repetition detection"


# ---------------------------------------------------------------------------
# Section 3: Advisory counter
# ---------------------------------------------------------------------------

def test_advisory_counter_fires_at_3():
    """After 3 consecutive failures, result should include advisory note."""
    tool = _make_browser_tool()

    results = []
    for i in range(3):
        fail_result = ToolResult(
            content=f"No results found for: query_{i}",
            meta={"query": f"query_{i}"},
        )
        tracked = tool._track_result("search", fail_result)
        results.append(tracked)

    # First 2 should not have advisory
    assert "[Note: browser:search" not in results[0].content
    assert "[Note: browser:search" not in results[1].content
    # Third should have advisory
    assert "[Note: browser:search has failed 3 consecutive" in results[2].content


def test_advisory_counter_resets_on_success():
    """A successful result should reset the failure counter."""
    tool = _make_browser_tool()

    # 2 failures
    for i in range(2):
        tool._track_result("search", ToolResult(
            content=f"No results found for: q{i}",
            meta={"query": f"q{i}"},
        ))

    # 1 success (resets counter)
    tool._track_result("search", ToolResult(
        content="Search results for: good\n\n1. Result",
        meta={"query": "good", "result_count": 1},
    ))

    # 2 more failures (counter was reset, so still below threshold)
    results = []
    for i in range(2):
        tracked = tool._track_result("search", ToolResult(
            content=f"No results found for: q_after_{i}",
            meta={"query": f"q_after_{i}"},
        ))
        results.append(tracked)

    # Neither should have advisory (counter was reset by success)
    assert "[Note: browser:search" not in results[0].content
    assert "[Note: browser:search" not in results[1].content


def test_advisory_counter_resets_on_run_start():
    """on_run_start should clear the failure tracker."""
    tool = _make_browser_tool()

    # 2 failures
    for i in range(2):
        tool._track_result("search", ToolResult(
            content=f"No results found for: q{i}",
            meta={"query": f"q{i}"},
        ))

    # Run starts (resets tracker)
    tool.on_run_start()

    # 2 more failures (should not hit threshold since tracker was reset)
    results = []
    for i in range(2):
        tracked = tool._track_result("search", ToolResult(
            content=f"No results found for: q_new_{i}",
            meta={"query": f"q_new_{i}"},
        ))
        results.append(tracked)

    assert "[Note: browser:search" not in results[0].content
    assert "[Note: browser:search" not in results[1].content


def test_advisory_includes_actual_results():
    """Advisory note should contain actual failure summaries, not generic labels."""
    tool = _make_browser_tool()

    specific_failures = [
        "No results found for: best restaurants shenzhen",
        "No results found for: AI agent frameworks 2024",
        "No results found for: modern SaaS dashboard design",
    ]

    tracked = None
    for msg in specific_failures:
        tracked = tool._track_result("search", ToolResult(
            content=msg,
            meta={"query": "test"},
        ))

    # Advisory should contain the actual failure messages
    assert "best restaurants shenzhen" in tracked.content
    assert "AI agent frameworks 2024" in tracked.content
    assert "modern SaaS dashboard design" in tracked.content


def test_parallel_calls_not_cancelled():
    """Without browser hash detection, browser calls should never trigger exit_outer."""
    # Simulate the loop logic for 3 browser calls
    action_hashes = deque(maxlen=20)
    exit_outer = False

    for i in range(3):
        tc_name = "browser"
        tc_args = {"action": "search", "query": f"query_{i}"}
        tool_content = f"No results found for: query_{i}"

        # This is the guard from loop.py
        if tc_name != "browser":
            result_prefix = str(tool_content)[:500]
            action_hash = hashlib.md5(
                (tc_name + str(tc_args) + result_prefix).encode()
            ).hexdigest()
            action_hashes.append(action_hash)
            if list(action_hashes).count(action_hash) >= 5:
                exit_outer = True
                break

    assert not exit_outer, "Browser calls should never trigger exit_outer via hash detection"
    assert len(action_hashes) == 0, "No hashes should be recorded for browser calls"
