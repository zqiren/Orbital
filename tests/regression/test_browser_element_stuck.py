# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for browser tool infinite retry loop on unactionable elements.

Bug: agent made 36 browser calls over 15 minutes trying to click the same tweet.
Fix: hard refusal at 5 consecutive element-action failures, improved error messages.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.tools.browser import BrowserTool
from agent_os.agent.tools.base import ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_browser_manager():
    bm = MagicMock()
    page = AsyncMock()
    page.url = "https://example.com"
    page.title = AsyncMock(return_value="Example")
    bm.get_page = AsyncMock(return_value=page)
    bm.capture_screenshot = AsyncMock(return_value="/tmp/screenshot.png")
    bm.get_ref_map = MagicMock(return_value=None)
    bm.clear_ref_map = MagicMock()
    bm.store_ref_map = MagicMock()
    return bm


def _make_tool(bm=None):
    if bm is None:
        bm = _make_browser_manager()
    return BrowserTool(
        browser_manager=bm,
        project_id="test-project",
        workspace="/workspace",
        autonomy_preset="hands_off",
        session_id="default",
    )


def _fail_result(msg="Error: element not found"):
    return ToolResult(content=msg)


def _success_result(msg="Clicked element e5"):
    return ToolResult(content=msg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBrowserElementStuck:

    def test_hard_refusal_at_5_element_failures(self):
        """5 consecutive click failures triggers BLOCKED response."""
        tool = _make_tool()
        fail = _fail_result()

        # First 2: no advisory
        r1 = tool._track_result("click", fail)
        assert "Note:" not in r1.content
        assert "BLOCKED" not in r1.content

        r2 = tool._track_result("click", fail)
        assert "Note:" not in r2.content
        assert "BLOCKED" not in r2.content

        # 3-4: advisory note appended
        r3 = tool._track_result("click", fail)
        assert "Note:" in r3.content
        assert "BLOCKED" not in r3.content

        r4 = tool._track_result("click", fail)
        assert "Note:" in r4.content
        assert "BLOCKED" not in r4.content

        # 5: BLOCKED message
        r5 = tool._track_result("click", fail)
        assert "BLOCKED" in r5.content
        assert "Do NOT retry" in r5.content

        # After refusal, tracker["click"] is cleared
        assert "click" not in tool._action_failure_tracker

    def test_snapshot_does_not_reset_click_failures(self):
        """Snapshot success doesn't reset click failure counter."""
        tool = _make_tool()
        fail = _fail_result()
        snap_success = _success_result("Page snapshot: <html>...")

        # click fail -> count=1
        tool._track_result("click", fail)
        assert len(tool._action_failure_tracker.get("click", [])) == 1

        # snapshot success -> click counter stays 1
        tool._track_result("snapshot", snap_success)
        assert len(tool._action_failure_tracker.get("click", [])) == 1

        # click fail -> count=2
        tool._track_result("click", fail)
        assert len(tool._action_failure_tracker.get("click", [])) == 2

    def test_scroll_does_not_reset_click_failures(self):
        """Scroll success doesn't reset click failure counter."""
        tool = _make_tool()
        fail = _fail_result()
        scroll_success = _success_result("Scrolled down 500px")

        # click fail -> count=1
        tool._track_result("click", fail)
        assert len(tool._action_failure_tracker.get("click", [])) == 1

        # scroll success -> click counter stays 1
        tool._track_result("scroll", scroll_success)
        assert len(tool._action_failure_tracker.get("click", [])) == 1

        # click fail -> count=2
        tool._track_result("click", fail)
        assert len(tool._action_failure_tracker.get("click", [])) == 2

    def test_successful_click_resets_counter(self):
        """A successful click clears the failure counter."""
        tool = _make_tool()
        fail = _fail_result()
        success = _success_result()

        # 3 click failures
        tool._track_result("click", fail)
        tool._track_result("click", fail)
        tool._track_result("click", fail)
        assert len(tool._action_failure_tracker.get("click", [])) == 3

        # 1 success -> tracker cleared
        tool._track_result("click", success)
        assert "click" not in tool._action_failure_tracker

    def test_refusal_resets_tracker_for_retry_after_strategy_change(self):
        """After hard refusal, counter resets so agent can try after changing strategy."""
        tool = _make_tool()
        fail = _fail_result()

        # 5 failures -> BLOCKED -> tracker cleared
        for _ in range(5):
            tool._track_result("click", fail)

        assert "click" not in tool._action_failure_tracker

        # 1 more failure -> count=1 (fresh start)
        tool._track_result("click", fail)
        assert len(tool._action_failure_tracker.get("click", [])) == 1

    def test_non_element_actions_never_trigger_hard_refusal(self):
        """Snapshot failures get advisory but never hard refusal."""
        tool = _make_tool()
        fail = _fail_result("Error: snapshot empty")

        # 10 consecutive snapshot failures
        for i in range(10):
            r = tool._track_result("snapshot", fail)

        # Should get advisory at 3+ but never BLOCKED
        assert "Note:" in r.content
        assert "BLOCKED" not in r.content
        # Counter keeps growing, never cleared by refusal
        assert len(tool._action_failure_tracker.get("snapshot", [])) == 10

    def test_error_translation_discourages_retry(self):
        """_translate_error for 'waiting for locator' discourages retry."""
        tool = _make_tool()
        error = Exception("Timeout 30000ms exceeded. waiting for locator('button#submit')")
        result = tool._translate_error(error, "click", {"ref": "e5"})

        # Should NOT contain the old retry-encouraging message
        assert "check if it's still present" not in result.content
        # Should contain discouraging language
        assert "different element" in result.content or "different approach" in result.content

    def test_mixed_element_actions_tracked_independently(self):
        """Click and type failures tracked separately."""
        tool = _make_tool()
        click_fail = _fail_result("Error: click element not found")
        type_fail = _fail_result("Error: type element not found")

        # 3 click failures
        for _ in range(3):
            tool._track_result("click", click_fail)

        # 3 type failures
        for _ in range(3):
            tool._track_result("type", type_fail)

        # Each has its own counter
        assert len(tool._action_failure_tracker.get("click", [])) == 3
        assert len(tool._action_failure_tracker.get("type", [])) == 3

        # Both have advisory at 3
        r_click = tool._track_result("click", click_fail)
        assert "Note:" in r_click.content
        assert "browser:click" in r_click.content

        r_type = tool._track_result("type", type_fail)
        assert "Note:" in r_type.content
        assert "browser:type" in r_type.content

    def test_hard_refusal_result_meta(self):
        """Hard refusal result contains blocked_action and failure_count in meta."""
        tool = _make_tool()
        fail = _fail_result()

        # 5 click failures -> hard refusal
        for _ in range(4):
            tool._track_result("click", fail)
        r = tool._track_result("click", fail)

        assert r.meta is not None
        assert r.meta["blocked_action"] == "click"
        assert r.meta["failure_count"] == 5
