# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for stale browser ref invalidation on locator timeout.

Bug: when a click/type/fill action hits a "waiting for locator" / timeout
error, the ref stays in the RefMap and the agent retries the dead ref 4+
times before giving up. Evidence: Quick Tasks session had 4 failed clicks
on ref e217 over 74 seconds, all returning "did not become actionable."

Fix: on locator timeout, remove the specific failed ref from the project's
RefMap so the next attempt forces a fresh snapshot. Non-timeout errors
(strict mode, overlay) MUST NOT invalidate the ref.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.tools.browser import BrowserTool
from agent_os.agent.tools.browser_refs import RefEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_live_ref_map():
    """A ref_map dict shared across get_ref_map() calls (simulates live reference)."""
    return {
        "e5": RefEntry(role="button", name="Submit", nth=0),
        "e6": RefEntry(role="textbox", name="Email", nth=0),
        "e3": RefEntry(role="textbox", name="Name", nth=0),
    }


def _make_browser_manager(ref_map: dict):
    bm = MagicMock()
    page = AsyncMock()
    page.url = "https://example.com"
    page.title = AsyncMock(return_value="Example")
    bm.get_page = AsyncMock(return_value=page)
    bm.capture_screenshot = AsyncMock(return_value="/tmp/screenshot.png")
    # get_ref_map returns the SAME dict on every call (live reference, not a copy)
    bm.get_ref_map = MagicMock(return_value=ref_map)
    bm.clear_ref_map = MagicMock()
    bm.store_ref_map = MagicMock()
    return bm


def _make_tool(bm):
    return BrowserTool(
        browser_manager=bm,
        project_id="test-project",
        workspace="/workspace",
        autonomy_preset="hands_off",
        session_id="default",
    )


def _patch_resolve_ref(monkeypatch, side_effect):
    """Patch resolve_ref in browser module to raise a given exception."""
    async def _fake_resolve_ref(ref_map, ref, page):
        if callable(side_effect):
            return await side_effect(ref_map, ref, page)
        raise side_effect
    monkeypatch.setattr(
        "agent_os.agent.tools.browser.resolve_ref",
        _fake_resolve_ref,
    )


# ---------------------------------------------------------------------------
# Test A — stale ref removed from map after locator timeout (click)
# ---------------------------------------------------------------------------


class TestStaleRefInvalidation:

    @pytest.mark.asyncio
    async def test_click_timeout_removes_stale_ref(self, monkeypatch):
        """Click that hits locator timeout must remove the ref from the map."""
        ref_map = _make_live_ref_map()
        bm = _make_browser_manager(ref_map)
        tool = _make_tool(bm)

        _patch_resolve_ref(
            monkeypatch,
            Exception(
                "Timeout 30000ms exceeded.\n"
                "waiting for locator('button').filter({ hasText: 'Submit' })"
            ),
        )

        result = await tool.execute_async(action="click", ref="e5")

        content = result.content if isinstance(result.content, str) else str(result.content)
        # Existing behavior preserved
        assert "did not become actionable" in content
        # NEW: instructs the agent to re-snapshot
        assert "snapshot" in content.lower()
        # NEW: the dead ref is gone so the next attempt resolves to empty map
        assert "e5" not in ref_map
        # Other refs remain untouched
        assert "e6" in ref_map
        assert "e3" in ref_map

    @pytest.mark.asyncio
    async def test_type_timeout_removes_stale_ref(self, monkeypatch):
        """Type that hits locator timeout must also remove the ref."""
        ref_map = _make_live_ref_map()
        bm = _make_browser_manager(ref_map)
        tool = _make_tool(bm)

        _patch_resolve_ref(
            monkeypatch,
            Exception("Timeout 30000ms exceeded. waiting for locator('input#email')"),
        )

        result = await tool.execute_async(action="type", ref="e6", text="hi")

        content = result.content if isinstance(result.content, str) else str(result.content)
        assert "did not become actionable" in content
        assert "e6" not in ref_map
        assert "e5" in ref_map

    # -------------------------------------------------------------------
    # Test B — non-timeout errors do NOT invalidate the ref
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_strict_mode_violation_keeps_ref(self, monkeypatch):
        """Strict mode = multiple matches, not a stale ref. Ref MUST remain."""
        ref_map = _make_live_ref_map()
        bm = _make_browser_manager(ref_map)
        tool = _make_tool(bm)

        _patch_resolve_ref(
            monkeypatch,
            Exception("strict mode violation: locator resolved to 3 elements"),
        )

        result = await tool.execute_async(action="click", ref="e5")

        content = result.content if isinstance(result.content, str) else str(result.content)
        assert "Multiple elements" in content or "multiple" in content.lower()
        # The ref IS still in the map — strict mode is not a stale-ref signal
        assert "e5" in ref_map

    @pytest.mark.asyncio
    async def test_overlay_intercept_keeps_ref(self, monkeypatch):
        """Overlay/intercept errors must NOT invalidate the ref."""
        ref_map = _make_live_ref_map()
        bm = _make_browser_manager(ref_map)
        tool = _make_tool(bm)

        _patch_resolve_ref(
            monkeypatch,
            Exception("element intercepts pointer events"),
        )

        result = await tool.execute_async(action="click", ref="e5")

        content = result.content if isinstance(result.content, str) else str(result.content)
        assert "overlay" in content.lower() or "modal" in content.lower()
        # Overlay means the element is still there, just blocked
        assert "e5" in ref_map

    # -------------------------------------------------------------------
    # Test C — fill action also invalidates stale refs on timeout
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fill_timeout_removes_stale_ref(self, monkeypatch):
        """Fill per-field timeout must also remove the specific stale ref."""
        ref_map = _make_live_ref_map()
        bm = _make_browser_manager(ref_map)
        tool = _make_tool(bm)

        _patch_resolve_ref(
            monkeypatch,
            Exception(
                "Timeout 30000ms exceeded.\n"
                "waiting for locator('input[name=name]')"
            ),
        )

        result = await tool.execute_async(
            action="fill",
            fields=[{"ref": "e3", "value": "test"}],
        )

        content = result.content if isinstance(result.content, str) else str(result.content)
        # Fill surfaces the translated error per field
        assert "e3" in content  # field id is reported
        # NEW: the dead ref is removed from the live map
        assert "e3" not in ref_map
        # Other refs untouched
        assert "e5" in ref_map
        assert "e6" in ref_map
