# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for BrowserTool — 25 tests with mocked Playwright and dependencies."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from agent_os.agent.tools.browser import BrowserTool, BROWSER_WRITE_ACTIONS, BROWSER_OBSERVATION_ACTIONS
from agent_os.agent.tools.browser_refs import RefEntry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_page(url="https://example.com", title="Example"):
    """Create a mock Playwright Page object."""
    page = AsyncMock()
    page.url = url
    page.title = AsyncMock(return_value=title)
    page.is_closed = MagicMock(return_value=False)
    page.goto = AsyncMock()
    page.go_back = AsyncMock()
    page.go_forward = AsyncMock()
    page.reload = AsyncMock()
    page.inner_text = AsyncMock(return_value="Hello world page content")
    page.keyboard = AsyncMock()
    page.keyboard.press = AsyncMock()
    page.mouse = AsyncMock()
    page.mouse.wheel = AsyncMock()
    page.evaluate = AsyncMock(return_value=None)
    page.wait_for_selector = AsyncMock()
    page.wait_for_url = AsyncMock()
    page.wait_for_load_state = AsyncMock()
    page.pdf = AsyncMock()
    page.bring_to_front = AsyncMock()
    page.close = AsyncMock()
    page.accessibility = MagicMock()
    page.accessibility.snapshot = AsyncMock(return_value={
        "role": "WebArea",
        "name": "Example",
        "children": [
            {"role": "heading", "name": "Example", "level": 1},
            {"role": "link", "name": "Click me"},
        ],
    })
    page.screenshot = AsyncMock()
    return page


def _make_mock_locator():
    """Create a mock Playwright Locator."""
    locator = AsyncMock()
    locator.click = AsyncMock()
    locator.dblclick = AsyncMock()
    locator.fill = AsyncMock()
    locator.hover = AsyncMock()
    locator.select_option = AsyncMock()
    locator.scroll_into_view_if_needed = AsyncMock()
    locator.drag_to = AsyncMock()
    locator.set_input_files = AsyncMock()
    locator.bounding_box = AsyncMock(return_value={"x": 10, "y": 20, "width": 100, "height": 30})
    return locator


def _make_browser_manager(page=None):
    """Create a mock BrowserManager."""
    bm = MagicMock()
    if page is None:
        page = _make_mock_page()
    bm.get_page = AsyncMock(return_value=page)
    bm.capture_screenshot = AsyncMock(return_value="/workspace/orbital/output/screenshots/step_0001.png")
    bm.new_tab = AsyncMock(return_value=page)
    bm.get_all_pages = AsyncMock(return_value=[page])
    bm.close_project_pages = AsyncMock()
    bm.store_ref_map = MagicMock()
    bm.get_ref_map = MagicMock(return_value=None)
    bm.clear_ref_map = MagicMock()
    return bm


def _make_ref_map():
    """Create a sample RefMap with a few entries."""
    return {
        "e1": RefEntry(role="heading", name="Example", nth=0),
        "e2": RefEntry(role="link", name="Click me", nth=0),
        "e5": RefEntry(role="button", name="Submit", nth=0),
    }


def _make_tool(bm=None, page=None, autonomy_preset="hands_off", user_credential_store=None):
    """Create a BrowserTool with mocked dependencies."""
    if bm is None:
        bm = _make_browser_manager(page)
    return BrowserTool(
        browser_manager=bm,
        project_id="test-project",
        workspace="/workspace",
        autonomy_preset=autonomy_preset,
        session_id="default",
        user_credential_store=user_credential_store,
    )


# ---------------------------------------------------------------------------
# 1-3: Navigate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_navigate_valid_url():
    """Navigate to a valid URL returns success with title and URL."""
    page = _make_mock_page(url="https://example.com", title="Example Domain")
    bm = _make_browser_manager(page)
    tool = _make_tool(bm=bm)

    with patch("agent_os.agent.tools.browser.validate_url_pre_navigation", return_value=None), \
         patch("agent_os.agent.tools.browser.validate_url_post_navigation", return_value=None):
        result = await tool.execute_async(action="navigate", url="https://example.com")

    assert "Navigated to" in result.content
    assert "Example Domain" in result.content
    page.goto.assert_awaited_once()
    assert result.meta["url"] == "https://example.com"
    assert result.meta["title"] == "Example Domain"


@pytest.mark.asyncio
async def test_navigate_blocked_url():
    """Navigate to a blocked URL (file://) returns error from safety module."""
    tool = _make_tool()

    with patch("agent_os.agent.tools.browser.validate_url_pre_navigation",
               return_value="Cannot navigate to file:///etc/passwd: only http/https URLs are allowed."):
        result = await tool.execute_async(action="navigate", url="file:///etc/passwd")

    assert "Cannot navigate" in result.content
    assert "file:///etc/passwd" in result.content


@pytest.mark.asyncio
async def test_navigate_post_redirect_block():
    """Navigate succeeds but final URL is private IP — blocked post-navigation."""
    page = _make_mock_page(url="http://192.168.1.1/admin", title="Router")
    bm = _make_browser_manager(page)
    tool = _make_tool(bm=bm)

    with patch("agent_os.agent.tools.browser.validate_url_pre_navigation", return_value=None), \
         patch("agent_os.agent.tools.browser.validate_url_post_navigation",
               return_value="Navigation was redirected to a blocked address (http://192.168.1.1/admin)."):
        result = await tool.execute_async(action="navigate", url="https://redirect.example.com")

    assert "redirected to a blocked address" in result.content
    page.goto.assert_any_await("about:blank")


# ---------------------------------------------------------------------------
# 4-6: Click
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_click_with_valid_ref():
    """Click with a valid ref resolves locator and clicks."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)
    ref_map = _make_ref_map()
    bm.get_ref_map.return_value = ref_map
    tool = _make_tool(bm=bm)

    mock_locator = _make_mock_locator()
    with patch("agent_os.agent.tools.browser.resolve_ref", new_callable=AsyncMock, return_value=mock_locator):
        result = await tool.execute_async(action="click", ref="e5")

    assert "Clicked element ref=e5" in result.content
    mock_locator.click.assert_awaited_once()


@pytest.mark.asyncio
async def test_click_no_snapshot():
    """Click without prior snapshot returns helpful message."""
    bm = _make_browser_manager()
    bm.get_ref_map.return_value = None
    tool = _make_tool(bm=bm)

    result = await tool.execute_async(action="click", ref="e5")

    assert "No snapshot taken yet" in result.content
    assert "Run snapshot first" in result.content


@pytest.mark.asyncio
async def test_click_stale_ref():
    """Click with a ref not in the map returns stale ref error."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)
    ref_map = _make_ref_map()
    bm.get_ref_map.return_value = ref_map
    tool = _make_tool(bm=bm)

    with patch("agent_os.agent.tools.browser.resolve_ref", new_callable=AsyncMock,
               side_effect=ValueError("Ref e99 is no longer valid")):
        result = await tool.execute_async(action="click", ref="e99")

    assert "e99" in result.content


# ---------------------------------------------------------------------------
# 7-8: Type
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_type_with_secret_masking():
    """Type with <secret:gmail.password> pattern substitutes real value, result shows masked."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)
    bm.get_ref_map.return_value = _make_ref_map()
    mock_store = MagicMock()
    mock_store.get_value.return_value = "real_password_123"
    tool = _make_tool(bm=bm, user_credential_store=mock_store)

    mock_locator = _make_mock_locator()
    with patch("agent_os.agent.tools.browser.resolve_ref", new_callable=AsyncMock, return_value=mock_locator):
        result = await tool.execute_async(action="type", ref="e5", text="<secret:gmail.password>")

    # The locator should have been filled with the real value
    mock_locator.fill.assert_awaited_once_with("real_password_123")
    mock_store.get_value.assert_called_once_with("gmail", "password")
    # The result should show the masked version
    assert "<secret:gmail.password>" in result.content
    assert "real_password_123" not in result.content


@pytest.mark.asyncio
async def test_type_without_secrets():
    """Type with plain text passes through directly."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)
    bm.get_ref_map.return_value = _make_ref_map()
    tool = _make_tool(bm=bm)

    mock_locator = _make_mock_locator()
    with patch("agent_os.agent.tools.browser.resolve_ref", new_callable=AsyncMock, return_value=mock_locator):
        result = await tool.execute_async(action="type", ref="e5", text="hello@email.com")

    mock_locator.fill.assert_awaited_once_with("hello@email.com")
    assert "hello@email.com" in result.content


# ---------------------------------------------------------------------------
# 9-10: Fill
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fill_batch():
    """Fill with 3 fields fills all successfully."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)
    bm.get_ref_map.return_value = _make_ref_map()
    tool = _make_tool(bm=bm)

    mock_locator = _make_mock_locator()
    with patch("agent_os.agent.tools.browser.resolve_ref", new_callable=AsyncMock, return_value=mock_locator):
        result = await tool.execute_async(
            action="fill",
            fields=[
                {"ref": "e1", "value": "John"},
                {"ref": "e2", "value": "Doe"},
                {"ref": "e5", "value": "john@example.com"},
            ],
        )

    assert "Filled 3 fields" in result.content
    assert "e1: filled" in result.content
    assert "e2: filled" in result.content
    assert "e5: filled" in result.content
    assert mock_locator.fill.await_count == 3


@pytest.mark.asyncio
async def test_fill_partial_failure():
    """Fill where 2 succeed and 1 fails shows error only on failed field."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)
    bm.get_ref_map.return_value = _make_ref_map()
    tool = _make_tool(bm=bm)

    call_count = 0

    async def resolve_side_effect(ref_map, ref, pg):
        nonlocal call_count
        call_count += 1
        if ref == "e2":
            raise ValueError("Ref e2 is no longer valid")
        return _make_mock_locator()

    with patch("agent_os.agent.tools.browser.resolve_ref", side_effect=resolve_side_effect):
        result = await tool.execute_async(
            action="fill",
            fields=[
                {"ref": "e1", "value": "John"},
                {"ref": "e2", "value": "Doe"},
                {"ref": "e5", "value": "john@example.com"},
            ],
        )

    assert "Filled 3 fields" in result.content
    assert "e1: filled" in result.content
    assert "e2: ERROR" in result.content
    assert "e5: filled" in result.content


# ---------------------------------------------------------------------------
# 11-12: Snapshot
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_snapshot_stores_refmap():
    """Snapshot calls _get_ax_tree and stores the RefMap."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)
    tool = _make_tool(bm=bm)

    mock_tree = {
        "role": "WebArea",
        "name": "Example",
        "children": [
            {"role": "heading", "name": "Example", "level": 1},
            {"role": "link", "name": "Click me"},
        ],
    }
    with patch("agent_os.agent.tools.browser._get_ax_tree", new_callable=AsyncMock, return_value=mock_tree):
        result = await tool.execute_async(action="snapshot")

    # RefMap should be stored
    bm.store_ref_map.assert_called_once()
    call_args = bm.store_ref_map.call_args
    assert call_args[0][0] == "test-project"  # project_id
    assert call_args[0][1] == id(page)  # page_id
    assert isinstance(call_args[0][2], dict)  # ref_map dict
    # Result should contain untrusted content markers
    assert "BROWSER CONTENT" in result.content or "(empty page)" in result.content


@pytest.mark.asyncio
async def test_snapshot_interactive_only():
    """Snapshot with interactive_only=True passes the flag through."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)
    tool = _make_tool(bm=bm)

    mock_tree = {
        "role": "WebArea",
        "name": "Test",
        "children": [
            {"role": "button", "name": "Submit"},
            {"role": "heading", "name": "Title", "level": 1},
        ],
    }

    with patch("agent_os.agent.tools.browser._get_ax_tree", new_callable=AsyncMock, return_value=mock_tree), \
         patch("agent_os.agent.tools.browser.serialize_snapshot") as mock_serialize:
        mock_serialize.return_value = (
            "[ref=e1] button \"Submit\"",
            {"e1": RefEntry(role="button", name="Submit", nth=0)},
            MagicMock(lines=1, chars=25, estimated_tokens=7, refs=1, interactive_refs=1),
        )
        result = await tool.execute_async(action="snapshot", interactive_only=True)

    mock_serialize.assert_called_once_with(mock_tree, interactive_only=True)


# ---------------------------------------------------------------------------
# 13-17: Wait (5 modes)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wait_seconds():
    """Wait with seconds parameter calls asyncio.sleep."""
    tool = _make_tool()

    with patch("agent_os.agent.tools.browser.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await tool.execute_async(action="wait", seconds=2)

    mock_sleep.assert_awaited_once_with(2)
    assert "Waited 2 seconds" in result.content


@pytest.mark.asyncio
async def test_wait_text():
    """Wait with text parameter calls page.wait_for_selector with text=."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)
    tool = _make_tool(bm=bm)

    result = await tool.execute_async(action="wait", text="Loading")

    page.wait_for_selector.assert_awaited_once_with("text=Loading", timeout=30_000)
    assert "Loading" in result.content
    assert "appeared" in result.content


@pytest.mark.asyncio
async def test_wait_text_gone():
    """Wait with text_gone parameter calls wait_for_selector with state=hidden."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)
    tool = _make_tool(bm=bm)

    result = await tool.execute_async(action="wait", text_gone="Loading")

    page.wait_for_selector.assert_awaited_once_with("text=Loading", state="hidden", timeout=30_000)
    assert "Loading" in result.content
    assert "disappeared" in result.content


@pytest.mark.asyncio
async def test_wait_selector():
    """Wait with selector parameter calls page.wait_for_selector."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)
    tool = _make_tool(bm=bm)

    result = await tool.execute_async(action="wait", selector=".results")

    page.wait_for_selector.assert_awaited_once_with(".results", timeout=30_000)
    assert ".results" in result.content


@pytest.mark.asyncio
async def test_wait_load_state():
    """Wait with load_state parameter calls page.wait_for_load_state."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)
    tool = _make_tool(bm=bm)

    result = await tool.execute_async(action="wait", load_state="networkidle")

    page.wait_for_load_state.assert_awaited_once_with("networkidle", timeout=30_000)
    assert "networkidle" in result.content


# ---------------------------------------------------------------------------
# 18: PDF
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pdf_generates_file():
    """PDF action calls page.pdf() with a path."""
    page = _make_mock_page(title="Test Page")
    bm = _make_browser_manager(page)
    tool = _make_tool(bm=bm)

    with patch("os.makedirs"):
        result = await tool.execute_async(action="pdf")

    page.pdf.assert_awaited_once()
    call_kwargs = page.pdf.call_args[1]
    assert call_kwargs["path"].endswith(".pdf")
    assert "PDF saved" in result.content


# ---------------------------------------------------------------------------
# 19-21: Error translation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_error_translation_timeout():
    """TimeoutError is translated to human-friendly message."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)
    bm.get_ref_map.return_value = _make_ref_map()
    tool = _make_tool(bm=bm)

    with patch("agent_os.agent.tools.browser.resolve_ref", new_callable=AsyncMock,
               side_effect=Exception("Timeout 30000ms exceeded waiting for locator")):
        result = await tool.execute_async(action="click", ref="e5")

    assert "did not become actionable" in result.content


@pytest.mark.asyncio
async def test_error_translation_strict():
    """Strict mode violation is translated to 'Multiple elements matched'."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)
    bm.get_ref_map.return_value = _make_ref_map()
    tool = _make_tool(bm=bm)

    mock_locator = _make_mock_locator()
    mock_locator.click = AsyncMock(side_effect=Exception("strict mode violation: locator resolved to 3 elements"))

    with patch("agent_os.agent.tools.browser.resolve_ref", new_callable=AsyncMock, return_value=mock_locator):
        result = await tool.execute_async(action="click", ref="e5")

    assert "Multiple elements matched" in result.content


@pytest.mark.asyncio
async def test_error_translation_pointer():
    """Pointer interception error is translated to 'blocked by overlay'."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)
    bm.get_ref_map.return_value = _make_ref_map()
    tool = _make_tool(bm=bm)

    mock_locator = _make_mock_locator()
    mock_locator.click = AsyncMock(
        side_effect=Exception("element click intercepted: pointer event")
    )

    with patch("agent_os.agent.tools.browser.resolve_ref", new_callable=AsyncMock, return_value=mock_locator):
        result = await tool.execute_async(action="click", ref="e5")

    assert "blocked by overlay" in result.content


# ---------------------------------------------------------------------------
# 22-24: Batch execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_batch_execution_all_observation():
    """Batch of observation-only actions all execute in any mode."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)
    tool = _make_tool(bm=bm, autonomy_preset="check_in")

    with patch("agent_os.agent.tools.browser.validate_url_pre_navigation", return_value=None), \
         patch("agent_os.agent.tools.browser.validate_url_post_navigation", return_value=None):
        result = await tool.execute_async(
            action="batch",
            actions=[
                {"action": "navigate", "url": "https://example.com"},
                {"action": "snapshot"},
                {"action": "screenshot"},
            ],
        )

    assert "Batch execution:" in result.content
    assert "[navigate]" in result.content
    assert "[snapshot]" in result.content
    assert "[screenshot]" in result.content
    assert "PAUSED" not in result.content


@pytest.mark.asyncio
async def test_batch_pauses_at_write_in_checkin():
    """Batch with write action pauses in CHECK_IN mode."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)
    bm.get_ref_map.return_value = _make_ref_map()
    tool = _make_tool(bm=bm, autonomy_preset="check_in")

    with patch("agent_os.agent.tools.browser.validate_url_pre_navigation", return_value=None), \
         patch("agent_os.agent.tools.browser.validate_url_post_navigation", return_value=None):
        result = await tool.execute_async(
            action="batch",
            actions=[
                {"action": "navigate", "url": "https://example.com"},
                {"action": "click", "ref": "e5"},
                {"action": "snapshot"},
            ],
        )

    assert "PAUSED" in result.content
    assert "click" in result.content
    assert "2 actions pending" in result.content
    assert result.meta is not None
    assert result.meta["paused_at_action"] == "click"


@pytest.mark.asyncio
async def test_batch_all_execute_in_handsoff():
    """Batch with write actions all execute in HANDS_OFF mode."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)
    bm.get_ref_map.return_value = _make_ref_map()
    tool = _make_tool(bm=bm, autonomy_preset="hands_off")

    mock_locator = _make_mock_locator()
    with patch("agent_os.agent.tools.browser.resolve_ref", new_callable=AsyncMock, return_value=mock_locator), \
         patch("agent_os.agent.tools.browser.validate_url_pre_navigation", return_value=None), \
         patch("agent_os.agent.tools.browser.validate_url_post_navigation", return_value=None):
        result = await tool.execute_async(
            action="batch",
            actions=[
                {"action": "navigate", "url": "https://example.com"},
                {"action": "click", "ref": "e5"},
                {"action": "snapshot"},
            ],
        )

    assert "Batch execution:" in result.content
    assert "[navigate]" in result.content
    assert "[click]" in result.content
    assert "[snapshot]" in result.content
    assert "PAUSED" not in result.content


# ---------------------------------------------------------------------------
# 25: Done
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_done_returns_text():
    """Done action returns the provided text as-is."""
    tool = _make_tool()

    result = await tool.execute_async(action="done", text="Task complete")

    assert result.content == "Task complete"


# ---------------------------------------------------------------------------
# Extra: sync execute raises
# ---------------------------------------------------------------------------

def test_execute_is_async():
    """execute() is a coroutine function (async), matching the registry contract."""
    tool = _make_tool()
    import asyncio
    assert asyncio.iscoroutinefunction(tool.execute)


# ---------------------------------------------------------------------------
# Path validation tests: _resolve_upload_path
# ---------------------------------------------------------------------------

def test_resolve_upload_path_relative_inside(tmp_path):
    """Relative path 'report.html' resolves to workspace/report.html."""
    tool = _make_tool()
    tool._workspace = str(tmp_path)
    result = tool._resolve_upload_path("report.html")
    expected = os.path.realpath(os.path.join(str(tmp_path), "report.html"))
    assert result == expected


def test_resolve_upload_path_nested(tmp_path):
    """Nested relative 'output/reports/data.csv' resolves correctly."""
    tool = _make_tool()
    tool._workspace = str(tmp_path)
    result = tool._resolve_upload_path("output/reports/data.csv")
    expected = os.path.realpath(os.path.join(str(tmp_path), "output", "reports", "data.csv"))
    assert result == expected


def test_resolve_upload_path_absolute_inside(tmp_path):
    """Absolute path inside workspace is allowed."""
    tool = _make_tool()
    tool._workspace = str(tmp_path)
    inner = tmp_path / "subdir" / "file.txt"
    inner.parent.mkdir(parents=True, exist_ok=True)
    inner.write_text("test")
    result = tool._resolve_upload_path(str(inner))
    assert result == os.path.realpath(str(inner))


def test_resolve_upload_path_traversal_blocked(tmp_path):
    """'../../etc/passwd' and '../secret.txt' return None (traversal blocked)."""
    tool = _make_tool()
    tool._workspace = str(tmp_path)
    assert tool._resolve_upload_path("../../etc/passwd") is None
    assert tool._resolve_upload_path("../secret.txt") is None


def test_resolve_upload_path_absolute_outside_blocked(tmp_path):
    """Absolute path outside workspace returns None."""
    tool = _make_tool()
    tool._workspace = str(tmp_path)
    # Use the parent of tmp_path which is outside workspace
    outside = str(tmp_path.parent / "outside_file.txt")
    assert tool._resolve_upload_path(outside) is None


# ---------------------------------------------------------------------------
# upload_file action tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_file_rejects_outside_workspace(tmp_path):
    """upload_file with path outside workspace returns error containing 'outside workspace'."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)
    tool = _make_tool(bm=bm)
    tool._workspace = str(tmp_path)

    result = await tool.execute_async(action="upload_file", file_path="../../etc/passwd")

    assert "outside workspace" in result.content.lower()


@pytest.mark.asyncio
async def test_upload_file_rejects_nonexistent(tmp_path):
    """upload_file with path inside workspace but file doesn't exist returns 'not found'."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)
    tool = _make_tool(bm=bm)
    tool._workspace = str(tmp_path)

    result = await tool.execute_async(action="upload_file", file_path="does_not_exist.txt")

    assert "not found" in result.content.lower()


@pytest.mark.asyncio
async def test_upload_file_uses_pending_filechooser(tmp_path):
    """When BrowserManager has a pending filechooser, upload_file uses chooser.set_files()."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)

    # Create a real file to upload
    upload_file = tmp_path / "report.html"
    upload_file.write_text("<html></html>")

    tool = _make_tool(bm=bm)
    tool._workspace = str(tmp_path)

    # Set up a mock filechooser
    mock_chooser = MagicMock()
    mock_chooser.set_files = AsyncMock()
    bm.consume_file_chooser = MagicMock(return_value=mock_chooser)

    result = await tool.execute_async(action="upload_file", file_path="report.html")

    mock_chooser.set_files.assert_awaited_once()
    assert "file chooser" in result.content.lower()
    assert "report.html" in result.content


@pytest.mark.asyncio
async def test_upload_file_falls_back_to_locator(tmp_path):
    """When no pending filechooser but ref is provided, falls back to locator.set_input_files()."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)
    ref_map = _make_ref_map()
    bm.get_ref_map.return_value = ref_map

    # Create a real file
    upload_file = tmp_path / "data.csv"
    upload_file.write_text("a,b,c")

    tool = _make_tool(bm=bm)
    tool._workspace = str(tmp_path)

    # No pending filechooser
    bm.consume_file_chooser = MagicMock(return_value=None)

    mock_locator = _make_mock_locator()
    with patch("agent_os.agent.tools.browser.resolve_ref", new_callable=AsyncMock, return_value=mock_locator):
        result = await tool.execute_async(action="upload_file", file_path="data.csv", ref="e5")

    mock_locator.set_input_files.assert_awaited_once()
    assert "ref=e5" in result.content


@pytest.mark.asyncio
async def test_upload_file_no_chooser_no_ref_error(tmp_path):
    """When no pending filechooser and no ref, returns error about clicking upload button first."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)

    # Create a real file
    upload_file = tmp_path / "data.csv"
    upload_file.write_text("a,b,c")

    tool = _make_tool(bm=bm)
    tool._workspace = str(tmp_path)

    # No pending filechooser
    bm.consume_file_chooser = MagicMock(return_value=None)

    result = await tool.execute_async(action="upload_file", file_path="data.csv")

    assert "click the upload button first" in result.content.lower()


@pytest.mark.asyncio
async def test_upload_file_chooser_consumed(tmp_path):
    """After successful upload via filechooser, calling upload_file again without clicking returns error."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)

    # Create a real file
    upload_file = tmp_path / "file.txt"
    upload_file.write_text("content")

    tool = _make_tool(bm=bm)
    tool._workspace = str(tmp_path)

    # First call: filechooser available
    mock_chooser = MagicMock()
    mock_chooser.set_files = AsyncMock()
    consume_results = [mock_chooser, None]  # first call returns chooser, second returns None
    bm.consume_file_chooser = MagicMock(side_effect=consume_results)

    result1 = await tool.execute_async(action="upload_file", file_path="file.txt")
    assert "file chooser" in result1.content.lower()

    # Second call: no filechooser, no ref -> error
    result2 = await tool.execute_async(action="upload_file", file_path="file.txt")
    assert "click the upload button first" in result2.content.lower()


# ---------------------------------------------------------------------------
# Post-action stabilization (networkidle waits)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_navigate_waits_for_networkidle():
    """Navigate waits for networkidle after goto to let JS content render."""
    page = _make_mock_page(url="https://example.com", title="Example")
    bm = _make_browser_manager(page)
    tool = _make_tool(bm=bm)

    with patch("agent_os.agent.tools.browser.validate_url_pre_navigation", return_value=None), \
         patch("agent_os.agent.tools.browser.validate_url_post_navigation", return_value=None):
        await tool.execute_async(action="navigate", url="https://example.com")

    page.wait_for_load_state.assert_awaited_with("networkidle", timeout=10000)


@pytest.mark.asyncio
async def test_click_waits_for_networkidle():
    """Click waits for networkidle after interaction to capture post-action state."""
    page = _make_mock_page()
    bm = _make_browser_manager(page)
    bm.get_ref_map.return_value = _make_ref_map()
    tool = _make_tool(bm=bm)

    mock_locator = _make_mock_locator()
    with patch("agent_os.agent.tools.browser.resolve_ref", new_callable=AsyncMock, return_value=mock_locator):
        await tool.execute_async(action="click", ref="e2")

    page.wait_for_load_state.assert_awaited_with("networkidle", timeout=5000)


@pytest.mark.asyncio
async def test_networkidle_timeout_does_not_fail_action():
    """If networkidle times out, the action still succeeds with partial page."""
    page = _make_mock_page(url="https://example.com", title="Example")
    page.wait_for_load_state = AsyncMock(side_effect=TimeoutError("networkidle timeout"))
    bm = _make_browser_manager(page)
    tool = _make_tool(bm=bm)

    with patch("agent_os.agent.tools.browser.validate_url_pre_navigation", return_value=None), \
         patch("agent_os.agent.tools.browser.validate_url_post_navigation", return_value=None):
        result = await tool.execute_async(action="navigate", url="https://example.com")

    assert "Navigated to" in result.content


# ---------------------------------------------------------------------------
# Search action tests
# ---------------------------------------------------------------------------

def test_search_is_observation_action():
    """search should be classified as observation (no approval needed)."""
    assert "search" in BROWSER_OBSERVATION_ACTIONS
    assert "search" not in BROWSER_WRITE_ACTIONS


def test_fetch_is_observation_action():
    """fetch should be classified as observation (no approval needed)."""
    assert "fetch" in BROWSER_OBSERVATION_ACTIONS
    assert "fetch" not in BROWSER_WRITE_ACTIONS


@pytest.mark.asyncio
async def test_search_requires_query():
    """search action without query returns error."""
    tool = _make_tool()
    result = await tool.execute_async(action="search")
    assert "error" in result.content.lower()
    assert "query" in result.content.lower()


@pytest.mark.asyncio
async def test_search_requires_nonempty_query():
    """search action with empty/whitespace query returns error."""
    tool = _make_tool()
    result = await tool.execute_async(action="search", query="   ")
    assert "error" in result.content.lower()
    assert "query" in result.content.lower()


@pytest.mark.asyncio
async def test_fetch_requires_url():
    """fetch action without url returns error."""
    tool = _make_tool()
    result = await tool.execute_async(action="fetch")
    assert "error" in result.content.lower()
    assert "url" in result.content.lower()


@pytest.mark.asyncio
async def test_fetch_blocks_unsafe_url():
    """fetch action rejects URLs that fail safety validation."""
    tool = _make_tool()
    with patch("agent_os.agent.tools.browser.validate_url_pre_navigation",
               return_value="Blocked: file:// scheme not allowed"):
        result = await tool.execute_async(action="fetch", url="file:///etc/passwd")
    assert "error" in result.content.lower()
    assert "blocked" in result.content.lower()


@pytest.mark.asyncio
async def test_search_returns_formatted_results():
    """search action returns formatted results from Google."""
    page = _make_mock_page()
    search_page = _make_mock_page(url="https://www.google.com/search?q=test", title="test - Google Search")
    search_page.evaluate = AsyncMock(return_value=[
        {"title": "Result 1", "url": "https://example.com/1", "snippet": "First result snippet"},
        {"title": "Result 2", "url": "https://example.com/2", "snippet": "Second result snippet"},
    ])

    bm = _make_browser_manager(page)
    page.context = MagicMock()
    page.context.new_page = AsyncMock(return_value=search_page)

    tool = _make_tool(bm=bm)
    result = await tool.execute_async(action="search", query="test query")

    assert "search results for: test query" in result.content.lower()
    assert "Result 1" in result.content
    assert "Result 2" in result.content
    assert result.meta["result_count"] == 2
    search_page.close.assert_awaited_once()
    page.bring_to_front.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_handles_empty_results():
    """search action returns 'no results' when extraction returns empty."""
    page = _make_mock_page()
    search_page = _make_mock_page()
    search_page.evaluate = AsyncMock(return_value=[])

    bm = _make_browser_manager(page)
    page.context = MagicMock()
    page.context.new_page = AsyncMock(return_value=search_page)

    tool = _make_tool(bm=bm)
    result = await tool.execute_async(action="search", query="nonexistent thing")

    assert "no results" in result.content.lower()
    search_page.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_cleans_up_on_error():
    """search action closes search tab even when an error occurs."""
    page = _make_mock_page()
    search_page = _make_mock_page()
    search_page.goto = AsyncMock(side_effect=Exception("navigation failed"))
    search_page.is_closed = MagicMock(return_value=False)

    bm = _make_browser_manager(page)
    page.context = MagicMock()
    page.context.new_page = AsyncMock(return_value=search_page)

    tool = _make_tool(bm=bm)
    result = await tool.execute_async(action="search", query="test")

    assert "error" in result.content.lower()
    search_page.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_extracts_text():
    """fetch action returns extracted text content."""
    page = _make_mock_page()
    fetch_page = _make_mock_page()
    fetch_page.evaluate = AsyncMock(return_value="This is the main article content.")

    bm = _make_browser_manager(page)
    page.context = MagicMock()
    page.context.new_page = AsyncMock(return_value=fetch_page)

    tool = _make_tool(bm=bm)
    with patch("agent_os.agent.tools.browser.validate_url_pre_navigation", return_value=None):
        result = await tool.execute_async(action="fetch", url="https://example.com/article")

    assert "main article content" in result.content
    assert result.meta["url"] == "https://example.com/article"
    fetch_page.close.assert_awaited_once()
    page.bring_to_front.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_truncates_long_content():
    """fetch action truncates content exceeding 50K characters."""
    page = _make_mock_page()
    fetch_page = _make_mock_page()
    long_text = "x" * 60000
    fetch_page.evaluate = AsyncMock(return_value=long_text)

    bm = _make_browser_manager(page)
    page.context = MagicMock()
    page.context.new_page = AsyncMock(return_value=fetch_page)

    tool = _make_tool(bm=bm)
    with patch("agent_os.agent.tools.browser.validate_url_pre_navigation", return_value=None):
        result = await tool.execute_async(action="fetch", url="https://example.com/huge")

    assert len(result.content) <= 51000  # 50K + truncation notice
    assert "TRUNCATED" in result.content


@pytest.mark.asyncio
async def test_fetch_handles_empty_content():
    """fetch action returns error when page has no readable content."""
    page = _make_mock_page()
    fetch_page = _make_mock_page()
    fetch_page.evaluate = AsyncMock(return_value="")

    bm = _make_browser_manager(page)
    page.context = MagicMock()
    page.context.new_page = AsyncMock(return_value=fetch_page)

    tool = _make_tool(bm=bm)
    with patch("agent_os.agent.tools.browser.validate_url_pre_navigation", return_value=None):
        result = await tool.execute_async(action="fetch", url="https://example.com/empty")

    assert "no readable content" in result.content.lower()


@pytest.mark.asyncio
async def test_fetch_cleans_up_on_error():
    """fetch action closes fetch tab even when an error occurs."""
    page = _make_mock_page()
    fetch_page = _make_mock_page()
    fetch_page.goto = AsyncMock(side_effect=Exception("connection refused"))
    fetch_page.is_closed = MagicMock(return_value=False)

    bm = _make_browser_manager(page)
    page.context = MagicMock()
    page.context.new_page = AsyncMock(return_value=fetch_page)

    tool = _make_tool(bm=bm)
    with patch("agent_os.agent.tools.browser.validate_url_pre_navigation", return_value=None):
        result = await tool.execute_async(action="fetch", url="https://example.com")

    assert "error" in result.content.lower()
    fetch_page.close.assert_awaited_once()
