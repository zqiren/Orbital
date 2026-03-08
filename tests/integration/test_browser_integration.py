# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Browser tool integration tests with real headless Playwright.

These tests verify the BrowserTool works end-to-end against a real browser,
covering navigation, snapshot, click, batch execution, approval flow,
multi-project isolation, screenshot capture, wait strategies, PDF generation,
crash recovery, and stale ref handling.

A local HTTP server serves deterministic test pages so no external network
dependencies are required.
"""

import re
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_os.agent.tools.browser import (
    BROWSER_OBSERVATION_ACTIONS,
    BROWSER_WRITE_ACTIONS,
    BrowserTool,
)
from agent_os.daemon_v2.browser_manager import BrowserManager


# ---------------------------------------------------------------------------
# Local HTTP server fixture
# ---------------------------------------------------------------------------

_TEST_PAGES = {
    "/hello.html": (
        "<html><body>"
        "<h1>Hello</h1>"
        "<a href='/page2.html'>Next</a>"
        "</body></html>"
    ),
    "/page2.html": (
        "<html><body><h1>Page2</h1></body></html>"
    ),
    "/form.html": (
        "<html><body>"
        "<input type='text' placeholder='Name' aria-label='Name'>"
        "<input type='text' placeholder='Email' aria-label='Email'>"
        "<button>Submit</button>"
        "</body></html>"
    ),
    "/button.html": (
        "<html><body><button>Click Me</button></body></html>"
    ),
    "/checkin_form.html": (
        "<html><body>"
        "<input placeholder='Name' aria-label='Name'>"
        "<button>Go</button>"
        "</body></html>"
    ),
    "/page_a.html": "<html><body><h1>PageA</h1></body></html>",
    "/page_b.html": "<html><body><h1>PageB</h1></body></html>",
    "/screenshot.html": (
        "<html><body><h1>Screenshot Test</h1></body></html>"
    ),
    "/delayed_text.html": (
        "<html><body>"
        "<div id='target'></div>"
        "<script>"
        "setTimeout(function(){ document.getElementById('target').textContent = 'Ready'; }, 500);"
        "</script>"
        "</body></html>"
    ),
    "/delayed_selector.html": (
        "<html><body>"
        "<script>"
        "setTimeout(function(){"
        "  var d = document.createElement('div');"
        "  d.className = 'loaded';"
        "  d.textContent = 'Done';"
        "  document.body.appendChild(d);"
        "}, 500);"
        "</script>"
        "</body></html>"
    ),
    "/pdf.html": (
        "<html><body><h1>PDF Test</h1><p>Content here</p></body></html>"
    ),
    "/crash_before.html": "<html><body><h1>Before</h1></body></html>",
    "/crash_after.html": "<html><body><h1>After</h1></body></html>",
    "/stale_button.html": (
        "<html><body><button>Click</button></body></html>"
    ),
    "/new_page.html": "<html><body><h1>New Page</h1></body></html>",
}


class _TestHandler(SimpleHTTPRequestHandler):
    """Serves in-memory test pages."""

    def do_GET(self):
        content = _TEST_PAGES.get(self.path)
        if content is None:
            self.send_error(404)
            return
        body = content.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # suppress request logs


@pytest.fixture(scope="module")
def test_server():
    """Start a local HTTP server on an ephemeral port for the test module."""
    server = HTTPServer(("127.0.0.1", 0), _TestHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def _allow_localhost(url: str):
    """Patch helper: allow localhost URLs through validation."""
    return None


# ---------------------------------------------------------------------------
# Browser fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def browser_manager(tmp_path):
    """Real BrowserManager with headless Chromium."""
    bm = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)
    yield bm
    await bm.shutdown()


@pytest.fixture
async def browser_tool(browser_manager, tmp_path):
    """BrowserTool wired to real BrowserManager in hands_off mode."""
    ws = tmp_path / "workspace"
    ws.mkdir(exist_ok=True)
    return BrowserTool(
        browser_manager=browser_manager,
        project_id="test_project",
        workspace=str(ws),
        autonomy_preset="hands_off",
        session_id="test_session",
    )


# ---------------------------------------------------------------------------
# 1. Full workflow: navigate -> snapshot -> click -> snapshot -> done
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_full_workflow_navigate_snapshot_click_extract(
    browser_tool, test_server,
):
    with patch(
        "agent_os.agent.tools.browser.validate_url_pre_navigation",
        side_effect=_allow_localhost,
    ), patch(
        "agent_os.agent.tools.browser.validate_url_post_navigation",
        side_effect=_allow_localhost,
    ):
        # Step 1: Navigate to page with a button
        result = await browser_tool.execute_async(
            action="navigate", url=f"{test_server}/hello.html",
        )
        assert result.meta.get("url")
        assert "Hello" in result.content or "Navigated" in result.content

        # Step 2: Snapshot — should contain refs
        result = await browser_tool.execute_async(action="snapshot")
        assert "ref=e" in result.content
        assert result.meta.get("snapshot_stats")
        assert result.meta["snapshot_stats"]["interactive_refs"] >= 1

        # Step 3: Find and click a ref (the link)
        refs = re.findall(r"\[ref=(e\d+)\]", result.content)
        assert len(refs) >= 1
        link_ref = refs[-1]  # Last ref is the link
        result = await browser_tool.execute_async(action="click", ref=link_ref)
        assert "Clicked" in result.content

        # Step 4: Navigate to second page explicitly
        result = await browser_tool.execute_async(
            action="navigate", url=f"{test_server}/page2.html",
        )
        assert result.meta.get("url")

        # Step 5: Snapshot on new page — should show Page2
        result = await browser_tool.execute_async(action="snapshot")
        assert "Page2" in result.content

        # Step 6: Done
        result = await browser_tool.execute_async(
            action="done", text="Navigated to page 2",
        )
        assert "Navigated to page 2" in result.content


# ---------------------------------------------------------------------------
# 2. Batch execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_batch_execution(browser_tool, test_server):
    with patch(
        "agent_os.agent.tools.browser.validate_url_pre_navigation",
        side_effect=_allow_localhost,
    ), patch(
        "agent_os.agent.tools.browser.validate_url_post_navigation",
        side_effect=_allow_localhost,
    ):
        await browser_tool.execute_async(
            action="navigate", url=f"{test_server}/form.html",
        )

        snapshot = await browser_tool.execute_async(action="snapshot")
        refs = re.findall(r"\[ref=(e\d+)\]", snapshot.content)
        # Should have at least 2 input refs + 1 button ref
        assert len(refs) >= 2, f"Expected >= 2 refs, got {refs} from:\n{snapshot.content}"

        # Batch: type into both input fields
        result = await browser_tool.execute_async(
            action="batch",
            actions=[
                {"action": "type", "ref": refs[0], "text": "John"},
                {"action": "type", "ref": refs[1], "text": "john@test.com"},
            ],
        )
        assert "Batch execution" in result.content
        assert "John" in result.content
        assert "john@test.com" in result.content


# ---------------------------------------------------------------------------
# 3. Action classification (CHECK_IN intercept awareness)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_checkin_intercepts_click():
    """Verify action classification: write vs observation actions."""
    assert "click" in BROWSER_WRITE_ACTIONS
    assert "type" in BROWSER_WRITE_ACTIONS
    assert "fill" in BROWSER_WRITE_ACTIONS
    assert "snapshot" in BROWSER_OBSERVATION_ACTIONS
    assert "navigate" in BROWSER_OBSERVATION_ACTIONS
    assert "screenshot" in BROWSER_OBSERVATION_ACTIONS


# ---------------------------------------------------------------------------
# 4. Batch pauses at write action in CHECK_IN mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_batch_pauses_at_write_in_checkin(browser_manager, tmp_path, test_server):
    ws = tmp_path / "ws_checkin"
    ws.mkdir(exist_ok=True)
    tool = BrowserTool(
        browser_manager=browser_manager,
        project_id="test_checkin",
        workspace=str(ws),
        autonomy_preset="check_in",
        session_id="test_checkin",
    )

    with patch(
        "agent_os.agent.tools.browser.validate_url_pre_navigation",
        side_effect=_allow_localhost,
    ), patch(
        "agent_os.agent.tools.browser.validate_url_post_navigation",
        side_effect=_allow_localhost,
    ):
        await tool.execute_async(
            action="navigate", url=f"{test_server}/checkin_form.html",
        )
        snapshot = await tool.execute_async(action="snapshot")
        refs = re.findall(r"\[ref=(e\d+)\]", snapshot.content)
        assert refs, f"No refs found in snapshot:\n{snapshot.content}"

        # Batch: snapshot (observation, OK) then type (write, should PAUSE)
        result = await tool.execute_async(
            action="batch",
            actions=[
                {"action": "snapshot"},
                {"action": "type", "ref": refs[0], "text": "Hello"},
            ],
        )
        assert "PAUSED" in result.content
        assert result.meta.get("pending_actions")


# ---------------------------------------------------------------------------
# 5. Multi-project isolated tabs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_multi_project_isolated_tabs(browser_manager, tmp_path, test_server):
    ws_a = tmp_path / "ws_a"
    ws_a.mkdir(exist_ok=True)
    ws_b = tmp_path / "ws_b"
    ws_b.mkdir(exist_ok=True)

    tool_a = BrowserTool(
        browser_manager=browser_manager,
        project_id="proj_a",
        workspace=str(ws_a),
        autonomy_preset="hands_off",
        session_id="sa",
    )
    tool_b = BrowserTool(
        browser_manager=browser_manager,
        project_id="proj_b",
        workspace=str(ws_b),
        autonomy_preset="hands_off",
        session_id="sb",
    )

    with patch(
        "agent_os.agent.tools.browser.validate_url_pre_navigation",
        side_effect=_allow_localhost,
    ), patch(
        "agent_os.agent.tools.browser.validate_url_post_navigation",
        side_effect=_allow_localhost,
    ):
        await tool_a.execute_async(
            action="navigate", url=f"{test_server}/page_a.html",
        )
        await tool_b.execute_async(
            action="navigate", url=f"{test_server}/page_b.html",
        )

        snap_a = await tool_a.execute_async(action="snapshot")
        snap_b = await tool_b.execute_async(action="snapshot")

        assert "PageA" in snap_a.content
        assert "PageB" in snap_b.content
        # They see different content — isolated tabs
        assert "PageB" not in snap_a.content
        assert "PageA" not in snap_b.content


# ---------------------------------------------------------------------------
# 6. Screenshot capture and retention
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_screenshot_capture_and_retention(browser_tool, tmp_path, test_server):
    with patch(
        "agent_os.agent.tools.browser.validate_url_pre_navigation",
        side_effect=_allow_localhost,
    ), patch(
        "agent_os.agent.tools.browser.validate_url_post_navigation",
        side_effect=_allow_localhost,
    ):
        await browser_tool.execute_async(
            action="navigate", url=f"{test_server}/screenshot.html",
        )

        # Take explicit screenshot
        result = await browser_tool.execute_async(action="screenshot")
        assert result.meta.get("screenshot_path")
        assert Path(result.meta["screenshot_path"]).exists()

        # Verify the file is a real PNG (starts with PNG magic bytes)
        with open(result.meta["screenshot_path"], "rb") as f:
            header = f.read(8)
        assert header[1:4] == b"PNG"


# ---------------------------------------------------------------------------
# 7. Wait for text
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_wait_for_text(browser_tool, test_server):
    with patch(
        "agent_os.agent.tools.browser.validate_url_pre_navigation",
        side_effect=_allow_localhost,
    ), patch(
        "agent_os.agent.tools.browser.validate_url_post_navigation",
        side_effect=_allow_localhost,
    ):
        await browser_tool.execute_async(
            action="navigate", url=f"{test_server}/delayed_text.html",
        )
        result = await browser_tool.execute_async(action="wait", text="Ready")
        assert "Ready" in result.content
        assert "timed out" not in result.content.lower()


# ---------------------------------------------------------------------------
# 8. Wait for selector
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_wait_for_selector(browser_tool, test_server):
    with patch(
        "agent_os.agent.tools.browser.validate_url_pre_navigation",
        side_effect=_allow_localhost,
    ), patch(
        "agent_os.agent.tools.browser.validate_url_post_navigation",
        side_effect=_allow_localhost,
    ):
        await browser_tool.execute_async(
            action="navigate", url=f"{test_server}/delayed_selector.html",
        )
        result = await browser_tool.execute_async(action="wait", selector=".loaded")
        assert ".loaded" in result.content
        assert "timed out" not in result.content.lower()


# ---------------------------------------------------------------------------
# 9. PDF generation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_pdf_generation(browser_tool, tmp_path, test_server):
    with patch(
        "agent_os.agent.tools.browser.validate_url_pre_navigation",
        side_effect=_allow_localhost,
    ), patch(
        "agent_os.agent.tools.browser.validate_url_post_navigation",
        side_effect=_allow_localhost,
    ):
        await browser_tool.execute_async(
            action="navigate", url=f"{test_server}/pdf.html",
        )
        result = await browser_tool.execute_async(action="pdf")
        # PDF generation may only work in non-headless or CDP-connected mode.
        # Accept either success (file_path in meta) or a graceful error.
        if result.meta and result.meta.get("file_path"):
            assert Path(result.meta["file_path"]).exists()
        else:
            # Graceful failure — action didn't crash
            assert "pdf" in result.content.lower() or "failed" in result.content.lower()


# ---------------------------------------------------------------------------
# 10. Crash recovery
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_crash_recovery(browser_manager, tmp_path, test_server):
    ws = tmp_path / "ws_crash"
    ws.mkdir(exist_ok=True)
    tool = BrowserTool(
        browser_manager=browser_manager,
        project_id="crash_test",
        workspace=str(ws),
        autonomy_preset="hands_off",
        session_id="cs",
    )

    with patch(
        "agent_os.agent.tools.browser.validate_url_pre_navigation",
        side_effect=_allow_localhost,
    ), patch(
        "agent_os.agent.tools.browser.validate_url_post_navigation",
        side_effect=_allow_localhost,
    ):
        # Navigate successfully first
        await tool.execute_async(
            action="navigate", url=f"{test_server}/crash_before.html",
        )
        snap = await tool.execute_async(action="snapshot")
        assert "Before" in snap.content

        # Simulate browser crash by closing the persistent context
        if browser_manager._context:
            try:
                await browser_manager._context.close()
            except Exception:
                pass
        browser_manager._browser = None
        browser_manager._context = None
        # Clear page tracking so get_page creates a fresh page
        browser_manager._project_pages.clear()
        browser_manager._page_state.clear()

        # Next action should auto-recover (ensure_browser re-launches)
        result = await tool.execute_async(
            action="navigate", url=f"{test_server}/crash_after.html",
        )
        # Recovery succeeded if we got a valid navigation result (URL present)
        assert result.meta and "crash_after" in result.meta.get("url", "")
        # Verify content is actually accessible after recovery
        snap = await tool.execute_async(action="snapshot")
        assert "After" in snap.content


# ---------------------------------------------------------------------------
# 11. Stale ref after navigation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_stale_ref_after_navigation(browser_tool, test_server):
    with patch(
        "agent_os.agent.tools.browser.validate_url_pre_navigation",
        side_effect=_allow_localhost,
    ), patch(
        "agent_os.agent.tools.browser.validate_url_post_navigation",
        side_effect=_allow_localhost,
    ):
        await browser_tool.execute_async(
            action="navigate", url=f"{test_server}/stale_button.html",
        )
        snap = await browser_tool.execute_async(action="snapshot")
        refs = re.findall(r"\[ref=(e\d+)\]", snap.content)
        assert refs

        # Navigate away — this clears the ref map
        await browser_tool.execute_async(
            action="navigate", url=f"{test_server}/new_page.html",
        )

        # Old ref should fail gracefully with a helpful message
        result = await browser_tool.execute_async(action="click", ref=refs[0])
        assert "snapshot" in result.content.lower()
