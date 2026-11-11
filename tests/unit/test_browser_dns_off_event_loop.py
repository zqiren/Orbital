# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Browser URL validation resolves hostnames off the event loop.

Spec 171 A1: 3 of the 6 stall dumps in the owner's ``logs/hang.log`` had the
loop thread inside ``browser_safety.py`` ``socket.getaddrinfo`` (called from
the browser's fetch action). A slow resolver froze chat, the sidebar and every
project. The pre- and post-navigation checks now run in a worker thread; the
answer and the checks are unchanged.
"""

from __future__ import annotations

import socket
import threading
from unittest.mock import patch

import pytest

from tests.unit.test_browser_tool import _make_browser_manager, _make_mock_page, _make_tool


def _recording_getaddrinfo(seen: list[tuple[str, int]], ip: str = "93.184.216.34"):
    def fake(host, port, *args, **kwargs):
        seen.append((host, threading.get_ident()))
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]
    return fake


@pytest.mark.asyncio
async def test_navigate_resolves_both_checks_off_the_event_loop():
    loop_thread = threading.get_ident()
    seen: list[tuple[str, int]] = []
    page = _make_mock_page(url="https://example.org/landing", title="Example")
    tool = _make_tool(bm=_make_browser_manager(page))

    with patch("agent_os.agent.tools.browser_safety.socket.getaddrinfo",
               _recording_getaddrinfo(seen)):
        result = await tool.execute_async(action="navigate", url="https://example.com")

    assert "Navigated to" in result.content
    # Pre-navigation (requested URL) and post-navigation (final URL).
    assert [h for h, _ in seen] == ["example.com", "example.org"]
    assert all(tid != loop_thread for _, tid in seen)


@pytest.mark.asyncio
async def test_fetch_resolves_off_the_event_loop():
    loop_thread = threading.get_ident()
    seen: list[tuple[str, int]] = []
    page = _make_mock_page(url="about:blank")
    page.evaluate.return_value = "Readable page text"
    tool = _make_tool(bm=_make_browser_manager(page))

    with patch("agent_os.agent.tools.browser_safety.socket.getaddrinfo",
               _recording_getaddrinfo(seen)):
        result = await tool.execute_async(action="fetch", url="https://example.com/doc")

    assert "Readable page text" in result.content
    assert [h for h, _ in seen] == ["example.com"]
    assert all(tid != loop_thread for _, tid in seen)


@pytest.mark.asyncio
async def test_a_host_resolving_to_a_private_address_is_still_blocked():
    seen: list[tuple[str, int]] = []
    page = _make_mock_page()
    tool = _make_tool(bm=_make_browser_manager(page))

    with patch("agent_os.agent.tools.browser_safety.socket.getaddrinfo",
               _recording_getaddrinfo(seen, ip="10.0.0.7")):
        nav = await tool.execute_async(action="navigate", url="https://intranet.example")
        fetch = await tool.execute_async(action="fetch", url="https://intranet.example")

    assert "private address" in nav.content
    assert "URL blocked" in fetch.content and "private address" in fetch.content
    page.goto.assert_not_awaited()
