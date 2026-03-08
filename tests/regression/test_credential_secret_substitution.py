# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: dotted secret tokens like <secret:gmail.email> were not substituted.

The SECRET_PATTERN regex in browser_safety.py only matched [a-zA-Z0-9_],
so the dot in 'gmail.email' caused detect_secrets() to return [] and the
literal string '<secret:gmail.email>' was typed into the browser form.
Additionally, BrowserTool was not wired to the UserCredentialStore at all.

These tests fail without the fix and pass with it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.tools.browser import BrowserTool
from agent_os.agent.tools.browser_refs import RefEntry
from agent_os.agent.tools.browser_safety import detect_secrets, substitute_secrets


# --- Bug 1: regex didn't match dotted keys ---

def test_detect_secrets_matches_dotted_keys():
    """Regression: <secret:gmail.email> was not detected because dot was missing from regex."""
    assert detect_secrets("<secret:gmail.email>") == ["gmail.email"]
    assert detect_secrets("<secret:gmail.password>") == ["gmail.password"]


def test_detect_secrets_matches_dotted_in_sentence():
    """Regression: multiple dotted secrets in a string were not found."""
    text = "user=<secret:site.user> pw=<secret:site.pass>"
    assert detect_secrets(text) == ["site.user", "site.pass"]


# --- Bug 3: substitute_secrets now takes a callable resolver ---

def test_substitute_secrets_calls_resolver():
    """Regression: substitute_secrets used flat dict lookup, not callable resolver."""
    resolver = lambda key: "user@example.com" if key == "gmail.email" else None
    result = substitute_secrets("<secret:gmail.email>", resolver)
    assert result == "user@example.com"


def test_substitute_secrets_resolver_returns_none_raises():
    """Regression: missing credential should raise ValueError."""
    resolver = lambda key: None
    with pytest.raises(ValueError, match="not found"):
        substitute_secrets("<secret:gmail.email>", resolver)


# --- Bug 2: BrowserTool wired to UserCredentialStore ---

@pytest.mark.asyncio
async def test_browser_type_resolves_dotted_secret_from_store():
    """Regression: BrowserTool typed literal '<secret:gmail.email>' into form
    because (a) regex missed the dot, (b) credential store wasn't wired,
    (c) substitute_secrets couldn't call get_value."""
    # Mock UserCredentialStore
    mock_store = MagicMock()
    mock_store.get_value.return_value = "real_user@gmail.com"

    # Mock browser manager
    page = AsyncMock()
    page.url = "https://accounts.google.com/signin"
    page.title = AsyncMock(return_value="Sign in")
    bm = MagicMock()
    bm.get_page = AsyncMock(return_value=page)
    bm.get_ref_map.return_value = {
        "e2": RefEntry(role="textbox", name="Email", nth=0),
    }
    bm.capture_screenshot = AsyncMock(return_value="/tmp/shot.png")

    tool = BrowserTool(
        browser_manager=bm,
        project_id="test",
        workspace="/tmp",
        autonomy_preset="hands_off",
        user_credential_store=mock_store,
    )

    mock_locator = AsyncMock()
    with patch("agent_os.agent.tools.browser.resolve_ref",
               new_callable=AsyncMock, return_value=mock_locator):
        result = await tool.execute(action="type", ref="e2", text="<secret:gmail.email>")

    # Real value should have been typed
    mock_locator.fill.assert_awaited_once_with("real_user@gmail.com")
    mock_store.get_value.assert_called_once_with("gmail", "email")
    # Result should show masked token, not real value
    assert "<secret:gmail.email>" in result.content
    assert "real_user@gmail.com" not in result.content
