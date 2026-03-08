# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for browser_refs: snapshot pipeline, ref resolution, role classification."""

import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.tools.browser_refs import (
    CONTENT_ROLES,
    INTERACTIVE_ROLES,
    STRUCTURAL_ROLES,
    RefEntry,
    RefMap,
    SnapshotStats,
    resolve_ref,
    serialize_snapshot,
)


# ---------------------------------------------------------------------------
# Helpers: mock accessibility trees
# ---------------------------------------------------------------------------

def _tree(role, name=None, children=None, **extra):
    """Build a minimal a11y tree node."""
    node = {"role": role}
    if name is not None:
        node["name"] = name
    if children:
        node["children"] = children
    node.update(extra)
    return node


# ---------------------------------------------------------------------------
# 1. test_basic_serialization
# ---------------------------------------------------------------------------

def test_basic_serialization():
    """Simple tree with heading + navigation + links + button."""
    tree = _tree("main", children=[
        _tree("heading", "Welcome", level=1),
        _tree("navigation", children=[
            _tree("link", "Home"),
            _tree("link", "About"),
        ]),
        _tree("button", "Submit"),
    ])
    text, ref_map, stats = serialize_snapshot(tree)

    # Heading with name gets a ref
    assert 'heading "Welcome"' in text
    assert "[level=1]" in text

    # Links and button get refs
    assert '[ref=e' in text
    assert 'link "Home"' in text
    assert 'link "About"' in text
    assert 'button "Submit"' in text

    # Navigation is structural — rendered with colon
    assert "- navigation:" in text

    # Indentation — navigation children indented more than navigation itself
    for line in text.split("\n"):
        if "navigation:" in line:
            nav_indent = len(line) - len(line.lstrip())
        if 'link "Home"' in line:
            link_indent = len(line) - len(line.lstrip())
    assert link_indent > nav_indent

    # RefMap has entries for all interactive + named content
    assert len(ref_map) >= 3  # at least: 2 links + 1 button


# ---------------------------------------------------------------------------
# 2. test_interactive_only_mode
# ---------------------------------------------------------------------------

def test_interactive_only_mode():
    """Interactive-only mode outputs flat list of only interactive elements."""
    tree = _tree("main", children=[
        _tree("heading", "Title", level=2),
        _tree("navigation", children=[
            _tree("link", "Home"),
            _tree("button", "Go"),
        ]),
        _tree("paragraph", "Some text"),
    ])
    text, ref_map, stats = serialize_snapshot(tree, interactive_only=True)

    lines = [l for l in text.split("\n") if l.strip()]

    # Only interactive elements
    assert len(lines) == 2
    assert '[ref=e1] link "Home"' in text
    assert '[ref=e2] button "Go"' in text

    # No structural or content nodes
    assert "heading" not in text
    assert "paragraph" not in text
    assert "navigation" not in text

    # Flat — no indentation
    for line in lines:
        assert line == line.lstrip()


# ---------------------------------------------------------------------------
# 3. test_ref_assignment_order
# ---------------------------------------------------------------------------

def test_ref_assignment_order():
    """Refs are assigned in tree-walk (depth-first) order: e1, e2, e3..."""
    tree = _tree("main", children=[
        _tree("link", "First"),
        _tree("group", children=[
            _tree("button", "Second"),
            _tree("textbox", "Third"),
        ]),
        _tree("link", "Fourth"),
    ])
    text, ref_map, stats = serialize_snapshot(tree)

    assert ref_map["e1"].name == "First"
    assert ref_map["e1"].role == "link"

    assert ref_map["e2"].name == "Second"
    assert ref_map["e2"].role == "button"

    assert ref_map["e3"].name == "Third"
    assert ref_map["e3"].role == "textbox"

    assert ref_map["e4"].name == "Fourth"
    assert ref_map["e4"].role == "link"


# ---------------------------------------------------------------------------
# 4. test_nth_disambiguation
# ---------------------------------------------------------------------------

def test_nth_disambiguation():
    """Two elements with same (role, name) get different nth values."""
    tree = _tree("main", children=[
        _tree("button", "Save"),
        _tree("button", "Save"),
        _tree("button", "Cancel"),
    ])
    text, ref_map, stats = serialize_snapshot(tree)

    save_entries = [e for e in ref_map.values() if e.name == "Save"]
    assert len(save_entries) == 2
    assert save_entries[0].nth == 0
    assert save_entries[1].nth == 1

    cancel_entries = [e for e in ref_map.values() if e.name == "Cancel"]
    assert len(cancel_entries) == 1
    assert cancel_entries[0].nth == 0


# ---------------------------------------------------------------------------
# 5. test_role_classification
# ---------------------------------------------------------------------------

def test_role_classification():
    """Interactive roles always get refs, content only when named, structural never."""
    tree = _tree("main", children=[
        # Interactive — always gets ref
        _tree("button", "Click"),
        _tree("textbox"),  # interactive without name — still gets ref
        # Content — only when named
        _tree("heading", "Title", level=1),  # named → ref
        _tree("paragraph"),  # unnamed content → no ref
        # Structural — never
        _tree("navigation", children=[
            _tree("link", "Nav Link"),
        ]),
    ])
    text, ref_map, stats = serialize_snapshot(tree)

    roles_in_map = {e.role for e in ref_map.values()}

    # Interactive present
    assert "button" in roles_in_map
    assert "textbox" in roles_in_map

    # Named content present
    assert "heading" in roles_in_map

    # Navigation (structural) NOT in ref map
    assert "navigation" not in roles_in_map

    # The unnamed paragraph should NOT be in ref_map
    paragraph_refs = [e for e in ref_map.values() if e.role == "paragraph"]
    assert len(paragraph_refs) == 0

    # Verify the classification sets don't overlap
    assert not INTERACTIVE_ROLES & CONTENT_ROLES
    assert not INTERACTIVE_ROLES & STRUCTURAL_ROLES
    assert not CONTENT_ROLES & STRUCTURAL_ROLES


# ---------------------------------------------------------------------------
# 6. test_text_truncation
# ---------------------------------------------------------------------------

def test_text_truncation():
    """Long names stored in full for ref resolution, truncated in display text."""
    long_name = "A" * 150  # exceeds 100 char limit
    tree = _tree("main", children=[
        _tree("link", long_name),
        _tree("textbox", "Input", value="V" * 250),  # value exceeds 200
    ])
    text, ref_map, stats = serialize_snapshot(tree)

    # RefEntry stores full name (for accurate Playwright locator resolution)
    link_entry = [e for e in ref_map.values() if e.role == "link"][0]
    assert link_entry.name is not None
    assert len(link_entry.name) == 150  # full name preserved
    assert link_entry.name == long_name

    # Display text truncates the name to 100 + "..."
    assert "A" * 100 + "..." in text
    assert "A" * 150 not in text

    # Value truncated in output text
    assert "V" * 200 + "..." in text
    assert "V" * 250 not in text


# ---------------------------------------------------------------------------
# 7. test_snapshot_stats
# ---------------------------------------------------------------------------

def test_snapshot_stats():
    """Stats correctly report lines, chars, estimated_tokens, refs, interactive_refs."""
    tree = _tree("main", children=[
        _tree("heading", "Title", level=1),
        _tree("link", "Home"),
        _tree("button", "Submit"),
        _tree("paragraph", "Info"),
    ])
    text, ref_map, stats = serialize_snapshot(tree)

    assert stats.lines == len([l for l in text.split("\n") if True])  # total lines
    assert stats.chars == len(text)
    assert stats.estimated_tokens == math.ceil(len(text) / 4)
    assert stats.refs == len(ref_map)

    # Interactive refs: link + button = 2 (heading is content, paragraph might not have ref)
    interactive = sum(1 for e in ref_map.values() if e.role in INTERACTIVE_ROLES)
    assert stats.interactive_refs == interactive


# ---------------------------------------------------------------------------
# 8. test_value_and_state
# ---------------------------------------------------------------------------

def test_value_and_state():
    """Input with value and checkbox with checked state are rendered correctly."""
    tree = _tree("main", children=[
        _tree("textbox", "Email", value="user@test.com"),
        _tree("checkbox", "Accept", checked=True),
        _tree("switch", "Dark mode", pressed=False),
    ])
    text, ref_map, stats = serialize_snapshot(tree)

    assert 'value="user@test.com"' in text
    assert "checked" in text
    # Switch with pressed=False
    assert "not pressed" in text


# ---------------------------------------------------------------------------
# 9. test_heading_level
# ---------------------------------------------------------------------------

def test_heading_level():
    """Headings include [level=N] annotation."""
    tree = _tree("main", children=[
        _tree("heading", "Main Title", level=1),
        _tree("heading", "Subtitle", level=2),
        _tree("heading", level=3),  # unnamed heading — no ref, but still rendered
    ])
    text, ref_map, stats = serialize_snapshot(tree)

    assert "[level=1]" in text
    assert "[level=2]" in text
    # Level 3 unnamed heading has no ref but still has level in output
    assert "[level=3]" in text


# ---------------------------------------------------------------------------
# 10. test_resolve_ref
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolve_ref():
    """resolve_ref builds correct Playwright locator with exact=True and nth."""
    ref_map: RefMap = {
        "e1": RefEntry(role="link", name="Home", nth=0),
        "e2": RefEntry(role="button", name="Save", nth=1),
        "e3": RefEntry(role="textbox", name=None, nth=0),
        "e4": RefEntry(role="clickable", name="#add-to-cart", nth=0),
    }

    page = MagicMock()
    locator_mock = MagicMock()
    nth_mock = MagicMock()
    locator_mock.nth.return_value = nth_mock
    page.get_by_role.return_value = locator_mock
    page.locator.return_value = locator_mock

    # e1: link "Home" nth=0 — no nth call needed
    result = await resolve_ref(ref_map, "e1", page)
    page.get_by_role.assert_called_with("link", exact=True, name="Home")
    assert result == locator_mock

    # e2: button "Save" nth=1 — nth call needed
    page.reset_mock()
    locator_mock.reset_mock()
    page.get_by_role.return_value = locator_mock
    locator_mock.nth.return_value = nth_mock
    result = await resolve_ref(ref_map, "e2", page)
    page.get_by_role.assert_called_with("button", exact=True, name="Save")
    locator_mock.nth.assert_called_with(1)
    assert result == nth_mock

    # e3: textbox with no name
    page.reset_mock()
    locator_mock.reset_mock()
    page.get_by_role.return_value = locator_mock
    result = await resolve_ref(ref_map, "e3", page)
    page.get_by_role.assert_called_with("textbox", exact=True)

    # e4: clickable — uses page.locator with CSS selector
    page.reset_mock()
    locator_mock.reset_mock()
    page.locator.return_value = locator_mock
    result = await resolve_ref(ref_map, "e4", page)
    page.locator.assert_called_with("#add-to-cart")


# ---------------------------------------------------------------------------
# 11. test_stale_ref_error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stale_ref_error():
    """Accessing a ref not in the map raises ValueError with helpful message."""
    ref_map: RefMap = {"e1": RefEntry(role="link", name="Home", nth=0)}
    page = MagicMock()

    with pytest.raises(ValueError, match="Ref e99 is no longer valid"):
        await resolve_ref(ref_map, "e99", page)


# ---------------------------------------------------------------------------
# 12. test_empty_tree
# ---------------------------------------------------------------------------

def test_empty_tree():
    """Empty or None tree returns empty output, empty RefMap, zeroed stats."""
    # None input
    text, ref_map, stats = serialize_snapshot(None)
    assert text == ""
    assert ref_map == {}
    assert stats.lines == 0
    assert stats.chars == 0
    assert stats.estimated_tokens == 0
    assert stats.refs == 0
    assert stats.interactive_refs == 0

    # Empty dict
    text, ref_map, stats = serialize_snapshot({})
    assert text == ""
    assert ref_map == {}
    assert stats.refs == 0
