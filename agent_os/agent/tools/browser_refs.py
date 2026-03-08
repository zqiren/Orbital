# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Snapshot pipeline: accessibility tree serialization, ref assignment, and resolution."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RefEntry:
    """An element reference entry mapping a ref ID to a Playwright locator strategy."""
    role: str
    name: str | None
    nth: int  # disambiguation index for duplicate (role, name) pairs


@dataclass
class SnapshotStats:
    """Statistics about a serialized snapshot."""
    lines: int
    chars: int
    estimated_tokens: int  # ceil(chars / 4)
    refs: int
    interactive_refs: int


# Type alias
RefMap = dict[str, RefEntry]


# --- Role classification ---

INTERACTIVE_ROLES = {
    "button", "link", "textbox", "checkbox", "radio", "combobox",
    "slider", "tab", "menuitem", "switch", "searchbox", "spinbutton",
    "option", "menuitemcheckbox", "menuitemradio", "treeitem",
}

CONTENT_ROLES = {
    "heading", "cell", "listitem", "article", "paragraph", "img",
    "blockquote", "caption", "definition", "term", "figure",
    "columnheader", "rowheader",
}

STRUCTURAL_ROLES = {
    "generic", "group", "list", "table", "navigation", "main",
    "banner", "contentinfo", "complementary", "form", "region",
    "toolbar", "tablist", "tabpanel", "tree", "treegrid", "grid",
    "row", "rowgroup", "separator", "none", "presentation",
}

_MAX_TEXT_LENGTH = 200
_MAX_NAME_LENGTH = 100


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _should_get_ref(role: str, name: str | None) -> bool:
    """Determine whether a node should be assigned a ref."""
    if role in INTERACTIVE_ROLES:
        return True
    if role in CONTENT_ROLES and name:
        return True
    return False


def serialize_snapshot(
    ax_tree: dict | None,
    interactive_only: bool = False,
) -> tuple[str, RefMap, SnapshotStats]:
    """Convert a Playwright accessibility tree into LLM-readable text with refs.

    Args:
        ax_tree: Output of page.accessibility.snapshot() — a dict tree.
        interactive_only: If True, output only interactive elements in a flat list.

    Returns:
        (text, ref_map, stats) tuple.
    """
    if not ax_tree:
        stats = SnapshotStats(lines=0, chars=0, estimated_tokens=0, refs=0, interactive_refs=0)
        return ("", {}, stats)

    ref_map: RefMap = {}
    ref_counter = 0
    # Track (role, name) occurrences for nth disambiguation
    seen_pairs: dict[tuple[str, str | None], int] = {}
    lines: list[str] = []

    def _next_ref(role: str, name: str | None) -> str:
        nonlocal ref_counter
        ref_counter += 1
        ref_id = f"e{ref_counter}"
        pair = (role, name)
        nth = seen_pairs.get(pair, 0)
        seen_pairs[pair] = nth + 1
        ref_map[ref_id] = RefEntry(role=role, name=name, nth=nth)
        return ref_id

    def _format_node(node: dict, depth: int) -> None:
        role = node.get("role", "")
        name = node.get("name")
        children = node.get("children", [])
        value = node.get("value")
        checked = node.get("checked")
        pressed = node.get("pressed")
        level = node.get("level")

        # Keep full name for ref resolution, truncated version for display
        full_name = name
        display_name = _truncate(name, _MAX_NAME_LENGTH) if name else None

        is_interactive = role in INTERACTIVE_ROLES
        is_content = role in CONTENT_ROLES
        gets_ref = _should_get_ref(role, name)

        if interactive_only:
            # Only emit interactive elements, flat list
            if is_interactive:
                ref_id = _next_ref(role, full_name)
                parts = [f"[ref={ref_id}] {role}"]
                if display_name:
                    parts.append(f'"{display_name}"')
                if value is not None:
                    parts.append(f'value="{_truncate(str(value), _MAX_TEXT_LENGTH)}"')
                if checked is not None:
                    parts.append("checked" if checked else "unchecked")
                if pressed is not None:
                    parts.append("pressed" if pressed else "not pressed")
                lines.append(" ".join(parts))
            # Recurse into children regardless (interactive elements may be nested)
            for child in children:
                if isinstance(child, dict):
                    _format_node(child, depth)
            return

        # Full tree mode
        indent = "  " * depth

        if gets_ref:
            ref_id = _next_ref(role, full_name)
            parts = [f"{indent}- {role}"]
            if display_name:
                parts.append(f'"{display_name}"')
            parts.append(f"[ref={ref_id}]")
            if value is not None:
                parts.append(f'value="{_truncate(str(value), _MAX_TEXT_LENGTH)}"')
            if checked is not None:
                parts.append("checked" if checked else "unchecked")
            if pressed is not None:
                parts.append("pressed" if pressed else "not pressed")
            if level is not None:
                parts.append(f"[level={level}]")
            lines.append(" ".join(parts))
        elif is_content:
            # Content without name — render inline text from name or as plain role
            parts = [f"{indent}- {role}"]
            if level is not None:
                parts.append(f"[level={level}]")
            if display_name:
                parts.append(f'"{display_name}"')
            lines.append(" ".join(parts))
        elif role in STRUCTURAL_ROLES or role not in INTERACTIVE_ROLES | CONTENT_ROLES:
            # Structural or unknown — only render if has children
            if children:
                text_child = None
                # Check if node has only a text name and no children that are dicts
                if display_name and not any(isinstance(c, dict) for c in children):
                    text_child = _truncate(display_name, _MAX_TEXT_LENGTH)
                if text_child:
                    lines.append(f"{indent}- {role}: {text_child}")
                else:
                    lines.append(f"{indent}- {role}:")
            elif display_name:
                # Structural with name but no children — emit as text
                lines.append(f"{indent}- {role}: {_truncate(display_name, _MAX_TEXT_LENGTH)}")
            else:
                # Skip empty structural nodes
                pass
                # Don't recurse children below — handled in the children block
                # Actually, we need to handle the no-children case: just skip
                return

        # Recurse children
        for child in children:
            if isinstance(child, dict):
                _format_node(child, depth + 1)

    _format_node(ax_tree, 0)

    text = "\n".join(lines)
    total_refs = len(ref_map)
    interactive_refs = sum(1 for e in ref_map.values() if e.role in INTERACTIVE_ROLES)
    char_count = len(text)
    line_count = len(lines)
    stats = SnapshotStats(
        lines=line_count,
        chars=char_count,
        estimated_tokens=math.ceil(char_count / 4) if char_count > 0 else 0,
        refs=total_refs,
        interactive_refs=interactive_refs,
    )
    return (text, ref_map, stats)


async def resolve_ref(ref_map: RefMap, ref: str, page) -> object:
    """Resolve a ref string (e.g. 'e5') to a Playwright Locator.

    Args:
        ref_map: Current page RefMap.
        ref: Element reference string (e.g. "e5").
        page: Playwright Page object.

    Returns:
        A Playwright Locator targeting the referenced element.

    Raises:
        ValueError: If ref is not found in the map (stale ref).
    """
    if ref not in ref_map:
        raise ValueError(
            f"Ref {ref} is no longer valid — the page may have changed. "
            "Run snapshot to get updated refs."
        )
    entry = ref_map[ref]

    if entry.role == "clickable":
        # Cursor-interactive element — use CSS selector stored in name
        locator = page.locator(entry.name)
    else:
        kwargs: dict = {"exact": True}
        if entry.name:
            kwargs["name"] = entry.name
        locator = page.get_by_role(entry.role, **kwargs)

    if entry.nth > 0:
        locator = locator.nth(entry.nth)

    return locator


async def discover_cursor_interactive(page) -> list[dict]:
    """Find elements with cursor:pointer/onclick/tabindex that lack ARIA roles.

    Injects JS into the page to discover interactive elements that are not
    captured by the accessibility tree.

    Args:
        page: Playwright Page object.

    Returns:
        List of {selector, label} dicts for cursor-interactive elements.
    """
    js = """
    () => {
        function uniqueSelector(el) {
            if (el.id) return '#' + CSS.escape(el.id);
            const parts = [];
            while (el && el !== document.body) {
                let selector = el.tagName.toLowerCase();
                if (el.id) {
                    parts.unshift('#' + CSS.escape(el.id));
                    break;
                }
                const parent = el.parentElement;
                if (parent) {
                    const siblings = Array.from(parent.children).filter(
                        c => c.tagName === el.tagName
                    );
                    if (siblings.length > 1) {
                        const idx = siblings.indexOf(el) + 1;
                        selector += ':nth-of-type(' + idx + ')';
                    }
                }
                parts.unshift(selector);
                el = parent;
            }
            return parts.join(' > ');
        }

        const interactive = ['A','BUTTON','INPUT','SELECT','TEXTAREA'];
        return [...document.querySelectorAll('*')].filter(el => {
            const style = getComputedStyle(el);
            const hasPointer = style.cursor === 'pointer';
            const hasOnclick = el.hasAttribute('onclick') || el.hasAttribute('tabindex');
            const hasRole = el.getAttribute('role');
            const isInteractive = interactive.includes(el.tagName);
            return (hasPointer || hasOnclick) && !hasRole && !isInteractive;
        }).map(el => ({
            selector: uniqueSelector(el),
            label: (el.textContent || '').trim().slice(0, 50) || el.getAttribute('aria-label') || 'unlabeled'
        }));
    }
    """
    return await page.evaluate(js)
