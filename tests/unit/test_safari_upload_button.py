# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression test for Safari upload-button overlap bug.

On mobile Safari the bottom toolbar can hide the upload button in the
Files tab.  The root cause was that ``<main>`` in App.tsx was **not** a
flex-column container, so the mobile "Back" button (~44 px) plus a
child with ``h-full`` (100 %) overflowed the parent.  The bottom of
ProjectDetail — where the upload button lives — was clipped by
``overflow-hidden``.

The fix:
  * App.tsx ``<main>``  → add ``flex flex-col``
  * ProjectDetail.tsx   → ``flex-1 min-h-0``  (instead of ``h-full``)
  * GlobalSettings.tsx  → ``flex-1 min-h-0``  (instead of ``h-full``)
  * CreateProject.tsx   → ``flex-1 min-h-0``  (instead of ``h-full``)

These tests read the source files and assert the critical CSS classes
are present so the fix is not accidentally reverted.
"""

from pathlib import Path

WEB_SRC = Path(__file__).resolve().parents[2] / "web" / "src"


def _read(rel: str) -> str:
    return (WEB_SRC / rel).read_text()


# ---- App.tsx ---------------------------------------------------------------

def test_main_element_is_flex_column():
    """<main> must be a flex column so the back button and content share space."""
    source = _read("App.tsx")
    # The main element's className must include both "flex" and "flex-col".
    # We look for the template-literal fragment that builds the className.
    assert "flex-1 overflow-hidden flex flex-col" in source, (
        "<main> in App.tsx must have 'flex flex-col' to prevent the mobile "
        "back button from pushing content below the visible viewport"
    )


# ---- ProjectDetail.tsx -----------------------------------------------------

def test_project_detail_uses_flex1():
    """ProjectDetail root must use flex-1 min-h-0, NOT h-full."""
    source = _read("components/ProjectDetail.tsx")
    assert "flex-1 min-h-0" in source, (
        "ProjectDetail root should use 'flex-1 min-h-0' so it fills "
        "remaining space in the flex-column <main> without overflowing"
    )
    # h-full on the root div was the original bug trigger
    # Make sure the root div doesn't use h-full anymore
    assert 'className="flex flex-col h-full"' not in source, (
        "ProjectDetail root must NOT use 'h-full' — it causes overflow "
        "when a sibling (mobile back button) is present"
    )


# ---- GlobalSettings.tsx / CreateProject.tsx --------------------------------

def test_global_settings_uses_flex1():
    """GlobalSettings root must use flex-1 min-h-0, NOT h-full."""
    source = _read("components/GlobalSettings.tsx")
    assert "flex-1 min-h-0" in source


def test_create_project_uses_flex1():
    """CreateProject root must use flex-1 min-h-0, NOT h-full."""
    source = _read("components/CreateProject.tsx")
    assert "flex-1 min-h-0" in source


# ---- FileExplorer.tsx (upload button safe-area padding) --------------------

def test_upload_button_has_safe_area_padding():
    """Upload button wrapper must include safe-area-inset-bottom for notched devices."""
    source = _read("components/FileExplorer.tsx")
    assert "safe-area-inset-bottom" in source, (
        "Upload button wrapper must use env(safe-area-inset-bottom) "
        "to avoid the home-indicator area on iOS"
    )
