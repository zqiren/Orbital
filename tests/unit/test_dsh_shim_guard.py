# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The composition shims must stay deletable.

Task 7's shims are rc-ware built to be thrown away: the day upstream
``dsh-acp`` forwards tool activity (and grows ``session/load``) natively, the
removal is a composition edit plus a lockfile refresh — no Orbital code
changes. That promise only holds if no Orbital code ever learns their ids. The
contract on the wire is standard ACP: a capability flag and ordinary
``session/update`` notifications, which the transport already handles for every
ACP agent. If any Orbital module starts naming a shim, the shim has stopped
being a shim and this test is the tripwire.

``agent_os/agents/assets/dsh/`` is excluded on purpose — the composition itself
lives there, and that is exactly where these ids belong.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_OS = REPO_ROOT / "agent_os"
COMPOSITION_DIR = AGENT_OS / "agents" / "assets" / "dsh"

SHIM_IDS = ("orbital-acp-resume", "orbital-acp-activity")

# Binary and build noise: a byte scan of these proves nothing.
SKIP_SUFFIXES = {".pyc", ".pyo", ".so", ".dylib", ".dll", ".node", ".png",
                 ".jpg", ".jpeg", ".gif", ".ico", ".icns", ".woff", ".woff2",
                 ".zip", ".gz", ".exe"}
SKIP_DIR_NAMES = {"__pycache__", "node_modules", ".git"}


def _scannable_files():
    for path in AGENT_OS.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if COMPOSITION_DIR in path.parents:
            continue
        yield path


@pytest.mark.parametrize("shim_id", SHIM_IDS)
def test_no_orbital_code_references_a_shim_id(shim_id):
    offenders = []
    for path in _scannable_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if shim_id in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, (
        f"{shim_id!r} leaked out of the composition into Orbital code: "
        f"{offenders}. The shims are deletable only while their ids exist "
        f"solely in agent_os/agents/assets/dsh/; Orbital's side of the "
        f"contract is standard ACP, never a shim-specific id."
    )


def test_the_guard_is_actually_scanning_something():
    """A guard that silently walks an empty tree passes forever."""
    scanned = list(_scannable_files())

    assert len(scanned) > 50, (
        f"only {len(scanned)} files scanned — the exclusion rules have eaten "
        f"the tree and this guard proves nothing"
    )
    assert any(p.name == "acp_sdk_transport.py" for p in scanned), (
        "the ACP transport must be in scope: it is the single most likely "
        "place for a shim id to be hardcoded"
    )


def test_the_composition_is_excluded_and_does_carry_an_id():
    """The other half: the ids must live SOMEWHERE, or nothing is wired."""
    template = COMPOSITION_DIR / "cordis.template.yml"
    assert template.is_file(), f"missing shipped composition: {template}"

    text = template.read_text(encoding="utf-8")
    assert "orbital-acp-activity" in text, (
        "the activity shim's composition block is missing — the guard above "
        "would then pass vacuously"
    )
    assert not any(COMPOSITION_DIR in p.parents for p in _scannable_files()), (
        "the composition dir must be excluded from the scan"
    )
