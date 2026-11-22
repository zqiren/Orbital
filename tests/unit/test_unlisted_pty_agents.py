"""Spec 171 B2: the six terminal (PTY) agents come off the list.

"Unlisted" means: no list shows them (Settings' agent cards, the pin menu,
the dispatch tool's agent list, auto-detect, onboarding), because every list
is built from ``SetupEngine.check_all()``. The manifest stays registered, so
a reference a user already has (a session pinned to one, an automation
assigned to one, ``enabled_sub_agents``) still resolves by slug exactly as
before: nothing errors on upgrade, and a downgrade reads the same data.
"""

import os
from types import SimpleNamespace

from agent_os.agents.manifest import ManifestLoader
from agent_os.agents.registry import AgentRegistry
from agent_os.agents.setup_engine import SetupEngine

MANIFESTS = os.path.join(os.path.dirname(__file__), "..", "..",
                         "agent_os", "agents", "manifests")
# Spec 193 retired gemini-cli outright (Antigravity replaced it), so five remain.
PTY_AGENTS = {"aider", "cline", "continue-cli", "copilot-cli", "goose"}


def _bundled() -> AgentRegistry:
    registry = AgentRegistry()
    registry.load_directory(MANIFESTS)
    return registry


def _minimal(**extra) -> dict:
    return {"manifest_version": "1", "name": "X", "slug": "x", "description": "",
            "author": "a", "version": "1.0.0",
            "runtime": {"adapter": "cli", "command": "x"}, **extra}


def test_listed_defaults_true_and_reads_false():
    assert ManifestLoader._build(_minimal()).listed is True
    assert ManifestLoader._build(_minimal(listed=False)).listed is False


def test_exactly_the_pty_agents_are_unlisted():
    registry = _bundled()
    unlisted = {m.slug for m in registry.list_all() if not m.listed}
    assert unlisted == PTY_AGENTS
    for m in registry.list_all():
        if m.slug in PTY_AGENTS:
            assert m.runtime.transport == "pty"
    # Still registered: a saved pin or automation resolves by slug.
    for slug in PTY_AGENTS:
        assert registry.get(slug) is not None


def test_check_all_skips_unlisted_but_check_agent_still_answers(tmp_path):
    registry = _bundled()
    engine = SetupEngine(registry, data_dir=str(tmp_path))
    probed = []

    def check_agent(slug):
        probed.append(slug)
        return SimpleNamespace(slug=slug, name=slug, installed=True)

    engine.check_agent = check_agent
    slugs = {s.slug for s in engine.check_all()}
    assert slugs.isdisjoint(PTY_AGENTS)
    assert {"claude-code", "codex"} <= slugs
    # The sweep no longer probes those CLIs at all.
    assert PTY_AGENTS.isdisjoint(probed)
    assert engine.check_agent("aider").slug == "aider"


def test_check_all_tolerates_manifests_without_the_field(tmp_path):
    """Test doubles (and any manifest object) without ``listed`` stay listed."""
    from unittest.mock import MagicMock

    registry = MagicMock()
    registry.list_all.return_value = [SimpleNamespace(slug="a"), SimpleNamespace(slug="b")]
    engine = SetupEngine(registry, data_dir=str(tmp_path))
    engine.check_agent = lambda slug: SimpleNamespace(slug=slug, name=slug, installed=True)
    assert [s.slug for s in engine.check_all()] == ["a", "b"]
