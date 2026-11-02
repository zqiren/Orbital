# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The live model probes run the detected binary, not a bare name.

On Windows an npm-installed CLI exists only as ``codex.cmd``. A bare ``codex``
cannot be spawned (WinError 2), the probe reads "list unavailable", and a save
or a pin-menu pick then accepts any model name, which fails at dispatch.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent_os.agent.transports import claude_models, codex_models
from agent_os.api.routes import settings as settings_routes

CODEX = r"C:\npm\codex.cmd"
CLAUDE = r"C:\npm\claude.cmd"


class _Engine:
    def check_all(self, **_):
        return [SimpleNamespace(slug="codex", binary_path=CODEX),
                SimpleNamespace(slug="claude-code", binary_path=CLAUDE),
                SimpleNamespace(slug="aider", binary_path=None)]


@pytest.fixture
def probes(monkeypatch):
    seen = []

    async def probe(binary, **_):
        seen.append(binary)
        return None

    monkeypatch.setattr(codex_models, "get_codex_models_cached", probe)
    monkeypatch.setattr(claude_models, "get_claude_models_cached", probe)
    return seen


def test_codex_probe_without_a_binary_uses_the_detected_one(monkeypatch, probes):
    monkeypatch.setattr(settings_routes, "_setup_engine", _Engine())
    asyncio.run(settings_routes._codex_live_models())
    assert probes == [CODEX]


def test_claude_probe_without_a_binary_uses_the_detected_one(monkeypatch, probes):
    monkeypatch.setattr(settings_routes, "_setup_engine", _Engine())
    asyncio.run(settings_routes._claude_live_models())
    assert probes == [CLAUDE]


def test_a_given_binary_wins(monkeypatch, probes):
    monkeypatch.setattr(settings_routes, "_setup_engine", _Engine())
    asyncio.run(settings_routes._codex_live_models(r"C:\elsewhere\codex.exe"))
    assert probes == [r"C:\elsewhere\codex.exe"]


def test_bare_name_when_nothing_was_detected(monkeypatch, probes):
    monkeypatch.setattr(settings_routes, "_setup_engine", None)
    asyncio.run(settings_routes._codex_live_models())
    assert probes == ["codex"]
