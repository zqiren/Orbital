# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared in-process harness for the keyless-onboarding tests.

A real ``AgentManager`` + ``ProjectStore`` behind the real ``agents_v2``
router, with three doubles: a card store (``FakeCardStore``, keyed or not), a
setup engine that answers ``check_all`` from a fixed status list, and a
sub-agent manager whose ``send`` records the dispatch instead of spawning a
CLI. Not a test module (no ``test_`` prefix): imported by
``test_keyless_gate.py``, ``test_project_default_pin.py`` and
``test_worker_onboarding.py``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.agents.setup_types import AgentSetupStatus
from tests.card_doubles import FakeCardStore


def agent_status(slug: str, *, installed: bool = True, creds: bool = True,
                 update_needed: bool = False, name: str | None = None
                 ) -> AgentSetupStatus:
    return AgentSetupStatus(
        slug=slug, name=name or slug, installed=installed,
        binary_path=f"/usr/local/bin/{slug}" if installed else None,
        version="1.0.0", dependencies_met=True, missing_dependencies=[],
        credentials_configured=creds,
        missing_credentials=[] if creds else ["login"],
        update_needed=update_needed,
    )


class FakeSetupEngine:
    """``check_all`` from a fixed list; ``statuses`` may be mutated by a test."""

    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = 0

    def check_all(self, allow_stale: bool = False, **_kw):
        self.calls += 1
        return list(self.statuses)


DEFAULT_STATUSES = (
    ("built-in", True, True),
    ("claude-code", True, True),
    ("codex", True, True),
)


def default_statuses():
    return [agent_status(s, installed=i, creds=c) for s, i, c in DEFAULT_STATUSES]


def make_sub_agent_manager():
    sub = MagicMock()
    sub.send = AsyncMock(side_effect=lambda project_id, handle, *_a, **_k:
                         f"Message sent to {handle}")
    sub.list_active.return_value = []
    return sub


def build_client(tmp_path, *, keyed: bool, statuses=None, settings_store=None):
    """Return ``(TestClient, deps)`` over a fresh data dir.

    ``keyed`` decides whether the default card carries a key (the manager is
    ready) or not (keyless install).
    """
    from agent_os.api.routes import agents_v2
    from agent_os.api.ws import WebSocketManager
    from agent_os.config.provider_registry import ProviderRegistry
    from agent_os.daemon_v2.activity_translator import ActivityTranslator
    from agent_os.daemon_v2.agent_manager import AgentManager
    from agent_os.daemon_v2.process_manager import ProcessManager
    from agent_os.daemon_v2.project_store import ProjectStore

    data_dir = tmp_path / "orbital-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    project_store = ProjectStore(data_dir=str(data_dir))
    ws_manager = WebSocketManager()
    activity_translator = ActivityTranslator(ws_manager)
    process_manager = ProcessManager(ws_manager, activity_translator)
    if settings_store is None:
        settings_store = FakeCardStore.with_default(
            provider="anthropic", model="claude-sonnet-4-20250514",
            key="sk-test-key" if keyed else "",
        )
    setup_engine = FakeSetupEngine(
        statuses if statuses is not None else default_statuses())
    platform = MagicMock()
    platform.get_capabilities.return_value = MagicMock(
        platform="macos", setup_complete=True)
    sub_agent_manager = make_sub_agent_manager()
    agent_manager = AgentManager(
        project_store=project_store, settings_store=settings_store,
        credential_store=MagicMock(), ws_manager=ws_manager,
        activity_translator=activity_translator,
        process_manager=process_manager, platform_provider=platform,
        sub_agent_manager=sub_agent_manager, setup_engine=setup_engine,
        provider_registry=ProviderRegistry(),
    )
    agents_v2.configure(
        project_store=project_store, agent_manager=agent_manager,
        ws_manager=ws_manager, sub_agent_manager=sub_agent_manager,
        setup_engine=setup_engine, settings_store=settings_store,
        credential_store=MagicMock(), trigger_manager=MagicMock(),
        provider_registry=ProviderRegistry(),
    )
    app = FastAPI()
    app.include_router(agents_v2.router)
    return TestClient(app), {
        "project_store": project_store,
        "agent_manager": agent_manager,
        "settings_store": settings_store,
        "setup_engine": setup_engine,
        "sub_agent_manager": sub_agent_manager,
        "ws_manager": ws_manager,
    }


def create_project(tc, workspace, name="kl-test", **extra):
    resp = tc.post("/api/v2/projects", json={
        "name": name, "workspace": str(workspace), **extra,
    })
    assert resp.status_code == 201, resp.text
    return resp.json()
