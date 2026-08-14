# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration tests for the sub-agent inheritance flow.

Exercises the full chain through SubAgentManager._start_from_registry:

  1. test_full_inheritance_dispatch_flow — Layer-1 files seeded, sub-agent
     started; verify (a) prompt is rendered with correct paths, (b)
     MEMORY.md is created, (c) transport receives the system prompt.
  2. test_memory_md_persistence_across_dispatches — Append to MEMORY.md
     between dispatches; verify content survives.
  3. test_other_sub_agents_memory_visible — Pre-write codex MEMORY.md;
     dispatch claude-code; verify claude-code's prompt references codex
     MEMORY.md path.

These tests stub the actual transport.start() so no real claude.exe or
LLM is invoked. They DO exercise the real SubAgentManager, registry,
setup engine, project store, and prompt renderer.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Explicit parent session id: SubAgentManager hard-raises on session_id=None
# since the "default" sentinel retirement (433912a — seam 3 / D1).
SID = "proj_1_sess0001"

from agent_os.agent.project_paths import ProjectPaths
from agent_os.agent.sub_agent_prompt import (
    MEMORY_MD_HEADER,
    ensure_memory_md,
    render_sub_agent_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    paths = ProjectPaths(str(ws))
    os.makedirs(paths.orbital_dir, exist_ok=True)
    os.makedirs(paths.instructions_dir, exist_ok=True)
    return str(ws)


@pytest.fixture
def seeded_workspace(workspace):
    """Pre-populate orbital layer-1 files with realistic content."""
    paths = ProjectPaths(workspace)
    with open(paths.project_state, "w", encoding="utf-8") as f:
        f.write("# Project State\n\nWe are working on the chat refactor.\n")
    with open(paths.decisions, "w", encoding="utf-8") as f:
        f.write("# Decisions\n\n- Use SDK transport for sub-agents.\n")
    with open(paths.lessons, "w", encoding="utf-8") as f:
        f.write("# Lessons\n\n- Always check token limits.\n")
    with open(paths.index, "w", encoding="utf-8") as f:
        f.write("# Index\n\nReference: spec PROMPT-TEMPLATE.\n")
    with open(paths.user_directives, "w", encoding="utf-8") as f:
        f.write("- always run pytest first\n")
    return workspace


def _make_manager(workspace: str, ws_manager=None):
    """Build a SubAgentManager wired to a real registry + setup_engine,
    project_store stubs sufficient for _start_from_registry.
    """
    from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
    from agent_os.agents.registry import AgentRegistry
    from agent_os.agents.manifest import (
        AgentManifest,
        ManifestRuntime,
        ManifestSetup,
        ManifestCapabilities,
        ManifestPermissions,
    )

    registry = AgentRegistry()
    # Fabricate a claude-code manifest so _start_from_registry succeeds
    registry.register(AgentManifest(
        manifest_version="1",
        name="Claude Code",
        slug="claude-code",
        description="test",
        author="test",
        version="1.0.0",
        runtime=ManifestRuntime(adapter="cli", mode="pipe", transport="sdk"),
        setup=ManifestSetup(),
        capabilities=ManifestCapabilities(),
        permissions=ManifestPermissions(),
    ))
    registry.register(AgentManifest(
        manifest_version="1",
        name="Codex",
        slug="codex",
        description="test",
        author="test",
        version="1.0.0",
        runtime=ManifestRuntime(adapter="cli", mode="pipe", transport="sdk"),
        setup=ManifestSetup(),
        capabilities=ManifestCapabilities(),
        permissions=ManifestPermissions(),
    ))

    # Setup engine: returns claude-code as installed
    setup_engine = MagicMock()
    setup_engine.get_adapter_config = MagicMock(return_value={
        "command": "claude",
        "workspace": workspace,
        "approval_patterns": [],
        "env": {},
        "args": [],
        "network_domains": [],
    })
    available_codex = MagicMock(slug="codex", installed=True)
    available_claude = MagicMock(slug="claude-code", installed=True)
    setup_engine.check_all = MagicMock(return_value=[available_claude, available_codex])

    # Project store
    project_store = MagicMock()
    project_store.get_project = MagicMock(return_value={
        "workspace": workspace,
        "autonomy": "check_in",
        "enabled_sub_agents": ["claude-code", "codex"],
    })

    pm = MagicMock()
    pm.start = AsyncMock()
    pm.stop = AsyncMock()
    pm._ws = ws_manager or MagicMock()

    return SubAgentManager(
        process_manager=pm,
        registry=registry,
        setup_engine=setup_engine,
        project_store=project_store,
        ws_manager=ws_manager,
    )


# ---------------------------------------------------------------------------
# 1. Full inheritance dispatch flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_inheritance_dispatch_flow(seeded_workspace):
    """Boot SubAgentManager with seeded project; dispatch claude-code; verify
    prompt rendering, MEMORY.md creation, and transport receives the prompt."""
    workspace = seeded_workspace
    paths = ProjectPaths(workspace)

    captured: dict = {}

    # Patch SDKTransport so we capture the system_prompt without spawning claude.
    class FakeTransport:
        def __init__(self, autonomy=None, system_prompt=None, **kwargs):  # kwargs: resume_session_id/model (006bcec, 7724315)
            captured["system_prompt"] = system_prompt
            captured["autonomy"] = autonomy
            self._system_prompt = system_prompt

        async def start(self, command, args, workspace, env=None):
            captured["start_workspace"] = workspace
            captured["start_command"] = command

        async def stop(self):
            pass

        def is_alive(self):
            return True

        def is_idle(self):
            return False

    mgr = _make_manager(workspace)

    with patch(
        "agent_os.agent.transports.sdk_transport.SDKTransport",
        FakeTransport,
    ), patch(
        "agent_os.agent.transports.sdk_transport.HAS_SDK",
        True,
    ):
        result = await mgr._start_from_registry("proj_1", "claude-code", session_id=SID)

    # Adapter started successfully
    assert "Started" in result, result

    # MEMORY.md created at the canonical path with the prescribed header
    memory_path = os.path.join(paths.sub_agent_dir("claude-code"), "MEMORY.md")
    assert os.path.isfile(memory_path)
    with open(memory_path, "r", encoding="utf-8") as f:
        assert f.read() == MEMORY_MD_HEADER.format(agent_slug="claude-code")

    # Transport received the rendered system prompt
    sp = captured.get("system_prompt")
    assert sp is not None and isinstance(sp, str)
    assert paths.project_state in sp
    assert paths.decisions in sp
    assert paths.lessons in sp
    assert paths.index in sp
    assert "SESSION_LOG.md" not in sp  # retired in the Layer-1 memory redesign
    assert paths.instructions_dir in sp
    assert memory_path in sp


# ---------------------------------------------------------------------------
# 2. MEMORY.md persistence across dispatches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_md_persistence_across_dispatches(seeded_workspace):
    workspace = seeded_workspace
    memory_path = ensure_memory_md(workspace, "claude-code")

    # Agent appended a learning between dispatches
    with open(memory_path, "a", encoding="utf-8") as f:
        f.write("\n## 2026-05-09\n- Always read PROJECT_STATE first.\n")

    captured_prompts: list[str] = []

    class FakeTransport:
        def __init__(self, autonomy=None, system_prompt=None, **kwargs):  # kwargs: resume_session_id/model (006bcec, 7724315)
            captured_prompts.append(system_prompt)
            self._system_prompt = system_prompt

        async def start(self, command, args, workspace, env=None):
            pass

        async def stop(self):
            pass

        def is_alive(self):
            return True

        def is_idle(self):
            return False

    mgr = _make_manager(workspace)

    with patch(
        "agent_os.agent.transports.sdk_transport.SDKTransport",
        FakeTransport,
    ), patch(
        "agent_os.agent.transports.sdk_transport.HAS_SDK",
        True,
    ):
        # Dispatch #1 — completes
        await mgr._start_from_registry("proj_1", "claude-code", session_id=SID)
        # Stop and re-dispatch (simulate a second turn)
        await mgr.stop("proj_1", "claude-code", session_id=SID)
        await mgr._start_from_registry("proj_1", "claude-code", session_id=SID)

    # MEMORY.md content was preserved
    with open(memory_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Always read PROJECT_STATE first." in content
    assert "# MEMORY for claude-code" in content


# ---------------------------------------------------------------------------
# 3. Other sub-agents' memory is visible in the rendered prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_other_sub_agents_memory_visible(seeded_workspace):
    workspace = seeded_workspace

    # Pre-create codex MEMORY.md with content directly on disk
    codex_path = ensure_memory_md(workspace, "codex")
    with open(codex_path, "a", encoding="utf-8") as f:
        f.write("- codex finds Python imports tricky\n")

    captured: dict = {}

    class FakeTransport:
        def __init__(self, autonomy=None, system_prompt=None, **kwargs):  # kwargs: resume_session_id/model (006bcec, 7724315)
            captured["system_prompt"] = system_prompt

        async def start(self, command, args, workspace, env=None):
            pass

        async def stop(self):
            pass

        def is_alive(self):
            return True

        def is_idle(self):
            return False

    mgr = _make_manager(workspace)
    with patch(
        "agent_os.agent.transports.sdk_transport.SDKTransport",
        FakeTransport,
    ), patch(
        "agent_os.agent.transports.sdk_transport.HAS_SDK",
        True,
    ):
        await mgr._start_from_registry("proj_1", "claude-code", session_id=SID)

    sp = captured["system_prompt"]
    assert sp is not None
    # codex MEMORY.md path appears in the "Other sub-agents" section
    assert codex_path in sp
    assert "Other sub-agents in this project" in sp


# ---------------------------------------------------------------------------
# Bonus: Re-dispatch must re-render the prompt (no caching)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_is_freshly_rendered_per_dispatch(seeded_workspace):
    """Adding a peer's MEMORY.md between dispatches must surface it in the
    second dispatch's prompt — proving the renderer is not cached."""
    workspace = seeded_workspace
    captured_prompts: list[str] = []

    class FakeTransport:
        def __init__(self, autonomy=None, system_prompt=None, **kwargs):  # kwargs: resume_session_id/model (006bcec, 7724315)
            captured_prompts.append(system_prompt)

        async def start(self, command, args, workspace, env=None):
            pass

        async def stop(self):
            pass

        def is_alive(self):
            return True

        def is_idle(self):
            return False

    mgr = _make_manager(workspace)
    with patch(
        "agent_os.agent.transports.sdk_transport.SDKTransport",
        FakeTransport,
    ), patch(
        "agent_os.agent.transports.sdk_transport.HAS_SDK",
        True,
    ):
        await mgr._start_from_registry("proj_1", "claude-code", session_id=SID)
        await mgr.stop("proj_1", "claude-code", session_id=SID)

        # Now create codex MEMORY.md
        codex_path = ensure_memory_md(workspace, "codex")
        await mgr._start_from_registry("proj_1", "claude-code", session_id=SID)

    assert len(captured_prompts) == 2
    # First dispatch must NOT mention codex memory (didn't exist yet)
    assert codex_path not in captured_prompts[0]
    # Second dispatch MUST mention it
    assert codex_path in captured_prompts[1]


# ---------------------------------------------------------------------------
# 4. Transport-aware MEMORY.md creation (Track F3 / dispatch 2026-05-20 §4)
#
# PTY and ACP transports drop system_prompt entirely (Track D investigation
# 2026-05-20). Creating MEMORY.md for those sub-agents produces orphan stub
# files the sub-agent never learns about. The dispatch decision: skip the
# create when the manifest's transport doesn't forward system_prompt.
# ---------------------------------------------------------------------------


def _make_manager_with_transport(workspace: str, transport: str, mode: str = "interactive"):
    """Variant of _make_manager with a configurable transport for the
    primary 'claude-code' manifest. Used to exercise PTY/ACP paths."""
    from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
    from agent_os.agents.registry import AgentRegistry
    from agent_os.agents.manifest import (
        AgentManifest,
        ManifestRuntime,
        ManifestSetup,
        ManifestCapabilities,
        ManifestPermissions,
    )

    registry = AgentRegistry()
    registry.register(AgentManifest(
        manifest_version="1",
        name="Claude Code",
        slug="claude-code",
        description="test",
        author="test",
        version="1.0.0",
        runtime=ManifestRuntime(adapter="cli", mode=mode, transport=transport),
        setup=ManifestSetup(),
        capabilities=ManifestCapabilities(),
        permissions=ManifestPermissions(),
    ))

    setup_engine = MagicMock()
    setup_engine.get_adapter_config = MagicMock(return_value={
        "command": "claude",
        "workspace": workspace,
        "approval_patterns": [],
        "env": {},
        "args": [],
        "network_domains": [],
    })
    available_claude = MagicMock(slug="claude-code", installed=True)
    setup_engine.check_all = MagicMock(return_value=[available_claude])

    project_store = MagicMock()
    project_store.get_project = MagicMock(return_value={
        "workspace": workspace,
        "autonomy": "check_in",
        "enabled_sub_agents": ["claude-code"],
    })

    pm = MagicMock()
    pm.start = AsyncMock()
    pm.stop = AsyncMock()

    return SubAgentManager(
        process_manager=pm,
        registry=registry,
        setup_engine=setup_engine,
        project_store=project_store,
        ws_manager=MagicMock(),
    )


@pytest.mark.asyncio
async def test_pty_transport_does_not_create_memory_md(seeded_workspace):
    """PTY-transport sub-agents do NOT receive system_prompt (Track D), so
    Orbital should NOT lazily create their MEMORY.md stub at dispatch time.
    Regression target: orphan stub files left in the workspace."""
    workspace = seeded_workspace
    paths = ProjectPaths(workspace)
    memory_path = os.path.join(paths.sub_agent_dir("claude-code"), "MEMORY.md")
    assert not os.path.exists(memory_path), "precondition: memory stub absent"

    class FakePTYTransport:
        def __init__(self, approval_patterns=None):
            pass

        async def start(self, command, args, workspace, env=None):
            pass

        async def stop(self):
            pass

        def is_alive(self):
            return True

        def is_idle(self):
            return False

    mgr = _make_manager_with_transport(workspace, transport="pty")

    with patch(
        "agent_os.agent.transports.pty_transport.PTYTransport",
        FakePTYTransport,
    ):
        result = await mgr._start_from_registry("proj_1", "claude-code", session_id=SID)

    assert "Started" in result, result
    # Critical assertion: no orphan MEMORY.md stub was created.
    assert not os.path.exists(memory_path), (
        f"PTY transport should NOT create MEMORY.md stub; "
        f"file unexpectedly exists at {memory_path}"
    )


@pytest.mark.asyncio
async def test_acp_sdk_transport_does_not_create_memory_md(seeded_workspace):
    """ACP-SDK sub-agents also drop system_prompt today. Same rule as PTY:
    skip the orphan stub creation."""
    from agent_os.agents.manifest import AgentManifest, ManifestRuntime, ManifestSetup, ManifestCapabilities, ManifestPermissions
    from agent_os.agents.registry import AgentRegistry
    from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

    workspace = seeded_workspace
    paths = ProjectPaths(workspace)
    memory_path = os.path.join(paths.sub_agent_dir("gemini-cli"), "MEMORY.md")
    assert not os.path.exists(memory_path), "precondition: memory stub absent"

    registry = AgentRegistry()
    registry.register(AgentManifest(
        manifest_version="1",
        name="Gemini CLI",
        slug="gemini-cli",
        description="test",
        author="test",
        version="1.0.0",
        runtime=ManifestRuntime(adapter="cli", mode="interactive", transport="acp-sdk", command="gemini"),
        setup=ManifestSetup(),
        capabilities=ManifestCapabilities(),
        permissions=ManifestPermissions(),
    ))
    setup_engine = MagicMock()
    setup_engine.get_adapter_config = MagicMock(return_value={
        "command": "gemini",
        "workspace": workspace,
        "approval_patterns": [],
        "env": {},
        "args": [],
        "network_domains": [],
    })
    setup_engine.check_all = MagicMock(return_value=[MagicMock(slug="gemini-cli", installed=True)])
    project_store = MagicMock()
    project_store.get_project = MagicMock(return_value={
        "workspace": workspace,
        "autonomy": "check_in",
        "enabled_sub_agents": ["gemini-cli"],
    })
    pm = MagicMock()
    pm.start = AsyncMock()
    pm.stop = AsyncMock()
    mgr = SubAgentManager(
        process_manager=pm,
        registry=registry,
        setup_engine=setup_engine,
        project_store=project_store,
        ws_manager=MagicMock(),
    )

    class FakeACPSDKTransport:
        def __init__(self, *, resume_record=None):
            pass

        async def start(self, command, args, workspace, env=None):
            pass

        async def stop(self):
            pass

        def is_alive(self):
            return True

        def is_idle(self):
            return False

    with patch(
        "agent_os.agent.transports.acp_sdk_transport.ACPSDKTransport",
        FakeACPSDKTransport,
    ):
        result = await mgr._start_from_registry("proj_1", "gemini-cli", session_id=SID)

    assert "Started" in result, result
    assert not os.path.exists(memory_path), (
        f"ACP transport should NOT create MEMORY.md stub; "
        f"file unexpectedly exists at {memory_path}"
    )


@pytest.mark.asyncio
async def test_sdk_transport_still_creates_memory_md(seeded_workspace):
    """Positive control: SDK-transport sub-agents (claude-code today) still
    receive their MEMORY.md stub at dispatch. The F3 change must not
    regress the existing behavior on the forwarding-transport path."""
    workspace = seeded_workspace
    paths = ProjectPaths(workspace)
    memory_path = os.path.join(paths.sub_agent_dir("claude-code"), "MEMORY.md")
    assert not os.path.exists(memory_path), "precondition: memory stub absent"

    class FakeSDKTransport:
        def __init__(self, autonomy=None, system_prompt=None, **kwargs):  # kwargs: resume_session_id/model (006bcec, 7724315)
            self._system_prompt = system_prompt

        async def start(self, command, args, workspace, env=None):
            pass

        async def stop(self):
            pass

        def is_alive(self):
            return True

        def is_idle(self):
            return False

    mgr = _make_manager_with_transport(workspace, transport="sdk", mode="pipe")

    with patch(
        "agent_os.agent.transports.sdk_transport.SDKTransport",
        FakeSDKTransport,
    ), patch(
        "agent_os.agent.transports.sdk_transport.HAS_SDK",
        True,
    ):
        result = await mgr._start_from_registry("proj_1", "claude-code", session_id=SID)

    assert "Started" in result, result
    assert os.path.isfile(memory_path), (
        f"SDK transport should create MEMORY.md stub at dispatch; "
        f"file missing at {memory_path}"
    )
    with open(memory_path, "r", encoding="utf-8") as f:
        assert f.read() == MEMORY_MD_HEADER.format(agent_slug="claude-code")
