# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Portal scoping (Spec 12 §2b — the leak fix).

Portals used to be a single flat dict on the provider singleton, baked into
*every* Seatbelt profile. Once projects A and B had both started, A's shell
could read/write B's workspace. Portals are now keyed by the *owning workspace
realpath* (the scope); a profile is built from that scope's portals ONLY.

These tests assert the isolation property at the profile-assembly seam without
spawning ``sandbox-exec`` (safe to run in the unit suite).
"""

import os
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="macOS Seatbelt profile assembly; Windows uses a separate ACL model",
)


def _provider():
    from agent_os.platform.macos.provider import MacOSPlatformProvider
    return MacOSPlatformProvider()


def test_profile_excludes_other_scopes_portals(tmp_path):
    """A profile built for scope A must not carry scope B's portals."""
    from agent_os.platform.macos.sandbox import generate_profile

    ws_a = str(tmp_path / "wsA")
    ws_b = str(tmp_path / "wsB")
    os.makedirs(ws_a)
    os.makedirs(ws_b)

    provider = _provider()
    provider.grant_folder_access(ws_a, "read_write", scope=ws_a)
    provider.grant_folder_access(ws_b, "read_write", scope=ws_b)

    portals_a = provider._portals_for_working_dir(ws_a)
    profile_a = generate_profile(ws_a, portal_paths=portals_a)

    assert os.path.realpath(ws_a) in profile_a
    assert os.path.realpath(ws_b) not in profile_a


def test_read_mode_portal_renders_read_only(tmp_path):
    """A ``read`` portal grants file-read* and never allows file-write* on it."""
    from agent_os.platform.macos.sandbox import generate_profile

    portal = str(tmp_path / "ref")
    os.makedirs(portal)
    real = os.path.realpath(portal)

    profile = generate_profile(str(tmp_path / "ws"), portal_paths={portal: "read"})

    assert f'(allow file-read* (subpath "{real}"))' in profile
    assert f'(allow file-write* (subpath "{real}"))' not in profile


def test_revoke_only_affects_own_scope(tmp_path):
    """Revoking a path in scope A leaves the same path granted under scope B."""
    ws_a = str(tmp_path / "wsA")
    ws_b = str(tmp_path / "wsB")
    shared = str(tmp_path / "shared")
    for d in (ws_a, ws_b, shared):
        os.makedirs(d)

    provider = _provider()
    provider.grant_folder_access(shared, "read", scope=ws_a)
    provider.grant_folder_access(shared, "read", scope=ws_b)

    provider.revoke_folder_access(shared, scope=ws_a)

    assert shared not in provider._portals_for_working_dir(ws_a)
    assert shared in provider._portals_for_working_dir(ws_b)


def test_working_dir_always_gets_read_write(tmp_path):
    """The working directory itself is always a read_write portal in its profile."""
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    provider = _provider()
    portals = provider._portals_for_working_dir(ws)
    assert portals.get(ws) == "read_write"


@pytest.mark.asyncio
async def test_request_access_grant_is_scoped(tmp_path):
    """approve() of a request_access tool grants scoped to the project's workspace."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    from agent_os.daemon_v2.agent_manager import AgentManager
    from agent_os.platform.types import PermissionResult

    workspace = str(tmp_path / "proj_ws")
    os.makedirs(workspace)

    provider = MagicMock()
    provider.grant_folder_access.return_value = PermissionResult(success=True, path="/x")

    mgr = AgentManager(
        project_store=MagicMock(),
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
        platform_provider=provider,
    )

    sid = "proj_1_sess0001"
    pending = {
        "tool_name": "request_access",
        "tool_args": {"path": "/some/target", "reason": "r", "access_type": "read"},
    }
    session = MagicMock()
    session.has_result_for.return_value = False
    interceptor = MagicMock()
    interceptor.get_pending.return_value = pending
    task = asyncio.get_event_loop().create_future()
    task.set_result(None)
    handle = MagicMock(
        session=session, task=task, interceptor=interceptor,
        config_snapshot={"workspace": workspace},
    )
    mgr._handles[("proj_1", sid)] = handle

    with patch.object(mgr, "_start_loop", new_callable=AsyncMock):
        await mgr.approve("proj_1", "tc_1", session_id=sid)

    provider.grant_folder_access.assert_called_once_with(
        "/some/target", "read_only", scope=os.path.realpath(workspace)
    )
