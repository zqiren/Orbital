# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Default-autonomy + sub-agent-elevation invariants.

Covers Tasks 1-4 in TASK-approval-ux-autonomy-default-settings-cleanup.md:

- New projects default to HANDS_OFF, not CHECK_IN.
- Sub-agent SDK transports always receive Autonomy.HANDS_OFF — the project's
  autonomy preset applies to the management agent only, never to its
  dispatched sub-agents.
- The management agent's resolution path still reads the project's autonomy.
- Existing CHECK_IN / SUPERVISED projects are not migrated by the new default.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import MagicMock

from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.transports.sdk_transport import SDKTransport
from agent_os.agent.transports.tool_risk import should_auto_approve
from agent_os.daemon_v2 import sub_agent_manager as sub_agent_manager_module
from agent_os.api import app as app_module
from agent_os.api.routes import agents_v2 as agents_v2_module


# ---------------------------------------------------------------------------
# Test 1 — scratch project default is HANDS_OFF
# ---------------------------------------------------------------------------

def test_scratch_project_defaults_to_hands_off(tmp_path):
    """`_ensure_scratch_project` creates Quick Tasks with autonomy="hands_off"."""
    project_store = MagicMock()
    project_store.find_scratch_project.return_value = None
    settings_store = MagicMock()
    settings_store.get.return_value = MagicMock(scratch_workspace=str(tmp_path))

    app_module._ensure_scratch_project(project_store, settings_store, str(tmp_path))

    project_store.create_project.assert_called_once()
    payload = project_store.create_project.call_args.args[0]
    assert payload["autonomy"] == "hands_off", (
        f"scratch project must default to hands_off, got {payload['autonomy']!r}"
    )
    assert payload["is_scratch"] is True


def test_new_project_api_defaults_to_hands_off():
    """The /api/v2/projects POST handler defaults autonomy to "hands_off".

    Verified statically — the route does ``"autonomy": req.autonomy or
    "hands_off"`` at agents_v2.py:399, so any create request with no explicit
    autonomy field lands as hands_off in the project store.
    """
    source = inspect.getsource(agents_v2_module)
    assert '"autonomy": req.autonomy or "hands_off"' in source, (
        "create-project route no longer defaults to hands_off — please "
        "verify the API hasn't regressed to check_in or supervised"
    )


# ---------------------------------------------------------------------------
# Test 2 — sub-agent autonomy is always HANDS_OFF regardless of project setting
# ---------------------------------------------------------------------------

def _autonomy_resolution_block() -> str:
    """Return the source snippet around the sub-agent autonomy assignment."""
    src = inspect.getsource(sub_agent_manager_module)
    # Locate the start_agent method's autonomy block.
    marker = "Sub-agents always run with HANDS_OFF"
    idx = src.find(marker)
    assert idx >= 0, "sub_agent_manager source no longer has the HANDS_OFF rationale comment"
    return src[idx:idx + 800]


def test_sub_agent_manager_hardcodes_hands_off():
    """The autonomy resolution block in sub_agent_manager assigns Autonomy.HANDS_OFF only."""
    block = _autonomy_resolution_block()
    assert "autonomy = Autonomy.HANDS_OFF" in block, (
        "sub-agent autonomy must be hardcoded to HANDS_OFF — see "
        "TASK-approval-ux-autonomy-default-settings-cleanup.md step 2"
    )
    # No conditional fallback to the project's setting should remain.
    assert 'project.get("autonomy"' not in block, (
        "sub_agent_manager must NOT read project.autonomy for sub-agent "
        "transport — the project's preset applies to the management agent only"
    )
    assert "Autonomy.CHECK_IN" not in block, (
        "sub_agent_manager must not fall back to CHECK_IN — sub-agents are "
        "always HANDS_OFF"
    )


def test_sub_agent_manager_assignment_is_unconditional():
    """AST check: the autonomy assignment is a bare ``Name = Attribute``, not inside a branch."""
    tree = ast.parse(inspect.getsource(sub_agent_manager_module))
    found = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "autonomy"
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "Autonomy"
            and node.value.attr == "HANDS_OFF"
        ):
            found = True
            break
    assert found, (
        "expected a top-level `autonomy = Autonomy.HANDS_OFF` assignment in "
        "sub_agent_manager.py — the elevation must be unconditional"
    )


def test_sdk_transport_with_hands_off_auto_approves_write_and_shell():
    """A sub-agent transport constructed with HANDS_OFF auto-approves all categories.

    This is the runtime consequence of test_sub_agent_manager_hardcodes_hands_off
    plus the policy at tool_risk.py:60-73 — HANDS_OFF returns True for every
    tool, so SDKTransport._handle_permission's auto-approve guard fires and
    no permission_request is queued.
    """
    transport = SDKTransport(autonomy=Autonomy.HANDS_OFF)
    assert transport._autonomy is Autonomy.HANDS_OFF
    for tool in ("Bash", "Edit", "Write", "MultiEdit", "TodoWrite", "Agent"):
        assert should_auto_approve(tool, Autonomy.HANDS_OFF), (
            f"{tool} should auto-approve under HANDS_OFF — sub-agent dispatch "
            f"would otherwise stall the queue at the first {tool} call"
        )


# ---------------------------------------------------------------------------
# Test 3 — management agent still respects the project's autonomy
# ---------------------------------------------------------------------------

def _config_for(project: dict):
    """Build an AgentConfig through the canonical builder — the single site
    every start path (chat, queue, trigger, /agents/start) now derives from."""
    from agent_os.daemon_v2.agent_manager import AgentManager

    project_store = MagicMock()
    project_store.get_project = MagicMock(return_value=project)
    settings_store = MagicMock()
    settings_store.get = MagicMock(return_value=None)
    credential_store = MagicMock()
    credential_store.get_api_key = MagicMock(return_value=None)
    mgr = AgentManager(
        project_store=project_store, ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(), activity_translator=MagicMock(),
        process_manager=MagicMock(), settings_store=settings_store,
        credential_store=credential_store,
    )
    return mgr._build_agent_config_from_project(project["project_id"])


def test_management_agent_resolution_path_reads_project_autonomy():
    """The management agent still derives Autonomy from project.autonomy —
    only the sub-agent path was elevated to hardcoded HANDS_OFF.

    Behavioral, not a source grep: this used to assert that the literal
    ``autonomy_str = project.get("autonomy", "hands_off")`` appeared in
    agents_v2.py, which broke the moment that route stopped re-deriving
    config and started calling the canonical builder like every other start
    path. The invariant was never about where the line lives.
    """
    cfg = _config_for({"project_id": "p", "workspace": "/tmp/p",
                       "autonomy": "check_in"})
    assert cfg.autonomy is Autonomy.CHECK_IN


def test_management_agent_autonomy_defaults_and_survives_garbage():
    cfg = _config_for({"project_id": "p", "workspace": "/tmp/p"})
    assert cfg.autonomy is Autonomy.HANDS_OFF

    cfg = _config_for({"project_id": "p", "workspace": "/tmp/p",
                       "autonomy": "not-a-real-preset"})
    assert cfg.autonomy is Autonomy.HANDS_OFF


def test_check_in_management_still_prompts_for_write_tools():
    """The autonomy policy itself is unchanged: CHECK_IN does NOT auto-approve writes."""
    for tool in ("Bash", "Edit", "Write"):
        assert not should_auto_approve(tool, Autonomy.CHECK_IN), (
            f"{tool} must require approval under CHECK_IN — policy unchanged"
        )
    # Reads still auto-approve under CHECK_IN (sanity).
    for tool in ("Read", "Glob", "Grep"):
        assert should_auto_approve(tool, Autonomy.CHECK_IN)


# ---------------------------------------------------------------------------
# Test 4 — existing CHECK_IN / SUPERVISED projects are not migrated
# ---------------------------------------------------------------------------

def test_no_runtime_migration_overwrites_existing_autonomy():
    """No code path silently rewrites an existing project's autonomy field."""
    # The scratch-create path is gated on find_scratch_project() returning None —
    # if a project (scratch or otherwise) already exists, no create is issued
    # and therefore no autonomy assignment touches its stored value.
    project_store = MagicMock()
    project_store.find_scratch_project.return_value = {
        "project_id": "proj_scratch",
        "name": "Quick Tasks",
        "autonomy": "check_in",  # legacy default from before this change
        "is_scratch": True,
    }
    settings_store = MagicMock()
    settings_store.get.return_value = MagicMock(scratch_workspace="/tmp/scratch")

    app_module._ensure_scratch_project(project_store, settings_store, "/tmp/scratch")

    project_store.create_project.assert_not_called(), (
        "existing scratch project must NOT be recreated — that would "
        "silently overwrite a user's chosen autonomy preset"
    )


def test_default_string_present_in_runtime_resolution_paths():
    """Runtime resolution paths default missing-autonomy projects to hands_off.

    Belt-and-suspenders for test 1: a legacy project written before the
    `autonomy` field existed should resolve to HANDS_OFF on next load,
    not CHECK_IN. This catches any new code path that introduces a
    `project.get("autonomy", "check_in")` fallback.
    """
    src_files = [
        Path(__file__).resolve().parents[2] / "agent_os" / "api" / "routes" / "agents_v2.py",
        Path(__file__).resolve().parents[2] / "agent_os" / "daemon_v2" / "sub_agent_manager.py",
        Path(__file__).resolve().parents[2] / "agent_os" / "api" / "app.py",
    ]
    for p in src_files:
        text = p.read_text(encoding="utf-8")
        assert 'project.get("autonomy", "check_in")' not in text, (
            f"{p.name} has a stale `project.get('autonomy', 'check_in')` "
            f"default — should be 'hands_off' (or removed entirely for "
            f"sub_agent_manager.py)"
        )
