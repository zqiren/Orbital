# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for sub-agent context inheritance.

Covers:
  1. render_sub_agent_prompt — content of the rendered prompt
  2. MEMORY.md lazy creation lifecycle
  3. SDKTransport / PipeTransport system_prompt forwarding
  4. Workspace CLAUDE.md interference detection + WS banner

Mirrors the spec's "regression tests required" list.
"""

import asyncio
import os
import subprocess
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.project_paths import ProjectPaths
from agent_os.agent.sub_agent_prompt import (
    MEMORY_MD_HEADER,
    detect_claudemd_conflict,
    ensure_memory_md,
    hash_claudemd,
    render_sub_agent_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path):
    """Provide a fresh workspace directory for each test."""
    ws = tmp_path / "ws"
    ws.mkdir()
    # Pre-create the orbital subtree so paths exist for some tests
    paths = ProjectPaths(str(ws))
    os.makedirs(paths.orbital_dir, exist_ok=True)
    return str(ws)


# ---------------------------------------------------------------------------
# 1-3. render_sub_agent_prompt
# ---------------------------------------------------------------------------


class TestRenderSubAgentPromptPaths:
    def test_render_sub_agent_prompt_includes_all_required_paths(self, workspace):
        """All authoritative file paths must appear in the rendered prompt."""
        paths = ProjectPaths(workspace)
        prompt = render_sub_agent_prompt(
            workspace=workspace,
            namespace=None,
            agent_slug="claude-code",
            enabled_sub_agents=["claude-code"],
        )

        # Paths that must appear
        assert paths.project_state in prompt
        assert paths.decisions in prompt
        assert paths.lessons in prompt
        # SESSION_LOG.md was retired in the Layer-1 memory redesign; the
        # navigation file CONTEXT.md was renamed to INDEX.md (paths.index).
        assert paths.index in prompt
        assert paths.instructions_dir in prompt
        assert paths.skills_dir in prompt

        # The agent's own MEMORY.md path
        own_memory = os.path.join(
            paths.sub_agent_dir("claude-code"), "MEMORY.md"
        )
        assert own_memory in prompt

    def test_render_sub_agent_prompt_omits_other_agents_section_when_alone(
        self, workspace
    ):
        """When only the current agent is enabled, the 'Other sub-agents'
        section MUST be omitted (not rendered as an empty list)."""
        prompt = render_sub_agent_prompt(
            workspace=workspace,
            namespace=None,
            agent_slug="claude-code",
            enabled_sub_agents=["claude-code"],
        )
        assert "Other sub-agents in this project" not in prompt

    def test_render_sub_agent_prompt_omits_section_when_peer_has_no_memory(
        self, workspace
    ):
        """If a peer is enabled but has no MEMORY.md on disk, it must NOT
        appear in the rendered prompt (avoid steering toward Read calls
        that would fail)."""
        prompt = render_sub_agent_prompt(
            workspace=workspace,
            namespace=None,
            agent_slug="claude-code",
            enabled_sub_agents=["claude-code", "codex"],
        )
        # codex MEMORY.md does not exist → no section
        assert "Other sub-agents in this project" not in prompt
        codex_memory = os.path.join(
            ProjectPaths(workspace).sub_agent_dir("codex"), "MEMORY.md"
        )
        assert codex_memory not in prompt

    def test_render_sub_agent_prompt_includes_other_agents_memory_paths(
        self, workspace
    ):
        """Peers that have an existing MEMORY.md must appear in the
        'Other sub-agents' section."""
        # Create codex MEMORY.md (peer)
        codex_path = ensure_memory_md(workspace, "codex")
        assert os.path.isfile(codex_path)

        prompt = render_sub_agent_prompt(
            workspace=workspace,
            namespace=None,
            agent_slug="claude-code",
            enabled_sub_agents=["claude-code", "codex"],
        )
        assert "Other sub-agents in this project" in prompt
        assert codex_path in prompt

    def test_render_handles_empty_enabled_list(self, workspace):
        """Empty enabled list still produces a valid prompt."""
        prompt = render_sub_agent_prompt(
            workspace=workspace,
            namespace=None,
            agent_slug="claude-code",
            enabled_sub_agents=[],
        )
        assert "claude-code" in prompt
        assert "Other sub-agents in this project" not in prompt

    def test_skills_index_renders_when_skills_present(self, workspace):
        paths = ProjectPaths(workspace)
        os.makedirs(paths.skills_dir, exist_ok=True)
        # Top-level .md skill
        with open(os.path.join(paths.skills_dir, "alpha.md"), "w") as f:
            f.write("# alpha")
        # Subdir-style skill
        beta_dir = os.path.join(paths.skills_dir, "beta")
        os.makedirs(beta_dir, exist_ok=True)
        with open(os.path.join(beta_dir, "SKILL.md"), "w") as f:
            f.write("# beta")

        prompt = render_sub_agent_prompt(
            workspace=workspace,
            namespace=None,
            agent_slug="claude-code",
            enabled_sub_agents=["claude-code"],
        )
        assert "Available skills:" in prompt
        assert "alpha.md" in prompt
        assert "beta/SKILL.md" in prompt or "beta\\SKILL.md" in prompt

    def test_skills_index_renders_none_when_no_skills(self, workspace):
        prompt = render_sub_agent_prompt(
            workspace=workspace,
            namespace=None,
            agent_slug="claude-code",
            enabled_sub_agents=["claude-code"],
        )
        assert "Available skills: (none)" in prompt

    def test_render_sub_agent_prompt_contains_no_newlines(self, workspace):
        """Regression guard against the Windows claude.exe argv-newline hang.

        Any `\\n` in the string passed via ClaudeAgentOptions.system_prompt
        (preset/append dict) is forwarded to claude.exe's
        ``--append-system-prompt`` argv value, which on Windows v2.1.138 under
        streaming SDK mode causes a 60s+ silent stdout hang. The renderer must
        emit a single-line prompt so no newline ever reaches argv.
        See docs/investigations/TRIGGER-orbital-prompt-content.md.
        """
        # Realistic inputs: peer with existing MEMORY.md (so the
        # "Other sub-agents" block renders), skills present.
        paths = ProjectPaths(workspace)
        os.makedirs(paths.skills_dir, exist_ok=True)
        with open(os.path.join(paths.skills_dir, "alpha.md"), "w") as f:
            f.write("# alpha")
        ensure_memory_md(workspace, "codex")  # peer w/ MEMORY.md

        prompt = render_sub_agent_prompt(
            workspace=workspace,
            namespace=None,
            agent_slug="claude-code",
            enabled_sub_agents=["claude-code", "codex"],
        )
        assert "\n" not in prompt, (
            f"render_sub_agent_prompt produced a newline-containing string "
            f"(len={len(prompt)}, first newline at "
            f"{prompt.find(chr(10))}); see "
            f"docs/investigations/TRIGGER-orbital-prompt-content.md"
        )


# ---------------------------------------------------------------------------
# 4-5. MEMORY.md lifecycle
# ---------------------------------------------------------------------------


class TestMemoryMdLifecycle:
    def test_ensure_memory_md_creates_with_header(self, workspace):
        path = ensure_memory_md(workspace, "claude-code")
        assert os.path.isfile(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert content == MEMORY_MD_HEADER.format(agent_slug="claude-code")
        assert "# MEMORY for claude-code" in content

    def test_ensure_memory_md_idempotent_preserves_existing(self, workspace):
        path = ensure_memory_md(workspace, "claude-code")
        # Append user content
        with open(path, "a", encoding="utf-8") as f:
            f.write("MY CUSTOM ENTRY\n")

        # Call ensure again — must NOT overwrite
        again = ensure_memory_md(workspace, "claude-code")
        assert again == path
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "MY CUSTOM ENTRY" in content

    def test_ensure_memory_md_creates_parent_dir(self, tmp_path):
        # Workspace exists but no orbital tree
        ws = tmp_path / "fresh"
        ws.mkdir()
        path = ensure_memory_md(str(ws), "claude-code")
        assert os.path.isfile(path)
        assert "sub_agents" in path


# ---------------------------------------------------------------------------
# 6. SDKTransport.system_prompt forwarding
# ---------------------------------------------------------------------------


class TestSDKTransportSystemPrompt:
    @pytest.mark.asyncio
    async def test_sdk_transport_passes_system_prompt_as_preset_append_dict(self):
        """system_prompt is forwarded as a preset/append dict, NOT a plain string.

        Plain-string `system_prompt` maps to claude-agent-sdk's
        ``--system-prompt <text>`` (REPLACE) which stalls claude.exe's
        ``Query.initialize`` for the full 60s timeout on Windows
        (Tier 3 trace W1 classification, ``TRACE-windows-dispatch-bug.md``).
        The dict shape ``{type:preset, preset:claude_code, append:<text>}``
        maps to ``--append-system-prompt <text>``, which preserves
        claude-code's default system prompt and avoids the stall.
        """
        from agent_os.agent.transports.sdk_transport import SDKTransport

        transport = SDKTransport(system_prompt="YOU ARE A SUB-AGENT")
        with patch(
            "agent_os.agent.transports.sdk_transport.ClaudeSDKClient"
        ) as MockClient, patch(
            "agent_os.agent.transports.sdk_transport.ClaudeAgentOptions"
        ) as MockOptions:
            MockClient.return_value.connect = AsyncMock()
            await transport.start("claude", [], "/workspace")

            opts_kwargs = MockOptions.call_args[1]
            sp = opts_kwargs.get("system_prompt")
            assert isinstance(sp, dict), f"expected dict, got {type(sp).__name__}: {sp!r}"
            assert sp.get("type") == "preset"
            assert sp.get("preset") == "claude_code"
            assert sp.get("append") == "YOU ARE A SUB-AGENT"

    @pytest.mark.asyncio
    async def test_sdk_transport_omits_system_prompt_when_none(self):
        from agent_os.agent.transports.sdk_transport import SDKTransport

        transport = SDKTransport()  # no system_prompt
        with patch(
            "agent_os.agent.transports.sdk_transport.ClaudeSDKClient"
        ) as MockClient, patch(
            "agent_os.agent.transports.sdk_transport.ClaudeAgentOptions"
        ) as MockOptions:
            MockClient.return_value.connect = AsyncMock()
            await transport.start("claude", [], "/workspace")
            opts_kwargs = MockOptions.call_args[1]
            assert "system_prompt" not in opts_kwargs


# ---------------------------------------------------------------------------
# 7. PipeTransport writes system_prompt file + flag + cleans up
# ---------------------------------------------------------------------------


class TestPipeTransportSystemPrompt:
    @pytest.mark.asyncio
    async def test_pipe_transport_writes_system_prompt_file(self, workspace):
        from agent_os.agent.transports.pipe_transport import (
            PipeTransport,
            PipeTransportConfig,
        )

        transport = PipeTransport(
            config=PipeTransportConfig(prompt_flag="-p"),
            system_prompt="HELLO SYSTEM",
            agent_slug="claude-code",
        )
        await transport.start("claude", ["--verbose"], workspace)

        captured_path: list[str] = []

        def fake_run(cmd, **kwargs):
            # Capture --append-system-prompt-file <path> from argv
            for i, a in enumerate(cmd):
                if a == "--append-system-prompt-file":
                    captured_path.append(cmd[i + 1])
            # Verify the file actually exists at this point
            for p in captured_path:
                assert os.path.isfile(p), f"temp file missing during dispatch: {p}"
                with open(p, "r", encoding="utf-8") as f:
                    assert f.read() == "HELLO SYSTEM"
            mock = MagicMock()
            mock.returncode = 0
            mock.stdout = (
                b'{"type":"assistant","message":{"content":[{"type":"text","text":"ok"}]}}\n'
            )
            mock.stderr = b""
            return mock

        with patch(
            "agent_os.agent.transports.pipe_transport.subprocess.run",
            side_effect=fake_run,
        ):
            response = await transport.send("hi")

        assert response == "ok"
        assert captured_path, "expected --append-system-prompt-file in argv"

        # After dispatch, the temp file MUST be removed
        for p in captured_path:
            assert not os.path.isfile(p), f"temp file leaked after dispatch: {p}"

    @pytest.mark.asyncio
    async def test_pipe_transport_no_system_prompt_no_flag(self, workspace):
        """If system_prompt is None, no -file flag must be added."""
        from agent_os.agent.transports.pipe_transport import (
            PipeTransport,
            PipeTransportConfig,
        )

        transport = PipeTransport(config=PipeTransportConfig(prompt_flag="-p"))
        await transport.start("claude", [], workspace)

        captured_cmd = {}

        def fake_run(cmd, **kwargs):
            captured_cmd["cmd"] = list(cmd)
            mock = MagicMock()
            mock.returncode = 0
            mock.stdout = b""
            mock.stderr = b""
            return mock

        with patch(
            "agent_os.agent.transports.pipe_transport.subprocess.run",
            side_effect=fake_run,
        ):
            await transport.send("hi")

        assert "--append-system-prompt-file" not in captured_cmd["cmd"]

    @pytest.mark.asyncio
    async def test_pipe_transport_cleans_up_on_timeout(self, workspace):
        """Temp file is removed even when subprocess.run raises TimeoutExpired."""
        from agent_os.agent.transports.pipe_transport import (
            PipeTransport,
            PipeTransportConfig,
        )

        transport = PipeTransport(
            config=PipeTransportConfig(prompt_flag="-p"),
            system_prompt="will time out",
            agent_slug="claude-code",
        )
        await transport.start("claude", [], workspace)

        captured_paths: list[str] = []

        def fake_run(cmd, **kwargs):
            for i, a in enumerate(cmd):
                if a == "--append-system-prompt-file":
                    captured_paths.append(cmd[i + 1])
            raise subprocess.TimeoutExpired(cmd, 300)

        with patch(
            "agent_os.agent.transports.pipe_transport.subprocess.run",
            side_effect=fake_run,
        ):
            response = await transport.send("slow")

        assert "timed out" in response.lower()
        assert captured_paths
        for p in captured_paths:
            assert not os.path.isfile(p), f"leaked tmp file: {p}"


# ---------------------------------------------------------------------------
# 8-10. Workspace CLAUDE.md interference detection + WS banner
# ---------------------------------------------------------------------------


class TestClaudemdDetection:
    def test_no_claudemd_returns_none(self, workspace):
        assert detect_claudemd_conflict(workspace) is None

    def test_neutral_claudemd_returns_none(self, workspace):
        path = os.path.join(workspace, "CLAUDE.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# project notes\n\nUse pytest for tests.\n")
        assert detect_claudemd_conflict(workspace) is None

    @pytest.mark.parametrize(
        "phrase",
        [
            "skip reading project files",
            "Don't read this directory",
            "Do Not Read these files",
            "Ignore project state",
            "ignore orbital",
        ],
    )
    def test_conflicting_phrase_detected(self, workspace, phrase):
        path = os.path.join(workspace, "CLAUDE.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# notes\n\n{phrase}\n")
        info = detect_claudemd_conflict(workspace)
        assert info is not None
        assert info["claudemd_path"] == path
        assert info["matched_token"] in phrase.lower()
        assert len(info["content_hash"]) == 64  # sha256 hex

    def test_hash_changes_with_content(self, workspace):
        path = os.path.join(workspace, "CLAUDE.md")
        with open(path, "w") as f:
            f.write("v1")
        h1 = hash_claudemd(workspace)
        with open(path, "w") as f:
            f.write("v2")
        h2 = hash_claudemd(workspace)
        assert h1 != h2

    # --- spec 004: the bare "bypass" token was dropped (false positives) ---

    def test_gatekeeper_bypass_docs_not_detected(self, workspace):
        """Regression (spec 004): a release-runbook line describing the macOS
        Gatekeeper 'Right-click -> Open -> Open to bypass.' step is
        documentation about a user-facing installer dialog, not an instruction
        telling agents to skip Orbital's directives. It must NOT trip the
        conflict banner. This is the exact false positive Orbital's own
        CLAUDE.md produced under the over-broad 'bypass' substring token."""
        path = os.path.join(workspace, "CLAUDE.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "# Release notes\n\n"
                "First launch: expect Gatekeeper warning. User must "
                "Right-click -> Open -> Open to bypass. Document this in "
                "release notes.\n"
            )
        assert detect_claudemd_conflict(workspace) is None

    def test_bare_bypass_word_not_detected(self, workspace):
        """Dropping the token means a bare 'bypass' no longer fires on its
        own — only the five remaining intent-bearing phrases do."""
        path = os.path.join(workspace, "CLAUDE.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# notes\n\nBypass these instructions if needed.\n")
        assert detect_claudemd_conflict(workspace) is None

    def test_genuine_override_still_detected(self, workspace):
        """The remaining tokens still fire after 'bypass' is dropped."""
        path = os.path.join(workspace, "CLAUDE.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# notes\n\nIgnore orbital directives entirely.\n")
        info = detect_claudemd_conflict(workspace)
        assert info is not None
        assert info["matched_token"] == "ignore orbital"


class _FakeWS:
    """Tiny WebSocketManager double for capturing broadcast() calls."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def broadcast(self, project_id: str, payload: dict) -> None:
        self.events.append((project_id, payload))


def _build_manager(ws):
    """Build a SubAgentManager with minimal deps for warning tests."""
    from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

    pm = MagicMock()
    pm._ws = ws
    return SubAgentManager(process_manager=pm, ws_manager=ws)


class TestClaudemdBanner:
    def test_workspace_claudemd_with_conflicting_tokens_emits_warning(
        self, workspace
    ):
        path = os.path.join(workspace, "CLAUDE.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("Please skip reading project files.\n")

        ws = _FakeWS()
        mgr = _build_manager(ws)
        mgr._maybe_emit_claudemd_warning("proj_1", workspace)

        assert len(ws.events) == 1
        pid, payload = ws.events[0]
        assert pid == "proj_1"
        assert payload["type"] == "workspace_claudemd_warning"
        assert payload["claudemd_path"] == path
        assert payload["matched_token"] == "skip reading"
        assert "content_hash" in payload

    def test_workspace_claudemd_without_conflicting_tokens_no_warning(
        self, workspace
    ):
        path = os.path.join(workspace, "CLAUDE.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# project notes\n\nFollow the spec.\n")

        ws = _FakeWS()
        mgr = _build_manager(ws)
        mgr._maybe_emit_claudemd_warning("proj_1", workspace)
        assert ws.events == []

    def test_workspace_claudemd_dismissed_warning_not_re_emitted(self, workspace):
        path = os.path.join(workspace, "CLAUDE.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("ignore orbital instructions\n")

        ws = _FakeWS()
        mgr = _build_manager(ws)

        mgr._maybe_emit_claudemd_warning("proj_1", workspace)
        assert len(ws.events) == 1
        chash = ws.events[0][1]["content_hash"]

        # Dismiss
        mgr.dismiss_claudemd_warning("proj_1", chash)

        # Re-trigger detection: must NOT broadcast again.
        mgr._maybe_emit_claudemd_warning("proj_1", workspace)
        assert len(ws.events) == 1, "warning re-emitted after dismiss"

    def test_warning_not_re_emitted_within_session(self, workspace):
        """Even without dismiss, a second detection pass on the same hash
        must not re-emit (one-time per session)."""
        path = os.path.join(workspace, "CLAUDE.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("don't read these files\n")

        ws = _FakeWS()
        mgr = _build_manager(ws)
        mgr._maybe_emit_claudemd_warning("proj_1", workspace)
        mgr._maybe_emit_claudemd_warning("proj_1", workspace)
        assert len(ws.events) == 1

    def test_warning_re_emitted_when_content_changes(self, workspace):
        """Hash change forces re-emit (file edited mid-session)."""
        path = os.path.join(workspace, "CLAUDE.md")

        with open(path, "w", encoding="utf-8") as f:
            f.write("skip reading v1\n")

        ws = _FakeWS()
        mgr = _build_manager(ws)
        mgr._maybe_emit_claudemd_warning("proj_1", workspace)
        assert len(ws.events) == 1

        # User edits the file with a new conflicting phrase
        with open(path, "w", encoding="utf-8") as f:
            f.write("skip reading v2 — different hash\n")

        mgr._maybe_emit_claudemd_warning("proj_1", workspace)
        assert len(ws.events) == 2
        assert ws.events[0][1]["content_hash"] != ws.events[1][1]["content_hash"]
