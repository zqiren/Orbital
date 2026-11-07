# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Attach by local path (desktop app, 2026-11-07).

In the desktop app a composer attachment is the file's absolute path, not an
uploaded copy, with no size cap. The daemon validates the path and writes it
into the same ``<attached_files>`` prefix marked as the user's local file;
the attachment is a read grant for every agent in the project.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agent_os.api.routes._attachment_formatter import (
    format_prefix,
    validate_attachments,
)
from agent_os.api.routes.agents_v2 import InjectAttachment
from agent_os.daemon_v2 import attachment_grants


@pytest.fixture(autouse=True)
def _clean_grants():
    attachment_grants.reset()
    yield
    attachment_grants.reset()


def _write(path, payload=b"first line\nsecond\n"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(payload)
    return len(payload)


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class TestInjectAttachmentLocal:
    def test_local_absolute_path_accepted(self, tmp_path):
        a = InjectAttachment(path=str(tmp_path / "a.txt"), mime="text/plain",
                             size=3, local=True)
        assert a.local is True

    def test_local_has_no_size_cap(self, tmp_path):
        a = InjectAttachment(path=str(tmp_path / "big.bin"),
                             mime="application/octet-stream",
                             size=50 * 1024 * 1024, local=True)
        assert a.size == 50 * 1024 * 1024

    def test_local_relative_path_rejected(self):
        with pytest.raises(ValidationError):
            InjectAttachment(path="uploads/a.txt", mime="text/plain", size=1, local=True)

    def test_upload_absolute_path_still_rejected(self):
        with pytest.raises(ValidationError):
            InjectAttachment(path="/etc/hosts", mime="text/plain", size=1)

    def test_upload_over_10mb_still_rejected(self):
        with pytest.raises(ValidationError):
            InjectAttachment(path="uploads/a.bin", mime="application/octet-stream",
                             size=10 * 1024 * 1024 + 1)

    def test_local_defaults_false(self):
        assert InjectAttachment(path="uploads/a", mime="text/plain", size=1).local is False


# ---------------------------------------------------------------------------
# Validation + prefix
# ---------------------------------------------------------------------------


def _local(path, mime="text/plain", size=0):
    # SimpleNamespace bypasses the pydantic layer so the filesystem checks
    # are exercised on their own.
    return SimpleNamespace(path=path, mime=mime, size=size, local=True)


class TestValidateLocal:
    def test_missing_file_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="file not found"):
            validate_attachments(str(tmp_path / "ws"), [_local(str(tmp_path / "nope.txt"))])

    def test_directory_rejected(self, tmp_path):
        d = tmp_path / "folder"
        d.mkdir()
        with pytest.raises(ValueError, match="not a regular file"):
            validate_attachments(str(tmp_path / "ws"), [_local(str(d))])

    def test_relative_path_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="absolute"):
            validate_attachments(str(tmp_path / "ws"), [_local("notes.txt")])

    def test_symlink_resolves_to_realpath_and_real_size(self, tmp_path):
        real = tmp_path / "real" / "notes.txt"
        size = _write(str(real))
        link = tmp_path / "link.txt"
        os.symlink(real, link)
        out = validate_attachments(str(tmp_path / "ws"), [_local(str(link), size=0)])
        assert out[0].path == os.path.realpath(str(real))
        assert out[0].size == size
        assert out[0].local is True

    def test_upload_path_unchanged(self, tmp_path):
        ws = tmp_path / "ws"
        size = _write(str(ws / "uploads" / "a.txt"))
        att = InjectAttachment(path="uploads/a.txt", mime="text/plain", size=size)
        out = validate_attachments(str(ws), [att])
        assert out[0].path == "uploads/a.txt"
        assert out[0].local is False
        assert out[0].as_row() == {"path": "uploads/a.txt", "mime": "text/plain", "size": size}

    def test_local_row_carries_flag(self, tmp_path):
        f = tmp_path / "n.txt"
        size = _write(str(f))
        out = validate_attachments(str(tmp_path / "ws"), [_local(str(f))])
        assert out[0].as_row() == {"path": os.path.realpath(str(f)), "mime": "text/plain",
                                   "size": size, "local": True}


class TestPrefix:
    def test_local_line_is_marked(self):
        att = SimpleNamespace(path="/Users/me/Downloads/report.pdf", mime="application/pdf",
                              size=50 * 1024 * 1024, local=True)
        assert format_prefix([att]) == (
            "<attached_files>\n"
            "- /Users/me/Downloads/report.pdf (application/pdf, 50.0 MB, "
            "the user's local file, not copied)\n"
            "</attached_files>\n\n"
        )

    def test_mixed_upload_and_local(self):
        up = SimpleNamespace(path="uploads/shot.png", mime="image/png", size=2048, local=False)
        lo = SimpleNamespace(path="/tmp/x.txt", mime="text/plain", size=5, local=True)
        assert format_prefix([up, lo]) == (
            "<attached_files>\n"
            "- uploads/shot.png (image/png, 2.0 KB)\n"
            "- /tmp/x.txt (text/plain, 5 B, the user's local file, not copied)\n"
            "</attached_files>\n\n"
        )

    def test_upload_line_unchanged(self):
        up = InjectAttachment(path="uploads/a.txt", mime="text/plain", size=11)
        assert format_prefix([up]) == (
            "<attached_files>\n- uploads/a.txt (text/plain, 11 B)\n</attached_files>\n\n")


# ---------------------------------------------------------------------------
# Grants
# ---------------------------------------------------------------------------


class TestGrants:
    def test_grant_is_per_project_and_realpathed(self, tmp_path):
        f = tmp_path / "a.txt"
        _write(str(f))
        link = tmp_path / "l.txt"
        os.symlink(f, link)
        attachment_grants.grant("p1", [str(link)])
        assert attachment_grants.is_granted("p1", str(f))
        assert attachment_grants.is_granted("p1", str(link))
        assert not attachment_grants.is_granted("p2", str(f))
        assert not attachment_grants.is_granted("p1", str(tmp_path / "other.txt"))
        assert not attachment_grants.is_granted("p1", "a.txt")

    def test_local_paths_parsed_from_prefix(self):
        # A platform-absolute path ("C:\\Users\\…" on Windows, "/Users/…" elsewhere).
        local = os.path.abspath(os.path.join(os.sep, "Users", "me", "My (draft).txt"))
        content = ("<attached_files>\n"
                   "- uploads/shot.png (image/png, 2.0 KB)\n"
                   f"- {local} (text/plain, 5 B, the user's local file, not copied)\n"
                   "</attached_files>\n\nhi")
        assert attachment_grants.local_paths_in(content) == [local]

    def test_hydrate_from_session_rows_once_and_skips_missing(self, tmp_path):
        f = tmp_path / "keep.txt"
        _write(str(f))
        gone = tmp_path / "gone.txt"
        rows = [{"role": "user", "content": format_prefix([
            SimpleNamespace(path=str(f), mime="text/plain", size=3, local=True),
            SimpleNamespace(path=str(gone), mime="text/plain", size=3, local=True),
        ]) + "look"}]
        attachment_grants.hydrate_from_messages("p", "s", rows)
        assert attachment_grants.granted("p") == frozenset({os.path.realpath(str(f))})
        attachment_grants.reset()
        attachment_grants.hydrate_from_messages("p", "s", rows)
        attachment_grants.hydrate_from_messages("p", "s", [])  # second call: no-op
        assert attachment_grants.is_granted("p", str(f))


# ---------------------------------------------------------------------------
# Orbital manager read tool
# ---------------------------------------------------------------------------


class TestManagerRead:
    def test_safe_mode_read_tool_reads_attached_file_only(self, tmp_path):
        from agent_os.agent.tools.read import ReadTool
        from agent_os.daemon_v2.agent_manager import AgentManager

        ws = tmp_path / "ws"
        ws.mkdir()
        outside = tmp_path / "home" / "notes.txt"
        _write(str(outside), b"ATTACHED-FIRST\nmore\n")
        sibling = tmp_path / "home" / "secret.txt"
        _write(str(sibling), b"nope\n")
        attachment_grants.grant("p", [str(outside)])

        fake = SimpleNamespace(_handles={})
        roots_cb = lambda: AgentManager._read_roots_with_attached(fake, "p", "s", [str(ws)])
        tool = ReadTool(workspace=str(ws), read_roots=roots_cb,
                        open_reads=lambda: False, cross_project=False)
        assert "ATTACHED-FIRST" in str(tool.execute(path=str(outside)).content)
        assert "outside workspace" in str(tool.execute(path=str(sibling)).content)

    def test_roots_unchanged_without_grants(self, tmp_path):
        from agent_os.daemon_v2.agent_manager import AgentManager
        fake = SimpleNamespace(_handles={})
        assert AgentManager._read_roots_with_attached(fake, "p", "s", ["/ws"]) == ["/ws"]


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


def _grants_for(path):
    return lambda: frozenset({os.path.realpath(path)})


class TestClaudeCodeGrant:
    def test_supervised_read_of_attached_file_allowed_without_prompt(self, tmp_path):
        from claude_agent_sdk import PermissionResultAllow
        from agent_os.agent.prompt_builder import Autonomy
        from agent_os.agent.transports.sdk_transport import SDKTransport

        f = tmp_path / "a.txt"
        _write(str(f))
        t = SDKTransport(autonomy=Autonomy.SUPERVISED)
        t.set_read_grants(_grants_for(str(f)))

        async def go():
            return await t._handle_permission("Read", {"file_path": str(f)}, None)
        res = asyncio.run(go())
        assert isinstance(res, PermissionResultAllow)
        assert t._pending_approvals == {}

    def test_other_paths_and_writes_still_ask(self, tmp_path):
        from agent_os.agent.prompt_builder import Autonomy
        from agent_os.agent.transports.sdk_transport import SDKTransport

        f = tmp_path / "a.txt"
        _write(str(f))
        t = SDKTransport(autonomy=Autonomy.SUPERVISED)
        t.set_read_grants(_grants_for(str(f)))
        assert t._is_attached_file_read("Read", {"file_path": str(f)})
        assert t._is_attached_file_read("Grep", {"pattern": "x", "path": str(f)})
        assert not t._is_attached_file_read("Read", {"file_path": str(tmp_path / "b.txt")})
        assert not t._is_attached_file_read("Write", {"file_path": str(f)})
        assert not t._is_attached_file_read("Edit", {"file_path": str(f)})
        assert not t._is_attached_file_read("Bash", {"command": f"cat {f}"})

    def test_no_grants_callable_means_no_grant(self, tmp_path):
        from agent_os.agent.transports.sdk_transport import SDKTransport
        t = SDKTransport()
        assert not t._is_attached_file_read("Read", {"file_path": str(tmp_path / "a")})


class TestACPGrant:
    def test_ask_mode_read_of_attached_file_auto_allowed(self, tmp_path):
        from acp.schema import PermissionOption, ToolCallLocation, ToolCallUpdate
        from agent_os.agent.transports.acp_sdk_transport import ACPSDKTransport

        f = tmp_path / "a.txt"
        _write(str(f))
        t = ACPSDKTransport(permission_mode="ask")
        t.set_read_grants(_grants_for(str(f)))
        opts = [PermissionOption(option_id="allow-once", name="Allow", kind="allow_once"),
                PermissionOption(option_id="reject-once", name="Reject", kind="reject_once")]

        async def go():
            return await t.request_permission(
                "s", ToolCallUpdate(tool_call_id="1", title="Read", kind="read",
                                    locations=[ToolCallLocation(path=str(f))]), opts)
        res = asyncio.run(asyncio.wait_for(go(), 2))
        assert res.outcome.outcome == "selected"
        assert res.outcome.option_id == "allow-once"

    def test_edit_of_attached_file_is_not_granted(self, tmp_path):
        from acp.schema import ToolCallLocation, ToolCallUpdate
        from agent_os.agent.transports.acp_sdk_transport import ACPSDKTransport

        f = tmp_path / "a.txt"
        _write(str(f))
        t = ACPSDKTransport(permission_mode="ask")
        t.set_read_grants(_grants_for(str(f)))
        edit = ToolCallUpdate(tool_call_id="1", title="Edit", kind="edit",
                              locations=[ToolCallLocation(path=str(f))])
        other = ToolCallUpdate(tool_call_id="2", title="Read", kind="read",
                               locations=[ToolCallLocation(path=str(tmp_path / "b"))])
        assert not t._is_attached_file_read(edit)
        assert not t._is_attached_file_read(other)


class TestPipeGrant:
    def test_claude_pipe_adds_parent_dir_of_each_attached_file(self, tmp_path, monkeypatch):
        from agent_os.agent.transports.pipe_transport import (
            CLAUDE_CODE_PIPE_CONFIG, PipeTransport)

        f = tmp_path / "docs" / "a.txt"
        _write(str(f))
        t = PipeTransport(config=CLAUDE_CODE_PIPE_CONFIG, agent_slug="claude-code")
        t.set_read_grants(_grants_for(str(f)))
        seen = {}

        def fake_run(cmd, **kw):
            seen["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

        monkeypatch.setattr(subprocess, "run", fake_run)

        async def go():
            await t.start("claude", [], str(tmp_path / "ws"))
            await t.send("hi")
        asyncio.run(go())
        cmd = seen["cmd"]
        i = cmd.index("--add-dir")
        assert cmd[i + 1] == os.path.dirname(os.path.realpath(str(f)))

    def test_generic_pipe_gets_no_flag(self, tmp_path):
        from agent_os.agent.transports.pipe_transport import PipeTransport
        t = PipeTransport()
        t.set_read_grants(_grants_for(str(tmp_path / "a")))
        assert t._attached_dir_args() == []


# ---------------------------------------------------------------------------
# Desktop bridge
# ---------------------------------------------------------------------------


class TestDesktopPickFiles:
    def test_pick_returns_absolute_paths_names_sizes(self, tmp_path):
        from agent_os.desktop.main import pick_files_with
        a = tmp_path / "a.txt"
        size = _write(str(a))
        big = tmp_path / "big.bin"
        with open(big, "wb") as fh:
            fh.truncate(50 * 1024 * 1024)
        out = pick_files_with(lambda: (str(a), str(big)))
        assert out == [
            {"path": os.path.realpath(str(a)), "name": "a.txt", "size": size,
             "mime": "text/plain"},
            {"path": os.path.realpath(str(big)), "name": "big.bin",
             "size": 50 * 1024 * 1024, "mime": "application/octet-stream"},
        ]

    def test_cancel_returns_empty(self):
        from agent_os.desktop.main import pick_files_with
        assert pick_files_with(lambda: None) == []

    def test_dialog_error_returns_empty(self):
        from agent_os.desktop.main import pick_files_with

        def boom():
            raise RuntimeError("no window")
        assert pick_files_with(boom) == []

    def test_folders_and_missing_left_out(self, tmp_path):
        from agent_os.desktop.main import local_file_entries
        d = tmp_path / "dir"
        d.mkdir()
        assert local_file_entries([str(d), str(tmp_path / "missing"), "rel.txt"]) == []

    def test_dropped_entries_from_pywebview_event(self, tmp_path):
        from agent_os.desktop.main import dropped_file_entries, native_drop_js
        a = tmp_path / "a.txt"
        _write(str(a))
        event = {"dataTransfer": {"files": [
            {"name": "a.txt", "pywebviewFullPath": str(a)},
            {"name": "web.png"},  # no native path: the SPA uploads it
        ]}}
        out = dropped_file_entries(event)
        assert [e["name"] for e in out] == ["a.txt"]
        js = native_drop_js(out)
        assert js.startswith('window.dispatchEvent(new CustomEvent("orbital:native-drop"')
        # The path rides inside a JSON string, so compare it JSON-escaped
        # (backslashes double on Windows).
        assert json.dumps(os.path.realpath(str(a)))[1:-1] in js


class TestShellFocusNote:
    def test_attached_file_gets_no_focus_warning_other_paths_do(self, tmp_path):
        from agent_os.agent.tools.shell import ShellTool

        ws = tmp_path / "ws"
        ws.mkdir()
        f = tmp_path / "home" / "a.txt"
        _write(str(f))
        other = tmp_path / "home" / "b.txt"
        _write(str(other))
        attachment_grants.grant("p", [str(f)])
        tool = ShellTool(workspace=str(ws), os_type="macos", project_id="p",
                         safe_mode_cb=lambda: True)
        # pytest's tmp dir sits under the OS temp root, which the focus rule
        # treats as a toolchain path; count it as outside for this test.
        tool._is_toolchain_path = lambda path: False
        out = tool.execute(command=f"head -1 {f}")
        assert "[focus]" not in out.content
        out = tool.execute(command=f"head -1 {f} {other}")
        assert "[focus]" in out.content
        warning = out.content.split("\n", 1)[0]
        assert os.path.join("home", "b.txt") in warning
        assert os.path.join("home", "a.txt") not in warning
