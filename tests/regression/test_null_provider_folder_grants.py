# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Linux has no sandbox provider, so it runs on NullProvider. Folder grants used
to fail there, which turned every approved request_access into an error."""

from __future__ import annotations

import os

from agent_os.platform.null import NullProvider


def test_grant_is_listed_under_its_scope(tmp_path):
    provider = NullProvider()
    result = provider.grant_folder_access("/data", "read_only", scope=str(tmp_path))
    assert result.success is True
    assert provider.portal_paths_for_scope(str(tmp_path)) == {"/data": "read_only"}


def test_scope_is_compared_by_realpath(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(real, link)
    provider = NullProvider()
    provider.grant_folder_access("/data", "read_write", scope=str(link))
    assert provider.portal_paths_for_scope(str(real)) == {"/data": "read_write"}


def test_grants_do_not_leak_between_projects(tmp_path):
    provider = NullProvider()
    provider.grant_folder_access("/data", "read_only", scope=str(tmp_path / "a"))
    assert provider.portal_paths_for_scope(str(tmp_path / "b")) == {}


def test_revoke_removes_only_that_grant(tmp_path):
    provider = NullProvider()
    scope = str(tmp_path)
    provider.grant_folder_access("/a", "read_only", scope=scope)
    provider.grant_folder_access("/b", "read_only", scope=scope)
    assert provider.revoke_folder_access("/a", scope=scope).success is True
    assert provider.portal_paths_for_scope(scope) == {"/b": "read_only"}


def test_revoking_an_unknown_grant_is_harmless(tmp_path):
    provider = NullProvider()
    assert provider.revoke_folder_access("/a", scope=str(tmp_path)).success is True


def test_the_returned_mapping_is_a_copy(tmp_path):
    provider = NullProvider()
    scope = str(tmp_path)
    provider.grant_folder_access("/a", "read_only", scope=scope)
    provider.portal_paths_for_scope(scope).clear()
    assert provider.portal_paths_for_scope(scope) == {"/a": "read_only"}
