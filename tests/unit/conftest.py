# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit-suite isolation for session import (2026-10-21).

``list_sessions`` lists a project folder's native Claude Code / Codex sessions
(``agent_os.onboarding.session_import``). Point the importer at empty homes for
every unit test so none reads the developer's real ``~/.claude`` / ``~/.codex``
(or ``~/.pi`` / ``~/.kimi-code`` / ``~/.minimax``, spec 138);
tests that exercise the importer set these to their own fixture dirs.

Same for DeepSeek Harness: its spawn copies provider rows out of DSH Desktop's
profile under ``$DSH_HOME`` and reads keys from its credentials file there, so
every unit test gets an empty ``DSH_HOME`` instead of the developer's ``~/.dsh``.
"""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_native_session_homes(tmp_path_factory, monkeypatch):
    empty = tmp_path_factory.mktemp("native-homes")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(empty / "claude"))
    monkeypatch.setenv("ORBITAL_CODEX_HOME", str(empty / "codex"))
    monkeypatch.setenv("DSH_HOME", str(empty / "dsh"))
    # Spec 138: pi / Kimi Code / MiniMax Code stores are read under this home.
    monkeypatch.setenv("ORBITAL_HISTORY_HOME", str(empty / "history-home"))
    # Shared skills: the ~/.agents, ~/.claude, ~/.codex skill folders the
    # manager's prompt and the `/` picker list are read under this home, so
    # the developer's own skills never reach a prompt snapshot.
    monkeypatch.setenv("ORBITAL_SKILLS_HOME", str(empty / "skills-home"))
    yield


@pytest.fixture(autouse=True)
def _no_real_drive_roots_in_repo_walk():
    """Batch 1114: on Windows the import repo walk also starts at every fixed
    drive's root, so a test walking its own temp home would also list the
    machine's real repos (on CI the checkout on ``D:\\``). Tests that cover the
    drive starts pass their own ``drives=``."""
    with patch("agent_os.onboarding.import_scanner._windows_fixed_drives",
               lambda: []):
        yield


@pytest.fixture(autouse=True)
def _no_quota_store_between_tests():
    """Spec 197: Orbital's own agent picks read the process-wide quota
    store's limit marks. ``create_app()`` installs one per app and nothing
    removed it, so a limit one test recorded reordered a later test's picks.
    Each test starts with none; a test that needs one installs it (or
    builds an app, which does)."""
    from agent_os.daemon_v2 import quota_store
    quota_store.install(None)
    yield
    quota_store.install(None)


@pytest.fixture(autouse=True)
def _no_sub_agent_warm_up():
    """Spec 136: ``create_app()``'s lifespan starts a background sweep of
    every agent CLI. Unit tests that enter it must not probe the developer's
    real CLIs (or race a test's own ``subprocess.run`` patch);
    ``test_settings_sub_agents_136.py`` calls the route-level hook directly."""
    with patch("agent_os.api.app.start_sub_agent_status"):
        yield
