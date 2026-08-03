# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression guard for the providers.json path used by the pricing module.

Bug #36: ``_PROVIDERS_JSON`` was built with a literal ``".."`` component
(``.../agent_os/agent/../config/providers.json``). POSIX resolves ``..``
component-by-component against the real filesystem, so in a PyInstaller build —
where ``agent_os/agent/`` exists only inside the PYZ archive and never on disk —
the ``open()`` failed with ENOENT. The failure was swallowed, the caches were
poisoned with ``{}``, and every packaged install silently repriced every model
at the global $3/$15 fallback (and served an empty pricing table to the editor).

The path must therefore be collapsed *lexically*, and a load failure must leave
a log line behind instead of failing silently.
"""

import logging
import os

import pytest

import agent_os.agent.pricing as pricing_mod


@pytest.fixture(autouse=True)
def _reset_pricing_caches():
    pricing_mod._pricing_cache = None
    pricing_mod._full_providers_cache = None
    yield
    pricing_mod._pricing_cache = None
    pricing_mod._full_providers_cache = None


class TestProvidersJsonPath:
    def test_path_has_no_parent_directory_component(self):
        """No ``..`` may survive in the path — it breaks in frozen builds."""
        parts = pricing_mod._PROVIDERS_JSON.replace("\\", "/").split("/")
        assert ".." not in parts, (
            f"_PROVIDERS_JSON still contains a '..' component: "
            f"{pricing_mod._PROVIDERS_JSON}. Collapse it lexically "
            f"(dirname(dirname(abspath(__file__)))) so it resolves inside a "
            f"PyInstaller bundle where agent_os/agent/ has no on-disk directory."
        )

    def test_path_points_at_the_shipped_providers_json(self):
        assert os.path.isfile(pricing_mod._PROVIDERS_JSON)
        assert pricing_mod._PROVIDERS_JSON.replace("\\", "/").endswith(
            "agent_os/config/providers.json"
        )

    def test_defaults_actually_load_through_that_path(self):
        """Sanity: the resolved path yields real pricing, not an empty table."""
        assert pricing_mod._load_full_providers(), "no providers loaded"
        assert pricing_mod._load_pricing(), "no pricing loaded"


class TestLoadFailureIsLogged:
    def test_load_full_providers_warns_when_file_is_unreadable(self, tmp_path, caplog):
        pricing_mod._PROVIDERS_JSON_ORIG = pricing_mod._PROVIDERS_JSON
        missing = str(tmp_path / "nope" / "providers.json")
        pricing_mod._PROVIDERS_JSON = missing
        try:
            with caplog.at_level(logging.WARNING, logger="agent_os.agent.pricing"):
                assert pricing_mod._load_full_providers() == {}
        finally:
            pricing_mod._PROVIDERS_JSON = pricing_mod._PROVIDERS_JSON_ORIG

        assert any(
            rec.levelno >= logging.WARNING and missing in rec.getMessage()
            for rec in caplog.records
        ), "expected a warning naming the unreadable providers.json path"

    def test_load_pricing_warns_when_file_is_unreadable(self, tmp_path, caplog):
        pricing_mod._PROVIDERS_JSON_ORIG = pricing_mod._PROVIDERS_JSON
        missing = str(tmp_path / "nope" / "providers.json")
        pricing_mod._PROVIDERS_JSON = missing
        try:
            with caplog.at_level(logging.WARNING, logger="agent_os.agent.pricing"):
                assert pricing_mod._load_pricing() == {}
        finally:
            pricing_mod._PROVIDERS_JSON = pricing_mod._PROVIDERS_JSON_ORIG

        assert any(
            rec.levelno >= logging.WARNING and missing in rec.getMessage()
            for rec in caplog.records
        ), "expected a warning naming the unreadable providers.json path"
