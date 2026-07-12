# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""normalize_domain + build_network_rules — the TOFU allowlist core."""

import pytest

from agent_os.daemon_v2.network_rules_builder import (
    build_network_rules,
    normalize_domain,
)
from agent_os.platform.types import DEFAULT_ALLOWLIST_DOMAINS


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("x.com", "x.com"),
        ("X.COM", "x.com"),
        ("https://x.com/some/path?q=1", "x.com"),
        ("http://api.stripe.com:443", "api.stripe.com"),
        ("  docs.python.org  ", "docs.python.org"),
    ],
)
def test_normalize_domain_accepts(raw, expected):
    assert normalize_domain(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "*.x.com", "*", "127.0.0.1", "192.168.1.10", "https://", "not a domain"],
)
def test_normalize_domain_rejects(raw):
    assert normalize_domain(raw) is None


def test_build_rules_includes_defaults_and_wildcarded_grants():
    rules = build_network_rules(["x.com"])
    assert rules.mode == "allowlist"
    for d in DEFAULT_ALLOWLIST_DOMAINS:
        assert d in rules.domains
    assert "x.com" in rules.domains
    assert "*.x.com" in rules.domains


def test_build_rules_handles_none_and_extra():
    rules = build_network_rules(None, extra=["internal.corp.example"])
    assert "internal.corp.example" in rules.domains
    assert set(DEFAULT_ALLOWLIST_DOMAINS) <= set(rules.domains)


def test_build_rules_dedupes():
    rules = build_network_rules(["pypi.org"])  # already a default
    assert rules.domains.count("pypi.org") == 1
