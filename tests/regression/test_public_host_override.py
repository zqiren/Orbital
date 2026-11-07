# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""In a container the detected LAN address is the unreachable bridge IP, so the
phone-access URL has to be able to name the host people really use."""

from agent_os.api.routes.agents_v2 import _get_lan_ip


def test_public_host_replaces_the_detected_address(monkeypatch):
    monkeypatch.setenv("AGENT_OS_PUBLIC_HOST", "orbital.example.ts.net")
    assert _get_lan_ip() == "orbital.example.ts.net"


def test_detection_is_unchanged_without_the_override(monkeypatch):
    monkeypatch.delenv("AGENT_OS_PUBLIC_HOST", raising=False)
    assert _get_lan_ip() != "orbital.example.ts.net"
