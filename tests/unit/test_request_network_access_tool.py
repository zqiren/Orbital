# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import json

from agent_os.agent.tools.request_network_access import RequestNetworkAccessTool


def test_returns_pending_with_normalized_domain():
    tool = RequestNetworkAccessTool()
    result = tool.execute(domain="https://X.com/path", reason="verify handles")
    payload = json.loads(result.content)
    assert payload["status"] == "pending"
    assert payload["domain"] == "x.com"
    assert payload["reason"] == "verify handles"


def test_rejects_ungrantable_domain():
    tool = RequestNetworkAccessTool()
    result = tool.execute(domain="127.0.0.1", reason="x")
    assert "Error" in result.content
