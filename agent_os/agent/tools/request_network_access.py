# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""RequestNetworkAccessTool — TOFU domain grant request (always pending)."""

import json

from agent_os.daemon_v2.network_rules_builder import normalize_domain

from .base import Tool, ToolResult


class RequestNetworkAccessTool(Tool):
    """Ask the user to approve a network domain for this project.

    The AutonomyInterceptor intercepts this call in EVERY preset — approval
    persists the domain to the project allowlist permanently. In hands-off
    the ask auto-denies for the current run after a timeout; the request
    stays pending in Project Settings.
    """

    def __init__(self):
        self.name = "request_network_access"
        self.description = (
            "Request permanent approval for shell/network access to a domain "
            "not on this project's allowlist. Use when a command needs an "
            "unapproved API or host (a proxy 403 means policy, not a broken "
            "network). For merely READING web content, use the browser tool "
            "instead — it needs no approval."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Bare domain to approve, e.g. api.stripe.com",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this task needs it (shown to the user)",
                },
                "intent": {
                    "type": "string",
                    "description": "What you'll do: e.g. 'fetch docs from' or "
                                   "'upload build artifacts to' (display only)",
                },
            },
            "required": ["domain", "reason"],
        }

    def execute(self, **arguments) -> ToolResult:
        raw = arguments.get("domain", "")
        domain = normalize_domain(raw)
        if domain is None:
            return ToolResult(content=(
                f"Error: '{raw}' is not a grantable domain (bare registrable "
                f"domains only — no IPs, wildcards, or paths)."
            ))
        return ToolResult(content=json.dumps({
            "domain": domain,
            "reason": arguments.get("reason", ""),
            "intent": arguments.get("intent", ""),
            "status": "pending",
        }))
