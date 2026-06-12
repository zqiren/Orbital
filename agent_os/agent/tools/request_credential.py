# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""RequestCredentialTool — request website credentials via secure modal."""

import json

from .base import Tool, ToolResult


class RequestCredentialTool(Tool):
    """Request user credentials for a website. Triggers secure input modal."""

    def __init__(self, credential_store):
        self.name = "request_credential"
        self.description = (
            "Request website login credentials from the user. "
            "If credentials already exist, returns secret tokens. "
            "Otherwise, triggers a secure input modal (values never enter chat). "
            "Request exactly the fields the login form shows."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Identifier for the credential (e.g., 'twitter', 'amazon')",
                },
                "domain": {
                    "type": "string",
                    "description": "Website domain (e.g., 'twitter.com')",
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Request exactly the fields the login form shows, "
                        "e.g. ['email'], ['username', 'password'], or ['code']"
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "Why credentials are needed — shown to user",
                },
            },
            "required": ["name", "domain", "fields", "reason"],
        }
        self._credential_store = credential_store

    def execute(self, **arguments) -> ToolResult:
        try:
            name = arguments["name"]
            domain = arguments["domain"]
            fields = arguments["fields"]
            reason = arguments["reason"]

            # Check if credential already exists
            existing = self._credential_store.get_metadata(name)
            if existing is not None:
                tokens = {f: f"<secret:{name}.{f}>" for f in fields}
                return ToolResult(content=json.dumps({
                    "status": "ready",
                    "name": name,
                    "tokens": tokens,
                    "message": f"Credential '{name}' is stored. Use the <secret:> tokens.",
                }))

            # Credential doesn't exist — return pending with meta signal.
            # The pending result must already carry the usable token names:
            # resume does NOT re-execute this tool, so this is the agent's
            # only authoritative source of token names.
            tokens = {f: f"<secret:{name}.{f}>" for f in fields}
            return ToolResult(
                content=json.dumps({
                    "status": "pending",
                    "name": name,
                    "domain": domain,
                    "fields": fields,
                    "reason": reason,
                    "tokens": tokens,
                    "message": (
                        "Waiting for the user to provide credentials via the "
                        "secure modal. These tokens become usable AFTER the "
                        "user submits the modal — use exactly these tokens; "
                        "do not construct or guess token names."
                    ),
                }),
                meta={
                    "credential_request": True,
                    "name": name,
                    "domain": domain,
                    "fields": fields,
                    "reason": reason,
                },
            )
        except Exception as e:
            return ToolResult(content=f"Error: {str(e)}")
